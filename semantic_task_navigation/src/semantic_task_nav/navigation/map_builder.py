"""Navigation map in raw_image_px (IMAGE_SPACE_DEMO).

Layers are kept separate: static traversable region, static obstacles, no-go regions, dynamic
obstacles (from this frame's detections) and semantic regions (not used for collision).
Nothing is traversable unless it lies inside a traversable polygon; floor colour is not used.
The robot footprint is a disc of robot_radius_px (assumption). A grid cell is valid when the disc
centred on the cell centre has pixel-level clearance >= robot_radius_px.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from ..config import sha256_bytes
from ..contracts import RAW_PX, validate

_POLY_LIST = {"type": "array", "items": {"type": "object", "required": ["id", "polygon"],
                                         "properties": {"id": {"type": "string"},
                                                        "polygon": {"type": "array", "minItems": 3}}}}
MAP_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "map_id", "coordinate_frame", "image_size", "review_status", "traversable",
                 "static_obstacles", "no_go", "start_pose"],
    "properties": {"coordinate_frame": {"type": "string", "enum": [RAW_PX]},
                   "image_size": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                   "traversable": {**_POLY_LIST, "minItems": 1}, "static_obstacles": _POLY_LIST, "no_go": _POLY_LIST,
                   "start_pose": {"type": ["object", "null"]}},
}


class MapError(ValueError):
    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def load_map_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate(spec, MAP_SCHEMA)


def _fill(size, polys):
    w, h = size
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p, dtype=np.float64)).astype(np.int32)], 1)
    return m.astype(bool)


class NavMap:
    def __init__(self, spec, image_size, pcfg, dynamic_obstacles=()):
        w, h = image_size
        if list(spec["image_size"]) != [w, h]:
            raise MapError("MAP_IMAGE_SIZE_MISMATCH", f"map is for {spec['image_size']}, frame is {[w, h]}")
        self.spec, self.width, self.height = spec, w, h
        self.cell_px = pcfg["cell_px"]
        self.robot_radius_px = pcfg["robot_radius_px"]
        self.safety_margin_px = pcfg["safety_margin_px"]
        self.dynamic_obstacles = [dict(d) for d in dynamic_obstacles]
        full = {
            "traversable": _fill((w, h), [z["polygon"] for z in spec["traversable"]]),
            "static_obstacle": _fill((w, h), [z["polygon"] for z in spec["static_obstacles"]]),
            "no_go": _fill((w, h), [z["polygon"] for z in spec["no_go"]]),
            "dynamic_obstacle": _fill((w, h), [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
                                               for b in (d["bbox_xyxy"] for d in self.dynamic_obstacles)]),
        }
        self.full_layers = full
        self.free_full = full["traversable"] & ~full["static_obstacle"] & ~full["no_go"] & ~full["dynamic_obstacle"]
        padded = np.pad(self.free_full.astype(np.uint8), 1)          # outside the image = blocked
        self.clearance_full = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        self.rows, self.cols = h // self.cell_px, w // self.cell_px
        ys = (np.arange(self.rows) * self.cell_px + self.cell_px // 2)
        xs = (np.arange(self.cols) * self.cell_px + self.cell_px // 2)
        self.layers = {k: v[np.ix_(ys, xs)] for k, v in full.items()}
        self.free = self.free_full[np.ix_(ys, xs)]
        self.clearance = self.clearance_full[np.ix_(ys, xs)]
        self.valid = self.free & (self.clearance >= self.robot_radius_px)
        self.risk = np.where(self.valid, np.clip(1.0 - (self.clearance - self.robot_radius_px) / self.safety_margin_px,
                                                 0.0, 1.0), np.nan)

    def cell_center(self, r, c):
        return (c + 0.5) * self.cell_px, (r + 0.5) * self.cell_px

    def cell_of(self, x, y):
        return int(y // self.cell_px), int(x // self.cell_px)

    def in_grid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def summary(self):
        def h(a):
            return sha256_bytes(np.packbits(a.astype(bool)).tobytes())
        return {"coordinate_frame": RAW_PX, "map_mode_note": "image space; not metric", "cell_px": self.cell_px,
                "rows": self.rows, "cols": self.cols, "robot_radius_px": self.robot_radius_px,
                "robot_footprint": "disc (assumed, image space)", "safety_margin_px": self.safety_margin_px,
                "cell_counts": {k: int(v.sum()) for k, v in self.layers.items()} | {
                    "free": int(self.free.sum()), "valid_robot_center": int(self.valid.sum())},
                "layer_sha256": {k: h(v) for k, v in self.layers.items()} | {"valid": h(self.valid)},
                "dynamic_obstacles": self.dynamic_obstacles}


def check_alignment(spec, frame_bgr, resolve):
    """Is the current frame taken from the same camera pose as the map's reference image?
    Global shift by phase correlation (method as in EXP111 tools/camera_check.py)."""
    ref_path = spec.get("reference_image")
    if not ref_path or not Path(resolve(ref_path)).exists():
        return {"status": "NOT_AVAILABLE", "reason": "map has no reference image; alignment not verified"}
    ref = cv2.imread(str(resolve(ref_path)), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if ref.shape[:2] != frame_bgr.shape[:2]:
        return {"status": "MISALIGNED", "reason": f"reference size {ref.shape[1::-1]} != frame size"}
    a = np.float32(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
    b = np.float32(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    (dx, dy), resp = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    th = spec.get("alignment_check", {"max_shift_px": 3.0, "min_response": 0.3})
    ok = abs(dx) <= th["max_shift_px"] and abs(dy) <= th["max_shift_px"] and resp >= th["min_response"]
    return {"status": "ALIGNED" if ok else "MISALIGNED", "shift_px": [round(dx, 3), round(dy, 3)],
            "response": round(float(resp), 4), "thresholds": th, "reference_image": ref_path}


def dynamic_obstacles_from_detections(objects, classes, spec, image_size):
    """Accepted detections of the given classes that overlap the traversable region."""
    trav = _fill(image_size, [z["polygon"] for z in spec["traversable"]])
    out = []
    for o in objects:
        if o["class_name"] not in classes or not o.get("accepted"):
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in o["bbox_xyxy"])
        if trav[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].any():
            out.append({"id": o["object_id"], "bbox_xyxy": o["bbox_xyxy"], "source": o["source"],
                        "knowledge": o["knowledge"]})
    return out
