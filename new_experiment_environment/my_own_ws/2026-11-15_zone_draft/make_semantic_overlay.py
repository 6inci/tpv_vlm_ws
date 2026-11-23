#!/usr/bin/env python3
"""Render the P01 semantic zone draft without changing the source image or annotations."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = HERE / "semantic_zones_draft_v2.json"
OUT = HERE / "semantic_overlay_P01_v2.png"
COLORS = {
    "pedestrian_route": (70, 230, 70),
    "robot_drivable_outer_area": (0, 180, 255),
    "cargo_staging": (0, 120, 255),
    "material_staging": (0, 120, 255),
    "charging_area": (255, 50, 255),
    "loading_bay": (255, 130, 0),
    "robotic_workcell": (60, 60, 255),
    "worker_support": (180, 110, 255),
    "rack_storage": (240, 240, 240),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = json.loads(SPEC.read_text())
    image_path = Path(spec["image"])
    annotation_path = Path(spec["annotation"])
    source_path = Path(spec["source_geometry"]["path"])
    for path, expected in ((image_path, spec["image_sha256"]),
                           (annotation_path, spec["annotation_sha256"]),
                           (source_path, spec["source_geometry"]["sha256"])):
        if sha256(path) != expected:
            raise SystemExit(f"source changed: {path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"cannot read {image_path}")
    h, w = image.shape[:2]
    coord = spec["coordinate_system"]
    if (w, h) != (coord["width"], coord["height"]) or coord["undistorted"]:
        raise SystemExit("expected raw distorted 1920x1080 pixel coordinates")
    annotations = json.loads(annotation_path.read_text())
    if annotations["image_sha256"] != spec["image_sha256"]:
        raise SystemExit("annotation image hash differs")
    for obj in annotations["objects"]:
        x1, y1, x2, y2 = (round(v) for v in obj["box_xyxy"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (145, 145, 145), 1)

    for zone in spec["zones"]:
        pts = np.asarray(zone["polygon_xy"], dtype=np.int32)
        if len(pts) < 3 or np.any(pts < 0) or np.any(pts[:, 0] >= w) or np.any(pts[:, 1] >= h):
            raise SystemExit(f"invalid polygon: {zone['id']}")
        color = COLORS[zone["role"]]
        fill = image.copy()
        cv2.fillPoly(fill, [pts], color)
        cv2.addWeighted(fill, 0.075, image, 0.925, 0, image)
        cv2.polylines(image, [pts], True, color, 3, cv2.LINE_AA)
        x, y = zone["label_anchor_xy"]
        label = f"{zone['id']} {zone['overlay_label']}"
        cv2.rectangle(image, (x - 5, y - 24), (x + 13 * len(label), y + 8), (0, 0, 0), -1)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)

    for mark in spec["landmarks"]:
        x, y = mark["point_xy"]
        cv2.circle(image, (x, y), 16, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(image, mark["id"], (x + 20, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.rectangle(image, (10, 10), (1010, 88), (0, 0, 0), -1)
    cv2.putText(image, "P01 SEMANTIC ZONES - DRAFT / raw distorted pixels", (22, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Road access is proposed; obstacles and entry points need confirmation", (22, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (185, 185, 185), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(OUT), image):
        raise SystemExit(f"cannot write {OUT}")
    print(OUT)


if __name__ == "__main__":
    main()
