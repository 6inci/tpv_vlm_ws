"""Layer rendering for the GUI / replay and paper figures.

Rendering is a pure function of the stored run data (result.json, raw frame, ROI images, map
spec), so a replay reproduces the saved figures pixel for pixel. Colours have one meaning in
every view. Manual inputs are drawn dashed and tagged [MANUAL]; model outputs are solid.
Labels are ASCII (OpenCV Hershey fonts).
"""
import math

import cv2
import numpy as np

from .geometry.coordinate_frames import RoiTransform, box_center

COL = {  # BGR
    "conveyor": (0, 140, 255), "truck": (255, 128, 0), "mobile_robot": (255, 255, 0), "target": (0, 0, 255),
    "rejected": (150, 150, 150), "uncertain": (0, 215, 255), "region_candidate": (0, 200, 255),
    "selected_region": (255, 0, 255), "prior_zone": (230, 230, 230), "traversable": (60, 180, 75),
    "static_obstacle": (30, 30, 30), "no_go": (60, 20, 200), "dynamic_obstacle": (0, 90, 200),
    "valid": (60, 180, 75), "footprint_blocked": (80, 80, 230), "blocked": (20, 20, 20),
    "accessible": (180, 255, 180), "path_a": (255, 200, 0), "path_b": (200, 60, 200), "raw_path": (210, 210, 210),
    "start": (0, 255, 0), "pickup_goal": (0, 165, 255), "dropoff_goal": (255, 0, 255), "relation": (255, 255, 255),
    "placement": (0, 230, 255), "approach": (255, 255, 0), "reference": (255, 255, 255),
}
VIEWS = ("RAW", "UNDISTORTED", "OBJECT_DETECTION", "RED_CUBE_ROI", "OBJECT_RELATIONSHIP", "SEMANTIC_LOADING_REGION",
         "TRAVERSABLE_MAP", "OCCUPANCY_MAP", "PICKUP_PATH", "DELIVERY_PATH", "FINAL_RESULT", "DESTINATION_CHAIN")
# v0.2 layers (approach_region, placement_region, reference_regions) only draw v0.2 data; the relative draw
# order of the v0.1 layers is unchanged, so v0.1 runs still replay to identical figures.
LAYERS = ("traversable", "obstacles", "occupancy", "accessible", "prior_zones", "region_candidates", "selected_region",
          "approach_region", "placement_region", "detections", "rejected_candidates", "target", "relations", "raw_paths",
          "path_a", "path_b", "start_goals", "reference_regions", "legend")
VIEW_LAYERS = {
    "RAW": [], "UNDISTORTED": [],
    "OBJECT_DETECTION": ["detections", "rejected_candidates", "target", "legend"],
    "RED_CUBE_ROI": ["legend"],
    "OBJECT_RELATIONSHIP": ["detections", "target", "relations", "legend"],
    "SEMANTIC_LOADING_REGION": ["detections", "region_candidates", "selected_region", "legend"],
    "TRAVERSABLE_MAP": ["traversable", "obstacles", "start_goals", "legend"],
    "OCCUPANCY_MAP": ["occupancy", "start_goals", "legend"],
    "PICKUP_PATH": ["traversable", "obstacles", "accessible", "target", "raw_paths", "path_a", "start_goals", "legend"],
    "DELIVERY_PATH": ["traversable", "obstacles", "selected_region", "raw_paths", "path_b", "start_goals", "legend"],
    "FINAL_RESULT": ["traversable", "obstacles", "detections", "target", "selected_region", "path_a", "path_b",
                     "start_goals", "legend"],
    "DESTINATION_CHAIN": ["traversable", "obstacles", "region_candidates", "selected_region", "approach_region",
                          "placement_region", "detections", "path_b", "start_goals", "legend"],
}
FIGURES = (("01_raw", "RAW"), ("02_detections", "OBJECT_DETECTION"), ("03_red_cube_roi", "RED_CUBE_ROI"),
           ("04_semantic_regions", "SEMANTIC_LOADING_REGION"), ("05_occupancy_map", "OCCUPANCY_MAP"),
           ("06_pickup_path", "PICKUP_PATH"), ("07_delivery_path", "DELIVERY_PATH"),
           ("08_final_overview", "FINAL_RESULT"), ("09_object_relationship", "OBJECT_RELATIONSHIP"),
           ("10_traversable_map", "TRAVERSABLE_MAP"), ("11_destination_chain", "DESTINATION_CHAIN"))
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Canvas:
    def __init__(self, img):
        self.img = img.copy()
        self.legend = {}

    def note(self, label, color, style="box"):
        self.legend.setdefault(label, (color, style))


# ------------------------------------------------------------------ primitives
def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def _text(img, text, org, color, scale=0.6, thick=1):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    y = max(th + 4, y)
    x = min(max(0, x), max(0, img.shape[1] - tw - 4))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), FONT, scale, color, thick, cv2.LINE_AA)


def _dashed_line(img, p1, p2, color, thick, dash=10):
    d = math.dist(p1, p2)
    n = max(1, int(d // dash))
    for i in range(0, n, 2):
        a = (p1[0] + (p2[0] - p1[0]) * i / n, p1[1] + (p2[1] - p1[1]) * i / n)
        b = (p1[0] + (p2[0] - p1[0]) * min(i + 1, n) / n, p1[1] + (p2[1] - p1[1]) * min(i + 1, n) / n)
        cv2.line(img, _pt(a), _pt(b), color, thick, cv2.LINE_AA)


def _poly(img, pts, color, thick, dashed=False):
    pts = [tuple(p) for p in pts]
    if dashed:
        for a, b in zip(pts, pts[1:] + pts[:1]):
            _dashed_line(img, a, b, color, thick)
    else:
        cv2.polylines(img, [np.round(np.array(pts)).astype(np.int32)], True, color, thick, cv2.LINE_AA)


def _box(img, b, color, thick, dashed=False):
    _poly(img, [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])], color, thick, dashed)


def _blend(img, mask, color, alpha):
    if mask is None or not mask.any():
        return
    over = img.copy()
    over[mask] = color
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, dst=img)


def _grid_to_full(mask, cell_px, w, h):
    full = np.repeat(np.repeat(mask, cell_px, axis=0), cell_px, axis=1)
    out = np.zeros((h, w), bool)
    out[:full.shape[0], :full.shape[1]] = full[:h, :w]
    return out


def _is_manual(obj):
    return obj.get("knowledge") in ("manual_test_data", "manual_prior") or obj.get("source") in (
        "manual_test_data", "manual_prior")


def _polyline(img, pts, color, thick, dashed=False, arrow=True):
    for a, b in zip(pts, pts[1:]):
        if dashed:
            _dashed_line(img, a, b, color, thick, dash=8)
        else:
            cv2.line(img, _pt(a), _pt(b), color, thick, cv2.LINE_AA)
    if arrow and len(pts) >= 2 and not dashed:
        cv2.arrowedLine(img, _pt(pts[-2]), _pt(pts[-1]), color, thick, cv2.LINE_AA, tipLength=min(
            0.5, 30.0 / max(1.0, math.dist(pts[-2], pts[-1]))))


# ------------------------------------------------------------------ layers
def _objects(state):
    return (state["result"].get("detections") or {}).get("objects", [])


def _target(state):
    return (state["result"].get("target_inspection") or {}).get("target")


def draw_detections(cv, state):
    for o in _objects(state):
        if not o.get("accepted"):
            continue
        col = COL.get(o["class_name"], (255, 255, 255))
        man = _is_manual(o)
        _box(cv.img, o["bbox_xyxy"], col, 3, dashed=man)
        score = "" if o.get("score") is None else f" s={o['score']:.2f}"
        ver = " [MANUAL]" if man else (" verified" if (o.get("verification") or {}).get("status") == "yes" else "")
        _text(cv.img, f"{o['object_id']}{score}{ver}", (o["bbox_xyxy"][0], o["bbox_xyxy"][1] - 4), col)
        cv.note(f"{o['class_name']} ({'manual' if man else 'detector + VLM verified'})", col, "dash" if man else "box")


def draw_rejected(cv, state):
    for o in _objects(state):
        if o.get("accepted"):
            continue
        _box(cv.img, o["bbox_xyxy"], COL["rejected"], 1)
        st = (o.get("verification") or {}).get("status", "?")
        _text(cv.img, f"x {o['object_id']} ({st})", (o["bbox_xyxy"][0], o["bbox_xyxy"][3] + 16), COL["rejected"], 0.45)
        cv.note("rejected / unverified candidate", COL["rejected"])


def draw_target(cv, state):
    t = _target(state)
    if not t:
        return
    man = _is_manual(t)
    b = t["bbox_xyxy"]
    pad = 6
    _box(cv.img, [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad], COL["target"], 3, dashed=man)
    _text(cv.img, f"TARGET {t['object_id']}{' [MANUAL]' if man else ''}", (b[0] - pad, b[1] - pad - 6), COL["target"])
    cv.note(f"target ({'manual' if man else 'VLM-verified candidate'})", COL["target"], "dash" if man else "box")


def draw_relations(cv, state):
    sg = state["result"].get("scene_graph") or {}
    nodes = {n["id"]: n for n in sg.get("nodes", []) if n.get("bbox_xyxy")}
    for i, r in enumerate(sg.get("relations", [])):
        a, b = nodes.get(r["subject"]), nodes.get(r["object"])
        if not a or not b:
            continue
        pa, pb = box_center(a["bbox_xyxy"]), box_center(b["bbox_xyxy"])
        dashed = r["status"] != "COMPUTED"
        (_dashed_line if dashed else lambda im, p, q, c, t: cv2.line(im, _pt(p), _pt(q), c, t, cv2.LINE_AA))(
            cv.img, pa, pb, COL["relation"], 2)
        mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2 + 18 * (i % 3))
        _text(cv.img, f"{r['type']} [{r['status']}]", mid, COL["relation"], 0.5)
    if sg.get("relations"):
        cv.note("relation: computed (image px)", COL["relation"], "line")
        cv.note("relation: unverified / inferred", COL["relation"], "dashline")


def draw_region_candidates(cv, state):
    rg = state["result"].get("region_grounding") or {}
    for c in rg.get("loading_area_candidates", []):
        _poly(cv.img, c["polygon_xy"], COL["region_candidate"], 2, dashed=True)
        cx, cy = np.mean(np.array(c["polygon_xy"]), axis=0)
        _text(cv.img, f"cand {c['candidate_id']}", (cx - 30, cy), COL["region_candidate"], 0.55)
        if "source" in c:      # v0.2 record: semantic judgement only (no accessibility mixed in)
            _text(cv.img, f"VLM:{c.get('vlm_plausible')} [{c['source'].split(':')[0]}]", (cx - 30, cy + 22),
                  COL["region_candidate"], 0.45)
        elif "vlm_plausible" in c:
            acc = "-" if c.get("accessible_cells") is None else c["accessible_cells"]
            _text(cv.img, f"VLM:{c['vlm_plausible']} access:{acc}", (cx - 30, cy + 22), COL["region_candidate"], 0.45)
    if rg.get("loading_area_candidates"):
        cv.note("loading-area candidate (geometric)", COL["region_candidate"], "dash")


def draw_selected_region(cv, state):
    rg = state["result"].get("region_grounding") or {}
    reg = rg.get("selected_loading_region")
    if not reg:
        if rg.get("status"):
            _text(cv.img, f"LOADING AREA: {rg['status']}", (20, 60), COL["selected_region"], 0.7, 2)
        return
    man = reg["verification_status"] == "manual_unconfirmed"
    mask = np.zeros(cv.img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(np.array(reg["polygon_xy"])).astype(np.int32)], 1)
    _blend(cv.img, mask.astype(bool), COL["selected_region"], 0.25)
    _poly(cv.img, reg["polygon_xy"], COL["selected_region"], 3, dashed=man)
    x = min(p[0] for p in reg["polygon_xy"])
    y = max(p[1] for p in reg["polygon_xy"])
    tag = "[MANUAL]" if man else f"[{reg['verification_status'].upper()}]"
    _text(cv.img, f"LOADING AREA {tag}", (x, y + 22), COL["selected_region"], 0.6, 2)
    cv.note("loading area " + ("(manual)" if man else "(VLM-selected, unverified)"), COL["selected_region"], "fill")


def draw_prior_zones(cv, state):
    for z in ((state["result"].get("map") or {}).get("spec") or {}).get("semantic_reference_zones", []):
        _poly(cv.img, z["polygon"], COL["prior_zone"], 1, dashed=True)
        _text(cv.img, f"prior {z['id']} {z['role']} ({'confirmed' if z['confirmed'] else 'unconfirmed'})",
              z["polygon"][0], COL["prior_zone"], 0.45)
        cv.note("prior map zone [MANUAL] (evaluation only)", COL["prior_zone"], "dash")


def draw_traversable(cv, state):
    nav = state.get("nav")
    if nav is None:
        return
    _blend(cv.img, nav.full_layers["traversable"], COL["traversable"], 0.18)
    for z in nav.spec["traversable"]:
        _poly(cv.img, z["polygon"], COL["traversable"], 2, dashed=True)
    cv.note("traversable region [MANUAL map]", COL["traversable"], "fill")


def draw_obstacles(cv, state):
    nav = state.get("nav")
    if nav is None:
        return
    for key, alpha, label in (("static_obstacle", 0.55, "static obstacle [MANUAL map]"),
                              ("no_go", 0.35, "no-go region [MANUAL map]"),
                              ("dynamic_obstacle", 0.45, "dynamic obstacle (this frame)")):
        if nav.full_layers[key].any():
            _blend(cv.img, nav.full_layers[key], COL[key], alpha)
            cv.note(label, COL[key], "fill")
    for z in nav.spec["static_obstacles"]:
        _poly(cv.img, z["polygon"], (200, 200, 200), 1, dashed=True)
    for z in nav.spec["no_go"]:
        _poly(cv.img, z["polygon"], COL["no_go"], 1, dashed=True)


def draw_occupancy(cv, state):
    nav = state.get("nav")
    if nav is None:
        return
    h, w = cv.img.shape[:2]
    _blend(cv.img, _grid_to_full(nav.valid, nav.cell_px, w, h), COL["valid"], 0.35)
    _blend(cv.img, _grid_to_full(nav.free & ~nav.valid, nav.cell_px, w, h), COL["footprint_blocked"], 0.45)
    _blend(cv.img, _grid_to_full(~nav.free, nav.cell_px, w, h), COL["blocked"], 0.45)
    cv.note(f"valid robot centre (grid {nav.cell_px}px)", COL["valid"], "fill")
    cv.note(f"free but < robot radius {nav.robot_radius_px}px", COL["footprint_blocked"], "fill")
    cv.note("not traversable / obstacle", COL["blocked"], "fill")


def draw_accessible(cv, state):
    nav = state.get("nav")
    if nav is None:
        return
    goals = state["result"].get("goals") or {}
    for key in ("pickup", "dropoff"):
        g = goals.get(key) or {}
        for r, c in g.get("accessible_cells", []):
            x, y = nav.cell_center(r, c)
            cv2.circle(cv.img, _pt((x, y)), 2, COL["accessible"], -1)
    if any((goals.get(k) or {}).get("accessible_cells") for k in ("pickup", "dropoff")):
        cv.note("accessible + reachable goal cells", COL["accessible"], "dot")


def draw_raw_paths(cv, state):
    for key in ("A", "B"):
        p = (state["result"].get("paths") or {}).get(key) or {}
        if p.get("raw_waypoints_xy"):
            _polyline(cv.img, p["raw_waypoints_xy"], COL["raw_path"], 1, dashed=True)
            cv.note("raw A* path (before simplification)", COL["raw_path"], "dashline")


def _draw_path(cv, state, key, color, label):
    p = (state["result"].get("paths") or {}).get(key) or {}
    if not p:
        return
    if p.get("status") != "PATH_FOUND":
        _text(cv.img, f"PATH {key}: {p.get('status')}", (20, 90 if key == "A" else 120), color, 0.7, 2)
        return
    pts = p["waypoints_xy"]
    _polyline(cv.img, pts, (0, 0, 0), 7)
    _polyline(cv.img, pts, color, 4)
    for q in pts:
        cv2.circle(cv.img, _pt(q), 5, color, -1)
    v = ((state["result"].get("validation") or {}).get(key) or {}).get("valid")
    st = " valid" if v else (" INVALID" if v is False else "")
    extra = ""
    if key == "B":
        extra = f" [{(state['result'].get('handoff') or {}).get('path_b_status', '')}]"
    cv.note(f"{label} ({p['length_px']:.0f}px{st}){extra}", color, "line")


def draw_start_goals(cv, state):
    res = state["result"]
    nav = state.get("nav")
    r = nav.robot_radius_px if nav is not None else 20
    s = res.get("start")
    if s:
        man = s["knowledge"] == "manual_prior"
        cv2.circle(cv.img, _pt(s["xy"]), int(r), COL["start"], 2, cv2.LINE_AA)
        cv2.circle(cv.img, _pt(s["xy"]), 6, COL["start"], -1)
        _text(cv.img, "START" + (" [MANUAL]" if man else " (detected)"), (s["xy"][0] - r, s["xy"][1] + r + 22),
              COL["start"], 0.6, 2)
        cv.note("robot start (footprint)" + (" [MANUAL]" if man else ""), COL["start"], "circle")
    for key, col, label in (("pickup", COL["pickup_goal"], "PICKUP APPROACH"),
                            ("dropoff", COL["dropoff_goal"], "DROPOFF APPROACH")):
        g = (res.get("goals") or {}).get(key) or {}
        pose = g.get("pose")
        if not pose:
            continue
        x, y = pose["xy"]
        cv2.circle(cv.img, _pt((x, y)), int(r), col, 2, cv2.LINE_AA)
        if pose.get("heading_deg") is not None:
            a = math.radians(pose["heading_deg"])
            cv2.arrowedLine(cv.img, _pt((x, y)), _pt((x + r * 1.3 * math.cos(a), y + r * 1.3 * math.sin(a))), col, 2,
                            cv2.LINE_AA, tipLength=0.35)
        cv2.drawMarker(cv.img, _pt((x, y)), col, cv2.MARKER_STAR, 22, 2)
        _text(cv.img, label, (x - r, y - r - 8), col, 0.6, 2)
        cv.note(f"{label.lower()} (robot footprint + heading)", col, "circle")


def _destination(state):
    return state["result"].get("destination") or {}


def draw_placement_region(cv, state):
    d = _destination(state)
    if not d:
        return
    chain = (f"DESTINATION CHAIN: semantic={len(d.get('semantic_regions_plausible') or [])} plausible | final="
             f"{d.get('final_stage_status')} | failed stage={d.get('failed_stage')}")
    _text(cv.img, chain, (20, 60), COL["placement"], 0.6, 2)
    p = d.get("object_placement_region")
    if not isinstance(p, dict) or not p.get("polygons_xy"):
        return
    mask = np.zeros(cv.img.shape[:2], np.uint8)
    for poly in p["polygons_xy"]:
        cv2.fillPoly(mask, [np.round(np.array(poly)).astype(np.int32)], 1)
    _blend(cv.img, mask.astype(bool), COL["placement"], 0.55)
    for poly in p["polygons_xy"]:
        _poly(cv.img, poly, COL["placement"], 2)
    x = min(q[0] for poly in p["polygons_xy"] for q in poly)
    y = max(q[1] for poly in p["polygons_xy"] for q in poly)
    _text(cv.img, "PLACEMENT (geometric)", (x, y + 46), COL["placement"], 0.55, 2)
    cv.note("object placement region (geometric; surface & reach unmeasured)", COL["placement"], "fill")


def draw_approach_region(cv, state):
    d, nav = _destination(state), state.get("nav")
    a = d.get("robot_approach_region") if d else None
    if not a or nav is None:
        return
    for r, c in a["cells_rc"]:
        x, y = nav.cell_center(r, c)
        h = nav.cell_px // 2
        cv2.rectangle(cv.img, _pt((x - h + 1, y - h + 1)), _pt((x + h - 1, y + h - 1)), COL["approach"], 1)
    cv.note(f"robot approach region (reach {a['reach']['status']}: {a['reach']['reach_px']}px)", COL["approach"], "box")


def draw_reference_regions(cv, state):
    ref = state.get("reference") or {}
    items = ([("REF loading area [HUMAN]", ref["loading_region_polygon"])] if ref.get("loading_region_polygon") else [])
    items += [("REF placement [HUMAN]", p) for p in ref.get("placement_region_polygons") or []]
    for label, poly in items:
        _poly(cv.img, poly, COL["reference"], 2, dashed=True)
        _text(cv.img, label, poly[0], COL["reference"], 0.5)
    if items:
        cv.note("human reference region (evaluation only; never given to the models)", COL["reference"], "dash")


DRAW = {"traversable": draw_traversable, "placement_region": draw_placement_region,
        "approach_region": draw_approach_region, "reference_regions": draw_reference_regions, "obstacles": draw_obstacles, "occupancy": draw_occupancy,
        "accessible": draw_accessible, "prior_zones": draw_prior_zones, "region_candidates": draw_region_candidates,
        "selected_region": draw_selected_region, "detections": draw_detections, "rejected_candidates": draw_rejected,
        "target": draw_target, "relations": draw_relations, "raw_paths": draw_raw_paths,
        "path_a": lambda cv, s: _draw_path(cv, s, "A", COL["path_a"], "path A: start -> pickup"),
        "path_b": lambda cv, s: _draw_path(cv, s, "B", COL["path_b"], "path B: pickup -> dropoff"),
        "start_goals": draw_start_goals}


# ------------------------------------------------------------------ views
def _message(shape, lines):
    img = np.full(shape, 40, np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (40, 80 + 40 * i), FONT, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
    return img


def _render_roi(state, cv_note=None):
    ti = state["result"].get("target_inspection") or {}
    rois = ti.get("rois") or []
    imgs = state.get("roi_images") or {}
    if not rois or not imgs:
        return Canvas(_message(state["frame"].shape, ["RED CUBE ROI: not available", ti.get("status", "not run")]))
    target = ti.get("target") or {}
    idx = next((c.get("roi_index") for c in ti.get("candidates", []) if c.get("candidate_id") == target.get(
        "candidate_id")), None)
    roi_rec = next((r for r in rois if r["roi_index"] == idx), rois[0])   # full-image detector candidates: ROI 1
    idx = roi_rec["roi_index"]
    tf = RoiTransform(roi_rec["transform"]["crop_xyxy_raw"], roi_rec["transform"]["scale"])
    cv = Canvas(imgs[idx])
    colors = {"yes": COL["target"], "no": COL["rejected"], "uncertain": COL["uncertain"],
              "invalid_response": COL["uncertain"], "not_run": (255, 255, 255)}
    x0, y0, x1, y1 = tf.crop_xyxy
    for c in ti.get("candidates", []):
        cx, cy = box_center(c["bbox_xyxy"])
        if not (x0 <= cx < x1 and y0 <= cy < y1):
            continue
        st = c["verification"]["status"]
        b = tf.raw_to_roi_box(c["bbox_xyxy"])
        _box(cv.img, b, colors.get(st, (255, 255, 255)), 3 if st == "yes" else 2, dashed=st == "not_run")
        _text(cv.img, f"{c['candidate_id']} {st} ({'+'.join(sorted(set(c['sources'])))})", (b[0], b[1] - 6),
              colors.get(st, (255, 255, 255)), 0.6)
        cv.note(f"candidate: VLM answer '{st}'", colors.get(st, (255, 255, 255)), "dash" if st == "not_run" else "box")
    _text(cv.img, f"ROI {idx} around {roi_rec['anchor']} | zoom x{tf.scale:.2f} | roi_local_px -> raw = offset + roi/scale",
          (10, cv.img.shape[0] - 10), (255, 255, 255), 0.6)
    return cv


def _layers_canvas(state, view, layers):
    if view == "RED_CUBE_ROI":
        return _render_roi(state)
    if view == "UNDISTORTED":
        und = state.get("undistorted")
        if und is None:
            reasons = (state["result"].get("calibration") or {}).get("reasons", [])
            return Canvas(_message(state["frame"].shape, ["UNDISTORTED VIEW UNAVAILABLE"] + reasons[:6]))
        cv = Canvas(und)
        _text(cv.img, "undistorted_image_px (view only; intrinsics validity for this install UNVERIFIED)",
              (10, cv.img.shape[0] - 10), (255, 255, 255), 0.6)
        return cv
    cv = Canvas(state["frame"])
    for name in LAYERS:
        if name in layers and name in DRAW:
            DRAW[name](cv, state)
    return cv


def _legend_overlay(cv):
    if not cv.legend:
        return
    items = list(cv.legend.items())
    h = 24 * len(items) + 12
    w = 20 + max(cv2.getTextSize(k, FONT, 0.5, 1)[0][0] for k, _ in items) + 40
    y0 = cv.img.shape[0] - h - 10
    cv2.rectangle(cv.img, (10, y0), (10 + w, y0 + h), (0, 0, 0), -1)
    for i, (label, (col, style)) in enumerate(items):
        y = y0 + 18 + 24 * i
        _legend_symbol(cv.img, (20, y - 6), col, style)
        cv2.putText(cv.img, label, (52, y), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _legend_symbol(img, at, col, style):
    x, y = at
    if style in ("line", "dashline"):
        (_dashed_line if style == "dashline" else lambda im, a, b, c, t: cv2.line(im, a, b, c, t))(
            img, (x, y), (x + 24, y), col, 3)
    elif style == "circle":
        cv2.circle(img, (x + 12, y), 7, col, 2)
    elif style == "dot":
        cv2.circle(img, (x + 12, y), 3, col, -1)
    elif style == "fill":
        cv2.rectangle(img, (x, y - 7), (x + 24, y + 7), col, -1)
    else:
        _box(img, [x, y - 7, x + 24, y + 7], col, 2, dashed=style == "dash")


def banner_text(state, view):
    r = state["result"]
    return (f"{view} | {r.get('map_mode', '?')} (image px, not metric) | frame #{(r.get('frame') or {}).get('frame_id')} "
            f"| {r.get('analysis_mode')} | run {r.get('run_id')}")


def render_view(state, view, layers=None, overlay_legend=True, banner=True):
    layers = VIEW_LAYERS[view] if layers is None else list(layers)
    cv = _layers_canvas(state, view, layers)
    if banner:
        _text(cv.img, banner_text(state, view), (10, 24), (255, 255, 255), 0.55)
    if overlay_legend and "legend" in layers:
        _legend_overlay(cv)
    return cv.img


def _wrap(text, max_w, scale):
    out, cur = [], ""
    for word in text.split(" "):
        trial = (cur + " " + word).strip()
        if cur and cv2.getTextSize(trial, FONT, scale, 1)[0][0] > max_w:
            out.append(cur)
            cur = word
        else:
            cur = trial
    return out + [cur]


def paper_figure(state, view):
    """Original-resolution figure with the legend in a strip below the image (image not covered)."""
    cv = _layers_canvas(state, view, VIEW_LAYERS[view])
    img = cv.img
    items = list(cv.legend.items())
    ncol = 3 if img.shape[1] >= 1500 else (2 if img.shape[1] >= 900 else 1)
    rows = math.ceil(len(items) / ncol)
    lines = _wrap(banner_text(state, view) + " | dashed/[MANUAL] = manual input, solid = algorithm/model output",
                  img.shape[1] - 24, 0.55)
    top = 16 + 26 * len(lines)
    strip = np.full((top + 16 + 30 * rows, img.shape[1], 3), 255, np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(strip, ln, (12, 26 + 26 * i), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    colw = img.shape[1] // ncol
    for i, (label, (col, style)) in enumerate(items):
        x, y = 12 + (i % ncol) * colw, top + 18 + 30 * (i // ncol)
        _legend_symbol(strip, (x, y - 6), col, style)
        cv2.putText(strip, label, (x + 36, y), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, strip])


def export_paper_figures(state, out_dir, writer=None):
    """Writes every figure as a new file. Returns {name: path}."""
    out = {}
    for name, view in FIGURES:
        fig = paper_figure(state, view)
        rel = f"{out_dir}/{name}.png"
        out[name] = str(writer.write_image(rel, fig)) if writer else (cv2.imwrite(rel, fig) and rel)
    return out
