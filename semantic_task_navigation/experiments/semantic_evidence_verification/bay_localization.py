#!/usr/bin/env python3
"""From Qwen's semantic destination to an image-space candidate region, using visual evidence only. No model is run.

Inputs, all STORED: the Qwen L2 proposal (its box = where to look), the Grounding DINO + Qwen-crop-verified truck and cones
(what to keep out of the search and what anchors the region). Nothing typed in by a person: no bay coordinates, no answer region,
not Z06.

Method (classical image processing, no learned model):
  1 search window   = Qwen's proposed box
  2 white paint     = bright, unsaturated pixels whose 31x31 surroundings are dark (paint on the dark mat; the light floor outside the mat
                      and the white truck/cone stripes are excluded: the surroundings test, and the detected truck/cone boxes)
  3 dash components = small connected blobs of that mask
  4 dashed lines    = RANSAC on dash centres: two lines, each >= 5 dashes, regular spacing
  5 candidate bay   = the parallelogram spanned by the two dashed sides from their corner. The two other sides are NOT drawn on the
                      floor: they are the extent of the dashed lines (an open bay), reported as such
  6 checks          = lines perpendicular, corner consistent with both dash runs, the verified truck stands inside, the region lies
                      inside Qwen's box. If a check fails nothing is claimed.
  7 free floor      = v0.2 semantics.placement.placement_region on the bay, with the detected truck and cones as exclusions. Object
                      boxes are subtracted, never treated as placeable. Geometry only: surface, height, reach are unmeasured.

The threshold constants below were chosen by looking at this one photo (E1). `sensitivity()` shows how the result changes with them;
`stability()` repeats the pipeline on the 8 photos of the same session (same layout, so this is a repeatability check, not
independent generalisation).
"""
import copy
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(HERE))

from semantic_task_nav.config import load_config  # noqa: E402
from semantic_task_nav.contracts import OBSERVED, RAW_PX  # noqa: E402
from semantic_task_nav.semantics.placement import placement_region  # noqa: E402

PARAMS = {
    "white_v_min": 185, "white_s_max": 70,        # paint is bright and unsaturated
    "surround_window_px": 31, "surround_gray_max": 110,   # ... on a dark surrounding (the mat), not on the light floor
    "exclude_margin_px": 8,                      # detected truck/cone boxes are removed from the mask, dilated by this
    "dash_area_px": [25, 600], "dash_max_side_px": 40,
    "line_tolerance_px": 4.0, "min_dashes_per_line": 5, "max_spacing_cv": 0.35,
    "min_line_angle_deg": 70, "corner_gap_in_spacings": 1.5,
    "truck_footprint_min_inside": 0.9, "min_inside_qwen_box": 0.9,
}


def _box(b, W, H):
    return [max(0, int(round(b[0]))), max(0, int(round(b[1]))), min(W, int(round(b[2]))), min(H, int(round(b[3])))]


def find_dashes(img, roi, exclude, P):
    H, W = img.shape[:2]
    hsv, g = cv2.cvtColor(img, cv2.COLOR_BGR2HSV), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = P["surround_window_px"]
    m = ((hsv[..., 2] >= P["white_v_min"]) & (hsv[..., 1] <= P["white_s_max"]) & (cv2.blur(g, (k, k)) <= P["surround_gray_max"])).astype(np.uint8)
    for b in exclude:
        x1, y1, x2, y2 = _box(b, W, H)
        e = P["exclude_margin_px"]
        m[max(0, y1 - e):y2 + e, max(0, x1 - e):x2 + e] = 0
    keep = np.zeros_like(m)
    x1, y1, x2, y2 = _box(roi, W, H)
    keep[y1:y2, x1:x2] = 1
    m *= keep
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    lo, hi = P["dash_area_px"]
    comps = [{"center": [float(cen[i][0]), float(cen[i][1])], "area_px": int(st[i, 4]), "size_px": [int(st[i, 2]), int(st[i, 3])],
              "bbox_xyxy": [int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2]), int(st[i, 1] + st[i, 3])]}
             for i in range(1, n) if lo <= st[i, 4] <= hi and max(st[i, 2], st[i, 3]) <= P["dash_max_side_px"]]
    return comps, m


def fit_two_lines(comps, P):
    pts = np.array([c["center"] for c in comps]) if comps else np.zeros((0, 2))
    remaining, lines = list(range(len(pts))), []
    for _ in range(2):
        best = None
        for a, b in combinations(remaining, 2):
            d = pts[b] - pts[a]
            n = float(np.linalg.norm(d))
            if n < 8:
                continue
            nrm = np.array([-d[1], d[0]]) / n
            inl = [i for i in remaining if abs(float((pts[i] - pts[a]) @ nrm)) <= P["line_tolerance_px"]]
            if best is None or len(inl) > len(best):
                best = inl
        if not best or len(best) < P["min_dashes_per_line"]:
            break
        p = pts[best].astype(np.float32)
        vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(p, cv2.DIST_L2, 0, 0.01, 0.01).ravel()]
        d = np.array([vx, vy])
        t = (pts[best] - np.array([x0, y0])) @ d
        order = np.argsort(t)
        ts = t[order]
        sp = np.diff(ts)
        resid = np.abs((pts[best] - np.array([x0, y0])) @ np.array([-vy, vx]))
        lines.append({"inliers": [int(best[i]) for i in order], "n_dashes": len(best), "point": [x0, y0], "direction": [vx, vy],
                      "t_min": float(ts[0]), "t_max": float(ts[-1]), "median_spacing_px": round(float(np.median(sp)), 2),
                      "spacing_cv": round(float(np.std(sp) / np.mean(sp)), 3), "max_residual_px": round(float(resid.max()), 2),
                      "angle_deg": round(float(np.degrees(np.arctan2(vy, vx))) % 180, 1)})
        remaining = [i for i in remaining if i not in best]
    return lines


def _at(line, t):
    return np.array(line["point"]) + t * np.array(line["direction"])


def _intersect(l1, l2):
    p, d = np.array(l1["point"]), np.array(l1["direction"])
    q, e = np.array(l2["point"]), np.array(l2["direction"])
    A = np.array([d, -e]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t, s = np.linalg.solve(A, q - p)
    return p + t * d, float(t), float(s)


def build_bay(lines, comps, P):
    """Corner + the two dashed sides -> parallelogram. Returns (polygon | None, geometry, checks)."""
    checks = []

    def chk(name, ok, detail):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        return ok
    chk("two dashed lines found, each with enough dashes", len(lines) == 2, [l["n_dashes"] for l in lines])
    if len(lines) < 2:
        return None, {}, checks
    l1, l2 = lines
    chk("dash spacing is regular on both lines", all(l["spacing_cv"] <= P["max_spacing_cv"] for l in lines), [l["spacing_cv"] for l in lines])
    ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(l1["direction"], l2["direction"])))))))   # angle between the lines, 0..90
    chk("the two lines are roughly perpendicular", ang >= P["min_line_angle_deg"], round(ang, 1))
    inter = _intersect(l1, l2)
    if inter is None:
        chk("the lines intersect", False, "parallel")
        return None, {}, checks
    C, t1, t2 = inter
    sp = max(l1["median_spacing_px"], l2["median_spacing_px"])
    gaps = [min(abs(t1 - l1["t_min"]), abs(t1 - l1["t_max"])) if not (l1["t_min"] <= t1 <= l1["t_max"]) else 0.0,
            min(abs(t2 - l2["t_min"]), abs(t2 - l2["t_max"])) if not (l2["t_min"] <= t2 <= l2["t_max"]) else 0.0]
    chk("the corner lies at/near the end of both dash runs", all(g <= P["corner_gap_in_spacings"] * sp for g in gaps),
        {"gap_px": [round(g, 1) for g in gaps], "limit_px": round(P["corner_gap_in_spacings"] * sp, 1)})
    # far end of each side = the end of the dash run farther from the corner (dash half-length added)
    half = float(np.median([max(c["size_px"]) for c in comps])) / 2
    ends = []
    for l, t in ((l1, t1), (l2, t2)):
        far = l["t_max"] if abs(l["t_max"] - t) > abs(l["t_min"] - t) else l["t_min"]
        far += half if far == l["t_max"] else -half
        ends.append(_at(l, far))
    A, B = ends
    D = A + B - C
    poly = [C, A, D, B]                         # corner, far end of side 1, opposite corner, far end of side 2
    geo = {"corner_xy": C.round(1).tolist(), "side1_far_end_xy": A.round(1).tolist(), "side2_far_end_xy": B.round(1).tolist(),
           "opposite_vertex_xy": D.round(1).tolist(), "side_lengths_px": [round(float(np.linalg.norm(A - C)), 1), round(float(np.linalg.norm(B - C)), 1)],
           "open_sides_note": "the sides A-D and B-D are NOT painted: they are the extent of the dashed lines (the bay is open there)"}
    return np.array(poly), geo, checks


def polygon_checks(poly, truck, qwen_box, W, H, P):
    out = []
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    area = int(mask.sum())
    qb = np.zeros_like(mask)
    x1, y1, x2, y2 = _box(qwen_box, W, H)
    qb[y1:y2, x1:x2] = 1
    inside_q = float((mask & qb).sum() / max(1, area))
    out.append({"check": "the region lies inside Qwen's proposed box", "status": "PASS" if inside_q >= P["min_inside_qwen_box"] else "FAIL",
                "detail": round(inside_q, 3)})
    xs = np.linspace(truck[0], truck[2], 21)
    inside_pts = [cv2.pointPolygonTest(np.round(poly).astype(np.float32).reshape(-1, 1, 2), (float(x), float(truck[3] - 2)), False) >= 0 for x in xs]
    frac = float(np.mean(inside_pts))
    out.append({"check": "the verified truck's ground edge (bottom of its box) stands inside the region",
                "status": "PASS" if frac >= P["truck_footprint_min_inside"] else "FAIL", "detail": round(frac, 3)})
    return out, area, mask


def free_floor(poly, objects, size, v02cfg, margin):
    cfg = copy.deepcopy(v02cfg)
    cfg["placement"]["exclude_detected_classes"] = sorted(set(cfg["placement"]["exclude_detected_classes"]) | {"traffic_cone"})
    cfg["placement"]["default_margin_px"] = margin
    region = {"region_id": "bay_candidate", "polygon_xy": np.round(poly, 1).tolist()}
    objs = [{"object_id": o["object_id"], "class_name": o["class_name"], "bbox_xyxy": o["bbox_xyxy"], "accepted": True,
             "knowledge": OBSERVED, "coordinate_frame": RAW_PX} for o in objects if o["class_name"] in ("truck", "traffic_cone", "conveyor", "mobile_robot")]
    rec = placement_region(region, size, objs, None, None, cfg)
    # v0.2 keeps only the EXTERNAL contour of the free area, so an object standing fully inside the region leaves no hole in
    # polygons_xy (v0.2 is not modified). Detect that: the polygon fill must not exceed the exact free area by more than contour
    # simplification (2 %). For margin 0 the exact free area is areas_px["after_object_footprints"].
    exact, filled = rec["areas_px"]["after_object_footprints"], rec["area_px"]
    rec["hole_check"] = {"exact_free_px_after_object_boxes": exact, "polygon_fill_px": filled,
                         "interior_hole_lost": bool(margin == 0 and filled > 1.02 * exact),
                         "note": "polygons_xy of an object fully inside the region would NOT exclude it" if margin == 0 and filled > 1.02 * exact else "ok"}
    return rec


def localize(img, qwen_box, objects, v02cfg, P=PARAMS):
    H, W = img.shape[:2]
    truck = next(o["bbox_xyxy"] for o in objects if o["class_name"] == "truck")
    exclude = [o["bbox_xyxy"] for o in objects if o["class_name"] in ("truck", "traffic_cone")]
    comps, mask = find_dashes(img, qwen_box, exclude, P)
    lines = fit_two_lines(comps, P)
    poly, geo, checks = build_bay(lines, comps, P)
    res = {"params": P, "n_dash_components": len(comps), "dash_components": comps, "lines": [{k: v for k, v in l.items() if k != "inliers"} |
           {"dash_indices": l["inliers"]} for l in lines], "checks": checks, "bay_geometry": geo, "status": "FLOOR_MARKING_NOT_ESTABLISHED",
           "bay_polygon_xy": None, "free_floor": None}
    if poly is not None:
        pc, area, pmask = polygon_checks(poly, truck, qwen_box, W, H, P)
        res["checks"] += pc
        res["bay_area_px"] = area
    if poly is not None and all(c["status"] == "PASS" for c in res["checks"]):
        res["status"], res["bay_polygon_xy"] = "FLOOR_MARKING_ESTABLISHED", np.round(poly, 1).tolist()
        res["free_floor"] = {f"margin_{m}px": {k: v for k, v in free_floor(poly, objects, (W, H), v02cfg, m).items()
                                                if k in ("status", "area_px", "polygons_xy", "margin_px", "areas_px", "excluded", "basis", "feasibility", "hole_check")}
                             for m in (0, 8)}
    return res, mask


def sensitivity(img, qwen_box, objects, v02cfg):
    rows = []
    for name, upd in (("as used", {}), ("white_v_min 170 / s_max 80 / surround 120", {"white_v_min": 170, "white_s_max": 80, "surround_gray_max": 120}),
                      ("white_v_min 200 / s_max 60 / surround 100", {"white_v_min": 200, "white_s_max": 60, "surround_gray_max": 100}),
                      ("white_v_min 195", {"white_v_min": 195}), ("white_v_min 175", {"white_v_min": 175}),
                      ("surround_gray_max 100", {"surround_gray_max": 100}), ("surround_gray_max 125", {"surround_gray_max": 125}),
                      ("line_tolerance 3", {"line_tolerance_px": 3.0}), ("line_tolerance 6", {"line_tolerance_px": 6.0})):
        P = {**PARAMS, **upd}
        r, _ = localize(img, qwen_box, objects, v02cfg, P)
        rows.append({"variant": name, "n_dash_components": r["n_dash_components"], "dashes_per_line": [l["n_dashes"] for l in r["lines"]],
                     "status": r["status"], "failed_checks": [c["check"] for c in r["checks"] if c["status"] == "FAIL"],
                     "corner_xy": r["bay_geometry"].get("corner_xy"), "bay_area_px": r.get("bay_area_px")})
    return rows


def load_cfg():
    return load_config()["planning"]
