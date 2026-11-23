#!/usr/bin/env python3
"""Run bay_localization on the STORED E1 L2 result and write a new, separate run record. No model is called.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  PYTHONDONTWRITEBYTECODE=1 $PY experiments/semantic_evidence_verification/localize_bay.py

Writes results/<timestamp>_bay_localization/ : localization.json, input_provenance.json, sensitivity.json, stability_8_photos.json,
01_qwen_vs_python_region.png, 02_bay_zoom.png. Existing results are never touched.
"""
import glob
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bay_localization as B  # noqa: E402
import observations as O  # noqa: E402
import verifier2 as V  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
CAL = Path("/home/binci/Pictures/Calibration")
COL = {"qwen": (0, 165, 255), "truck": (255, 255, 255), "cone": (0, 110, 255), "dash": (0, 255, 255), "line": (255, 255, 0),
       "bay": (60, 200, 60), "free": (255, 60, 255)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _t(img, text, org, color, scale=0.55):
    text = str(text).encode("ascii", "replace").decode()
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(max(0, min(org[0], img.shape[1] - tw - 2))), int(max(th + 3, min(org[1], img.shape[0] - bl - 2)))
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + bl), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def _dashed(img, b, color, thick):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    for p, q in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        p, q = np.array(p, float), np.array(q, float)
        n = max(1, int(np.linalg.norm(q - p) // 16))
        for i in range(0, n, 2):
            a, c = p + (q - p) * i / n, p + (q - p) * min(i + 1, n) / n
            cv2.line(img, tuple(a.astype(int)), tuple(c.astype(int)), (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.line(img, tuple(a.astype(int)), tuple(c.astype(int)), color, thick, cv2.LINE_AA)


def _fill(img, polys, color, alpha):
    layer = img.copy()
    for p in polys:
        cv2.fillPoly(layer, [np.round(np.array(p)).astype(np.int32)], color)
    img[:] = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)


def draw(img, res, qwen_box, objects, with_python=True, with_qwen=True):
    out = img.copy()
    if with_python and res["status"] == "FLOOR_MARKING_ESTABLISHED":
        _fill(out, [res["bay_polygon_xy"]], COL["bay"], 0.22)
        ff = res["free_floor"]["margin_0px"]["polygons_xy"]
        if ff:
            _fill(out, ff, COL["free"], 0.45)
            for p in ff:
                cv2.polylines(out, [np.round(np.array(p)).astype(np.int32)], True, COL["free"], 1, cv2.LINE_AA)
        cv2.polylines(out, [np.round(np.array(res["bay_polygon_xy"])).astype(np.int32)], True, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.polylines(out, [np.round(np.array(res["bay_polygon_xy"])).astype(np.int32)], True, COL["bay"], 3, cv2.LINE_AA)
    for o in objects:
        b = [int(round(v)) for v in o["bbox_xyxy"]]
        if o["class_name"] == "truck":
            cv2.rectangle(out, tuple(b[:2]), tuple(b[2:]), COL["truck"], 2)
        elif o["class_name"] == "traffic_cone":
            cv2.rectangle(out, tuple(b[:2]), tuple(b[2:]), COL["cone"], 2)
    if with_python:
        for l in res["lines"]:
            p, d = np.array(l["point"]), np.array(l["direction"])
            a, c = p + l["t_min"] * d, p + l["t_max"] * d
            cv2.line(out, tuple(a.astype(int)), tuple(c.astype(int)), COL["line"], 1, cv2.LINE_AA)
        used = {i for l in res["lines"] for i in l["dash_indices"]}
        for i, c in enumerate(res["dash_components"]):
            cv2.circle(out, (int(c["center"][0]), int(c["center"][1])), 4 if i in used else 2, COL["dash"] if i in used else (120, 120, 120), -1)
    if with_qwen:
        _dashed(out, qwen_box, COL["qwen"], 4)
    return out


def panel(width, lines):
    pn = np.full((26 * len(lines) + 10, width, 3), 24, np.uint8)
    for i, (txt, col, sw) in enumerate(lines):
        y, x = 26 * (i + 1) - 6, 8
        if sw is not None:
            cv2.rectangle(pn, (8, y - 12), (22, y + 1), sw, -1)
            x = 30
        cv2.putText(pn, str(txt).encode("ascii", "replace").decode()[:235], (x, y), FONT, 0.52, col, 1, cv2.LINE_AA)
    return pn


def main():
    cfg, v02 = V.load_config(), B.load_cfg()
    cc = next(c for c in cfg["stored_cases"] if c["id"] == "E1")
    case = O.build_case(cc, HERE)
    img = cv2.imread(case["image_path"])
    H, W = img.shape[:2]
    qb = case["candidates"][0]["box_px"]
    objs = case["observations"]["objects"]
    res, mask = B.localize(img, qb, objs, v02)
    sens = B.sensitivity(img, qb, objs, v02)

    # repeatability over the 8 photos of the same session (same static layout; E1's window and exclusions are reused)
    stab = []
    base = np.array(res["bay_polygon_xy"]) if res["bay_polygon_xy"] else None
    for f in sorted(glob.glob(str(CAL / "undistorted_*.png"))):
        im = cv2.imread(f)
        r, _ = B.localize(im, qb, objs, v02)
        row = {"photo": Path(f).name, "sha256_16": sha(f)[:16], "status": r["status"], "dashes_per_line": [l["n_dashes"] for l in r["lines"]],
               "corner_xy": r["bay_geometry"].get("corner_xy"), "polygon": r["bay_polygon_xy"]}
        if base is not None and r["bay_polygon_xy"]:
            row["max_vertex_shift_vs_E1_px"] = round(float(np.abs(np.array(r["bay_polygon_xy"]) - base).max()), 2)
        stab.append(row)

    # comparison of Qwen's region with the marking-defined region (two outputs compared; no human answer is involved)
    cmp_ = {}
    if res["bay_polygon_xy"]:
        poly = np.array(res["bay_polygon_xy"])
        qa = (qb[2] - qb[0]) * (qb[3] - qb[1])
        cones = [o for o in objs if o["class_name"] == "traffic_cone"]
        truck = next(o for o in objs if o["class_name"] == "truck")
        pm = np.zeros((H, W), np.uint8)
        cv2.fillPoly(pm, [np.round(poly).astype(np.int32)], 1)
        cmp_ = {"qwen_box_area_px": round(qa), "bay_area_px": res["bay_area_px"], "bay_over_qwen_box_area": round(res["bay_area_px"] / qa, 3),
                "bay_inside_qwen_box_fraction": next(c["detail"] for c in res["checks"] if c["check"].startswith("the region lies inside")),
                "part_of_qwen_box_outside_the_bay": round(1 - res["bay_area_px"] / qa, 3),
                "verified_cones_with_centre_inside_bay": f"{sum(1 for c in cones if pm[int((c['bbox_xyxy'][1] + c['bbox_xyxy'][3]) / 2), int((c['bbox_xyxy'][0] + c['bbox_xyxy'][2]) / 2)])} of {len(cones)}",
                "nearest_cone_to_bay_px": round(min(float(cv2.pointPolygonTest(np.round(poly).astype(np.float32).reshape(-1, 1, 2),
                                                                                (float((c['bbox_xyxy'][0] + c['bbox_xyxy'][2]) / 2), float((c['bbox_xyxy'][1] + c['bbox_xyxy'][3]) / 2)), True) * -1)
                                                    for c in cones), 1),
                "truck_box_area_inside_bay_fraction": round(float(pm[int(truck['bbox_xyxy'][1]):int(truck['bbox_xyxy'][3]), int(truck['bbox_xyxy'][0]):int(truck['bbox_xyxy'][2])].sum()) /
                                                             max(1, (truck['bbox_xyxy'][2] - truck['bbox_xyxy'][0]) * (truck['bbox_xyxy'][3] - truck['bbox_xyxy'][1])), 3)}

    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_bay_localization")
    out.mkdir(parents=True, exist_ok=False)
    runs = case["_v1_runs"]
    prov = {"purpose": "connect Qwen's stored semantic destination to an image-space candidate region; no model was run",
            "instruction": case["instruction"], "image": {"path": case["image_path"], "sha256": sha(case["image_path"]), "size": [W, H]},
            "qwen_proposal": {"stored_in": [r["run_id"] for r in runs], "raw_response_file": case["vlm_raw_response_file"],
                              "raw_response_sha256_16": sha(case["vlm_raw_response_file"])[:16], "function_name": case["candidates"][0]["function_claimed"],
                              "box_px": [round(v, 1) for v in qb], "observed_evidence": case["candidates"][0]["observed_evidence"],
                              "same_reply_in_natural_language_abstraction_exploratory_E1_L2": (HERE.parent / "natural_language_abstraction/results/20261118_185930_exploratory_be08/E1/L2/raw_response.txt").read_text(encoding="utf-8")
                              == Path(case["vlm_raw_response_file"]).read_text(encoding="utf-8")},
            "detections_reused": [{"object_id": o["object_id"], "class": o["class_name"], "bbox_xyxy": [round(v, 1) for v in o["bbox_xyxy"]], "basis": o["basis"]}
                                  for o in objs if o["class_name"] in ("truck", "traffic_cone")],
            "not_used_as_input": ["any human-typed bay or answer coordinates", "the reference zone Z06 / semantic_reference_zones", "the driving map",
                                  "a new Qwen or Grounding DINO call", "the reviewer's own estimate of the bay"],
            "models_run_in_this_step": []}
    loc = {**res, "dash_mask_pixels": int(mask.sum()), "comparison_qwen_box_vs_bay": cmp_}
    loc["not_verified"] = [
        "the open sides of the bay (top and right) are the extent of the dashed lines, not painted lines: their position is an inference",
        "the bay polygon is a straight-line parallelogram in image space: perspective of the floor plane is not modelled (no verified homography)",
        "the truck's bottom edge lies inside the region only marginally (see check): the detector box is not the truck's ground footprint",
        "free floor = bay minus the detected truck box and cones. The truck box includes the truck's height, so it over-removes floor; the cube target may be the truck bed rather than the floor: not determined",
        "v0.2 placement_region keeps only the outer contour of the free area: an object standing fully inside the region would leave no hole (hole_check in localization.json; not triggered on E1 because the truck box crosses the bay's top edge)",
        "placement surface (height, occupancy), gripper reach and approach are unmeasured; this is not a placement or path result",
        "thresholds were chosen by looking at this photo; see sensitivity.json (the result is not stable under every setting)",
        "the 8-photo repeat uses the same layout and E1's window: repeatability, not independent generalisation",
        "no human-confirmed bay exists: nothing here is scored against an answer"]
    (out / "localization.json").write_text(json.dumps(loc, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "input_provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "sensitivity.json").write_text(json.dumps(sens, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "stability_8_photos.json").write_text(json.dumps(stab, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- images
    full = draw(img, res, qb, objs)
    est = res["status"] == "FLOOR_MARKING_ESTABLISHED"
    ff0 = res["free_floor"]["margin_0px"] if est else None
    lines = [("white = verified truck (Grounding DINO + Qwen crop check) | orange boxes = verified cones | yellow dots = detected dash paint | cyan = fitted dashed lines", (200, 200, 200), None),
             ("[Qwen3-VL] original proposed region: 'loading dock' (dashed box, unverified)", (255, 255, 255), COL["qwen"]),
             (("[Python] candidate destination region = bay defined by the two detected dashed sides (green) area %d px; status %s" % (res.get("bay_area_px", 0), res["status"])), (255, 255, 255), COL["bay"]),
             (("[Python, v0.2 placement_region] floor left in the bay after removing the detected truck/cone boxes: %s px exact (%s px after the 8 px margin; geometry only)" % (ff0["areas_px"]["after_object_footprints"] if ff0 else "-", res["free_floor"]["margin_8px"]["area_px"] if ff0 else "-")), (255, 255, 255), COL["free"])]
    if cmp_:
        lines.append(("comparison: bay = %.1f%% of Qwen's box; %s verified cones inside the bay; nearest cone %.0f px away; truck box %.0f%% inside the bay" %
                      (100 * cmp_["bay_over_qwen_box_area"], cmp_["verified_cones_with_centre_inside_bay"], cmp_["nearest_cone_to_bay_px"],
                       100 * cmp_["truck_box_area_inside_bay_fraction"]), (0, 255, 255), None))
    lines.append(("checks: %d of %d PASS (list in localization.json): dashed lines | spacing | angle | corner | inside Qwen box | truck stands in region" %
                  (sum(c["status"] == "PASS" for c in res["checks"]), len(res["checks"])), (170, 170, 170), None))
    lines.append(("NOT VERIFIED: open sides are inferred | floor-plane perspective | placement surface and truck-bed vs floor | reach | no human-confirmed bay. No path planning was run.", (120, 120, 255), None))
    cv2.imwrite(str(out / "01_qwen_vs_python_region.png"), np.vstack([full, panel(W, lines)]))

    x0, y0, x1, y1 = 1230, 0, 1850, 300
    z = 3
    def crop(im):
        return cv2.resize(im[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    raw = crop(img)
    mid = crop(img).copy()
    mm = cv2.resize(mask[y0:y1, x0:x1] * 255, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
    mid[mm > 0] = (0, 0, 255)
    for i, c in enumerate(res["dash_components"]):
        cv2.circle(mid, (int((c["center"][0] - x0) * z), int((c["center"][1] - y0) * z)), 7, COL["dash"], 2)
    for l in res["lines"]:
        p, d = np.array(l["point"]), np.array(l["direction"])
        a, c = (p + l["t_min"] * d - [x0, y0]) * z, (p + l["t_max"] * d - [x0, y0]) * z
        cv2.line(mid, tuple(a.astype(int)), tuple(c.astype(int)), COL["line"], 2, cv2.LINE_AA)
    fin = crop(draw(img, res, qb, objs))
    for t, im in (("A) the photo, unmodified (3x zoom)", raw), ("B) white-paint mask (red) -> dash components (yellow rings) -> fitted lines (cyan)", mid),
                  ("C) Qwen box (orange dashed), Python bay (green), free floor (magenta), verified objects", fin)):
        _t(im, t, (6, 20), (0, 255, 255), 0.6)
    cv2.imwrite(str(out / "02_bay_zoom.png"), np.vstack([raw, mid, fin]))
    print(f"wrote {out}")
    print(json.dumps({"status": res["status"], "dashes_per_line": [l["n_dashes"] for l in res["lines"]], "bay_polygon_xy": res["bay_polygon_xy"],
                      "comparison": cmp_, "free_floor_exact_px_margin0": (res["free_floor"] or {}).get("margin_0px", {}).get("areas_px", {}).get("after_object_footprints"),
                      "free_floor_areas": {k: v["area_px"] for k, v in (res["free_floor"] or {}).items()}, "hole_check": (res["free_floor"] or {}).get("margin_0px", {}).get("hole_check")}, indent=1))
    return out


if __name__ == "__main__":
    main()
