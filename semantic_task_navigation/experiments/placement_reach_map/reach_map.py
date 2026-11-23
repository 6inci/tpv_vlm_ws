#!/usr/bin/env python3
"""Required body-to-point distance over the placement candidate floor of the loading bay (E1, offline).

  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python experiments/placement_reach_map/reach_map.py

Reused unchanged: task_route_integration.build_context (provisional floor transform, DRAFT map, stored bay and free-floor candidate),
task_plan_prototype RectNavMap / make_leg (v0.2 A*, validator, whole-body sweep), v0.2 cost_field. New: the gap map, the R sweep, figures.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent / "task_route_integration"))
import route_integration as RI  # noqa: E402
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file  # noqa: E402
from semantic_task_nav.navigation.planner import cost_field  # noqa: E402

PT, FN = RI.PT, RI.FN


def load_cfg():
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def fill_mask(polys_cm, shape, pad, s):
    m = np.zeros(shape, np.uint8)
    for p in polys_cm:
        cv2.fillPoly(m, [np.round((np.asarray(p, np.float64) + pad) * s).astype(np.int32)], 1)
    return m.astype(bool)


def reachable_centres(nav, start_bev):
    """Valid body centres (full resolution) connected to the start. None if the body does not fit at the start."""
    fits = nav.footprint_fits_full
    x, y = int(round(start_bev[0])), int(round(start_bev[1]))
    if not fits[y, x]:
        return None
    _, lab = cv2.connectedComponents(fits.astype(np.uint8), connectivity=8)
    return lab == lab[y, x]


def body_union(centres, extent_cm, s):
    """Floor covered by the NOMINAL body over all centres. Kernel rounded inward: the body is never drawn larger, so gaps are never too small."""
    odd = lambda v: int(np.floor(v)) - (1 - int(np.floor(v)) % 2)       # noqa: E731  largest odd pixel count <= the nominal size
    kx, ky = odd(extent_cm[0] * s), odd(extent_cm[1] * s)
    return cv2.dilate(centres.astype(np.uint8), np.ones((ky, kx), np.uint8)).astype(bool)


def gap_map(union, s):
    """Distance (cm) from every floor pixel to the nearest pixel the nominal body can cover."""
    return cv2.distanceTransform((~union).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / s


def r_sweep(gap, cand, rs, s):
    total = int(cand.sum())
    return [{"R_cm": r, "area_cm2": round(int((cand & (gap <= r)).sum()) / s ** 2, 1), "fraction_of_candidate": round(float((cand & (gap <= r)).sum() / total), 3) if total else None} for r in rs]


def rect_point_dist(cx, cy, ext, p):
    return np.hypot(np.maximum(np.abs(cx - p[0]) - ext[0] / 2, 0), np.maximum(np.abs(cy - p[1]) - ext[1] / 2, 0))


def _t(img, text, org, col=(255, 255, 255), sc=0.5):
    PT._t(img, text, org, col, sc)


def chart(sweeps, out):
    W, H, L, B = 760, 420, 70, 50
    im = np.full((H, W, 3), 255, np.uint8)
    rmax = max(x["R_cm"] for x in next(iter(sweeps.values())))
    amax = max(max(x["area_cm2"] for x in v) for v in sweeps.values()) or 1
    px = lambda r, a: (int(L + (W - L - 20) * r / rmax), int(H - B - (H - B - 30) * a / amax))   # noqa: E731
    cv2.rectangle(im, (L, 30), (W - 20, H - B), (0, 0, 0), 1)
    for r in range(0, rmax + 1, 10):
        cv2.putText(im, str(r), (px(r, 0)[0] - 8, H - B + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(im, (px(r, 0)[0], H - B), (px(r, 0)[0], 30), (225, 225, 225), 1)
    for k in range(5):
        a = amax * k / 4
        cv2.putText(im, f"{a:.0f}", (8, px(0, a)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    for (name, v), col in zip(sweeps.items(), [(200, 90, 0), (0, 140, 255)]):
        pts = np.array([px(x["R_cm"], x["area_cm2"]) for x in v], np.int32)
        cv2.polylines(im, [pts], False, col, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(im, tuple(int(q) for q in p), 3, col, -1)
        cv2.putText(im, name, (L + 10, 50 + 18 * list(sweeps).index(name)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.putText(im, "hypothetical bridging distance R from the body edge [cm]  (NOT the gripper reach)", (L, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(im, "placeable candidate floor [cm^2]  (geometry only, PROVISIONAL frame)", (L, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), im)


def main():
    cfg, rcfg, v02 = load_cfg(), RI.load_cfg(), load_config()
    gcfg = PT.load_cfg()
    img_path = Path(rcfg["image"])
    assert sha256_file(img_path) == rcfg["image_sha256"]
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    fp_before = code_fingerprint()["sha256"]
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_reach")
    out.mkdir(parents=True, exist_ok=False)
    (out / "figures").mkdir()
    (out / "config_snapshot.yaml").write_text((HERE / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    ctx = RI.build_context(rcfg, gcfg, v02, img)
    assert ctx["ft"]["accepted"], "floor transform refused"
    fr, s, pad, m = ctx["fr"], ctx["fr"].s, ctx["fr"].pad, ctx["ft"]["footprint_margin_cm"]
    shape = (fr.bev_size[1], fr.bev_size[0])
    route = PT.jload((HERE / cfg["route_run"]).resolve() / "task_plan.json")
    start_cm = route["condition_A"]["plan"]["paths"]["A"]["goal_cm"]                 # pick-up approach = start of path B
    old_B = route["condition_A"]["plan"]["paths"]["B"]
    start_bev = fr.cm_to_bev([start_cm])[0].tolist()
    cand = fill_mask(ctx["place_candidate_cm"], shape, pad, s)
    bay = fill_mask([ctx["bay_cm"]], shape, pad, s)
    drivable = fill_mask([z["polygon"] for z in ctx["spec_cm"]["traversable"]], shape, pad, s)
    result = {"experiment_id": cfg["experiment_id"], "created": datetime.now().isoformat(timespec="seconds"),
              "placement_definition": "empty floor inside the dashed loading bay (provisional, user 2026-11-19); virtual staging scenario",
              "claims": {"placement_success": "NOT_CLAIMABLE", "gripper_reach": "NOT ASSUMED: R is a hypothetical body-edge-to-point distance",
                         "geometry": "PROVISIONAL floor transform + DRAFT map (unverified)"},
              "placement_candidate": {**cfg["placement_candidate"], "area_cm2": round(int(cand.sum()) / s ** 2, 1), "bay_area_cm2": round(int(bay.sum()) / s ** 2, 1),
                                      "polygons_cm": [np.round(p, 1).tolist() for p in ctx["place_candidate_cm"]],
                                      "overlap_with_drivable_cm2": round(int((cand & drivable).sum()) / s ** 2, 1)},
              "start_of_path_B_cm": start_cm, "footprint_margin_cm": m, "heading_sets": {}}
    if not cand.any():
        result["state"] = "NO_VALID_PLACEMENT (no candidate floor in the bay)"
        (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(result["state"])
        return out
    navs, reach, gaps = {}, {}, {}
    for h in cfg["robot"]["headings"]["with_rotation"]:
        navs[h] = FN.RectNavMap(ctx["spec_bev"], fr.bev_size, ctx["pcfg"], FN.footprint_dims_cm(35, 24, m, h), s)
        reach[h] = reachable_centres(navs[h], start_bev)
    for name, hs in cfg["robot"]["headings"].items():
        union = np.zeros(shape, bool)
        used = [h for h in hs if reach[h] is not None]
        for h in used:
            union |= body_union(reach[h], FN.footprint_dims_cm(35, 24, 0, h), s)
        g = gap_map(union, s)
        gaps[name] = g
        gc = g[cand]
        result["heading_sets"][name] = {"headings_deg": hs, "headings_usable_from_start": used, "min_gap_cm": round(float(gc.min()), 1), "median_gap_cm": round(float(np.median(gc)), 1),
                                        "max_gap_cm": round(float(gc.max()), 1), "R_sweep": r_sweep(g, cand, cfg["R_cm"], s),
                                        "smallest_listed_R_with_any_area_cm": next((x["R_cm"] for x in r_sweep(g, cand, cfg["R_cm"], s) if x["area_cm2"] > 0), None),
                                        "smallest_listed_R_covering_all_cm": next((x["R_cm"] for x in r_sweep(g, cand, cfg["R_cm"], s) if x["fraction_of_candidate"] == 1.0), None)}
    result["heading_sets"]["with_rotation"]["note"] = "needs a heading change; in-place rotation clearance is NOT met at the pick-up approach (earlier result): information only"

    # ---- placement point with the smallest gap (primary heading) -> nearest valid stop pose -> path B again
    g0 = np.where(cand, gaps["primary"], np.inf)
    py, px_ = np.unravel_index(int(np.argmin(g0)), g0.shape)
    p_cm = [float(px_ / s - pad), float(py / s - pad)]
    nav = navs[0]
    cells = np.isfinite(cost_field(nav, nav.cell_of(*start_bev), ctx["pcfg"]["planner"], ctx["pcfg"]["safety_weight"])) & nav.valid
    rr, cc = np.nonzero(cells)
    cx, cy = (cc + 0.5) * nav.cell_px / s - pad, (rr + 0.5) * nav.cell_px / s - pad
    d = rect_point_dist(cx, cy, FN.footprint_dims_cm(35, 24, 0, 0), p_cm)
    k = int(np.argmin(d))
    stop_cm = [float(cx[k]), float(cy[k])]
    goal = {"status": "GOAL_SELECTED", "xy_bev_px": fr.cm_to_bev([stop_cm])[0].tolist(), "xy_cm": [round(v, 1) for v in stop_cm]}
    leg, _ = PT.make_leg(nav, nav.extent_cm, 0, goal, start_bev, ctx["pcfg"], ctx["free_hi"], fr, gcfg["bev"], "B_to_placement_stop")
    result["placement_point"] = {"rule": "candidate floor point with the smallest gap (primary heading)", "point_cm": [round(v, 1) for v in p_cm],
                                 "point_image_px": [round(float(v), 1) for v in fr.cm_to_img([p_cm])[0]], "state": "PLACEMENT_UNVERIFIED",
                                 "why_unverified": ["candidate floor is geometry only (not visually confirmed as empty)", "the bay lies outside the drivable zone: the body cannot enter it",
                                                    "gripper mount, direction and reach unknown: the gap is not a reach", "PROVISIONAL floor transform, DRAFT map"],
                                 "nearest_valid_stop_pose_cm": [round(v, 1) for v in stop_cm], "stop_heading_deg": 0, "body_edge_to_point_gap_cm": round(float(d[k]), 1),
                                 "direction_stop_to_point_deg": round(float(np.degrees(np.arctan2(p_cm[1] - stop_cm[1], p_cm[0] - stop_cm[0]))), 1)}
    # stored, Qwen-checked cone boxes between the stop pose and the point (image px; tall objects, so this is information, not a floor test)
    seg = np.linspace(fr.cm_to_img([stop_cm])[0], fr.cm_to_img([p_cm])[0], 400)
    cones = [o for o in ctx["st"]["objects"] if o["class_name"] == "traffic_cone"]
    crossed = [o["object_id"] for o in cones if np.any((seg[:, 0] >= o["bbox_xyxy"][0]) & (seg[:, 0] <= o["bbox_xyxy"][2]) & (seg[:, 1] >= o["bbox_xyxy"][1]) & (seg[:, 1] <= o["bbox_xyxy"][3]))]
    result["placement_point"]["straight_line_stop_to_point"] = {"stored_cone_boxes_checked": len(cones), "cone_boxes_crossed": crossed,
                                                                "note": "a cone row stands between the drivable zone and the bay; only the stored verified cone boxes are tested, other objects are not"}
    result["path_B_to_placement_stop"] = {"generated": leg["feasible"], "failure": leg["failure"], "path": leg["path"] and {k2: leg["path"][k2] for k2 in ("waypoints_cm", "length_cm", "n_waypoints")},
                                          "v02_validation": leg["v02_validation"], "whole_body_sweep": leg["sweep_check"],
                                          "layers": RI.swept_layer_overlap(ctx, leg["path"]["waypoints_cm"], nav.extent_cm) if leg["path"] else None,
                                          "earlier_path_B": {"goal_cm": old_B["goal_cm"], "length_cm": old_B["length_cm"], "gap_to_bay_polygon_cm": old_B["approach"]["body_to_target_gap_cm"]}}
    pr = result["heading_sets"]["primary"]
    result["state"] = ("PLACEMENT_UNVERIFIED: candidate floor exists (geometry only); the body cannot reach any of it without bridging >= %.1f cm" % pr["min_gap_cm"])

    # ---- figures
    mp = PT.Mapper("photo", fr)
    im = img.copy()
    RI.layers(im, mp, ctx, text=False)
    PT._fill(im, mp, ctx["place_candidate_cm"], (0, 255, 0), 0.5)
    for p in ctx["place_candidate_cm"]:
        PT._outline(im, mp, p, (0, 255, 0), 2)
    PT._rect(im, mp, stop_cm, nav.extent_cm, (0, 255, 255), 2, 0.2)
    PT._rect(im, mp, stop_cm, FN.footprint_dims_cm(35, 24, 0, 0), (0, 255, 255), 1)
    cv2.line(im, tuple(mp(stop_cm)[0]), tuple(mp(p_cm)[0]), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(im, tuple(mp(p_cm)[0]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
    if leg["path"]:
        cv2.polylines(im, [mp(leg["path"]["waypoints_cm"])], False, (40, 150, 255), 3, cv2.LINE_AA)
    _t(im, "bright green = placement candidate floor (bay minus object boxes; GEOMETRY ONLY, not confirmed empty)", (10, 24), (0, 255, 0), 0.6)
    _t(im, f"red x = candidate point with the smallest gap {result['placement_point']['point_cm']} cm; yellow = nearest valid stop pose {result['placement_point']['nearest_valid_stop_pose_cm']} cm; gap {result['placement_point']['body_edge_to_point_gap_cm']} cm (NOT a reach)", (10, 50), (255, 255, 255), 0.6)
    _t(im, "PLACEMENT_UNVERIFIED - plan only; PROVISIONAL floor frame, DRAFT map", (10, 76), (0, 255, 255), 0.6)
    cv2.imwrite(str(out / "figures/01_placement_candidate_on_photo.png"), im)
    S = 4
    Hc, Wc = shape
    bev = (cv2.warpPerspective(img, PT.FT.bev_matrix(fr.H, s, pad), (Wc, Hc)) * 0.4).astype(np.uint8)
    for name, fn in (("primary", "02_gap_map_heading0.png"), ("with_rotation", "03_gap_map_heading0_or_90.png")):
        b = bev.copy()
        g = gaps[name]
        heat = cv2.applyColorMap(np.clip(g / 80 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        b[cand] = heat[cand]
        mb = PT.Mapper("bev", fr, s)
        for z in ctx["spec_cm"]["traversable"]:
            PT._outline(b, mb, z["polygon"], (255, 255, 255), 1)
        PT._outline(b, mb, ctx["bay_cm"], (230, 200, 40), 2)
        for r in (20, 30, 40, 50, 60):
            cs, _ = cv2.findContours(((g <= r) & cand).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(b, cs, -1, (255, 255, 255), 1)
        if name == "primary":
            PT._rect(b, mb, stop_cm, FN.footprint_dims_cm(35, 24, 0, 0), (0, 255, 255), 2)
            cv2.drawMarker(b, tuple(mb(p_cm)[0]), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        b = b[int((0) * s):int((90 + pad) * s), int((200 + pad) * s):]                 # zoom on the bay (x >= 200 cm, y <= 90 cm)
        b = cv2.resize(b, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)
        _t(b, f"required body-edge-to-point distance on the candidate floor [{name}]: min {result['heading_sets'][name]['min_gap_cm']} / median {result['heading_sets'][name]['median_gap_cm']} / max {result['heading_sets'][name]['max_gap_cm']} cm", (8, 22), (255, 255, 255), 0.55)
        _t(b, "colour: 0 cm (blue) .. 80 cm (red); white contours R = 20/30/40/50/60 cm; white outline = drivable zone; NOT a gripper reach", (8, 46), (255, 255, 255), 0.5)
        cv2.imwrite(str(out / "figures" / fn), b)
    chart({k: v["R_sweep"] for k, v in result["heading_sets"].items()}, out / "figures/04_area_vs_R.png")
    result.update({"figures": sorted(p.name for p in (out / "figures").iterdir()), "v02_code_before": fp_before, "v02_code_after": code_fingerprint()["sha256"], "model_calls": 0,
                   "floor_transform": {k: ctx["ft"][k] for k in ("status", "fit_rms_cm", "seam_max_abs_dev_cm", "footprint_margin_cm")}})
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "manifest.json").write_text(json.dumps({"files_sha256": files}, indent=2), encoding="utf-8")
    print("wrote", out)
    print(json.dumps({k: result[k] for k in ("placement_candidate", "heading_sets", "placement_point", "path_B_to_placement_stop", "state")}, indent=1, ensure_ascii=False, default=str)[:5000])
    return out


if __name__ == "__main__":
    main()
