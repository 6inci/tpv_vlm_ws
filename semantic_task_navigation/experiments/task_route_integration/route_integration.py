#!/usr/bin/env python3
"""Offline task-route integration on E1: condition A (REFERENCE_TARGET_PRELIMINARY) and condition B (retained detector candidates).

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
  $PY experiments/task_route_integration/route_integration.py

No model call, no GPU, no hardware. Reused (imported, unchanged): v0.2 NavMap / select_goal / plan / validate_path / rule_based_parse / APPROACH_CODES /
STEP_STATUSES, the stored loaders of first_vlm_navigation_path, and from task_plan_prototype the provisional floor transform, the rectangular-footprint
NavMap, approach_goal, make_leg and plan_variant (two fixed headings, v0.2 A*, v0.2 validator, independent whole-body sweep).
New here: target sources A/B, plan states, per-layer swept-body counts, the disc comparison, the re-check of the earlier 35 px path, figures.
"""
import hashlib
import json
import math
import shutil
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent / "task_plan_prototype"))
sys.path.insert(0, str(HERE.parent / "first_vlm_navigation_path"))
import plan_task as PT  # noqa: E402
from semantic_task_nav.agent.orchestrator import APPROACH_CODES  # noqa: E402
from semantic_task_nav.agent.task_parser import rule_based_parse  # noqa: E402
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file  # noqa: E402
from semantic_task_nav.contracts import STEP_STATUSES  # noqa: E402
from semantic_task_nav.geometry.coordinate_frames import box_gap, iou  # noqa: E402
from semantic_task_nav.navigation.map_builder import NavMap  # noqa: E402
from semantic_task_nav.navigation.path_validator import validate_path  # noqa: E402
from semantic_task_nav.navigation.planner import plan  # noqa: E402

FT, FN, FP = PT.FT, PT.FN, PT.FP
PLAN_STATES = ("APPROACH_PLANNED", "VERIFY_REQUIRED", "PICKUP_ASSUMED", "TRANSPORT_PLANNED", "PLACEMENT_ASSUMED")


def load_cfg():
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def res(p):
    return (HERE / p).resolve()


# ------------------------------------------------------------------------------------------------ context (stored inputs + geometry)
def build_context(cfg, gcfg, v02, img):
    h, w = img.shape[:2]
    st = FP.load_stored(FP.load_cfg(), (w, h))
    first_dir = res(cfg["stored"]["first_path_run"])
    first = PT.jload(first_dir / "result.json")
    robot, starts = PT.pose_list_from_stored(first)
    zmap = PT.jload(first_dir / "map_E1_transferred.json")
    ft = FT.fit_floor_transform(img, gcfg["floor"])
    ctx = {"cfg": gcfg, "img": img, "ft": ft, "st": st, "first": first, "robot": robot, "starts": starts, "zmap": zmap, "legend": []}
    if not ft["accepted"]:
        return ctx
    size_cm = (gcfg["floor"]["mat_cm"] * gcfg["floor"]["mats_cols"], gcfg["floor"]["mat_cm"] * gcfg["floor"]["mats_rows"])
    fr = PT.Frames(ft["H_img_to_floor_cm"], gcfg["bev"]["px_per_cm"], gcfg["bev"]["pad_cm"], size_cm)
    spec_cm, spec_bev, clipped = PT.build_specs(zmap, fr)
    bay_cm = fr.img_to_cm(st["bay_polygon"])
    free_place = (st["bay"].get("free_floor") or {}).get("margin_0px", {}).get("polygons_xy", [])
    ctx.update({"fr": fr, "size_cm": size_cm, "spec_cm": spec_cm, "spec_bev": spec_bev, "clipped": clipped,
                "pcfg": PT.v02_pcfg(v02, gcfg["bev"], gcfg["approach"]),
                "free_hi": FN.free_mask_cm(spec_cm, size_cm, fr.pad, gcfg["bev"]["check_px_per_cm"]),
                "bay_cm": bay_cm.tolist(), "bay_bev": fr.cm_to_bev(bay_cm).tolist(),
                "qwen_cm": fr.img_to_cm(PT.box_poly(st["qwen_box_px"])).tolist(),
                "place_candidate_cm": [fr.img_to_cm(p).tolist() for p in free_place]})
    return ctx


def set_target(ctx, box):
    tc = ctx["fr"].img_to_cm(PT.box_poly(box))
    ctx["target_cm"], ctx["target_bev"] = tc.tolist(), ctx["fr"].cm_to_bev(tc).tolist()
    return tc


# ------------------------------------------------------------------------------------------------ checks
def _fill(polys, w, h, pad, s):
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round((np.asarray(p, np.float64) + pad) * s).astype(np.int32)], 1)
    return m.astype(bool)


def swept_layer_overlap(ctx, wp_cm, extent, step_cm=0.25):
    """Union of the body rectangle along the polyline, counted against EACH layer separately (px at check_px_per_cm)."""
    s, pad, size = ctx["cfg"]["bev"]["check_px_per_cm"], ctx["fr"].pad, ctx["size_cm"]
    w, h = round((size[0] + 2 * pad) * s), round((size[1] + 2 * pad) * s)
    body, off_canvas = np.zeros((h, w), bool), 0
    for a, b in zip(wp_cm, wp_cm[1:]):
        n = max(1, int(math.ceil(math.dist(a, b) / step_cm)))
        for k in range(n + 1):
            c = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
            x0, x1 = math.floor((c[0] - extent[0] / 2 + pad) * s), math.ceil((c[0] + extent[0] / 2 + pad) * s)
            y0, y1 = math.floor((c[1] - extent[1] / 2 + pad) * s), math.ceil((c[1] + extent[1] / 2 + pad) * s)
            off_canvas += int(x0 < 0 or y0 < 0 or x1 > w or y1 > h)
            body[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    sp = ctx["spec_cm"]
    trav, stat, nogo = (_fill([z["polygon"] for z in sp[k]], w, h, pad, s) for k in ("traversable", "static_obstacles", "no_go"))
    ws = _fill([[[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]], w, h, pad, s)
    out = {"px_per_cm": s, "body_px": int(body.sum()), "on_static_obstacle_px": int((body & stat).sum()), "on_no_go_px": int((body & nogo).sum()),
           "outside_drivable_px": int((body & ~trav).sum()), "outside_workspace_px": int((body & ~ws).sum()), "samples_off_canvas": off_canvas}
    out["clear"] = all(out[k] == 0 for k in ("on_static_obstacle_px", "on_no_go_px", "outside_drivable_px", "outside_workspace_px", "samples_off_canvas"))
    return out


def leg_code(legs, which):
    """Aggregate failure code of a leg over the planned headings (None if one heading is feasible). v0.2 codes are kept per heading."""
    if not legs:
        return "NOT_ATTEMPTED (no pick-up approach to start from)" if which == "B" else "NOT_ATTEMPTED"
    if any(l["feasible"] for l in legs.values()):
        return None
    f = [l["failure"] or {} for l in legs.values()]
    if which == "A" and all(x.get("stage") == "approach_point" for x in f):
        return "START_POSE_INVALID" if all(str(x.get("code", "")).startswith("START_") for x in f) else "NO_VALID_PICKUP_APPROACH"
    if which == "B" and all(x.get("stage") == "approach_point" and not str(x.get("code", "")).startswith("START_") for x in f):
        return "NO_VALID_PLACE_APPROACH"
    return "NO_VALID_PATH_" + which


def summarize(var, ctx, L, W, margin, target_tag):
    """Plan states, chosen legs, path records and checks for one plan_variant() result."""
    fr, sel = ctx["fr"], var["selected"]
    hA, hB = sel.get("A_heading_deg"), sel.get("B_heading_deg")
    legA, legB = (var["legs_A"].get(hA) if hA is not None else None), (var["legs_B"].get(hB) if hB is not None else None)
    out = {"target": target_tag, "footprint_nominal_cm": [L, W], "margin_per_side_cm": margin, "start_cm": var["start_cm"], "start_status": var["start_status"],
           "selection": sel, "failure_A": leg_code(var["legs_A"], "A"), "failure_B": leg_code(var["legs_B"], "B"),
           "v02_codes_by_heading": {"A": {str(h): (l["failure"] or {}).get("code") for h, l in var["legs_A"].items()},
                                    "B": {str(h): (l["failure"] or {}).get("code") for h, l in var["legs_B"].items()}},
           "v02_approach_code_names": {k: APPROACH_CODES.get(k) for k in ("NO_ACCESSIBLE_CANDIDATE", "CANDIDATES_UNREACHABLE")},
           "states": [], "paths": {}, "checks": {}}
    for tag, leg in (("A", legA), ("B", legB)):
        if leg is None:
            continue
        ext = tuple(leg["footprint_extent_xy_cm"])
        g = leg["goal"]
        out["paths"][tag] = {"heading_deg": leg["heading_deg"], "footprint_extent_xy_cm": list(ext), "frame": "floor_cm (PROVISIONAL); image px = undistorted E1",
                             "start_cm": leg["path"]["waypoints_cm"][0], "goal_cm": g["xy_cm"], "goal_image_px": g["xy_image_px"],
                             "waypoints_cm": leg["path"]["waypoints_cm"], "waypoints_image_px": leg["path"]["waypoints_image_px"],
                             "length_cm": leg["path"]["length_cm"], "raw_grid_length_cm": leg["path"]["raw_grid_length_cm"],
                             "approach": {"purpose": g["purpose"], "nearest_reachable_centre_to_target_cm": g["nearest_reachable_centre_to_target_cm"],
                                          "body_to_target_gap_cm": g["body_to_target_gap_cm"], "facing_direction_deg": g["facing_direction_deg"],
                                          "reach": g["reach"]}}
        out["checks"][tag] = {"v02_validator_inscribed_disc": leg["v02_validation"], "whole_body_sweep": leg["sweep_check"],
                              "swept_layer_overlap": swept_layer_overlap(ctx, leg["path"]["waypoints_cm"], ext),
                              "approach_pose_body_free": FN.body_free(ctx["free_hi"], g["xy_cm"], ext, fr.pad, ctx["cfg"]["bev"]["check_px_per_cm"])[0],
                              "diagonal_corner_cutting": "v0.2 A* forbids diagonal moves past a blocked side cell; the continuous polyline is swept every 0.25 cm (catches cut corners)"}
    if legA:
        gA = legA["goal"]
        out["states"] += [
            {"state": "APPROACH_PLANNED", "path": "A", "to_cm": gA["xy_cm"], "executed": False},
            {"state": "VERIFY_REQUIRED", "at_cm": gA["xy_cm"], "executed": False,
             "meaning": "a further close-range observation is needed before any grasp; re-checking with the fixed camera adds no resolution (target ~17x15 px)",
             "facing_required_deg": gA["facing_direction_deg"], "rotation_in_place_clear": gA.get("rotation_in_place_at_approach")},
            {"state": "PICKUP_ASSUMED", "at_cm": gA["xy_cm"], "executed": False, "pickup_feasibility": "UNVERIFIED",
             "gripper_must_bridge_cm": gA["body_to_target_gap_cm"], "unmeasured": ["gripper reach", "grasp pose", "cube size", "conveyor surface height"]}]
    if legB:
        gB = legB["goal"]
        out["states"] += [
            {"state": "TRANSPORT_PLANNED", "path": "B", "from_cm": legA["goal"]["xy_cm"], "to_cm": gB["xy_cm"], "executed": False, "condition": "planned assuming pick-up"},
            {"state": "PLACEMENT_ASSUMED", "at_cm": gB["xy_cm"], "executed": False, "placement_feasibility": "UNVERIFIED",
             "gripper_must_bridge_cm": gB["body_to_target_gap_cm"], "undecided": ["boundary-crossing permission of the gripper", "placement surface"]}]
        out["checks"]["A_B_connected"] = {"B_starts_where_A_ends": out["paths"]["B"]["start_cm"] == out["paths"]["A"]["waypoints_cm"][-1],
                                          "heading_change": hA != hB, "rotation_check": legB.get("rotation_in_place_at_pickup_approach")}
    assert all(s["state"] in PLAN_STATES for s in out["states"])
    return out


def run_target(ctx, label, box, L, W, margin, start_img, tag):
    set_target(ctx, box)
    var, _ = PT.plan_variant(label, L, W, margin, start_img, ctx)
    return summarize(var, ctx, L, W, margin, tag), var


def plan_disc(ctx, r_cm, start_img):
    """Comparison: v0.2 disc footprint (NavMap unchanged) with radius r_cm; v0.2 validate_path is exact for a disc."""
    fr, s = ctx["fr"], ctx["fr"].s
    p = {**ctx["pcfg"], "robot_radius_px": r_cm * s}
    nav = NavMap(ctx["spec_bev"], fr.bev_size, p)
    slack = ctx["cfg"]["approach"]["slack_cm"] * s
    out = {"radius_cm": round(r_cm, 2), "A": None, "B": None}
    start = fr.cm_to_bev(fr.img_to_cm([start_img]))[0].tolist()
    for tag, anchor, poly in (("A", "cube", ctx["target_bev"]), ("B", "bay", ctx["bay_bev"])):
        g = PT.approach_goal(nav, anchor, poly, start, "pickup_approach" if tag == "A" else "dropoff_approach", p, slack, s, fr.pad)
        rec = {"goal_status": g["status"], "goal_cm": g.get("xy_cm"), "path": None}
        if g["status"] == "GOAL_SELECTED":
            pa = plan(nav, start, g["xy_bev_px"], p, tag)
            rec["path"] = {"status": pa["status"], "length_cm": round(pa["length_px"] / s, 1) if pa.get("length_px") else None,
                           "v02_valid": validate_path(nav, pa["waypoints_xy"], pa["start_xy"], pa["goal_xy"], p)["valid"] if pa["status"] == "PATH_FOUND" and len(pa["waypoints_xy"]) > 1 else None}
        out[tag] = rec
        if not (rec["path"] and rec["path"]["status"] == "PATH_FOUND"):
            break
        start = g["xy_bev_px"]
    out["result"] = "PATH_A_AND_B" if out["B"] and out["B"]["path"] and out["B"]["path"]["status"] == "PATH_FOUND" else "PATH_A_ONLY" if out["A"] and out["A"]["path"] and out["A"]["path"]["status"] == "PATH_FOUND" else "NO_PATH_A"
    return out


def old_path_check(ctx, cfg, margin):
    pv = ctx["first"]["stages"]["5_planning"]
    var = pv["variants"][cfg["old_path_check"]["variant"]]
    wp_img = var["path"]["waypoints_xy"]
    wp_cm = ctx["fr"].img_to_cm(wp_img).tolist()
    out = {"source": cfg["old_path_check"]["variant"], "planned_with": f"v0.2 disc r={var['robot_radius_px']} px in image space", "length_px": var["path"]["length_px"],
           "waypoints_image_px": wp_img, "waypoints_cm": [[round(x, 1), round(y, 1)] for x, y in wp_cm], "checks": {}}
    for name, mg in (("nominal_35x24", 0.0), ("primary_with_margin", margin)):
        for h in cfg["old_path_check"]["headings_deg"]:
            ext = FN.footprint_dims_cm(35, 24, mg, h)
            sw = FN.sweep_check(ctx["free_hi"], wp_cm, ext, ctx["fr"].pad, ctx["cfg"]["bev"]["check_px_per_cm"], 0.25)
            out["checks"][f"{name}_heading{h}"] = {"extent_cm": list(ext), "sweep_ok": sw["ok"], "first_violation": sw["first_violation"],
                                                   "layers": swept_layer_overlap(ctx, wp_cm, ext)}
    out["verdict"] = ("PASSES real-size body check at some heading" if any(c["sweep_ok"] for k, c in out["checks"].items() if k.startswith("primary"))
                      else "FAILS the real-size body check (primary footprint, both headings): recorded as a failure of the earlier plan")
    return out


# ------------------------------------------------------------------------------------------------ figures
C = {"A": (255, 160, 40), "B": (40, 150, 255), "goal": (0, 255, 255), "target": (0, 0, 255), "cand": (255, 0, 255), "robot": (255, 255, 255),
     "bay": (230, 200, 40), "qwen": (0, 165, 255), "place": (200, 255, 200)}
COND_COL = [(255, 0, 255), (0, 200, 255), (120, 255, 120)]


def layers(img, m, ctx, text=True):
    sp = ctx["spec_cm"]
    PT._fill(img, m, [z["polygon"] for z in sp["traversable"]], (80, 200, 80), 0.22)
    PT._fill(img, m, [z["polygon"] for z in sp["static_obstacles"]], (60, 60, 235), 0.45)
    PT._fill(img, m, [z["polygon"] for z in sp["no_go"]], (200, 60, 200), 0.30)
    for z in sp["traversable"] + sp["static_obstacles"] + sp["no_go"]:
        PT._outline(img, m, z["polygon"], (255, 255, 255), 1)
        if text:
            PT._t(img, z["id"], m(np.mean(z["polygon"], axis=0))[0], (255, 255, 255), 0.45)
    PT._fill(img, m, [ctx["bay_cm"]], C["bay"], 0.35)
    PT._outline(img, m, ctx["bay_cm"], C["bay"], 2)
    PT._outline(img, m, ctx["qwen_cm"], C["qwen"], 2, dashed=True)
    for p in ctx["place_candidate_cm"]:
        PT._outline(img, m, p, C["place"], 1, dashed=True)
    if text:
        PT._t(img, "observed loading bay (Python)", m(ctx["bay_cm"][0])[0] + np.array([-40, 22]), C["bay"], 0.45)
        PT._t(img, "Qwen semantic destination (area_5)", m(ctx["qwen_cm"][3])[0] + np.array([4, -6]), C["qwen"], 0.45)


def body_trail(img, m, wp, ext, col, step):
    for a, b in zip(wp, wp[1:]):
        n = max(1, int(math.dist(a, b) // step))
        for k in range(n + 1):
            PT._rect(img, m, (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n), ext, col, 1)


def draw_path(img, m, path, col, tag, step, coords=True):
    wp = path["waypoints_cm"]
    body_trail(img, m, wp, path["footprint_extent_xy_cm"], col, step)
    cv2.polylines(img, [m(wp)], False, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.polylines(img, [m(wp)], False, col, 3, cv2.LINE_AA)
    for i, p in enumerate(wp):
        cv2.circle(img, tuple(m(p)[0]), 6, col, -1, cv2.LINE_AA)
        PT._t(img, f"{tag}{i}" + (f" ({p[0]:.0f},{p[1]:.0f})cm" if coords else ""), m(p)[0] + np.array([8, -8 if i % 2 else 18]), col, 0.42)
    PT._rect(img, m, path["goal_cm"], path["footprint_extent_xy_cm"], C["goal"], 2, 0.25)


def robot_start(img, m, start_cm, ext_nom, ext_m):
    PT._rect(img, m, start_cm, ext_nom, C["robot"], 2)
    PT._rect(img, m, start_cm, ext_m, C["robot"], 1)
    cv2.drawMarker(img, tuple(m(start_cm)[0]), C["robot"], cv2.MARKER_CROSS, 16, 2)
    PT._t(img, f"START robot (ESTIMATED) ({start_cm[0]:.0f},{start_cm[1]:.0f})cm", m(start_cm)[0] + np.array([-150, 40]), C["robot"], 0.45)


def panel(h, lines, w=700):
    p = np.zeros((h, w, 3), np.uint8)
    y = 24
    for line in lines:
        col = (0, 255, 255) if line.startswith("#") else (255, 255, 255)
        for part in textwrap.wrap(line.lstrip("#"), 82) or [""]:
            PT._t(p, part, (10, y), col, 0.46, bg=False)
            y += 19
    return p


def render(ctx, A, B_plans, cand_meta, cfg, out):
    fd = out / "figures"
    fd.mkdir()
    fr, step = ctx["fr"], cfg["figures_sample_step_cm"]
    mp = PT.Mapper("photo", fr)
    nom = FN.footprint_dims_cm(35, 24, 0, (A["selection"].get("A_heading_deg") or 0))
    marg = FN.footprint_dims_cm(35, 24, A["margin_per_side_cm"], (A["selection"].get("A_heading_deg") or 0))
    gen = ["# PLAN ONLY - nothing executed; pick-up / placement NOT claimed",
           f"floor transform PROVISIONAL_UNVALIDATED (fit RMS {ctx['ft']['fit_rms_cm']} cm, independent seam dev <= {ctx['ft']['seam_max_abs_dev_cm']} cm): real-size safety NOT claimed",
           "map: hand-drawn DRAFT zones (MANUAL_MAP_PRIOR, human-unconfirmed); anything outside Z02 is not drivable",
           f"body: 35x24 cm user size + {A['margin_per_side_cm']} cm/side transform margin = {marg[0]:.0f}x{marg[1]:.0f} cm, fixed heading per leg"]
    figs = {}

    def save(name, img, lines):
        cv2.imwrite(str(fd / name), np.hstack([img, panel(img.shape[0], lines)]))
        figs[name] = str(fd / name)

    # 1 environment
    im = ctx["img"].copy()
    layers(im, mp, ctx)
    robot_start(im, mp, A["start_cm"], nom, marg)
    PT._outline(im, mp, ctx["targetA_cm"], C["target"], 3)
    PT._t(im, "A: REFERENCE_TARGET_PRELIMINARY (AI label)", mp(ctx["targetA_cm"][0])[0] + np.array([10, -40]), C["target"], 0.45)
    for (rank, meta), col in zip(cand_meta.items(), COND_COL):
        PT._outline(im, mp, meta["poly_cm"], col, 2)
        PT._t(im, f"B rank {rank}: {meta['state']}", mp(meta["poly_cm"][1])[0] + np.array([6, 6 + 18 * (rank % 3)]), col, 0.42)
    save("01_environment.png", im, gen + ["", "#Legend", "green = drivable (Z02); red = static obstacle; magenta = no-go", "cyan fill = observed loading bay; orange dashed = Qwen semantic destination",
                                          "pale dashed = placement region candidate (bay floor minus objects; UNVERIFIED)", "white rects = robot start body (35x24 nominal and +margin)",
                                          "red = condition A target; magenta/blue/green = condition B PROBABLE candidates (photo decision AMBIGUOUS)",
                                          "", f"command: {cfg['command']}", "(text in figures is ASCII; the command is in task_plan.json)"])
    # 2 path A, 3 path B
    for tag, name, title in (("A", "02_path_A.png", "PATH A: robot start -> pick-up approach"), ("B", "03_path_B.png", "PATH B: pick-up approach -> loading-bay approach")):
        im = ctx["img"].copy()
        layers(im, mp, ctx, text=False)
        lines = gen + ["", "#" + title]
        if tag in A["paths"]:
            p, ck = A["paths"][tag], A["checks"][tag]
            if tag == "A":
                robot_start(im, mp, A["start_cm"], nom, marg)
                PT._outline(im, mp, ctx["targetA_cm"], C["target"], 3)
            else:
                PT._rect(im, mp, A["paths"]["A"]["goal_cm"], A["paths"]["A"]["footprint_extent_xy_cm"], C["goal"], 2)
            draw_path(im, mp, p, C[tag], tag, step)
            lo = ck["swept_layer_overlap"]
            lines += [f"heading {p['heading_deg']} deg, body {p['footprint_extent_xy_cm'][0]:.0f}x{p['footprint_extent_xy_cm'][1]:.0f} cm, length {p['length_cm']} cm, {len(p['waypoints_cm'])} waypoints",
                      f"goal {p['goal_cm']} cm (image px {p['goal_image_px']})",
                      f"whole-body sweep: {'PASS' if ck['whole_body_sweep']['ok'] else 'FAIL'} ({ck['whole_body_sweep']['n_samples']} samples, 0.25 cm)",
                      f"v0.2 validator (inscribed disc): {ck['v02_validator_inscribed_disc']['valid']}",
                      f"swept body on: static {lo['on_static_obstacle_px']} px, no-go {lo['on_no_go_px']} px, outside drivable {lo['outside_drivable_px']} px, outside workspace {lo['outside_workspace_px']} px",
                      f"approach body on free floor: {ck['approach_pose_body_free']}",
                      f"gap the gripper must bridge: {p['approach']['body_to_target_gap_cm']} cm (reach UNVERIFIED)"]
        else:
            lines += [f"NOT GENERATED: {A['failure_' + tag]}"]
        save(name, im, lines + ["", "thin rects = whole body every %d cm along the path (display); yellow = approach body" % step])
    # 4 full plan
    im = ctx["img"].copy()
    layers(im, mp, ctx, text=False)
    robot_start(im, mp, A["start_cm"], nom, marg)
    PT._outline(im, mp, ctx["targetA_cm"], C["target"], 3)
    for tag in ("A", "B"):
        if tag in A["paths"]:
            draw_path(im, mp, A["paths"][tag], C[tag], tag, step, coords=False)
    if "A" in A["paths"]:
        PT._t(im, "PICKUP / VERIFY REQUIRED", mp(A["paths"]["A"]["goal_cm"])[0] + np.array([-90, -60]), C["goal"], 0.6)
    if "B" in A["paths"]:
        PT._t(im, "PLACE (assumed)", mp(A["paths"]["B"]["goal_cm"])[0] + np.array([-40, -50]), C["goal"], 0.6)
    chain = " -> ".join(s["state"] for s in A["states"]) or "no plan"
    lines = gen + ["", "#START -> PATH A -> PICKUP / VERIFY REQUIRED -> PATH B -> PLACE", f"states: {chain}", "(every state: executed = false)", "", "#Coordinates (floor cm, PROVISIONAL)"]
    for tag in ("A", "B"):
        if tag in A["paths"]:
            lines.append(f"path {tag} ({'blue' if tag == 'A' else 'orange'}): " + " -> ".join(f"({x:.0f},{y:.0f})" for x, y in A["paths"][tag]["waypoints_cm"]) + f"  = {A['paths'][tag]['length_cm']} cm")
    for s in A["states"]:
        if "gripper_must_bridge_cm" in s:
            lines.append(f"{s['state']}: gripper must bridge {s['gripper_must_bridge_cm']} cm (UNVERIFIED)")
    save("04_full_plan.png", im, lines)
    # 5 BEV
    S = 3
    W_, H_ = round((ctx["size_cm"][0] + 2 * fr.pad) * S), round((ctx["size_cm"][1] + 2 * fr.pad) * S)
    M = np.array([[S, 0, fr.pad * S], [0, S, fr.pad * S], [0, 0, 1]]) @ fr.H
    bev = (cv2.warpPerspective(ctx["img"], M, (W_, H_)) * 0.45).astype(np.uint8)
    mb = PT.Mapper("bev", fr, S)
    bev = cv2.copyMakeBorder(bev, 30, 40, 50, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    class Shift:
        def __call__(self, cm):
            return mb(cm) + np.array([50, 30])
    ms = Shift()
    for x in range(0, ctx["size_cm"][0] + 1, 50):
        cv2.line(bev, tuple(ms([[x, 0]])[0]), tuple(ms([[x, ctx["size_cm"][1]]])[0]), (90, 90, 90), 1)
        PT._t(bev, f"{x}", ms([[x, ctx["size_cm"][1]]])[0] + np.array([-10, 40]), (255, 255, 255), 0.42)
    for y in range(0, ctx["size_cm"][1] + 1, 50):
        cv2.line(bev, tuple(ms([[0, y]])[0]), tuple(ms([[ctx["size_cm"][0], y]])[0]), (90, 90, 90), 1)
        PT._t(bev, f"{y}", ms([[0, y]])[0] + np.array([-46, 5]), (255, 255, 255), 0.42)
    PT._t(bev, "x [cm] ->", ms([[ctx["size_cm"][0] - 60, ctx["size_cm"][1]]])[0] + np.array([0, 22]), (255, 255, 255), 0.45)
    PT._t(bev, "y [cm]", (2, 20), (255, 255, 255), 0.45)
    cv2.rectangle(bev, tuple(ms([[0, 0]])[0]), tuple(ms([[ctx["size_cm"][0], ctx["size_cm"][1]]])[0]), (255, 255, 255), 1)
    layers(bev, ms, ctx, text=True)
    robot_start(bev, ms, A["start_cm"], nom, marg)
    PT._outline(bev, ms, ctx["targetA_cm"], C["target"], 2)
    for tag in ("A", "B"):
        if tag in A["paths"]:
            draw_path(bev, ms, A["paths"][tag], C[tag], tag, step, coords=False)
    save("05_bev_map.png", bev, gen + ["", "#BEV of the PROVISIONAL floor frame: 3 x 2 mats = 315 x 210 cm, axes in cm",
                                       "the floor frame comes from 8 mat-seam nodes of this photo and the user's 105 cm; it is NOT validated by human reference points",
                                       "photo content above the floor (boxes, robot, truck) is smeared by perspective; only floor-level geometry is meaningful",
                                       "body rectangles are drawn to scale in this frame"])
    # 6 condition B
    im = ctx["img"].copy()
    layers(im, mp, ctx, text=False)
    robot_start(im, mp, A["start_cm"], nom, marg)
    lines = gen + ["", "#Condition B: retained detector candidates (photo decision AMBIGUOUS: no target is chosen)"]
    for (rank, plan_), col in zip(B_plans.items(), COND_COL):
        meta = cand_meta[rank]
        PT._outline(im, mp, meta["poly_cm"], col, 3)
        for tag in ("A", "B"):
            if tag in plan_["paths"]:
                wp = plan_["paths"][tag]["waypoints_cm"]
                cv2.polylines(im, [mp(wp)], False, col, 3 if tag == "A" else 2, cv2.LINE_AA)
                PT._rect(im, mp, plan_["paths"][tag]["goal_cm"], plan_["paths"][tag]["footprint_extent_xy_cm"], col, 1)
        PT._t(im, f"rank {rank}", mp(meta["poly_cm"][2])[0] + np.array([6, 14]), col, 0.5)
        lines.append(f"rank {rank} ({meta['state']}; Qwen views {meta['crop_views']}): A {plan_['failure_A'] or 'generated'} "
                     + (f"{plan_['paths']['A']['length_cm']} cm, gap {plan_['paths']['A']['approach']['body_to_target_gap_cm']} cm" if 'A' in plan_['paths'] else '')
                     + f" | B {plan_['failure_B'] or 'generated'}")
    save("06_condition_B_conditional.png", im, lines + ["", "every plan here is CONDITIONAL_ON_CANDIDATE; none is the confirmed target"])
    return figs


# ------------------------------------------------------------------------------------------------ main
def main():
    cfg, v02 = load_cfg(), load_config()
    assert res(cfg["geometry_config"]) == (PT.HERE / "config.yaml").resolve()
    gcfg = PT.load_cfg()
    img_path = Path(cfg["image"])
    assert sha256_file(img_path) == cfg["image_sha256"], "image differs from the declared one"
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    fp_before = code_fingerprint()["sha256"]
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_route")
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    shutil.copy2(img_path, out / "inputs/input_E1_undistorted.png")
    (out / "config_snapshot.yaml").write_text((HERE / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "geometry_config_snapshot.yaml").write_text((PT.HERE / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    steps = []

    def step(name, status, **kw):
        assert status in STEP_STATUSES
        steps.append({"step": name, "status": status, **kw})

    ctx = build_context(cfg, gcfg, v02, img)
    st = ctx["st"]
    e1 = res(cfg["stored"]["e1_run"])
    rule = rule_based_parse(cfg["command"])
    sem = {"command": cfg["command"],
           "target_and_source": {"backend": "v0.2 rule_based_parse (deterministic, no model)", "raw": rule, "target": rule["target"],
                                 "note": "v0.2 build_task raises MISSING_INFORMATION for this command (no destination keyword); the destination comes from the stored Qwen reply"},
           "destination": {"backend": "Qwen3-VL-4B, STORED L2 reply (no new call)", "run": st["qwen_run"].name, "raw_response_sha256_16": st["raw_response_sha256"][:16],
                           "instruction_of_stored_reply": cfg["semantics"]["stored_destination_instruction"], "reuse_status": cfg["semantics"]["destination_reuse_status"],
                           "area_id": st["task_analysis"]["destination_area_id"], "function_name": st["area"]["function_name"], "box_image_px": [round(v, 1) for v in st["qwen_box_px"]],
                           "evidence": st["area"]["observed_evidence"], "reason": st["task_analysis"]["reason"], "decision": st["task_analysis"]["decision"]},
           "role_split": {"Qwen": "semantic destination (stored)", "rule parser": "target class/colour and source relation", "DINO/HSV": "visual candidates (stored)",
                          "Python": "floor transform, footprint, approach points, A*, collision checks", "this script": "step order and plan states (AnalysisRun not instantiated)"}}
    step("load_semantics", "OK", output=f"target {rule['target']}, destination {sem['destination']['function_name']} (stored, shorter wording)")
    if not ctx["ft"]["accepted"]:
        step("build_floor_frame", "FAILED", error="GEOMETRY_UNVERIFIED: floor transform refused by its declared acceptance rule")
        (out / "task_plan.json").write_text(json.dumps({"steps": steps, "floor_transform": ctx["ft"]}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print("floor transform refused; stopped")
        return out
    step("build_floor_frame", "UNCERTAIN", output=f"PROVISIONAL_UNVALIDATED (fit RMS {ctx['ft']['fit_rms_cm']} cm, seam dev <= {ctx['ft']['seam_max_abs_dev_cm']} cm)")
    step("estimate_start", "UNCERTAIN", output=f"{ctx['robot']['object_id']} box centre, ESTIMATED; heading UNKNOWN")
    step("build_map", "UNCERTAIN", output="DRAFT zones as MANUAL_MAP_PRIOR (human-unconfirmed)")
    L, W, m = 35.0, 24.0, ctx["ft"]["footprint_margin_cm"]
    start_img = ctx["starts"][0]

    # ---- condition A
    fcfg = yaml.safe_load(res(cfg["stored"]["funnel_config"]).read_text(encoding="utf-8"))
    boxA = fcfg["labels"][cfg["condition_A"]["photo_id"]]
    ctx["targetA_cm"] = set_target(ctx, boxA).tolist()
    target_A = {"name": cfg["condition_A"]["name"], "bbox_image_px": boxA, "label_status": fcfg["label_status"], "not": "ORACLE / GROUND_TRUTH",
                "floor_position_cm": np.round(np.mean(ctx["targetA_cm"], axis=0), 1).tolist(),
                "floor_position_status": "ESTIMATED floor projection of the image box; the cube is on the conveyor surface (height UNKNOWN), so it is displaced by parallax"}
    A, _ = run_target(ctx, "A_reference", boxA, L, W, m, start_img, target_A)
    A_nominal, _ = run_target(ctx, "A_reference_nominal", boxA, L, W, 0.0, start_img, target_A)
    set_target(ctx, boxA)
    r_disc = math.hypot(L + 2 * m, W + 2 * m) / 2
    disc = plan_disc(ctx, r_disc, start_img)
    step("condition_A_plan", "OK" if A["selection"]["status"] == "PATH_A_AND_B" else "FAILED", output=A["selection"]["status"], error=A["failure_A"] or A["failure_B"])

    # ---- condition B
    rr = PT.jload(res(cfg["stored"]["retention_run"]) / "result.json")
    e1p = next(p for p in rr["photos"] if p["photo_id"] == cfg["condition_A"]["photo_id"])
    qc = PT.jload(res(cfg["stored"]["crop_comparison_run"]) / "result.json")
    conveyor = next(o for o in PT.jload(e1 / "analysis/detections.json")["objects"] if o["object_id"] == "conveyor_1")
    near_px = v02["planning"]["relations"]["near_px"]
    B_plans, cand_meta = {}, {}
    for c in e1p["candidates"]:
        if c["state"] not in cfg["condition_B"]["plannable_states"]:
            continue
        views = None
        for q in qc["candidates"]:
            if iou(c["bbox_xyxy"], q["bbox_xyxy_raw_px"]) >= cfg["condition_B"]["crop_verdict_match_iou"]:
                views = {"A_stored": q["stored_A"]["status"], "B_no_marker": q["B"]["status"], "C_more_context": q["C"]["status"], "matched": q["id"]}
        gap = box_gap(c["bbox_xyxy"], conveyor["bbox_xyxy"])
        tag = {"name": "CONDITIONAL_ON_CANDIDATE", "retention_rank": c["rank"], "state": c["state"], "bbox_image_px": c["bbox_xyxy"], "source_types": c["source_types"],
               "stored_qwen": c["stored_qwen"], "disagreement": c["disagreement"], "qwen_crop_views": views,
               "context_info_only": {"gap_to_verified_conveyor_px": round(gap, 1), f"within_v02_near_px_{near_px}": gap <= near_px,
                                     "note": "recorded for the next study; NOT used to choose a target here"}}
        plan_c, _ = run_target(ctx, f"B_rank{c['rank']}", c["bbox_xyxy"], L, W, m, start_img, tag)
        B_plans[c["rank"]] = plan_c
        cand_meta[c["rank"]] = {"poly_cm": ctx["target_cm"], "state": c["state"], "crop_views": None if views is None else f"{views['A_stored']}/{views['B_no_marker']}/{views['C_more_context']}"}
    decision = e1p["photo_decision_from_stored_answers"]
    condB = {"photo_decision": decision, "selected_target": None if decision == "AMBIGUOUS" else decision,
             "rule": "candidates in VLM_ACCEPTED/PROBABLE each get a CONDITIONAL plan; with AMBIGUOUS no candidate is chosen and none is CONFIRMED",
             "n_candidates_preserved": len(e1p["candidates"]), "candidates_all": e1p["candidates"], "conditional_plans": {str(k): v for k, v in B_plans.items()}}
    step("condition_B_plan", "UNCERTAIN", output=f"{len(B_plans)} conditional plans, photo decision {decision}")
    old = old_path_check(ctx, cfg, m)
    step("recheck_old_path", "FAILED" if old["verdict"].startswith("FAILS") else "OK", output=old["verdict"])

    # ---- validation summary
    def plan_checks(p):
        c = p["checks"]
        return {k: {"sweep_ok": c[k]["whole_body_sweep"]["ok"], "v02_valid": c[k]["v02_validator_inscribed_disc"]["valid"], "layers_clear": c[k]["swept_layer_overlap"]["clear"],
                    "approach_body_free": c[k]["approach_pose_body_free"]} for k in ("A", "B") if k in c} | ({"A_B_connected": c["A_B_connected"]["B_starts_where_A_ends"]} if "A_B_connected" in c else {})
    validation = {
        "inputs": {"image_sha256": cfg["image_sha256"], "stored_e1_input_equals_image": sha256_file(e1 / "input/raw_frame.png") == cfg["image_sha256"],
                   "condition_A_box_equals_funnel_label": boxA == fcfg["labels"][cfg["condition_A"]["photo_id"]],
                   "condition_B_boxes_equal_retention_record": all(B_plans[k]["target"]["bbox_image_px"] == next(c["bbox_xyxy"] for c in e1p["candidates"] if c["rank"] == k) for k in B_plans),
                   "retained_candidates_preserved": len(condB["candidates_all"]) == e1p["n_candidates"]},
        "footprint_frame": {"primary_extent_cm": [L + 2 * m, W + 2 * m], "margin_cm": m, "bev_px_per_cm": ctx["fr"].s,
                            "cm_bev_roundtrip_max_err_cm": float(np.abs(ctx["fr"].bev_to_cm(ctx["fr"].cm_to_bev(ctx["targetA_cm"])) - np.array(ctx["targetA_cm"])).max())},
        "condition_A": plan_checks(A), "condition_A_nominal": plan_checks(A_nominal), "condition_B": {str(k): plan_checks(v) for k, v in B_plans.items()},
        "uncertainty": {"no_candidate_confirmed": all(c["state"] != "CONFIRMED" for c in e1p["candidates"]), "condition_B_selected_target": condB["selected_target"]},
        "failure_handling": {"old_35px_path": old["verdict"], "disc_comparison": disc["result"]},
        "geometry_status": "GEOMETRY_UNVERIFIED for physical safety: PROVISIONAL floor transform + DRAFT map; checks are exact only inside that model"}
    task_plan = {"experiment_id": cfg["experiment_id"], "created": datetime.now().isoformat(timespec="seconds"), "claims": cfg["claims"],
                 "input_image": {"path": str(img_path), "sha256": cfg["image_sha256"], "size": [img.shape[1], img.shape[0]]}, "semantics": sem,
                 "sources": {k: str(res(v)) for k, v in cfg["stored"].items()}, "steps": steps,
                 "robot": {"object_id": ctx["robot"]["object_id"], "bbox_image_px": ctx["robot"]["bbox_xyxy"], "source": ctx["robot"]["source"], "start_image_px": start_img,
                           "start_cm": A["start_cm"], "status": "ESTIMATED (box centre; heading UNKNOWN)", "footprint_user_cm": [35, 24, 16], "wheel_inclusion": "UNKNOWN"},
                 "destination_layers": {"1_qwen_semantic_destination_cm": np.round(ctx["qwen_cm"], 1).tolist(), "2_observed_bay_cm": np.round(ctx["bay_cm"], 1).tolist(),
                                        "3_robot_place_approach": (A["paths"].get("B") or {}).get("goal_cm"),
                                        "4_placement_region_candidate_cm": [np.round(p, 1).tolist() for p in ctx["place_candidate_cm"]], "4_status": "UNVERIFIED (geometry only)"},
                 "map": {"zones_cm": ctx["spec_cm"], "status": "MANUAL_MAP_PRIOR (DRAFT_UNCONFIRMED)", "clipped": ctx["clipped"]},
                 "floor_transform": {k: v for k, v in ctx["ft"].items() if k not in ("H_floor_cm_to_img",)},
                 "condition_A": {"plan": A, "comparisons": {"nominal_rectangle_no_margin": {"selection": A_nominal["selection"], "failure_A": A_nominal["failure_A"], "failure_B": A_nominal["failure_B"],
                                                                                         "lengths_cm": {k: v["length_cm"] for k, v in A_nominal["paths"].items()}},
                                                         "circumscribed_disc_with_margin": disc}},
                 "condition_B": condB, "old_path_recheck": old, "figures": [], "v02_code_before": fp_before, "no_hardware_command": True}
    task_plan["figures"] = render(ctx, A, B_plans, cand_meta, cfg, out)
    task_plan["v02_code_after"] = code_fingerprint()["sha256"]
    paths = {"condition_A": A["paths"], **{f"condition_B_rank{k}": v["paths"] for k, v in B_plans.items()}}
    for tag in ("A", "B"):
        (out / f"path_{tag.lower()}.json").write_text(json.dumps({k: v.get(tag) for k, v in paths.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "task_plan.json").write_text(json.dumps(task_plan, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "manifest.json").write_text(json.dumps({"files_sha256": files}, indent=2), encoding="utf-8")
    print("wrote", out)
    print(json.dumps({"steps": steps, "A": {"sel": A["selection"], "fA": A["failure_A"], "fB": A["failure_B"]}, "A_nominal": A_nominal["selection"]["status"], "disc": disc,
                      "B": {k: (v["selection"]["status"], v["failure_A"], v["failure_B"], {t: p["length_cm"] for t, p in v["paths"].items()}) for k, v in B_plans.items()},
                      "old": old["verdict"], "validation": {k: validation[k] for k in ("condition_A", "condition_B", "uncertainty", "inputs")}}, indent=1, default=str, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
