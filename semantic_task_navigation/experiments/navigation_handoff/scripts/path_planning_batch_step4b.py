#!/usr/bin/env python3
"""STEP 4-B batch: preliminary Path A / Path B for photos the user has not approved (config step4b_batch).

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/navigation_handoff/scripts/path_planning_batch_step4b.py        # new results/<time>_step4b_batch/

Planner and checks are imported unchanged (path_planning.py, path_planning_wheel_boundary.py) at the 2.0 cm development margin.
New here: PROVISIONAL positions (robot by ECC against photo_5, cube = AI-preliminary label, B = STEP 4-A empty pallet), per-photo obstacle
cells (all inventory pallets block wheels and body; the static PGM is not changed), result statuses, conditional Path B, figures, the
8-photo comparison (photo_1 and photo_5 are READ from their existing runs, not re-run). No model call, no ROS, no robot.
"""
import json
import math
import platform
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coordinate_transform as CT  # noqa: E402
import map_export as ME  # noqa: E402
import object_positions as OP  # noqa: E402
import path_planning as PP  # noqa: E402
import path_planning_wheel_boundary as WB  # noqa: E402
import survey_photos_step4a as SV  # noqa: E402
from semantic_task_nav.config import code_fingerprint, sha256_file  # noqa: E402

STATUS = {"VALID": "path found and every collision / boundary check passed in the current map model",
          "INVALID_PATH": "a path exists but a collision / boundary check failed", "NOT_FOUND": "no valid path or stop pose under the current conditions",
          "CONDITIONAL": "computed from an assumed start (grasp-pose candidate or robot start) because Path A is not valid",
          "REVIEW_REQUIRED": "positions are provisional or a target is missing: user confirmation needed",
          "UNVERIFIED": "physical conditions (gripper offset / reach, dynamics) not verified"}


def robot_pose(cfg, photo, frame):
    rc = cfg["step4b_batch"]["robot_estimate"]
    ref_photo = next(p for p in CT.resolve_photos(cfg) if p["name"] == rc["reference_photo"])
    x0, y0, x1, y1 = rc["window_xyxy"]
    mx0, my0, mx1, my1 = rc["mask_xyxy"]
    g = lambda p: cv2.GaussianBlur(cv2.cvtColor(cv2.imread(p)[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)  # noqa: E731
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    mask[my0 - y0:my1 - y0, mx0 - x0:mx1 - x0] = 1
    cc, warp = cv2.findTransformECC(g(ref_photo["path"]), g(photo["path"]), np.eye(2, 3, dtype=np.float32), cv2.MOTION_EUCLIDEAN,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), mask, 5)
    tf = lambda p: warp @ np.array([p[0] - x0, p[1] - y0, 1.0]) + [x0, y0]  # noqa: E731
    w = {k: tf(v) for k, v in rc["wheel_reading_px"].items()}
    front, rear = (w["left_top"] + w["left_bottom"]) / 2, (w["right_top"] + w["right_bottom"]) / 2
    centre = np.mean(list(w.values()), axis=0)
    return {"base_link_px": [round(float(v), 1) for v in centre], "yaw_map_deg": OP.yaw_from_points(frame, rear, front),
            "ecc_correlation": round(float(cc), 4), "ecc_rotation_deg": round(float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))), 2),
            "ecc_shift_px": [round(float(v), 1) for v in warp[:, 2]], "method": rc["method"], "status": "PROVISIONAL (REVIEW_REQUIRED)"}


def obstacle_cells(grid, frame, boxes, px_per_cm):
    sub = int(round(grid.res_cm * px_per_cm))
    m = np.zeros((grid.rows * sub, grid.cols * sub), np.uint8)
    for b in boxes:
        corners = frame.img_to_floor([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
        cv2.fillPoly(m, [np.round((corners - [grid.x0, grid.y0]) * px_per_cm).astype(np.int32)], 1)
    return m.reshape(grid.rows, sub, grid.cols, sub).any(axis=(1, 3))


def gripper_line_hits(dm, cand, target_f, other_obst, nominal, step_cm=0.5):
    """Layers crossed by the straight line from the nominal body front to the target (static obstacles / other per-photo objects)."""
    yaw = math.radians(cand["yaw_map_deg"])
    f = np.array([math.cos(yaw), -math.sin(yaw)])                          # facing in floor cm (y towards the camera)
    front = np.array(cand["centre_floor_cm"]) + f * nominal[0] / 2
    n = max(1, int(math.ceil(np.linalg.norm(target_f - front) / step_cm)))
    hits = []
    for k in range(n + 1):
        r, c = dm.cell_of_floor(front + (target_f - front) * k / n)
        if not dm.in_grid(r, c):
            continue
        if dm.layers[r, c] == ME.STATIC_OBSTACLE and "STATIC_OBSTACLE" not in hits:
            hits.append("STATIC_OBSTACLE")
        if other_obst[r, c] and "PER_PHOTO_OBJECT" not in hits:
            hits.append("PER_PHOTO_OBJECT")
    return hits


def select_and_plan_checked(dm, kind, target_f, start_f, start_yaw, nominal, m, pcfg, gripper, esc, other_obst):
    """path_planning.select_and_plan with the gripper-line rule applied BEFORE the legs are planned (config step4b_batch.gripper_line_rule)."""
    cands = PP.approach_candidates(dm, target_f, nominal, m, pcfg["approach_yaws_map_deg"], pcfg["approach_search_cm"], gripper)
    for c in cands:
        if c["status"] != "POSE_FOUND":
            continue
        hits = gripper_line_hits(dm, c, target_f, other_obst, nominal)
        c["gripper_line_crosses"] = hits
        if hits:
            c["status"] = "GRIPPER_LINE_BLOCKED"
            continue
        best, tried = PP.plan_leg(dm, kind, start_f, start_yaw, tuple(c["cell"]), c["yaw_map_deg"], nominal, m, pcfg, esc)
        c["leg_attempts"] = [{k: v for k, v in t.items() if k != "raw_cells"} for t in tried]
        c["leg"] = best
        if best is None:
            c["status"] = "NO_FEASIBLE_LEG"
            continue
        c["leg_length_cm"] = round((len(best["raw_cells"]) - 1) * dm.grid.res_cm, 1)
        c["status"] = "FEASIBLE_PLAN_GRASP_UNVERIFIED"
    feas = [c for c in cands if c["status"] == "FEASIBLE_PLAN_GRASP_UNVERIFIED"]
    return cands, (min(feas, key=lambda c: (c["front_to_target_gap_cm"], c["leg_length_cm"])) if feas else None)


def dual_map(cfg, pgm, layers, obst, grid, frame, urdf_env, nominal):
    vals, pc, wc = cfg["map_export"]["values"], cfg["planning"], cfg["wheel_boundary_experiment"]
    code_of = {v.split(" (")[0]: k for k, v in ME.CODE_NAMES.items()}
    pgm_w = pgm.copy()
    pgm_w[obst] = vals["occupied"]
    body_ok = ~np.isin(layers, [code_of[c] for c in wc["body_blocking_codes"]]) & ~obst
    pm_w = PP.PgmMap(pgm_w, layers, vals, grid, frame, pc["raster_px_per_cm"], pc["safety_margin_cm"])
    pm_b = PP.PgmMap(np.where(body_ok, vals["free"], vals["occupied"]).astype(np.uint8), layers, vals, grid, frame, pc["raster_px_per_cm"], pc["safety_margin_cm"])
    return WB.DualMap(pm_w, pm_b, urdf_env, nominal)


def leg_summary(rec):
    """Works on a describe_leg-based record of this script AND on path_A / path_B of an existing wheel-boundary run."""
    if not rec:
        return None
    rots = [e for e in rec["events"] if e["type"] == "rotate"]
    return {"length_cm": rec["length_cm"], "direction_changes": rec["n_turns"], "body_rotations": len(rots),
            "rotations": [{"at_map": None, "from": e["from_yaw_map_deg"], "to": e["to_yaw_map_deg"], "direction": e["direction"],
                           "sweep_deg": e["per_direction"][e["direction"]]["sweep_deg"]} for e in rots],
            "lateral_cm": round(sum(s["length_cm"] for s in rec["segments"] if s["body_vs_move"] == "lateral"), 1),
            "checks_passed": rec["collision_checks_passed"], "strategy": rec["strategy"]}


def cause_of(cands, kind):
    if not cands:
        return f"{kind}: no target"
    st = {c["yaw_map_deg"]: c["status"] for c in cands}
    if all(s == "NO_COLLISION_FREE_POSE_ON_AXIS" for s in st.values()):
        return f"{kind}: no collision-free stop on the target axis for any of 0/90/180/270 (wheel boundary / body / per-photo obstacles)"
    blocked = {c["yaw_map_deg"]: c.get("gripper_line_crosses") for c in cands if c["status"] == "GRIPPER_LINE_BLOCKED"}
    if blocked and all(s in ("NO_COLLISION_FREE_POSE_ON_AXIS", "GRIPPER_LINE_BLOCKED") for s in st.values()):
        return f"{kind}: the only collision-free stops on the target axis need the gripper to reach across {blocked} -> rejected"
    legs = {c["yaw_map_deg"]: [(a["strategy"], a["status"]) for a in c.get("leg_attempts", [])] for c in cands if c["status"] == "NO_FEASIBLE_LEG"}
    return f"{kind}: stop pose found but no feasible leg {legs}" if legs else f"{kind}: {st}"


def plan_photo(cfg, ctx_sv, photo, frame, grid, pgm, layers, urdf, nominal, out):
    bc, pcfg = cfg["step4b_batch"], cfg["planning"]
    m = float(bc["margin_cm"])
    labels = yaml.safe_load(CT.resolve(bc["cube_labels"]).read_text(encoding="utf-8"))["labels"]
    inv = cfg["step4a"]["inventory"][photo["name"]]
    robot = robot_pose(cfg, photo, frame)
    lab = labels[photo["earlier_id"]]
    cube_px = [(lab[0] + lab[2]) / 2, (lab[1] + lab[3]) / 2]
    place_px = None if not inv["pallet_empty"] else [(inv["pallet_empty"][0][0] + inv["pallet_empty"][0][2]) / 2, (inv["pallet_empty"][0][1] + inv["pallet_empty"][0][3]) / 2]
    boxes = [{"role": "A cube pallet (target A, obstacle for wheels/body)", "box_px": inv["cube"]}] + \
            [{"role": f"empty pallet {i + 1}" + (" (target B, obstacle for wheels/body)" if i == 0 else " (obstacle)"), "box_px": b} for i, b in enumerate(inv["pallet_empty"])]
    obst = obstacle_cells(grid, frame, [b["box_px"] for b in boxes], cfg["planning"]["raster_px_per_cm"])
    px = cfg["planning"]["raster_px_per_cm"]
    other_A = obstacle_cells(grid, frame, [b["box_px"] for b in boxes[1:]], px)                 # everything except the cube pallet
    other_B = obstacle_cells(grid, frame, [b["box_px"] for i, b in enumerate(boxes) if i != 1], px) if len(boxes) > 1 else obst
    dm = dual_map(cfg, pgm, layers, obst, grid, frame, urdf["envelope_cm"], nominal)
    mt = WB.Metrics(dm)
    gripper = {**cfg["gripper"], "offset_y_m_assumed": pcfg["gripper_dy_assumption_m"]}
    pt = lambda uv: OP._pt(frame, uv)  # noqa: E731
    pos = {"robot_base_link": {**pt(robot["base_link_px"]), "yaw_map_deg": robot["yaw_map_deg"]}, "A_grasp_target": pt(cube_px),
           "B_placement_target": pt(place_px) if place_px else None}
    start_f, start_yaw = np.array(pos["robot_base_link"]["floor_cm"]), float(robot["yaw_map_deg"])
    cube_f = np.array(pos["A_grasp_target"]["floor_cm"])
    place_f = np.array(pos["B_placement_target"]["floor_cm"]) if place_px else None

    gc, gsel = select_and_plan_checked(dm, "A", cube_f, start_f, start_yaw, nominal, m, pcfg, gripper, True, other_A)
    legs, desc = {}, {}
    if gsel:
        legs["A"], desc["A"] = gsel["leg"], WB.relabel_body_vs_move(PP.describe_leg(dm, gsel["leg"], nominal, m, pcfg))
        a_status = "VALID" if desc["A"]["collision_checks_passed"] else "INVALID_PATH"
    else:
        a_status = "NOT_FOUND"
    b_basis, pc, psel = None, [], None
    if place_f is None:
        b_status = "REVIEW_REQUIRED"
        b_basis = "no empty pallet in this photo: placement target must be given by the user (none invented)"
    else:
        if a_status == "VALID":
            b_start, b_yaw, esc, b_basis = np.array(gsel["centre_floor_cm"]), gsel["yaw_map_deg"], False, "grasp stop of Path A"
        else:
            cand = [c for c in gc if c.get("centre_floor_cm") is not None and c["status"] != "GRIPPER_LINE_BLOCKED"]
            if cand:
                c0 = min(cand, key=lambda c: c["front_to_target_gap_cm"])
                b_start, b_yaw, esc = np.array(c0["centre_floor_cm"]), c0["yaw_map_deg"], False
                b_basis = f"CONDITIONAL: grasp-pose candidate yaw {c0['yaw_map_deg']} {c0['pose_map']} (Path A not valid)"
            else:
                b_start, b_yaw, esc = start_f, start_yaw, True
                b_basis = "CONDITIONAL: from the robot start (no grasp-pose candidate); not a task sequence"
        pc, psel = select_and_plan_checked(dm, "B", place_f, b_start, b_yaw, nominal, m, pcfg, gripper, esc, other_B)
        if psel:
            legs["B"], desc["B"] = psel["leg"], WB.relabel_body_vs_move(PP.describe_leg(dm, psel["leg"], nominal, m, pcfg))
            b_status = ("VALID" if a_status == "VALID" else "CONDITIONAL") if desc["B"]["collision_checks_passed"] else "INVALID_PATH"
        else:
            b_status = "NOT_FOUND"
    for key in legs:
        PP.write_csv(out / f"path_{key}.csv", legs[key], desc[key])

    def leg_rec(key):
        if key not in legs:
            return None
        lg, d = legs[key], desc[key]
        return {"strategy": lg["strategy"], "rotation_cell": lg.get("rotation_cell"), "length_cm": d["length_cm"], "n_raw_points": len(lg["raw_cells"]),
                "n_turns": d["n_turns"], "events": lg["events"], "segments": d["segments"], "poses": PP.pose_list(lg, d), "checks": d["checks"],
                "collision_checks_passed": d["collision_checks_passed"], "nominal_footprint_metrics": mt.path(d, cfg["wheel_boundary_experiment"]["metrics_step_cm"]),
                "baseline_v02_astar_4neighbours": lg["baseline_v02_astar_4neighbours"], "raw_path_file": f"path_{key}.csv"}
    yaws = pcfg["approach_yaws_map_deg"]
    refA = None if gsel else WB.nearest_reference_pose(dm, cube_f, nominal, m, yaws)
    refB = None if (psel or place_f is None) else WB.nearest_reference_pose(dm, place_f, nominal, m, yaws)
    strip = lambda cs: [{k: v for k, v in c.items() if k != "leg"} for c in cs]  # noqa: E731
    rec = {"photo": {k: photo[k] for k in ("name", "sha256", "earlier_id", "current_file")}, "photo_path_now": photo["path"],
           "file_name_matches_config": photo["file_name_matches_config"], "positions": pos, "robot_estimate": robot,
           "position_status": "PROVISIONAL (REVIEW_REQUIRED): STEP 4-A inventory, AI-preliminary cube label, ECC robot estimate; not user-approved",
           "per_photo_obstacles": boxes, "per_photo_obstacle_cells": int(obst.sum()), "margin_cm": m,
           "path_A_status": a_status, "path_B_status": b_status, "path_B_basis": b_basis,
           "path_A": leg_rec("A"), "path_B": leg_rec("B"), "summary_A": leg_summary(leg_rec("A")), "summary_B": leg_summary(leg_rec("B")),
           "grasp_stop": WB.stop_record(dm, mt, gsel, "A grasp target"), "place_stop": WB.stop_record(dm, mt, psel, "B placement target"),
           "grasp_candidates": strip(gc), "place_candidates": strip(pc),
           "cause_A": None if gsel else cause_of(gc, "A"), "cause_B": None if (psel or place_f is None) else cause_of(pc, "B"),
           "reference_nearest_stop_A": refA, "reference_nearest_stop_B": refB,
           "straight_line_crossings": {"robot_to_A": SV.segment_crossings(ctx_sv, start_f, cube_f),
                                       "A_to_B": SV.segment_crossings(ctx_sv, cube_f, place_f) if place_f is not None else None},
           "grasp_placement": "UNVERIFIED (gripper offset and reach not measured)", "driving": "NOT_FOR_DRIVING (2.0 cm margin < position uncertainty)"}
    (out / "path_plan.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (out / "object_positions_provisional.json").write_text(json.dumps({k: rec[k] for k in ("photo", "positions", "robot_estimate", "position_status", "per_photo_obstacles")},
                                                                      indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    wy = {"frame_id": "map", "status": f"A {a_status}, B {b_status}; positions PROVISIONAL / REVIEW_REQUIRED; grasp/placement UNVERIFIED; NOT_FOR_DRIVING",
          "note": "yaw = body heading (gripper side), not the movement direction", "path_B_basis": b_basis,
          "path_A": rec["path_A"] and rec["path_A"]["poses"], "path_B": rec["path_B"] and rec["path_B"]["poses"],
          "grasp_stop_pose": gsel and {**gsel["pose_map"], "status": "UNVERIFIED"}, "place_stop_pose": psel and {**psel["pose_map"], "status": "UNVERIFIED"}}
    (out / "waypoints.yaml").write_text(yaml.safe_dump(wy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    val = {"photo": photo["name"], "path_A_status": a_status, "path_B_status": b_status, "path_B_basis": b_basis, "status_meaning": STATUS,
           "path_A_checks": rec["path_A"] and {"passed": rec["path_A"]["collision_checks_passed"], **rec["path_A"]["checks"], **rec["path_A"]["nominal_footprint_metrics"]},
           "path_B_checks": rec["path_B"] and {"passed": rec["path_B"]["collision_checks_passed"], **rec["path_B"]["checks"], **rec["path_B"]["nominal_footprint_metrics"]},
           "cause_A": rec["cause_A"], "cause_B": rec["cause_B"], "positions": "REVIEW_REQUIRED", "grasp_placement": "UNVERIFIED", "driving": rec["driving"]}
    (out / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    masks = {"A": (other_A, obst & ~other_A), "B": (other_B, (obst & ~other_B) if len(boxes) > 1 else None)}   # (other objects, target's own pallet)
    render(photo, dm, rec, desc, gsel, psel, cube_f, place_f, start_f, start_yaw, refA, refB, nominal, boxes, out, ctx_sv, masks)
    return rec


LINE_COL = {"FREE": (0, 200, 0), "NOT_DRIVABLE": (150, 150, 150), "NO_GO": (200, 80, 160), "UNKNOWN": (90, 90, 90), "OUTSIDE_MAP": (90, 90, 90),
            "TARGET_PALLET": (0, 165, 255), "STATIC_OBSTACLE": (0, 0, 255), "OTHER_PER_PHOTO_OBJECT": (0, 0, 255)}
BLOCKING = ("STATIC_OBSTACLE", "OTHER_PER_PHOTO_OBJECT")                # same layers gripper_line_hits rejects


def run_name(r):
    return "/".join(r["zones"]) or {"OTHER_PER_PHOTO_OBJECT": "other per-photo object"}.get(r["layer"], r["layer"])


def gripper_line_runs(dm, ctx_sv, cand, target_f, other, own, nominal, step_cm=0.5):
    """Diagnostic only: the floor-level centre line from the nominal body front to the target (same sampling as gripper_line_hits),
    split into runs by layer. 2D only: gripper width / height / 3D clearance are not measured."""
    yaw = math.radians(cand["yaw_map_deg"])
    front = np.array(cand["centre_floor_cm"]) + np.array([math.cos(yaw), -math.sin(yaw)]) * nominal[0] / 2
    n = max(1, int(math.ceil(np.linalg.norm(target_f - front) / step_cm)))
    runs = []
    for k in range(n + 1):
        p = front + (target_f - front) * k / n
        r, c = dm.cell_of_floor(p)
        if not dm.in_grid(r, c):
            name = "OUTSIDE_MAP"
        elif other[r, c]:
            name = "OTHER_PER_PHOTO_OBJECT"
        elif dm.layers[r, c] == ME.STATIC_OBSTACLE:
            name = "STATIC_OBSTACLE"
        elif own is not None and own[r, c]:
            name = "TARGET_PALLET"
        else:
            name = ME.CODE_NAMES[int(dm.layers[r, c])].split(" (")[0]
        if runs and runs[-1]["layer"] == name:              # runs split by layer only; edge cells of a zone raster can lie outside its polygon
            runs[-1]["to_floor_cm"] = p.round(2).tolist()
        else:
            runs.append({"layer": name, "zones": [], "from_floor_cm": p.round(2).tolist(), "to_floor_cm": p.round(2).tolist()})
        zone = ctx_sv.zone_at(p) if name in ("STATIC_OBSTACLE", "NO_GO") else None
        if zone and zone not in runs[-1]["zones"]:
            runs[-1]["zones"].append(zone)
    for r_ in runs:
        r_["length_cm"] = round(float(np.linalg.norm(np.subtract(r_["to_floor_cm"], r_["from_floor_cm"]))), 1)
        r_["from_map_m"], r_["to_map_m"] = [dm.frame.floor_to_map([q])[0].round(3).tolist() for q in (r_["from_floor_cm"], r_["to_floor_cm"])]
        r_["blocking"] = r_["layer"] in BLOCKING
    return front, runs


def render(photo, dm, rec, desc, gsel, psel, cube_f, place_f, start_f, start_yaw, refA, refB, nominal, boxes, out, ctx_sv, masks):
    """Basic figure: computed paths and valid stops only. Diagnostic figure: every stop candidate, reference stops and the gripper
    baselines coloured by layer (not paths). Candidate data in path_plan.json is not changed."""
    g = dm.grid
    to_map = lambda f: (np.asarray(f, float) - [g.x0, g.y0]) / g.res_cm * 6 + [70, 40]   # noqa: E731  render_map: margin 70/40 px, 6 px per cell
    to_img = lambda f: dm.frame.floor_to_img(np.asarray(f, float))                    # noqa: E731
    start = {"floor": start_f.tolist(), "yaw": start_yaw, "size": nominal, "color": (200, 200, 200), "dot": (255, 255, 255),
             "label": f"START yaw {start_yaw:.1f} (PROVISIONAL)"}
    poses = [start]
    if gsel:
        poses.append({"floor": gsel["centre_floor_cm"], "yaw": gsel["yaw_map_deg"], "size": nominal, "color": (0, 0, 255), "dot": (0, 0, 255),
                      "label": f"GRASP STOP yaw {gsel['yaw_map_deg']:.0f} UNVERIFIED", "target": cube_f.tolist()})
    if psel:
        poses.append({"floor": psel["centre_floor_cm"], "yaw": psel["yaw_map_deg"], "size": nominal, "color": (255, 0, 0), "dot": (255, 0, 0),
                      "label": f"PLACE STOP yaw {psel['yaw_map_deg']:.0f} UNVERIFIED" + (" (CONDITIONAL)" if rec["path_B_status"] == "CONDITIONAL" else ""),
                      "target": place_f.tolist()})
    head = f"{photo['current_file']} = {photo['name']} ({photo['earlier_id']}), positions PROVISIONAL -> REVIEW_REQUIRED"
    lines = [head,
             f"Path A: {rec['path_A_status']}" + (f", {desc['A']['length_cm']} cm, {desc['A']['n_turns']} turns" if 'A' in desc else f" - {rec['cause_A']}"),
             f"Path B: {rec['path_B_status']}" + (f", {desc['B']['length_cm']} cm, {desc['B']['n_turns']} turns" if 'B' in desc else f" - {rec['cause_B'] or rec['path_B_basis']}")]
    if rec["path_B_status"] == "CONDITIONAL":
        lines.append("B start: " + rec["path_B_basis"])
    lines = [w for ln in lines for w in textwrap.wrap(ln, 90, subsequent_indent="    ")]      # long causes were cut off at the image edge
    lines += ["orange dashed: per-photo pallets (obstacles for wheels/body)", "margin 2.0 cm: NOT_FOR_DRIVING; gripper offset not measured -> UNVERIFIED",
              "body overhang beyond the yellow line: unmapped objects other than the pallets -> UNVERIFIED",
              "only computed paths and valid stops are drawn here;", "    rejected / reference stops and gripper baselines: path_planning_diag_*.png"]
    targets = [{"kind": "cube", "floor": cube_f.tolist(), "label": "A? cube (REVIEW_REQUIRED)"}]
    if place_f is not None:
        targets.append({"kind": "place", "floor": place_f.tolist(), "label": "B? place (REVIEW_REQUIRED)"})
    for key in ("A", "B"):
        fv = desc.get(key, {}).get("checks", {}).get("whole_body_sweep_generic", {}).get("first_violation") if key in desc else None
        if fv:
            targets.append({"kind": "collision", "floor": fv["floor_cm"], "label": f"COLLISION on Path {key}"})
    img = cv2.imread(photo["path"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    PP.render_photo(img, dm, {"legs": desc, "poses": poses, "status_lines": lines, "targets": targets}, out / "path_planning_photo.png")
    PP.render_map(dm, {"legs": desc, "poses": poses, "status_lines": lines, "targets": targets}, out / "path_planning_map.png")

    # ---- diagnostic: candidates and gripper baselines (data: gripper_line_diagnostics.json; path_plan.json unchanged)
    diag, dposes, dlines = [], [start], [head, "DIAGNOSTIC FIGURE: stop candidates and gripper baselines - NOT paths, NOT stops to drive to",
                                         "baseline colours: green free | grey not drivable | violet no-go (gripper allowed)",
                                         "    orange target's own pallet (allowed) | RED static obstacle / other object (BLOCKING, X)",
                                         "baseline = 2D floor-level centre line; gripper width / height / 3D clearance not measured -> UNVERIFIED", ""]
    for key, lab, tgt, sel in (("grasp_candidates", "A", cube_f, gsel), ("place_candidates", "B", place_f, psel)):
        other, own = masks[lab]
        nopose = [c for c in rec[key] if not c.get("centre_floor_cm")]     # one panel line per leg (the map panel overflowed with 8 candidates)
        if nopose:
            dlines.append(f"{lab} yaw {'/'.join(str(c['yaw_map_deg']) for c in nopose)}: {nopose[0]['status']} (searched {nopose[0].get('searched_cm')} cm)")
        for c in rec[key]:
            d = {"leg": lab, "yaw_map_deg": c["yaw_map_deg"], "status": c["status"], "selected": bool(sel) and c.get("pose_map") == sel.get("pose_map")}
            if not c.get("centre_floor_cm"):
                d["reason"] = f"no collision-free stop on the target axis (searched {c.get('searched_cm')} cm)"
            else:
                front, runs = gripper_line_runs(dm, ctx_sv, c, tgt, other, own, nominal)
                blk = [r for r in runs if r["blocking"]]
                d.update({"stop_pose_map": c["pose_map"], "body_front_map_m": dm.frame.floor_to_map([front])[0].round(3).tolist(),
                          "target_map_m": dm.frame.floor_to_map([tgt])[0].round(4).tolist(), "front_to_target_gap_cm": c.get("front_to_target_gap_cm"),
                          "gripper_line_crosses_planner": c.get("gripper_line_crosses"), "runs": runs})
                cross = ", ".join(f"{run_name(r)} {r['length_cm']} cm" for r in blk)
                d["reason"] = {"GRIPPER_LINE_BLOCKED": f"rejected: gripper baseline crosses {cross}",
                               "NO_FEASIBLE_LEG": "stop is collision-free but no feasible leg (A* / rotation)",
                               "FEASIBLE_PLAN_GRASP_UNVERIFIED": "selected stop" if d["selected"] else "feasible, not selected (larger gap / longer leg)"}.get(c["status"], c["status"])
                col, dot, short = {"GRIPPER_LINE_BLOCKED": ((80, 0, 80), (220, 160, 255), "REJECTED"), "NO_FEASIBLE_LEG": ((0, 140, 140), (0, 200, 200), "NO LEG")}.get(
                    c["status"], (((0, 0, 255) if lab == "A" else (255, 0, 0)), ((0, 0, 255) if lab == "A" else (255, 0, 0)), "SELECTED" if d["selected"] else "FEASIBLE"))
                dposes.append({"floor": c["centre_floor_cm"], "yaw": c["yaw_map_deg"], "size": nominal, "color": col, "dot": dot,
                               "label": f"{short} {lab} yaw {c['yaw_map_deg']}"})
                d["_draw"] = (front, tgt, runs)
            if c.get("pose_map"):
                dlines.append(f"{lab} yaw {c['yaw_map_deg']}: {c['status']} at ({c['pose_map']['x_m']}, {c['pose_map']['y_m']}), "
                              f"gap {c.get('front_to_target_gap_cm')} cm - {d['reason']}")
            diag.append(d)
    for ref, lab in ((refA, "A"), (refB, "B")):
        if ref:
            f = dm.frame.map_to_floor([[ref["pose_map"]["x_m"], ref["pose_map"]["y_m"]]])[0]
            dposes.append({"floor": f.tolist(), "yaw": ref["yaw_map_deg"], "size": nominal, "color": (128, 128, 128), "dot": (128, 128, 128),
                           "label": f"REF ONLY {lab} (not a stop)"})
            dlines.append(f"REF ONLY {lab}: nearest collision-free body pose ({ref['pose_map']['x_m']}, {ref['pose_map']['y_m']}, {ref['yaw_map_deg']}) - gripper not facing the target")
    dlines = [w for ln in dlines for w in (textwrap.wrap(ln, 90, subsequent_indent="    ") or [""])]
    dscene = {"legs": {}, "poses": dposes, "status_lines": dlines, "targets": targets}
    PP.render_photo(img, dm, dscene, out / "path_planning_diag_photo.png")
    PP.render_map(dm, dscene, out / "path_planning_diag_map.png")
    (out / "gripper_line_diagnostics.json").write_text(json.dumps(
        {"photo": photo["name"], "rule": "step4b_batch.gripper_line_rule (config)", "frame": "map_m (REP-103); floor_cm for raw samples",
         "note": "diagnostic only, recomputed from path_plan.json candidates; 2D floor-level centre line; gripper width/height/3D UNVERIFIED",
         "candidates": [{k: v for k, v in d.items() if k != "_draw"} for d in diag],
         "reference_stops": {"A": refA, "B": refB}}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")

    # ---- post-draw with the same transforms: per-photo pallet outlines (both figures), coloured gripper baselines (diagnostic only)
    for name, conv in (("path_planning_photo.png", to_img), ("path_planning_map.png", to_map), ("path_planning_diag_photo.png", to_img), ("path_planning_diag_map.png", to_map)):
        im = cv2.imread(str(out / name))
        for b in boxes:
            x0, y0, x1, y1 = b["box_px"]
            cp = conv(dm.frame.img_to_floor([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
            cv2.polylines(im, [np.round(cp).astype(np.int32)], True, (0, 165, 255), 3 if conv is to_img else 2)
        if "diag" in name:
            for d in diag:
                if "_draw" not in d:
                    continue
                front, tgt, runs = d["_draw"]
                for r in runs:
                    q0, q1 = conv([r["from_floor_cm"], r["to_floor_cm"]])
                    cv2.line(im, PP._ip(q0), PP._ip(q1), LINE_COL[r["layer"]], 5 if r["blocking"] else 3, cv2.LINE_AA)
                    if r["blocking"]:
                        m = (q0 + q1) / 2
                        cv2.drawMarker(im, PP._ip(m), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 26, 4)
                        PP._text(im, f"X {run_name(r)} {r['length_cm']} cm", m + [14, 5], (0, 0, 255), 0.6, 2)
        for tf, ok, txt in ((cube_f, bool(gsel), f"Path A {rec['path_A_status']}"), (place_f, bool(psel), f"Path B {rec['path_B_status']}")):
            if tf is not None and not ok:                     # failure stated at the target itself (label kept short so it is not cut off)
                PP._text(im, txt, conv([tf])[0] + [8, 16], (0, 0, 255), 0.5, 1)
        cv2.imwrite(str(out / name), im)


def existing_row(cfg, name, rel):
    d = CT.resolve(rel)
    p = json.loads((d / "path_plan.json").read_text(encoding="utf-8"))
    return {"photo": name, "source": str(d), "positions": "USER_APPROVED", "path_A_status": "VALID" if p["path_A"] and p["path_A"]["collision_checks_passed"] else "NOT_FOUND",
            "path_B_status": "VALID" if p["path_B"] and p["path_B"]["collision_checks_passed"] else "NOT_FOUND",
            "summary_A": leg_summary(p["path_A"]), "summary_B": leg_summary(p["path_B"]),
            "note": "existing run read, not re-run; target pallets were NOT modelled as per-photo obstacles in that run"}, d


def main():
    cfg = CT.load_config()
    bc = cfg["step4b_batch"]
    before = code_fingerprint()["sha256"]
    photos = {p["name"]: p for p in CT.resolve_photos(cfg)}
    frame, fc, map_run = OP.load_step2_frame(cfg)
    g = fc["grid"]
    grid = CT.MapGrid(frame, g["extent_floor_cm"], g["resolution_m"])
    pgm, _ = ME.read_pgm(map_run / "maps/map.pgm")
    layers = np.load(map_run / "maps/map_layers.npy")
    map_sha = {k: sha256_file(map_run / "maps" / k) for k in ("map.pgm", "map.yaml", "map_layers.npy")}
    urdf = WB.read_urdf_wheels(cfg["wheel_boundary_experiment"]["urdf"])
    nominal = (cfg["robot"]["length_cm"], cfg["robot"]["width_cm"])
    ctx_sv = SV.Ctx(cfg)
    stamp = datetime.now().astimezone()
    root = CT.resolve(cfg["results_dir"]) / (stamp.strftime("%Y%m%d_%H%M%S") + "_step4b_batch")
    (root / "maps").mkdir(parents=True, exist_ok=False)
    for k in ("map.pgm", "map.yaml", "map_preview.png"):
        (root / "maps" / k).write_bytes((map_run / "maps" / k).read_bytes())
    rows = {}
    for name in bc["photos"]:
        out = root / name
        out.mkdir()
        rec = plan_photo(cfg, ctx_sv, photos[name], frame, grid, pgm, layers, urdf, nominal, out)
        rows[name] = {"photo": name, "source": str(out), "positions": "PROVISIONAL (REVIEW_REQUIRED)", "path_A_status": rec["path_A_status"],
                      "path_B_status": rec["path_B_status"], "path_B_basis": rec["path_B_basis"], "summary_A": rec["summary_A"], "summary_B": rec["summary_B"],
                      "cause_A": rec["cause_A"], "cause_B": rec["cause_B"], "crossings": rec["straight_line_crossings"],
                      "grasp_stop": rec["grasp_stop"] and {k: rec["grasp_stop"][k] for k in ("pose_map", "front_to_target_gap_cm", "lateral_offset_cm")},
                      "place_stop": rec["place_stop"] and {k: rec["place_stop"][k] for k in ("pose_map", "front_to_target_gap_cm", "lateral_offset_cm")},
                      "robot": rec["robot_estimate"], "file_name_matches_config": rec["file_name_matches_config"], "photo_path_now": rec["photo_path_now"]}
        print(name, rows[name]["path_A_status"], rows[name]["path_B_status"], json.dumps(rows[name]["summary_A"], default=float)[:160],
              json.dumps(rows[name]["summary_B"], default=float)[:160], rows[name]["cause_A"], rows[name]["cause_B"], flush=True)
    srcs = {}
    for name, rel in bc["compare_existing"].items():
        rows[name], srcs[name] = existing_row(cfg, name, rel)
    order = [f"photo_{i}" for i in range(1, 9)]
    figs_m = [(Path(rows[n]["source"]) / "path_planning_map.png") for n in order]
    figs_p = [(Path(rows[n]["source"]) / "path_planning_photo.png") for n in order]

    def tile(fs, w):
        ims = [cv2.imread(str(f)) for f in fs]
        ims = [cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1]))) for im in ims]
        for im, n in zip(ims, order):
            cv2.rectangle(im, (0, 0), (w, 30), (0, 0, 0), -1)
            r = rows[n]
            cv2.putText(im, f"{n}: A {r['path_A_status']} / B {r['path_B_status']} ({r['positions'].split(' ')[0]})", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return np.vstack([np.hstack(ims[i:i + 2]) for i in range(0, 8, 2)])
    cv2.imwrite(str(root / "comparison_8photos_maps.png"), tile(figs_m, 900))
    cv2.imwrite(str(root / "comparison_8photos_photos.png"), tile(figs_p, 880))
    after_ok = code_fingerprint()["sha256"] == before and all(sha256_file(map_run / "maps" / k) == v for k, v in map_sha.items()) \
        and all(sha256_file(p["path"]) == p["sha256"] for p in photos.values())
    (root / "comparison.json").write_text(json.dumps({"rows": [rows[n] for n in order], "status_meaning": STATUS, "margin_cm": bc["margin_cm"],
                                                      "environment_unchanged_after_run": after_ok}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (root / "config_snapshot.yaml").write_text(CT.CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    files = sorted(q for q in root.rglob("*") if q.is_file())
    (root / "manifest.json").write_text(json.dumps({
        "step": "STEP4B_batch", "run_id": root.name, "created": stamp.isoformat(timespec="seconds"), "python": platform.python_version(),
        "code_sha256": {p.name: sha256_file(p) for p in (Path(__file__).resolve(), HERE / "path_planning.py", HERE / "path_planning_wheel_boundary.py",
                                                         HERE / "survey_photos_step4a.py", HERE / "object_positions.py", HERE / "coordinate_transform.py")},
        "config_sha256": sha256_file(CT.CONFIG_PATH), "map_sha256": map_sha, "photos": {n: photos[n]["sha256"] for n in order},
        "photo_file_names_now": {n: Path(photos[n]["path"]).name for n in order}, "v02_code_fingerprint": before, "environment_unchanged_after_run": after_ok,
        "existing_runs_read": {k: str(v) for k, v in srcs.items()}, "outputs_sha256": {str(q.relative_to(root)): sha256_file(q) for q in files}}, indent=1), encoding="utf-8")
    print("run_dir", root, "| environment unchanged", after_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
