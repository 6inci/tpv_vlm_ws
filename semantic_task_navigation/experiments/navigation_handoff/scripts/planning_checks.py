#!/usr/bin/env python3
"""Shared deterministic checks for Path A / Path B (unified planning, user approval 2026-11-20). Policy: config/unified_planning.yaml
= the STEP 4-B batch checks:
  - wheel envelope (URDF, + margin) on FREE cells of map.pgm; body (+ margin) off STATIC_OBSTACLE / NO_GO / UNKNOWN,
  - every per-photo pallet blocks wheels and body (the static PGM is not changed),
  - gripper 2D baseline: a stop is rejected when the floor-level centre line from the nominal body front to the target crosses a blocking
    layer (static obstacle, a per-photo object other than the target's own pallet). Passing it is NOT evidence of a grasp or a placement:
    reach, width, height and 3D clearance stay UNVERIFIED and no reach limit is set,
  - 4-connected A*, static rotation check, whole-path sweep: path_planning.py / path_planning_wheel_boundary.py imported UNCHANGED.
One implementation of the gripper baseline serves both the decision and the diagnostic figure (the batch script had three).
Pure functions (no file output, no model, no ROS). Callers: path_planning_unified.py, later the Orchestrator.
Frames: floor_cm (STEP 2 floor frame, y towards the camera), map_m (ROS map, REP-103), image px = undistorted px of the photo.
"""
import math

import cv2
import numpy as np

import map_export as ME
import path_planning as PP
import path_planning_wheel_boundary as WB

STATUS = {"VALID": "path found and every collision / boundary check passed in the current map model",
          "INVALID_PATH": "a path exists but a collision / boundary check failed", "NOT_FOUND": "no valid path or stop pose under the current conditions",
          "CONDITIONAL": "DIAGNOSTIC: computed from an assumed start (grasp-pose candidate or robot start) because Path A is not valid; not a task sequence",
          "REVIEW_REQUIRED": "positions are provisional or a target is missing: user confirmation needed",
          "UNVERIFIED": "physical conditions (gripper reach / width / height / 3D interference, dynamics) not verified"}
GRIPPER_3D = "UNVERIFIED: 2D floor-level centre line only; gripper reach, width, height and 3D clearance are not measured"
REQUIRED_BLOCKING = ("STATIC_OBSTACLE", "OTHER_PER_PHOTO_OBJECT")
REQUIRED_BODY_BLOCKING = ("STATIC_OBSTACLE", "NO_GO", "UNKNOWN")
LAYER_NAME = {k: v.split(" (")[0] for k, v in ME.CODE_NAMES.items()}


def check_policy(ucfg):
    """Refuse a unified config that would relax the approved policy (it may only be stricter)."""
    g = ucfg["gripper_baseline"]
    missing = [b for b in REQUIRED_BLOCKING if b not in g["blocking_layers"]]
    if missing:
        raise ValueError(f"gripper_baseline.blocking_layers lacks {missing}: relaxing the approved rule is not allowed")
    if g.get("max_reach_cm") is not None:
        raise ValueError("gripper reach is not measured: gripper_baseline.max_reach_cm must stay null (no invented limit)")
    missing = [b for b in REQUIRED_BODY_BLOCKING if b not in ucfg["body_blocking_codes"]]
    if missing:
        raise ValueError(f"body_blocking_codes lacks {missing}: relaxing the approved rule is not allowed")
    if not float(ucfg["margin_cm"]) >= 0.0 or not float(g["step_cm"]) > 0.0:
        raise ValueError("margin_cm must be >= 0 and gripper_baseline.step_cm > 0")
    return True


def facing(yaw_map_deg):
    """Body heading (gripper side) as a floor-cm direction (floor y points towards the camera, so map yaw turns the other way)."""
    a = math.radians(yaw_map_deg)
    return np.array([math.cos(a), -math.sin(a)])


def body_front(centre_floor, yaw_map_deg, length_cm):
    return np.asarray(centre_floor, np.float64) + facing(yaw_map_deg) * length_cm / 2


def obstacle_mask(grid, frame, boxes_px, px_per_cm):
    """Grid cells touched by the image boxes projected at z = 0 (same raster as path_planning_batch_step4b.obstacle_cells)."""
    sub = int(round(grid.res_cm * px_per_cm))
    m = np.zeros((grid.rows * sub, grid.cols * sub), np.uint8)
    for b in boxes_px:
        corners = frame.img_to_floor([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
        cv2.fillPoly(m, [np.round((corners - [grid.x0, grid.y0]) * px_per_cm).astype(np.int32)], 1)
    return m.reshape(grid.rows, sub, grid.cols, sub).any(axis=(1, 3))


def target_masks(grid, frame, boxes_px, target_px, px_per_cm):
    """(own pallet mask, other objects mask, own index). Own = the first box containing the target image point; none -> all boxes are other."""
    own_i = None
    if target_px is not None:
        own_i = next((i for i, b in enumerate(boxes_px) if b[0] <= target_px[0] <= b[2] and b[1] <= target_px[1] <= b[3]), None)
    own = obstacle_mask(grid, frame, [boxes_px[own_i]] if own_i is not None else [], px_per_cm)
    other = obstacle_mask(grid, frame, [b for i, b in enumerate(boxes_px) if i != own_i], px_per_cm)
    return own, other, own_i


def build_world(pgm, layers, values, grid, frame, pcfg, body_blocking_codes, wheel_envelope_cm, nominal, obstacles):
    """Wheel-boundary DualMap with the per-photo obstacle cells in BOTH maps (= path_planning_batch_step4b.dual_map).
    An all-False obstacle mask gives the model of path_planning_wheel_boundary.main."""
    code_of = {v: k for k, v in LAYER_NAME.items()}
    pgm_w = pgm.copy()
    pgm_w[obstacles] = values["occupied"]
    body_ok = ~np.isin(layers, [code_of[c] for c in body_blocking_codes]) & ~obstacles
    pm_w = PP.PgmMap(pgm_w, layers, values, grid, frame, pcfg["raster_px_per_cm"], pcfg["safety_margin_cm"])
    pm_b = PP.PgmMap(np.where(body_ok, values["free"], values["occupied"]).astype(np.uint8), layers, values, grid, frame,
                     pcfg["raster_px_per_cm"], pcfg["safety_margin_cm"])
    return WB.DualMap(pm_w, pm_b, wheel_envelope_cm, nominal)


def gripper_baseline_2d(world, centre_floor, yaw_map_deg, target_floor, own, other, nominal, gcfg, zone_of=None):
    """Floor-level centre line from the nominal body front to the target, split into runs by layer. Blocking = gcfg blocking_layers.
    zone_of(floor_cm) -> zone id (optional, names only). 2D only: no reach limit, no width / height / 3D check."""
    front = body_front(centre_floor, yaw_map_deg, nominal[0])
    target = np.asarray(target_floor, np.float64)
    n = max(1, int(math.ceil(np.linalg.norm(target - front) / gcfg["step_cm"])))
    runs = []
    for k in range(n + 1):
        p = front + (target - front) * k / n
        r, c = world.cell_of_floor(p)
        if not world.in_grid(r, c):
            name = "OUTSIDE_MAP"
        elif other[r, c]:
            name = "OTHER_PER_PHOTO_OBJECT"
        elif world.layers[r, c] == ME.STATIC_OBSTACLE:
            name = "STATIC_OBSTACLE"
        elif own[r, c]:
            name = "TARGET_PALLET"
        else:
            name = LAYER_NAME[int(world.layers[r, c])]
        if runs and runs[-1]["layer"] == name:
            runs[-1]["to_floor_cm"] = p.round(2).tolist()
        else:
            runs.append({"layer": name, "zones": [], "from_floor_cm": p.round(2).tolist(), "to_floor_cm": p.round(2).tolist()})
        zone = zone_of(p) if zone_of is not None and name in ("STATIC_OBSTACLE", "NO_GO") else None
        if zone and zone not in runs[-1]["zones"]:
            runs[-1]["zones"].append(zone)
    for r_ in runs:
        r_["length_cm"] = round(float(np.linalg.norm(np.subtract(r_["to_floor_cm"], r_["from_floor_cm"]))), 1)
        r_["from_map_m"], r_["to_map_m"] = [world.frame.floor_to_map([q])[0].round(3).tolist() for q in (r_["from_floor_cm"], r_["to_floor_cm"])]
        r_["blocking"] = r_["layer"] in gcfg["blocking_layers"]
    blocked_by = list(dict.fromkeys(r_["layer"] for r_ in runs if r_["blocking"]))
    return {"check": "2D_FLOOR_BASELINE", "front_floor_cm": front.round(2).tolist(), "front_map_m": world.frame.floor_to_map([front])[0].round(3).tolist(),
            "target_map_m": world.frame.floor_to_map([target])[0].round(4).tolist(), "step_cm": gcfg["step_cm"], "runs": runs,
            "blocked": bool(blocked_by), "blocked_by": blocked_by, "max_reach_cm": None, "gripper_3d": GRIPPER_3D}


def select_and_plan(world, kind, target_f, start_f, start_yaw, nominal, margin, pcfg, gripper, allow_escape, own, other, gcfg, zone_of=None):
    """path_planning.select_and_plan with the gripper baseline rule applied BEFORE the legs are planned (as the batch script)."""
    cands = PP.approach_candidates(world, target_f, nominal, margin, pcfg["approach_yaws_map_deg"], pcfg["approach_search_cm"], gripper)
    for c in cands:
        if c["status"] != "POSE_FOUND":
            continue
        gb = gripper_baseline_2d(world, c["centre_floor_cm"], c["yaw_map_deg"], target_f, own, other, nominal, gcfg, zone_of)
        c["gripper_baseline_2d"] = gb
        if gb["blocked"]:
            c["status"] = "GRIPPER_LINE_BLOCKED"
            continue
        best, tried = PP.plan_leg(world, kind, start_f, start_yaw, tuple(c["cell"]), c["yaw_map_deg"], nominal, margin, pcfg, allow_escape)
        c["leg_attempts"] = [{k: v for k, v in t.items() if k != "raw_cells"} for t in tried]
        c["leg"] = best
        if best is None:
            c["status"] = "NO_FEASIBLE_LEG"
            continue
        c["leg_length_cm"] = round((len(best["raw_cells"]) - 1) * world.grid.res_cm, 1)
        c["status"] = "FEASIBLE_PLAN_GRASP_UNVERIFIED"
    feas = [c for c in cands if c["status"] == "FEASIBLE_PLAN_GRASP_UNVERIFIED"]
    return cands, (min(feas, key=lambda c: (c["front_to_target_gap_cm"], c["leg_length_cm"])) if feas else None)


def evaluate_leg(world, metrics, leg, nominal, margin, pcfg, metrics_step_cm):
    """Unchanged describe_leg checks + tolerant forward/lateral labels + nominal footprint metrics; status VALID / INVALID_PATH."""
    d = WB.relabel_body_vs_move(PP.describe_leg(world, leg, nominal, margin, pcfg))
    d["status"] = "VALID" if d["collision_checks_passed"] else "INVALID_PATH"
    d["nominal_footprint_metrics"] = metrics.path(d, metrics_step_cm)
    return d


def cause_of(cands, kind):
    if not cands:
        return f"{kind}: no target"
    st = {c["yaw_map_deg"]: c["status"] for c in cands}
    if all(s == "NO_COLLISION_FREE_POSE_ON_AXIS" for s in st.values()):
        return f"{kind}: no collision-free stop on the target axis for any of 0/90/180/270 (wheel boundary / body / per-photo obstacles)"
    blocked = {c["yaw_map_deg"]: c["gripper_baseline_2d"]["blocked_by"] for c in cands if c["status"] == "GRIPPER_LINE_BLOCKED"}
    if blocked and all(s in ("NO_COLLISION_FREE_POSE_ON_AXIS", "GRIPPER_LINE_BLOCKED") for s in st.values()):
        return f"{kind}: the only collision-free stops on the target axis need the gripper baseline to cross {blocked} -> rejected"
    legs = {c["yaw_map_deg"]: [(a["strategy"], a["status"]) for a in c.get("leg_attempts", [])] for c in cands if c["status"] == "NO_FEASIBLE_LEG"}
    return f"{kind}: stop pose found but no feasible leg {legs}" if legs else f"{kind}: {st}"


def plan_task(world, start_f, start_yaw, a_f, b_f, masks, nominal, margin, pcfg, gripper, gcfg, metrics_step_cm, zone_of=None):
    """Path A (robot start -> grasp stop) and Path B (grasp stop -> placement stop). masks = {"A": (own, other), "B": (own, other)}.
    A not VALID -> B is DIAGNOSTIC only (CONDITIONAL). b_f None -> B REVIEW_REQUIRED (no target is invented)."""
    mt = WB.Metrics(world)
    gc, gsel = select_and_plan(world, "A", a_f, start_f, start_yaw, nominal, margin, pcfg, gripper, True, *masks["A"], gcfg, zone_of)
    A = {"candidates": gc, "selected": gsel, "desc": None, "cause": None, "reference": None}
    if gsel:
        A["desc"] = evaluate_leg(world, mt, gsel["leg"], nominal, margin, pcfg, metrics_step_cm)
        A["status"] = A["desc"]["status"]
    else:
        A["status"], A["cause"] = "NOT_FOUND", cause_of(gc, "A")
        A["reference"] = WB.nearest_reference_pose(world, a_f, nominal, margin, pcfg["approach_yaws_map_deg"])
    B = {"candidates": [], "selected": None, "desc": None, "cause": None, "reference": None, "basis": None, "role": None}
    if b_f is None:
        B["status"], B["basis"] = "REVIEW_REQUIRED", "no placement target: it must be given by the user (none invented)"
        return {"A": A, "B": B, "metrics": mt}
    if A["status"] == "VALID":
        b_start, b_yaw, esc, B["basis"], B["role"] = np.array(gsel["centre_floor_cm"]), gsel["yaw_map_deg"], False, "grasp stop of Path A", "TASK_SEQUENCE"
    else:
        B["role"] = "DIAGNOSTIC_ONLY"
        cand = [c for c in gc if c.get("centre_floor_cm") is not None and c["status"] != "GRIPPER_LINE_BLOCKED"]
        if cand:
            c0 = min(cand, key=lambda c: c["front_to_target_gap_cm"])
            b_start, b_yaw, esc = np.array(c0["centre_floor_cm"]), c0["yaw_map_deg"], False
            B["basis"] = f"CONDITIONAL: grasp-pose candidate yaw {c0['yaw_map_deg']} {c0['pose_map']} (Path A not valid)"
        else:
            b_start, b_yaw, esc = np.asarray(start_f, np.float64), start_yaw, True
            B["basis"] = "CONDITIONAL: from the robot start (no grasp-pose candidate); not a task sequence"
    pc, psel = select_and_plan(world, "B", b_f, b_start, b_yaw, nominal, margin, pcfg, gripper, esc, *masks["B"], gcfg, zone_of)
    B["candidates"], B["selected"] = pc, psel
    if psel:
        B["desc"] = evaluate_leg(world, mt, psel["leg"], nominal, margin, pcfg, metrics_step_cm)
        B["status"] = B["desc"]["status"] if B["desc"]["status"] == "INVALID_PATH" or A["status"] == "VALID" else "CONDITIONAL"
    else:
        B["status"], B["cause"] = "NOT_FOUND", cause_of(pc, "B")
        B["reference"] = WB.nearest_reference_pose(world, b_f, nominal, margin, pcfg["approach_yaws_map_deg"])
    return {"A": A, "B": B, "metrics": mt}
