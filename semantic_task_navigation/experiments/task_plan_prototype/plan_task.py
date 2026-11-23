#!/usr/bin/env python3
"""Image-based task-plan prototype on the E1 photo (no model call, no GPU, no hardware).

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
  $PY experiments/task_plan_prototype/plan_task.py

instruction (stored Qwen result) -> PROVISIONAL_TARGET (stored E1 cube candidate, Qwen crop gate said "no", kept) -> stored robot start ->
PROVISIONAL floor transform (mat seams) -> BEV map (stored DRAFT zones as MANUAL_MAP_PRIOR) -> path A (start -> pick-up approach) -> VIRTUAL pick-up ->
path B (pick-up approach -> loading-bay approach) -> VIRTUAL placement -> figures + plan.json.

Reused unchanged: v0.2 NavMap (subclassed for the footprint), select_goal, plan (A*), validate_path, handoff.make_request/TaskExecution, the stored loaders of
first_vlm_navigation_path. No grasp, placement or driving success is claimed; every physical quantity that is not measured is listed as UNVERIFIED.
"""
import copy
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "first_vlm_navigation_path"))
import floor_transform as FT  # noqa: E402
import footprint_nav as FN  # noqa: E402
import run_first_path as FP  # noqa: E402  (stored loaders of the first path experiment; read only)
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file  # noqa: E402
from semantic_task_nav.navigation.goal_selector import _distance_to_polygon, select_goal  # noqa: E402
from semantic_task_nav.navigation.path_validator import validate_path  # noqa: E402
from semantic_task_nav.navigation.planner import plan  # noqa: E402
from semantic_task_nav.robot.handoff import TaskExecution, make_request  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
BEV_FRAME, FLOOR_FRAME, IMG_FRAME = "bev_px", "floor_cm (PROVISIONAL)", "undistorted_image_px (E1; v0.2 labels it raw_image_px)"
REQUIRED_INFO = [
    {"item": "robot outer dimensions incl. wheels and the real heading of the robot at the start", "why": "35x24 cm is the user's initial value; the verified outline maps to ~46x32 cm on the floor (parallax and/or wheels)"},
    {"item": "gripper reach, grasp pose, cube size and the height of the conveyor surface", "why": "the approach points leave a gap the gripper must bridge; the target's floor projection is displaced by its height"},
    {"item": "permission/geometry for the gripper to cross the drivable boundary and the placement surface of the bay", "why": "the bay and the cube both lie outside the drivable zone Z02"},
    {"item": "human-confirmed floor reference points (and a re-check of the intrinsics for the current installation)", "why": "the floor transform is PROVISIONAL; mat edges are curved near the border (residual lens distortion)"},
    {"item": "human confirmation of the DRAFT zone polygons (edge accuracy at the cm level)", "why": "the footprint margin covers the transform error only, not an error of the hand-drawn polygon edges"},
    {"item": "mecanum dynamics (acceleration, corner behaviour) and obstacle overhang beyond the floor outlines", "why": "only the static whole-body footprint at a fixed heading is checked"}]


def load_cfg():
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ stored inputs
def pick_target(ti, tcfg):
    """Label-free rule from the config: rank 1 of the stored merged list with >= min_distinct_sources source types."""
    c = ti["candidates"][0] if ti["candidates"] else None
    if c is None or len(set(c["sources"])) < tcfg["min_distinct_sources"]:
        return None
    return c


def pose_list_from_stored(first):
    s3 = first["stages"]["3_robot_start"]
    if s3["status"] != "ROBOT_DETECTED":
        return None, None
    return s3["robot"], {s["shift_px"]: s["xy"] for s in s3["start_candidates"]}


# ------------------------------------------------------------------------------------------------ BEV spec
class Frames:
    """image px <-> floor cm <-> BEV px for the PROVISIONAL transform."""

    def __init__(self, H_img_to_floor, px_per_cm, pad_cm, size_cm):
        self.H, self.Hinv = np.asarray(H_img_to_floor, np.float64), np.linalg.inv(np.asarray(H_img_to_floor, np.float64))
        self.s, self.pad, self.size_cm = px_per_cm, pad_cm, size_cm
        self.bev_size = (round((size_cm[0] + 2 * pad_cm) * px_per_cm), round((size_cm[1] + 2 * pad_cm) * px_per_cm))

    def img_to_cm(self, pts):
        return FT.project(self.H, pts)

    def cm_to_img(self, pts):
        return FT.project(self.Hinv, pts)

    def cm_to_bev(self, pts):
        return (np.asarray(pts, np.float64).reshape(-1, 2) + self.pad) * self.s

    def bev_to_cm(self, pts):
        return np.asarray(pts, np.float64).reshape(-1, 2) / self.s - self.pad


def box_poly(b):
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]


def build_specs(zmap, fr):
    """DRAFT zones (image px) -> cm polygons -> BEV px polygons. Anything not inside a traversable zone is NOT drivable."""
    cm, bev, clipped = {"traversable": [], "static_obstacles": [], "no_go": []}, {"traversable": [], "static_obstacles": [], "no_go": []}, []
    for layer in cm:
        for z in zmap[layer]:
            pc = fr.img_to_cm(z["polygon"])
            cm[layer].append({"id": z["id"], "polygon": pc.tolist()})
            bev[layer].append({"id": z["id"], "polygon": fr.cm_to_bev(pc).tolist()})
            if layer == "traversable" and (pc.min() < -fr.pad or pc[:, 0].max() > fr.size_cm[0] + fr.pad or pc[:, 1].max() > fr.size_cm[1] + fr.pad):
                clipped.append(z["id"])
    spec_bev = {**bev, "image_size": list(fr.bev_size), "coordinate_frame": BEV_FRAME, "review_status": zmap["review_status"], "start_pose": None}
    return cm, spec_bev, clipped


# ------------------------------------------------------------------------------------------------ planning
def v02_pcfg(v02, bcfg, acfg):
    p = copy.deepcopy(v02["planning"])
    p.update({"cell_px": bcfg["cell_px"], "safety_margin_px": bcfg["safety_margin_px"], "validation_tolerance_px": bcfg["validation_tolerance_px"],
              "validation_sample_step_px": 1.0})
    for k in ("pickup", "dropoff"):
        p["goal_selection"][k].update({"max_dist_px": None, "distance_weight": acfg["distance_weight"]})   # reach is NOT assumed (see approach_goal)
    return p


def approach_goal(nav, anchor_id, poly_bev, start_xy, purpose, pcfg, slack_px, s, pad_cm):
    """v0.2 select_goal, twice: (1) all reachable positions -> the nearest one defines the distance the approach has to bridge;
    (2) lowest score within `slack` of that distance. The gripper reach is NOT assumed: the distance to bridge is reported instead."""
    key = "pickup" if purpose == "pickup_approach" else "dropoff"
    g1 = select_goal(nav, anchor_id, poly_bev, start_xy, purpose, {"max_dist_px": 1e9, "min_dist_px": 0, "distance_weight": pcfg["goal_selection"][key]["distance_weight"], "top_k_report": 1}, pcfg)
    if g1["status"] != "GOAL_SELECTED":
        return _plain_goal(g1, s, pad_cm)
    dist = _distance_to_polygon(nav, poly_bev)
    cells = np.array(g1["accessible_cells"])
    d_near = float(dist[cells[:, 0], cells[:, 1]].min())
    g2 = select_goal(nav, anchor_id, poly_bev, start_xy, purpose, {"max_dist_px": d_near + slack_px, "min_dist_px": 0, "distance_weight": pcfg["goal_selection"][key]["distance_weight"], "top_k_report": 5}, pcfg)
    out = _plain_goal(g2, s, pad_cm)
    out["nearest_reachable_centre_to_target_cm"] = round(d_near / s, 1)
    out["n_reachable_collision_free_positions"] = len(cells)
    return out


def _plain_goal(g, s, pad_cm):
    """Drop the v0.2 fields that would mislead here (reach basis, raw_image_px labels, huge cell lists) and add cm."""
    out = {"status": g["status"], "reason": g.get("reason"), "purpose": g["purpose"], "anchor": g["anchor"], "stats": g.get("stats"),
           "reach": "NOT_A_REACH_CLAIM: gripper reach is UNVERIFIED; the distance the approach has to bridge is reported as body_to_target_gap_cm",
           "xy_bev_px": None, "xy_cm": None, "frame_note": "v0.2 select_goal labels its output raw_image_px; the values are BEV px"}
    if g.get("pose"):
        x, y = g["pose"]["xy"]
        out.update({"xy_bev_px": [x, y], "xy_cm": [round(x / s - pad_cm, 1), round(y / s - pad_cm, 1)], "path_cost": g["pose"]["path_cost"],
                    "facing_direction_deg": g["pose"]["heading_deg"], "facing_note": "direction from the approach point towards the target centroid (x right, y down)"})
        out["candidates_top_k"] = [{"xy_cm": [round(c["xy"][0] / s - pad_cm, 1), round(c["xy"][1] / s - pad_cm, 1)], "distance_to_target_cm": round(c["distance_to_anchor_px"] / s, 1),
                                   "path_cost_px": c["path_cost"]} for c in g["candidates_top_k"]]
    return out


def make_leg(nav, extent_cm, heading, goal, start_bev, pcfg, free_hi, fr, bcfg, label):
    """Path (v0.2 A*) + v0.2 validator + independent whole-body sweep for one leg and one heading."""
    s, pad = fr.s, fr.pad
    rec = {"heading_deg": heading, "footprint_extent_xy_cm": [round(extent_cm[0], 1), round(extent_cm[1], 1)], "goal": goal, "path": None,
           "v02_validation": None, "sweep_check": None, "feasible": False, "failure": None}
    if goal["status"] != "GOAL_SELECTED":
        rec["failure"] = {"stage": "approach_point", "code": goal["status"], "reason": goal.get("reason")}
        return rec, None
    raw = plan(nav, start_bev, goal["xy_bev_px"], pcfg, label)
    if raw["status"] != "PATH_FOUND":
        rec["failure"] = {"stage": "path", "code": raw["status"], "reason": raw.get("error")}
        return rec, None
    wp_cm = fr.bev_to_cm(raw["waypoints_xy"])
    path = {"status": "PATH_FOUND", "frame_note": "v0.2 plan() labels its record raw_image_px; waypoints_bev_px are BEV px", "waypoints_bev_px": raw["waypoints_xy"],
            "waypoints_cm": [[round(x, 1), round(y, 1)] for x, y in wp_cm], "waypoints_image_px": [[round(x, 1), round(y, 1)] for x, y in fr.cm_to_img(wp_cm)], "n_waypoints": raw["n_waypoints"],
            "length_cm": round(raw["length_px"] / s, 1), "raw_grid_length_cm": round(raw["raw_length_px"] / s, 1), "astar_cost": raw["astar_cost"], "planner": raw["planner"]}
    rec["path"] = path
    if raw["n_waypoints"] < 2:                      # start == goal: v0.2 validate_path needs >= 2 waypoints; reported, not treated as a generated path
        rec["failure"] = {"stage": "path", "code": "ZERO_LENGTH_PATH", "reason": "start and goal are the same cell: no path is generated for this leg"}
        return rec, raw
    v = validate_path(nav, raw["waypoints_xy"], raw["start_xy"], raw["goal_xy"], pcfg)
    rec["v02_validation"] = {"valid": v["valid"], "min_clearance_px": v["min_clearance_px"], "required_px": v["required_clearance_px"],
                             "note": "v0.2 validator on the INSCRIBED disc of the footprint: a necessary check only"}
    sw = FN.sweep_check(free_hi, wp_cm.tolist(), extent_cm, pad, bcfg["check_px_per_cm"], bcfg["check_step_cm"])
    rec["sweep_check"] = sw
    rec["feasible"] = bool(v["valid"] and sw["ok"])
    if not rec["feasible"]:
        rec["failure"] = {"stage": "validation", "code": "PATH_FAILED_FOOTPRINT_CHECK", "reason": f"v0.2 validator {v['valid']}, whole-body sweep {sw['ok']}"}
    return rec, raw


def plan_variant(label, base_L, base_W, margin, start_img, ctx):
    fr, cfg, pcfg, bcfg = ctx["fr"], ctx["cfg"], ctx["pcfg"], ctx["cfg"]["bev"]
    s, pad, size_cm = fr.s, fr.pad, fr.size_cm
    free_hi = ctx["free_hi"]
    start_cm = fr.img_to_cm([start_img])[0]
    start_bev = fr.cm_to_bev([start_cm])[0]
    rec = {"label": label, "nominal_size_cm": [round(base_L, 1), round(base_W, 1)], "margin_per_side_cm": margin, "start_image_px": [round(v, 1) for v in start_img],
           "start_cm": [round(float(start_cm[0]), 1), round(float(start_cm[1]), 1)], "start_status": "ESTIMATED (outline centre moved towards the principal point; heading UNKNOWN)",
           "legs_A": {}, "legs_B": {}, "selected": None}
    navs = {}
    for h in cfg["robot"]["headings_deg"]:
        ext = FN.footprint_dims_cm(base_L, base_W, margin, h)
        navs[h] = FN.RectNavMap(ctx["spec_bev"], fr.bev_size, pcfg, ext, s)
    runtime = {"navs": navs, "A": {}, "B": {}}
    slack = cfg["approach"]["slack_cm"] * s
    # ---- leg A
    for h, nav in navs.items():
        g = approach_goal(nav, "cube_candidate", ctx["target_bev"], list(start_bev), "pickup_approach", pcfg, slack, s, pad)
        leg, raw = make_leg(nav, nav.extent_cm, h, g, list(start_bev), pcfg, free_hi, fr, bcfg, "A_start_to_pickup_approach")
        if g["status"] == "GOAL_SELECTED":
            g["body_to_target_gap_cm"] = round(FN.body_gap_cm(ctx["target_cm"], g["xy_cm"], footprint_nominal(base_L, base_W, h), size_cm, pad, 8), 1)
            g["xy_image_px"] = [round(v, 1) for v in fr.cm_to_img([g["xy_cm"]])[0]]
            g["rotation_in_place_at_approach"] = nav.rotation_clear(g["xy_bev_px"])
        leg["start_heading_alignment"] = {"note": "actual start heading UNKNOWN; rotation to the planning heading checked with the circumscribed disc (information only)", **nav.rotation_clear(start_bev)}
        rec["legs_A"][h] = leg
        runtime["A"][h] = raw
    feas_A = [(h, l) for h, l in rec["legs_A"].items() if l["feasible"]]
    if not feas_A:
        rec["selected"] = {"status": "NO_PATH_A", "reason": {h: l["failure"] for h, l in rec["legs_A"].items()}}
        return rec, runtime
    hA, legA = min(feas_A, key=lambda t: (t[1]["goal"]["body_to_target_gap_cm"], t[1]["path"]["length_cm"]))   # smallest gap to bridge first, then the shortest path
    gA_bev = rec["legs_A"][hA]["goal"]["xy_bev_px"]
    # ---- leg B starts at the pick-up approach point; a heading change there needs the circumscribed disc free (conservative)
    for h, nav in navs.items():
        g = approach_goal(nav, "loading_bay", ctx["bay_bev"], list(gA_bev), "dropoff_approach", pcfg, slack, s, pad)
        leg, raw = make_leg(nav, nav.extent_cm, h, g, list(gA_bev), pcfg, free_hi, fr, bcfg, "B_pickup_approach_to_bay_approach")
        leg["heading_change_from_A"] = h != hA
        if g["status"] == "GOAL_SELECTED":
            g["body_to_target_gap_cm"] = round(FN.body_gap_cm(ctx["bay_cm"], g["xy_cm"], footprint_nominal(base_L, base_W, h), size_cm, pad, 8), 1)
            g["xy_image_px"] = [round(v, 1) for v in fr.cm_to_img([g["xy_cm"]])[0]]
        if h != hA:
            rot = navs[hA].rotation_clear(gA_bev)
            leg["rotation_in_place_at_pickup_approach"] = rot
            if leg["feasible"] and not rot["ok"]:
                leg["feasible"] = False
                leg["failure"] = {"stage": "heading_change", "code": "ROTATION_CLEARANCE_NOT_MET",
                                  "reason": f"circumscribed disc needs {rot['needed_px'] / s:.1f} cm clearance at the pick-up approach, only {rot['clearance_px'] / s:.1f} cm"}
        rec["legs_B"][h] = leg
        runtime["B"][h] = raw
    feas_B = [(h, l) for h, l in rec["legs_B"].items() if l["feasible"]]
    if not feas_B:
        rec["selected"] = {"status": "PATH_A_ONLY", "A_heading_deg": hA, "reason": {h: l["failure"] for h, l in rec["legs_B"].items()}}
        return rec, runtime
    hB, legB = min(feas_B, key=lambda t: (t[1]["goal"]["body_to_target_gap_cm"], t[1]["path"]["length_cm"]))
    rec["selected"] = {"status": "PATH_A_AND_B", "A_heading_deg": hA, "B_heading_deg": hB, "total_length_cm": round(legA["path"]["length_cm"] + legB["path"]["length_cm"], 1),
                       "rule": "per leg: among the feasible headings (v0.2 A*, v0.2 validator and whole-body sweep all pass) the smallest body-to-target gap first (the gap is what the unverified gripper must bridge), then the shortest path; B starts at the pick-up approach point"}
    return rec, runtime


def footprint_nominal(L, W, h):
    return FN.footprint_dims_cm(L, W, 0, h)


# ------------------------------------------------------------------------------------------------ drawing
def _t(img, text, org, color, scale=0.5, bg=True):
    text = str(text).encode("ascii", "replace").decode()
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(max(0, min(org[0], img.shape[1] - tw - 2))), int(max(th + 3, min(org[1], img.shape[0] - bl - 2)))
    if bg:
        cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + bl), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


class Mapper:
    def __init__(self, kind, fr, scale=3):
        self.kind, self.fr, self.scale = kind, fr, scale

    def __call__(self, cm):
        cm = np.asarray(cm, np.float64).reshape(-1, 2)
        p = self.fr.cm_to_img(cm) if self.kind == "photo" else (cm + self.fr.pad) * self.scale
        return np.round(p).astype(np.int32)


def _fill(img, m, polys_cm, color, alpha):
    layer = img.copy()
    for p in polys_cm:
        cv2.fillPoly(layer, [m(p)], color)
    img[:] = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)


def _outline(img, m, poly_cm, color, thick=2, dashed=False):
    p = m(poly_cm)
    if not dashed:
        cv2.polylines(img, [p], True, color, thick, cv2.LINE_AA)
        return
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        n = max(1, int(np.linalg.norm(b - a) // 14))
        for i in range(0, n, 2):
            cv2.line(img, tuple(a + (b - a) * i // n), tuple(a + (b - a) * min(i + 1, n) // n), color, thick, cv2.LINE_AA)


def _rect(img, m, c, ext, color, thick=2, fill_alpha=0.0):
    pts = [[c[0] - ext[0] / 2, c[1] - ext[1] / 2], [c[0] + ext[0] / 2, c[1] - ext[1] / 2], [c[0] + ext[0] / 2, c[1] + ext[1] / 2], [c[0] - ext[0] / 2, c[1] + ext[1] / 2]]
    if fill_alpha:
        _fill(img, m, [pts], color, fill_alpha)
    _outline(img, m, pts, color, thick)


def draw_plan(img, m, ctx, rec, rt, scale_txt):
    C = {"free": (80, 200, 80), "obs": (60, 60, 235), "nogo": (200, 60, 200), "bay": (230, 200, 40), "qwen": (0, 165, 255), "A": (255, 160, 40), "B": (40, 150, 255),
         "cube": (0, 0, 255), "robot": (255, 255, 255), "goal": (0, 255, 255)}
    spec = ctx["spec_cm"]
    _fill(img, m, [z["polygon"] for z in spec["traversable"]], C["free"], 0.22)
    _fill(img, m, [z["polygon"] for z in spec["static_obstacles"]], C["obs"], 0.45)
    _fill(img, m, [z["polygon"] for z in spec["no_go"]], C["nogo"], 0.30)
    for z in spec["traversable"] + spec["static_obstacles"] + spec["no_go"]:
        _outline(img, m, z["polygon"], (255, 255, 255), 1)
        _t(img, z["id"], m(np.mean(z["polygon"], axis=0))[0], (255, 255, 255), 0.45)
    _fill(img, m, [ctx["bay_cm"]], C["bay"], 0.35)
    _outline(img, m, ctx["bay_cm"], C["bay"], 2)
    _t(img, "semantic destination: loading bay (stored, Python-localised)", m(np.mean(ctx["bay_cm"], axis=0))[0] + np.array([-120, -26]), C["bay"], 0.45)
    _outline(img, m, ctx["qwen_cm"], C["qwen"], 2, dashed=True)
    _outline(img, m, ctx["target_cm"], C["cube"], 3)
    _t(img, "PROVISIONAL_TARGET (cube cand., Qwen gate: no)", m(ctx["target_cm"][0])[0] + np.array([8, -22]), C["cube"], 0.45)
    if rec is None:
        return
    st = rec["start_cm"]
    sel = rec.get("selected") or {}
    nominal = rec["nominal_size_cm"]
    hA, hB = sel.get("A_heading_deg"), sel.get("B_heading_deg")
    if hA is None and rec["legs_A"]:
        hA = next(iter(rec["legs_A"]))
    ext0 = FN.footprint_dims_cm(nominal[0], nominal[1], 0, hA if hA is not None else 0)
    extm = FN.footprint_dims_cm(nominal[0], nominal[1], rec["margin_per_side_cm"], hA if hA is not None else 0)
    _rect(img, m, st, ext0, C["robot"], 2)
    _rect(img, m, st, extm, C["robot"], 1)
    cv2.drawMarker(img, tuple(m(st)[0]), C["robot"], cv2.MARKER_CROSS, 14, 2)
    _t(img, "robot start (ESTIMATED)", m(st)[0] + np.array([-70, 30]), C["robot"], 0.45)
    for tag, legs, hsel, col in (("A", rec["legs_A"], hA, C["A"]), ("B", rec["legs_B"], hB, C["B"])):
        leg = legs.get(hsel) if hsel is not None else None
        if leg is None or not leg["feasible"]:
            continue
        wp = leg["path"]["waypoints_cm"]
        cv2.polylines(img, [m(wp)], False, col, 3, cv2.LINE_AA)
        for i, p in enumerate(wp):
            cv2.circle(img, tuple(m(p)[0]), 5, col, -1, cv2.LINE_AA)
            _t(img, f"{tag}{i}", m(p)[0] + np.array([6, -6]), col, 0.42)
        g = leg["goal"]
        ext = FN.footprint_dims_cm(nominal[0], nominal[1], rec["margin_per_side_cm"], leg["heading_deg"])
        _rect(img, m, g["xy_cm"], ext, C["goal"], 2, 0.25)
        _t(img, ("pick-up approach (virtual pick-up)" if tag == "A" else "bay approach (virtual placement)") + f" gap {g['body_to_target_gap_cm']}cm", m(g["xy_cm"])[0] + np.array([10, 24 if tag == "A" else -14]), C["goal"], 0.45)
        for p in wp[1:-1]:                              # whole body at every waypoint
            _rect(img, m, p, ext, col, 1)
    _t(img, scale_txt, (10, 24), (255, 255, 255), 0.55)


def render(ctx, rec, rt, out):
    fr, cfg = ctx["fr"], ctx["cfg"]
    sel = rec["selected"]
    head = [f"TASK PLAN PROTOTYPE - PLAN ONLY, NOTHING EXECUTED - variant {rec['label']}",
            f"floor transform PROVISIONAL_UNVALIDATED (fit RMS {ctx['ft']['fit_rms_cm']} cm, seam dev <= {ctx['ft']['seam_max_abs_dev_cm']} cm); target PROVISIONAL_TARGET (stored Qwen gate said NO)",
            f"result: {sel['status']}" + (f"  A {rec['legs_A'][sel['A_heading_deg']]['path']['length_cm']} cm" if sel.get('A_heading_deg') is not None else "")
            + (f"  B {rec['legs_B'][sel['B_heading_deg']]['path']['length_cm']} cm" if sel.get('B_heading_deg') is not None else "")
            + "   gripper reach / grasp / placement: UNVERIFIED, success NOT claimed"]
    figs = []
    # BEV
    S = 3
    Wc, Hc = ctx["size_cm"][0] + 2 * fr.pad, ctx["size_cm"][1] + 2 * fr.pad
    M = np.array([[S, 0, fr.pad * S], [0, S, fr.pad * S], [0, 0, 1]]) @ fr.H
    bev = cv2.warpPerspective(ctx["img"], M, (round(Wc * S), round(Hc * S)))
    bev = (bev * 0.45).astype(np.uint8)
    mb = Mapper("bev", fr, S)
    for i in range(cfg["floor"]["mats_cols"] + 1):                      # mat grid lines (the reference geometry)
        cv2.line(bev, tuple(mb([[i * cfg["floor"]["mat_cm"], 0]])[0]), tuple(mb([[i * cfg["floor"]["mat_cm"], ctx["size_cm"][1]]])[0]), (200, 200, 200), 1)
    for j in range(cfg["floor"]["mats_rows"] + 1):
        cv2.line(bev, tuple(mb([[0, j * cfg["floor"]["mat_cm"]]])[0]), tuple(mb([[ctx["size_cm"][0], j * cfg["floor"]["mat_cm"]]])[0]), (200, 200, 200), 1)
    draw_plan(bev, mb, ctx, rec, rt, "BEV (floor plane, PROVISIONAL): 3x2 mats of 105 cm; photo content above the floor is smeared by perspective")
    panel = np.zeros((bev.shape[0], 900, 3), np.uint8)
    y = 24
    for line in head + [""] + ctx["legend"]:
        for part in (textwrap.wrap(line, 98) or [""]):
            _t(panel, part, (8, y), (255, 255, 255), 0.45, bg=False)
            y += 19
    cv2.imwrite(str(out / "01_task_plan_bev.png"), np.hstack([bev, panel]))
    figs.append("01_task_plan_bev.png")
    # photo
    ph = ctx["img"].copy()
    draw_plan(ph, Mapper("photo", fr), ctx, rec, rt, "Photo overlay (floor-level geometry projected back with the PROVISIONAL transform; objects above the floor are not floor footprints)")
    for i, line in enumerate(head):
        _t(ph, line, (10, 52 + 20 * i), (255, 255, 255), 0.5)
    cv2.imwrite(str(out / "02_task_plan_on_photo.png"), ph)
    figs.append("02_task_plan_on_photo.png")
    # floor transform check
    ck = ctx["img"].copy()
    mp = Mapper("photo", fr)
    for i in range(cfg["floor"]["mats_cols"] + 1):
        cv2.polylines(ck, [mp([[i * cfg["floor"]["mat_cm"], j] for j in np.linspace(0, ctx["size_cm"][1], 40)])], False, (0, 255, 255), 1, cv2.LINE_AA)
    for j in range(cfg["floor"]["mats_rows"] + 1):
        cv2.polylines(ck, [mp([[i, j * cfg["floor"]["mat_cm"]] for i in np.linspace(0, ctx["size_cm"][0], 40)])], False, (0, 255, 255), 1, cv2.LINE_AA)
    for n in ctx["ft"]["nodes"]:
        if n["used"]:
            p = tuple(int(round(v)) for v in n["xy_px"])
            cv2.drawMarker(ck, p, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            _t(ck, f"node {n['node']} fit {n['fit_residual_cm']}cm LOO {n['leave_one_out_cm']}cm", (p[0] + 12, p[1] - 10 if p[1] < 900 else p[1] - 40), (0, 255, 255), 0.45)
    _t(ck, "yellow: mat grid predicted by the PROVISIONAL transform from the 8 red nodes. Bottom/left/right edges bow outward (residual lens distortion).", (10, 24), (255, 255, 255), 0.55)
    _t(ck, "independent seam deviation cm (max): " + ", ".join(f"{s['id']} {s['max_abs_dev_cm']}" for s in ctx["ft"]["independent_seam_check"]), (10, 50), (255, 255, 255), 0.55)
    cv2.imwrite(str(out / "03_floor_transform_check.png"), ck)
    figs.append("03_floor_transform_check.png")
    return figs


# ------------------------------------------------------------------------------------------------ main
def main():
    cfg, v02 = load_cfg(), load_config()
    img_path = Path(cfg["image"])
    assert sha256_file(img_path) == cfg["image_sha256"], "image differs from the declared one"
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    H_, W_ = img.shape[:2]
    fp_before = code_fingerprint()["sha256"]
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_task_plan")
    out.mkdir(parents=True, exist_ok=False)

    # 1 stored Qwen result (no new inference)
    fcfg = FP.load_cfg()
    st = FP.load_stored(fcfg, (W_, H_))
    e1 = (HERE / cfg["stored"]["e1_run"]).resolve()
    first_dir = (HERE / cfg["stored"]["first_path_run"]).resolve()
    task_parse = jload(e1 / "analysis/task_parse.json")
    ta, area = st["task_analysis"], st["area"]
    qwen = {"source": "STORED (no new Qwen call)", "run": st["qwen_run"].name, "instruction": cfg["instruction"],
            "task_type": task_parse["task"]["task_type"], "target": task_parse["task"]["target"],
            "target_parse_backend": task_parse["task"]["parser"]["backend"], "actions_source": task_parse["task"]["actions_source"],
            "destination_inference": {"model": "Qwen3-VL (stored L2 reply)", "chosen_area_id": ta["destination_area_id"], "function_name": area["function_name"],
                                      "box_px": [round(v, 1) for v in st["qwen_box_px"]], "observed_evidence": area["observed_evidence"], "reason": ta["reason"],
                                      "decision": ta["decision"], "confidence": area.get("confidence"), "raw_response_sha256_16": st["raw_response_sha256"][:16]},
            "ok": ta["destination_area_id"] == fcfg["stored"]["proposal_area_id"] and ta["decision"] == "proposed"}
    # 2 provisional target (label free)
    ti = jload(e1 / "analysis/target_inspection.json")
    cand = pick_target(ti, cfg["target_candidate"])
    target = None if cand is None else {
        "status": cfg["target_candidate"]["status"], "candidate_id": cand["candidate_id"], "bbox_xyxy_image_px": cand["bbox_xyxy"], "score": cand["score"],
        "sources": cand["sources"], "member_boxes": cand["member_boxes"], "shape_stats": cand["shape_stats"], "selection_rule": cfg["target_candidate"]["rule"],
        "rule_gave_expected_id": cand["candidate_id"] == cfg["target_candidate"]["expected_candidate_id"],
        "stored_qwen_crop_gate": cand["verification"], "v02_decision_for_this_photo": ti["status"],
        "not_used_as": "evidence of detection accuracy or of pick-up success; the Qwen gate answer 'no' is preserved above",
        "height_note": "the cube stands on the conveyor surface, not on the mat floor: its floor-plane projection is displaced by parallax (height UNKNOWN)"}
    inp = out / "inputs"
    inp.mkdir()
    shutil.copy2(img_path, inp / "input_E1_undistorted.png")
    shutil.copy2(e1 / "vlm_proposal/structured.json", inp / "qwen_destination_structured.json")
    shutil.copy2(e1 / "vlm_proposal/raw_response.txt", inp / "qwen_destination_raw_response.txt")
    if cand is not None:
        shutil.copy2(e1 / cand["verification"]["crop"]["path"], inp / "target_candidate_qwen_gate_crop.png")
        (inp / "target_candidate_record.json").write_text(json.dumps(cand, indent=2, ensure_ascii=False), encoding="utf-8")
    # 3 stored robot start
    first = jload(first_dir / "result.json")
    robot, starts = pose_list_from_stored(first)
    # 4 stored DRAFT map (prior)
    zmap = jload(first_dir / "map_E1_transferred.json")
    blocked = [k for k, ok in (("qwen", qwen["ok"]), ("target_candidate", target is not None), ("robot", robot is not None)) if not ok]

    # 5 floor transform
    ft = FT.fit_floor_transform(img, cfg["floor"])
    if not ft["accepted"] or blocked:
        res = {"experiment_id": cfg["experiment_id"], "status": "BLOCKED", "blocked_by": blocked, "floor_transform": {k: v for k, v in ft.items() if k != "nodes"}, "claims": cfg["claims"]}
        (out / "plan.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print("BLOCKED", blocked, ft.get("reason"))
        return out
    size_cm = (cfg["floor"]["mat_cm"] * cfg["floor"]["mats_cols"], cfg["floor"]["mat_cm"] * cfg["floor"]["mats_rows"])
    fr = Frames(ft["H_img_to_floor_cm"], cfg["bev"]["px_per_cm"], cfg["bev"]["pad_cm"], size_cm)
    spec_cm, spec_bev, clipped = build_specs(zmap, fr)
    tgt_cm = fr.img_to_cm(box_poly(target["bbox_xyxy_image_px"]))
    bay_cm = fr.img_to_cm(st["bay_polygon"])
    qwen_cm = fr.img_to_cm(box_poly(qwen["destination_inference"]["box_px"]))
    ro = robot["bbox_xyxy"]
    ro_cm = fr.img_to_cm(box_poly(ro))
    outline_cm = (float(ro_cm[:, 0].max() - ro_cm[:, 0].min()), float(ro_cm[:, 1].max() - ro_cm[:, 1].min()))
    ctx = {"cfg": cfg, "fr": fr, "img": img, "ft": ft, "size_cm": size_cm, "spec_cm": spec_cm, "spec_bev": spec_bev, "pcfg": v02_pcfg(v02, cfg["bev"], cfg["approach"]),
           "free_hi": FN.free_mask_cm(spec_cm, size_cm, fr.pad, cfg["bev"]["check_px_per_cm"]), "target_cm": tgt_cm.tolist(), "target_bev": fr.cm_to_bev(tgt_cm).tolist(),
           "bay_cm": bay_cm.tolist(), "bay_bev": fr.cm_to_bev(bay_cm).tolist(), "qwen_cm": qwen_cm.tolist(), "legend": []}

    # 6 plan variants
    L, W, m = cfg["robot"]["length_cm"], cfg["robot"]["width_cm"], ft["footprint_margin_cm"]
    plan_list = [("conservative_margin", L, W, m, starts[0])]
    plan_list += [(f"conservative_margin_start_shift{s}px", L, W, m, starts[s]) for s in cfg["robot"]["start_shift_toward_principal_point_px"] if s != 0]
    plan_list += [("nominal_no_margin", L, W, 0.0, starts[0]), ("outline_box_stress", outline_cm[0], outline_cm[1], 0.0, starts[0])]
    variants, runtimes = {}, {}
    for label, bl, bw, mg, sxy in plan_list:
        variants[label], runtimes[label] = plan_variant(label, bl, bw, mg, sxy, ctx)
    primary = variants["conservative_margin"]

    # margin sweep (diagnostic): the plan is recomputed with growing uniform inflation; the primary result above is unchanged
    msw = cfg["margin_sweep"]
    sweep = []
    for k in range(int(round(msw["max_cm"] / msw["step_cm"])) + 1):
        mg = k * msw["step_cm"]
        r, _ = plan_variant(f"margin_{mg}", L, W, mg, starts[0], ctx)
        sweep.append({"margin_per_side_cm": mg, "selected": r["selected"]["status"], "total_length_cm": r["selected"].get("total_length_cm")})
    survive = [x["margin_per_side_cm"] for x in sweep if x["selected"] == "PATH_A_AND_B"]

    # 7 virtual steps (never reported as success). TaskExecution stays in WAITING_FOR_PICKUP_CONFIRMATION: no confirmation is made here.
    sel = primary["selected"]
    steps, exe = [], None
    tobj = {"object_id": target["candidate_id"], "class_name": "cube", "color": "red"}
    if sel["status"] in ("PATH_A_ONLY", "PATH_A_AND_B"):
        gA = primary["legs_A"][sel["A_heading_deg"]]["goal"]
        poseA = {"xy": fr.cm_to_img([gA["xy_cm"]])[0].tolist(), "heading_deg": gA["facing_direction_deg"], "coordinate_frame": "raw_image_px"}
        reqs = [make_request("PICKUP_HANDOFF", tobj, poseA, "PROVISIONAL_FLOOR_PLANE_BEV", "PLANNED_VIRTUAL_NOT_EXECUTED")]
        if sel["status"] == "PATH_A_AND_B":
            gB = primary["legs_B"][sel["B_heading_deg"]]["goal"]
            reqs.append(make_request("PLACEMENT_HANDOFF", tobj, {"xy": fr.cm_to_img([gB["xy_cm"]])[0].tolist(), "heading_deg": gB["facing_direction_deg"], "coordinate_frame": "raw_image_px"},
                                     "PROVISIONAL_FLOOR_PLANE_BEV", "PLANNED_VIRTUAL_NOT_EXECUTED"))
        exe = TaskExecution("transport_and_place", reqs, "WAITING_FOR_PICKUP_CONFIRMATION", path_b_status="PLANNED_ASSUMING_PICKUP").to_dict()
        steps = [
            {"step": "NAVIGATE_TO_PICKUP_APPROACH", "path": "A", "status": "PLANNED_NOT_EXECUTED", "at_cm": primary["start_cm"]},
            {"step": "VIRTUAL_PICKUP", "status": "SIMULATED_NOT_EXECUTED", "at_cm": gA["xy_cm"], "gripper_must_bridge_cm": gA["body_to_target_gap_cm"],
             "requires": ["gripper reach >= the gap (UNVERIFIED)", "grasp pose / cube size (UNMEASURED)", "target confirmed (now PROVISIONAL_TARGET)"]},
        ]
        if sel["status"] == "PATH_A_AND_B":
            steps += [{"step": "CARRY_TO_BAY_APPROACH", "path": "B", "status": "PLANNED_ASSUMING_PICKUP", "at_cm": gB["xy_cm"]},
                      {"step": "VIRTUAL_PLACEMENT", "status": "SIMULATED_NOT_EXECUTED", "at_cm": gB["xy_cm"], "gripper_must_bridge_cm": gB["body_to_target_gap_cm"],
                       "requires": ["gripper reach >= the gap (UNVERIFIED)", "placement surface / boundary-crossing permission (UNDECIDED)"]}]
    ctx["legend"] = ["legend:  white rect = robot start (nominal 35x24 cm, and +margin)   yellow rect = approach pose (body incl. margin)",
                     "blue/orange line = path A / B with waypoints; thin rects = whole body at each waypoint",
                     "green tint = drivable (DRAFT zone Z02, human-unconfirmed); red = static obstacle; magenta = no-go; cyan = bay; orange dashed = Qwen box",
                     "anything not inside a drivable zone is NOT drivable (unknown is never free)"]
    figs = render(ctx, primary, runtimes["conservative_margin"], out)

    summary = {k: {"selected": v["selected"]["status"], "A": {h: (l["feasible"], (l["failure"] or {}).get("code")) for h, l in v["legs_A"].items()},
                   "B": {h: (l["feasible"], (l["failure"] or {}).get("code")) for h, l in v["legs_B"].items()}} for k, v in variants.items()}
    legA = primary["legs_A"].get(sel.get("A_heading_deg")) if sel.get("A_heading_deg") is not None else None
    legB = primary["legs_B"].get(sel.get("B_heading_deg")) if sel.get("B_heading_deg") is not None else None
    fail = lambda legs: {str(h): (l["failure"] or {}).get("code") for h, l in legs.items()}      # noqa: E731
    flags = {
        "1_target_input": {"state": cfg["target_candidate"]["status"], "verified_detection": False, "stored_qwen_crop_gate": target["stored_qwen_crop_gate"]["status"],
                           "meaning": "temporary input to test the plan chain; not evidence of detection accuracy or pick-up success"},
        "2_approach_paths": {"A_start_to_pickup_approach": "GENERATED" if legA else "NOT_GENERATED", "B_pickup_to_bay_approach": "GENERATED" if legB else "NOT_GENERATED",
                             "A_failures_by_heading": fail(primary["legs_A"]), "B_failures_by_heading": fail(primary["legs_B"])},
        "3_approach_feasibility_verified": {"verified": False, "why": ["gripper reach UNVERIFIED (distances the gripper would have to bridge are reported)", "grasp pose / cube size UNMEASURED",
                                                                       "start heading UNKNOWN; in-place rotation clearance not met at the start nor at the pick-up approach (see heading_alignment)"]},
        "4_pickup_and_placement": "VIRTUAL steps only (SIMULATED_NOT_EXECUTED); TaskExecution stays WAITING_FOR_PICKUP_CONFIRMATION",
        "5_driving_space": {"primary_footprint": "SUFFICIENT (paths A and B found, validated by v0.2 and by the whole-body sweep)" if sel["status"] == "PATH_A_AND_B"
                            else "INSUFFICIENT or no connection: " + json.dumps(sel.get("reason"), default=str),
                            "plan_survives_uniform_inflation_up_to_cm_per_side": max(survive) if survive else None,
                            "outline_box_stress": variants["outline_box_stress"]["selected"]["status"]},
        "6_required_physical_information": REQUIRED_INFO}
    heading_alignment = {"start": legA["start_heading_alignment"] if legA else None, "pickup_approach": legA["goal"].get("rotation_in_place_at_approach") if legA else None,
                         "facing_vs_planning_heading": {"pickup_facing_direction_deg": legA["goal"]["facing_direction_deg"] if legA else None, "planning_heading_deg": sel.get("A_heading_deg"),
                                                        "note": "the front gripper would have to face this direction; the plan holds an axis-aligned heading along each leg, so the facing angle is NOT satisfied/verified"}}
    failures = [{"variant": k, "leg": leg[-1], "heading_deg": h, "stage": l["failure"]["stage"], "code": l["failure"]["code"], "reason": l["failure"]["reason"]}
                for k, v in variants.items() for leg in ("legs_A", "legs_B") for h, l in v[leg].items() if l["failure"]]
    failures_and_unverified = {
        "failed_or_blocked_legs": failures,
        "primary_variant_note": "failures of non-selected headings are expected (the other heading does not fit the passages or the start); they are kept, not hidden",
        "heading_alignment": heading_alignment, "unmeasured_or_unverified": REQUIRED_INFO,
        "floor_transform_limits": ft["known_limits"], "margin_sweep_stops_at_cm": next((x["margin_per_side_cm"] for x in sweep if x["selected"] != "PATH_A_AND_B"), None)}
    result = {"experiment_id": cfg["experiment_id"], "created": datetime.now().isoformat(timespec="seconds"), "claims": cfg["claims"],
              "frames": {"model_output": "Qwen boxes are 0-1000 normalised, converted to undistorted_image_px (v0.2 helper); DINO/HSV boxes are undistorted_image_px",
                         "pixel": IMG_FRAME, "transformed": FLOOR_FRAME + " via the PROVISIONAL homography; BEV px = (cm + pad) * px_per_cm", "world": "NOT available: no validated floor transform"},
              "inputs": {"image": {"path": str(img_path), "sha256": cfg["image_sha256"], "size": [W_, H_]}, "instruction": cfg["instruction"], "qwen": qwen,
                         "target_candidate": target,
                         "robot": {"object_id": robot["object_id"], "bbox_xyxy_image_px": ro, "source": robot["source"], "detector_score": robot["detector_score"],
                                   "stored_run": first_dir.name, "outline_on_floor_cm": [round(outline_cm[0], 1), round(outline_cm[1], 1)],
                                   "outline_vs_user_size_note": "outline mapped to the floor vs the user-given 35 x 24 cm: it is larger (height parallax and/or wheels outside the given size) -> UNKNOWN, kept as a stress variant",
                                   "start_candidates_image_px": {str(k): v for k, v in starts.items()}},
                         "map_prior": {"status": "MANUAL_MAP_PRIOR (DRAFT_UNCONFIRMED, hand-drawn; transferred by the first_vlm_navigation_path image registration)",
                                       "source": str(first_dir / "map_E1_transferred.json"), "zones_cm": spec_cm, "traversable_zones_clipped_by_canvas": clipped,
                                       "rule": "only the traversable zone Z02 is drivable; static obstacles and no-go zones are subtracted; everything else is not drivable"}},
              "floor_transform": {**{k: v for k, v in ft.items() if k != "H_floor_cm_to_img"}, "legacy_2026_09_06_homography": "NOT USED (other layout, other undistorted frame 2016x1133)"},
              "targets_cm": {"cube_candidate_polygon": np.round(tgt_cm, 1).tolist(), "loading_bay_polygon": np.round(bay_cm, 1).tolist(), "qwen_box_polygon": np.round(qwen_cm, 1).tolist()},
              "robot_footprint": {"user_size_cm": [L, W, cfg["robot"]["height_cm"]], "wheel_inclusion": "UNKNOWN (to be confirmed)", "transform_margin_per_side_cm": m,
                                  "primary_extent_cm_by_heading": {str(h): list(FN.footprint_dims_cm(L, W, m, h)) for h in cfg["robot"]["headings_deg"]},
                                  "motion": cfg["robot"]["motion"], "headings_planned_deg": cfg["robot"]["headings_deg"]},
              "bev_map": {"frame": BEV_FRAME, "px_per_cm": fr.s, "pad_cm": fr.pad, "size_px": list(fr.bev_size), "cell_px": cfg["bev"]["cell_px"],
                          "layers_summary_primary_heading0": runtimes["conservative_margin"]["navs"][cfg["robot"]["headings_deg"][0]].summary()},
              "primary_variant": "conservative_margin", "heading_alignment": heading_alignment, "margin_sweep": sweep, "plan_variants": variants, "variant_summary": summary, "virtual_steps": steps, "task_execution_simulated": exe,
              "status_flags": flags, "failures_and_unverified": failures_and_unverified, "figures": figs, "saved_inputs": sorted(x.name for x in inp.iterdir()), "v02_code_fingerprint_before": fp_before, "v02_code_fingerprint_after": code_fingerprint()["sha256"], "no_hardware_command": True}
    (out / "plan.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "config_snapshot.yaml").write_text((HERE / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out / "manifest.json").write_text(json.dumps({"files_sha256": files, "config_sha256": sha256_file(HERE / "config.yaml")}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({"floor": {k: ft[k] for k in ("status", "fit_rms_cm", "seam_max_abs_dev_cm", "footprint_margin_cm")}, "flags": flags, "variants": summary,
                      "primary_selected": primary["selected"]}, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
