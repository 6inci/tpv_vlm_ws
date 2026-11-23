#!/usr/bin/env python3
"""STEP 3 variant (2026-11-20): wheel-boundary collision model, several margins, one approved photo.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/navigation_handoff/scripts/path_planning_wheel_boundary.py --photo photo_1   # new results/<time>_wheelbound_<photo>/

Model (config wheel_boundary_experiment):
  - wheel envelope from the URDF (+ margin) must lie on FREE cells of map.pgm (inside the yellow zone, off obstacles),
  - body 35 x 24 cm (+ margin) must not overlap STATIC_OBSTACLE / NO_GO / UNKNOWN; it may overhang NOT_DRIVABLE floor (objects there are NOT
    mapped: every overhang is measured and reported as UNVERIFIED),
  - gripper may reach beyond the boundary (unchanged).
Everything else is path_planning.py imported unchanged: 4-neighbour A* with turn cost on the v0.2 edge cost, approach candidates
(0/90/180/270), start escape, rotating-rectangle check, turn-point waypoints, v0.2 validate_path, footprint_nav.sweep_check, figures.
path_planning.py and its results are not modified. No model call, no ROS, no robot.
"""
import argparse
import json
import math
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
from semantic_task_nav.config import code_fingerprint, sha256_file  # noqa: E402


def read_urdf_wheels(path):
    """Wheel envelope (along heading, lateral) in cm from the URDF xacro properties."""
    t = Path(path).read_text(encoding="utf-8")
    v = {k: float(re.search(rf'name="{k}"\s+value="([0-9.eE+-]+)"', t).group(1)) for k in ("half_wheelbase", "half_track", "wheel_radius", "wheel_width")}
    along = 2 * (v["half_wheelbase"] + v["wheel_radius"]) * 100
    lateral = 2 * (v["half_track"] + v["wheel_width"] / 2) * 100
    return {"properties_m": v, "envelope_cm": [round(along, 3), round(lateral, 3)],
            "formula": "along = 2 (half_wheelbase + wheel_radius), lateral = 2 (half_track + wheel_width / 2)"}


class DualMap:
    """Same interface as path_planning.PgmMap. The (L, W) a reused function passes are BODY dims (+ 2 margin); the wheel envelope gets the
    same margin. A pose is valid when the wheel envelope is on FREE and the body is off the blocking layers."""

    def __init__(self, pm_wheel, pm_body, wheel_nominal, body_nominal):
        self.w, self.b = pm_wheel, pm_body
        self.wheel_nominal, self.body_nominal = tuple(wheel_nominal), tuple(body_nominal)
        for k in ("grid", "frame", "layers", "rows", "cols", "s", "sub", "cell_px", "height", "width", "xs", "ys", "free"):
            setattr(self, k, getattr(pm_wheel, k))
        self.free_full = pm_body.free_full          # reused footprint_nav.sweep_check then checks the BODY against the blocking layers
        self._cache = {}

    def wheel_dims(self, L, W):
        m = (L - self.body_nominal[0]) / 2
        return self.wheel_nominal[0] + 2 * m, self.wheel_nominal[1] + 2 * m

    def cspace(self, L, W, yaw):
        key = (round(L, 3), round(W, 3), round(yaw % 360, 3))
        if key not in self._cache:
            Lw, Ww = self.wheel_dims(L, W)
            valid = self.w.cspace(Lw, Ww, yaw)[0] & self.b.cspace(L, W, yaw)[0]
            rr = min(Lw, Ww) / 2 * self.s
            risk = np.where(valid, np.clip(1.0 - (self.w.clearance - rr) / self.w.safety_margin_px, 0.0, 1.0), np.nan)
            self._cache[key] = (valid, risk)
        return self._cache[key]

    def view(self, L, W, yaw):
        valid, risk = self.cspace(L, W, yaw)
        return SimpleNamespace(valid=valid, risk=risk, rows=self.rows, cols=self.cols, cell_px=self.cell_px)

    def body_free(self, centre, L, W, yaw):
        Lw, Ww = self.wheel_dims(L, W)
        okw, bw = self.w.body_free(centre, Lw, Ww, yaw)
        okb, bb = self.b.body_free(centre, L, W, yaw)
        return okw and okb, bw + bb

    def v02_nav(self, L, W):
        Lw, Ww = self.wheel_dims(L, W)               # v0.2 validator: inscribed disc of the WHEEL envelope stays on FREE (necessary only)
        return SimpleNamespace(cell_px=self.cell_px, robot_radius_px=min(Lw, Ww) / 2 * self.s, width=self.width, height=self.height,
                               clearance_full=self.w.clearance_full)

    def centre_floor(self, r, c):
        return self.w.centre_floor(r, c)

    def cell_of_floor(self, p):
        return self.w.cell_of_floor(p)

    def in_grid(self, r, c):
        return self.w.in_grid(r, c)

    def floor_to_px(self, pts):
        return self.w.floor_to_px(pts)

    def body_polygon_px(self, centre, L, W, yaw):
        return self.w.body_polygon_px(centre, L, W, yaw)


class Metrics:
    """Clearances of the NOMINAL footprints (so the effect of the margin is visible) from distance transforms at raster resolution."""

    def __init__(self, dm):
        self.dm = dm
        wheel_ok, body_ok = dm.w.free_full.astype(np.uint8), dm.b.free_full.astype(np.uint8)
        dt = lambda m: cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]  # noqa: E731
        self.wheel_ok, self.body_ok = wheel_ok.astype(bool), body_ok.astype(bool)
        self.dt_wheel, self.dt_body = dt(wheel_ok), dt(body_ok)
        self.dt_out_to_free = dt((~self.wheel_ok).astype(np.uint8))

    def _fp(self, centre, L, W, yaw):
        poly = self.dm.body_polygon_px(centre, L, W, yaw)
        x0, y0 = np.floor(poly.min(0)).astype(int) - 1
        x1, y1 = np.ceil(poly.max(0)).astype(int) + 2
        m = PP.PgmMap._mask(poly, x0, y0, x1 - x0, y1 - y0)
        H, Wd = self.wheel_ok.shape
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, Wd), min(y1, H)
        return (slice(sy0, sy1), slice(sx0, sx1)), m[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0], int(m.sum() - m[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0].sum())

    def pose(self, centre, yaw):
        s = self.dm.s
        (sw, mw, outw) = self._fp(centre, *self.dm.wheel_nominal, yaw)
        (sb, mb, outb) = self._fp(centre, *self.dm.body_nominal, yaw)
        wheel_out = int((mw & ~self.wheel_ok[sw]).sum()) + outw
        body_hit = int((mb & ~self.body_ok[sb]).sum()) + outb
        over = mb & ~self.wheel_ok[sb] & self.body_ok[sb]
        return {"wheels_outside_yellow_px": wheel_out, "body_on_blocking_layer_px": body_hit,
                "wheel_clearance_to_yellow_boundary_cm": round(float(self.dt_wheel[sw][mw].min()) / s, 2) if wheel_out == 0 else 0.0,
                "body_clearance_to_mapped_obstacle_cm": round(float(self.dt_body[sb][mb].min()) / s, 2) if body_hit == 0 else 0.0,
                "body_overhang_beyond_yellow_cm": round(float(self.dt_out_to_free[sb][over].max()) / s, 2) if over.any() else 0.0,
                "body_overhang_area_cm2": round(float(over.sum()) / s ** 2, 1)}

    def path(self, d, step_cm):
        """Every step_cm along the waypoint polyline, per-segment heading; the start-escape segment is included (nominal footprints anyway)."""
        worst = {"wheels_outside_yellow_px": 0, "body_on_blocking_layer_px": 0, "wheel_clearance_to_yellow_boundary_cm": math.inf,
                 "body_clearance_to_mapped_obstacle_cm": math.inf, "body_overhang_beyond_yellow_cm": 0.0, "body_overhang_area_cm2": 0.0}
        n_samples = 0
        for sgm in d["segments"]:
            a, b = d["floor"][sgm["from_index"]], d["floor"][sgm["to_index"]]
            n = max(1, int(math.ceil(np.linalg.norm(b - a) / step_cm)))
            for k in range(n + 1):
                q = self.pose(a + (b - a) * k / n, sgm["body_yaw_map_deg"])
                n_samples += 1
                for key in ("wheels_outside_yellow_px", "body_on_blocking_layer_px", "body_overhang_beyond_yellow_cm", "body_overhang_area_cm2"):
                    worst[key] = max(worst[key], q[key])
                for key in ("wheel_clearance_to_yellow_boundary_cm", "body_clearance_to_mapped_obstacle_cm"):
                    worst[key] = min(worst[key], q[key])
        worst = {("max_" + k if k.startswith(("wheels", "body_on", "body_overhang")) else "min_" + k): v for k, v in worst.items()}
        return {**worst, "n_samples": n_samples, "footprints": "NOMINAL wheel envelope and body (margin excluded)"}


def nearest_reference_pose(dm, target, nominal, margin, yaws):
    """Closest collision-free stop (any of the axis headings) to the target: REFERENCE ONLY, not a grasp / placement pose."""
    best = None
    for yaw in yaws:
        valid, _ = dm.cspace(nominal[0] + 2 * margin, nominal[1] + 2 * margin, yaw)
        rr, cc = np.nonzero(valid)
        if not len(rr):
            continue
        xs, ys = dm.grid.cell_center_floor(rr, cc)
        dist = np.hypot(xs - target[0], ys - target[1])
        k = int(np.argmin(dist))
        if best is None or dist[k] < best["centre_to_target_cm"]:
            m = dm.frame.floor_to_map([[xs[k], ys[k]]])[0]
            best = {"yaw_map_deg": yaw, "pose_map": {"x_m": round(float(m[0]), 4), "y_m": round(float(m[1]), 4), "yaw_deg": float(yaw)},
                    "centre_to_target_cm": round(float(dist[k]), 1), "status": "REFERENCE_ONLY (not a grasp / placement pose)"}
    return best


def relabel_body_vs_move(d, tol_deg=15.0):
    """path_planning.describe_leg compares move and body heading exactly, so a non-axis start heading (e.g. -178.2) is labelled 'lateral'
    while moving along -x. Relabel with a tolerance; the raw headings in the record are unchanged."""
    for s in d["segments"]:
        rel = abs(CT.wrap_deg(s["move_yaw_map_deg"] - s["body_yaw_map_deg"]))
        s["body_vs_move"] = "forward" if rel <= tol_deg else "backward" if rel >= 180 - tol_deg else "lateral"
        s["body_vs_move_rule"] = f"|move - body| <= {tol_deg} deg forward, >= {180 - tol_deg} backward, else lateral"
    return d


def run_condition(dm, mt, pcfg, gripper, nominal, margin, start_floor, start_yaw, cube_floor, place_floor):
    gc, gsel = PP.select_and_plan(dm, "A", cube_floor, start_floor, start_yaw, nominal, margin, pcfg, gripper, True)
    pc, psel = ([], None)
    if gsel:
        pc, psel = PP.select_and_plan(dm, "B", place_floor, np.array(gsel["centre_floor_cm"]), gsel["yaw_map_deg"], nominal, margin, pcfg, gripper, False)
    return gc, gsel, pc, psel


def stop_record(dm, mt, sel, kind):
    if not sel:
        return None
    return {"target": kind, "pose_map": sel["pose_map"], "centre_to_target_along_heading_cm": sel["centre_to_target_along_heading_cm"],
            "front_to_target_gap_cm": sel["front_to_target_gap_cm"], "lateral_offset_cm": sel["lateral_offset_cm"],
            "required_gripper_offset_if_dy0": sel["required_gripper_offset_if_dy0"], "gripper_alignment": "UNVERIFIED (gripper dx, dy not measured)",
            "gripper_corridor": sel["gripper_corridor"], "nominal_footprints_at_stop": mt.pose(np.array(sel["centre_floor_cm"]), sel["yaw_map_deg"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", required=True)
    args = ap.parse_args()
    cfg = CT.load_config()
    pcfg, mcfg, wcfg = cfg["planning"], cfg["map_export"], cfg["wheel_boundary_experiment"]
    before = {"v02": code_fingerprint()["sha256"]}
    doc, pos_path = OP.load_approved(cfg, args.photo)
    photo = next(p for p in CT.resolve_photos(cfg) if p["name"] == args.photo)
    frame, fc, map_run = OP.load_step2_frame(cfg)
    map_files = {k: map_run / "maps" / k for k in ("map.pgm", "map.yaml", "map_layers.npy", "map_preview.png")}
    before["inputs"] = {**{k: sha256_file(v) for k, v in map_files.items()}, "approved": sha256_file(pos_path), "urdf": sha256_file(wcfg["urdf"])}
    pgm, _ = ME.read_pgm(map_files["map.pgm"])
    layers = np.load(map_files["map_layers.npy"])
    g = fc["grid"]
    grid = CT.MapGrid(frame, g["extent_floor_cm"], g["resolution_m"])
    code_of = {v.split(" (")[0]: k for k, v in ME.CODE_NAMES.items()}
    body_ok = ~np.isin(layers, [code_of[c] for c in wcfg["body_blocking_codes"]])
    vals = mcfg["values"]
    pm_w = PP.PgmMap(pgm, layers, vals, grid, frame, pcfg["raster_px_per_cm"], pcfg["safety_margin_cm"])
    pm_b = PP.PgmMap(np.where(body_ok, vals["free"], vals["occupied"]).astype(np.uint8), layers, vals, grid, frame, pcfg["raster_px_per_cm"], pcfg["safety_margin_cm"])
    urdf = read_urdf_wheels(wcfg["urdf"])
    nominal = (cfg["robot"]["length_cm"], cfg["robot"]["width_cm"])
    dm = DualMap(pm_w, pm_b, urdf["envelope_cm"], nominal)
    mt = Metrics(dm)
    gripper = {**cfg["gripper"], "offset_y_m_assumed": pcfg["gripper_dy_assumption_m"]}
    pts = {"robot_base_link": OP._pt(frame, doc["robot"]["base_link_image_px"]), "A_grasp_target": OP._pt(frame, doc["A_cube"]["grasp_target_image_px"]),
           "B_placement_target": OP._pt(frame, doc["B_placement"]["placement_target_image_px"])}
    start_floor, start_yaw = np.array(pts["robot_base_link"]["floor_cm"]), float(doc["robot"]["yaw_map_deg"])
    cube_floor, place_floor = np.array(pts["A_grasp_target"]["floor_cm"]), np.array(pts["B_placement_target"]["floor_cm"])

    stamp = datetime.now().astimezone()
    root = CT.resolve(cfg["results_dir"]) / (stamp.strftime("%Y%m%d_%H%M%S") + f"_wheelbound_{args.photo}")
    (root / "maps").mkdir(parents=True, exist_ok=False)
    for k in ("map.pgm", "map.yaml", "map_preview.png"):
        (root / "maps" / k).write_bytes(map_files[k].read_bytes())
    img = cv2.imread(photo["path"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    unc = wcfg["position_uncertainty_cm"]
    summary, map_figs, photo_figs = [], [], []
    for cond in wcfg["conditions"]:
        m = float(cond["margin_cm"])
        out = root / cond["label"] / args.photo
        out.mkdir(parents=True)
        gc, gsel, pc, psel = run_condition(dm, mt, pcfg, gripper, nominal, m, start_floor, start_yaw, cube_floor, place_floor)
        legs, desc = {}, {}
        for key, sel in (("A", gsel), ("B", psel)):
            if sel:
                legs[key], desc[key] = sel["leg"], relabel_body_vs_move(PP.describe_leg(dm, sel["leg"], nominal, m, pcfg))
                PP.write_csv(out / f"path_{key}.csv", sel["leg"], desc[key])
        driving = ("NOT_FOR_DRIVING: margin below the position uncertainty" if m < unc else
                   "PROVISIONAL: margin = independent transform error only (node residual up to 5.2 cm, DRAFT map, unmapped objects beyond the yellow line)")
        lines = [f"CONDITION {cond['label']}: margin {m} cm/side on wheels and body ({cond['role']})", driving,
                 f"wheels (URDF envelope {urdf['envelope_cm'][0]:.1f} x {urdf['envelope_cm'][1]:.1f} cm) inside yellow; body 35 x 24 cm off obstacles, overhang allowed"]
        poses = [{"floor": start_floor.tolist(), "yaw": start_yaw, "size": nominal, "color": (200, 200, 200), "dot": (255, 255, 255), "label": f"START yaw {start_yaw:.1f}"}]
        for key, sel, col, lab, tgt in (("A", gsel, (0, 0, 255), "GRASP STOP", cube_floor), ("B", psel, (255, 0, 0), "PLACE STOP", place_floor)):
            if sel:
                poses.append({"floor": sel["centre_floor_cm"], "yaw": sel["yaw_map_deg"], "size": nominal, "color": col, "dot": col,
                              "label": f"{lab} yaw {sel['yaw_map_deg']:.0f} UNVERIFIED", "target": tgt.tolist()})
                lines.append(f"Path {key}: {desc[key]['length_cm']} cm, {desc[key]['n_turns']} turns, checks {'PASS' if desc[key]['collision_checks_passed'] else 'FAIL'}; "
                             f"gripper must reach {sel['front_to_target_gap_cm']} cm beyond the body front")
            else:
                lines.append(f"Path {key}: NOT FOUND" + (" (no grasp stop)" if key == "B" and not gsel else ""))
        lines += ["body overhang beyond the yellow line: objects there are NOT mapped -> UNVERIFIED", "gripper offset not measured -> grasp / placement UNVERIFIED"]
        scene = {"legs": desc, "poses": poses, "status_lines": lines,
                 "targets": [{"kind": "cube", "floor": cube_floor.tolist(), "label": "A grasp target"}, {"kind": "place", "floor": place_floor.tolist(), "label": "B placement target"}]}
        PP.render_photo(img, dm, scene, out / "path_planning_photo.png")
        PP.render_map(dm, scene, out / "path_planning_map.png")
        map_figs.append(out / "path_planning_map.png")
        photo_figs.append(out / "path_planning_photo.png")

        def leg_rec(key):
            if key not in legs:
                return None
            lg, d = legs[key], desc[key]
            return {"strategy": lg["strategy"], "rotation_cell": lg.get("rotation_cell"), "length_cm": d["length_cm"], "n_raw_points": len(lg["raw_cells"]),
                    "n_turns": d["n_turns"], "events": lg["events"], "segments": d["segments"], "poses": PP.pose_list(lg, d), "checks": d["checks"],
                    "collision_checks_passed": d["collision_checks_passed"], "nominal_footprint_metrics": mt.path(d, wcfg["metrics_step_cm"]),
                    "baseline_v02_astar_4neighbours": lg["baseline_v02_astar_4neighbours"], "raw_path_file": f"path_{key}.csv"}

        strip = lambda cs: [{k: v for k, v in c.items() if k != "leg"} for c in cs]  # noqa: E731
        rec = {"condition": cond, "driving_status": driving, "path_A": leg_rec("A"), "path_B": leg_rec("B"),
               "grasp_stop": stop_record(dm, mt, gsel, "A grasp target"), "place_stop": stop_record(dm, mt, psel, "B placement target"),
               "grasp_candidates": strip(gc), "place_candidates": strip(pc),
               "reference_nearest_stop_A": None if gsel else nearest_reference_pose(dm, cube_floor, nominal, m, pcfg["approach_yaws_map_deg"]),
               "reference_nearest_stop_B": None if psel else nearest_reference_pose(dm, place_floor, nominal, m, pcfg["approach_yaws_map_deg"]),
               "rotation_check": "static rotating-rectangle test of wheels (on FREE) and body (off blocking layers) at the rotation cell; dynamics UNVERIFIED"}
        (out / "path_plan.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        wy = {"frame_id": "map", "condition": cond["label"], "driving_status": driving,
              "note": "yaw = body heading (gripper side), not the movement direction; Nav2 goals make Nav2 plan its own path, not follow path_*.csv",
              "path_A": rec["path_A"] and rec["path_A"]["poses"], "path_B": rec["path_B"] and rec["path_B"]["poses"],
              "grasp_stop_pose": gsel and {**gsel["pose_map"], "status": "UNVERIFIED"}, "place_stop_pose": psel and {**psel["pose_map"], "status": "UNVERIFIED"}}
        (out / "waypoints.yaml").write_text(yaml.safe_dump(wy, sort_keys=False, allow_unicode=True), encoding="utf-8")
        val = {"condition": cond["label"], "margin_cm": m, "driving_status": driving,
               "path_A_status": "PATH_FOUND" if gsel else "NOT_FOUND", "path_B_status": "PATH_FOUND" if psel else ("NOT_FOUND" if gsel else "NOT_COMPUTED (no grasp stop)"),
               "path_A_checks": rec["path_A"] and {"passed": rec["path_A"]["collision_checks_passed"], **rec["path_A"]["checks"], **rec["path_A"]["nominal_footprint_metrics"]},
               "path_B_checks": rec["path_B"] and {"passed": rec["path_B"]["collision_checks_passed"], **rec["path_B"]["checks"], **rec["path_B"]["nominal_footprint_metrics"]},
               "grasp_alignment": "UNVERIFIED", "placement_alignment": "UNVERIFIED", "body_overhang_objects": "UNVERIFIED (not mapped beyond the yellow line)"}
        (out / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        summary.append({"label": cond["label"], "margin_cm": m, "driving_status": driving, "A": val["path_A_status"], "B": val["path_B_status"],
                        "A_len_cm": rec["path_A"] and rec["path_A"]["length_cm"], "B_len_cm": rec["path_B"] and rec["path_B"]["length_cm"],
                        "A_checks": rec["path_A"] and rec["path_A"]["collision_checks_passed"], "B_checks": rec["path_B"] and rec["path_B"]["collision_checks_passed"],
                        "grasp_stop": rec["grasp_stop"] and {k: rec["grasp_stop"][k] for k in ("pose_map", "front_to_target_gap_cm", "lateral_offset_cm")},
                        "place_stop": rec["place_stop"] and {k: rec["place_stop"][k] for k in ("pose_map", "front_to_target_gap_cm", "lateral_offset_cm")},
                        "A_metrics": rec["path_A"] and rec["path_A"]["nominal_footprint_metrics"], "B_metrics": rec["path_B"] and rec["path_B"]["nominal_footprint_metrics"],
                        "grasp_stop_footprints": rec["grasp_stop"] and rec["grasp_stop"]["nominal_footprints_at_stop"],
                        "place_stop_footprints": rec["place_stop"] and rec["place_stop"]["nominal_footprints_at_stop"],
                        "reference_A": rec["reference_nearest_stop_A"], "reference_B": rec["reference_nearest_stop_B"]})

    sweep = []
    for m in wcfg["margin_sweep_cm"]:
        gc, gsel, pc, psel = run_condition(dm, mt, pcfg, gripper, nominal, float(m), start_floor, start_yaw, cube_floor, place_floor)
        sweep.append({"margin_cm": m, "A": "PATH_FOUND" if gsel else "NOT_FOUND", "B": "PATH_FOUND" if psel else ("NOT_FOUND" if gsel else "NOT_COMPUTED"),
                      "grasp_yaw": gsel and gsel["yaw_map_deg"], "place_yaw": psel and psel["yaw_map_deg"],
                      "A_len_cm": gsel and gsel.get("leg_length_cm"), "B_len_cm": psel and psel.get("leg_length_cm")})
    w = 1300
    stack = lambda fs: np.vstack([cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1]))) for im in (cv2.imread(str(f)) for f in fs)])  # noqa: E731
    cv2.imwrite(str(root / "comparison_maps.png"), stack(map_figs))
    cv2.imwrite(str(root / "comparison_photos.png"), stack(photo_figs))
    after_ok = code_fingerprint()["sha256"] == before["v02"] and all(sha256_file(v) == before["inputs"][k] for k, v in map_files.items()) \
        and sha256_file(pos_path) == before["inputs"]["approved"] and sha256_file(photo["path"]) == photo["sha256"]
    comp = {"photo": photo["name"], "approved_positions": str(pos_path), "positions": pts, "start_yaw_map_deg": start_yaw,
            "robot_geometry": {"body_nominal_cm": list(nominal), "body_source": "user decision 2026-11-19 (centred on base_link: UNVERIFIED)",
                               "wheel_envelope_cm": urdf["envelope_cm"], "wheel_source": wcfg["urdf"], "urdf_values": urdf["properties_m"],
                               "urdf_status": wcfg["urdf_status"], "gripper": "UNKNOWN (dx, dy, size not measured)"},
            "baseline_previous_model": {"run": wcfg["baseline_run"], "rule": "whole body 35 x 24 + 3.5 cm/side inside FREE", "A": "NOT_FOUND", "B": "NOT_COMPUTED"},
            "conditions": summary, "margin_sweep": sweep, "environment_unchanged_after_run": after_ok}
    (root / "comparison.json").write_text(json.dumps(comp, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (root / "config_snapshot.yaml").write_text(CT.CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    (root / "manifest.json").write_text(json.dumps({
        "experiment_id": cfg["experiment_id"], "step": "STEP3_wheel_boundary_variant", "run_id": root.name, "created": stamp.isoformat(timespec="seconds"),
        "git_commit": None, "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
        "code_sha256": {p.name: sha256_file(p) for p in (Path(__file__).resolve(), HERE / "path_planning.py", HERE / "coordinate_transform.py",
                                                         HERE / "map_export.py", HERE / "object_positions.py")},
        "config_sha256": sha256_file(CT.CONFIG_PATH), "inputs": {"photo": photo["sha256"], **before["inputs"]}, "v02_code_fingerprint": before["v02"],
        "environment_unchanged_after_run": after_ok, "hardware_condition": "offline; no model, GPU, ROS or robot",
        "outputs_sha256": {str(p.relative_to(root)): sha256_file(p) for p in files}}, indent=1), encoding="utf-8")
    print(json.dumps({"run_dir": str(root), "conditions": [{k: s[k] for k in ("label", "A", "B", "A_len_cm", "B_len_cm", "A_checks", "B_checks")} for s in summary],
                      "sweep": [(s["margin_cm"], s["A"], s["B"]) for s in sweep], "environment_unchanged": after_ok}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
