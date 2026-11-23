#!/usr/bin/env python3
"""Unified Path A / Path B planning (user approval 2026-11-20): ONE rule set for every photo (config/unified_planning.yaml = the STEP 4-B
batch checks) through planning_checks.py. Existing scripts, configs, results and the hand-off package are not modified.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/navigation_handoff/scripts/path_planning_unified.py                          # photos of unified_planning.yaml -> results/<time>_unified/
  $PY experiments/navigation_handoff/scripts/path_planning_unified.py --photos photo_2,photo_7 --out-root <dir>

Positions: data/approved_positions/<photo>.yaml when it exists (USER_APPROVED; a malformed file raises), otherwise the STEP 4-B estimate
(ECC robot vs photo_5, AI-preliminary cube label, first STEP 4-A empty pallet) = PROVISIONAL -> REVIEW_REQUIRED. Per-photo pallets: STEP 4-A
inventory (AI reading, PROVISIONAL) for every photo. Each photo is compared with its earlier run (compare_previous, read only).
No model call, no ROS, no robot.
"""
import argparse
import json
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
import path_planning_batch_step4b as BB  # noqa: E402  (reused unchanged: robot_pose, leg_summary, LINE_COL, run_name)
import path_planning_wheel_boundary as WB  # noqa: E402
import planning_checks as PC  # noqa: E402
import survey_photos_step4a as SV  # noqa: E402
from semantic_task_nav.config import code_fingerprint, sha256_file  # noqa: E402

UCFG_PATH = CT.CONFIG_PATH.parent / "unified_planning.yaml"
EXISTING_CODE = ["path_planning.py", "path_planning_wheel_boundary.py", "path_planning_batch_step4b.py", "make_handoff_package.py",
                 "position_picker.py", "object_positions.py", "coordinate_transform.py", "map_export.py", "survey_photos_step4a.py"]


def load_unified():
    u = yaml.safe_load(UCFG_PATH.read_text(encoding="utf-8"))
    if u["base_config"] != CT.CONFIG_PATH.name:
        raise ValueError(f"{UCFG_PATH}: base_config {u['base_config']!r} is not {CT.CONFIG_PATH.name}")
    PC.check_policy(u)
    return u


def positions(cfg, photo, frame):
    """USER_APPROVED file when it exists (object_positions.load_approved refuses a malformed one), else the STEP 4-B PROVISIONAL estimate."""
    p = CT.resolve(cfg["approved_positions_dir"]) / f"{photo['name']}.yaml"
    if p.exists():
        doc, p = OP.load_approved(cfg, photo["name"])
        r = doc["robot"]
        return {"status": "USER_APPROVED", "source": str(p), "source_sha256": sha256_file(p), "approved_at": doc["approved_at"],
                "robot_px": r["base_link_image_px"], "yaw_map_deg": float(r["yaw_map_deg"]), "A_px": doc["A_cube"]["grasp_target_image_px"],
                "B_px": doc["B_placement"]["placement_target_image_px"], "robot_source": f"{r.get('base_link_source')}; heading {r.get('yaw_source')}",
                "targets_source": f"A {doc['A_cube'].get('grasp_target_source')}, B {doc['B_placement'].get('placement_target_source')}"}
    bc = cfg["step4b_batch"]
    robot = BB.robot_pose(cfg, photo, frame)
    lab = yaml.safe_load(CT.resolve(bc["cube_labels"]).read_text(encoding="utf-8"))["labels"][photo["earlier_id"]]
    empty = cfg["step4a"]["inventory"][photo["name"]]["pallet_empty"]
    return {"status": "PROVISIONAL", "source": "STEP 4-B estimate (REVIEW_REQUIRED)", "robot_estimate": robot,
            "robot_px": robot["base_link_px"], "yaw_map_deg": float(robot["yaw_map_deg"]), "A_px": [(lab[0] + lab[2]) / 2, (lab[1] + lab[3]) / 2],
            "B_px": [(empty[0][0] + empty[0][2]) / 2, (empty[0][1] + empty[0][3]) / 2] if empty else None, "robot_source": robot["method"],
            "targets_source": "A = AI-preliminary cube label centre, B = centre of the first STEP 4-A empty pallet"}


def compare_previous(ucfg, name, out, rec):
    """Earlier run of this photo (read only): statuses, byte equality of the path files, stop poses."""
    ent = ucfg["compare_previous"].get(name)
    if not ent:
        return None
    d = CT.resolve(ent["run"])
    p = json.loads((d / "path_plan.json").read_text(encoding="utf-8"))
    st = {k: p.get(f"path_{k}_status") or ("VALID" if p.get(f"path_{k}") and p[f"path_{k}"]["collision_checks_passed"] else "NOT_FOUND") for k in ("A", "B")}
    files = {}
    for k in ("A", "B"):
        a, b = d / f"path_{k}.csv", out / f"path_{k}.csv"
        files[f"path_{k}.csv"] = ("BOTH_ABSENT" if not a.exists() and not b.exists() else "ONLY_PREVIOUS" if not b.exists() else
                                  "ONLY_UNIFIED" if not a.exists() else "IDENTICAL" if a.read_bytes() == b.read_bytes() else "DIFFERENT")
    stops = {k: (p.get(k) or {}).get("pose_map") == (rec.get(k) or {}).get("pose_map") for k in ("grasp_stop", "place_stop")}
    return {"run": str(d), "rules": ent["rules"], "previous_A": st["A"], "previous_B": st["B"], "unified_A": rec["path_A_status"], "unified_B": rec["path_B_status"],
            "files": files, "same_stop_pose": stops}


def plan_photo(cfg, ucfg, ctx_sv, photo, frame, grid, pgm, layers, urdf, nominal, out):
    pcfg, gcfg = cfg["planning"], ucfg["gripper_baseline"]
    m, px = float(ucfg["margin_cm"]), pcfg["raster_px_per_cm"]
    pos = positions(cfg, photo, frame)
    inv = cfg["step4a"]["inventory"][photo["name"]]
    boxes_px = [inv["cube"]] + inv["pallet_empty"]
    obst = PC.obstacle_mask(grid, frame, boxes_px, px)
    world = PC.build_world(pgm, layers, cfg["map_export"]["values"], grid, frame, pcfg, ucfg["body_blocking_codes"], urdf["envelope_cm"], nominal, obst)
    own_a, other_a, ia = PC.target_masks(grid, frame, boxes_px, pos["A_px"], px)
    own_b, other_b, ib = PC.target_masks(grid, frame, boxes_px, pos["B_px"], px) if pos["B_px"] else (None, None, None)
    pt = lambda uv: OP._pt(frame, uv)  # noqa: E731
    P = {"robot_base_link": {**pt(pos["robot_px"]), "yaw_map_deg": pos["yaw_map_deg"]}, "A_grasp_target": pt(pos["A_px"]),
         "B_placement_target": pt(pos["B_px"]) if pos["B_px"] else None}
    start_f, start_yaw = np.array(P["robot_base_link"]["floor_cm"]), pos["yaw_map_deg"]
    a_f = np.array(P["A_grasp_target"]["floor_cm"])
    b_f = np.array(P["B_placement_target"]["floor_cm"]) if pos["B_px"] else None
    gripper = {**cfg["gripper"], "offset_y_m_assumed": pcfg["gripper_dy_assumption_m"]}
    res = PC.plan_task(world, start_f, start_yaw, a_f, b_f, {"A": (own_a, other_a), "B": (own_b, other_b)}, nominal, m, pcfg, gripper, gcfg,
                       cfg["wheel_boundary_experiment"]["metrics_step_cm"], ctx_sv.zone_at)
    A, B = res["A"], res["B"]
    for k, L in (("A", A), ("B", B)):
        if L["desc"]:
            PP.write_csv(out / f"path_{k}.csv", L["selected"]["leg"], L["desc"])

    def leg_rec(L, k):
        if not L["desc"]:
            return None
        lg, d = L["selected"]["leg"], L["desc"]
        return {"status": d["status"], "strategy": lg["strategy"], "rotation_cell": lg.get("rotation_cell"), "length_cm": d["length_cm"],
                "n_raw_points": len(lg["raw_cells"]), "n_turns": d["n_turns"], "events": lg["events"], "segments": d["segments"], "poses": PP.pose_list(lg, d),
                "checks": d["checks"], "collision_checks_passed": d["collision_checks_passed"], "nominal_footprint_metrics": d["nominal_footprint_metrics"],
                "baseline_v02_astar_4neighbours": lg["baseline_v02_astar_4neighbours"], "raw_path_file": f"path_{k}.csv"}

    def stop_rec(L, kind):
        s = WB.stop_record(world, res["metrics"], L["selected"], kind)
        return s and {**s, "gripper_baseline_2d": L["selected"]["gripper_baseline_2d"],
                      "gripper_corridor_note": "gripper_corridor = layer count of path_planning.approach_candidates (record only); the decision is gripper_baseline_2d"}

    unc = cfg["wheel_boundary_experiment"]["position_uncertainty_cm"]
    driving = f"NOT_FOR_DRIVING (margin {m} cm < position uncertainty {unc} cm)" if m < unc else "PROVISIONAL (margin = transform error only; DRAFT map)"
    roles = ["cube pallet"] + [f"empty pallet {i}" for i in range(1, len(boxes_px))]
    boxes = [{"role": roles[i] + (" = own pallet of A" if i == ia else "") + (" = own pallet of B" if i == ib else "") + " (obstacle for wheels/body)",
              "box_px": b} for i, b in enumerate(boxes_px)]
    strip = lambda cs: [{k: v for k, v in c.items() if k != "leg"} for c in cs]  # noqa: E731
    path_a, path_b = leg_rec(A, "A"), leg_rec(B, "B")
    rec = {"photo": {k: photo[k] for k in ("name", "sha256", "earlier_id", "current_file")}, "photo_path_now": photo["path"],
           "file_name_matches_config": photo["file_name_matches_config"], "input_status": pos["status"],
           "positions_source": {k: v for k, v in pos.items() if not k.endswith("_px") and k != "yaw_map_deg"}, "positions": P,
           "per_photo_obstacles": boxes, "per_photo_obstacle_cells": int(obst.sum()), "obstacles_status": ucfg["positions"]["obstacles_status"],
           "own_pallet_index": {"A": ia, "B": ib}, "policy": {"config": str(UCFG_PATH), "sha256": sha256_file(UCFG_PATH), "margin_cm": m, "gripper_baseline": gcfg},
           "path_A_status": A["status"], "path_B_status": B["status"], "path_B_basis": B["basis"], "path_B_role": B["role"],
           "path_A": path_a, "path_B": path_b, "summary_A": BB.leg_summary(path_a), "summary_B": BB.leg_summary(path_b),
           "grasp_stop": stop_rec(A, "A grasp target"), "place_stop": stop_rec(B, "B placement target"),
           "grasp_candidates": strip(A["candidates"]), "place_candidates": strip(B["candidates"]), "cause_A": A["cause"], "cause_B": B["cause"],
           "reference_nearest_stop_A": A["reference"], "reference_nearest_stop_B": B["reference"],
           "straight_line_crossings": {"robot_to_A": SV.segment_crossings(ctx_sv, start_f, a_f),
                                       "A_to_B": SV.segment_crossings(ctx_sv, a_f, b_f) if b_f is not None else None},
           "grasp_placement": "UNVERIFIED (gripper offset, reach and 3D clearance not measured; the 2D baseline is not a grasp claim)",
           "gripper_3d": PC.GRIPPER_3D, "driving": driving, "status_meaning": PC.STATUS}
    (out / "path_plan.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (out / "positions_used.json").write_text(json.dumps({k: rec[k] for k in ("photo", "input_status", "positions_source", "positions", "per_photo_obstacles",
                                                                             "obstacles_status", "own_pallet_index")}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    review = "" if pos["status"] == "USER_APPROVED" else " / REVIEW_REQUIRED"
    wy = {"frame_id": "map", "status": f"A {A['status']}, B {B['status']}; positions {pos['status']}{review}; obstacles PROVISIONAL; grasp/placement UNVERIFIED; {driving}",
          "note": "yaw = body heading (gripper side), not the movement direction", "path_B_basis": B["basis"], "path_B_role": B["role"],
          "path_A": path_a and path_a["poses"], "path_B": path_b and path_b["poses"],
          "grasp_stop_pose": A["selected"] and {**A["selected"]["pose_map"], "status": "UNVERIFIED"},
          "place_stop_pose": B["selected"] and {**B["selected"]["pose_map"], "status": "UNVERIFIED"}}
    (out / "waypoints.yaml").write_text(yaml.safe_dump(wy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    val = {"photo": photo["name"], "input_status": pos["status"], "path_A_status": A["status"], "path_B_status": B["status"], "path_B_basis": B["basis"],
           "path_B_role": B["role"], "status_meaning": PC.STATUS,
           "path_A_checks": path_a and {"passed": path_a["collision_checks_passed"], **path_a["checks"], **path_a["nominal_footprint_metrics"]},
           "path_B_checks": path_b and {"passed": path_b["collision_checks_passed"], **path_b["checks"], **path_b["nominal_footprint_metrics"]},
           "cause_A": A["cause"], "cause_B": B["cause"], "positions": pos["status"] + review, "obstacles": "PROVISIONAL",
           "grasp_placement": "UNVERIFIED", "gripper_3d": PC.GRIPPER_3D, "driving": driving}
    (out / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    render(photo, world, rec, res, a_f, b_f, start_f, start_yaw, nominal, boxes, out, m, driving)
    return rec


def render(photo, world, rec, res, a_f, b_f, start_f, start_yaw, nominal, boxes, out, margin, driving):
    """Copy of path_planning_batch_step4b.render with the input status / margin in the labels (the batch figure says PROVISIONAL and 2.0 cm
    for every photo). The diagnostic figure draws the SAME gripper_baseline_2d runs the decision used (no recomputation)."""
    g = world.grid
    to_map = lambda f: (np.asarray(f, float) - [g.x0, g.y0]) / g.res_cm * 6 + [70, 40]   # noqa: E731  render_map: margin 70/40 px, 6 px per cell
    to_img = lambda f: world.frame.floor_to_img(np.asarray(f, float))                 # noqa: E731
    status = rec["input_status"]
    tag = "" if status == "USER_APPROVED" else " (REVIEW_REQUIRED)"
    gsel, psel = res["A"]["selected"], res["B"]["selected"]
    start = {"floor": start_f.tolist(), "yaw": start_yaw, "size": nominal, "color": (200, 200, 200), "dot": (255, 255, 255),
             "label": f"START yaw {start_yaw:.1f} ({status})"}
    poses = [start]
    if gsel:
        poses.append({"floor": gsel["centre_floor_cm"], "yaw": gsel["yaw_map_deg"], "size": nominal, "color": (0, 0, 255), "dot": (0, 0, 255),
                      "label": f"GRASP STOP yaw {gsel['yaw_map_deg']:.0f} UNVERIFIED", "target": a_f.tolist()})
    if psel:
        poses.append({"floor": psel["centre_floor_cm"], "yaw": psel["yaw_map_deg"], "size": nominal, "color": (255, 0, 0), "dot": (255, 0, 0),
                      "label": f"PLACE STOP yaw {psel['yaw_map_deg']:.0f} UNVERIFIED" + (" (CONDITIONAL)" if rec["path_B_status"] == "CONDITIONAL" else ""),
                      "target": b_f.tolist()})
    desc = {k: res[k]["desc"] for k in ("A", "B") if res[k]["desc"]}
    head = f"{photo['current_file']} = {photo['name']} ({photo['earlier_id']}), positions {status}{tag}, unified rules"
    lines = [head,
             f"Path A: {rec['path_A_status']}" + (f", {desc['A']['length_cm']} cm, {desc['A']['n_turns']} turns" if "A" in desc else f" - {rec['cause_A']}"),
             f"Path B: {rec['path_B_status']}" + (f", {desc['B']['length_cm']} cm, {desc['B']['n_turns']} turns" if "B" in desc else f" - {rec['cause_B'] or rec['path_B_basis']}")]
    if rec["path_B_status"] == "CONDITIONAL":
        lines.append("B start: " + rec["path_B_basis"] + " -> DIAGNOSTIC only")
    lines = [w for ln in lines for w in textwrap.wrap(ln, 90, subsequent_indent="    ")]
    lines += ["orange outline: per-photo pallets (STEP 4-A AI reading, PROVISIONAL), obstacles for wheels/body",
              f"margin {margin} cm: {driving.split(' (')[0]}; gripper offset / reach not measured -> UNVERIFIED",
              "magenta dashes: 2D baseline only; not a grasp / placement claim (3D UNVERIFIED)",
              "body overhang beyond the yellow line: unmapped objects other than the pallets -> UNVERIFIED",
              "only computed paths and valid stops are drawn here;", "    rejected / reference stops and gripper baselines: path_planning_diag_*.png"]
    targets = [{"kind": "cube", "floor": a_f.tolist(), "label": f"A cube{tag}"}]
    if b_f is not None:
        targets.append({"kind": "place", "floor": b_f.tolist(), "label": f"B place{tag}"})
    for key in desc:
        fv = desc[key]["checks"]["whole_body_sweep_generic"]["first_violation"]
        if fv:
            targets.append({"kind": "collision", "floor": fv["floor_cm"], "label": f"COLLISION on Path {key}"})
    img = cv2.imread(photo["path"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    PP.render_photo(img, world, {"legs": desc, "poses": poses, "status_lines": lines, "targets": targets}, out / "path_planning_photo.png")
    PP.render_map(world, {"legs": desc, "poses": poses, "status_lines": lines, "targets": targets}, out / "path_planning_map.png")

    # ---- diagnostic: every stop candidate and its gripper baseline (the decision's own runs)
    diag, dposes, dlines = [], [start], [head, "DIAGNOSTIC FIGURE: stop candidates and gripper baselines - NOT paths, NOT stops to drive to",
                                         "baseline colours: green free | grey not drivable | violet no-go (gripper allowed)",
                                         "    orange target's own pallet (allowed) | RED static obstacle / other object (BLOCKING, X)",
                                         "baseline = 2D floor-level centre line; gripper reach / width / height / 3D clearance not measured -> UNVERIFIED", ""]
    for key, lab, sel in (("grasp_candidates", "A", gsel), ("place_candidates", "B", psel)):
        nopose = [c for c in rec[key] if not c.get("centre_floor_cm")]
        if nopose:
            dlines.append(f"{lab} yaw {'/'.join(str(c['yaw_map_deg']) for c in nopose)}: {nopose[0]['status']} (searched {nopose[0].get('searched_cm')} cm)")
        for c in rec[key]:
            d = {"leg": lab, "yaw_map_deg": c["yaw_map_deg"], "status": c["status"], "selected": bool(sel) and c.get("pose_map") == sel.get("pose_map")}
            if not c.get("centre_floor_cm"):
                d["reason"] = f"no collision-free stop on the target axis (searched {c.get('searched_cm')} cm)"
            else:
                gb = c["gripper_baseline_2d"]
                runs = gb["runs"]
                cross = ", ".join(f"{BB.run_name(r)} {r['length_cm']} cm" for r in runs if r["blocking"])
                d.update({"stop_pose_map": c["pose_map"], "body_front_map_m": gb["front_map_m"], "target_map_m": gb["target_map_m"],
                          "front_to_target_gap_cm": c.get("front_to_target_gap_cm"), "blocked_by": gb["blocked_by"], "runs": runs})
                d["reason"] = {"GRIPPER_LINE_BLOCKED": f"rejected: gripper baseline crosses {cross}",
                               "NO_FEASIBLE_LEG": "stop is collision-free but no feasible leg (A* / rotation)",
                               "FEASIBLE_PLAN_GRASP_UNVERIFIED": "selected stop" if d["selected"] else "feasible, not selected (larger gap / longer leg)"}.get(c["status"], c["status"])
                col, dot, short = {"GRIPPER_LINE_BLOCKED": ((80, 0, 80), (220, 160, 255), "REJECTED"), "NO_FEASIBLE_LEG": ((0, 140, 140), (0, 200, 200), "NO LEG")}.get(
                    c["status"], (((0, 0, 255) if lab == "A" else (255, 0, 0)), ((0, 0, 255) if lab == "A" else (255, 0, 0)), "SELECTED" if d["selected"] else "FEASIBLE"))
                dposes.append({"floor": c["centre_floor_cm"], "yaw": c["yaw_map_deg"], "size": nominal, "color": col, "dot": dot, "label": f"{short} {lab} yaw {c['yaw_map_deg']}"})
                d["_draw"] = runs
            if c.get("pose_map"):
                dlines.append(f"{lab} yaw {c['yaw_map_deg']}: {c['status']} at ({c['pose_map']['x_m']}, {c['pose_map']['y_m']}), "
                              f"gap {c.get('front_to_target_gap_cm')} cm - {d['reason']}")
            diag.append(d)
    for ref, lab in ((rec["reference_nearest_stop_A"], "A"), (rec["reference_nearest_stop_B"], "B")):
        if ref:
            f = world.frame.map_to_floor([[ref["pose_map"]["x_m"], ref["pose_map"]["y_m"]]])[0]
            dposes.append({"floor": f.tolist(), "yaw": ref["yaw_map_deg"], "size": nominal, "color": (128, 128, 128), "dot": (128, 128, 128),
                           "label": f"REF ONLY {lab} (not a stop)"})
            dlines.append(f"REF ONLY {lab}: nearest collision-free body pose ({ref['pose_map']['x_m']}, {ref['pose_map']['y_m']}, {ref['yaw_map_deg']}) - gripper not facing the target")
    dlines = [w for ln in dlines for w in (textwrap.wrap(ln, 90, subsequent_indent="    ") or [""])]
    dscene = {"legs": {}, "poses": dposes, "status_lines": dlines, "targets": targets}
    PP.render_photo(img, world, dscene, out / "path_planning_diag_photo.png")
    PP.render_map(world, dscene, out / "path_planning_diag_map.png")
    (out / "gripper_line_diagnostics.json").write_text(json.dumps(
        {"photo": photo["name"], "rule": "unified_planning.yaml gripper_baseline (2D_FLOOR_BASELINE)", "frame": "map_m (REP-103); floor_cm for raw samples",
         "note": "the runs are the ones the decision used (planning_checks.gripper_baseline_2d); gripper reach / width / height / 3D UNVERIFIED",
         "candidates": [{k: v for k, v in d.items() if k != "_draw"} for d in diag],
         "reference_stops": {"A": rec["reference_nearest_stop_A"], "B": rec["reference_nearest_stop_B"]}}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")

    for name, conv in (("path_planning_photo.png", to_img), ("path_planning_map.png", to_map), ("path_planning_diag_photo.png", to_img), ("path_planning_diag_map.png", to_map)):
        im = cv2.imread(str(out / name))
        for b in boxes:
            x0, y0, x1, y1 = b["box_px"]
            cp = conv(world.frame.img_to_floor([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
            cv2.polylines(im, [np.round(cp).astype(np.int32)], True, (0, 165, 255), 3 if conv is to_img else 2)
        if "diag" in name:
            for d in diag:
                for r in d.get("_draw", []):
                    q0, q1 = conv([r["from_floor_cm"], r["to_floor_cm"]])
                    cv2.line(im, PP._ip(q0), PP._ip(q1), BB.LINE_COL[r["layer"]], 5 if r["blocking"] else 3, cv2.LINE_AA)
                    if r["blocking"]:
                        mid = (q0 + q1) / 2
                        cv2.drawMarker(im, PP._ip(mid), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 26, 4)
                        PP._text(im, f"X {BB.run_name(r)} {r['length_cm']} cm", mid + [14, 5], (0, 0, 255), 0.6, 2)
        for tf, ok, txt in ((a_f, bool(gsel), f"Path A {rec['path_A_status']}"), (b_f, bool(psel), f"Path B {rec['path_B_status']}")):
            if tf is not None and not ok:
                PP._text(im, txt, conv([tf])[0] + [8, 16], (0, 0, 255), 0.5, 1)
        cv2.imwrite(str(out / name), im)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", help="comma-separated photo names (default: unified_planning.yaml photos)")
    ap.add_argument("--out-root", help="parent folder of the new run folder (default: results_dir of navigation_config.yaml)")
    args = ap.parse_args()
    cfg, ucfg = CT.load_config(), load_unified()
    names = args.photos.split(",") if args.photos else ucfg["photos"]
    photos = {p["name"]: p for p in CT.resolve_photos(cfg)}
    unknown = [n for n in names if n not in photos]
    if unknown:
        raise ValueError(f"unknown photos {unknown}")
    frame, fc, map_run = OP.load_step2_frame(cfg)
    g = fc["grid"]
    grid = CT.MapGrid(frame, g["extent_floor_cm"], g["resolution_m"])
    pgm, _ = ME.read_pgm(map_run / "maps/map.pgm")
    layers = np.load(map_run / "maps/map_layers.npy")
    urdf = WB.read_urdf_wheels(cfg["wheel_boundary_experiment"]["urdf"])
    nominal = (cfg["robot"]["length_cm"], cfg["robot"]["width_cm"])
    ctx_sv = SV.Ctx(cfg)
    approved = {n: CT.resolve(cfg["approved_positions_dir"]) / f"{n}.yaml" for n in names}
    watched = {**{f"map/{k}": map_run / "maps" / k for k in ("map.pgm", "map.yaml", "map_layers.npy")},
               **{f"photo/{n}": Path(photos[n]["path"]) for n in names}, **{f"approved/{n}": p for n, p in approved.items() if p.exists()},
               "config/navigation_config.yaml": CT.CONFIG_PATH, "config/unified_planning.yaml": UCFG_PATH,
               **{f"code/{c}": HERE / c for c in EXISTING_CODE}, "urdf": Path(cfg["wheel_boundary_experiment"]["urdf"])}
    before = {"v02": code_fingerprint()["sha256"], **{k: sha256_file(v) for k, v in watched.items()}}
    stamp = datetime.now().astimezone()
    parent = (Path(args.out_root) if Path(args.out_root).is_absolute() else CT.resolve(args.out_root)) if args.out_root else CT.resolve(cfg["results_dir"])
    root = parent / (stamp.strftime("%Y%m%d_%H%M%S") + "_unified")
    (root / "maps").mkdir(parents=True, exist_ok=False)
    for k in ("map.pgm", "map.yaml", "map_preview.png"):
        (root / "maps" / k).write_bytes((map_run / "maps" / k).read_bytes())
    rows = []
    for n in names:
        out = root / n
        out.mkdir()
        rec = plan_photo(cfg, ucfg, ctx_sv, photos[n], frame, grid, pgm, layers, urdf, nominal, out)
        row = {"photo": n, "input_status": rec["input_status"], "path_A_status": rec["path_A_status"], "path_B_status": rec["path_B_status"],
               "path_B_basis": rec["path_B_basis"], "path_B_role": rec["path_B_role"], "summary_A": rec["summary_A"], "summary_B": rec["summary_B"],
               "cause_A": rec["cause_A"], "cause_B": rec["cause_B"],
               "grasp_stop": rec["grasp_stop"] and {k: rec["grasp_stop"][k] for k in ("pose_map", "front_to_target_gap_cm", "lateral_offset_cm")},
               "place_stop": rec["place_stop"] and {k: rec["place_stop"][k] for k in ("pose_map", "front_to_target_gap_cm", "lateral_offset_cm")},
               "previous": compare_previous(ucfg, n, out, rec)}
        rows.append(row)
        print(n, rec["input_status"], "A", rec["path_A_status"], "B", rec["path_B_status"], "| previous", row["previous"] and
              (row["previous"]["previous_A"], row["previous"]["previous_B"], row["previous"]["files"], row["previous"]["same_stop_pose"]), flush=True)
    after = {"v02": code_fingerprint()["sha256"], **{k: sha256_file(v) for k, v in watched.items()}}
    unchanged = after == before
    (root / "comparison.json").write_text(json.dumps({"rows": rows, "status_meaning": PC.STATUS, "margin_cm": ucfg["margin_cm"],
                                                      "environment_unchanged_after_run": unchanged}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (root / "config_snapshot.yaml").write_text(CT.CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (root / "unified_planning_snapshot.yaml").write_text(UCFG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    files = sorted(q for q in root.rglob("*") if q.is_file())
    (root / "manifest.json").write_text(json.dumps({
        "experiment_id": ucfg["experiment_id"], "step": "UNIFIED_planning", "run_id": root.name, "created": stamp.isoformat(timespec="seconds"),
        "git_commit": None, "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
        "code_sha256": {p.name: sha256_file(p) for p in (Path(__file__).resolve(), HERE / "planning_checks.py", *(HERE / c for c in EXISTING_CODE))},
        "inputs_sha256": before, "photos": names, "environment_unchanged_after_run": unchanged,
        "hardware_condition": "offline; no model, GPU, ROS or robot", "outputs_sha256": {str(q.relative_to(root)): sha256_file(q) for q in files}}, indent=1), encoding="utf-8")
    print(json.dumps({"run_dir": str(root), "environment_unchanged": unchanged}, indent=1))
    return 0 if unchanged else 1


if __name__ == "__main__":
    sys.exit(main())
