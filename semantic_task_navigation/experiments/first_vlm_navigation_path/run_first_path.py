#!/usr/bin/env python3
"""First real VLM -> approach-path run on the E1 photo (image space, no hardware).

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
  $PY experiments/first_vlm_navigation_path/run_first_path.py                      # one Qwen crop check for the robot, then everything offline
  $PY experiments/first_vlm_navigation_path/run_first_path.py --reuse-verification results/<run>/robot_verification.json   # no model call

Reused unchanged: v0.2 NavMap, select_goal, plan (A*), validate_path, grounding.verify_box, check_alignment. Stored, not recomputed: the Qwen
L2 reply, Grounding DINO truck/conveyor/cones, the painted-bay localization. New here: map transfer, robot start, extra checks, figures.
Scope: current robot position -> safe APPROACH point of the loading bay. No pickup, grasp or placement; no success claim for shipping.
"""
import argparse
import copy
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
sys.path.insert(0, str(HERE))
import map_transfer as MT  # noqa: E402
import path_checks as PC  # noqa: E402
import render as RD  # noqa: E402
import robot_start as RS  # noqa: E402
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file  # noqa: E402
from semantic_task_nav.navigation.goal_selector import select_goal  # noqa: E402
from semantic_task_nav.navigation.map_builder import NavMap, dynamic_obstacles_from_detections  # noqa: E402
from semantic_task_nav.navigation.path_validator import compare_raw_and_simplified, validate_path  # noqa: E402
from semantic_task_nav.navigation.planner import plan  # noqa: E402

ANCHOR_ID = "loading_bay_candidate"
CLAIMS = {"robot_task_success": "NOT_CLAIMABLE", "shipping_success": "NOT_CLAIMABLE",
          "scope": "current robot position -> approach point of the loading bay only (no pickup, grasp or placement)",
          "metric_valid": False, "map_mode": "IMAGE_SPACE (raw_image_px of the undistorted E1 photo; not metric)"}


def load_cfg():
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def load_stored(cfg, size):
    """The stored Qwen reply, the DINO-checked objects and the Python-computed bay. Nothing is recomputed."""
    q = (HERE / cfg["stored"]["qwen_run"]).resolve()
    c = (HERE / cfg["stored"]["cone_run"]).resolve()
    b = (HERE / cfg["stored"]["bay_localization"]).resolve()
    structured = json.loads((q / "vlm_proposal/structured.json").read_text(encoding="utf-8"))
    area = next(a for a in structured["functional_areas"] if a["id"] == cfg["stored"]["proposal_area_id"])
    ta = structured["task_analysis"]
    w, h = size
    qbox = [area["bbox_2d"][0] / 1000 * w, area["bbox_2d"][1] / 1000 * h, area["bbox_2d"][2] / 1000 * w, area["bbox_2d"][3] / 1000 * h]
    det = json.loads((q / "analysis/detections.json").read_text(encoding="utf-8"))
    cone_boxes = json.loads((c / "integration_result.json").read_text(encoding="utf-8"))["verification"]["cones"]["traffic_cone"]["verified"]
    loc = json.loads((b / "localization.json").read_text(encoding="utf-8"))
    bprov = json.loads((b / "input_provenance.json").read_text(encoding="utf-8"))
    objects = [dict(o, accepted=True) for o in det["objects"] if o["object_id"] in ("truck_1", "conveyor_1") and o["accepted"]]
    objects += [{"object_id": f"traffic_cone_{i}", "class_name": "traffic_cone", "bbox_xyxy": bx, "accepted": True, "source": "grounding_dino",
                 "knowledge": "observed_by_model", "basis": f"Grounding DINO + Qwen crop check 'yes' [stored run {c.name}]"}
                for i, bx in enumerate(cone_boxes, 1)]
    return {"qwen_run": q, "cone_run": c, "bay_run": b, "structured": structured, "area": area, "task_analysis": ta, "qwen_box_px": qbox,
            "detections": det, "objects": objects, "bay": loc, "bay_prov": bprov,
            "raw_response_sha256": sha256_file(q / "vlm_proposal/raw_response.txt"),
            "bay_polygon": loc["bay_polygon_xy"], "bay_status": loc["status"]}


def serialisable_goal(g):
    g = dict(g)
    cells = g.pop("accessible_cells", [])
    g["accessible_cells_count"] = len(cells)
    return g


def plan_variant(label, spec, objs, size, pcfg, start_xy, bay_poly, yellow, robot_box, tol_note):
    """One planning variant with the v0.2 modules. Returns (record, runtime objects for the figures)."""
    dyn = dynamic_obstacles_from_detections(objs, pcfg["dynamic_obstacle_classes"], spec, size)
    nav = NavMap(spec, size, pcfg, dyn)
    goal = select_goal(nav, ANCHOR_ID, bay_poly, start_xy, "dropoff_approach", pcfg["goal_selection"]["dropoff"], pcfg)
    rec = {"label": label, "robot_radius_px": pcfg["robot_radius_px"], "robot_radius_status": "ASSUMED (image-space, not measured)",
           "start_xy": start_xy, "n_dynamic_obstacles_in_map": len(dyn), "dynamic_obstacle_ids": [d["id"] for d in dyn],
           "map_summary": nav.summary(), "goal": serialisable_goal(goal), "path": None, "v02_validation": None, "independent_checks": None}
    # the robot must not be an obstacle, and the start must be a valid robot cell
    rec["start_checks"] = {"robot_in_obstacle_classes": "mobile_robot" in pcfg["dynamic_obstacle_classes"],
                           "robot_box_overlap_px": {k: PC.box_overlap_px(robot_box, nav.full_layers[k])
                                                    for k in ("traversable", "static_obstacle", "no_go", "dynamic_obstacle")}}
    r0, c0 = nav.cell_of(*start_xy)
    rec["start_checks"].update({"cell": [r0, c0], "in_traversable": bool(nav.layers["traversable"][r0, c0]),
                                "valid_robot_cell": bool(nav.valid[r0, c0]), "clearance_px": round(float(nav.clearance_full[int(start_xy[1]), int(start_xy[0])]), 1)})
    trav_dist = cv2.distanceTransform((~nav.full_layers["traversable"]).astype(np.uint8), cv2.DIST_L2, 5)
    rec["obstacle_reflection"] = []
    for o in objs + [{"object_id": "mobile_robot_1", "class_name": "mobile_robot", "bbox_xyxy": robot_box}]:
        x1, y1, x2, y2 = (int(round(v)) for v in o["bbox_xyxy"])
        ov = PC.box_overlap_px(o["bbox_xyxy"], nav.full_layers["traversable"])
        rec["obstacle_reflection"].append({
            "object_id": o["object_id"], "class": o["class_name"], "overlap_with_drivable_px": ov,
            "gap_to_drivable_px": round(float(trav_dist[max(0, y1):y2, max(0, x1):x2].min()), 1),
            "as_obstacle_in_grid": o["object_id"] in rec["dynamic_obstacle_ids"],
            "meaning": ("the robot itself: deliberately NOT an obstacle" if o["class_name"] == "mobile_robot" else
                        "in the obstacle layer" if o["object_id"] in rec["dynamic_obstacle_ids"] else
                        "outside the drivable area (no overlap): already non-free, so the goal/path cannot enter it"
                        if ov == 0 else "overlaps the drivable area but was not added (check)")})
    verdict = {"goal_selected": goal["status"] == "GOAL_SELECTED", "path_found": False, "v02_validator": None, "independent": None}
    runtime = {"nav": nav, "goal": goal, "path": None}
    if verdict["goal_selected"]:
        gxy = goal["pose"]["xy"]
        path = plan(nav, start_xy, gxy, pcfg, "A_start_to_goal")
        rec["path"] = path
        runtime["path"] = path
        verdict["path_found"] = path["status"] == "PATH_FOUND"
        gc = PC.goal_checks(nav, gxy, bay_poly, [o for o in objs if o["class_name"] != "mobile_robot"])
        rec["goal_checks"] = gc
        if verdict["path_found"]:
            v = validate_path(nav, path["waypoints_xy"], path["start_xy"], path["goal_xy"], pcfg)
            v["raw_path_check"] = validate_path(nav, path["raw_waypoints_xy"], path["start_xy"], path["goal_xy"], pcfg)["valid"]
            v["raw_vs_simplified"] = compare_raw_and_simplified(path)
            rec["v02_validation"] = v
            verdict["v02_validator"] = v["valid"]
            r = pcfg["robot_radius_px"]
            tol = pcfg["validation_tolerance_px"]
            wp = path["waypoints_xy"]
            sw_tol, sw_full = PC.swept_overlap(nav, wp, r - tol), PC.swept_overlap(nav, wp, r)
            yc = PC.yellow_crossing(wp, yellow)
            ind = {"swept_disc_at_radius_minus_tolerance": sw_tol, "swept_disc_at_full_radius_informational": sw_full,
                   "tolerance_px": tol, "tolerance_note": tol_note, "yellow_tape": yc, "tape_overlap_of_footprint": PC.tape_overlap_of_footprint(wp, yellow, r),
                   "goal": gc}
            ind["pass"] = bool(all(sw_tol[k] == 0 for k in ("on_static_obstacle_px", "on_no_go_px", "on_dynamic_obstacle_px", "outside_traversable_px"))
                               and yc["centre_line_samples_on_tape"] == 0 and gc["ok"])
            rec["independent_checks"] = ind
            verdict["independent"] = ind["pass"]
    rec["verdict"] = verdict
    return rec, runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-verification", help="results/<run>/robot_verification.json from an earlier run: no model call")
    args = ap.parse_args()
    cfg, v02 = load_cfg(), load_config()
    pcfg0 = v02["planning"]
    img_path = Path(cfg["image"])
    assert sha256_file(img_path) == cfg["image_sha256"], "image differs from the declared one"
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    size = (W, H)
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_first_path")
    out.mkdir(parents=True, exist_ok=False)
    stages = {}

    # ---- STEP 1-2: stored Qwen proposal + stored DINO objects + stored bay
    st = load_stored(cfg, size)
    ta = st["task_analysis"]
    stages["1_qwen_destination"] = {
        "source": "STORED Qwen3-VL L2 reply (no new semantic inference)", "run": st["qwen_run"].name,
        "raw_response_sha256_16": st["raw_response_sha256"][:16], "instruction": cfg["instruction"],
        "chosen_area_id": ta["destination_area_id"], "function_name": st["area"]["function_name"], "box_px": [round(v, 1) for v in st["qwen_box_px"]],
        "observed_evidence": st["area"]["observed_evidence"], "reason": ta["reason"], "decision": ta["decision"],
        "qwen_confidence": st["area"].get("confidence"),
        "also_listed_by_qwen": [{"id": a["id"], "function_name": a["function_name"], "bbox_2d": a["bbox_2d"]} for a in st["structured"]["functional_areas"] if a["id"] != st["area"]["id"]],
        "ok": ta["destination_area_id"] == cfg["stored"]["proposal_area_id"] and ta["decision"] == "proposed"}
    stages["2_bay_and_objects"] = {
        "bay_source": f"STORED Python localization {st['bay_run'].name}: {st['bay_status']}", "bay_polygon_xy": st["bay_polygon"],
        "bay_matches_stored_qwen_box": [round(v, 1) for v in st["bay_prov"]["qwen_proposal"]["box_px"]] == [round(v, 1) for v in st["qwen_box_px"]],
        "objects_reused": [{"object_id": o["object_id"], "class": o["class_name"], "bbox_xyxy": [round(v, 1) for v in o["bbox_xyxy"]]} for o in st["objects"]],
        "ok": st["bay_status"] == "FLOOR_MARKING_ESTABLISHED"}

    # ---- STEP 3: robot (DINO candidate stored by v0.2, Qwen crop check = the single new model call)
    window = cfg["robot"]["search_window_xyxy"]
    cands = RS.stored_candidates(st["detections"], window, cfg["robot"]["stored_candidate_class"])
    robot = None
    s3 = {"search_window_xyxy": window, "window_source": "coarse region from the task text ('lower right'); not a coordinate of the answer",
          "stored_candidates_in_window": [{"object_id": c["object_id"], "bbox_xyxy": c["bbox_xyxy"], "score": c["score"], "v02_status": c["verification"]["status"]} for c in cands]}
    if args.reuse_verification:
        prev = json.loads(Path(args.reuse_verification).read_text(encoding="utf-8"))
        s3["verification"] = {"reused_from": str(args.reuse_verification), "model_call_made_in_this_run": False, **prev}
        recs = prev["checks"]
        yes = next((r for r in recs if r["status"] == "yes"), None)
        robot = next((c for c in cands if yes and c["object_id"] == yes["candidate_id"]), None)
    else:
        from semantic_task_nav.perception import grounding  # noqa: E402
        from semantic_task_nav.perception.model_client import ModelWorkerClient  # noqa: E402
        client = ModelWorkerClient(v02["models"], out / "worker_logs")
        client.start()
        try:
            robot, recs = RS.verify_in_order(client, v02["models"], img, cands, out / "robot_crops", grounding.verify_box)
        finally:
            client.close()
        for r in recs:
            r["crop"] = {k: (str(Path(v)) if k == "path" else v) for k, v in r["crop"].items()}
        s3["verification"] = {"model_call_made_in_this_run": True, "model": v02["models"]["qwen"]["model_id"], "checks": recs}
        (out / "robot_verification.json").write_text(json.dumps({"checks": recs}, indent=2, ensure_ascii=False), encoding="utf-8")
    K, D, _ = MT.load_calibration(v02["camera"]["calibration"]["intrinsics_file"])
    if robot is not None:
        box = robot["bbox_xyxy"]
        starts, direction = RS.start_estimates(box, (K[0, 2], K[1, 2]), cfg["robot"]["start_shift_toward_principal_point_px"])
        s3.update({"status": "ROBOT_DETECTED", "robot": {"object_id": robot["object_id"], "bbox_xyxy": box, "detector_score": robot["score"],
                                                         "source": "grounding_dino 'mobile robot.' + Qwen crop check yes"},
                   "start_candidates": starts, "direction_to_principal_point_unit": direction, "principal_point_px": [round(K[0, 2], 1), round(K[1, 2], 1)],
                   "footprint": RS.footprint_record(box, pcfg0["robot_radius_px"]),
                   "separate_from_stack": "box lies below the yellow storage-row line; verified box does not overlap the stored goods row polygon (see start_checks)"})
        robot_obj = {"object_id": "mobile_robot_1", "class_name": "mobile_robot", "bbox_xyxy": box, "accepted": True, "source": "grounding_dino",
                     "knowledge": "observed_by_model"}
    else:
        s3.update({"status": "ROBOT_NOT_DETECTED", "note": "no manual start is substituted; planning stages are not run"})
        robot_obj = None
    stages["3_robot_start"] = s3

    # ---- STEP 4: map prior transferred into the E1 frame (or refused)
    spec, mrec = MT.run(cfg, ROOT / "config/maps/reference_scene_map.json", v02["camera"]["calibration"]["intrinsics_file"],
                        ROOT / "config/maps/reference_scene_map.reference_P01.jpg", img, out)
    yellow = MT.yellow_mask(img, cfg["yellow"])
    if spec is not None:
        mrec["edge_offsets_informational"] = MT.edge_offsets(spec, yellow, cfg["map_transfer"]["yellow_edge_check"])
        (out / "map_E1_transferred.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    stages["4_map"] = mrec

    # ---- STEP 5-6: goal selection + A* + validation (v0.2 modules), several documented variants
    variants, runtime = {}, {}
    blocked = [k for k, ok in (("qwen_destination", stages["1_qwen_destination"]["ok"]), ("bay", stages["2_bay_and_objects"]["ok"]),
                               ("robot_start", robot is not None), ("map", spec is not None)) if not ok]
    if not blocked:
        pc = copy.deepcopy(pcfg0)
        pc["dynamic_obstacle_classes"] = list(pc["dynamic_obstacle_classes"]) + ["traffic_cone"]   # v0.2 default lacks cones (documented)
        objs_nav = st["objects"]                                                                  # the robot is deliberately not in this list
        robot_box = robot["bbox_xyxy"]
        fp = RS.footprint_record(robot_box, pcfg0["robot_radius_px"])
        r_stress = fp["planning_discs"]["image_inscribed_radius_px"]["radius_px"]
        tol_note = "v0.2 validation_tolerance_px (grid discretisation); the same convention as path_validator"
        plan_list = [("primary_assumed_r%d_start_shift0" % pc["robot_radius_px"], pc["robot_radius_px"], 0)]
        plan_list += [("start_shift%d_r%d" % (s["shift_px"], pc["robot_radius_px"]), pc["robot_radius_px"], s["shift_px"])
                      for s in s3["start_candidates"] if s["shift_px"] != 0]
        plan_list += [("stress_inscribed_r%d_start_shift0" % r_stress, r_stress, 0)]
        for label, radius, shift in plan_list:
            p = copy.deepcopy(pc)
            p["robot_radius_px"] = radius
            sxy = next(s["xy"] for s in s3["start_candidates"] if s["shift_px"] == shift)
            variants[label], runtime[label] = plan_variant(label, spec, objs_nav, size, p, sxy, st["bay_polygon"], yellow, robot_box, tol_note)
            variants[label]["start_shift_px"] = shift
        stress = plan_list[-1][0]
        sg = variants[stress]["goal"]
        if sg["status"] == "NO_ACCESSIBLE_CANDIDATE" and sg.get("stats", {}).get("nearest_valid_cell_to_anchor_px") is not None:
            p = copy.deepcopy(pc)
            p["robot_radius_px"] = r_stress
            relaxed = sg["stats"]["nearest_valid_cell_to_anchor_px"] + cfg["stress_reach_relax_cells"] * p["cell_px"]
            p["goal_selection"]["dropoff"]["max_dist_px"] = relaxed
            label = "diagnostic_r%d_reach_relaxed_to_%dpx" % (r_stress, relaxed)
            sxy = next(s["xy"] for s in s3["start_candidates"] if s["shift_px"] == 0)
            variants[label], runtime[label] = plan_variant(label, spec, objs_nav, size, p, sxy, st["bay_polygon"], yellow, robot_box, tol_note)
            variants[label]["start_shift_px"] = 0
            variants[label]["diagnostic_note"] = ("DIAGNOSTIC ONLY: collision radius unchanged; the (unmeasured, assumed) reach was relaxed from %s to %s px "
                                                  "to show what reach a robot of the image-inscribed size would need" % (pc["goal_selection"]["dropoff"]["max_dist_px"], relaxed))
        primary = plan_list[0][0]
    else:
        primary = None
    stages["5_planning"] = {"blocked_by": blocked, "primary": primary, "variants": variants}

    # ---- success criteria (the first 7 lines of the task)
    pv = variants.get(primary, {}) if primary else {}
    vd = pv.get("verdict", {})
    crit = {
        "1_stored_qwen_result_used": stages["1_qwen_destination"]["ok"],
        "2_approach_goal_computed": bool(vd.get("goal_selected")),
        "3_robot_start_confirmed_by_detection": robot is not None,
        "4_drivable_area_and_obstacles_in_same_frame": spec is not None and bool(pv),
        "5_astar_path_generated": bool(vd.get("path_found")),
        "6_path_validated_independently": bool(vd.get("v02_validator") and vd.get("independent")),
        "7_visualised_on_original_photo": None}
    ctx = {"image": img, "qwen_box": st["qwen_box_px"], "bay_polygon": st["bay_polygon"], "objects": st["objects"], "robot": robot_obj,
           "spec": spec, "yellow": yellow, "variants": variants, "runtime": runtime, "primary": primary, "stages": stages,
           "start_candidates": s3.get("start_candidates"), "map_record": mrec, "free_floor": st["bay"].get("free_floor")}
    figs = RD.render_all(ctx, out)
    crit["7_visualised_on_original_photo"] = bool(figs)

    result = {"experiment_id": cfg["experiment_id"], "created": datetime.now().isoformat(timespec="seconds"), "claims": CLAIMS,
              "image": {"path": str(img_path), "sha256": cfg["image_sha256"], "size": [W, H]}, "instruction": cfg["instruction"],
              "success_criteria": crit, "all_criteria_met": all(crit.values()), "stages": stages, "figures": figs,
              "v02_code_fingerprint": code_fingerprint()["sha256"], "no_hardware_command": True,
              "not_used_as_input": ["Z06 / semantic_reference_zones", "the manual start (700, 740) of the v0.2 map", "any typed goal or bay coordinate",
                                    "a new Qwen semantic inference or a new prompt"]}
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out / "manifest.json").write_text(json.dumps({"files_sha256": files, "config_sha256": sha256_file(HERE / "config.yaml")}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({"criteria": crit, "blocked_by": blocked, "robot": s3["status"],
                      "map_accepted": mrec["decision"]["accepted"],
                      "variants": {k: {"verdict": v["verdict"], "goal_status": v["goal"]["status"],
                                       "goal_xy": (v["goal"].get("pose") or {}).get("xy"), "length_px": (v["path"] or {}).get("length_px")}
                                   for k, v in variants.items()}}, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
