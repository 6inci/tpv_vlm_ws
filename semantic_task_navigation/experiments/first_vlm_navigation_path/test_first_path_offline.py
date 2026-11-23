"""Offline checks (no GPU, no model). From the project root:
  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/first_vlm_navigation_path/test_first_path_offline.py

(1) the new pieces on synthetic inputs, including cases where a check MUST fail; (2) design rules (no answer-side inputs, robot is not an
obstacle); (3) pins of what the stored final run recorded (observed behaviour, not evidence that the path is right for a real robot).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import map_transfer as MT  # noqa: E402
import path_checks as PC  # noqa: E402
import robot_start as RS  # noqa: E402
from semantic_task_nav.config import code_fingerprint, load_config  # noqa: E402
from semantic_task_nav.navigation.goal_selector import select_goal  # noqa: E402
from semantic_task_nav.navigation.map_builder import NavMap  # noqa: E402
from semantic_task_nav.navigation.planner import plan  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
PCFG = load_config()["planning"]
SIZE = (400, 300)


def synth_spec():
    return {"schema_version": "stnav-map-0.1", "map_id": "t", "coordinate_frame": "raw_image_px", "image_size": list(SIZE), "review_status": "DRAFT_UNCONFIRMED",
            "traversable": [{"id": "T", "polygon": [[20, 20], [380, 20], [380, 280], [20, 280]]}],
            "static_obstacles": [{"id": "O", "polygon": [[150, 100], [250, 100], [250, 200], [150, 200]]}], "no_go": [], "start_pose": None}


def synth_nav():
    return NavMap(synth_spec(), SIZE, PCFG, [])


# ------------------------------------------------------------------ map transfer
def test_homography_recovers_a_known_warp():
    img = cv2.imread(CFG["image"])
    if img is None:
        pytest.skip("E1 photo not available")
    H0 = np.array([[1.02, 0.015, -40.0], [-0.01, 1.01, 8.0], [0, 0, 1.0]])
    ref = cv2.warpPerspective(img, H0, (img.shape[1], img.shape[0]))          # ref pixel = H0 * frame pixel
    H, rec = MT.estimate_homography(ref, img, CFG["map_transfer"])
    assert rec["status"] == "ESTIMATED"
    pts = np.float32([[300, 200], [1500, 300], [900, 800], [1700, 700]]).reshape(-1, 1, 2)
    back = cv2.perspectiveTransform(cv2.perspectiveTransform(pts, np.linalg.inv(H0)), H0)   # sanity of the test itself
    assert np.allclose(back, pts, atol=1e-3)
    est = cv2.perspectiveTransform(cv2.perspectiveTransform(pts, H0), H)                    # frame -> ref -> (estimated) frame
    assert np.abs(est - pts).max() < 2.0


def test_transform_is_identity_with_identity_inputs():
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    pts = [[100.0, 200.0], [1800.0, 900.0]]
    assert np.allclose(MT.transform_points(pts, K, np.zeros(5), np.eye(3)), pts, atol=1e-3)


def test_transferred_map_does_not_carry_the_reference_zone_or_the_manual_start():
    src = json.loads((ROOT / "config/maps/reference_scene_map.json").read_text(encoding="utf-8"))
    ref = HERE / "config.yaml"                                                              # any existing file works as the 'reference image' here
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    rec = {k: {} for k in ("method", "n_matches", "n_inliers", "inlier_residual", "holdout")}
    spec = MT.build_spec(src, K, np.zeros(5), np.eye(3), (1920, 1080), ref, rec)
    assert "semantic_reference_zones" not in spec and spec["start_pose"] is None
    assert spec["review_status"] == "DRAFT_UNCONFIRMED" and "MANUAL_MAP_PRIOR" in spec["map_status"]
    ids = [z["id"] for k in ("traversable", "static_obstacles", "no_go") for z in spec[k]]
    assert "Z06" not in ids and "[1402, 99]" not in json.dumps(spec) and "1402" not in json.dumps(spec["traversable"] + spec["static_obstacles"] + spec["no_go"])
    assert "Z06" in " ".join(spec["provenance"]["not_carried_over"])           # recorded as NOT carried over


def test_a_bad_transfer_is_refused():
    rec = {"status": "ESTIMATED", "n_inliers": 40, "inlier_residual": {"rms_px": 5.0, "median_px": 4.0, "p95_px": 9.0},
           "holdout": {k: {"median_px": 9.0, "p95_px": 20.0} for k in ("a", "b", "c", "d")}}
    d = MT.accept(rec, {"status": "MISALIGNED"}, CFG["map_transfer"])
    assert d["accepted"] is False and sum(not c["ok"] for c in d["checks"]) == 5


# ------------------------------------------------------------------ robot start
def test_start_estimates_move_toward_the_principal_point_only():
    box = [1585.0, 620.0, 1814.0, 784.0]
    pp = (905.0, 513.0)
    est, u = RS.start_estimates(box, pp, [0, 20, 40])
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    assert np.allclose(est[0]["xy"], c, atol=0.1)
    d0 = np.linalg.norm(np.array(pp) - c)
    assert np.linalg.norm(np.array(pp) - np.array(est[2]["xy"])) == pytest.approx(d0 - 40, abs=0.2)


def test_candidates_are_taken_from_the_window_in_score_order_and_no_manual_substitute():
    det = {"objects": [{"object_id": "a", "class_name": "mobile_robot", "bbox_xyxy": [100, 100, 200, 200], "score": 0.9},     # outside the window
                       {"object_id": "b", "class_name": "mobile_robot", "bbox_xyxy": [1500, 700, 1600, 800], "score": 0.2},
                       {"object_id": "c", "class_name": "mobile_robot", "bbox_xyxy": [1600, 600, 1800, 800], "score": 0.4},
                       {"object_id": "d", "class_name": "truck", "bbox_xyxy": [1600, 600, 1800, 800], "score": 0.99}]}
    cands = RS.stored_candidates(det, [960, 540, 1920, 1080])
    assert [c["object_id"] for c in cands] == ["c", "b"]
    mcfg = load_config()["models"]
    img = np.zeros((1080, 1920, 3), np.uint8)
    answers = iter(["no", "no"])
    fake = lambda client, mc, im, box, desc, path: {"status": next(answers), "answer": {}, "crop": {}, "call": {}}   # noqa: E731
    robot, recs = RS.verify_in_order(None, mcfg, img, cands, "/nonexistent", fake)
    assert robot is None and [r["status"] for r in recs] == ["no", "no"]          # all 'no' -> no robot, nothing invented
    answers = iter(["yes"])
    robot, recs = RS.verify_in_order(None, mcfg, img, cands, "/nonexistent", fake)
    assert robot["object_id"] == "c" and len(recs) == 1                            # stops at the first yes


# ------------------------------------------------------------------ path / goal checks MUST be able to fail
def test_swept_disc_check_flags_a_path_through_an_obstacle_and_passes_a_clean_one():
    nav = synth_nav()
    bad = PC.swept_overlap(nav, [[60, 150], [340, 150]], 20)
    good = PC.swept_overlap(nav, [[60, 60], [340, 60]], 20)
    assert bad["on_static_obstacle_px"] > 0
    assert all(good[k] == 0 for k in ("on_static_obstacle_px", "on_no_go_px", "on_dynamic_obstacle_px", "outside_traversable_px"))
    off = PC.swept_overlap(nav, [[60, 60], [60, 10]], 15)
    assert off["outside_traversable_px"] > 0


def test_tape_check_flags_a_crossing():
    tape = np.zeros((300, 400), bool)
    tape[:, 200] = True
    assert PC.yellow_crossing([[100, 50], [300, 50]], tape)["centre_line_samples_on_tape"] > 0
    assert PC.yellow_crossing([[100, 50], [180, 50]], tape)["centre_line_samples_on_tape"] == 0


def test_goal_must_not_be_inside_the_bay_or_a_detected_object():
    nav = synth_nav()
    bay = [[50, 40], [120, 40], [120, 90], [50, 90]]
    objs = [{"object_id": "truck_1", "class_name": "truck", "bbox_xyxy": [300, 40, 360, 90]}]
    assert PC.goal_checks(nav, (80, 60), bay, objs)["inside_bay_polygon"] is True
    assert PC.goal_checks(nav, (80, 60), bay, objs)["ok"] is False
    assert PC.goal_checks(nav, (330, 60), bay, objs)["inside_detected_object_boxes"] == ["truck_1"]
    assert PC.goal_checks(nav, (80, 240), bay, objs)["ok"] is True


def test_v02_goal_and_astar_on_the_synthetic_map_avoid_the_obstacle():
    nav = synth_nav()
    bay = [[60, 20], [140, 20], [140, 40], [60, 40]]
    g = select_goal(nav, "bay", bay, (340, 240), "dropoff_approach", PCFG["goal_selection"]["dropoff"], PCFG)
    assert g["status"] == "GOAL_SELECTED"
    p = plan(nav, (340, 240), g["pose"]["xy"], PCFG, "A_start_to_goal")
    assert p["status"] == "PATH_FOUND"
    sw = PC.swept_overlap(nav, p["waypoints_xy"], PCFG["robot_radius_px"] - PCFG["validation_tolerance_px"])
    assert sw["on_static_obstacle_px"] == 0 and sw["outside_traversable_px"] == 0


# ------------------------------------------------------------------ design rules
def test_the_robot_is_never_in_the_obstacle_classes_and_inputs_are_not_answer_side():
    src = (HERE / "run_first_path.py").read_text(encoding="utf-8")
    assert 'pc["dynamic_obstacle_classes"]) + ["traffic_cone"]' in src
    assert '"mobile_robot" in pcfg["dynamic_obstacle_classes"]' in src                  # recorded in every variant as robot_in_obstacle_classes
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"', "'")))
    for forbidden in ("semantic_reference_zones", "start_pose\"]", "(700, 740)", "Z06"):
        assert not [l for l in code.splitlines() if forbidden in l and ("open(" in l or "read_text" in l or "load" in l)], forbidden


def test_v02_code_is_unchanged():
    assert code_fingerprint()["sha256"].startswith("254cfb31da7f")


# ------------------------------------------------------------------ pins of the stored final run
def latest_run():
    runs = sorted(p for p in (HERE / "results").glob("*_first_path") if (p / "result.json").exists())
    return json.loads((runs[-1] / "result.json").read_text(encoding="utf-8")) if runs else None


def test_stored_final_run_pin():
    r = latest_run()
    if r is None:
        pytest.skip("no stored run")
    s = r["stages"]
    assert r["claims"]["robot_task_success"] == "NOT_CLAIMABLE" and r["claims"]["shipping_success"] == "NOT_CLAIMABLE" and r["no_hardware_command"] is True
    assert s["1_qwen_destination"]["function_name"] == "loading dock" and "STORED" in s["1_qwen_destination"]["source"]
    assert s["3_robot_start"]["status"] == "ROBOT_DETECTED" and s["3_robot_start"]["robot"]["source"].endswith("Qwen crop check yes")
    assert s["4_map"]["decision"]["accepted"] is True and s["4_map"]["homography"]["n_inliers"] > 1000
    v = s["5_planning"]["variants"]
    pri = v[s["5_planning"]["primary"]]
    assert pri["robot_radius_status"].startswith("ASSUMED") and pri["start_checks"]["robot_in_obstacle_classes"] is False
    assert all(x == 0 for k, x in pri["start_checks"]["robot_box_overlap_px"].items() if k != "traversable")     # the robot is not on any obstacle layer
    assert pri["verdict"] == {"goal_selected": True, "path_found": True, "v02_validator": True, "independent": True}
    assert pri["goal"]["pose"]["xy"] == [1348.0, 332.0] and pri["path"]["length_px"] == pytest.approx(1014.47, abs=0.01)
    stress = next(x for k, x in v.items() if k.startswith("stress"))
    assert stress["goal"]["status"] == "NO_ACCESSIBLE_CANDIDATE"                   # with the image-inscribed radius the assumed reach finds no goal
