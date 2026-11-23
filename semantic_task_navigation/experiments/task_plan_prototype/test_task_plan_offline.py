"""Offline checks (no GPU, no model). From the project root:
  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/task_plan_prototype

(1) floor transform on a synthetic grid with a known homography, and refusal without evidence; (2) rectangular footprint: the corridor test depends on the
heading, the independent sweep CAN fail; (3) design rules (label-free target rule, reach not assumed); (4) pins of the latest run: what it recorded and
that nothing is claimed beyond the plan (observed behaviour on one photo, not evidence of accuracy)."""
import inspect
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
import floor_transform as FT  # noqa: E402
import footprint_nav as FN  # noqa: E402
import plan_task as PT  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ (1) floor transform
W2I = np.array([[3.2, 0.4, 150.0], [0.1, 2.6, 120.0], [1e-4, 4e-4, 1.0]])           # synthetic world(cm) -> image


def to_img(pts):
    return FT.project(W2I, pts)


def grid_image():
    img = np.full((900, 1400, 3), 90, np.uint8)
    for i in range(4):
        cv2.polylines(img, [np.round(to_img([[105 * i, y] for y in np.linspace(-10, 220, 60)])).astype(np.int32)], False, (10, 10, 10), 2, cv2.LINE_AA)
    for j in range(3):
        cv2.polylines(img, [np.round(to_img([[x, 105 * j] for x in np.linspace(-10, 325, 60)])).astype(np.int32)], False, (10, 10, 10), 2, cv2.LINE_AA)
    return img


def synthetic_fcfg(offset=4.0):
    seeds = {}
    for i in range(4):
        for j in range(3):
            c = to_img([[105 * i, 105 * j]])[0]
            v = to_img([[105 * i, 105 * j + 10]])[0] - to_img([[105 * i, 105 * j - 10]])[0]
            h = to_img([[105 * i + 10, 105 * j]])[0] - to_img([[105 * i - 10, 105 * j]])[0]
            seeds[f"{i},{j}"] = {"xy": [c[0] + offset, c[1] - offset], "v_dir": v.tolist(), "h_dir": h.tolist(), "v_side": 0, "h_side": 0}
    p0, p1 = to_img([[105, 20]])[0], to_img([[105, 190]])[0]
    return {**CFG["floor"], "grid_seeds": seeds, "seam_checks": [{"id": "V1", "axis": "x", "world_cm": 105, "p0": p0.tolist(), "p1": p1.tolist(), "inlier_px": 3.0}]}


def test_floor_transform_recovers_a_known_homography_from_perturbed_seeds():
    r = FT.fit_floor_transform(grid_image(), synthetic_fcfg())
    assert r["accepted"] and r["n_nodes_used"] == 12
    assert r["fit_rms_cm"] < 0.5 and r["seam_max_abs_dev_cm"] < 0.5
    pts = [[30, 40], [200, 150], [300, 20]]
    back = FT.project(r["H_img_to_floor_cm"], to_img(pts))
    assert np.abs(back - np.array(pts)).max() < 0.5                      # world -> image (truth) -> floor (fit) returns the world point


def test_floor_transform_refuses_without_seam_evidence():
    r = FT.fit_floor_transform(np.full((900, 1400, 3), 90, np.uint8), synthetic_fcfg())
    assert not r["accepted"] and r["H_img_to_floor_cm"] is None


# ------------------------------------------------------------------------------------------------ (2) rectangular footprint
S = 4


def corridor_spec(height_cm, length_cm=200, pad=10):
    poly = [[0, 0], [length_cm, 0], [length_cm, height_cm], [0, height_cm]]
    size = (round((length_cm + 2 * pad) * S), round((height_cm + 2 * pad) * S))
    sp = {"traversable": [{"id": "Z", "polygon": (np.array(poly) + pad).__mul__(S).tolist()}], "static_obstacles": [], "no_go": [], "image_size": list(size)}
    return sp, {"traversable": [{"id": "Z", "polygon": poly}], "static_obstacles": [], "no_go": []}, size


def navmap(height_cm, ext):
    sp, _, size = corridor_spec(height_cm)
    pcfg = PT.v02_pcfg(PT.load_config(), {**CFG["bev"]}, CFG["approach"])
    return FN.RectNavMap(sp, size, pcfg, ext, S)


@pytest.mark.parametrize("height,heading,expect", [(30, 0, False), (30, 90, False), (40, 0, True), (40, 90, False), (45, 0, True), (45, 90, True)])
def test_corridor_validity_depends_on_footprint_and_heading(height, heading, expect):
    ext = FN.footprint_dims_cm(42, 31, 0, heading)                       # heading 0: 42 x 31 cm along x,y; heading 90: 31 x 42 (y extent 42 needs a corridor > 42 cm)
    assert bool(navmap(height, ext).valid.any()) == expect


def test_footprint_dims_swap_with_heading_and_never_shrink():
    assert FN.footprint_dims_cm(35, 24, 3.5, 0) == (42.0, 31.0) and FN.footprint_dims_cm(35, 24, 3.5, 90) == (31.0, 42.0)
    assert min(FN.footprint_dims_cm(35, 24, 0, 0)) == 24


def test_independent_sweep_can_fail_and_is_not_the_planner():
    sp, spec_cm, size = corridor_spec(40)
    spec_cm["static_obstacles"] = [{"id": "O", "polygon": [[90, 0], [110, 0], [110, 18], [90, 18]]}]
    free = FN.free_mask_cm(spec_cm, (200, 40), 10, S)
    ext = (42.0, 31.0)
    clear = FN.sweep_check(free, [[30, 29], [170, 29]], ext, 10, S, 0.25)            # 31 cm tall body at y 13.5..44.5: leaves the 40 cm corridor and hits the obstacle
    ext_ok = (42.0, 20.0)
    assert not clear["ok"]
    assert FN.sweep_check(free, [[30, 29], [170, 29]], ext_ok, 10, S, 0.25)["ok"]     # 20 cm tall body at y 19..39 passes below the obstacle (y 0..18)
    assert not FN.sweep_check(free, [[30, 10], [170, 10]], ext_ok, 10, S, 0.25)["ok"]  # same body through the obstacle fails
    assert FN.body_free(free, (-50, 10), ext_ok, 10, S)[0] is False                  # outside the canvas is blocked


# ------------------------------------------------------------------------------------------------ (3) design rules
def test_target_rule_is_label_free_and_needs_agreement():
    assert "label" not in inspect.signature(PT.pick_target).parameters
    one = {"candidates": [{"candidate_id": "c1", "sources": ["hsv_color_shape"]}]}
    many = {"candidates": [{"candidate_id": "c1", "sources": ["hsv_color_shape", "grounding_dino_roi"]}]}
    assert PT.pick_target(one, CFG["target_candidate"]) is None
    assert PT.pick_target(many, CFG["target_candidate"])["candidate_id"] == "c1"
    assert PT.pick_target({"candidates": []}, CFG["target_candidate"]) is None


def test_stored_e1_gives_the_expected_candidate_and_its_gate_answer_is_no():
    e1 = (HERE / CFG["stored"]["e1_run"]).resolve()
    ti = json.loads((e1 / "analysis/target_inspection.json").read_text(encoding="utf-8"))
    c = PT.pick_target(ti, CFG["target_candidate"])
    assert c["candidate_id"] == CFG["target_candidate"]["expected_candidate_id"]
    assert c["verification"]["status"] == "no" and ti["status"] == "TARGET_NOT_FOUND"


def test_approach_goal_reports_the_distance_to_bridge_and_makes_no_reach_claim():
    sp, spec_cm, size = corridor_spec(60)
    pcfg = PT.v02_pcfg(PT.load_config(), CFG["bev"], CFG["approach"])
    nav = FN.RectNavMap(sp, size, pcfg, (42.0, 31.0), S)
    target_cm = [[-12, 25], [-8, 25], [-8, 29], [-12, 29]]                           # small target just outside the left end of the corridor
    target_bev = ((np.array(target_cm) + 10) * S).tolist()
    start = ((np.array([[150.0, 30.0]]) + 10) * S)[0].tolist()
    g = PT.approach_goal(nav, "t", target_bev, start, "pickup_approach", pcfg, 6 * S, S, 10)
    assert g["status"] == "GOAL_SELECTED" and g["reach"].startswith("NOT_A_REACH_CLAIM")
    assert g["nearest_reachable_centre_to_target_cm"] == pytest.approx(21 + 8, abs=2)   # half length 21 cm + 8 cm to the target edge


def test_v02_planning_values_are_not_reused_as_reach():
    p = PT.v02_pcfg(PT.load_config(), CFG["bev"], CFG["approach"])
    assert p["goal_selection"]["pickup"]["max_dist_px"] is None and p["goal_selection"]["dropoff"]["max_dist_px"] is None
    assert p["cell_px"] == CFG["bev"]["cell_px"] and p["safety_margin_px"] == CFG["bev"]["safety_margin_px"]
    assert p["validation_tolerance_px"] == CFG["bev"]["validation_tolerance_px"] != PT.load_config()["planning"]["validation_tolerance_px"]   # v0.2: 8 image px, here 0.5 cm


# ------------------------------------------------------------------------------------------------ (4) pins of the latest run
def latest():
    runs = sorted((HERE / CFG["results_dir"]).glob("*_task_plan"))
    assert runs, "run plan_task.py first"
    return runs[-1], json.loads((runs[-1] / "plan.json").read_text(encoding="utf-8"))


def test_latest_run_claims_nothing_beyond_the_plan():
    _, r = latest()
    assert r["claims"]["robot_task_success"] == "NOT_CLAIMABLE" and r["claims"]["grasp_or_placement_success"] == "NOT_CLAIMABLE" and r["claims"]["detection_accuracy"] == "NOT_COMPUTED"
    assert r["no_hardware_command"] and r["v02_code_fingerprint_before"] == r["v02_code_fingerprint_after"] and r["v02_code_fingerprint_before"].startswith("254cfb31da7f")
    assert r["floor_transform"]["status"] == "PROVISIONAL_UNVALIDATED" and "NOT USED" in r["floor_transform"]["legacy_2026_09_06_homography"]
    f = r["status_flags"]
    assert f["1_target_input"]["state"] == "PROVISIONAL_TARGET" and f["1_target_input"]["verified_detection"] is False and f["1_target_input"]["stored_qwen_crop_gate"] == "no"
    assert f["3_approach_feasibility_verified"]["verified"] is False and "VIRTUAL" in f["4_pickup_and_placement"] and len(f["6_required_physical_information"]) >= 5
    tc = r["inputs"]["target_candidate"]
    assert tc["stored_qwen_crop_gate"]["answer"]["observed_object"] == "round object" and tc["v02_decision_for_this_photo"] == "TARGET_NOT_FOUND"


def test_latest_run_virtual_steps_are_never_success_and_execution_stays_unconfirmed():
    _, r = latest()
    assert r["virtual_steps"], "primary variant produced no plan"
    for s in r["virtual_steps"]:
        assert s["status"] in ("PLANNED_NOT_EXECUTED", "SIMULATED_NOT_EXECUTED", "PLANNED_ASSUMING_PICKUP")
    ex = r["task_execution_simulated"]
    assert ex["state"] == "WAITING_FOR_PICKUP_CONFIRMATION" and ex["completed"] is False and ex["hardware_enabled"] is False
    assert all(q["hardware_executable"] is False for q in ex["requests"])


def test_latest_run_footprint_is_never_smaller_than_the_user_size_plus_the_measured_transform_error():
    _, r = latest()
    ft, fp = r["floor_transform"], r["robot_footprint"]
    assert fp["transform_margin_per_side_cm"] >= ft["seam_max_abs_dev_cm"]
    for h, ext in fp["primary_extent_cm_by_heading"].items():
        assert min(ext) >= CFG["robot"]["width_cm"] + 2 * ft["seam_max_abs_dev_cm"] and max(ext) >= CFG["robot"]["length_cm"] + 2 * ft["seam_max_abs_dev_cm"]
    for leg in list(r["plan_variants"]["conservative_margin"]["legs_A"].values()) + list(r["plan_variants"]["conservative_margin"]["legs_B"].values()):
        if leg["path"]:
            assert leg["footprint_extent_xy_cm"] in ([42.0, 31.0], [31.0, 42.0])


def test_latest_run_feasibility_never_returns_when_the_footprint_grows():
    _, r = latest()
    ok = [x["selected"] == "PATH_A_AND_B" for x in r["margin_sweep"]]
    assert ok[0], "nominal footprint must be plannable for the sweep to mean anything"
    first_fail = ok.index(False) if False in ok else len(ok)
    assert not any(ok[first_fail:]), f"non-monotone sweep: {ok}"


def test_latest_primary_paths_pass_a_fresh_whole_body_sweep_and_fail_when_the_body_is_much_bigger():
    run, r = latest()
    p = r["plan_variants"]["conservative_margin"]
    assert p["selected"]["status"] == "PATH_A_AND_B"
    spec_cm = {k: [{"id": z["id"], "polygon": z["polygon"]} for z in v] for k, v in r["inputs"]["map_prior"]["zones_cm"].items()}
    free = FN.free_mask_cm(spec_cm, (315, 210), CFG["bev"]["pad_cm"], CFG["bev"]["check_px_per_cm"])
    for tag, legs, h in (("A", p["legs_A"], p["selected"]["A_heading_deg"]), ("B", p["legs_B"], p["selected"]["B_heading_deg"])):
        leg = legs[str(h)]
        ext = FN.footprint_dims_cm(35, 24, r["robot_footprint"]["transform_margin_per_side_cm"], h)
        wp = leg["path"]["waypoints_cm"]
        assert FN.sweep_check(free, wp, ext, CFG["bev"]["pad_cm"], CFG["bev"]["check_px_per_cm"], 0.25)["ok"], tag
        big = FN.footprint_dims_cm(35, 24, 14.0, h)                                     # negative control: a body 28 cm wider and longer must collide
        assert not FN.sweep_check(free, wp, big, CFG["bev"]["pad_cm"], CFG["bev"]["check_px_per_cm"], 0.25)["ok"], tag
    sw = {"A": p["legs_A"][str(p["selected"]["A_heading_deg"])]["path"]["waypoints_cm"][0], "B": p["legs_B"][str(p["selected"]["B_heading_deg"])]["path"]["waypoints_cm"][0]}
    assert sw["B"] == p["legs_A"][str(p["selected"]["A_heading_deg"])]["path"]["waypoints_cm"][-1]  # B starts exactly where A ended


def test_latest_run_figures_and_manifest_exist():
    run, r = latest()
    for f in r["figures"] + ["plan.json", "manifest.json", "config_snapshot.yaml"] + ["inputs/" + n for n in r["saved_inputs"]]:
        assert (run / f).exists(), f
    assert {"input_E1_undistorted.png", "qwen_destination_raw_response.txt", "target_candidate_qwen_gate_crop.png"} <= set(r["saved_inputs"])
    assert r["failures_and_unverified"]["failed_or_blocked_legs"] and r["failures_and_unverified"]["unmeasured_or_unverified"]
