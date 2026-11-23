"""Offline checks of the task-route integration (no GPU, no model). From the project root:
  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/task_route_integration

Numbers 1-12 follow the requested verification list. Synthetic tests check the logic; pins check what the latest run recorded (one photo; no accuracy)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import route_integration as RI  # noqa: E402

PT, FN = RI.PT, RI.FN
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
GCFG = PT.load_cfg()
S, PAD = GCFG["bev"]["check_px_per_cm"], GCFG["bev"]["pad_cm"]


def latest():
    runs = sorted((HERE / CFG["results_dir"]).glob("*_route"))
    assert runs, "run route_integration.py first"
    r = runs[-1]
    return r, *(json.loads((r / f).read_text(encoding="utf-8")) for f in ("task_plan.json", "validation.json", "path_a.json", "path_b.json"))


def all_plans(tp):
    yield "A", tp["condition_A"]["plan"]
    for k, v in tp["condition_B"]["conditional_plans"].items():
        yield "B" + k, v


# ------------------------------------------------------------------------------------------------ synthetic context (image px == cm)
def synth_ctx(rooms, size=(200, 60)):
    fr = PT.Frames(np.eye(3), GCFG["bev"]["px_per_cm"], PAD, size)
    spec_cm = {"traversable": [{"id": f"R{i}", "polygon": p} for i, p in enumerate(rooms)], "static_obstacles": [], "no_go": []}
    spec_bev = {k: [{"id": z["id"], "polygon": fr.cm_to_bev(z["polygon"]).tolist()} for z in v] for k, v in spec_cm.items()}
    spec_bev.update({"image_size": list(fr.bev_size), "coordinate_frame": "bev_px", "review_status": "SYNTHETIC", "start_pose": None})
    return {"cfg": GCFG, "fr": fr, "size_cm": size, "spec_cm": spec_cm, "spec_bev": spec_bev, "pcfg": PT.v02_pcfg(PT.load_config(), GCFG["bev"], GCFG["approach"]),
            "free_hi": FN.free_mask_cm(spec_cm, size, PAD, S), "bay_cm": [[190, 2], [198, 2], [198, 8], [190, 8]],
            "bay_bev": fr.cm_to_bev([[190, 2], [198, 2], [198, 8], [190, 8]]).tolist()}


def rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# 1 / 2 data and coordinates
def test_1_2_stored_inputs_load_and_match_the_image():
    _, tp, val, _, _ = latest()
    assert all(val["inputs"][k] for k in ("stored_e1_input_equals_image", "condition_A_box_equals_funnel_label", "condition_B_boxes_equal_retention_record", "retained_candidates_preserved"))
    H = np.array(tp["floor_transform"]["H_img_to_floor_cm"])
    for _, p in all_plans(tp):
        for leg in p["paths"].values():                                   # image px and floor cm of every approach point agree through H
            assert np.abs(PT.FT.project(np.linalg.inv(H), [leg["goal_cm"]])[0] - np.array(leg["goal_image_px"])).max() < 0.2


# 3 candidates preserved
def test_3_candidates_and_states_are_preserved_unchanged():
    _, tp, _, _, _ = latest()
    rr = json.loads(((HERE / CFG["stored"]["retention_run"]).resolve() / "result.json").read_text(encoding="utf-8"))
    e1 = next(p for p in rr["photos"] if p["photo_id"] == CFG["condition_A"]["photo_id"])
    assert tp["condition_B"]["candidates_all"] == e1["candidates"]


# 4 footprint in the map frame
def test_4_footprint_extent_matches_user_size_plus_margin_and_heading():
    _, tp, val, _, _ = latest()
    m = tp["floor_transform"]["footprint_margin_cm"]
    assert val["footprint_frame"]["primary_extent_cm"] == [35 + 2 * m, 24 + 2 * m] and val["footprint_frame"]["cm_bev_roundtrip_max_err_cm"] < 1e-9
    for _, p in all_plans(tp):
        for leg in p["paths"].values():
            assert tuple(leg["footprint_extent_xy_cm"]) == FN.footprint_dims_cm(35, 24, m, leg["heading_deg"])
    nav = PT.FN.RectNavMap(synth_ctx([rect(0, 0, 200, 60)])["spec_bev"], synth_ctx([rect(0, 0, 200, 60)])["fr"].bev_size, synth_ctx([rect(0, 0, 200, 60)])["pcfg"], (42, 31), GCFG["bev"]["px_per_cm"])
    ys, xs = np.nonzero(nav.footprint_fits_full)                           # valid centres keep >= half the body from the room edge
    assert (xs.min() / GCFG["bev"]["px_per_cm"] - PAD) >= 21 - 0.26 and (ys.min() / GCFG["bev"]["px_per_cm"] - PAD) >= 15.5 - 0.26


# 5 / 6 no-go and obstacles: recomputed here from the stored zones, independently of the run's own numbers
def test_5_6_every_generated_path_keeps_the_whole_body_off_no_go_obstacles_and_outside():
    _, tp, _, _, _ = latest()
    spec = tp["map"]["zones_cm"]
    free = FN.free_mask_cm(spec, (315, 210), PAD, S)
    n = 0
    for name, p in all_plans(tp):
        for tag, leg in p["paths"].items():
            assert FN.sweep_check(free, leg["waypoints_cm"], leg["footprint_extent_xy_cm"], PAD, S, 0.25)["ok"], (name, tag)
            lo = p["checks"][tag]["swept_layer_overlap"]
            assert lo["clear"] and lo["on_no_go_px"] == 0 and lo["on_static_obstacle_px"] == 0 and lo["outside_drivable_px"] == 0, (name, tag)
            n += 1
    assert n == 8                                                          # A, B for condition A and for 3 conditional candidates


# 7 corner cutting
def test_7_a_diagonal_that_cuts_an_obstacle_corner_fails_and_the_planner_goes_around():
    ctx = synth_ctx([rect(0, 0, 200, 120)], size=(200, 120))
    ctx["spec_cm"]["static_obstacles"] = [{"id": "O", "polygon": rect(80, 0, 120, 70)}]
    ctx["spec_bev"]["static_obstacles"] = [{"id": "O", "polygon": ctx["fr"].cm_to_bev(rect(80, 0, 120, 70)).tolist()}]
    ctx["free_hi"] = FN.free_mask_cm(ctx["spec_cm"], (200, 120), PAD, S)
    body = (30, 20)
    assert not FN.sweep_check(ctx["free_hi"], [[40, 30], [160, 95]], body, PAD, S, 0.25)["ok"]          # straight line clips the corner
    nav = FN.RectNavMap(ctx["spec_bev"], ctx["fr"].bev_size, ctx["pcfg"], body, GCFG["bev"]["px_per_cm"])
    path = RI.plan(nav, ctx["fr"].cm_to_bev([[40, 30]])[0].tolist(), ctx["fr"].cm_to_bev([[160, 30]])[0].tolist(), ctx["pcfg"], "x")
    assert path["status"] == "PATH_FOUND"
    wp = ctx["fr"].bev_to_cm(path["waypoints_xy"]).tolist()
    assert FN.sweep_check(ctx["free_hi"], wp, body, PAD, S, 0.25)["ok"] and max(y for _, y in wp) > 70 + 10   # goes below the obstacle


# 8 / 9 approach points
def test_8_9_approach_poses_are_drivable_for_the_whole_body():
    _, tp, _, _, _ = latest()
    free = FN.free_mask_cm(tp["map"]["zones_cm"], (315, 210), PAD, S)
    for name, p in all_plans(tp):
        for tag, leg in p["paths"].items():
            assert FN.body_free(free, leg["goal_cm"], leg["footprint_extent_xy_cm"], PAD, S)[0], (name, tag)
            assert p["checks"][tag]["approach_pose_body_free"]
            assert leg["approach"]["body_to_target_gap_cm"] > 0          # the robot stops beside the target/bay, never on its box (centre)


# 10 A-B connection
def test_10_path_b_starts_where_path_a_ends():
    _, tp, _, pa, pb = latest()
    for name, p in all_plans(tp):
        if "B" in p["paths"]:
            assert p["paths"]["B"]["waypoints_cm"][0] == p["paths"]["A"]["waypoints_cm"][-1] and p["checks"]["A_B_connected"]["B_starts_where_A_ends"]
    for k in pa:
        if pa[k] and pb.get(k):
            assert pb[k]["waypoints_cm"][0] == pa[k]["waypoints_cm"][-1]


# 11 uncertainty
def test_11_uncertain_candidates_are_never_confirmed_or_chosen():
    _, tp, val, _, _ = latest()
    text = json.dumps(tp["condition_B"], ensure_ascii=False)
    assert '"CONFIRMED"' not in text and tp["condition_B"]["photo_decision"] == "AMBIGUOUS" and tp["condition_B"]["selected_target"] is None
    assert all(v["target"]["name"] == "CONDITIONAL_ON_CANDIDATE" for v in tp["condition_B"]["conditional_plans"].values())
    a = tp["condition_A"]["plan"]["target"]
    assert a["name"] == "REFERENCE_TARGET_PRELIMINARY" and a["label_status"] == "AI_PRELIMINARY_REVIEW" and "ORACLE" in a["not"]
    for _, p in all_plans(tp):
        assert all(s["executed"] is False for s in p["states"]) and [s["state"] for s in p["states"]] == list(RI.PLAN_STATES)
        assert next(s for s in p["states"] if s["state"] == "PICKUP_ASSUMED")["pickup_feasibility"] == "UNVERIFIED"


# 12 failure handling
def test_12_start_that_cannot_hold_the_body_is_reported_not_planned():
    ctx = synth_ctx([rect(0, 0, 200, 25)], size=(200, 25))                       # 25 cm high strip: a 31 cm body never fits
    ctx["cfg"] = {**GCFG, "robot": {**GCFG["robot"], "headings_deg": [0, 90]}}
    plan_rec, var = RI.run_target(ctx, "x", [20, 10, 24, 14], 35, 24, 3.5, [150, 12], {"name": "SYN"})
    assert plan_rec["failure_A"] == "START_POSE_INVALID" and plan_rec["paths"] == {} and plan_rec["states"] == []
    assert plan_rec["failure_B"].startswith("NOT_ATTEMPTED")


@pytest.mark.parametrize("legs,which,code", [
    ({0: {"feasible": False, "failure": {"stage": "approach_point", "code": "NO_ACCESSIBLE_CANDIDATE"}}}, "A", "NO_VALID_PICKUP_APPROACH"),
    ({0: {"feasible": False, "failure": {"stage": "path", "code": "NO_PATH"}}}, "A", "NO_VALID_PATH_A"),
    ({0: {"feasible": False, "failure": {"stage": "approach_point", "code": "CANDIDATES_UNREACHABLE"}}}, "B", "NO_VALID_PLACE_APPROACH"),
    ({0: {"feasible": False, "failure": {"stage": "validation", "code": "PATH_FAILED_FOOTPRINT_CHECK"}}, 90: {"feasible": False, "failure": {"stage": "heading_change", "code": "ROTATION_CLEARANCE_NOT_MET"}}}, "B", "NO_VALID_PATH_B"),
    ({0: {"feasible": True, "failure": None}}, "A", None)])
def test_12_failure_codes(legs, which, code):
    assert RI.leg_code(legs, which) == code


def test_12_earlier_35px_path_fails_the_real_size_check_and_is_recorded():
    _, tp, val, _, _ = latest()
    o = tp["old_path_recheck"]
    assert o["verdict"].startswith("FAILS") and not any(c["sweep_ok"] for c in o["checks"].values())
    assert any(s["step"] == "recheck_old_path" and s["status"] == "FAILED" for s in tp["steps"])
    assert tp["condition_A"]["comparisons"]["circumscribed_disc_with_margin"]["A"]["goal_status"] == "START_IN_COLLISION"


def test_run_provenance_and_outputs():
    run, tp, _, _, _ = latest()
    assert tp["v02_code_before"] == tp["v02_code_after"] and tp["v02_code_before"].startswith("254cfb31da7f") and tp["no_hardware_command"]
    assert tp["claims"]["pickup_or_placement_success"] == "NOT_CLAIMABLE" and tp["floor_transform"]["status"] == "PROVISIONAL_UNVALIDATED"
    for f in ["task_plan.json", "path_a.json", "path_b.json", "validation.json", "manifest.json", "config_snapshot.yaml"] + [Path(p).relative_to(run).as_posix() for p in tp["figures"].values()]:
        assert (run / f).exists(), f
    assert len(tp["figures"]) == 6
