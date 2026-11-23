"""Offline checks of V2 (no GPU, no model). From the project root:
  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q experiments/semantic_evidence_verification/test_verifier2_offline.py

Two kinds of tests: (1) the design rules, on small synthetic inputs; (2) regression pins of what V1 and V2 decide on the three
STORED proposals. The pins record the behaviour that was observed when the verifier was built; they are not evidence that
V2 is right, and no threshold was changed to produce them.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import observations as O  # noqa: E402
import scenarios as SC  # noqa: E402
import v1_replay as R  # noqa: E402
import verifier2 as V  # noqa: E402
from semantic_task_nav.config import load_config  # noqa: E402

CFG = V.load_config()
V02 = load_config()["planning"]
SC_LIST = {s["id"]: s for s in SC.scenarios()}


def run(sc_id):
    sc = SC_LIST[sc_id]
    out = V.verify_case(sc["case"], SC.scenario_cfg(CFG, sc), V02)
    return sc, out, next(c for c in out["candidates"] if c["candidate_id"] == sc["case"]["chosen"])


# ------------------------------------------------------------------ design rules
def test_purpose_comes_from_the_instruction_text_only():
    p = V.purpose_from_instruction(SC.SHIPPING, CFG["purposes"])
    assert p["purposes"] == ["outbound_shipping"] and p["requires_any_of"] == ["carrier_vehicle"]
    assert not V.purpose_from_instruction(SC.NO_PURPOSE, CFG["purposes"])["known"]


def test_claims_are_typed_and_unverifiable_text_is_never_assumed_true():
    cl = V.extract_claims(["white truck parked", "painted lines on the floor", "crates stacked", "red cube marker (assumed)"],
                          CFG["registry"], CFG["marking_terms"])
    assert [c["kind"] for c in cl] == ["registry", "marking", "unmatched_object", "registry"]
    assert cl[3]["hedged"] and cl[3]["classes"] == ["red_cube"]


def test_the_vlms_function_wording_changes_nothing():
    sc, out, base = run("S1_consistent_and_localized")
    altered = copy.deepcopy(sc["case"])
    altered["candidates"][0]["function_claimed"] = "absolutely certainly the perfect shipping dock (confidence 100%)"
    altered["candidates"][0]["confidence"] = "high"
    again = V.verify_case(altered, CFG, V02)["candidates"][0]
    assert again["final_disposition"] == base["final_disposition"] and again["semantic_verdict"] == base["semantic_verdict"]


def test_region_size_is_not_a_criterion_for_the_semantic_verdict_or_for_readiness():
    sc, _, base = run("S1_consistent_and_localized")
    big = copy.deepcopy(sc["case"])
    big["candidates"][0]["box_px"] = [1000, 0, 1900, 300]                    # much larger, still contains truck and localization
    r = V.verify_case(big, CFG, V02)["candidates"][0]
    assert r["semantic_verdict"] == base["semantic_verdict"] == V.SEMANTICALLY_SUPPORTED
    assert r["placement"]["readiness"] == base["placement"]["readiness"] == "ESTABLISHED"
    assert r["placement"]["region_size_info"]["area_px"] > base["placement"]["region_size_info"]["area_px"]
    assert "NOT used" in r["placement"]["region_size_info"]["role"]


def test_placement_needs_a_verified_localization_and_nothing_else_proves_it():
    _, _, wide = run("S5_semantic_ok_placement_unverified")
    assert wide["semantic_verdict"] == V.SEMANTICALLY_SUPPORTED and wide["final_disposition"] == V.PLACEMENT_UNVERIFIED
    assert wide["placement"]["legacy_v1_size_gate"]["would_block_in_v1"] is True                 # shown next to the verdict, not used by it
    assert run("S1_consistent_and_localized")[2]["placement"]["legacy_v1_size_gate"]["would_block_in_v1"] is False
    sc = SC_LIST["S1_consistent_and_localized"]
    bad = copy.deepcopy(sc["case"])
    bad["observations"]["localizations"] = [{"candidate_id": "area_1", "box_px": [100, 100, 300, 300],
                                             "source": "evidence_grounded_refinement", "query": "x"}]   # outside the proposal
    assert V.verify_case(bad, CFG, V02)["candidates"][0]["placement"]["readiness"] == "NOT_ESTABLISHED"
    unsourced = copy.deepcopy(sc["case"])
    unsourced["observations"]["localizations"][0]["source"] = "vlm_own_box"                       # not an accepted source
    assert V.verify_case(unsourced, CFG, V02)["candidates"][0]["placement"]["readiness"] == "NOT_ESTABLISHED"


def test_not_found_and_not_run_are_different_states():
    _, _, absent = run("S2_evidence_object_absent")
    assert absent["class_evidence"][0]["status"] == V.NOT_FOUND                    # truck WAS searched for
    assert absent["semantic_verdict"] == V.EVIDENCE_NOT_ESTABLISHED
    sc = SC_LIST["S1_consistent_and_localized"]
    not_run = copy.deepcopy(sc["case"])
    not_run["observations"]["objects"] = [o for o in not_run["observations"]["objects"] if o["class_name"] != "traffic_cone"]
    not_run["observations"]["searched"].pop("traffic_cone", None)                  # cones were never looked for
    r = V.verify_case(not_run, CFG, V02)["candidates"][0]
    assert next(c for c in r["class_evidence"] if c["class"] == "traffic_cone")["status"] == V.NOT_RUN
    assert r["semantic_verdict"] == V.SEMANTICALLY_SUPPORTED                       # the truck still carries the purpose
    assert r["planner_support"]["status"] == "UNSUPPORTED_OBSTACLE_COVERAGE_INCOMPLETE"


def test_the_object_to_be_carried_is_never_evidence_for_where_to_put_it():
    sc = SC_LIST["S1_consistent_and_localized"]
    c = copy.deepcopy(sc["case"])
    c["candidates"][0]["observed_evidence"] = ["a red cube on the floor"]
    c["observations"]["objects"] = [{"object_id": "red_cube_1", "class_name": "red_cube", "bbox_xyxy": SC.TRUCK, "basis": "x"}]
    r = V.verify_case(c, CFG, V02)["candidates"][0]
    assert r["claims"][0]["status"] == V.NOT_EVIDENCE and r["semantic_verdict"] == V.EVIDENCE_NOT_ESTABLISHED


def test_a_facility_missing_from_the_registry_is_held_not_rejected_and_extending_the_registry_needs_no_code():
    _, _, held = run("S10_other_function_verified")
    assert held["semantic_verdict"] == V.PURPOSE_FIT_UNDETERMINED                  # verified pallets show staging, not shipping
    assert "goods_staging" in held["purpose_fit"]["established_capabilities"]
    sc = SC_LIST["S6_non_truck_carrier"]
    assert V.verify_case(sc["case"], CFG, V02)["candidates"][0]["final_disposition"] == V.EVIDENCE_NOT_ESTABLISHED   # unregistered class
    with_van = V.verify_case(sc["case"], SC.scenario_cfg(CFG, sc), V02)["candidates"][0]                           # config extension only
    assert with_van["final_disposition"] == V.READY
    assert [e["class"] for e in with_van["class_evidence"]] == ["delivery_van"]                                     # no truck involved


def test_a_purpose_no_registered_detector_can_verify_stays_undetermined():
    _, _, r = run("S7_purpose_mismatch")
    assert r["purpose"]["purposes"] == ["charging"] and r["semantic_verdict"] == V.PURPOSE_FIT_UNDETERMINED
    assert "no registered detector class can verify" in r["semantic_reasons"][0]


def test_multiple_candidates_are_judged_separately_and_the_verifier_does_not_replace_the_choice():
    sc, out, chosen = run("S4_multiple_candidates")
    by_id = {c["candidate_id"]: c["final_disposition"] for c in out["candidates"]}
    assert by_id == {"area_1": V.READY, "area_2": V.EVIDENCE_NOT_ESTABLISHED}
    assert out["chosen_by_vlm"] == "area_2" and chosen["hand_over_to_planner"] is False
    assert out["ready_candidates"] == ["area_1"] and "NOT_MADE_BY_THE_VERIFIER" in out["selection"]


def test_preconditions_are_separate_from_the_semantic_verdict():
    sc = SC_LIST["S1_consistent_and_localized"]
    c = copy.deepcopy(sc["case"])
    c["observations"]["map"] = {"alignment": "MISALIGNED", "layers": None}
    c["observations"]["preconditions"]["cube"] = "TARGET_NOT_FOUND"
    r = V.verify_case(c, CFG, V02)["candidates"][0]
    assert r["final_disposition"] == V.READY and r["hand_over_to_planner"] is False
    assert len(r["planning_preconditions"]["blocked_by"]) == 2


@pytest.mark.parametrize("sid", list(SC_LIST))
def test_each_synthetic_scenario_matches_the_expectation_written_before_it_was_run(sid):
    sc, out, chosen = run(sid)
    assert chosen["final_disposition"] == sc["expect"]["v2"] and chosen["hand_over_to_planner"] == sc["expect"]["hand_over"]
    v1 = R.load_v1()
    rp = R.replay(v1, SC.v1_inputs(sc["case"]), R.v1_config(), V02)
    assert rp["decision"] == sc["expect"]["v1"]
    for k, v in sc["expect"].get("v2_other", {}).items():
        assert next(c for c in out["candidates"] if c["candidate_id"] == k)["final_disposition"] == v


# ------------------------------------------------------------------ stored proposals (regression pins)
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stored(cid):
    cc = next(c for c in CFG["stored_cases"] if c["id"] == cid)
    return O.build_case(cc, HERE)


def test_v1_replay_reproduces_every_stored_v1_result():
    v1, icfg = R.load_v1(), R.v1_config()
    n = 0
    for cc in CFG["stored_cases"]:
        case = O.build_case(cc, HERE)
        for run_ in case["_v1_runs"]:
            stored_v1 = O.v1_stored(run_)
            rp = R.replay(v1, R.inputs_from_run(run_, case), R.icfg_with_overrides(icfg, stored_v1["overrides"]), V02)
            rep = R.reproduces(stored_v1, rp)
            assert rep["checks_equal"] and rep["decision_equal"], (run_["run_id"], rep)
            n += 1
    assert n == 4


@pytest.mark.parametrize("cid,final,semantic", [
    ("DEV_REF", V.EVIDENCE_RELATION_INCONSISTENT, V.EVIDENCE_RELATION_INCONSISTENT),
    ("E1", V.PLACEMENT_UNVERIFIED, V.SEMANTICALLY_SUPPORTED),
    ("REF_P01", V.EVIDENCE_NOT_ESTABLISHED, V.EVIDENCE_NOT_ESTABLISHED)])
def test_v2_on_the_stored_proposals(cid, final, semantic):
    out = V.verify_case(stored(cid), CFG, V02)["candidates"][0]
    assert out["final_disposition"] == final and out["semantic_verdict"] == semantic and out["hand_over_to_planner"] is False


def test_stored_e1_evidence_details():
    c = V.verify_case(stored("E1"), CFG, V02)["candidates"][0]
    kinds = {cl["text"]: (cl["kind"], cl["status"]) for cl in c["claims"]}
    assert kinds["truck"] == ("registry", V.VERIFIED_PRESENT) and kinds["cones"] == ("registry", V.VERIFIED_PRESENT)
    assert kinds["barriers"] == ("unmatched_object", V.UNVERIFIABLE)
    cones = next(e for e in c["class_evidence"] if e["class"] == "traffic_cone")
    assert len(cones["instances"]) == 6 and cones["n_related"] == 6
    assert c["planner_support"]["unmodeled_physical_claims"] == ["barriers"]


def test_stored_dev_ref_truck_exists_but_is_unrelated_to_the_proposed_region():
    c = V.verify_case(stored("DEV_REF"), CFG, V02)["candidates"][0]
    t = next(e for e in c["class_evidence"] if e["class"] == "truck")
    assert t["status"] == V.VERIFIED_PRESENT and t["n_related"] == 0
    assert t["instances"][0]["gap_px"] == pytest.approx(227.7, abs=0.2) and t["instances"][0]["fraction_of_object_inside_region"] == 0.0


def test_loading_stored_runs_modifies_nothing():
    files = []
    for cc in CFG["stored_cases"]:
        for r in cc["runs"] + cc.get("supplementary_runs", []):
            d = (HERE / r).resolve()
            files += [d / "integration_result.json", d / "result.json", d / "vlm_proposal/raw_response.txt", d / "analysis/detections.json"]
    before = [_sha(f) for f in files]
    for cc in CFG["stored_cases"]:
        V.verify_case(O.build_case(cc, HERE), CFG, V02)
    assert before == [_sha(f) for f in files]


def test_supplementary_run_must_be_the_same_image_and_the_same_vlm_reply():
    cc = copy.deepcopy(next(c for c in CFG["stored_cases"] if c["id"] == "E1"))
    cc["supplementary_runs"] = ["../semantic_to_navigation_integration/results/20261118_202916_fbe7"]      # DEV_REF run: different image
    with pytest.raises(ValueError):
        O.build_case(cc, HERE)


def test_v2_never_reads_the_reference_zone_and_size_never_enters_a_decision_function():
    import ast
    import inspect
    for f in ("verifier2.py", "observations.py", "scenarios.py", "compare.py"):
        tree = ast.parse((HERE / f).read_text(encoding="utf-8"))
        literals = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert "semantic_reference_zones" not in literals and "Z06" not in literals, f     # never used as a key or value
    for fn in (V.semantic_verdict, V.placement_readiness, V.planner_support, V.class_evidence, V.relation_of, V.region_conflicts):
        assert "max_unrefined_area_ratio" not in inspect.getsource(fn) and "area_ratio" not in inspect.getsource(fn), fn.__name__
