"""v0.2 destination chain: semantic region -> placement region -> robot approach region -> goal -> path.

All scenes here are SYNTHETIC or use fake model answers (no new photographs exist for these conditions);
they test the logic, not model accuracy. Real-model behaviour is recorded in docs/VERIFICATION_v0.2_*.md.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import EXAMPLE_COMMAND, REF_IMAGE, rect, simple_map
from semantic_task_nav import visualization
from semantic_task_nav.agent.orchestrator import AnalysisRun, StepFailure, replay_state
from semantic_task_nav.config import load_config
from semantic_task_nav.contracts import OBSERVED, RAW_PX
from semantic_task_nav.navigation.map_builder import NavMap
from semantic_task_nav.perception.camera import Frame, ImageSource
from semantic_task_nav.semantics import region_grounding as rg
from semantic_task_nav.semantics.placement import placement_region

CFG = load_config()
TRUCK = {"object_id": "truck_1", "class_name": "truck", "bbox_xyxy": [1430, 69, 1619, 225], "knowledge": OBSERVED,
         "coordinate_frame": RAW_PX, "accepted": True}


def fake_judge(answers):
    """answers: {candidate_id: 'yes'|'no'|'uncertain'}; records every call."""
    calls = []

    def fake(client, mcfg, key, prompt, image_path, max_new, schema):
        cid = int(Path(image_path).stem.replace("region_judge_cand", ""))
        calls.append({"key": key, "prompt": prompt, "image": str(image_path), "cid": cid})
        a = answers.get(cid, "no")
        if a == "BAD":
            return None, {"model": "fake", "commit_hash": None, "raw_response": "x", "error": "INVALID_RESPONSE"}
        return {"answer": a, "reason": f"fake {a}"}, {"model": "fake", "commit_hash": None, "raw_response": a, "error": None}
    return fake, calls


def ground(monkeypatch, tmp_path, answers, sources=("geometric_bands",), boxes=(), truck=TRUCK):
    pcfg = json.loads(json.dumps(CFG["planning"]))
    pcfg["destination"]["candidate_sources"] = list(sources)
    fake, calls = fake_judge(answers)
    monkeypatch.setattr(rg, "qwen_json", fake)
    monkeypatch.setattr(rg, "qwen_ground_boxes", lambda c, m, p, size, q: (
        [{"bbox_xyxy": list(b), "bbox_norm1000": [0, 0, 1, 1], "score": None, "prompt": q,
          "source": "qwen3_vl_grounding"} for b in boxes], {"model": "fake", "raw_response": "boxes", "error": None}))
    img = np.zeros((1080, 1920, 3), np.uint8)
    return rg.ground_loading_region(object(), CFG["models"], pcfg, img, "frame.png", truck, tmp_path), calls


# ------------------------------------------------------------------ semantic stage
def test_one_judgement_per_candidate_and_no_map_in_model_input(monkeypatch, tmp_path):
    res, calls = ground(monkeypatch, tmp_path, {1: "yes"})
    assert len(calls) == len(res["loading_area_candidates"]) == 4
    assert len({c["image"] for c in calls}) == 4                       # one candidate per image
    assert all(c["prompt"] == CFG["models"]["prompts"]["region_judge"]["text"] for c in calls)  # nothing injected
    assert res["status"] == "SEMANTIC_REGION_FOUND"
    cand = res["loading_area_candidates"][0]
    assert res["selected_loading_region"]["polygon_xy"] == cand["polygon_xy"]       # geometry, not model coords
    assert res["selected_loading_region"]["verification_status"] == "unverified"


def test_truck_present_but_no_valid_loading_region(monkeypatch, tmp_path):
    res, _ = ground(monkeypatch, tmp_path, {1: "no", 2: "no", 3: "no", 4: "no"})
    assert res["status"] == "NO_LOADING_REGION_FOUND" and res["semantic_regions"] == []


def test_uncertain_or_invalid_answers_are_not_a_region(monkeypatch, tmp_path):
    res, _ = ground(monkeypatch, tmp_path, {1: "uncertain", 2: "BAD"})
    assert res["status"] == "SEMANTIC_REGION_UNCERTAIN" and res["selected_loading_region"] is None


def test_multiple_plausible_regions_defer_the_choice(monkeypatch, tmp_path):
    res, _ = ground(monkeypatch, tmp_path, {1: "yes", 4: "yes"})
    assert res["status"] == "SEMANTIC_REGIONS_MULTIPLE" and len(res["semantic_regions"]) == 2
    assert res["selected_loading_region"] is None


def test_structure_candidates_are_checked_against_the_truck(monkeypatch, tmp_path):
    t = TRUCK["bbox_xyxy"]
    bay = [1398, 60, 1651, 240]
    boxes = [bay, [1400, 62, 1650, 238], [0, 0, 1920, 1080], t, [100, 900, 300, 1000]]
    res, _ = ground(monkeypatch, tmp_path, {}, sources=("grounded_structure",), boxes=boxes)
    rows = [b for r in res["structure_grounding"] for b in r["boxes"]]
    accepted = [b for b in rows if b["accepted"]]
    assert [b["bbox_xyxy"] for b in accepted] == [bay]                     # one per query, first query only
    reasons = {b["reject_reason"] for b in rows if not b["accepted"]}
    assert {"duplicate of an earlier structure candidate", "scene-sized box",
            "same as the truck box (the vehicle, not an area)",
            "not associated with the truck (neither contains it nor adjacent)"} <= reasons
    assert all(c["source"].startswith("grounded_structure:") for c in res["loading_area_candidates"])


def test_candidates_follow_a_moved_truck(monkeypatch, tmp_path):
    moved = {**TRUCK, "bbox_xyxy": [400, 600, 589, 756]}
    a, _ = ground(monkeypatch, tmp_path, {})
    b, _ = ground(monkeypatch, tmp_path / "b", {}, truck=moved)
    below_a = next(c for c in a["loading_area_candidates"] if c["side_of_truck_box"] == "below")
    below_b = next(c for c in b["loading_area_candidates"] if c["side_of_truck_box"] == "below")
    assert below_b["polygon_xy"][0] == [400, 756] and below_a["polygon_xy"][0] == [1430, 225]


# ------------------------------------------------------------------ placement stage
def test_placement_empty_when_region_is_the_truck_itself():
    reg = {"region_id": "r", "polygon_xy": rect(1430, 69, 1619, 225)}
    p = placement_region(reg, (1920, 1080), [TRUCK], None, None, CFG["planning"])
    assert p["status"] == "PLACEMENT_REGION_EMPTY" and "entirely" in p["reason"]


def test_placement_excludes_truck_and_keeps_free_floor():
    reg = {"region_id": "r", "polygon_xy": rect(1400, 60, 1660, 300)}
    target = {"object_id": "red_cube_1", "bbox_xyxy": [0, 0, 16, 12]}
    p = placement_region(reg, (1920, 1080), [TRUCK], target, None, CFG["planning"])
    assert p["status"] == "PLACEMENT_REGION_FOUND" and p["margin_px"] == 8
    m = np.zeros((1080, 1920), np.uint8)
    for poly in p["polygons_xy"]:
        cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    assert m[69:225, 1430:1619].sum() == 0                              # nothing on the truck
    assert m[60:300, 1400:1660].sum() == m.sum() > 0                    # inside the semantic region


def test_placement_respects_map_no_go(pcfg):
    spec = simple_map(no_go=[rect(100, 100, 200, 200)])
    nav = NavMap(spec, (400, 300), pcfg)
    reg = {"region_id": "r", "polygon_xy": rect(100, 100, 200, 200)}
    p = placement_region(reg, (400, 300), [], None, nav, CFG["planning"])
    assert p["status"] == "PLACEMENT_REGION_EMPTY" and {"map_layer": "no_go", "overlap_px": p["areas_px"]["semantic"]} in p["excluded"]


# ------------------------------------------------------------------ approach + choice (orchestrator stage)
def chain_run(cfg, pcfg, placements, start=(40, 150), obstacles=()):
    cfg["planning"].update({k: pcfg[k] for k in ("cell_px", "robot_radius_px", "safety_margin_px",
                                                  "validation_tolerance_px", "validation_sample_step_px")})
    cfg["planning"]["goal_selection"] = pcfg["goal_selection"]
    spec = simple_map(obstacles=obstacles, start=start)
    frame = Frame(np.zeros((300, 400, 3), np.uint8), 1, 0.0, {})
    run = AnalysisRun(cfg, "IMAGE_DEMO", frame, EXAMPLE_COMMAND, map_spec=spec)
    run.ctx.update(task={"task_type": "transport_and_place"}, nav=NavMap(spec, (400, 300), cfg["planning"]),
                   semantic_regions=[{"region_id": k, "polygon_xy": v} for k, v in placements.items()],
                   placements={k: {"status": "PLACEMENT_REGION_FOUND", "polygons_xy": [v]} for k, v in placements.items()})
    run.ctx["goals"]["pickup"] = {"pose": {"xy": list(start)}}
    return run


def test_multiple_feasible_regions_choose_lowest_score(cfg, pcfg):
    run = chain_run(cfg, pcfg, {"near": rect(110, 140, 120, 160), "far": rect(330, 140, 340, 160)})
    run._select_dropoff_pose()
    assert run.ctx["selected_semantic_region_id"] == "near"
    assert run.ctx["dropoff_choice"]["feasible"] == ["far", "near"]
    assert "lowest goal score" in run.ctx["dropoff_choice"]["rule"]


def test_region_without_robot_approach_is_its_own_failure(cfg, pcfg):
    # placement region inside a big obstacle: semantic + placement fine, no collision-free robot cell within reach
    run = chain_run(cfg, pcfg, {"blocked": rect(195, 145, 205, 155)}, obstacles=[rect(100, 60, 300, 240)])
    with pytest.raises(StepFailure) as e:
        run._select_dropoff_pose()
    assert e.value.code == "ROBOT_APPROACH_NOT_FOUND"
    d = run._destination_record()
    assert d["failed_stage"] == "robot_approach" and d["final_stage_status"] == "ROBOT_APPROACH_NOT_FOUND"


def test_approach_exists_but_unreachable(cfg, pcfg):
    run = chain_run(cfg, pcfg, {"other_side": rect(330, 140, 340, 160)}, obstacles=[rect(190, 0, 210, 300)])
    with pytest.raises(StepFailure) as e:
        run._select_dropoff_pose()
    assert e.value.code == "ROBOT_APPROACH_UNREACHABLE"


# ------------------------------------------------------------------ end-to-end (IMAGE_DEMO, manual inputs)
def demo_cfg(cfg, tmp_path, loading_region):
    demo = json.loads((REF_IMAGE.parents[2] / "config/demo/reference_scene_manual_inputs.json").read_text())
    if loading_region is None:
        demo.pop("loading_region")
    else:
        demo["loading_region"]["polygon_xy"] = loading_region
    p = tmp_path / "demo.json"
    p.write_text(json.dumps(demo))
    cfg["app"]["demo_inputs_file"] = str(p)
    return cfg


def run_demo(cfg):
    r = AnalysisRun(cfg, "IMAGE_DEMO", ImageSource(REF_IMAGE).read(), EXAMPLE_COMMAND)
    return r.run()


def test_e2e_region_and_approach_exist(cfg):
    res = run_demo(cfg)
    d = res["destination"]
    assert d["final_stage_status"] == "DESTINATION_PLANNED" and d["failed_stage"] is None
    assert d["object_placement_region"]["status"] == "PLACEMENT_REGION_FOUND"
    assert d["robot_approach_region"]["stats"]["cells_reachable_from_start"] > 0
    assert d["claims"]["robot_task_success"].startswith("NOT_CLAIMABLE")
    assert d["robot_approach_region"]["reach"]["status"] == "PROVISIONAL"


def test_e2e_region_exists_but_robot_cannot_approach(cfg, tmp_path):
    res = run_demo(demo_cfg(cfg, tmp_path, rect(60, 20, 260, 120)))   # floor top-left, far from the robot road
    d = res["destination"]
    assert d["semantic_loading_region"] is not None
    assert d["chain"][0]["placement"]["status"] == "PLACEMENT_REGION_FOUND"
    assert d["failed_stage"] == "robot_approach" and d["final_stage_status"] == "ROBOT_APPROACH_NOT_FOUND"
    assert "B" not in res["paths"] and res["overall_status"] == "PARTIAL_PLAN"


def test_e2e_region_is_only_the_truck(cfg, tmp_path):
    res = run_demo(demo_cfg(cfg, tmp_path, rect(1430, 69, 1619, 225)))
    d = res["destination"]
    assert d["failed_stage"] == "placement" and d["final_stage_status"] == "PLACEMENT_REGION_EMPTY"
    assert {s["name"]: s["status"] for s in res["steps"]}["select_dropoff_pose"] == "SKIPPED"


def test_e2e_no_loading_region(cfg, tmp_path):
    res = run_demo(demo_cfg(cfg, tmp_path, None))
    assert res["destination"]["failed_stage"] == "semantic_region"
    assert res["destination"]["semantic_loading_region"] is None


# ------------------------------------------------------------------ v0.1 replay regression
V01_RUN = REF_IMAGE.parents[2] / "runs" / "20261115_225944_49f5"


@pytest.mark.skipif(not V01_RUN.exists(), reason="v0.1 reference run not present")
def test_v01_run_still_replays_identically():
    state, _ = replay_state(V01_RUN)
    for name, view in visualization.FIGURES:
        f = V01_RUN / "figures" / f"{name}.png"
        if f.exists():
            assert np.array_equal(cv2.imread(str(f)), visualization.paper_figure(state, view)), name
