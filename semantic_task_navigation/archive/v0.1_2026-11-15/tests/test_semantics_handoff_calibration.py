"""Region grounding, scene graph, Test L (hand-off), calibration gating."""
import cv2
import numpy as np
import pytest

from semantic_task_nav.config import load_config
from semantic_task_nav.contracts import OBSERVED, RAW_PX
from semantic_task_nav.geometry.calibration import assess_calibration
from semantic_task_nav.perception.scene_graph import build_scene_graph
from semantic_task_nav.robot.handoff import HandoffError, HardwareDisabledError, TaskExecution
from semantic_task_nav.semantics import region_grounding as rg

CFG = load_config()
TRUCK = {"object_id": "truck_1", "class_name": "truck", "bbox_xyxy": [1430, 69, 1619, 225], "knowledge": OBSERVED,
         "coordinate_frame": RAW_PX}


def test_candidates_are_geometric_and_clipped():
    c = rg.generate_candidates(TRUCK["bbox_xyxy"], (1920, 1080), CFG["planning"]["region_grounding"])
    assert {x["side_of_truck_box"] for x in c} == {"below", "above", "left", "right"}
    for x in c:
        assert all(0 <= p[0] <= 1920 and 0 <= p[1] <= 1080 for p in x["polygon_xy"])


def _fake_qwen(answer):
    def fake(client, mcfg, key, prompt, image_path, max_new, schema):
        rec = {"model": "fake", "commit_hash": None, "raw_response": str(answer), "error": None}
        if answer == "BAD":
            return None, {**rec, "error": "INVALID_RESPONSE"}
        return answer, rec
    return fake


def judged(best, **plaus):
    return {"candidates": [{"id": int(k[1:]), "plausible": v, "reason": "r"} for k, v in plaus.items()],
            "best": best, "evidence": ["cargo bed"], "uncertain": []}


# candidates: 1 below, 2 above, 3 left, 4 right. accessible: only 1 and 3 (fake map)
ACCESS = {1: 50, 2: 0, 3: 7, 4: 0}


@pytest.mark.parametrize("answer,status,selected", [
    (judged(1, c1="yes", c2="no", c3="no", c4="no"), "REGION_SELECTED_UNVERIFIED", 1),
    (judged(2, c1="yes", c2="yes", c3="no", c4="no"), "REGION_SELECTED_UNVERIFIED", 1),   # best inaccessible
    (judged(2, c1="no", c2="yes", c3="no", c4="yes"), "SEMANTIC_REGION_INACCESSIBLE", None),
    (judged(None, c1="yes", c2="no", c3="yes", c4="no"), "SEMANTIC_REGION_UNCERTAIN", None),  # no unique choice
    (judged(None, c1="no", c2="no", c3="uncertain", c4="no"), "SEMANTIC_REGION_UNCERTAIN", None),
    ("BAD", "SEMANTIC_REGION_UNCERTAIN", None),
])
def test_region_selection(monkeypatch, tmp_path, answer, status, selected):
    monkeypatch.setattr(rg, "qwen_json", _fake_qwen(answer))
    img = np.zeros((1080, 1920, 3), np.uint8)
    cands = rg.generate_candidates(TRUCK["bbox_xyxy"], (1920, 1080), CFG["planning"]["region_grounding"])
    access = {str(c["polygon_xy"]): ACCESS[c["candidate_id"]] for c in cands}
    res = rg.ground_loading_region(None, CFG["models"], CFG["planning"], img, TRUCK, tmp_path,
                                   [{"id": "Z", "role": "loading_bay", "polygon": [[1430, 225], [1619, 225], [1619, 330], [1430, 330]],
                                     "confirmed": False}], accessible_cells=lambda poly: access[str(poly)])
    assert res["status"] == status
    if status == "REGION_SELECTED_UNVERIFIED":
        cand = next(c for c in res["loading_area_candidates"] if c["candidate_id"] == selected)
        assert res["selected_loading_region"]["polygon_xy"] == cand["polygon_xy"]    # coordinates from geometry
        assert res["selected_loading_region"]["verification_status"] == "unverified"
        assert res["validated_loading_region"] is None                               # prior not confirmed
        assert res["prior_zone_consistency"][0]["iou"] > 0
    else:
        assert res["selected_loading_region"] is None


def test_scene_graph_marks_unobservable_relation_unverified():
    conv = {"object_id": "conveyor_1", "class_name": "conveyor", "bbox_xyxy": [296, 371, 461, 715], "knowledge": OBSERVED,
            "coordinate_frame": RAW_PX}
    cube = {"object_id": "red_cube_1", "class_name": "red_cube", "bbox_xyxy": [581, 296, 597, 308],
            "knowledge": OBSERVED, "coordinate_frame": RAW_PX}
    task = {"target": {"class": "cube", "color": "red", "source_reference": "conveyor", "source_relation": "produced_by"}}
    g = build_scene_graph([conv], cube, task, None, CFG["planning"])
    rel = {r["type"]: r for r in g["relations"]}
    assert rel["near"]["status"] == "COMPUTED" and rel["near"]["verified"] is False
    assert rel["produced_by"]["status"] == "UNVERIFIED" and rel["produced_by"]["verified"] is False


def test_handoff_requires_explicit_confirmation():
    exe = TaskExecution("transport_and_place", {}, "WAITING_FOR_PICKUP_CONFIRMATION", "PLANNED_ASSUMING_PICKUP")
    assert not exe.completed
    with pytest.raises(HandoffError):
        exe.report("placement_ok", "user_simulation_confirmation")       # cannot skip pickup
    with pytest.raises(HandoffError):
        exe.report("pickup_ok", "planner")                               # only explicit sources
    with pytest.raises(HardwareDisabledError):
        exe.send_to_hardware({"cmd": "go"})
    assert exe.state == "WAITING_FOR_PICKUP_CONFIRMATION" and exe.path_b_status == "PLANNED_ASSUMING_PICKUP"
    exe.report("pickup_ok", "user_simulation_confirmation")
    assert exe.path_b_status == "ACTIVE_AFTER_CONFIRMED_PICKUP" and not exe.completed
    exe.report("placement_ok", "external_simulator")
    assert exe.completed and exe.state == "COMPLETED_SIMULATED"
    with pytest.raises(HardwareDisabledError):
        TaskExecution("transport_and_place", {}, "WAITING_FOR_PICKUP_CONFIRMATION", hardware_enabled=True)


def test_pickup_failure_is_terminal():
    exe = TaskExecution("transport_and_place", {}, "WAITING_FOR_PICKUP_CONFIRMATION", "PLANNED_ASSUMING_PICKUP")
    exe.report("pickup_failed", "user_simulation_confirmation")
    assert exe.state == "PICKUP_FAILED" and not exe.completed
    with pytest.raises(HandoffError):
        exe.report("placement_ok", "user_simulation_confirmation")


def test_calibration_gating():
    ok = assess_calibration(CFG["camera"], 1920, 1080, "HD Web Camera")
    assert ok["undistort_view_available"] and ok["map_mode"] == "IMAGE_SPACE_DEMO" and not ok["metric_output_enabled"]
    assert any("homography" in r for r in ok["reasons"])
    other = assess_calibration(CFG["camera"], 1920, 1080, "HP True Vision FHD Camera: HP T")
    assert not other["undistort_view_available"]
    small = assess_calibration(CFG["camera"], 1280, 720, "HD Web Camera")
    assert not small["undistort_view_available"] and small["map_mode"] == "IMAGE_SPACE_DEMO"
