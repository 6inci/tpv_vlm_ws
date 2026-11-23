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


# v0.1 test_region_selection (multi-candidate judgement + accessibility inside the semantic stage) was
# replaced in v0.2 by tests/test_destination_chain.py; the v0.1 behaviour stays tested in
# archive/v0.1_2026-11-15/tests.


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
