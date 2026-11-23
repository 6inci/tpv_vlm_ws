"""Test F (map layers), G (valid path), H (collision rejected), K (no path -> no line), goal selection."""
import cv2
import numpy as np
import pytest

from conftest import REF_IMAGE, rect, simple_map
from semantic_task_nav.navigation.goal_selector import select_goal
from semantic_task_nav.navigation.map_builder import MapError, NavMap, check_alignment
from semantic_task_nav.navigation.path_validator import validate_path
from semantic_task_nav.navigation.planner import line_of_sight, plan

WALL = rect(190, 10, 210, 220)          # wall from the top, gap at the bottom


def nav_of(spec, pcfg, dyn=()):
    return NavMap(spec, tuple(spec["image_size"]), pcfg, dyn)


def test_layers_are_separate(pcfg):
    nav = nav_of(simple_map(obstacles=[rect(100, 100, 140, 140)], no_go=[rect(300, 50, 340, 90)]), pcfg,
                 [{"id": "d", "bbox_xyxy": [250, 200, 270, 220]}])
    assert nav.full_layers["traversable"][120, 120] and not nav.free_full[120, 120]       # obstacle inside
    assert nav.full_layers["no_go"][70, 320] and not nav.free_full[70, 320]
    assert nav.full_layers["dynamic_obstacle"][210, 260] and not nav.free_full[210, 260]
    assert not nav.full_layers["traversable"][5, 5]                                      # outside polygon
    assert nav.free[nav.cell_of(120, 145)]               # free pixel ...
    assert not nav.valid[nav.cell_of(120, 145)]          # ... but closer than the robot radius to the obstacle
    assert nav.valid[nav.cell_of(60, 250)]


def test_map_size_mismatch_is_rejected(pcfg):
    with pytest.raises(MapError) as e:
        NavMap(simple_map(), (640, 480), pcfg)
    assert e.value.code == "MAP_IMAGE_SIZE_MISMATCH"


def test_normal_path_is_found_and_valid(pcfg):
    nav = nav_of(simple_map(obstacles=[WALL]), pcfg)
    p = plan(nav, (60, 60), (340, 60), pcfg, "A_start_to_pickup")
    assert p["status"] == "PATH_FOUND" and p["n_waypoints"] < p["n_raw_waypoints"]
    v = validate_path(nav, p["waypoints_xy"], (60, 60), (340, 60), pcfg)
    assert v["valid"], v
    assert max(y for _, y in p["waypoints_xy"]) > 220           # went around the wall through the gap


@pytest.mark.parametrize("start,goal,code", [((200, 100), (340, 60), "START_IN_COLLISION"),
                                             ((60, 60), (200, 100), "GOAL_IN_COLLISION"),
                                             ((-5, 60), (340, 60), "START_OUTSIDE_MAP")])
def test_invalid_endpoints(pcfg, start, goal, code):
    p = plan(nav_of(simple_map(obstacles=[WALL]), pcfg), start, goal, pcfg, "A_start_to_pickup")
    assert p["status"] == code and p["waypoints_xy"] == []


def test_disconnected_goal_gives_no_path_and_no_line(pcfg):
    nav = nav_of(simple_map(obstacles=[rect(190, 0, 210, 300)]), pcfg)
    p = plan(nav, (60, 60), (340, 60), pcfg, "A_start_to_pickup")
    assert p["status"] == "NO_PATH" and p["waypoints_xy"] == [] and p["length_px"] is None


def test_validator_rejects_path_through_obstacle(pcfg):
    nav = nav_of(simple_map(obstacles=[WALL]), pcfg)
    v = validate_path(nav, [[62.5, 62.5], [337.5, 62.5]], (62.5, 62.5), (337.5, 62.5), pcfg)
    assert not v["valid"] and v["first_violation"]["segment_index"] == 0


def test_simplification_keeps_clearance(pcfg):
    nav = nav_of(simple_map(obstacles=[WALL, rect(100, 150, 300, 170)]), pcfg)
    p = plan(nav, (60, 60), (340, 60), pcfg, "A_start_to_pickup")
    assert p["status"] == "PATH_FOUND"
    assert validate_path(nav, p["waypoints_xy"], (60, 60), (340, 60), pcfg)["valid"]
    assert not line_of_sight(nav.valid, nav.cell_of(60, 60), nav.cell_of(340, 60))


def test_goal_selection_prefers_reachable_cheaper_candidate(pcfg):
    # object sits in a slot of the wall; both sides are within reach but only the left is cheap
    nav = nav_of(simple_map(obstacles=[WALL]), pcfg)
    obj = rect(195, 60, 205, 70)
    g = select_goal(nav, "obj", obj, (60, 60), "pickup_approach", pcfg["goal_selection"]["pickup"], pcfg)
    assert g["status"] == "GOAL_SELECTED" and g["pose"]["xy"][0] < 190
    assert g["stats"]["cells_reachable_from_start"] >= len(g["candidates_top_k"]) > 1
    assert g["pose"]["metric_valid"] is False


def test_goal_selection_blocked_region(pcfg):
    nav = nav_of(simple_map(obstacles=[rect(100, 100, 300, 260)]), pcfg)
    g = select_goal(nav, "obj", rect(195, 175, 205, 185), (40, 40), "pickup_approach",
                    pcfg["goal_selection"]["pickup"], pcfg)
    assert g["status"] == "NO_ACCESSIBLE_CANDIDATE" and g["pose"] is None


def test_goal_selection_separated_from_start(pcfg):
    nav = nav_of(simple_map(obstacles=[rect(190, 0, 210, 300)]), pcfg)
    g = select_goal(nav, "obj", rect(330, 140, 340, 150), (40, 150), "pickup_approach",
                    pcfg["goal_selection"]["pickup"], pcfg)
    assert g["status"] == "CANDIDATES_UNREACHABLE" and g["pose"] is None


def test_camera_alignment_check():
    from semantic_task_nav.config import resolve_path
    spec = {"reference_image": str(REF_IMAGE), "alignment_check": {"max_shift_px": 3.0, "min_response": 0.3}}
    img = cv2.imread(str(REF_IMAGE))
    assert check_alignment(spec, img, resolve_path)["status"] == "ALIGNED"
    moved = np.roll(img, 20, axis=1)
    a = check_alignment(spec, moved, resolve_path)
    assert a["status"] == "MISALIGNED" and abs(a["shift_px"][0] - 20) < 1
    assert check_alignment({}, img, resolve_path)["status"] == "NOT_AVAILABLE"
