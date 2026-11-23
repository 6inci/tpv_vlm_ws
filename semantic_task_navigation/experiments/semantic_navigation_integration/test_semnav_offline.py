"""Offline checks for the semantic-navigation integration (synthetic inputs; no model, no GPU, no stored run needed).

  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/semantic_navigation_integration/test_semnav_offline.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent / "task_plan_prototype"))
sys.path.insert(0, str(HERE.parent / "first_vlm_navigation_path"))
sys.path.insert(0, str(HERE))
import context_map as CM  # noqa: E402
import cost_map as KM  # noqa: E402
import decision as DM  # noqa: E402
import plan_task as PT  # noqa: E402
from semantic_task_nav.config import load_config  # noqa: E402
from semantic_task_nav.navigation.map_builder import NavMap  # noqa: E402
from semantic_task_nav.navigation.planner import astar, cost_field  # noqa: E402


def _pcfg(radius=20):
    p = PT.v02_pcfg(load_config(), {"cell_px": 8, "safety_margin_px": 40, "validation_tolerance_px": 2}, {"distance_weight": 1.5})
    p["robot_radius_px"] = radius
    return p


def _nav(radius=20):
    spec = {"traversable": [{"id": "T", "polygon": [[0, 0], [400, 0], [400, 240], [0, 240]]}],
            "static_obstacles": [{"id": "S", "polygon": [[150, 0], [200, 0], [200, 160], [150, 160]]}], "no_go": [], "image_size": [400, 240]}
    p = _pcfg(radius)
    return NavMap(spec, (400, 240), p), p


def test_cost_map_equals_v02_astar_on_every_reachable_cell():
    nav, p = _nav()
    s = nav.cell_of(40, 200)
    cost = cost_field(nav, s, p["planner"], p["safety_weight"])
    cells = np.argwhere(nav.valid & np.isfinite(cost))
    assert len(cells) > 100
    for r, c in cells[::7]:
        a = astar(nav, s, (int(r), int(c)), p["planner"], p["safety_weight"])
        assert a["status"] == "PATH_FOUND" and abs(a["cost"] - cost[r, c]) < 1e-9


def test_multi_source_matches_cost_field_and_min_of_two():
    nav, p = _nav()
    a, b = nav.cell_of(40, 200), nav.cell_of(360, 200)
    fa, fb = (cost_field(nav, x, p["planner"], p["safety_weight"]) for x in (a, b))
    one = KM.multi_source_cost(nav, {a: 0.0}, p["planner"], p["safety_weight"])
    assert np.array_equal(np.isfinite(one), np.isfinite(fa)) and np.allclose(one[np.isfinite(one)], fa[np.isfinite(fa)])
    two = KM.multi_source_cost(nav, {a: 0.0, b: 5.0}, p["planner"], p["safety_weight"])
    ref = np.minimum(fa, fb + 5.0)
    fin = np.isfinite(ref)
    assert np.array_equal(np.isfinite(two), fin) and np.allclose(two[fin], ref[fin])


def test_no_diagonal_corner_cutting():
    valid = np.ones((3, 3), bool)
    valid[0, 1] = valid[1, 0] = False                     # (0,0) touches (1,1) only diagonally between two blocked cells
    nav = SimpleNamespace(valid=valid, risk=np.zeros((3, 3)), rows=3, cols=3, cell_px=8)
    assert not np.isfinite(cost_field(nav, (0, 0), "DISTANCE_ONLY_ASTAR", 0.0)[1, 1])
    assert not np.isfinite(KM.multi_source_cost(nav, {(0, 0): 0.0}, "DISTANCE_ONLY_ASTAR", 0.0)[1, 1])


def test_cell_classes_priority_and_unknown_is_not_free():
    nav, p = _nav(radius=20)
    s = nav.cell_of(40, 200)
    cost = cost_field(nav, s, p["planner"], p["safety_weight"])
    ws = np.ones(nav.valid.shape, bool)
    ws[:, -2:] = False
    cl = KM.classify(nav, cost, ws)
    names = np.array(KM.CLASSES)[cl]
    assert (names[nav.layers["static_obstacle"] & ws] == "STATIC_OBSTACLE").all()
    assert (names[:, -2:] == "OUTSIDE_WORKSPACE").all()
    assert (names[nav.free & ~nav.valid & ws] == "UNREACHABLE_FOOTPRINT").all()
    assert set(np.unique(names[np.isfinite(cost) & ws])) == {"REACHABLE"}


def test_costmap_approach_equals_task_plan_prototype_approach_goal():
    nav, p = _nav()
    poly = [[300, 20], [320, 20], [320, 40], [300, 40]]
    start = [40.0, 200.0]
    cost = cost_field(nav, nav.cell_of(*start), p["planner"], p["safety_weight"])
    a = KM.approach(nav, cost, poly, 6 * 4, 1.5)
    g = PT.approach_goal(nav, "t", poly, start, "pickup_approach", p, 6 * 4, 4, 0)
    assert g["status"] == "GOAL_SELECTED" and a["xy_bev_px"] == list(g["xy_bev_px"])


def _c(i, vis, sem, feas, score, gap=10.0):
    return {"id": i, "rank": int(i[-1]), "visual": vis, "semantic": sem, "source_types": ["a", "b"], "score": 0.3,
            "nav": {"path_A_B_feasible": feas, "v02_goal_score_cm": score, "gap_cm": gap}}


DCFG = {"visual_level": {"VLM_ACCEPTED": 3, "PROBABLE": 2, "PENDING_VERIFICATION": 1, "REJECTED": 0}, "plannable_min_visual_level": 2}
GRADED, BINARY = {"on": 2, "near": 1, "none": 0}, {"on": 1, "near": 1, "none": 0}


def test_ties_stay_ambiguous_and_tie_breaks_never_decide():
    cs = [_c("c1", "PROBABLE", "near", True, 100), _c("c2", "PROBABLE", "near", True, 50)]
    r = DM.target_conditions(cs, DCFG, GRADED)
    assert r["A_visual"]["status"] == "AMBIGUOUS" and r["C_visual_semantic_navigation"]["status"] == "AMBIGUOUS"
    assert r["C_visual_semantic_navigation"]["selected"] is None and set(r["C_visual_semantic_navigation"]["tied_set"]) == {"c1", "c2"}


def test_navigation_never_overrides_semantics():
    cs = [_c("c1", "PROBABLE", "on", False, 900), _c("c2", "PROBABLE", "near", True, 10), _c("c3", "REJECTED", "none", True, 1)]
    r = DM.target_conditions(cs, DCFG, GRADED)
    C = r["C_visual_semantic_navigation"]
    assert C["status"] == "DEFERRED_NAVIGATION" and C["selected"] is None and C["ranking"][0]["id"] == "c1"
    assert r["B_visual_semantic"]["selected"] == "c1"
    assert r["D_navigation_only"]["selected"] == "c3"                       # the naive baseline picks the REJECTED but nearest one
    assert DM.target_conditions(cs, DCFG, BINARY)["C_visual_semantic_navigation"]["selected"] == "c2"   # same tier: feasibility decides


def test_rejected_never_selected_in_A_to_C():
    cs = [_c("c1", "REJECTED", "on", True, 1), _c("c2", "PENDING_VERIFICATION", "on", True, 1)]
    r = DM.target_conditions(cs, DCFG, GRADED)
    for k in ("A_visual", "B_visual_semantic", "C_visual_semantic_navigation"):
        assert r[k]["status"] == "NO_ELIGIBLE_CANDIDATE" and r[k]["selected"] is None


def test_source_relation_on_near_none():
    src = [100, 100, 300, 300]
    assert CM.source_relation([150, 150, 160, 160], src, 150, 0.8)[0] == "on"
    assert CM.source_relation([350, 150, 360, 160], src, 150, 0.8)[0] == "near"
    assert CM.source_relation([600, 600, 610, 610], src, 150, 0.8)[0] == "none"


def _inp():
    cand = lambda i, b: {"id": i, "rank": int(i[-1]), "bbox_xyxy": b, "source_types": ["hsv"], "score": None, "state": "PROBABLE", "stored_qwen": None,   # noqa: E731
                         "crop_views": None, "floor_cm": [[b[0] / 4, b[1] / 4], [b[2] / 4, b[1] / 4], [b[2] / 4, b[3] / 4], [b[0] / 4, b[3] / 4]]}
    return {"image_size": (400, 300), "floor_canvas_wh": (100, 80), "floor_px_per_cm": 1, "floor_pad_cm": 0, "command": "c",
            "rule_parse": {"target": {"class": "cube", "color": "red", "source_reference": "conveyor", "source_relation": "produced_by"}},
            "robot": {"bbox_xyxy": [300, 200, 350, 250], "start_cm": [50.0, 50.0], "crop_answer": "yes", "body_fits_at_start": True},
            "objects": [{"object_id": "conveyor_1", "class_name": "conveyor", "bbox_xyxy": [10, 10, 110, 110], "source": "dino", "crop": "yes"}],
            "bay": {"polygon_px": [[200, 10], [280, 10], [280, 60], [200, 60]], "polygon_cm": [[60, 0], [90, 0], [90, 10], [60, 10]], "status": "X", "not_verified": [],
                    "stored_truck_check": None},
            "placement_floor": {"polygons_px": [[[210, 20], [270, 20], [270, 50], [210, 50]]], "polygons_cm": [[[62, 1], [88, 1], [88, 9], [62, 9]]], "basis": "b"},
            "qwen_structured": {"functional_areas": [{"id": "area_1", "function_name": "all", "bbox_2d": [0, 0, 1000, 1000], "observed_evidence": [], "inference": ""},
                                                     {"id": "area_5", "function_name": "loading dock", "bbox_2d": [475, 0, 750, 266], "observed_evidence": [], "inference": ""}],
                                "task_analysis": {"destination_area_id": "area_5", "decision": "proposed", "reason": "r", "place_needed_for_task": "p",
                                                  "info_added_by_you": [{"statement": "The red cube is not visible in the image.", "basis": "image"}]},
                                "observed_objects": [{"label": "conveyor belt", "count": 1}]},
            "qwen_meta": {"run": "x", "instruction": "y"}, "area_boxes_px": {"area_1": [0, 0, 400, 300], "area_5": [190, 0, 300, 80]},
            "area_floor_cm": {"area_5": [[47, 0], [75, 0], [75, 20], [47, 20]]},
            "zones": [{"id": "Z02", "layer": "traversable", "polygon_px": [[0, 120], [400, 120], [400, 300], [0, 300]], "polygon_cm": [[0, 20], [100, 20], [100, 80], [0, 80]]}],
            "candidates": [cand("cand_r1", [50, 50, 60, 60]), cand("cand_r2", [380, 280, 390, 290])]}


def test_context_map_only_evidenced_relations_with_provenance():
    g = CM.build(_inp(), {"excluded_vlm_areas": {"area_1": "whole image"}, "floor_adjacent_cm": 5}, 150, 0.8)
    rels = {(r["type"], r["subject"], r["object"]): r for r in g["relations"]}
    assert rels[("source_referenced_by_instruction", "task", "conveyor_1")]["provenance"] == "INSTRUCTION_GIVEN"
    assert rels[("source_referenced_by_instruction", "task", "conveyor_1")]["status"] == "UNVERIFIED"
    assert rels[("candidate_destination_for", "area_5", "task")]["status"] == "UNVERIFIED"
    assert ("on", "cand_r1", "conveyor_1") in rels and CM.source_level(g, "cand_r1")[0] == "on"
    assert CM.source_level(g, "cand_r2")[0] == "none" and not any(r["subject"] == "cand_r2" and r["object"] == "conveyor_1" for r in g["relations"])
    assert ("associated_with", "area_5", "loading_bay") in rels and "area_1" not in {n["id"] for n in g["nodes"]}
    assert all(r["status"] != "SUPPORTED" for r in g["relations"] if r["provenance"] != "GEOMETRY_DERIVED")
    assert g["conflicts"] and len({(r["type"], r["subject"], r["object"]) for r in g["relations"]}) == len(g["relations"])
