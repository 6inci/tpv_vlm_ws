import copy
import json
import math
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(WORKSPACE / "scripts"))

import run_dynamic_path_invalidation as runner
from vlm_reposition.planning import (
    DynamicStateError,
    apply_obstacle_event,
    build_dynamic_collision_world,
    build_grid_map,
    initial_obstacle_state,
    load_dynamic_config,
    load_planner_config,
    validate_current_path,
    validate_dynamic_config,
)
from vlm_reposition.world import load_physical_geometry, load_zones


PROTECTED_DIRS = [
    WORKSPACE / "results" / name
    for name in ("workspace_state", "physical_geometry", "collision_geometry", "path_planning_baseline", "path_planning_stress")
]


class DynamicPathInvalidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_dynamic_config(WORKSPACE / "config" / "dynamic_obstacle_scenarios.json")
        cls.planner = load_planner_config(WORKSPACE / "config" / "planner_baseline.json")
        cls.zones = load_zones(WORKSPACE / "config" / "zones.json")
        cls.physical = load_physical_geometry(WORKSPACE / "config" / "physical_geometry.json")
        cls.readiness = json.loads((WORKSPACE / "results" / "physical_geometry" / "physical_geometry_readiness.json").read_text())["planning_geometry_readiness"]
        cls.results, cls.contexts = {}, {}
        for scenario in cls.config["scenarios"]:
            result, contexts = runner.run_scenario(
                cls.config, scenario, cls.zones, cls.physical, cls.readiness, cls.planner
            )
            cls.results[scenario["scenario_id"]] = result
            cls.contexts[scenario["scenario_id"]] = contexts

    def scenario(self, scenario_id):
        return next(item for item in self.config["scenarios"] if item["scenario_id"] == scenario_id)

    def assert_config_error(self, mutate, code):
        config = copy.deepcopy(self.config)
        mutate(config)
        with self.assertRaises(DynamicStateError) as caught:
            validate_dynamic_config(config)
        self.assertEqual(caught.exception.code, code)

    def test_01_dynamic_scenario_config_load(self):
        self.assertEqual([item["scenario_id"] for item in self.config["scenarios"]], ["D1", "D2", "D3", "D4", "D5", "D6"])

    def test_02_stable_obstacle_identity(self):
        scenario = self.scenario("D3")
        before = initial_obstacle_state(scenario)
        after = apply_obstacle_event(before, scenario["events"][0])
        self.assertIn("dynamic_box_03", {item["id"] for item in before["obstacles"]})
        self.assertIn("dynamic_box_03", {item["id"] for item in after["obstacles"]})

    def test_03_add_operation(self):
        scenario = self.scenario("D1")
        after = apply_obstacle_event(initial_obstacle_state(scenario), scenario["events"][0])
        self.assertEqual(len(after["obstacles"]), 2)

    def test_04_duplicate_add_rejected(self):
        scenario = self.scenario("D1")
        event = copy.deepcopy(scenario["events"][0])
        event["obstacle"] = copy.deepcopy(scenario["initial_static_obstacles"][0])
        with self.assertRaises(DynamicStateError) as caught:
            apply_obstacle_event(initial_obstacle_state(scenario), event)
        self.assertEqual(caught.exception.code, "DUPLICATE_OBSTACLE_ID")

    def test_05_remove_operation(self):
        scenario = self.scenario("D4")
        state = apply_obstacle_event(initial_obstacle_state(scenario), scenario["events"][0])
        state = apply_obstacle_event(state, scenario["events"][1])
        self.assertNotIn("dynamic_box_04", {item["id"] for item in state["obstacles"]})

    def test_06_unknown_remove_rejected(self):
        event = {"operation": "REMOVE", "obstacle": {"id": "missing"}}
        with self.assertRaises(DynamicStateError) as caught:
            apply_obstacle_event(initial_obstacle_state(self.scenario("D1")), event)
        self.assertEqual(caught.exception.code, "UNKNOWN_OBSTACLE_ID")

    def test_07_move_operation(self):
        scenario = self.scenario("D3")
        state = apply_obstacle_event(initial_obstacle_state(scenario), scenario["events"][0])
        moved = next(item for item in state["obstacles"] if item["id"] == "dynamic_box_03")
        self.assertEqual(moved["bounds"]["x_min"], 0.88)

    def test_08_unknown_move_rejected(self):
        event = copy.deepcopy(self.scenario("D3")["events"][0])
        event["obstacle"]["id"] = "missing"
        with self.assertRaises(DynamicStateError) as caught:
            apply_obstacle_event(initial_obstacle_state(self.scenario("D1")), event)
        self.assertEqual(caught.exception.code, "UNKNOWN_OBSTACLE_ID")

    def test_09_world_revision_increments_deterministically(self):
        scenario = self.scenario("D4")
        state = initial_obstacle_state(scenario)
        revisions = [state["world_revision"]]
        for event in scenario["events"]:
            state = apply_obstacle_event(state, event)
            revisions.append(state["world_revision"])
        self.assertEqual(revisions, [0, 1, 2])

    def test_10_d1_obstacle_away_valid(self):
        self.assertEqual(self.results["D1"]["events"][-1]["path_state_after"], "VALID")

    def test_11_d2_obstacle_on_path_blocked(self):
        self.assertEqual(self.results["D2"]["events"][-1]["path_state_after"], "BLOCKED")

    def test_12_d2_blocking_obstacle_id(self):
        self.assertEqual(self.results["D2"]["events"][-1]["blocking_obstacle_ids"], ["dynamic_box_02"])

    def test_13_first_blocked_location_deterministic(self):
        event = self.results["D2"]["events"][-1]
        self.assertEqual((event["first_blocked_waypoint_index"], event["first_blocked_edge_index"]), (50, 49))

    def test_14_d3_move_onto_path_blocked(self):
        self.assertEqual(self.results["D3"]["events"][-1]["path_state_after"], "BLOCKED")

    def test_15_d4_blocker_removal_valid(self):
        states = [item["path_state_after"] for item in self.results["D4"]["events"]]
        self.assertEqual(states, ["BLOCKED", "VALID"])

    def test_16_d5_move_away_valid(self):
        states = [item["path_state_after"] for item in self.results["D5"]["events"]]
        self.assertEqual(states, ["BLOCKED", "VALID"])

    def test_17_d6_near_path_valid(self):
        self.assertEqual(self.results["D6"]["events"][-1]["path_state_after"], "VALID")

    def test_18_d6_risk_changes_without_hard_invalidation(self):
        event = self.results["D6"]["events"][-1]
        self.assertGreater(event["accumulated_safety_risk_after"], event["accumulated_safety_risk_before"])
        self.assertFalse(event["replan_required"])

    def test_19_occupancy_rebuilt_after_event(self):
        event = self.results["D2"]["events"][-1]
        self.assertNotEqual(event["occupancy_sha256_before"], event["occupancy_sha256_after"])

    def test_20_safety_layer_rebuilt_after_event(self):
        event = self.results["D6"]["events"][-1]
        self.assertNotEqual(event["safety_risk_sha256_before"], event["safety_risk_sha256_after"])

    def test_21_no_double_inflation(self):
        obstacle = next(item for item in self.results["D2"]["revisions"][1]["collision_obstacles"] if item["id"] == "dynamic_box_02")
        self.assertEqual(obstacle["raw_bounds"], {"x_min": 0.88, "x_max": 0.92, "y_min": 0.85, "y_max": 0.95})
        self.assertEqual(obstacle["collision_bounds"], {"x_min": 0.735, "x_max": 1.065, "y_min": 0.727, "y_max": 1.073})
        self.assertEqual(self.results["D2"]["revisions"][1]["grid"]["additional_inflation_m"], 0.0)

    def _coarse_event_validation(self, raw_bounds, path_grid, path_world):
        scenario = copy.deepcopy(self.scenario("D1"))
        event = copy.deepcopy(scenario["events"][0])
        event["obstacle"]["bounds"] = raw_bounds
        state = apply_obstacle_event(initial_obstacle_state(scenario), event)
        world = build_dynamic_collision_world(self.config, state, self.zones, self.physical, self.readiness)
        planner = copy.deepcopy(self.planner)
        planner["grid_resolution_m"] = 0.3
        grid = build_grid_map(world, planner)
        return validate_current_path(path_grid, path_world, world, grid)

    def test_22_cardinal_edge_blocked_between_free_waypoints(self):
        result = self._coarse_event_validation(
            {"x_min": 0.149, "x_max": 0.151, "y_min": 0.29, "y_max": 0.31},
            [[0, 0], [1, 0]], [[0.15, 0.15], [0.15, 0.45]],
        )
        self.assertEqual(result["reason"], "EDGE_BLOCKED")
        self.assertEqual(result["first_blocked_edge_index"], 0)

    def test_23_corner_cut_invalidation_after_obstacle_update(self):
        result = self._coarse_event_validation(
            {"x_min": 0.149, "x_max": 0.151, "y_min": 0.449, "y_max": 0.451},
            [[0, 0], [1, 1]], [[0.15, 0.15], [0.45, 0.45]],
        )
        self.assertEqual(result["reason"], "CORNER_CUT_BLOCKED")
        self.assertEqual(result["blocking_obstacle_ids"], ["dynamic_box_01"])

    def test_24_path_outside_valid_bounds_rejected(self):
        world, grid, _ = self.contexts["D1"][0]
        result = validate_current_path([[0, 0]], [[0.0, 0.0]], world, grid)
        self.assertEqual(result["reason"], "WORKSPACE_INVALID")
        self.assertEqual(result["blocking_obstacle_ids"], ["WORKSPACE_BOUNDARY"])

    def test_25_invalid_geometry_rejected(self):
        self.assert_config_error(lambda value: value["scenarios"][0]["events"][0]["obstacle"]["bounds"].update(x_max=0.1), "INVALID_RECTANGLE")

    def test_26_frame_mismatch_rejected(self):
        self.assert_config_error(lambda value: value["scenarios"][0]["events"][0]["obstacle"].update(coordinate_frame="camera"), "FRAME_MISMATCH")

    def test_27_unit_mismatch_rejected(self):
        self.assert_config_error(lambda value: value["scenarios"][0]["events"][0]["obstacle"].update(unit="mm"), "UNIT_MISMATCH")

    def test_28_nonfinite_event_geometry_rejected(self):
        self.assert_config_error(lambda value: value["scenarios"][0]["events"][0]["obstacle"]["bounds"].update(x_min=math.nan), "NON_FINITE_GEOMETRY")

    def test_29_deterministic_event_sequence(self):
        scenario = self.scenario("D5")
        repeated, _ = runner.run_scenario(self.config, scenario, self.zones, self.physical, self.readiness, self.planner)
        self.assertEqual(runner.semantic_result(self.results["D5"]), runner.semantic_result(repeated))

    def test_30_step12_through_step16_artifacts_unchanged(self):
        before = {path: path.read_bytes() for directory in PROTECTED_DIRS for path in directory.rglob("*") if path.is_file()}
        scenario = self.scenario("D2")
        runner.run_scenario(self.config, scenario, self.zones, self.physical, self.readiness, self.planner)
        after = {path: path.read_bytes() for directory in PROTECTED_DIRS for path in directory.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_31_unsupported_operation_rejected(self):
        self.assert_config_error(lambda value: value["scenarios"][0]["events"][0].update(operation="REPLACE"), "UNSUPPORTED_OPERATION")

    def test_32_missing_provenance_rejected(self):
        self.assert_config_error(lambda value: value["scenarios"][0]["events"][0]["obstacle"].update(source_detail=""), "MISSING_PROVENANCE")

    def test_33_outside_workspace_rejected(self):
        self.assert_config_error(lambda value: value["scenarios"][0]["events"][0]["obstacle"]["bounds"].update(x_max=1.3), "OBSTACLE_OUTSIDE_WORKSPACE")

    def test_34_invalid_event_order_rejected(self):
        self.assert_config_error(lambda value: value["scenarios"][0]["events"][0].update(event_index=2), "INVALID_EVENT_ORDER")

    def test_35_replan_required_logic(self):
        self.assertTrue(self.results["D2"]["events"][-1]["replan_required"])
        self.assertFalse(self.results["D1"]["events"][-1]["replan_required"])

    def test_36_no_post_event_planning(self):
        self.assertTrue(all(result["post_event_planner_run_count"] == 0 for result in self.results.values()))


if __name__ == "__main__":
    unittest.main()
