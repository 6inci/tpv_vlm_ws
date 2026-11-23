import copy
import json
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(WORKSPACE / "scripts"))

import run_automatic_replanning as runner
from vlm_reposition.planning import (
    ReplanningError,
    apply_obstacle_event,
    build_dynamic_collision_world,
    build_grid_map,
    create_active_path,
    current_robot_pose,
    evaluate_event_and_replan,
    initial_replanning_state,
    load_planner_config,
    load_replanning_config,
    plan_path,
    replanning_scenario_by_id,
    validate_current_path,
    validate_replanning_config,
)
from vlm_reposition.world import load_physical_geometry, load_zones


PROTECTED_DIRS = [
    WORKSPACE / "results" / name
    for name in (
        "workspace_state", "physical_geometry", "collision_geometry", "path_planning_baseline",
        "path_planning_stress", "dynamic_path_invalidation",
    )
]


class AutomaticReplanningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_replanning_config(WORKSPACE / "config" / "automatic_replanning_scenarios.json")
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
        return replanning_scenario_by_id(self.config, scenario_id)

    def initial_runtime(self, scenario_id="R1"):
        scenario = self.scenario(scenario_id)
        planner = runner.planner_config_for_scenario(self.planner, scenario)
        state = initial_replanning_state(scenario)
        world = build_dynamic_collision_world(self.config, state, self.zones, self.physical, self.readiness)
        grid = build_grid_map(world, planner)
        result = plan_path(world, grid, planner, self.config["planner"], scenario["initial_start"], scenario["goal"])
        active = create_active_path(result, scenario_id, 0, 0, scenario["goal"])
        return scenario, planner, state, world, grid, active

    def endpoint_failure(self, bounds):
        scenario, planner, state, _, _, active = self.initial_runtime()
        event = copy.deepcopy(scenario["events"][0])
        event["event_id"] = "TEST_ENDPOINT_FAILURE"
        event["obstacle"]["id"] = "test_endpoint_box"
        event["obstacle"]["bounds"] = bounds
        return evaluate_event_and_replan(
            self.config, scenario, planner, self.zones, self.physical, self.readiness,
            state, active, [], event,
        )[-1]

    def test_01_replanning_config_validation(self):
        self.assertEqual([item["scenario_id"] for item in self.config["scenarios"]], ["R1", "R2", "R3", "R4", "R5", "R6", "R7"])

    def test_02_initial_active_path_revision_zero(self):
        self.assertTrue(all(result["initial_active_path"]["path_revision"] == 0 for result in self.results.values()))

    def test_03_current_robot_waypoint_extraction(self):
        pose = current_robot_pose(self.results["R2"]["initial_active_path"], 20)
        self.assertEqual(pose["grid"], [16, 30])
        self.assertEqual([pose["x"], pose["y"]], [0.61, 0.33])

    def test_04_invalid_robot_waypoint_index_rejected(self):
        with self.assertRaises(ReplanningError) as caught:
            current_robot_pose(self.results["R1"]["initial_active_path"], 999)
        self.assertEqual(caught.exception.code, "INVALID_ROBOT_PROGRESS")

    def test_05_backward_progress_rejected(self):
        with self.assertRaises(ReplanningError) as caught:
            current_robot_pose(self.results["R1"]["initial_active_path"], 10, 11)
        self.assertEqual(caught.exception.code, "BACKWARD_ROBOT_PROGRESS")

    def test_06_valid_path_event_has_no_planner_rerun(self):
        self.assertEqual(self.results["R1"]["events"][0]["planner_runs_after_event"], 0)

    def test_07_r1_active_path_unchanged(self):
        result = self.results["R1"]
        self.assertEqual(result["initial_active_path"], result["final_active_path"])
        self.assertEqual(result["events"][0]["system_action"], "CONTINUE_CURRENT_PATH")

    def test_08_r2_blocked_detected(self):
        self.assertEqual(self.results["R2"]["events"][0]["old_path_state"], "BLOCKED")

    def test_09_r2_replan_executed(self):
        event = self.results["R2"]["events"][0]
        self.assertEqual(event["planner_runs_after_event"], 1)
        self.assertEqual(event["replanning"]["status"], "REPLAN_SUCCESS")

    def test_10_r2_new_path_found(self):
        self.assertEqual(self.results["R2"]["events"][0]["replanning"]["planner_result"], "PATH_FOUND")

    def test_11_r2_new_path_collision_free(self):
        validation = self.results["R2"]["events"][0]["replanned_path_validation"]
        self.assertEqual(validation["status"], "VALID")
        self.assertTrue(all(validation["checks"].values()))

    def test_12_r2_replan_starts_at_current_pose(self):
        result = self.results["R2"]
        event = result["events"][0]
        new_path = result["final_active_path"]
        self.assertEqual(new_path["start_pose"]["cell_center_m"], [event["current_robot_pose"]["x"], event["current_robot_pose"]["y"]])

    def test_13_r2_original_goal_retained(self):
        result = self.results["R2"]
        self.assertEqual(result["final_active_path"]["goal_pose"]["mission_input_m"], [0.95, 1.5])

    def test_14_r2_path_revision_increments(self):
        self.assertEqual(self.results["R2"]["final_active_path"]["path_revision"], 1)

    def test_15_r2_old_path_superseded(self):
        self.assertEqual(self.results["R2"]["path_records"][0]["status"], "SUPERSEDED")

    def test_16_r3_move_triggers_replan(self):
        event = self.results["R3"]["events"][0]
        self.assertEqual((event["operation"], event["replanning"]["status"]), ("MOVE", "REPLAN_SUCCESS"))

    def test_17_r4_no_path_replan_failed(self):
        event = self.results["R4"]["events"][0]
        self.assertEqual(event["replanning"]["planner_result"], "NO_PATH")
        self.assertEqual(event["replanning"]["status"], "REPLAN_FAILED")

    def test_18_r4_stop_required(self):
        event = self.results["R4"]["events"][0]
        self.assertTrue(event["stop_required"])
        self.assertEqual(event["system_action"], "STOP")

    def test_19_r4_no_partial_path_activated(self):
        result = self.results["R4"]
        self.assertIsNone(result["final_active_path"])
        self.assertIsNone(result["events"][0]["new_active_path"])

    def test_20_r5_soft_risk_change_no_replan(self):
        event = self.results["R5"]["events"][0]
        self.assertEqual(event["old_path_state"], "VALID")
        self.assertGreater(event["old_path_accumulated_safety_risk"], event["accumulated_safety_risk_before_event"])
        self.assertEqual(event["planner_runs_after_event"], 0)

    def test_21_r5_path_revision_unchanged(self):
        self.assertEqual(self.results["R5"]["final_active_path"]["path_revision"], 0)

    def test_22_r6_first_replan_revision_one(self):
        self.assertEqual(self.results["R6"]["events"][0]["new_active_path"]["path_revision"], 1)

    def test_23_r6_second_replan_revision_two(self):
        result = self.results["R6"]
        self.assertEqual(result["events"][1]["new_active_path"]["path_revision"], 2)
        self.assertEqual(result["final_active_path"]["path_revision"], 2)

    def test_24_r6_path_lineage_correct(self):
        records = self.results["R6"]["path_records"]
        self.assertEqual([item["path_revision"] for item in records], [0, 1, 2])
        self.assertEqual(records[1]["lineage"]["previous_path_revision"], 0)
        self.assertEqual(records[2]["lineage"]["previous_path_revision"], 1)

    def test_25_r7_removed_blocker_keeps_valid_active_path(self):
        result = self.results["R7"]
        self.assertEqual(result["events"][1]["old_path_state"], "VALID")
        self.assertEqual(result["final_active_path"]["path_revision"], 1)

    def test_26_no_automatic_original_path_restoration(self):
        result = self.results["R7"]
        self.assertEqual(result["events"][1]["planner_runs_after_event"], 0)
        self.assertEqual(result["final_active_path"]["path_id"], "R7_path_r1")
        self.assertNotEqual(result["final_active_path"]["path_id"], result["initial_active_path"]["path_id"])

    def test_27_robot_pose_blocked_explicit_failure(self):
        event = self.endpoint_failure({"x_min": 0.60, "x_max": 0.62, "y_min": 0.32, "y_max": 0.34})
        self.assertEqual(event["blocking_reason"], "ROBOT_POSE_INVALID")
        self.assertEqual(event["replanning"]["status"], "NOT_RUN")
        self.assertTrue(event["stop_required"])

    def test_28_goal_blocked_explicit_failure(self):
        event = self.endpoint_failure({"x_min": 0.94, "x_max": 0.96, "y_min": 1.49, "y_max": 1.51})
        self.assertEqual(event["blocking_reason"], "GOAL_BLOCKED")
        self.assertEqual(event["replanning"]["status"], "NOT_RUN")
        self.assertEqual(event["system_action"], "STOP")

    def test_29_no_double_footprint_inflation(self):
        world = self.contexts["R2"][1]["world"]
        obstacle = next(item for item in world["static_obstacles"] if item["id"] == "r2_box")
        self.assertEqual(obstacle["collision_bounds"], {"x_min": 0.735, "x_max": 1.065, "y_min": 0.727, "y_max": 1.073})
        self.assertEqual(self.results["R2"]["events"][0]["additional_inflation_m"], 0.0)

    def test_30_new_costmap_used_for_replanning(self):
        event = self.results["R2"]["events"][0]
        revisions = self.results["R2"]["world_revisions"]
        self.assertEqual(event["costmap_world_revision"], event["world_revision_after"])
        self.assertNotEqual(revisions[0]["grid"]["occupancy_sha256"], revisions[1]["grid"]["occupancy_sha256"])

    def _coarse_validation(self, bounds, path_grid, path_world):
        scenario, planner, state, _, _, _ = self.initial_runtime()
        event = copy.deepcopy(scenario["events"][0])
        event["obstacle"]["id"] = "coarse_box"
        event["obstacle"]["bounds"] = bounds
        state = apply_obstacle_event(state, event)
        world = build_dynamic_collision_world(self.config, state, self.zones, self.physical, self.readiness)
        planner["grid_resolution_m"] = 0.3
        grid = build_grid_map(world, planner)
        return validate_current_path(path_grid, path_world, world, grid)

    def test_31_diagonal_corner_cutting_still_forbidden(self):
        result = self._coarse_validation(
            {"x_min": 0.149, "x_max": 0.151, "y_min": 0.449, "y_max": 0.451},
            [[0, 0], [1, 1]], [[0.15, 0.15], [0.45, 0.45]],
        )
        self.assertEqual(result["reason"], "CORNER_CUT_BLOCKED")

    def test_32_continuous_edge_collision_still_validated(self):
        result = self._coarse_validation(
            {"x_min": 0.149, "x_max": 0.151, "y_min": 0.29, "y_max": 0.31},
            [[0, 0], [1, 0]], [[0.15, 0.15], [0.15, 0.45]],
        )
        self.assertEqual(result["reason"], "EDGE_BLOCKED")

    def test_33_deterministic_replanned_path(self):
        scenario = self.scenario("R2")
        repeated, _ = runner.run_scenario(self.config, scenario, self.zones, self.physical, self.readiness, self.planner)
        self.assertEqual(runner.semantic_result(self.results["R2"]), runner.semantic_result(repeated))

    def test_34_deterministic_path_revisions(self):
        self.assertEqual([item["path_revision"] for item in self.results["R6"]["path_records"]], [0, 1, 2])

    def test_35_step12_through_step17_artifacts_unchanged(self):
        before = {path: path.read_bytes() for directory in PROTECTED_DIRS for path in directory.rglob("*") if path.is_file()}
        scenario = self.scenario("R6")
        runner.run_scenario(self.config, scenario, self.zones, self.physical, self.readiness, self.planner)
        after = {path: path.read_bytes() for directory in PROTECTED_DIRS for path in directory.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_36_replanning_does_not_use_original_start(self):
        event = self.results["R2"]["events"][0]
        new_start = self.results["R2"]["final_active_path"]["start_pose"]["input_m"]
        self.assertEqual(new_start, [event["current_robot_pose"]["x"], event["current_robot_pose"]["y"]])
        self.assertNotEqual(new_start, [0.2, 0.3])

    def test_37_state_machine_traces(self):
        self.assertEqual(self.results["R1"]["events"][0]["state_trace"], ["PATH_VALID"])
        self.assertEqual(self.results["R2"]["events"][0]["state_trace"], ["REPLAN_REQUIRED", "REPLANNING", "REPLAN_SUCCESS"])
        self.assertEqual(self.results["R4"]["events"][0]["state_trace"], ["REPLAN_REQUIRED", "REPLANNING", "REPLAN_FAILED", "STOP_REQUIRED"])

    def test_38_replan_metrics_recorded(self):
        replanning = self.results["R2"]["events"][0]["replanning"]
        self.assertGreater(replanning["nodes_expanded"], 0)
        self.assertGreater(replanning["new_path_length_m"], 0)
        self.assertGreater(replanning["waypoint_count"], 0)

    def test_39_safety_cost_planner_only(self):
        self.assertTrue(all(result["planner"] == "SAFETY_COST_ASTAR" for result in self.results.values()))
        self.assertTrue(all(record["source_planner"] == "SAFETY_COST_ASTAR" for result in self.results.values() for record in result["path_records"]))

    def test_40_invalid_config_progress_rejected(self):
        invalid = copy.deepcopy(self.config)
        invalid["scenarios"][0]["events"][0]["robot_waypoint_index"] = -1
        with self.assertRaises(ReplanningError) as caught:
            validate_replanning_config(invalid)
        self.assertEqual(caught.exception.code, "INVALID_ROBOT_PROGRESS")


if __name__ == "__main__":
    unittest.main()
