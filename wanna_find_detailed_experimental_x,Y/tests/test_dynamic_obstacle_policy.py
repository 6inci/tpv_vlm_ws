import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(WORKSPACE / "scripts"))

import run_dynamic_obstacle_policy as runner
from vlm_reposition.planning import (
    ObstaclePolicyError,
    load_obstacle_policy_config,
    load_planner_config,
    policy_scenario_by_id,
    validate_obstacle_policy_config,
)
from vlm_reposition.world import load_physical_geometry, load_zones


PROTECTED_DIRS = [
    WORKSPACE / "results" / name
    for name in (
        "workspace_state", "physical_geometry", "collision_geometry", "path_planning_baseline",
        "path_planning_stress", "dynamic_path_invalidation", "automatic_replanning",
    )
]


def manifest(directories):
    result = {}
    for directory in directories:
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            result[str(path.relative_to(WORKSPACE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class DynamicObstaclePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protected_before = manifest(PROTECTED_DIRS)
        cls.config = load_obstacle_policy_config(WORKSPACE / "config" / "dynamic_obstacle_policy.json")
        cls.planner = load_planner_config(WORKSPACE / "config" / "planner_baseline.json")
        cls.zones = load_zones(WORKSPACE / "config" / "zones.json")
        cls.physical = load_physical_geometry(WORKSPACE / "config" / "physical_geometry.json")
        cls.readiness = json.loads(
            (WORKSPACE / "results" / "physical_geometry" / "physical_geometry_readiness.json").read_text()
        )["planning_geometry_readiness"]
        cls.results, cls.contexts = {}, {}
        for scenario in cls.config["scenarios"]:
            result, context = runner.run_scenario(
                cls.config, scenario, cls.zones, cls.physical, cls.readiness, cls.planner
            )
            cls.results[scenario["scenario_id"]] = result
            cls.contexts[scenario["scenario_id"]] = context

    def scenario(self, scenario_id):
        return policy_scenario_by_id(self.config, scenario_id)

    def test_01_policy_config_validation(self):
        self.assertEqual([item["scenario_id"] for item in self.config["scenarios"]], [f"P{i}" for i in range(1, 12)])

    def test_02_logical_observation_time_validation(self):
        invalid = copy.deepcopy(self.config)
        invalid["scenarios"][0]["observations"][1]["logical_time_s"] = 0.19
        with self.assertRaises(ObstaclePolicyError) as caught:
            validate_obstacle_policy_config(invalid)
        self.assertEqual(caught.exception.code, "INVALID_LOGICAL_TIME")

    def test_03_development_parameters_explicit(self):
        parameters = self.config["parameters"]
        self.assertEqual((parameters["observation_interval_s"], parameters["persistence_threshold_s"]), (0.2, 1.0))
        self.assertEqual(parameters["parameter_status"], "DEVELOPMENT_PARAMETER")

    def test_04_single_frame_presence_pending(self):
        observation = self.results["P1"]["observations"][0]
        self.assertEqual(observation["temporal_obstacle_states"]["p1_box"]["state"], "PRESENCE_PENDING")
        self.assertFalse(observation["confirmed_obstacle_ids"])

    def test_05_presence_confirmation_count(self):
        observation = self.results["P1"]["observations"][1]
        state = observation["temporal_obstacle_states"]["p1_box"]
        self.assertEqual((state["state"], state["consecutive_presence"]), ("CONFIRMED_PRESENT", 2))

    def test_06_single_frame_absence_pending(self):
        observation = self.results["P1"]["observations"][3]
        self.assertEqual(observation["temporal_obstacle_states"]["p1_box"]["state"], "ABSENCE_PENDING")
        self.assertIn("p1_box", observation["confirmed_obstacle_ids"])

    def test_07_absence_confirmation_count(self):
        observation = self.results["P1"]["observations"][4]
        state = observation["temporal_obstacle_states"]["p1_box"]
        self.assertEqual(state["state"], "CLEARED")
        self.assertFalse(state["confirmed_present"])

    def test_08_raw_pending_does_not_mutate_world(self):
        observation = self.results["P1"]["observations"][0]
        self.assertEqual(observation["world_revision"], 0)
        self.assertFalse(observation["confirmed_world_changed"])

    def test_09_confirmation_updates_world_once(self):
        observations = self.results["P1"]["observations"]
        self.assertEqual([item["world_revision"] for item in observations], [0, 1, 1, 1, 2])

    def test_10_blocking_since_recorded(self):
        episode = self.results["P1"]["blocking_episodes"][0]
        self.assertEqual((episode["start_observation_index"], episode["blocking_since_s"]), (1, 0.2))

    def test_11_persistence_duration_is_logical_time(self):
        observations = self.results["P2"]["observations"]
        durations = [item["blocking_duration_s"] for item in observations[1:]]
        self.assertEqual(durations, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def test_12_below_threshold_is_temporary(self):
        episode = self.results["P7"]["blocking_episodes"][0]
        self.assertEqual((episode["classification"], episode["blocking_duration_s"]), ("TEMPORARY_BLOCKING", 0.8))

    def test_13_equal_threshold_is_persistent(self):
        episode = self.results["P8"]["blocking_episodes"][0]
        self.assertEqual((episode["classification"], episode["blocking_duration_s"]), ("PERSISTENT_BLOCKING", 1.0))

    def test_14_p1_temporary_blocker_stops(self):
        self.assertEqual(self.results["P1"]["observations"][1]["navigation_state"], "STOPPED_FOR_OBSTACLE")

    def test_15_p1_temporary_blocker_waits(self):
        self.assertEqual(self.results["P1"]["observations"][2]["navigation_state"], "WAITING_FOR_CLEARANCE")

    def test_16_p1_clear_resumes(self):
        self.assertEqual(self.results["P1"]["observations"][-1]["action"], "RESUME_CURRENT_PATH")

    def test_17_p1_has_no_planner_run(self):
        self.assertEqual(self.results["P1"]["temporal_metrics"]["planner_run_count"], 0)

    def test_18_p1_path_revision_unchanged(self):
        result = self.results["P1"]
        self.assertEqual(result["final_active_path"]["path_revision"], 0)
        self.assertEqual(result["initial_active_path"], result["final_active_path"])

    def test_19_p2_exactly_one_replan(self):
        result = self.results["P2"]
        self.assertEqual(result["temporal_metrics"]["planner_run_count"], 1)
        self.assertEqual(sum(item["planner_runs_this_observation"] for item in result["observations"]), 1)

    def test_20_p2_replan_starts_at_stopped_pose(self):
        final = self.results["P2"]["observations"][-1]
        self.assertEqual(final["replan_result"]["current_robot_pose"]["waypoint_index"], 20)
        self.assertEqual(final["replan_result"]["current_robot_pose"]["x"], 0.61)

    def test_21_p2_switches_to_replanned_path(self):
        result = self.results["P2"]
        self.assertEqual(result["observations"][-1]["action"], "SWITCH_TO_REPLANNED_PATH")
        self.assertEqual(result["final_active_path"]["path_revision"], 1)

    def test_22_original_goal_preserved(self):
        self.assertEqual(self.results["P2"]["final_active_path"]["goal_pose"]["mission_input_m"], [0.95, 1.5])

    def test_23_current_robot_pose_is_replan_start(self):
        result = self.results["P2"]
        pose = result["observations"][-1]["replan_result"]["current_robot_pose"]
        self.assertEqual(result["final_active_path"]["start_pose"]["cell_center_m"], [pose["x"], pose["y"]])

    def test_24_path_lineage_preserved(self):
        lineage = self.results["P2"]["final_active_path"]["lineage"]
        self.assertEqual((lineage["previous_path_revision"], lineage["new_path_revision"]), (0, 1))
        self.assertEqual(lineage["trigger_event_id"], "P2_persistence_episode_001")

    def test_25_p3_no_path_stops(self):
        result = self.results["P3"]
        self.assertEqual(result["final_navigation_state"], "STOP_REQUIRED")
        self.assertEqual(result["observations"][-1]["replan_status"], "REPLAN_FAILED")

    def test_26_p3_has_no_partial_active_path(self):
        self.assertIsNone(self.results["P3"]["final_active_path"])
        self.assertIsNone(self.results["P3"]["observations"][-1]["replan_result"]["new_active_path"])

    def test_27_p4_flicker_debounce(self):
        observations = self.results["P4"]["observations"]
        self.assertEqual([item["world_revision"] for item in observations[:3]], [0, 0, 0])
        self.assertEqual(observations[3]["world_revision"], 1)

    def test_28_p4_has_no_unnecessary_replan(self):
        self.assertEqual(self.results["P4"]["temporal_metrics"]["planner_run_count"], 0)

    def test_29_p5_off_path_never_stops(self):
        states = [item["navigation_state"] for item in self.results["P5"]["observations"]]
        self.assertEqual(set(states), {"MOVING"})

    def test_30_p5_off_path_has_no_episode_or_replan(self):
        result = self.results["P5"]
        self.assertFalse(result["blocking_episodes"])
        self.assertEqual(result["temporal_metrics"]["planner_run_count"], 0)

    def test_31_p6_soft_risk_never_stops(self):
        self.assertTrue(all(item["action"] == "CONTINUE_CURRENT_PATH" for item in self.results["P6"]["observations"]))

    def test_32_p6_soft_risk_increases(self):
        observations = self.results["P6"]["observations"]
        self.assertGreater(observations[1]["accumulated_safety_risk"], observations[0]["accumulated_safety_risk"])

    def test_33_p6_soft_risk_has_no_replan(self):
        self.assertEqual(self.results["P6"]["temporal_metrics"]["planner_run_count"], 0)

    def test_34_p7_clear_before_threshold_no_replan(self):
        result = self.results["P7"]
        self.assertEqual(result["blocking_episodes"][0]["blocking_duration_s"], 0.8)
        self.assertEqual(result["temporal_metrics"]["planner_run_count"], 0)

    def test_35_p8_threshold_triggers_replan(self):
        observation = self.results["P8"]["observations"][-1]
        self.assertEqual((observation["blocking_duration_s"], observation["replan_status"]), (1.0, "REPLAN_SUCCESS"))

    def test_36_p9_other_blocker_prevents_resume(self):
        actions = [item["action"] for item in self.results["P9"]["observations"]]
        self.assertNotIn("RESUME_CURRENT_PATH", actions)
        self.assertEqual(actions[-1], "STOP_AND_WAIT")

    def test_37_p9_second_blocker_remains_confirmed(self):
        self.assertEqual(self.results["P9"]["observations"][-1]["confirmed_obstacle_ids"], ["p9_box_b"])

    def test_38_wait_does_not_advance_robot_progress(self):
        waits = [
            item["current_robot_pose"]["waypoint_index"]
            for item in self.results["P2"]["observations"]
            if item["navigation_state"] in {"STOPPED_FOR_OBSTACLE", "WAITING_FOR_CLEARANCE"}
        ]
        self.assertEqual(set(waits), {20})

    def test_39_confirmed_world_is_collision_source(self):
        observations = self.results["P1"]["observations"]
        self.assertEqual((observations[0]["path_state"], observations[0]["confirmed_obstacle_ids"]), ("VALID", []))
        self.assertEqual((observations[1]["path_state"], observations[1]["confirmed_obstacle_ids"]), ("BLOCKED", ["p1_box"]))

    def test_40_robot_footprint_is_applied_once(self):
        world = self.contexts["P1"]["blocking_world"]
        obstacle = next(item for item in world["static_obstacles"] if item["id"] == "p1_box")
        half = world["robot"]["half_extents_m"]
        self.assertAlmostEqual(obstacle["raw_bounds"]["x_min"] - obstacle["collision_bounds"]["x_min"], half["x"])
        self.assertAlmostEqual(obstacle["raw_bounds"]["y_min"] - obstacle["collision_bounds"]["y_min"], half["y"])

    def test_41_same_episode_does_not_repeat_replan(self):
        result = self.results["P10"]
        self.assertEqual(len(result["blocking_episodes"]), 1)
        self.assertEqual(sum(item["planner_runs_this_observation"] for item in result["observations"]), 1)

    def test_42_new_episode_can_later_replan(self):
        result = self.results["P11"]
        self.assertEqual([item["blocking_episode_id"] for item in result["blocking_episodes"]], ["episode_001", "episode_002"])
        self.assertEqual([item["classification"] for item in result["blocking_episodes"]], ["TEMPORARY_BLOCKING", "PERSISTENT_BLOCKING"])
        self.assertEqual(result["temporal_metrics"]["planner_run_count"], 1)

    def test_43_p10_clear_keeps_replanned_path(self):
        result = self.results["P10"]
        self.assertEqual(result["observations"][-1]["action"], "CONTINUE_CURRENT_PATH")
        self.assertEqual(result["final_active_path"]["path_revision"], 1)

    def test_44_temporal_results_are_deterministic(self):
        scenario = self.scenario("P2")
        repeated, _ = runner.run_scenario(
            self.config, scenario, self.zones, self.physical, self.readiness, self.planner
        )
        self.assertEqual(runner.semantic_result(self.results["P2"]), runner.semantic_result(repeated))

    def test_45_temporal_metrics_record_confirmation_latency(self):
        metrics = self.results["P1"]["temporal_metrics"]
        self.assertEqual(metrics["time_to_presence_confirmation_s"], [0.2])
        self.assertEqual(metrics["time_to_clear_confirmation_s"], [0.2])

    def test_46_step_12_through_18_artifacts_unchanged(self):
        self.assertEqual(self.protected_before, manifest(PROTECTED_DIRS))


if __name__ == "__main__":
    unittest.main()
