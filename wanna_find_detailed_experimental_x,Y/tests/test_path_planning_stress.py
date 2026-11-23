import copy
import json
import math
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(WORKSPACE / "scripts"))

import run_path_planning_stress as stress_runner
from vlm_reposition.planning import clearance_at, load_planner_config, semantic_path_result
from vlm_reposition.world import load_physical_geometry, load_zones


PROTECTED_DIRS = [
    WORKSPACE / "results" / "workspace_state",
    WORKSPACE / "results" / "physical_geometry",
    WORKSPACE / "results" / "collision_geometry",
    WORKSPACE / "results" / "path_planning_baseline",
]


class PathPlanningStressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stress = stress_runner.load_stress_config(WORKSPACE / "config" / "planner_stress_scenarios.json")
        cls.planner = load_planner_config(WORKSPACE / "config" / "planner_baseline.json")
        cls.zones = load_zones(WORKSPACE / "config" / "zones.json")
        cls.physical = load_physical_geometry(WORKSPACE / "config" / "physical_geometry.json")
        cls.readiness = json.loads((WORKSPACE / "results" / "physical_geometry" / "physical_geometry_readiness.json").read_text())["planning_geometry_readiness"]
        cls.results, cls.contexts = {}, {}
        for scenario in cls.stress["scenarios"]:
            output = stress_runner.run_scenario(
                cls.stress, scenario, cls.zones, cls.physical, cls.readiness, cls.planner
            )
            cls.results[scenario["scenario_id"]] = output[0]
            cls.contexts[scenario["scenario_id"]] = output[1:]
        scenario = stress_runner.scenario_by_id(cls.stress, "S1_baseline")
        cls.s1_world = cls.contexts["S1_baseline"][0]
        cls.resolution = stress_runner.run_resolution_sweep(cls.s1_world, scenario, cls.planner)
        cls.radius = stress_runner.run_safety_radius_sweep(cls.s1_world, scenario, cls.planner)
        cls.weight = stress_runner.run_safety_weight_sweep(cls.s1_world, scenario, cls.planner)

    def test_01_stress_scenario_config_validation(self):
        self.assertEqual(len(self.stress["scenarios"]), 6)
        self.assertTrue(all(item["geometry_status"] == "DEVELOPMENT_SYNTHETIC" for item in self.stress["scenarios"]))

    def test_02_multiple_obstacle_loading(self):
        scenario = stress_runner.scenario_by_id(self.stress, "S3_two_obstacles")
        self.assertEqual(len(scenario["static_obstacles"]), 2)
        self.assertEqual(len(self.contexts["S3_two_obstacles"][0]["static_obstacles"]), 2)

    def test_03_s1_regression_path_found(self):
        result = self.results["S1_baseline"]
        self.assertEqual(result["distance_only_astar"]["status"], "PATH_FOUND")
        baseline = json.loads((WORKSPACE / "results" / "path_planning_baseline" / "distance_only_astar.json").read_text())
        self.assertEqual(semantic_path_result(result["distance_only_astar"]), semantic_path_result(baseline))

    def test_04_s2_path_found(self):
        result = self.results["S2_shifted_left"]
        self.assertEqual(result["distance_only_astar"]["status"], "PATH_FOUND")
        self.assertEqual(result["safety_cost_astar"]["status"], "PATH_FOUND")

    def test_05_s3_multi_obstacle_path_found(self):
        result = self.results["S3_two_obstacles"]
        self.assertEqual(result["distance_only_astar"]["status"], "PATH_FOUND")
        self.assertEqual(result["safety_cost_astar"]["status"], "PATH_FOUND")

    def test_06_s4_narrow_corridor_geometry_valid(self):
        world, grid, _ = self.contexts["S4_narrow_passage"]
        left, right = world["static_obstacles"]
        gap = right["collision_bounds"]["x_min"] - left["collision_bounds"]["x_max"]
        self.assertAlmostEqual(gap, 0.03)
        free_columns = [col for col in range(grid["cols"]) if grid["occupancy"][40][col] == 0]
        self.assertGreaterEqual(len(free_columns), 1)

    def test_07_s4_paths_collision_free(self):
        validation = self.results["S4_narrow_passage"]["independent_validation"]
        self.assertEqual(validation["distance_only_astar"]["status"], "PATH_VALID")
        self.assertEqual(validation["safety_cost_astar"]["status"], "PATH_VALID")

    def test_08_s5_distance_no_path(self):
        result = self.results["S5_completely_blocked"]["distance_only_astar"]
        self.assertEqual(result["status"], "NO_PATH")
        self.assertEqual(result["path_grid"], [])

    def test_09_s5_safety_no_path(self):
        result = self.results["S5_completely_blocked"]["safety_cost_astar"]
        self.assertEqual(result["status"], "NO_PATH")
        self.assertEqual(result["path_world"], [])

    def test_10_s6_alternate_endpoints_valid(self):
        result = self.results["S6_alternate_endpoints"]
        self.assertEqual(result["distance_only_astar"]["start_input_m"], [0.95, 0.3])
        self.assertEqual(result["distance_only_astar"]["goal_input_m"], [0.2, 1.5])
        self.assertTrue(result["safety_cost_astar"]["collision_free"])

    def test_11_multiple_obstacle_clearance(self):
        world = self.contexts["S3_two_obstacles"][0]
        self.assertAlmostEqual(clearance_at(0.5, 0.4, world), 0.127)

    def test_12_overlapping_collision_obstacles_supported(self):
        obstacles = self.contexts["S3_two_obstacles"][0]["static_obstacles"]
        first, second = obstacles[0]["collision_bounds"], obstacles[1]["collision_bounds"]
        self.assertLess(max(first["x_min"], second["x_min"]), min(first["x_max"], second["x_max"]))
        self.assertLess(max(first["y_min"], second["y_min"]), min(first["y_max"], second["y_max"]))

    def test_13_resolution_001_grid_valid(self):
        entry = self.resolution["entries"][0]
        self.assertEqual(entry["grid_cell_count"], 21600)
        self.assertTrue(entry["distance_only_astar"]["collision_free"])

    def test_14_resolution_002_regression(self):
        entry = self.resolution["entries"][1]
        self.assertEqual(entry["grid_cell_count"], 5400)
        self.assertAlmostEqual(entry["distance_only_astar"]["path_length_m"], 1.740832611206853)

    def test_15_resolution_004_grid_valid(self):
        entry = self.resolution["entries"][2]
        self.assertEqual(entry["grid_cell_count"], 1350)
        self.assertTrue(entry["safety_cost_astar"]["collision_free"])

    def test_16_no_double_inflation_across_resolution(self):
        expected = [item["collision_bounds"] for item in self.s1_world["static_obstacles"]]
        for entry in self.resolution["entries"]:
            self.assertEqual(entry["collision_bounds"], expected)
            self.assertEqual(entry["additional_inflation_m"], 0.0)

    def test_17_safety_radius_010_valid(self):
        entry = self.radius["entries"][0]
        self.assertEqual(entry["safety_cost_radius_m"], 0.1)
        self.assertTrue(entry["safety_cost_astar"]["collision_free"])

    def test_18_safety_radius_020_regression(self):
        entry = self.radius["entries"][1]
        baseline = self.results["S1_baseline"]["safety_cost_astar"]
        self.assertEqual(entry["safety_cost_astar"]["path_length_m"], baseline["path_length_m"])

    def test_19_safety_radius_030_valid(self):
        entry = self.radius["entries"][2]
        self.assertEqual(entry["safety_cost_radius_m"], 0.3)
        self.assertTrue(entry["safety_cost_astar"]["collision_free"])

    def test_20_safety_radius_does_not_change_hard_occupancy(self):
        self.assertEqual(len({entry["occupancy_sha256"] for entry in self.radius["entries"]}), 1)

    def test_21_safety_weight_000_valid(self):
        self.assertTrue(self.weight["entries"][0]["safety_cost_astar"]["collision_free"])

    def test_22_safety_weight_050_valid(self):
        self.assertEqual(self.weight["entries"][1]["safety_weight"], 0.5)

    def test_23_safety_weight_100_valid(self):
        self.assertEqual(self.weight["entries"][2]["safety_weight"], 1.0)

    def test_24_safety_weight_200_regression(self):
        entry = self.weight["entries"][3]
        baseline = self.results["S1_baseline"]["safety_cost_astar"]
        self.assertEqual(entry["safety_cost_astar"]["path_length_m"], baseline["path_length_m"])
        self.assertEqual(entry["safety_cost_astar"]["accumulated_safety_risk"], baseline["accumulated_safety_risk"])

    def test_25_safety_weight_400_valid(self):
        self.assertTrue(self.weight["entries"][4]["safety_cost_astar"]["collision_free"])

    def test_26_safety_weight_does_not_change_hard_occupancy(self):
        self.assertEqual(len({entry["occupancy_sha256"] for entry in self.weight["entries"]}), 1)

    def test_27_all_found_paths_independently_validated(self):
        statuses = [
            validation[planner]["status"]
            for result in self.results.values()
            for planner, validation in (("distance_only_astar", result["independent_validation"]), ("safety_cost_astar", result["independent_validation"]))
            if result[planner]["status"] == "PATH_FOUND"
        ]
        self.assertEqual(statuses, ["PATH_VALID"] * 10)

    def test_28_scenario_paths_deterministic(self):
        self.assertTrue(all(all(result["determinism"].values()) for result in self.results.values()))

    def test_29_invalid_scenario_rejected(self):
        invalid = copy.deepcopy(self.stress)
        invalid["scenarios"][0]["geometry_status"] = "MEASURED"
        with self.assertRaises(stress_runner.StressConfigError):
            stress_runner.validate_stress_config(invalid)

    def test_30_step12_through_step15_artifacts_unchanged(self):
        before = {path: path.read_bytes() for directory in PROTECTED_DIRS for path in directory.iterdir()}
        scenario = stress_runner.scenario_by_id(self.stress, "S2_shifted_left")
        stress_runner.run_scenario(self.stress, scenario, self.zones, self.physical, self.readiness, self.planner)
        self.assertEqual(before, {path: path.read_bytes() for directory in PROTECTED_DIRS for path in directory.iterdir()})

    def test_31_zero_weight_matches_distance_objective(self):
        self.assertTrue(self.weight["zero_weight_distance_objective_equivalent"])
        zero = self.weight["entries"][0]["safety_cost_astar"]
        distance = self.results["S1_baseline"]["distance_only_astar"]
        self.assertTrue(math.isclose(zero["total_objective_cost"], distance["total_objective_cost"]))


if __name__ == "__main__":
    unittest.main()
