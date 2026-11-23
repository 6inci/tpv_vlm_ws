import copy
import json
import math
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.planning import (
    PlannerError,
    accumulated_safety_risk,
    astar_grid,
    build_grid_map,
    clearance_at,
    grid_to_metric,
    is_robot_center_valid,
    load_planner_config,
    metric_to_grid,
    neighbors,
    path_length_m,
    plan_path,
    safety_risk_from_clearance,
    semantic_path_result,
)


CONFIG_PATH = WORKSPACE / "config" / "planner_baseline.json"
WORLD_PATH = WORKSPACE / "results" / "collision_geometry" / "static_collision_world.json"
PROTECTED_DIRS = [
    WORKSPACE / "results" / "workspace_state",
    WORKSPACE / "results" / "physical_geometry",
    WORKSPACE / "results" / "collision_geometry",
]


def tiny_grid(occupancy, risks=None, resolution=1.0):
    if risks is None:
        risks = [[None if value else 0.0 for value in row] for row in occupancy]
    return {
        "rows": len(occupancy),
        "cols": len(occupancy[0]),
        "resolution_m": resolution,
        "origin_m": [0.0, 0.0],
        "occupancy": occupancy,
        "safety_risk": risks,
    }


class PathPlanningBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_planner_config(CONFIG_PATH)
        cls.world = json.loads(WORLD_PATH.read_text())
        cls.grid = build_grid_map(cls.world, cls.config)
        cls.distance = plan_path(cls.world, cls.grid, cls.config, "DISTANCE_ONLY_ASTAR")
        cls.safety = plan_path(cls.world, cls.grid, cls.config, "SAFETY_COST_ASTAR")

    def test_01_planner_config_load(self):
        self.assertEqual(self.config["parameter_status"], "DEVELOPMENT_PARAMETER")
        self.assertEqual(self.config["neighbor_order"], ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

    def test_02_grid_dimensions_and_resolution(self):
        self.assertEqual((self.grid["rows"], self.grid["cols"]), (90, 60))
        self.assertEqual(self.grid["resolution_m"], 0.02)

    def test_03_metric_to_grid_conversion(self):
        self.assertEqual(metric_to_grid(0.20, 0.30, self.grid), [15, 10])

    def test_04_grid_to_metric_conversion(self):
        self.assertEqual(grid_to_metric(15, 10, self.grid), [0.21, 0.31])

    def test_05_occupancy_uses_step14_collision_query(self):
        for x, y in ((0.21, 0.31), (0.51, 0.81), (0.95, 1.51)):
            row, col = metric_to_grid(x, y, self.grid)
            expected = 0 if self.world_point_is_free(x, y) else 1
            self.assertEqual(self.grid["occupancy"][row][col], expected)

    def world_point_is_free(self, x, y):
        return is_robot_center_valid(x, y, self.world)

    def test_06_expanded_obstacle_cells_are_occupied(self):
        row, col = metric_to_grid(0.51, 0.81, self.grid)
        self.assertEqual(self.grid["occupancy"][row][col], 1)
        self.assertIsNone(self.grid["safety_risk"][row][col])

    def test_07_workspace_boundary_cells_are_occupied(self):
        self.assertEqual(self.grid["occupancy"][0][0], 1)
        self.assertEqual(self.grid["occupancy"][89][59], 1)

    def test_08_safety_risk_zero_outside_radius(self):
        self.assertEqual(safety_risk_from_clearance(0.20, 0.20), 0.0)
        self.assertEqual(safety_risk_from_clearance(0.50, 0.20), 0.0)

    def test_09_safety_risk_increases_near_collision(self):
        self.assertGreater(safety_risk_from_clearance(0.01, 0.20), safety_risk_from_clearance(0.10, 0.20))

    def test_10_occupied_cell_is_not_traversable(self):
        grid = tiny_grid([[0, 1], [0, 0]])
        self.assertNotIn((0, 1), {item[0] for item in neighbors((0, 0), grid)})
        with self.assertRaises(PlannerError) as context:
            astar_grid(grid, (0, 0), (0, 1), "DISTANCE_ONLY_ASTAR", 0.0)
        self.assertEqual(context.exception.code, "GOAL_IN_COLLISION")

    def test_11_valid_start_and_goal(self):
        self.assertEqual(self.distance["start_grid"], [15, 10])
        self.assertEqual(self.distance["goal_grid"], [75, 47])

    def test_12_start_in_collision_rejected(self):
        with self.assertRaises(PlannerError) as context:
            plan_path(self.world, self.grid, self.config, "DISTANCE_ONLY_ASTAR", start={"x": 0.5, "y": 0.8})
        self.assertEqual(context.exception.code, "START_IN_COLLISION")

    def test_13_goal_in_collision_rejected(self):
        with self.assertRaises(PlannerError) as context:
            plan_path(self.world, self.grid, self.config, "DISTANCE_ONLY_ASTAR", goal={"x": 0.5, "y": 0.8})
        self.assertEqual(context.exception.code, "GOAL_IN_COLLISION")

    def test_14_non_finite_start_and_goal_rejected(self):
        for label, kwargs, code in (
            ("start", {"start": {"x": float("nan"), "y": 0.3}}, "START_NON_FINITE"),
            ("goal", {"goal": {"x": 0.9, "y": float("inf")}}, "GOAL_NON_FINITE"),
        ):
            with self.subTest(label=label), self.assertRaises(PlannerError) as context:
                plan_path(self.world, self.grid, self.config, "DISTANCE_ONLY_ASTAR", **kwargs)
            self.assertEqual(context.exception.code, code)

    def test_15_cardinal_movement_cost(self):
        north = next(item for item in neighbors((1, 1), tiny_grid([[0] * 3 for _ in range(3)], resolution=0.02)) if item[2] == "N")
        self.assertAlmostEqual(north[1], 0.02)

    def test_16_diagonal_movement_cost(self):
        northeast = next(item for item in neighbors((1, 1), tiny_grid([[0] * 3 for _ in range(3)], resolution=0.02)) if item[2] == "NE")
        self.assertAlmostEqual(northeast[1], math.sqrt(2) * 0.02)

    def test_17_corner_cutting_prevented(self):
        grid = tiny_grid([[0, 1], [1, 0]])
        self.assertNotIn((1, 1), {item[0] for item in neighbors((0, 0), grid)})
        self.assertEqual(astar_grid(grid, (0, 0), (1, 1), "DISTANCE_ONLY_ASTAR", 0.0)["status"], "NO_PATH")

    def test_18_simple_free_space_shortest_path(self):
        result = astar_grid(tiny_grid([[0] * 3 for _ in range(3)]), (0, 0), (2, 2), "DISTANCE_ONLY_ASTAR", 0.0)
        self.assertAlmostEqual(result["total_cost"], 2 * math.sqrt(2))
        self.assertEqual(len(result["path_grid"]), 3)

    def test_19_obstacle_detour(self):
        result = astar_grid(tiny_grid([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), (0, 0), (2, 2), "DISTANCE_ONLY_ASTAR", 0.0)
        self.assertEqual(result["status"], "PATH_FOUND")
        self.assertAlmostEqual(result["total_cost"], 4.0)

    def test_20_no_path(self):
        grid = tiny_grid([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        self.assertEqual(astar_grid(grid, (0, 0), (2, 2), "DISTANCE_ONLY_ASTAR", 0.0)["status"], "NO_PATH")

    def test_21_distance_only_path_collision_free(self):
        self.assertTrue(self.distance["collision_free"])
        self.assertTrue(all(self.world_point_is_free(*point) for point in self.distance["path_world"]))

    def test_22_safety_path_collision_free(self):
        self.assertTrue(self.safety["collision_free"])
        self.assertTrue(all(self.world_point_is_free(*point) for point in self.safety["path_world"]))

    def test_23_path_length_calculation(self):
        self.assertAlmostEqual(path_length_m([[0, 0], [0.02, 0], [0.04, 0.02]]), 0.02 + math.sqrt(2) * 0.02)
        self.assertAlmostEqual(self.distance["path_length_m"], path_length_m(self.distance["path_world"]))

    def test_24_minimum_clearance_calculation(self):
        expected = min(clearance_at(*point, self.world) for point in self.safety["path_world"])
        self.assertAlmostEqual(self.safety["minimum_clearance_m"], expected)

    def test_25_accumulated_safety_risk_calculation(self):
        expected = accumulated_safety_risk(self.safety["path_grid"], self.safety["path_world"], self.grid)
        self.assertAlmostEqual(self.safety["accumulated_safety_risk"], expected)

    def test_26_deterministic_distance_only_path(self):
        rebuilt = plan_path(self.world, self.grid, self.config, "DISTANCE_ONLY_ASTAR")
        self.assertEqual(semantic_path_result(self.distance), semantic_path_result(rebuilt))

    def test_27_deterministic_safety_path(self):
        rebuilt = plan_path(self.world, self.grid, self.config, "SAFETY_COST_ASTAR")
        self.assertEqual(semantic_path_result(self.safety), semantic_path_result(rebuilt))

    def test_28_safety_weight_changes_edge_cost(self):
        grid = tiny_grid([[0, 0]], risks=[[0.0, 0.5]])
        unweighted = astar_grid(grid, (0, 0), (0, 1), "SAFETY_COST_ASTAR", 0.0)
        weighted = astar_grid(grid, (0, 0), (0, 1), "SAFETY_COST_ASTAR", 2.0)
        self.assertAlmostEqual(unweighted["total_cost"], 1.0)
        self.assertAlmostEqual(weighted["total_cost"], 2.0)

    def test_29_no_step14_double_inflation(self):
        row, col = metric_to_grid(0.29, 0.81, self.grid)
        self.assertEqual(grid_to_metric(row, col, self.grid), [0.29, 0.81])
        self.assertEqual(self.grid["occupancy"][row][col], 0)
        self.assertEqual(self.grid["additional_inflation_m"], 0.0)

    def test_30_step12_step13_step14_artifacts_unchanged(self):
        before = {path: path.read_bytes() for directory in PROTECTED_DIRS for path in directory.iterdir()}
        build_grid_map(self.world, self.config)
        self.assertEqual(before, {path: path.read_bytes() for directory in PROTECTED_DIRS for path in directory.iterdir()})

    def test_31_outside_start_and_goal_rejected(self):
        for label, kwargs, code in (
            ("start", {"start": {"x": 0.10, "y": 0.30}}, "START_OUTSIDE_WORKSPACE"),
            ("goal", {"goal": {"x": 1.10, "y": 1.50}}, "GOAL_OUTSIDE_WORKSPACE"),
        ):
            with self.subTest(label=label), self.assertRaises(PlannerError) as context:
                plan_path(self.world, self.grid, self.config, "DISTANCE_ONLY_ASTAR", **kwargs)
            self.assertEqual(context.exception.code, code)


if __name__ == "__main__":
    unittest.main()
