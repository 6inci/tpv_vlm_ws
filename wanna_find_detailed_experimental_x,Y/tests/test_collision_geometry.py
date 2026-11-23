import copy
import json
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.planning import (
    CollisionGeometryError,
    build_collision_world,
    check_robot_center,
    compute_valid_robot_center_bounds,
    expand_axis_aligned_obstacle,
    is_robot_center_valid,
    load_static_obstacles,
    robot_half_extents,
    validate_static_obstacles,
)
from vlm_reposition.world import load_physical_geometry, load_zones


ZONES_PATH = WORKSPACE / "config" / "zones.json"
PHYSICAL_PATH = WORKSPACE / "config" / "physical_geometry.json"
OBSTACLES_PATH = WORKSPACE / "config" / "static_obstacles.json"
STEP12_DIR = WORKSPACE / "results" / "workspace_state"
STEP13_DIR = WORKSPACE / "results" / "physical_geometry"


class CollisionGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zones = load_zones(ZONES_PATH)
        cls.physical = load_physical_geometry(PHYSICAL_PATH)
        cls.obstacles = load_static_obstacles(OBSTACLES_PATH)
        cls.world = build_collision_world(cls.zones, cls.physical, cls.obstacles, "PARTIAL")

    def test_01_static_obstacle_config_load(self):
        self.assertEqual(self.obstacles["obstacles"][0]["geometry_status"], "DEVELOPMENT_SYNTHETIC")
        self.assertEqual(self.obstacles["obstacles"][0]["planning_role"], "STATIC_OBSTACLE")

    def test_02_robot_footprint_from_physical_config(self):
        self.assertEqual(self.world["robot"]["footprint_m"], self.physical["robot"]["source_footprint_m"])
        self.assertEqual(self.world["robot"]["source_path"], self.physical["robot"]["source_path"])

    def test_03_robot_half_extents(self):
        self.assertEqual(robot_half_extents(self.physical["robot"]), {"x": 0.145, "y": 0.123})

    def test_04_valid_robot_center_workspace(self):
        actual = self.world["workspace"]["valid_robot_center_bounds"]
        expected = {"x_min": 0.145, "x_max": 1.055, "y_min": 0.123, "y_max": 1.677}
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key])

    def test_05_obstacle_expansion(self):
        expanded = expand_axis_aligned_obstacle(
            {"x_min": 0.4, "x_max": 0.6, "y_min": 0.7, "y_max": 1.0},
            {"x": 0.1, "y": 0.2},
        )
        self.assertEqual(expanded, {"x_min": 0.30000000000000004, "x_max": 0.7, "y_min": 0.49999999999999994, "y_max": 1.2})

    def test_06_expected_development_obstacle_expansion(self):
        bounds = self.world["static_obstacles"][0]["collision_bounds"]
        for actual, expected in zip(bounds.values(), (0.305, 0.795, 0.527, 1.273)):
            self.assertAlmostEqual(actual, expected)

    def test_07_inside_collision_obstacle(self):
        result = check_robot_center(0.5, 0.8, self.world)
        self.assertEqual(result, {"status": "COLLISION", "reason": "STATIC_OBSTACLE", "obstacle_id": "dev_obstacle_01"})

    def test_08_outside_collision_obstacle(self):
        self.assertTrue(is_robot_center_valid(0.9, 0.9, self.world))
        self.assertEqual(check_robot_center(0.9, 0.9, self.world)["status"], "COLLISION_FREE")

    def test_09_touching_collision_boundary(self):
        bounds = self.world["static_obstacles"][0]["collision_bounds"]
        for point in ((bounds["x_min"], 0.8), (bounds["x_max"], 0.8), (0.5, bounds["y_min"]), (0.5, bounds["y_max"])):
            with self.subTest(point=point):
                self.assertEqual(check_robot_center(*point, self.world)["status"], "COLLISION")

    def test_10_outside_or_touching_workspace_boundary(self):
        bounds = self.world["workspace"]["valid_robot_center_bounds"]
        for point in ((0.1, 0.3), (bounds["x_min"], 0.3), (bounds["x_max"], 0.3)):
            with self.subTest(point=point):
                result = check_robot_center(*point, self.world)
                self.assertEqual(result["reason"], "WORKSPACE_BOUNDARY")

    def test_11_frame_mismatch_rejected(self):
        invalid = copy.deepcopy(self.obstacles)
        invalid["coordinate_frame"] = "OTHER_FRAME"
        invalid["obstacles"][0]["coordinate_frame"] = "OTHER_FRAME"
        with self.assertRaises(CollisionGeometryError) as context:
            build_collision_world(self.zones, self.physical, invalid, "PARTIAL")
        self.assertEqual(context.exception.code, "FRAME_MISMATCH")

    def test_12_unit_mismatch_rejected(self):
        invalid = copy.deepcopy(self.obstacles)
        invalid["unit"] = "cm"
        invalid["obstacles"][0]["unit"] = "cm"
        with self.assertRaises(CollisionGeometryError) as context:
            build_collision_world(self.zones, self.physical, invalid, "PARTIAL")
        self.assertEqual(context.exception.code, "UNIT_MISMATCH")

    def test_13_non_finite_geometry_rejected(self):
        invalid = copy.deepcopy(self.obstacles)
        invalid["obstacles"][0]["bounds"]["x_min"] = float("nan")
        with self.assertRaises(CollisionGeometryError) as context:
            validate_static_obstacles(invalid)
        self.assertEqual(context.exception.code, "NON_FINITE_GEOMETRY")

    def test_14_invalid_rectangle_and_schema_rejected(self):
        invalid = copy.deepcopy(self.obstacles)
        invalid["obstacles"][0]["bounds"]["x_min"] = 0.65
        with self.assertRaises(CollisionGeometryError) as context:
            validate_static_obstacles(invalid)
        self.assertEqual(context.exception.code, "INVALID_RECTANGLE")
        for field, value, code in (
            ("shape", "circle", "UNSUPPORTED_GEOMETRY_SHAPE"),
            ("geometry_status", "MEASURED", "UNSUPPORTED_GEOMETRY_STATUS"),
            ("source_detail", "", "MISSING_PROVENANCE"),
        ):
            invalid = copy.deepcopy(self.obstacles)
            invalid["obstacles"][0][field] = value
            with self.subTest(field=field), self.assertRaises(CollisionGeometryError) as context:
                validate_static_obstacles(invalid)
            self.assertEqual(context.exception.code, code)

    def test_15_obstacle_outside_workspace_rejected(self):
        invalid = copy.deepcopy(self.obstacles)
        invalid["obstacles"][0]["bounds"]["x_min"] = -0.01
        with self.assertRaises(CollisionGeometryError) as context:
            build_collision_world(self.zones, self.physical, invalid, "PARTIAL")
        self.assertEqual(context.exception.code, "OBSTACLE_OUTSIDE_WORKSPACE")

    def test_16_no_safety_margin_applied(self):
        raw = self.world["static_obstacles"][0]["raw_bounds"]
        expanded = self.world["static_obstacles"][0]["collision_bounds"]
        half = self.world["robot"]["half_extents_m"]
        self.assertEqual(self.world["safety_margin_m"], 0.0)
        self.assertAlmostEqual(expanded["x_min"], raw["x_min"] - half["x"])
        self.assertAlmostEqual(expanded["y_max"], raw["y_max"] + half["y"])

    def test_17_step12_results_unchanged(self):
        before = {path: path.read_bytes() for path in STEP12_DIR.iterdir()}
        build_collision_world(self.zones, self.physical, self.obstacles, "PARTIAL")
        self.assertEqual(before, {path: path.read_bytes() for path in STEP12_DIR.iterdir()})

    def test_18_step13_results_unchanged(self):
        before = {path: path.read_bytes() for path in STEP13_DIR.iterdir()}
        build_collision_world(self.zones, self.physical, self.obstacles, "PARTIAL")
        self.assertEqual(before, {path: path.read_bytes() for path in STEP13_DIR.iterdir()})

    def test_19_deterministic_rebuild_and_readiness(self):
        rebuilt = build_collision_world(self.zones, self.physical, self.obstacles, "PARTIAL")
        self.assertEqual(self.world, rebuilt)
        self.assertEqual(rebuilt["readiness"]["overall_manipulation_geometry_readiness"], "PARTIAL")
        self.assertEqual(rebuilt["readiness"]["robot_only_static_planning_geometry_readiness"], "READY")
        self.assertEqual(rebuilt["path_planning"]["status"], "NOT_RUN")


if __name__ == "__main__":
    unittest.main()
