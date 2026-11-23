import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
SPEC = importlib.util.spec_from_file_location(
    "step25n", WORKSPACE / "scripts" / "integrate_navigation_dynamic_world.py"
)
STEP25N = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP25N)

from vlm_reposition.planning import (
    apply_obstacle_event,
    build_step17_dynamic_obstacle,
    dynamic_scenario_by_id,
    grid_cell_bounds,
    initial_obstacle_state,
    load_dynamic_config,
)
import vlm_reposition.planning.navigation_dynamic_bridge as bridge


class NavigationDynamicWorldIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observation = STEP25N.load_step24n_observation()
        cls.grid = STEP25N.read_json(STEP25N.GRID_ARTIFACT)
        cls.config = load_dynamic_config(STEP25N.DYNAMIC_CONFIG)
        cls.obstacle = build_step17_dynamic_obstacle(cls.observation, cls.grid, cls.config)
        cls.record = STEP25N.read_json(STEP25N.OUTPUT / "dynamic_obstacle_record.json")
        cls.before = STEP25N.read_json(STEP25N.OUTPUT / "dynamic_world_before.json")
        cls.after = STEP25N.read_json(STEP25N.OUTPUT / "dynamic_world_after.json")
        cls.effect = STEP25N.read_json(STEP25N.OUTPUT / "occupancy_effect.json")
        cls.summary = STEP25N.read_json(STEP25N.OUTPUT / "step25n_summary.json")

    def test_01_step24n_artifact_load(self):
        self.assertEqual(self.observation["provenance"]["source_step"], "STEP_22D")

    def test_02_image_id_preserved(self):
        self.assertEqual(self.obstacle["observation_identity"]["source_image_id"], "image_014_photo_08")

    def test_03_track_id_preserved(self):
        self.assertEqual(self.obstacle["observation_identity"]["track_id"], "track_039")

    def test_04_reference_point_preserved(self):
        self.assertEqual(self.obstacle["reference_point_world"], self.observation["reference_point_world"])

    def test_05_single_occupied_cell_loaded(self):
        self.assertEqual(self.observation["navigation_representation"]["type"], "SINGLE_OCCUPIED_CELL")

    def test_06_grid_metadata_consistency(self):
        cell = self.obstacle["navigation_cell"]
        self.assertEqual(cell["resolution_m"], self.grid["resolution_m"])
        self.assertEqual(self.obstacle["coordinate_frame"], self.grid["coordinate_frame"])

    def test_07_deterministic_cell_world_bounds(self):
        self.assertEqual(self.obstacle["bounds"], {"x_min": 0.76, "x_max": 0.78, "y_min": 0.7, "y_max": 0.72})

    def test_08_row_column_convention(self):
        self.assertEqual(self.obstacle["navigation_cell"]["row"], 35)
        self.assertEqual(self.obstacle["navigation_cell"]["col"], 38)

    def test_09_cell_boundary_convention(self):
        bounds = grid_cell_bounds(35, 38, self.grid)
        self.assertTrue(bounds["x_min"] <= 0.7646719813346863 < bounds["x_max"])
        self.assertTrue(bounds["y_min"] <= 0.713485598564148 < bounds["y_max"])

    def test_10_no_arbitrary_radius(self):
        self.assertNotIn("radius", json.dumps(self.obstacle))

    def test_11_no_physical_size_inference(self):
        text = json.dumps(self.obstacle)
        for prohibited in ('"dimensions"', '"physical_footprint"', '"object_width"', '"object_length"'):
            self.assertNotIn(prohibited, text)

    def test_12_physical_geometry_remains_not_ready(self):
        self.assertEqual(self.obstacle["physical_geometry_status"], "NOT_READY")
        self.assertFalse(self.obstacle["provenance"]["physical_geometry_used"])

    def test_13_bridge_geometry_is_navigation_discretization(self):
        self.assertEqual(self.obstacle["bridge_geometry"], "NAVIGATION_DISCRETIZATION_CELL")

    def test_14_step17_rectangle_generated(self):
        self.assertEqual(self.obstacle["shape"], "axis_aligned_rectangle")
        self.assertTrue(math.isclose(self.obstacle["bounds"]["x_max"] - self.obstacle["bounds"]["x_min"], 0.02))

    def test_15_observed_source_not_labeled_development_synthetic(self):
        self.assertEqual(self.obstacle["geometry_status"], "NAVIGATION_APPROXIMATION")
        self.assertEqual(self.obstacle["source_type"], "STEP24N_NAVIGATION_APPROXIMATION")
        self.assertNotEqual(self.obstacle["geometry_status"], "DEVELOPMENT_SYNTHETIC")

    def test_16_existing_development_synthetic_regression_preserved(self):
        scenario = dynamic_scenario_by_id(self.config, "D1")
        self.assertEqual(scenario["initial_static_obstacles"][0]["geometry_status"], "DEVELOPMENT_SYNTHETIC")

    def test_17_deterministic_obstacle_id(self):
        repeated = build_step17_dynamic_obstacle(self.observation, self.grid, self.config)
        self.assertEqual(self.obstacle["id"], "dynamic_image_014_photo_08_track_039")
        self.assertEqual(repeated["id"], self.obstacle["id"])

    def test_18_track_not_permanent_identity(self):
        identity = self.obstacle["observation_identity"]
        self.assertEqual(identity["identity_scope"], "OFFLINE_TEMPORAL_OBSERVATION")
        self.assertEqual(identity["persistent_physical_object_identity"], "NOT_CLAIMED")

    def test_19_timestamp_remains_unavailable(self):
        self.assertEqual(self.obstacle["timestamp"], {"status": "NOT_AVAILABLE"})

    def test_20_step17_add_through_normal_api(self):
        before = initial_obstacle_state(dynamic_scenario_by_id(self.config, "D1"))
        after = apply_obstacle_event(before, {"operation": "ADD", "obstacle": self.obstacle})
        self.assertIn(self.obstacle, after["obstacles"])
        self.assertEqual(after["world_revision"], 1)

    def test_21_dynamic_world_before_after_state(self):
        self.assertEqual(self.before["obstacle_state"]["world_revision"], 0)
        self.assertEqual(self.after["obstacle_state"]["world_revision"], 1)
        self.assertEqual(len(self.after["obstacle_state"]["obstacles"]), len(self.before["obstacle_state"]["obstacles"]) + 1)

    def test_22_static_obstacle_unchanged(self):
        before = [item for item in self.before["obstacle_state"]["obstacles"] if item["state_role"] == "STATIC_OBSTACLE"]
        after = [item for item in self.after["obstacle_state"]["obstacles"] if item["state_role"] == "STATIC_OBSTACLE"]
        self.assertEqual(before, after)
        self.assertTrue(self.summary["static_obstacles_unchanged"])

    def test_23_existing_occupied_cell_overlap_recorded(self):
        self.assertEqual(self.effect["source_cell_occupancy_before"], 1)
        self.assertEqual(self.effect["source_cell_occupancy_after"], 1)

    def test_24_zero_source_delta_does_not_erase_dynamic_observation(self):
        self.assertEqual(self.effect["source_cell_delta"], 0)
        self.assertEqual(self.summary["dynamic_obstacle"], "REGISTERED")
        self.assertEqual(self.effect["free_to_occupied_count"], 78)

    def test_25_no_step18_execution(self):
        self.assertEqual(self.summary["step18_call"], "NOT_RUN")

    def test_26_no_step19_execution(self):
        self.assertEqual(self.summary["step19_call"], "NOT_RUN")

    def test_27_no_astar_execution(self):
        self.assertEqual(self.summary["astar"], "NOT_RUN")
        self.assertNotIn("plan_path", inspect.getsource(STEP25N))

    def test_28_no_replanning(self):
        self.assertEqual(self.summary["replanning"], "NOT_RUN")

    def test_29_no_qwen_inference(self):
        self.assertEqual(self.summary["qwen_inference_runs"], 0)

    def test_30_no_homography_recomputation(self):
        self.assertEqual(self.summary["homography_recomputation"], "NOT_RUN")
        self.assertFalse(self.obstacle["provenance"]["reference_point_recomputed"])

    def test_31_step23m_unchanged(self):
        measurement = STEP25N.read_json(STEP25N.STEP23M)
        self.assertEqual(measurement["measurement_input_status"], "WAITING_FOR_USER_MEASUREMENT")
        self.assertEqual(measurement["representative_physical_geometry"], "NOT_READY")

    def test_32_protected_previous_artifacts_unchanged(self):
        self.assertEqual(STEP25N.snapshot_protected(), STEP25N.PROTECTED_BASELINE)
        self.assertTrue(self.summary["protected_artifacts_unchanged"])


if __name__ == "__main__":
    unittest.main()
