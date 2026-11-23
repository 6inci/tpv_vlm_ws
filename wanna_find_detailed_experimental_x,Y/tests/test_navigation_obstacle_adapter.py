import copy
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
    "step24n", WORKSPACE / "scripts" / "build_navigation_obstacle_observation.py"
)
STEP24N = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP24N)

from vlm_reposition.planning import (
    NavigationObstacleAdapterError,
    build_navigation_obstacle_observation,
    validate_semantic_metric_observation,
)
import vlm_reposition.planning.navigation_obstacle_adapter as adapter


class NavigationObstacleAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input = STEP24N.load_representative_observation()
        cls.config = STEP24N.read_json(STEP24N.PLANNER_CONFIG)
        cls.grid = STEP24N.read_json(STEP24N.GRID)
        cls.output = build_navigation_obstacle_observation(cls.input, cls.config, cls.grid)
        cls.summary = STEP24N.read_json(STEP24N.OUTPUT / "step24n_summary.json")

    def invalid(self, mutate, code):
        value = copy.deepcopy(self.input)
        mutate(value)
        with self.assertRaises(NavigationObstacleAdapterError) as caught:
            validate_semantic_metric_observation(value)
        self.assertEqual(caught.exception.code, code)

    def test_01_step22d_representative_observation_load(self):
        self.assertEqual(self.input["provenance"]["source_step"], "STEP_22D")

    def test_02_expected_image_id_preserved(self):
        self.assertEqual(self.output["source_image_id"], "image_014_photo_08")

    def test_03_track_id_preserved(self):
        self.assertEqual(self.output["track_id"], "track_039")

    def test_04_semantic_label_preserved(self):
        self.assertEqual(self.output["semantic"]["label"], "tennis racket")

    def test_05_visible_entity_required(self):
        self.invalid(lambda x: x["semantic"].update(status="NO_VISIBLE_ENTITY"), "SEMANTIC_NOT_VISIBLE")

    def test_06_association_required(self):
        self.invalid(lambda x: x["association"].update(status="NOT_ASSOCIATED"), "NOT_ASSOCIATED")

    def test_07_world_xy_required(self):
        self.invalid(lambda x: x["metric"].update(world_xy=None), "WORLD_POINT_MISSING")

    def test_08_finite_coordinates_required(self):
        for invalid in (float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                self.invalid(lambda x, v=invalid: x["metric"].update(world_xy=[v, 0.7]), "WORLD_POINT_NON_FINITE")

    def test_09_correct_world_frame_required(self):
        self.invalid(lambda x: x["metric"].update(coordinate_frame="map"), "FRAME_MISMATCH")

    def test_10_workspace_inside_accepted(self):
        self.assertIs(validate_semantic_metric_observation(self.input), self.input)

    def test_11_workspace_outside_rejected(self):
        self.invalid(lambda x: x["metric"].update(workspace_status="OUTSIDE"), "OUTSIDE_WORKSPACE")

    def test_12_physical_geometry_not_ready_does_not_block(self):
        self.assertNotIn("physical_geometry", self.input)
        self.assertEqual(self.output["physical_geometry"], {"status": "NOT_READY", "required_for_adapter": False})

    def test_13_no_physical_size_generation(self):
        text = json.dumps(self.output)
        for prohibited in ('"dimensions"', '"length"', '"width"', '"physical_size"'):
            self.assertNotIn(prohibited, text)

    def test_14_no_semantic_class_size_lookup(self):
        source = inspect.getsource(adapter)
        self.assertNotIn("tennis racket", source)
        self.assertNotIn("class_size", source)

    def test_15_no_bbox_to_metric_size_conversion(self):
        representation = self.output["navigation_representation"]
        self.assertNotIn("bbox", representation)
        self.assertNotIn("extent_m", representation)

    def test_16_no_arbitrary_object_radius(self):
        self.assertNotIn('"radius', json.dumps(self.output))

    def test_17_reference_not_labeled_physical_center(self):
        reference = self.output["reference_point_world"]
        self.assertEqual(reference["reference_type"], "BBOX_BOTTOM_CENTER_DEVELOPMENT_GROUND_REFERENCE")
        self.assertFalse(any(key in reference for key in ("physical_center", "object_center", "footprint_center")))

    def test_18_navigation_role_obstacle_candidate(self):
        self.assertEqual(self.output["navigation_role"], "OBSTACLE_CANDIDATE")

    def test_19_point_source_representation_retained(self):
        self.assertEqual(self.output["navigation_representation"]["source_representation"], "NAVIGATION_POINT_OBSTACLE")

    def test_20_deterministic_containing_cell_projection(self):
        representation = self.output["navigation_representation"]
        self.assertEqual(representation["grid_index"], {"row": 35, "col": 38})
        self.assertEqual(representation["grid_cell_center_m"], [0.77, 0.71])

    def test_21_grid_resolution_from_existing_config(self):
        self.assertTrue(math.isclose(self.output["navigation_representation"]["grid_resolution_m"], self.config["grid_resolution_m"]))

    def test_22_physical_navigation_and_planner_layers_distinct(self):
        self.assertEqual(self.output["navigation_representation"]["geometry_status"], "NAVIGATION_APPROXIMATION")
        self.assertEqual(self.output["physical_geometry"]["status"], "NOT_READY")
        self.assertEqual(self.output["planner_injection"], "NOT_RUN")

    def test_23_step23m_measurement_status_unchanged(self):
        physical = STEP24N.read_json(STEP24N.STEP23M)
        self.assertEqual(physical["measurement_input_status"], "WAITING_FOR_USER_MEASUREMENT")
        self.assertEqual(physical["representative_physical_geometry"], "NOT_READY")

    def test_24_qwen_inference_count_zero(self):
        self.assertEqual(self.summary["qwen_inference_runs"], 0)

    def test_25_no_homography_recomputation(self):
        self.assertFalse(self.input["provenance"]["world_xy_recomputed"])
        self.assertEqual(self.summary["homography_recomputation"], "NOT_RUN")

    def test_26_no_step17_call(self):
        self.assertEqual(self.summary["step17_call"], "NOT_RUN")

    def test_27_no_step18_call(self):
        self.assertEqual(self.summary["step18_call"], "NOT_RUN")

    def test_28_no_step19_call(self):
        self.assertEqual(self.summary["step19_call"], "NOT_RUN")

    def test_29_no_astar_execution(self):
        self.assertEqual(self.summary["astar"], "NOT_RUN")

    def test_30_no_costmap_execution(self):
        self.assertEqual(self.summary["costmap"], "NOT_RUN")

    def test_31_no_ros_nav2_execution(self):
        self.assertEqual(self.summary["ros_nav2"], "NOT_RUN")

    def test_32_protected_step12_to_step23m_artifacts_unchanged(self):
        self.assertEqual(STEP24N.snapshot_protected(), STEP24N.PROTECTED_BASELINE)
        self.assertTrue(self.summary["protected_artifacts_unchanged"])


if __name__ == "__main__":
    unittest.main()
