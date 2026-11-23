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
    "step23", WORKSPACE / "scripts" / "audit_physical_obstacle_geometry.py"
)
STEP23 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP23)

from vlm_reposition.world import (
    ANCHOR_STATUSES,
    GEOMETRY_SOURCES,
    ObjectFootprintError,
    build_local_footprint,
    load_physical_obstacle_config,
    measurement_requirements,
    place_world_footprint,
    planning_readiness,
    validate_geometry_record,
    validate_physical_obstacle_config,
)
import vlm_reposition.world.object_footprint as object_footprint


class ObjectFootprintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_physical_obstacle_config(STEP23.CONFIG)
        cls.representative_item = cls.config["representative_observations"][0]
        cls.representative = cls.representative_item["physical_geometry"]
        cls.fixture = cls.config["development_fixtures"][0]

    def measured_rectangle(self):
        record = copy.deepcopy(self.fixture)
        record.update(
            object_id="measured_fixture",
            semantic_label="measured fixture",
            geometry_source="MEASURED_PHYSICAL_GEOMETRY",
            source_status="CONFIRMED",
        )
        record["anchor"].update(status="MEASURED_FOOTPRINT_CENTER")
        record["provenance"] = {
            "source_type": "PHYSICAL_MEASUREMENT",
            "source_path": "/tmp/measurement_record.json",
            "source_detail": "test-only explicit measurement record",
        }
        return record

    def test_01_geometry_source_enum_validation(self):
        self.assertEqual(self.config["source_policy"]["priority"], list(GEOMETRY_SOURCES))
        invalid = copy.deepcopy(self.config)
        invalid["source_policy"]["priority"] = ["UNKNOWN"]
        with self.assertRaises(ObjectFootprintError) as caught:
            validate_physical_obstacle_config(invalid)
        self.assertEqual(caught.exception.code, "INVALID_SOURCE_PRIORITY")
        planar = self.measured_rectangle()
        planar.update(geometry_source="PLANAR_GROUND_CONTACT_GEOMETRY")
        planar["provenance"]["source_type"] = "VALIDATED_PLANAR_GROUND_CONTACT_GEOMETRY"
        with self.assertRaises(ObjectFootprintError) as caught:
            validate_geometry_record(planar)
        self.assertEqual(caught.exception.code, "UNVALIDATED_GROUND_CONTACT")

    def test_02_unknown_source_rejected_for_footprint_creation(self):
        with self.assertRaises(ObjectFootprintError) as caught:
            build_local_footprint(self.representative)
        self.assertEqual(caught.exception.code, "UNKNOWN_SOURCE")

    def test_03_measured_source_accepted(self):
        record = self.measured_rectangle()
        self.assertIs(validate_geometry_record(record), record)
        self.assertEqual(planning_readiness(record), "READY")

    def test_04_development_synthetic_provenance_preserved(self):
        self.assertEqual(self.fixture["provenance"]["source_type"], "STEP14_EXISTING_DEVELOPMENT_SYNTHETIC")
        self.assertEqual(Path(self.fixture["provenance"]["source_path"]), WORKSPACE / "config" / "static_obstacles.json")
        self.assertEqual(planning_readiness(self.fixture), "DEVELOPMENT_ONLY")

    def assert_prohibited_source(self, source):
        record = self.measured_rectangle()
        record["geometry_source"] = source
        with self.assertRaises(ObjectFootprintError) as caught:
            validate_geometry_record(record)
        self.assertEqual(caught.exception.code, "PROHIBITED_GEOMETRY_SOURCE")

    def test_05_bbox_cannot_source_physical_geometry(self):
        self.assert_prohibited_source("VLM_BBOX")

    def test_06_semantic_label_cannot_generate_dimensions(self):
        self.assert_prohibited_source("SEMANTIC_LABEL")

    def test_07_vlm_cannot_generate_physical_dimensions(self):
        self.assert_prohibited_source("VLM_GENERATED_DIMENSIONS")

    def test_08_circle_footprint_generation(self):
        record = self.measured_rectangle()
        record.update(shape_type="CIRCLE", dimensions_m={"diameter_m": 0.4}, physical_footprint_local=None)
        self.assertEqual(build_local_footprint(record), {"center_xy_m": [0.0, 0.0], "radius_m": 0.2})

    def test_09_rectangle_local_footprint_generation(self):
        self.assertEqual(build_local_footprint(self.fixture), [[-0.1, -0.25], [0.1, -0.25], [0.1, 0.25], [-0.1, 0.25]])

    def test_10_polygon_validation(self):
        record = self.measured_rectangle()
        points = [[0.0, 0.0], [0.3, 0.0], [0.0, 0.2]]
        record.update(shape_type="POLYGON", dimensions_m=None, physical_footprint_local=points)
        self.assertEqual(build_local_footprint(record), points)
        self.assertIs(validate_geometry_record(record), record)

    def test_11_invalid_polygon_rejected(self):
        record = self.measured_rectangle()
        record.update(shape_type="POLYGON", dimensions_m=None, physical_footprint_local=[[0, 0], [1, 1], [2, 2]])
        with self.assertRaises(ObjectFootprintError) as caught:
            validate_geometry_record(record)
        self.assertEqual(caught.exception.code, "INVALID_POLYGON")

    def test_12_anchor_status_validation(self):
        self.assertIn("UNKNOWN", ANCHOR_STATUSES)
        invalid = self.measured_rectangle()
        invalid["anchor"]["status"] = "LIKELY_CENTER"
        with self.assertRaises(ObjectFootprintError) as caught:
            validate_geometry_record(invalid)
        self.assertEqual(caught.exception.code, "INVALID_ANCHOR_STATUS")

    def test_13_unknown_anchor_blocks_world_placement(self):
        record = self.measured_rectangle()
        record["anchor"].update(status="UNKNOWN", offset_local_m=None)
        self.assertEqual(planning_readiness(record), "NOT_READY_MISSING_ANCHOR")
        with self.assertRaises(ObjectFootprintError) as caught:
            place_world_footprint(record, [0.5, 0.5])
        self.assertEqual(caught.exception.code, "NOT_READY_MISSING_ANCHOR")

    def test_14_development_assumed_center_explicit(self):
        self.assertEqual(self.fixture["anchor"], {"type": "FOOTPRINT_CENTER", "status": "DEVELOPMENT_ASSUMED_CENTER", "offset_local_m": [0.0, 0.0]})
        self.assertFalse(self.fixture["planning_ready"])

    def test_15_local_to_world_translation(self):
        result = place_world_footprint(self.fixture, [0.55, 0.9])
        expected = [[0.45, 0.65], [0.65, 0.65], [0.65, 1.15], [0.45, 1.15]]
        for actual, wanted in zip(result["physical_footprint_world"], expected):
            self.assertTrue(all(math.isclose(a, b) for a, b in zip(actual, wanted)))
        self.assertTrue(math.isclose(result["physical_bounds_world_m"]["x_min"], 0.45))

    def test_16_unknown_required_orientation_blocks_placement(self):
        record = self.measured_rectangle()
        record.update(shape_type="POLYGON", dimensions_m=None, physical_footprint_local=[[0, 0], [0.2, 0], [0, 0.1]], orientation_required=True)
        record["orientation"] = {"status": "UNKNOWN", "yaw_rad": None}
        self.assertEqual(planning_readiness(record), "NOT_READY_MISSING_ORIENTATION")
        with self.assertRaises(ObjectFootprintError) as caught:
            place_world_footprint(record, [0.5, 0.5])
        self.assertEqual(caught.exception.code, "NOT_READY_MISSING_ORIENTATION")

    def test_17_physical_and_planning_geometry_separate(self):
        result = place_world_footprint(self.fixture, [0.55, 0.9])
        self.assertEqual(result["planning_geometry"], {"status": "NOT_GENERATED"})
        self.assertIn("physical_footprint_world", result)

    def test_18_safety_margin_absent_from_physical_dimensions(self):
        self.assertNotIn("safety_margin_m", self.fixture["dimensions_m"])
        self.assertEqual(self.config["layer_policy"]["safety_margin"], "SEPARATE_NOT_APPLIED")
        invalid = copy.deepcopy(self.fixture)
        invalid["safety_margin_m"] = 0.1
        with self.assertRaises(ObjectFootprintError) as caught:
            validate_geometry_record(invalid)
        self.assertEqual(caught.exception.code, "PHYSICAL_LAYER_CONTAMINATION")

    def test_19_robot_inflation_absent_from_object_geometry(self):
        self.assertNotIn("robot_inflation_m", self.fixture)
        self.assertEqual(self.config["layer_policy"]["robot_inflation"], "SEPARATE_NOT_APPLIED")

    def test_20_representative_step22d_metric_point_import(self):
        result = STEP23.load_representative_observation(self.config)
        self.assertEqual(result["semantic_label"], "tennis racket")
        self.assertEqual(result["world_xy"], [0.7646719813346863, 0.713485598564148])
        self.assertEqual(result["ground_reference"]["meaning"], "DEVELOPMENT_GROUND_REFERENCE")

    def test_21_tennis_racket_dimensions_remain_unknown(self):
        self.assertEqual(self.representative["geometry_source"], "UNKNOWN")
        self.assertIsNone(self.representative["dimensions_m"])
        self.assertIsNone(self.representative["physical_footprint_local"])
        self.assertEqual(planning_readiness(self.representative), "NOT_READY_UNKNOWN_SOURCE")

    def test_22_measurement_requirement_generated(self):
        requirements = measurement_requirements(self.representative)
        self.assertEqual(len(requirements), 3)
        self.assertTrue(any("anchor relation" in item for item in requirements))

    def test_23_no_planner_import(self):
        source = inspect.getsource(object_footprint) + inspect.getsource(STEP23)
        self.assertNotIn("vlm_reposition.planning", source)
        self.assertNotIn("collision_geometry", source)

    def test_24_no_step17_18_19_call(self):
        source = inspect.getsource(STEP23)
        self.assertNotIn("dynamic_state", source)
        self.assertNotIn("automatic_replanning", source)
        self.assertNotIn("obstacle_policy", source)

    def test_25_no_astar_or_costmap_call(self):
        source = inspect.getsource(STEP23)
        self.assertNotIn("grid_astar", source)
        self.assertNotIn("costmap.", source)

    def test_26_protected_step12_to_step22d_artifacts_unchanged(self):
        self.assertEqual(STEP23.snapshot_protected(), STEP23.PROTECTED_BASELINE)
        step22d = json.loads((STEP23.STEP22D / "step22d_summary.json").read_text())
        self.assertEqual(step22d["qwen_inference_runs"], 1)
        self.assertEqual(step22d["step_result"], "PASS")


if __name__ == "__main__":
    unittest.main()
