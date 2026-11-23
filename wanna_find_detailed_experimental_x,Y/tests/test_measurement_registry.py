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
    "step23m", WORKSPACE / "scripts" / "register_representative_object_measurement.py"
)
STEP23M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP23M)

from vlm_reposition.world import (
    assess_measurement,
    build_measured_world_placement,
    register_ready_measurement,
    to_m,
)
import vlm_reposition.world.measurement_registry as measurement_registry


class MeasurementRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = json.loads(STEP23M.TEMPLATE.read_text())
        cls.registry = json.loads(STEP23M.REGISTRY.read_text())
        cls.metric = STEP23M.load_representative_metric_point()

    def rectangle_measurement(self):
        return {
            "schema_version": "0.1",
            "object_id": "test_measured_object",
            "semantic_label": "test-only measured rectangle",
            "measurement_status": "USER_MEASUREMENT_PROVIDED",
            "geometry_source": "MEASURED_PHYSICAL_GEOMETRY",
            "shape_type": "AXIS_ALIGNED_RECTANGLE",
            "dimensions": {"length": 70.0, "width": 25.0, "unit": "cm"},
            "polygon_local": None,
            "polygon_unit": None,
            "anchor": {
                "type": "MEASURED_FOOTPRINT_CENTER",
                "offset_local": [0.0, 0.0],
                "unit": "cm",
                "measurement_method": "test-only ruler fixture",
            },
            "orientation": {
                "status": "KNOWN",
                "reference_definition": "test local +x along measured length",
                "value_rad": 0.25,
            },
            "measurement_uncertainty": {"value": 2.0, "unit": "mm"},
            "notes": "synthetic unit-test input; not representative-object evidence",
        }

    def test_01_missing_measurement_waits_for_user(self):
        self.assertEqual(assess_measurement(None)["status"], "WAITING_FOR_USER_MEASUREMENT")

    def test_02_template_contains_no_fake_dimensions(self):
        self.assertIsNone(self.template["shape_type"])
        self.assertIsNone(self.template["dimensions"])
        self.assertIsNone(self.template["polygon_local"])
        self.assertIsNone(self.template["anchor"]["offset_local"])
        self.assertIsNone(self.template["orientation"]["value_rad"])

    def test_03_prohibited_or_default_dimensions_absent(self):
        self.assertFalse(STEP23M.ACTUAL_INPUT.exists())
        self.assertEqual(self.registry["objects"], [])
        self.assertNotIn("default_dimensions", self.template)
        self.assertNotIn("vlm_dimensions", self.template)

    def test_04_rectangle_validation(self):
        result = assess_measurement(self.rectangle_measurement())
        self.assertEqual(result["status"], "READY")
        dimensions = result["physical_geometry"]["dimensions_m"]
        self.assertTrue(math.isclose(dimensions["length_m"], 0.7))
        self.assertTrue(math.isclose(dimensions["width_m"], 0.25))

    def test_05_circle_validation(self):
        value = self.rectangle_measurement()
        value.update(shape_type="CIRCLE", dimensions={"radius": 10.0, "unit": "cm"})
        value["orientation"] = {"status": None, "reference_definition": None, "value_rad": None}
        result = assess_measurement(value)
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["physical_geometry"]["dimensions_m"], {"diameter_m": 0.2})
        self.assertEqual(result["physical_geometry"]["orientation"]["status"], "NOT_REQUIRED")

    def test_06_polygon_validation(self):
        value = self.rectangle_measurement()
        value.update(shape_type="POLYGON", dimensions=None, polygon_local=[[0, 0], [300, 0], [0, 200]], polygon_unit="mm")
        result = assess_measurement(value)
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["physical_geometry"]["physical_footprint_local"], [[0.0, 0.0], [0.3, 0.0], [0.0, 0.2]])

    def test_07_negative_zero_and_nonfinite_rejected(self):
        for invalid in (0, -1, float("nan"), float("inf")):
            value = self.rectangle_measurement()
            value["dimensions"]["length"] = invalid
            with self.subTest(invalid=invalid):
                self.assertEqual(assess_measurement(value)["status"], "INVALID_MEASUREMENT")

    def test_08_mm_to_m(self):
        self.assertEqual(to_m(250, "mm"), 0.25)

    def test_09_cm_to_m(self):
        self.assertEqual(to_m(25, "cm"), 0.25)

    def test_10_m_unchanged(self):
        self.assertEqual(to_m(0.25, "m"), 0.25)

    def test_11_raw_input_provenance_retained(self):
        value = self.rectangle_measurement()
        result = assess_measurement(value, "/tmp/test_measurement.json")
        provenance = result["physical_geometry"]["provenance"]
        self.assertEqual(provenance["source"], "USER_PHYSICAL_MEASUREMENT")
        self.assertEqual(provenance["raw_input"], value)
        self.assertEqual(provenance["input_unit"], "cm")
        self.assertTrue(math.isclose(provenance["canonical_metric_values"]["dimensions_m"]["length_m"], 0.7))

    def test_12_missing_anchor_blocks_ready(self):
        value = self.rectangle_measurement()
        value["anchor"] = {"type": None, "offset_local": None, "unit": None, "measurement_method": None}
        self.assertEqual(assess_measurement(value)["status"], "NOT_READY_MISSING_ANCHOR")

    def test_13_missing_required_orientation_blocks_ready(self):
        value = self.rectangle_measurement()
        value["orientation"] = {"status": None, "reference_definition": None, "value_rad": None}
        self.assertEqual(assess_measurement(value)["status"], "NOT_READY_MISSING_ORIENTATION")

    def test_14_complete_measurement_can_be_ready(self):
        assessment = assess_measurement(self.rectangle_measurement())
        updated = register_ready_measurement(self.registry, assessment)
        self.assertEqual(assessment["status"], "READY")
        self.assertEqual(updated["registry_status"], "HAS_READY_MEASUREMENTS")
        self.assertEqual(len(updated["objects"]), 1)

    def test_15_representative_step22d_metric_point_reused(self):
        self.assertEqual(self.metric["world_xy"], [0.7646719813346863, 0.713485598564148])
        self.assertEqual(self.metric["ground_reference"]["meaning"], "DEVELOPMENT_GROUND_REFERENCE")

    def test_16_world_placement_only_after_ready(self):
        waiting = assess_measurement(None)
        self.assertEqual(build_measured_world_placement(waiting, self.metric)["status"], "NOT_RUN")
        ready = assess_measurement(self.rectangle_measurement())
        placed = build_measured_world_placement(ready, self.metric)
        self.assertEqual(placed["status"], "AVAILABLE")
        self.assertTrue(math.isclose(placed["anchor_world_xy"][0], self.metric["world_xy"][0]))

    def test_17_physical_and_planning_geometry_separated(self):
        record = assess_measurement(self.rectangle_measurement())["physical_geometry"]
        self.assertFalse(record["planning_ready"])
        self.assertEqual(record["planning_geometry"], {"status": "NOT_GENERATED"})

    def test_18_no_safety_margin_insertion(self):
        record = assess_measurement(self.rectangle_measurement())["physical_geometry"]
        self.assertNotIn("safety_margin_m", record)
        self.assertNotIn("robot_inflation_m", record)

    def test_19_no_planner_injection(self):
        source = inspect.getsource(measurement_registry) + inspect.getsource(STEP23M)
        self.assertNotIn("vlm_reposition.planning", source)
        self.assertNotIn("collision_geometry", source)

    def test_20_no_step17_18_19_calls(self):
        source = inspect.getsource(STEP23M)
        self.assertNotIn("dynamic_state", source)
        self.assertNotIn("automatic_replanning", source)
        self.assertNotIn("obstacle_policy", source)
        self.assertNotIn("grid_astar", source)

    def test_21_no_qwen_inference(self):
        source = inspect.getsource(STEP23M)
        self.assertNotIn("load_local_qwen", source)
        self.assertNotIn("generate_qwen_text", source)

    def test_22_protected_step12_to_step23_artifacts_unchanged(self):
        self.assertEqual(STEP23M.snapshot_protected(), STEP23M.PROTECTED_BASELINE)
        status = json.loads(STEP23M.STEP23_RESULT.read_text())
        self.assertEqual(status["representative_physical_geometry"], "NOT_READY_MEASUREMENT_REQUIRED")


if __name__ == "__main__":
    unittest.main()
