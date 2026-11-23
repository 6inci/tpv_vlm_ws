import copy
import json
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.world import (
    PhysicalGeometryError,
    build_physical_planning_state,
    load_physical_geometry,
    rectangle_footprint,
    validate_physical_geometry,
)


CONFIG_PATH = WORKSPACE / "config" / "physical_geometry.json"
STATE_DIR = WORKSPACE / "results" / "workspace_state"


class PhysicalGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_physical_geometry(CONFIG_PATH)
        cls.red_state = json.loads((STATE_DIR / "red_cup_zone_a_state.json").read_text())

    def build(self, state=None, config=None):
        return build_physical_planning_state(
            self.red_state if state is None else state,
            self.config if config is None else config,
        )

    def test_physical_config_load(self):
        self.assertEqual(self.config["schema_version"], "0.1")
        self.assertEqual(self.config["coordinate_units"], "m")

    def test_geometry_status_validation(self):
        invalid = copy.deepcopy(self.config)
        invalid["robot"]["geometry_status"] = "PROBABLY_OK"
        with self.assertRaises(PhysicalGeometryError) as context:
            validate_physical_geometry(invalid)
        self.assertEqual(context.exception.code, "INVALID_GEOMETRY_STATUS")
        invalid = copy.deepcopy(self.config)
        invalid["robot"].update(geometry_status="INFERRED", footprint_type="circle")
        with self.assertRaises(PhysicalGeometryError) as context:
            validate_physical_geometry(invalid)
        self.assertEqual(context.exception.code, "UNSUPPORTED_FOOTPRINT_SHAPE")

    def test_confirmed_dimensions_must_be_finite_and_positive(self):
        for value in (-1, 0, float("nan"), float("inf")):
            invalid = copy.deepcopy(self.config)
            invalid["robot"]["length_m"] = value
            with self.subTest(value=value), self.assertRaises(PhysicalGeometryError) as context:
                validate_physical_geometry(invalid)
            self.assertEqual(context.exception.code, "INVALID_DIMENSION")

    def test_confirmed_geometry_requires_source(self):
        invalid = copy.deepcopy(self.config)
        invalid["robot"]["source_path"] = None
        with self.assertRaises(PhysicalGeometryError) as context:
            validate_physical_geometry(invalid)
        self.assertEqual(context.exception.code, "MISSING_PROVENANCE")

    def test_unknown_dimensions_are_preserved_as_null(self):
        result = self.build()
        target = result["physical_geometry"]["target_object"]
        self.assertEqual(target["geometry_status"], "UNKNOWN")
        self.assertIsNone(target["length_m"])
        self.assertIsNone(target["footprint_m"])

    def test_inferred_dimension_is_not_promoted(self):
        inferred = copy.deepcopy(self.config)
        inferred["robot"]["geometry_status"] = "INFERRED"
        result = self.build(config=inferred)
        robot = result["physical_geometry"]["robot"]
        self.assertEqual(robot["geometry_status"], "INFERRED")
        self.assertEqual(robot["geometry_readiness"], "NOT_READY")
        self.assertIsNone(robot["footprint_m"])

    def test_confirmed_robot_footprint_generation(self):
        expected = [[0.145, 0.123], [0.145, -0.123], [-0.145, -0.123], [-0.145, 0.123]]
        self.assertEqual(rectangle_footprint(0.29, 0.246), expected)
        self.assertEqual(self.build()["physical_geometry"]["robot"]["footprint_m"], expected)

    def test_gripper_unknown_handling(self):
        gripper = self.build()["physical_geometry"]["gripper"]
        self.assertEqual(gripper["geometry_status"], "UNKNOWN")
        self.assertEqual(gripper["geometry_readiness"], "NOT_READY")
        self.assertIsNone(gripper["front_extension_m"])

    def test_all_task_objects_remain_unknown(self):
        cases = (
            ("red_cup_zone_a_state.json", "red cup"),
            ("white_bag_zone_b_state.json", "white bag"),
            ("trophy_zone_f_state.json", "gold trophy-like object"),
        )
        for filename, target in cases:
            state = json.loads((STATE_DIR / filename).read_text())
            result = self.build(state=state)
            with self.subTest(target=target):
                self.assertEqual(result["physical_geometry"]["target_object"]["geometry_status"], "UNKNOWN")

    def test_planning_obstacle_unknown_geometry(self):
        candidates = [
            entity for entity in self.build()["physical_geometry"]["planning_entities"]
            if entity["planning_role"] == "STATIC_OBSTACLE_CANDIDATE"
        ]
        self.assertTrue(candidates)
        self.assertTrue(all(entity["geometry_status"] == "UNKNOWN" for entity in candidates))
        self.assertTrue(all(entity["footprint_m"] is None for entity in candidates))

    def test_semantic_type_and_planning_role_are_separate(self):
        entities = self.build()["physical_geometry"]["planning_entities"]
        red = next(entity for entity in entities if entity["scene_entity_id"] == "entity_007")
        person = next(entity for entity in entities if entity["semantic_type"] == "person")
        self.assertEqual(red["semantic_type"], "workspace_element")
        self.assertEqual(red["planning_role"], "TASK_TARGET")
        self.assertEqual(person["planning_role"], "DYNAMIC_ENTITY")

    def test_readiness_and_step12_input_immutability(self):
        original = copy.deepcopy(self.red_state)
        result = self.build()
        self.assertEqual(self.red_state, original)
        self.assertEqual(result["planning_geometry_readiness"]["robot"], "READY")
        self.assertEqual(result["planning_geometry_readiness"]["overall"], "PARTIAL")
        self.assertEqual(result["collision_checking"]["status"], "NOT_RUN")
        self.assertEqual(result["planning"]["status"], "NOT_RUN")


if __name__ == "__main__":
    unittest.main()
