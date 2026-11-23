import json
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.geometry import LegacyGeometry
from vlm_reposition.perception import (
    GroundingValidationError,
    bbox_iou,
    ground_scene_entities,
    parse_scene_entities,
)


CONFIG = json.loads((WORKSPACE / "config" / "qwen_grounding.json").read_text())
VALID_OUTPUT = '''{"entities":[
  {"label":"red cup","semantic_type":"movable_object","navigation_role":"target_candidate","bbox_2d":[448,525,483,595]},
  {"label":"person","semantic_type":"person","navigation_role":"obstacle_candidate","bbox_2d":[850,250,990,900]}
]}'''


class SceneGroundingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = LegacyGeometry.load(
            CONFIG["calibration_path"], CONFIG["legacy_geometry_path"]
        )

    def test_multi_entity_parse(self):
        entities = parse_scene_entities(f"```json\n{VALID_OUTPUT}\n```")
        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0]["semantic_type"], "movable_object")
        self.assertEqual(entities[1]["navigation_role"], "obstacle_candidate")

    def test_invalid_schema_values_and_bbox_are_rejected(self):
        invalid = (
            VALID_OUTPUT.replace('"movable_object"', '"portable_thing"', 1),
            VALID_OUTPUT.replace('"target_candidate"', '"target"', 1),
            VALID_OUTPUT.replace("[448,525,483,595]", "[448,525,400,595]", 1),
        )
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(GroundingValidationError):
                parse_scene_entities(output)

    def test_scene_grounding_assigns_ids_and_metric_coordinates(self):
        entities = ground_scene_entities(
            parse_scene_entities(VALID_OUTPUT),
            (2016, 1133),
            self.geometry,
            (0.0, 1.2, 0.0, 1.8),
        )
        self.assertEqual([entity["id"] for entity in entities], ["entity_001", "entity_002"])
        self.assertEqual(entities[0]["geometry_status"], "VALID")
        self.assertEqual(entities[0]["world_coordinate"]["frame"], "LEGACY_WORKSPACE_COORDINATE")

    def test_geometry_status_outside_and_invalid(self):
        red_cup = parse_scene_entities(VALID_OUTPUT)[0:1]
        outside = ground_scene_entities(
            red_cup, (2016, 1133), self.geometry, (0.0, 0.0, 0.0, 0.0)
        )
        self.assertEqual(outside[0]["geometry_status"], "OUTSIDE_LEGACY_WORKSPACE")

        invalid = [dict(red_cup[0], bbox_2d=[448, 525, 483, 1000])]
        grounded = ground_scene_entities(
            invalid, (2016, 1133), self.geometry, (0.0, 1.2, 0.0, 1.8)
        )
        self.assertEqual(grounded[0]["geometry_status"], "INVALID")
        self.assertIsNone(grounded[0]["world_coordinate"])

    def test_bbox_iou_is_analysis_only(self):
        self.assertEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)


if __name__ == "__main__":
    unittest.main()
