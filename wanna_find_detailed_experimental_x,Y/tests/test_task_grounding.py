import json
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.geometry import LegacyGeometry
from vlm_reposition.perception import (
    ground_target_result,
    load_partial_semantic_reference,
)


CONFIG = json.loads((WORKSPACE / "config" / "qwen_grounding.json").read_text())


class TaskGroundingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = LegacyGeometry.load(
            CONFIG["calibration_path"], CONFIG["legacy_geometry_path"]
        )
        cls.reference = load_partial_semantic_reference(CONFIG["semantic_reference_path"])
        cls.bounds = (0.0, 1.2, 0.0, 1.8)

    def result(self, output, reference_index=0, query="red cup"):
        return ground_target_result(
            query,
            output,
            (2016, 1133),
            self.geometry,
            self.bounds,
            self.reference["entities"][reference_index],
            0.5,
        )

    def test_target_found_reference_match_and_geometry(self):
        result = self.result(
            '{"candidates":[{"label":"red cup","bbox_2d":[447,527,483,595]}]}'
        )
        self.assertEqual(result["target_status"], "TARGET_FOUND")
        self.assertEqual(result["position_status"], "REFERENCE_POSITION_MATCH")
        self.assertEqual(result["semantic_status"], "REFERENCE_ALIAS_MATCH")
        self.assertEqual(result["active_workspace_status"], "INSIDE")
        self.assertEqual(result["world_coordinate"]["frame"], "LEGACY_WORKSPACE_COORDINATE")

    def test_target_not_found(self):
        result = self.result('{"candidates":[]}')
        self.assertEqual(result["target_status"], "TARGET_NOT_FOUND")
        self.assertEqual(result["position_status"], "NOT_EVALUATED")

    def test_target_ambiguous_does_not_select_first(self):
        result = self.result(
            '{"candidates":['
            '{"label":"red cup","bbox_2d":[447,527,483,595]},'
            '{"label":"red cup","bbox_2d":[500,500,550,600]}]}'
        )
        self.assertEqual(result["target_status"], "TARGET_AMBIGUOUS")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertNotIn("world_coordinate", result)

    def test_invalid_output(self):
        result = self.result('{"label":"red cup","bbox_2d":[500,500,400,600]}')
        self.assertEqual(result["target_status"], "INVALID_OUTPUT")

    def test_label_and_position_status_are_separate(self):
        result = self.result(
            '{"candidates":[{"label":"red mug","bbox_2d":[448,527,482,595]}]}'
        )
        self.assertEqual(result["position_status"], "REFERENCE_POSITION_MATCH")
        self.assertEqual(result["semantic_status"], "HUMAN_ALIAS_CANDIDATE")
        self.assertFalse(result["alias_match"])

    def test_reference_position_mismatch_does_not_pass_grounding(self):
        result = self.result(
            '{"candidates":[{"label":"red cup","bbox_2d":[100,100,150,150]}]}'
        )
        self.assertEqual(result["semantic_status"], "REFERENCE_ALIAS_MATCH")
        self.assertEqual(result["position_status"], "REFERENCE_POSITION_MISMATCH")


if __name__ == "__main__":
    unittest.main()
