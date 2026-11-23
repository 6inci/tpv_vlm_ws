import json
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.geometry import LegacyGeometry
from vlm_reposition.perception import (
    apply_semantic_reference,
    ground_scene_entities,
    load_partial_semantic_reference,
    parse_semantic_entities,
    semantic_label_status,
)


CONFIG = json.loads((WORKSPACE / "config" / "qwen_grounding.json").read_text())


class SemanticRefinementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = LegacyGeometry.load(
            CONFIG["calibration_path"], CONFIG["legacy_geometry_path"]
        )
        cls.reference = load_partial_semantic_reference(CONFIG["semantic_reference_path"])

    def test_specific_generic_and_unknown_labels(self):
        self.assertEqual(semantic_label_status("blue box"), "SPECIFIC")
        self.assertEqual(semantic_label_status("person"), "SPECIFIC")
        self.assertEqual(semantic_label_status("workspace_element"), "GENERIC")
        self.assertEqual(semantic_label_status("furniture"), "GENERIC")
        self.assertEqual(semantic_label_status("unknown"), "UNKNOWN")

    def test_partial_reference_matches_legacy_source_bboxes(self):
        legacy = json.loads(Path(CONFIG["stored_legacy_result_path"]).read_text())
        legacy_by_label = {entity["label"]: entity["bbox_2d"] for entity in legacy["objects"]}
        self.assertEqual(self.reference["reference_status"], "PARTIAL_SEMANTIC_REFERENCE")
        self.assertEqual(len(self.reference["entities"]), 3)
        for entity in self.reference["entities"]:
            self.assertEqual(
                entity["bbox_normalized_0_1000"], legacy_by_label[entity["source_label"]]
            )

    def test_navigation_role_is_not_trusted_as_semantic_output(self):
        parsed = parse_semantic_entities(
            '{"entities":[{"label":"blue box","semantic_type":"movable_object",'
            '"navigation_role":"target_candidate","bbox_2d":[100,100,200,200]}]}'
        )
        self.assertNotIn("navigation_role", parsed[0])

    def test_reference_bbox_and_alias_matching(self):
        output = {"entities": [
            {"label": "red cup", "semantic_type": "movable_object", "bbox_2d": [447,527,483,595]},
            {"label": "white bag", "semantic_type": "movable_object", "bbox_2d": [713,517,794,696]},
            {"label": "trophy", "semantic_type": "movable_object", "bbox_2d": [585,243,617,347]},
        ]}
        parsed = parse_semantic_entities(json.dumps(output))
        grounded = ground_scene_entities(
            parsed, (2016, 1133), self.geometry, (0.0, 1.2, 0.0, 1.8)
        )
        entities, comparisons = apply_semantic_reference(grounded, self.reference, 0.5)
        self.assertTrue(all(item["result"] == "PASS" for item in comparisons))
        self.assertTrue(all(item["reference_review_status"] == "MATCHED_REFERENCE" for item in entities))

    def test_outside_entity_is_preserved_and_unreviewed(self):
        parsed = parse_semantic_entities(
            '{"entities":[{"label":"office desk","semantic_type":"furniture",'
            '"bbox_2d":[0,0,350,998]}]}'
        )
        grounded = ground_scene_entities(
            parsed, (2016, 1133), self.geometry, (0.0, 1.2, 0.0, 1.8)
        )
        entities, comparisons = apply_semantic_reference(grounded, self.reference, 0.5)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["active_workspace_status"], "OUTSIDE")
        self.assertEqual(entities[0]["reference_review_status"], "UNREVIEWED")
        self.assertEqual(len(comparisons), 3)


if __name__ == "__main__":
    unittest.main()
