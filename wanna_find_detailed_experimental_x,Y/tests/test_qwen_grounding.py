import json
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.geometry import LegacyGeometry
from vlm_reposition.perception import (
    GroundingValidationError,
    bbox_bottom_center,
    ground_normalized_bbox,
    normalized_bbox_to_pixel_bbox,
    parse_single_detection,
    validate_normalized_bbox,
)


CONFIG = json.loads((WORKSPACE / "config" / "qwen_grounding.json").read_text())


class QwenGroundingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = LegacyGeometry.load(
            CONFIG["calibration_path"], CONFIG["legacy_geometry_path"]
        )

    def test_normalized_bbox_to_pixel_and_bottom_center(self):
        pixel = normalized_bbox_to_pixel_bbox([447, 527, 483, 595], (2016, 1133))
        self.assertEqual(pixel, (901.152, 597.091, 973.728, 674.135))
        self.assertEqual(bbox_bottom_center(pixel), (937.44, 674.135))

    def test_bbox_validation_rejects_invalid_values(self):
        for bbox in ([1, 2, 3], [-1, 2, 3, 4], [1, 2, 1, 4], [1, 5, 3, 4]):
            with self.subTest(bbox=bbox), self.assertRaises(GroundingValidationError):
                validate_normalized_bbox(bbox)

    def test_parse_single_detection(self):
        parsed = parse_single_detection(
            '```json\n{"label":"red cup","bbox_2d":[447,527,483,595]}\n```'
        )
        self.assertEqual(parsed["label"], "red cup")
        self.assertEqual(parsed["bbox_2d"], [447.0, 527.0, 483.0, 595.0])

    def test_multiple_detections_are_rejected(self):
        with self.assertRaisesRegex(GroundingValidationError, "expected one target"):
            parse_single_detection(
                '{"objects":[{"label":"cup","bbox_2d":[1,2,3,4]},'
                '{"label":"cup","bbox_2d":[5,6,7,8]}]}'
            )

    def test_stored_bbox_integrates_with_geometry(self):
        stored = json.loads(Path(CONFIG["stored_legacy_result_path"]).read_text())["objects"][0]
        result = ground_normalized_bbox(
            "red cup", stored["label"], stored["bbox_2d"], (2016, 1133), self.geometry
        )
        self.assertEqual(result["reference_point_type"], "bbox_bottom_center")
        self.assertEqual(result["world_coordinate"]["frame"], "LEGACY_WORKSPACE_COORDINATE")
        self.assertEqual(
            [result["world_coordinate"]["x"], result["world_coordinate"]["y"]],
            list(self.geometry.raw_pixel_to_world(937.44, 674.135)),
        )

    def test_resolution_mismatch_is_rejected(self):
        with self.assertRaisesRegex(GroundingValidationError, "Expected 2016x1133"):
            ground_normalized_bbox(
                "red cup", "red cup", [447, 527, 483, 595], (1920, 1080), self.geometry
            )


if __name__ == "__main__":
    unittest.main()
