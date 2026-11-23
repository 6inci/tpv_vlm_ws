import sys
from pathlib import Path
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.geometry import GeometryCompatibilityError, LegacyGeometry


CALIBRATION = Path("/home/binci/camera_calibration_1920x1080/calibration_result.json")
GEOMETRY = Path("/home/binci/qwen_vision_test/homography_12pt_undistorted.json")
RAW_POINTS = Path("/home/binci/qwen_vision_test/homography_click_points.json")


class LegacyGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = LegacyGeometry.load(CALIBRATION, GEOMETRY)

    def test_metadata_shapes(self):
        self.assertEqual(self.geometry.camera_matrix.shape, (3, 3))
        self.assertEqual(self.geometry.new_camera_matrix.shape, (3, 3))
        self.assertEqual(self.geometry.homography.shape, (3, 3))
        self.assertEqual(self.geometry.image_size, (2016, 1133))

    def test_resolution_mismatch_is_rejected(self):
        with self.assertRaisesRegex(GeometryCompatibilityError, "Expected 2016x1133"):
            self.geometry.validate_image(np.zeros((1080, 1920, 3), dtype=np.uint8))

    def test_reference_point_reproduction(self):
        import json

        raw = json.loads(RAW_POINTS.read_text())["points"][0]
        expected = self.geometry.reference_undistorted_pixels[0]
        actual = self.geometry.raw_pixel_to_undistorted_pixel(*raw)
        np.testing.assert_array_equal(actual, expected)

    def test_pixel_to_world_is_deterministic(self):
        first = self.geometry.raw_pixel_to_world(708, 964)
        second = self.geometry.raw_pixel_to_world(708, 964)
        self.assertEqual(first, second)

    def test_invalid_pixel_is_rejected(self):
        with self.assertRaises(GeometryCompatibilityError):
            self.geometry.raw_pixel_to_world(float("nan"), 0)


if __name__ == "__main__":
    unittest.main()
