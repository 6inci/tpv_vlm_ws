import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.geometry import LegacyGeometry
from vlm_reposition.world import (
    ZoneValidationError,
    find_zone_for_point,
    get_zone_center,
    load_zones,
    point_in_zone,
    resolve_zone_alias,
    validate_zones,
    world_to_undistorted_pixel,
)


ZONES_PATH = WORKSPACE / "config" / "zones.json"
GROUNDING_CONFIG = json.loads((WORKSPACE / "config" / "qwen_grounding.json").read_text())


class GoalZoneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zones = load_zones(ZONES_PATH)
        cls.geometry = LegacyGeometry.load(
            GROUNDING_CONFIG["calibration_path"], GROUNDING_CONFIG["legacy_geometry_path"]
        )

    def test_zone_config_load(self):
        self.assertEqual(self.zones["status"], "OFFLINE_DEVELOPMENT_ZONE_SET")
        self.assertEqual(len(self.zones["zones"]), 6)

    def test_zone_ids_are_unique(self):
        ids = [zone["id"] for zone in self.zones["zones"]]
        self.assertEqual(set(ids), {f"zone_{letter}" for letter in "ABCDEF"})
        invalid = copy.deepcopy(self.zones)
        invalid["zones"][1]["id"] = "zone_A"
        with self.assertRaisesRegex(ZoneValidationError, "duplicate ID"):
            validate_zones(invalid)

    def test_alias_lookup(self):
        for alias in ("A", "A zone", "zone A", "A 구역", "A구역", "zone_A"):
            self.assertEqual(resolve_zone_alias(alias, self.zones), "zone_A")

    def test_unknown_alias_and_duplicate_alias_are_rejected(self):
        self.assertEqual(resolve_zone_alias("왼쪽", self.zones), "ZONE_NOT_RESOLVED")
        invalid = copy.deepcopy(self.zones)
        invalid["zones"][1]["aliases"][0] = "A"
        with self.assertRaisesRegex(ZoneValidationError, "duplicate zone alias"):
            validate_zones(invalid)

    def test_polygon_bounds(self):
        invalid = copy.deepcopy(self.zones)
        invalid["zones"][0]["polygon"][0] = [-0.1, 0.0]
        with self.assertRaisesRegex(ZoneValidationError, "outside workspace"):
            validate_zones(invalid)

    def test_center_is_inside_polygon(self):
        self.assertEqual(get_zone_center(self.zones, "zone_F"), (0.9, 1.5))
        invalid = copy.deepcopy(self.zones)
        invalid["zones"][0]["center"] = [0.6, 0.3]
        with self.assertRaisesRegex(ZoneValidationError, "strictly inside"):
            validate_zones(invalid)

    def test_zone_interiors_do_not_overlap(self):
        invalid = copy.deepcopy(self.zones)
        invalid["zones"][1]["polygon"] = copy.deepcopy(invalid["zones"][0]["polygon"])
        invalid["zones"][1]["center"] = copy.deepcopy(invalid["zones"][0]["center"])
        with self.assertRaisesRegex(ZoneValidationError, "interiors overlap"):
            validate_zones(invalid)

    def test_point_inside_and_outside(self):
        self.assertEqual(point_in_zone(0.3, 0.3, self.zones["zones"][0]), "INSIDE")
        self.assertEqual(point_in_zone(0.9, 0.3, self.zones["zones"][0]), "OUTSIDE")
        self.assertEqual(find_zone_for_point(0.3, 0.9, self.zones)["zone_id"], "zone_C")
        self.assertEqual(find_zone_for_point(1.3, 0.9, self.zones)["status"], "OUTSIDE_WORKSPACE")

    def test_boundary_is_not_silently_assigned(self):
        result = find_zone_for_point(0.6, 0.3, self.zones)
        self.assertEqual(result["status"], "BOUNDARY")
        self.assertEqual(result["zone_ids"], ["zone_A", "zone_B"])
        self.assertIsNone(result["zone_id"])

    def test_white_bag_remains_outside(self):
        result = json.loads(
            (Path(GROUNDING_CONFIG["task_grounding_result_dir"]) / "white_bag_run01.json").read_text()
        )
        world = result["world_coordinate"]
        self.assertEqual(result["active_workspace_status"], "OUTSIDE")
        self.assertEqual(
            find_zone_for_point(world["x"], world["y"], self.zones)["status"],
            "OUTSIDE_WORKSPACE",
        )

    def test_world_to_undistorted_inverse_homography(self):
        pixels = self.geometry.reference_undistorted_pixels
        fitted_world = [
            self.geometry.undistorted_pixel_to_world(u, v) for u, v in pixels
        ]
        round_trip = [
            world_to_undistorted_pixel(x, y, self.geometry.homography)
            for x, y in fitted_world
        ]
        np.testing.assert_allclose(round_trip, pixels, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
