import hashlib
import inspect
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.perception import scene_monitor
from vlm_reposition.perception import image_geometry_audit as audit_module
from vlm_reposition.perception.image_geometry_audit import (
    DEVELOPMENT_PROVENANCE,
    DIAGNOSTIC_ROLE,
    _register,
    audit_geometry,
    audit_legacy_geometry,
    build_replay_manifest,
    classify_registration,
    inventory_images,
    load_audit_config,
    metric_transform_policy,
    orb_features,
    ordered_ratio_matches,
    read_bgr,
    run_offline_replay,
)
from vlm_reposition.perception.scene_monitor import load_scene_monitor_config


PRIOR_RESULT_NAMES = (
    "workspace_state", "physical_geometry", "collision_geometry", "path_planning_baseline",
    "path_planning_stress", "dynamic_path_invalidation", "automatic_replanning",
    "dynamic_obstacle_policy", "live_scene_monitor",
)
PROTECTED_PATHS = [WORKSPACE / "results" / name for name in PRIOR_RESULT_NAMES] + [
    WORKSPACE / "src" / "vlm_reposition" / "geometry" / "legacy.py",
    WORKSPACE / "src" / "vlm_reposition" / "perception" / "scene_monitor.py",
]


def protected_manifest():
    result = {}
    for root in PROTECTED_PATHS:
        paths = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in paths:
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class OfflineImageReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protected_before = protected_manifest()
        cls.config = load_audit_config(WORKSPACE / "config" / "offline_image_geometry_audit.json")
        cls.inventory = inventory_images(cls.config["candidate_directory"], cls.config["reference_image"])
        cls.audit, _ = audit_geometry(cls.config, cls.inventory)
        cls.by_name = {item["filename"]: item for item in cls.audit["images"]}
        cls.inventory_by_name = {item["filename"]: item for item in cls.inventory["images"]}
        cls.manifest = build_replay_manifest(cls.config, cls.audit)
        cls.scene_config = load_scene_monitor_config(WORKSPACE / "config" / "live_scene_monitor.json")
        cls.observations = run_offline_replay(cls.manifest, cls.scene_config)

    def test_01_image_inventory_is_deterministic(self):
        repeated = inventory_images(self.config["candidate_directory"], self.config["reference_image"])
        self.assertEqual(self.inventory, repeated)

    def test_02_photo_04_reference_resolution_is_verified(self):
        item = self.inventory_by_name["photo_04.png"]
        self.assertEqual((item["width"], item["height"]), (2016, 1133))
        self.assertEqual(self.by_name["photo_04.png"]["geometry_compatibility_status"], "REFERENCE_VERIFIED")

    def test_03_raw_like_image_classification(self):
        self.assertEqual(self.inventory_by_name["photo_01.png"]["category"], "RAW_LIKE_SOURCE")
        self.assertEqual(self.inventory_by_name["qwen_frame.jpg"]["category"], "RAW_LIKE_SOURCE")

    def test_04_derived_image_classification(self):
        self.assertEqual(self.inventory_by_name["photo_04_undistorted.png"]["category"], "UNDISTORTED_DERIVED")
        self.assertEqual(self.inventory_by_name["photo_04_bbox_world.png"]["category"], "ANNOTATED_DERIVED")
        self.assertEqual(self.inventory_by_name["photo_04_grid_lines.png"]["category"], "VISUALIZATION_DERIVED")

    def test_05_derived_images_are_excluded_from_replay(self):
        replay_names = {Path(item["path"]).name for item in self.manifest["images"]}
        derived_names = {item["filename"] for item in self.inventory["images"] if item["category"] != "RAW_LIKE_SOURCE"}
        self.assertTrue(replay_names.isdisjoint(derived_names))

    def test_06_same_resolution_detection(self):
        self.assertEqual(self.by_name["photo_01.png"]["resolution_status"], "SAME_RESOLUTION")

    def test_07_resolution_mismatch_classification(self):
        item = self.by_name["qwen_frame.jpg"]
        self.assertEqual(item["resolution_status"], "RESOLUTION_MISMATCH")
        self.assertEqual(item["geometry_compatibility_status"], "RESOLUTION_MISMATCH")

    def test_08_same_resolution_alone_does_not_imply_compatibility(self):
        metrics = dict(self.by_name["photo_01.png"])
        metrics["reference_corner_displacement_px"] = self.config["maximum_corner_displacement_px"] + 1
        self.assertEqual(classify_registration(metrics, "SAME_RESOLUTION", self.config), "POSE_MISMATCH")

    def test_09_orb_setup_is_deterministic(self):
        gray = cv2.cvtColor(read_bgr(self.config["reference_image"]), cv2.COLOR_BGR2GRAY)
        first_keypoints, first_descriptors = orb_features(gray, self.config)
        second_keypoints, second_descriptors = orb_features(gray, self.config)
        self.assertEqual([item.pt for item in first_keypoints], [item.pt for item in second_keypoints])
        self.assertTrue(np.array_equal(first_descriptors, second_descriptors))

    def test_10_match_ordering_is_deterministic(self):
        ref = cv2.cvtColor(read_bgr(self.config["reference_image"]), cv2.COLOR_BGR2GRAY)
        candidate = cv2.cvtColor(read_bgr(self.by_name["photo_01.png"]["absolute_path"]), cv2.COLOR_BGR2GRAY)
        _, ref_descriptors = orb_features(ref, self.config)
        _, candidate_descriptors = orb_features(candidate, self.config)
        _, first = ordered_ratio_matches(candidate_descriptors, ref_descriptors, self.config["ratio_test"])
        _, second = ordered_ratio_matches(candidate_descriptors, ref_descriptors, self.config["ratio_test"])
        signature = lambda values: [(item.distance, item.queryIdx, item.trainIdx) for item in values]
        self.assertEqual(signature(first), signature(second))
        self.assertEqual(signature(first), sorted(signature(first)))

    def test_11_registration_metric_schema(self):
        required = {
            "reference_keypoint_count", "candidate_keypoint_count", "raw_match_count",
            "accepted_match_count", "inlier_count", "inlier_ratio", "reprojection_error_px",
            "median_reprojection_error_px", "reference_corner_displacement_px", "registration_status",
        }
        self.assertTrue(required.issubset(self.by_name["photo_01.png"]))

    def test_12_insufficient_features_are_explicit(self):
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        reference_features = orb_features(cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY), self.config)
        metrics, diagnostic = _register(blank, reference_features, blank.copy(), self.config)
        self.assertEqual(metrics["registration_status"], "INSUFFICIENT_FEATURES")
        self.assertIsNone(diagnostic)

    def test_13_pose_mismatch_classification(self):
        metrics = dict(self.by_name["photo_01.png"])
        metrics["reprojection_error_px"] = self.config["maximum_reprojection_error_px"] + 1
        self.assertEqual(classify_registration(metrics, "SAME_RESOLUTION", self.config), "POSE_MISMATCH")

    def test_14_reference_status_is_not_inferred(self):
        statuses = [item["filename"] for item in self.audit["images"] if item["geometry_compatibility_status"] == "REFERENCE_VERIFIED"]
        self.assertEqual(statuses, ["photo_04.png"])

    def test_15_diagnostic_transform_is_not_metric_homography(self):
        self.assertEqual(self.by_name["photo_01.png"]["diagnostic_transform_role"], DIAGNOSTIC_ROLE)
        self.assertFalse(audit_legacy_geometry(self.config)["metric_transform_executed"])

    def test_16_legacy_h_is_not_scaled_for_mismatch(self):
        item = self.by_name["qwen_frame.jpg"]
        self.assertEqual(item["registration_status"], "RESOLUTION_MISMATCH")
        self.assertEqual(item["legacy_metric_transform_policy"], "NOT_ALLOWED")

    def test_17_images_are_not_resized_to_bypass_compatibility(self):
        qwen = read_bgr(self.by_name["qwen_frame.jpg"]["absolute_path"])
        reference = read_bgr(self.config["reference_image"])
        self.assertEqual(qwen.shape[:2], (480, 640))
        self.assertNotEqual(qwen.shape, reference.shape)

    def test_18_raw_and_undistorted_domains_are_distinct(self):
        raw = self.inventory_by_name["photo_04.png"]
        undistorted = self.inventory_by_name["photo_04_undistorted.png"]
        self.assertEqual((raw["category"], undistorted["category"]), ("RAW_LIKE_SOURCE", "UNDISTORTED_DERIVED"))

    def test_19_pose_mismatch_policy_is_not_allowed(self):
        self.assertEqual(metric_transform_policy("POSE_MISMATCH"), "NOT_ALLOWED")

    def test_20_resolution_mismatch_policy_is_not_allowed(self):
        self.assertEqual(metric_transform_policy("RESOLUTION_MISMATCH"), "NOT_ALLOWED")

    def test_21_insufficient_evidence_policy_is_unknown(self):
        self.assertEqual(metric_transform_policy("INSUFFICIENT_EVIDENCE"), "UNKNOWN")

    def test_22_compatible_candidate_has_development_provenance(self):
        item = self.by_name["photo_01.png"]
        self.assertEqual(item["legacy_metric_transform_policy"], "DEVELOPMENT_INFERRED_ALLOWED")
        self.assertEqual(item["metric_transform_provenance"], DEVELOPMENT_PROVENANCE)

    def test_23_manifest_excludes_incompatible_images(self):
        paths = {item["path"] for item in self.manifest["images"]}
        self.assertNotIn(self.by_name["qwen_frame.jpg"]["absolute_path"], paths)

    def test_24_replay_order_is_deterministic(self):
        filenames = [Path(item["path"]).name for item in self.manifest["images"]]
        self.assertEqual(filenames, sorted(filenames))

    def test_25_replay_order_is_not_called_capture_timeline(self):
        self.assertEqual(self.manifest["ordering_policy"], "DETERMINISTIC_FILENAME_ORDER_NOT_REAL_CAPTURE_TIMELINE")

    def test_26_existing_step_20_scene_monitor_is_reused(self):
        self.assertIs(audit_module.SceneMonitor, scene_monitor.SceneMonitor)

    def test_27_no_qwen_model_load(self):
        source = inspect.getsource(audit_module)
        self.assertNotIn("transformers", source)
        self.assertNotIn("Qwen3VL", source)

    def test_28_no_world_coordinate_generation(self):
        self.assertNotIn("world_xy", repr(self.observations))
        self.assertTrue(all(item["metric_conversion"] == "NOT_PERFORMED" for item in self.observations))

    def test_29_no_physical_obstacle_geometry_generation(self):
        text = repr(self.observations)
        self.assertNotIn("footprint", text)
        self.assertNotIn("width_m", text)

    def test_30_no_planner_injection(self):
        self.assertTrue(all(item["planner_injection"] == "NOT_PERFORMED" for item in self.observations))

    def test_31_offline_replay_is_deterministic(self):
        repeated = run_offline_replay(self.manifest, self.scene_config)
        self.assertEqual(self.observations, repeated)

    def test_32_step_12_through_20_artifacts_are_unchanged(self):
        self.assertEqual(self.protected_before, protected_manifest())


if __name__ == "__main__":
    unittest.main()
