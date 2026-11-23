import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(WORKSPACE / "scripts"))

import run_live_scene_monitor as live_runner
import verify_scene_monitor_offline as offline_runner
from vlm_reposition.perception import (
    CameraSource,
    CameraSourceError,
    ChangeTracker,
    SceneMonitor,
    SceneMonitorError,
    audit_camera_sources,
    detect_change_candidates,
    load_scene_monitor_config,
    pixel_bbox_iou,
    preprocess_frame,
    select_stable_reference,
    validate_frame,
    validate_scene_monitor_config,
)


PROTECTED_DIRS = [
    WORKSPACE / "results" / name
    for name in (
        "workspace_state", "physical_geometry", "collision_geometry", "path_planning_baseline",
        "path_planning_stress", "dynamic_path_invalidation", "automatic_replanning",
        "dynamic_obstacle_policy",
    )
]


def manifest(directories):
    result = {}
    for directory in directories:
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            result[str(path.relative_to(WORKSPACE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def candidate(index, bbox, centroid):
    return {
        "candidate_index": index,
        "bbox_px": bbox,
        "centroid_px": centroid,
        "area_px": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
        "area_ratio": 0.01,
        "frame_index": 0,
        "source": "FAST_SCENE_CHANGE",
        "candidate_type": "CHANGE_CANDIDATE",
        "semantic_status": "NOT_EVALUATED",
        "metric_geometry_status": "NOT_AVAILABLE",
        "planning_role": "NOT_ASSIGNED",
    }


class SceneMonitorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protected_before = manifest(PROTECTED_DIRS)
        cls.config = load_scene_monitor_config(WORKSPACE / "config" / "live_scene_monitor.json")
        cls.outputs, cls.frames = offline_runner.run_offline_scenarios(cls.config)
        cls.empty = offline_runner.frame_with_rectangles()

    def test_01_camera_selector_config_validation(self):
        selector = self.config["camera_selector"]
        self.assertEqual(selector["policy"], "EXACT_EXTERNAL_BY_ID_ONLY")
        self.assertFalse(selector["allow_fallback"])

    def test_02_fallback_enabled_is_rejected(self):
        invalid = copy.deepcopy(self.config)
        invalid["camera_selector"]["allow_fallback"] = True
        with self.assertRaises(SceneMonitorError) as caught:
            validate_scene_monitor_config(invalid)
        self.assertEqual(caught.exception.code, "INVALID_CONFIG")

    def test_03_even_gaussian_kernel_is_rejected(self):
        invalid = copy.deepcopy(self.config)
        invalid["preprocess"]["gaussian_kernel_size"] = 4
        with self.assertRaises(SceneMonitorError):
            validate_scene_monitor_config(invalid)

    def test_04_frame_shape_validation(self):
        validate_frame(self.empty)
        for invalid in (np.zeros((10, 10), np.uint8), np.zeros((10, 10, 4), np.uint8), np.zeros((10, 10, 3), np.float32)):
            with self.subTest(shape=invalid.shape), self.assertRaises(SceneMonitorError) as caught:
                validate_frame(invalid)
            self.assertEqual(caught.exception.code, "INVALID_FRAME_SHAPE")

    def test_05_preprocess_is_deterministic(self):
        first = preprocess_frame(self.empty, self.config)
        second = preprocess_frame(self.empty.copy(), self.config)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, self.empty.shape[:2])

    def test_06_stable_reference_selection(self):
        reference, result = select_stable_reference([self.empty.copy() for _ in range(4)], self.config)
        self.assertEqual(result["status"], "REFERENCE_STABLE")
        self.assertEqual(result["attempts"], 4)
        self.assertTrue(np.array_equal(reference, self.empty))

    def test_07_unstable_reference_is_explicit(self):
        config = copy.deepcopy(self.config)
        config["maximum_reference_attempts"] = 4
        frames = [offline_runner.frame_with_rectangles(value=value) for value in (20, 220, 20, 220)]
        with self.assertRaises(SceneMonitorError) as caught:
            select_stable_reference(frames, config)
        self.assertEqual(caught.exception.code, "REFERENCE_NOT_STABLE")

    def test_08_no_change_mask(self):
        state, score, candidates, mask = detect_change_candidates(self.empty, self.empty.copy(), self.config, 0)
        self.assertEqual(state, "NO_SIGNIFICANT_CHANGE")
        self.assertEqual(score["changed_pixel_count"], 0)
        self.assertFalse(candidates)
        self.assertEqual(int(mask.max()), 0)

    def test_09_local_change_detection(self):
        frame = offline_runner.frame_with_rectangles([(50, 40, 80, 70)])
        state, _, candidates, _ = detect_change_candidates(self.empty, frame, self.config, 7)
        self.assertEqual(state, "LOCAL_CHANGE")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["frame_index"], 7)

    def test_10_minimum_area_filter(self):
        frame = offline_runner.frame_with_rectangles([(20, 20, 23, 23)])
        state, _, candidates, _ = detect_change_candidates(self.empty, frame, self.config, 0)
        self.assertEqual(state, "NO_SIGNIFICANT_CHANGE")
        self.assertFalse(candidates)

    def test_11_component_bbox_calculation(self):
        frame = offline_runner.frame_with_rectangles([(50, 40, 80, 70)])
        candidate_result = detect_change_candidates(self.empty, frame, self.config, 0)[2][0]
        self.assertEqual(candidate_result["bbox_px"], [49, 39, 81, 71])

    def test_12_component_centroid_calculation(self):
        frame = offline_runner.frame_with_rectangles([(50, 40, 80, 70)])
        centroid = detect_change_candidates(self.empty, frame, self.config, 0)[2][0]["centroid_px"]
        self.assertEqual(centroid, [64.5, 54.5])

    def test_13_changed_pixel_ratio(self):
        observation = self.outputs["F3"]["observations"][0]
        score = observation["change_score"]
        self.assertAlmostEqual(score["changed_pixel_ratio"], score["changed_pixel_count"] / (120 * 160))

    def test_14_global_change_detection(self):
        observation = self.outputs["F6"]["observations"][0]
        self.assertEqual(observation["scene_state"], "GLOBAL_CHANGE")
        self.assertGreaterEqual(observation["change_score"]["changed_pixel_ratio"], self.config["global_change_threshold"])

    def test_15_global_change_is_not_candidate(self):
        observation = self.outputs["F6"]["observations"][0]
        self.assertFalse(observation["candidates"])
        self.assertFalse(observation["semantic_trigger"])

    def test_16_single_frame_noise_no_stable_trigger(self):
        self.assertFalse(any(item["semantic_trigger"] for item in self.outputs["F2"]["observations"]))

    def test_17_persistent_change_stable_trigger(self):
        self.assertEqual([item["semantic_trigger"] for item in self.outputs["F3"]["observations"]], [False, True, True])

    def test_18_track_id_is_deterministic(self):
        self.assertEqual([item["candidates"][0]["track_id"] for item in self.outputs["F3"]["observations"]], ["track_001"] * 3)

    def test_19_iou_association(self):
        tracker = ChangeTracker(self.config["tracking"])
        tracker.update([candidate(1, [10, 10, 30, 30], [20.0, 20.0])], 0)
        tracked = tracker.update([candidate(1, [12, 10, 32, 30], [22.0, 20.0])], 1)
        self.assertEqual(tracked[0]["track_id"], "track_001")
        self.assertGreater(pixel_bbox_iou([10, 10, 30, 30], [12, 10, 32, 30]), 0.1)

    def test_20_centroid_fallback_association(self):
        tracker = ChangeTracker(self.config["tracking"])
        tracker.update([candidate(1, [10, 10, 20, 20], [15.0, 15.0])], 0)
        tracked = tracker.update([candidate(1, [25, 10, 35, 20], [30.0, 15.0])], 1)
        self.assertEqual(tracked[0]["track_id"], "track_001")

    def test_21_association_tie_break_is_lexical(self):
        tracker = ChangeTracker(self.config["tracking"])
        tracker.update([
            candidate(1, [0, 0, 10, 10], [5.0, 5.0]),
            candidate(2, [20, 0, 30, 10], [25.0, 5.0]),
        ], 0)
        tracked = tracker.update([candidate(1, [10, 20, 20, 30], [15.0, 25.0])], 1)
        self.assertEqual(tracked[0]["track_id"], "track_001")

    def test_22_track_missing_count(self):
        tracks = self.outputs["F5"]["observations"][2]["active_tracks"]
        self.assertEqual((tracks[0]["track_state"], tracks[0]["consecutive_missing_count"]), ("MISSING", 1))

    def test_23_track_retirement(self):
        result = self.outputs["F5"]
        self.assertFalse(result["observations"][-1]["active_tracks"])
        self.assertEqual(result["retired_tracks"][0]["track_state"], "RETIRED")

    def test_24_two_candidates_have_separate_ids(self):
        ids = [item["track_id"] for item in self.outputs["F7"]["observations"][-1]["candidates"]]
        self.assertEqual(ids, ["track_001", "track_002"])

    def test_25_merge_split_does_not_crash(self):
        result = self.outputs["F8"]
        self.assertEqual(len(result["observations"]), 3)
        self.assertEqual([len(item["candidates"]) for item in result["observations"]], [2, 1, 2])

    def test_26_candidate_semantic_fields_are_deferred(self):
        item = self.outputs["F3"]["observations"][0]["candidates"][0]
        self.assertEqual(item["semantic_status"], "NOT_EVALUATED")
        self.assertEqual(item["metric_geometry_status"], "NOT_AVAILABLE")
        self.assertEqual(item["planning_role"], "NOT_ASSIGNED")

    def test_27_no_world_coordinate_generated(self):
        item = self.outputs["F3"]["observations"][0]["candidates"][0]
        self.assertNotIn("world_xy", item)
        self.assertNotIn("world_coordinate", item)

    def test_28_no_planner_obstacle_generated(self):
        observation = self.outputs["F3"]["observations"][1]
        self.assertEqual(observation["planner_injection"], "NOT_PERFORMED")
        self.assertNotIn("obstacle", observation["candidates"][0])

    def test_29_latest_observation_overwrites_old_frame(self):
        monitor = SceneMonitor(self.config, self.empty, {"source_id": "test", "source_path": "fixture"})
        monitor.process(self.empty, 0)
        monitor.process(self.empty, 1)
        self.assertEqual(monitor.latest_observation["frame_index"], 1)

    def test_30_get_latest_trigger_only_returns_confirmed_local(self):
        monitor = SceneMonitor(self.config, self.empty, {"source_id": "test", "source_path": "fixture"})
        rectangle = offline_runner.frame_with_rectangles([(50, 40, 80, 70)])
        monitor.process(rectangle, 0)
        self.assertIsNone(monitor.get_latest_trigger())
        monitor.process(rectangle, 1)
        self.assertEqual(monitor.get_latest_trigger()["frame_index"], 1)

    def test_31_camera_unavailable_is_explicit(self):
        audit = audit_camera_sources(self.config, by_id_paths=[], video_paths=[])
        self.assertEqual(audit["audit_status"], "CAMERA_NOT_FOUND")
        self.assertIsNone(audit["selected_device_path"])

    def test_32_builtin_camera_is_not_fallback(self):
        audit = audit_camera_sources(
            self.config,
            by_id_paths=["/dev/v4l/by-id/usb-HP_True_Vision-video-index0"],
            video_paths=[],
        )
        self.assertIsNone(audit["selected_device_path"])
        self.assertFalse(audit["fallback_used"])

    def test_33_exact_external_path_can_be_selected(self):
        path = self.config["camera_selector"]["device_path"]
        audit = audit_camera_sources(self.config, by_id_paths=[path], video_paths=[])
        self.assertEqual(audit["selected_device_path"], path)

    def test_34_camera_source_does_not_open_when_unavailable(self):
        audit = audit_camera_sources(self.config, by_id_paths=[], video_paths=[])
        with self.assertRaises(CameraSourceError) as caught:
            CameraSource(self.config, audit).open()
        self.assertEqual(caught.exception.code, "CAMERA_NOT_FOUND")

    def test_35_bounded_live_runner_terminates_without_camera(self):
        audit = audit_camera_sources(self.config, by_id_paths=[], video_paths=[])
        result = live_runner.run_live_validation(self.config, audit, requested_frames=5)
        self.assertEqual(result["live_external_camera_validation"], "NOT_AVAILABLE")
        self.assertEqual((result["frames_requested"], result["frames_captured"]), (5, 0))

    def test_36_offline_observations_are_deterministic(self):
        repeated, _ = offline_runner.run_offline_scenarios(self.config)
        self.assertEqual(offline_runner.semantic_result(self.outputs), offline_runner.semantic_result(repeated))

    def test_37_f1_no_change(self):
        self.assertEqual({item["scene_state"] for item in self.outputs["F1"]["observations"]}, {"NO_SIGNIFICANT_CHANGE"})

    def test_38_f4_track_continuity(self):
        ids = {item["candidates"][0]["track_id"] for item in self.outputs["F4"]["observations"]}
        self.assertEqual(ids, {"track_001"})

    def test_39_f6_global_change_not_triggered(self):
        observation = self.outputs["F6"]["observations"][0]
        self.assertEqual((observation["scene_state"], observation["semantic_trigger"]), ("GLOBAL_CHANGE", False))

    def test_40_no_qwen_homography_or_planning_import(self):
        source = (WORKSPACE / "src" / "vlm_reposition" / "perception" / "scene_monitor.py").read_text()
        for forbidden in ("transformers", "Qwen", "LegacyGeometry", "Homography", "vlm_reposition.planning"):
            self.assertNotIn(forbidden, source)

    def test_41_ignore_region_suppresses_change(self):
        config = copy.deepcopy(self.config)
        config["ignore_regions_px"] = [[45, 35, 85, 75]]
        frame = offline_runner.frame_with_rectangles([(50, 40, 80, 70)])
        state, _, candidates, _ = detect_change_candidates(self.empty, frame, config, 0)
        self.assertEqual(state, "NO_SIGNIFICANT_CHANGE")
        self.assertFalse(candidates)

    def test_42_step_12_through_19_artifacts_unchanged(self):
        self.assertEqual(self.protected_before, manifest(PROTECTED_DIRS))


if __name__ == "__main__":
    unittest.main()
