import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.geometry import LegacyGeometry
from vlm_reposition.perception.qwen_grounding import ground_normalized_bbox
from vlm_reposition.perception import semantic_metric_grounding as module
from vlm_reposition.perception.semantic_metric_grounding import (
    DEVELOPMENT_PROVENANCE,
    SemanticMetricGroundingError,
    bbox_association,
    build_metric_observation,
    candidate_prompt_payload,
    load_semantic_metric_config,
    make_candidate_prompt,
    parse_candidate_response,
    process_frame_response,
    select_triggered_frames,
    workspace_status,
)


PROTECTED_PATHS = [
    WORKSPACE / "results" / name
    for name in (
        "workspace_state", "physical_geometry", "collision_geometry", "path_planning_baseline",
        "path_planning_stress", "dynamic_path_invalidation", "automatic_replanning",
        "dynamic_obstacle_policy", "live_scene_monitor", "offline_image_replay",
        "object_grounding", "task_conditioned_grounding", "semantic_refinement",
    )
] + [
    Path("/home/binci/qwen_vision_test/homography_12pt_undistorted.json"),
    Path("/home/binci/qwen_vision_test/photo_04_undistorted.png"),
    Path("/home/binci/camera_calibration_1920x1080/calibration_result.json"),
]


def protected_manifest():
    result = {}
    for root in PROTECTED_PATHS:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in paths:
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class SemanticMetricGroundingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protected_before = protected_manifest()
        cls.config = load_semantic_metric_config(WORKSPACE / "config" / "offline_semantic_metric_grounding.json")
        cls.qwen_config = json.loads(Path(cls.config["model_grounding_config"]).read_text())
        cls.manifest = json.loads(Path(cls.config["input_replay_manifest"]).read_text())
        cls.frames = select_triggered_frames(cls.manifest, cls.config["scene_observation_directory"])
        cls.geometry = LegacyGeometry.load(cls.qwen_config["calibration_path"], cls.qwen_config["legacy_geometry_path"])
        bounds = cls.qwen_config["legacy_workspace_bounds_m"]
        cls.bounds = (*bounds["x"], *bounds["y"])
        cls.candidate = {
            "track_id": "track_test",
            "bbox_px": [900, 500, 1000, 600],
            "track_state": "CONFIRMED",
        }
        cls.frame = {
            "image_id": "image_test",
            "manifest_image": {
                "path": "/home/binci/qwen_vision_test/photo_02.png",
                "compatibility_status": "DEVELOPMENT_COMPATIBLE_CANDIDATE",
                "metric_transform_policy": "DEVELOPMENT_INFERRED_ALLOWED",
            },
            "scene_observation": {
                "source_image": "/home/binci/qwen_vision_test/photo_02.png",
                "scene_state": "LOCAL_CHANGE",
                "semantic_trigger": True,
            },
            "confirmed_candidates": [cls.candidate],
        }
        cls.visible = {
            "track_id": "track_test",
            "candidate_bbox_px": cls.candidate["bbox_px"],
            "status": "VISIBLE_ENTITY_FOUND",
            "visible_label": "test box",
            "semantic_type": "movable_object",
            "object_bbox_normalized_1000": [440, 400, 520, 600],
            "confidence_note": "fixture",
        }
        cls.raw_visible = json.dumps({"image_id": "image_test", "candidate_results": [cls.visible]})

    def observation(self, policy="DEVELOPMENT_INFERRED_ALLOWED", image_size=(2016, 1133)):
        frame = copy.deepcopy(self.frame)
        frame["manifest_image"]["metric_transform_policy"] = policy
        return build_metric_observation(
            frame, frame["confirmed_candidates"][0], self.visible, frame["manifest_image"],
            image_size, self.geometry, self.bounds, self.config["candidate_association"],
        )

    def test_01_step21_replay_manifest_load(self):
        self.assertEqual(self.manifest["group_id"], "legacy_geometry_offline_group_01")
        self.assertEqual(self.manifest["geometry_status"], "OFFLINE_REPLAY_READY")

    def test_02_only_compatible_images_are_accepted_for_metric_path(self):
        self.assertEqual(len(self.frames), 5)
        self.assertTrue(all(frame["manifest_image"]["metric_transform_policy"] in module.ALLOWED_METRIC_POLICIES for frame in self.frames))

    def test_03_not_allowed_image_blocks_metric_grounding(self):
        result = self.observation("NOT_ALLOWED")
        self.assertEqual(result["metric_grounding"]["status"], "METRIC_GROUNDING_NOT_ALLOWED")

    def test_04_semantic_trigger_false_means_qwen_not_run(self):
        selected_names = {Path(frame["scene_observation"]["source_image"]).name for frame in self.frames}
        self.assertNotIn("photo_04.png", selected_names)
        self.assertNotIn("photo_01.png", selected_names)

    def test_05_triggered_candidate_serialization(self):
        self.assertEqual(candidate_prompt_payload([self.candidate]), [{"track_id": "track_test", "candidate_bbox_px": [900, 500, 1000, 600]}])

    def test_06_strict_prompt_preserves_candidate_ids(self):
        prompt = make_candidate_prompt("image_test", [self.candidate])
        self.assertIn("track_id: track_test", prompt)
        self.assertIn("candidate_bbox_px: [900,500,1000,600]", prompt)
        self.assertIn("INTEGER coordinates", prompt)
        with self.assertRaises(SemanticMetricGroundingError):
            make_candidate_prompt("image_test", [self.candidate, self.candidate])

    def test_07_valid_semantic_response_parse(self):
        result = parse_candidate_response(self.raw_visible, "image_test", [self.candidate])
        self.assertEqual(result[0]["visible_label"], "test box")

    def test_08_no_visible_entity_parse(self):
        item = dict(self.visible, status="NO_VISIBLE_ENTITY", visible_label=None, semantic_type="unknown", object_bbox_normalized_1000=None)
        result = parse_candidate_response(json.dumps({"image_id": "image_test", "candidate_results": [item]}), "image_test", [self.candidate])
        self.assertEqual(result[0]["status"], "NO_VISIBLE_ENTITY")

    def test_09_ambiguous_parse(self):
        item = dict(self.visible, status="AMBIGUOUS", visible_label=None, semantic_type="unknown", object_bbox_normalized_1000=None)
        result = parse_candidate_response(json.dumps({"image_id": "image_test", "candidate_results": [item]}), "image_test", [self.candidate])
        self.assertEqual(result[0]["status"], "AMBIGUOUS")

    def test_10_invalid_json_rejected(self):
        with self.assertRaises(SemanticMetricGroundingError) as caught:
            parse_candidate_response("not json", "image_test", [self.candidate])
        self.assertEqual(caught.exception.code, "FRAME_SEMANTIC_PARSE_FAILED")

    def test_11_duplicate_candidate_ids_rejected(self):
        raw = json.dumps({"image_id": "image_test", "candidate_results": [self.visible, self.visible]})
        with self.assertRaises(SemanticMetricGroundingError) as caught:
            parse_candidate_response(raw, "image_test", [self.candidate])
        self.assertEqual(caught.exception.code, "DUPLICATE_CANDIDATE_ID")

    def test_12_unknown_candidate_ids_rejected(self):
        unknown = dict(self.visible, track_id="track_unknown")
        with self.assertRaises(SemanticMetricGroundingError) as caught:
            parse_candidate_response(json.dumps({"image_id": "image_test", "candidate_results": [unknown]}), "image_test", [self.candidate])
        self.assertEqual(caught.exception.code, "UNKNOWN_CANDIDATE_ID")

    def test_13_invalid_normalized_bbox_rejected(self):
        for bbox in ([500, 500, 400, 600], [0.4, 0.3, 0.8, 0.9], [125.0, 240.0, 680.0, 910.0]):
            invalid = dict(self.visible, object_bbox_normalized_1000=bbox)
            with self.subTest(bbox=bbox), self.assertRaises(SemanticMetricGroundingError) as caught:
                parse_candidate_response(json.dumps({"image_id": "image_test", "candidate_results": [invalid]}), "image_test", [self.candidate])
            self.assertEqual(caught.exception.code, "BBOX_INVALID")

    def test_14_bbox_normalized_to_raw_conversion(self):
        result = self.observation()
        self.assertEqual(result["visual_grounding"]["bbox_raw_px"], [887.04, 453.2, 1048.32, 679.8])

    def test_15_bbox_bottom_center_calculation(self):
        result = self.observation()
        self.assertEqual(result["visual_grounding"]["ground_reference_px"], [967.68, 679.8])

    def test_16_change_candidate_bbox_is_not_metric_reference(self):
        result = self.observation()
        self.assertNotEqual(result["visual_grounding"]["ground_reference_px"], [950.0, 600.0])

    def test_17_association_iou_calculation(self):
        result = bbox_association([0, 0, 20, 20], [10, 10, 30, 30], (100, 100), self.config["candidate_association"])
        self.assertAlmostEqual(result["iou"], 100 / 700)

    def test_18_associated_candidate_accepted(self):
        self.assertEqual(self.observation()["association"]["status"], "ASSOCIATED")

    def test_19_weak_association_is_explicit(self):
        thresholds = dict(self.config["candidate_association"], minimum_associated_iou=0.9, maximum_associated_center_distance_ratio=0.0)
        result = bbox_association([0, 0, 10, 10], [9, 9, 20, 20], (100, 100), thresholds)
        self.assertEqual(result["status"], "WEAK_ASSOCIATION")

    def test_20_not_associated_entity_has_no_metric_point(self):
        semantic = dict(self.visible, object_bbox_normalized_1000=[0, 0, 50, 50])
        result = build_metric_observation(self.frame, self.candidate, semantic, self.frame["manifest_image"], self.geometry.image_size, self.geometry, self.bounds, self.config["candidate_association"])
        self.assertEqual(result["association"]["status"], "NOT_ASSOCIATED")
        self.assertIsNone(result["metric_grounding"]["world_xy"])

    def test_21_existing_deterministic_geometry_is_reused(self):
        self.assertIsInstance(self.geometry, LegacyGeometry)
        self.assertEqual(self.geometry.coordinate_system, "LEGACY_WORKSPACE_COORDINATE")

    def test_22_raw_undistorted_world_order_is_preserved(self):
        result = self.observation()
        self.assertEqual(result["metric_grounding"]["pipeline"], "RAW_BOTTOM_CENTER_PX_TO_STORED_UNDISTORTION_TO_LEGACY_HOMOGRAPHY")
        self.assertIn("undistorted_ground_reference_px", result["visual_grounding"])

    def test_23_no_runtime_camera_matrix_recomputation(self):
        self.assertNotIn("getOptimalNewCameraMatrix", inspect.getsource(module))

    def test_24_resolution_mismatch_rejects_metric_transform(self):
        candidate = dict(self.candidate, bbox_px=[280, 200, 350, 300])
        semantic = dict(self.visible, candidate_bbox_px=candidate["bbox_px"], object_bbox_normalized_1000=[400, 350, 600, 700])
        result = build_metric_observation(self.frame, candidate, semantic, self.frame["manifest_image"], (640, 480), self.geometry, self.bounds, self.config["candidate_association"])
        self.assertEqual(result["metric_grounding"]["status"], "METRIC_GROUNDING_NOT_ALLOWED")

    def test_25_reference_allowed_provenance(self):
        result = self.observation("REFERENCE_ALLOWED")
        self.assertEqual(result["source_image"]["geometry_provenance"], "LEGACY_REFERENCE_VERIFIED")

    def test_26_development_inferred_allowed_provenance(self):
        self.assertEqual(self.observation()["source_image"]["geometry_provenance"], DEVELOPMENT_PROVENANCE)

    def test_27_workspace_inside(self):
        self.assertEqual(workspace_status((0.5, 0.8), self.bounds), "INSIDE")

    def test_28_workspace_outside_is_preserved(self):
        self.assertEqual(workspace_status((1.3, 0.8), self.bounds), "OUTSIDE")

    def test_29_physical_geometry_remains_not_available(self):
        self.assertEqual(self.observation()["physical_geometry"], {"status": "NOT_AVAILABLE"})

    def test_30_planning_role_remains_not_assigned(self):
        result = self.observation()
        self.assertEqual(result["semantic"]["planning_role"], "NOT_ASSIGNED")
        self.assertEqual(result["planning"]["role"], "NOT_ASSIGNED")

    def test_31_no_obstacle_polygon_or_radius_generated(self):
        text = repr(self.observation())
        self.assertNotIn("polygon", text)
        self.assertNotIn("radius_m", text)
        self.assertNotIn("width_m", text)

    def test_32_no_step17_18_19_call(self):
        source = inspect.getsource(module)
        self.assertNotIn("vlm_reposition.planning", source)
        self.assertNotIn("dynamic_state", source)
        self.assertNotIn("obstacle_policy", source)

    def test_33_no_astar_execution(self):
        self.assertNotIn("grid_astar", inspect.getsource(module))

    def test_34_semantic_metric_observation_is_deterministic(self):
        first = process_frame_response(self.frame, self.raw_visible, self.geometry, self.bounds, self.config["candidate_association"])
        second = process_frame_response(self.frame, self.raw_visible, self.geometry, self.bounds, self.config["candidate_association"])
        self.assertEqual(first, second)

    def test_35_existing_step05_08_grounding_regression(self):
        first = ground_normalized_bbox("red cup", "red cup", [447, 527, 483, 595], self.geometry.image_size, self.geometry)
        second = ground_normalized_bbox("red cup", "red cup", [447, 527, 483, 595], self.geometry.image_size, self.geometry)
        self.assertEqual(first, second)
        self.assertEqual(first["world_coordinate"]["frame"], "LEGACY_WORKSPACE_COORDINATE")

    def test_36_step12_through_21_artifacts_are_unchanged(self):
        self.assertEqual(self.protected_before, protected_manifest())


if __name__ == "__main__":
    unittest.main()
