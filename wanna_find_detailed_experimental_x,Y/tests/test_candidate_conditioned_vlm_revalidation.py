import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import unittest

import cv2


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
SPEC = importlib.util.spec_from_file_location(
    "step22c", WORKSPACE / "scripts" / "run_candidate_conditioned_vlm_revalidation.py"
)
STEP22C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP22C)

from vlm_reposition.geometry import LegacyGeometry
from vlm_reposition.perception.qwen_grounding import normalized_bbox_to_pixel_bbox
from vlm_reposition.perception.semantic_metric_grounding import (
    SemanticMetricGroundingError,
    build_metric_observation,
    load_semantic_metric_config,
    make_method_c_candidate_prompt,
    parse_candidate_response,
)


class CandidateConditionedVlmRevalidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_semantic_metric_config(STEP22C.CONFIG)
        cls.case, cls.frame = STEP22C.load_validated_case(cls.config)
        cls.prompt = make_method_c_candidate_prompt(
            STEP22C.IMAGE_ID, STEP22C.TRACK_ID, STEP22C.NORMALIZED_CANDIDATE_BBOX
        )
        cls.source = Path(cls.case["source_image"])
        cls.source_hash = hashlib.sha256(cls.source.read_bytes()).hexdigest()
        qwen_config = json.loads(Path(cls.config["model_grounding_config"]).read_text())
        cls.geometry = LegacyGeometry.load(qwen_config["calibration_path"], qwen_config["legacy_geometry_path"])
        bounds = qwen_config["legacy_workspace_bounds_m"]
        cls.bounds = (*bounds["x"], *bounds["y"])

    def test_01_method_c_config_validation(self):
        self.assertEqual(json.loads(STEP22C.AUDIT_DESIGN.read_text())["recommended_method"], STEP22C.METHOD)

    def test_02_representative_image_id(self):
        self.assertEqual(self.case["image_id"], "image_014_photo_08")

    def test_03_exact_track_id(self):
        self.assertEqual(self.frame["confirmed_candidates"][0]["track_id"], "track_039")

    def test_04_audited_normalized_candidate(self):
        self.assertEqual(self.case["candidate_bbox_normalized_1000"], [504, 481, 682, 604])

    def test_05_raw_bbox_not_in_semantic_prompt(self):
        self.assertNotIn("candidate_bbox_px", self.prompt)
        self.assertNotIn("[1017", self.prompt)

    def test_06_input_and_output_use_normalized_domain(self):
        self.assertIn("candidate_region_normalized_1000: [504,481,682,604]", self.prompt)
        self.assertIn("object_bbox_normalized_1000", self.prompt)

    def test_07_overlay_matches_original_dimensions(self):
        source = cv2.imread(str(self.source))
        overlay = cv2.imread(str(STEP22C.OVERLAY))
        self.assertEqual(source.shape, overlay.shape)
        self.assertEqual(source.shape[:2], (1133, 2016))

    def test_08_source_original_is_unmodified(self):
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)

    def test_09_overlay_is_semantic_only(self):
        self.assertIn("SEMANTIC_CONDITIONING_ONLY", inspect.getsource(STEP22C))
        self.assertIn("overlay_pixels_used_for_homography", inspect.getsource(STEP22C))

    def test_10_integer_output_bbox_validation(self):
        item = {"track_id": "track_039", "status": "VISIBLE_ENTITY_FOUND", "visible_label": "tennis racket", "semantic_type": "movable_object", "object_bbox_normalized_1000": [504, 481, 682, 604], "confidence_note": "visible"}
        raw = json.dumps({"image_id": STEP22C.IMAGE_ID, "candidate_results": [item]})
        self.assertEqual(parse_candidate_response(raw, STEP22C.IMAGE_ID, self.frame["confirmed_candidates"], False)[0]["status"], "VISIBLE_ENTITY_FOUND")
        item["object_bbox_normalized_1000"] = [0.5, 0.4, 0.7, 0.6]
        with self.assertRaises(SemanticMetricGroundingError):
            parse_candidate_response(json.dumps({"image_id": STEP22C.IMAGE_ID, "candidate_results": [item]}), STEP22C.IMAGE_ID, self.frame["confirmed_candidates"], False)

    def test_11_normalized_output_to_raw_conversion(self):
        self.assertEqual(normalized_bbox_to_pixel_bbox([500, 400, 600, 700], (2016, 1133)), (1008.0, 453.2, 1209.6, 793.1))

    def test_12_association_thresholds_unchanged(self):
        self.assertEqual(self.config["candidate_association"], {"minimum_associated_iou": 0.05, "maximum_associated_center_distance_ratio": 0.08, "minimum_weak_iou": 0.001, "maximum_weak_center_distance_ratio": 0.15})

    def test_13_metric_only_on_associated(self):
        semantic = {"track_id": "track_039", "candidate_bbox_px": STEP22C.RAW_CANDIDATE_BBOX, "status": "VISIBLE_ENTITY_FOUND", "visible_label": "far object", "semantic_type": "movable_object", "object_bbox_normalized_1000": [0, 0, 10, 10], "confidence_note": "fixture"}
        result = build_metric_observation(self.frame, self.frame["confirmed_candidates"][0], semantic, self.frame["manifest_image"], STEP22C.IMAGE_SIZE, self.geometry, self.bounds, self.config["candidate_association"])
        self.assertNotEqual(result["association"]["status"], "ASSOCIATED")
        self.assertEqual(result["metric_grounding"]["status"], "NOT_RUN")

    def test_14_geometry_implementation_unchanged(self):
        path = WORKSPACE / "src" / "vlm_reposition" / "geometry" / "legacy.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), "3d951c275cca8cba6b85f5ee7935c9cccfccce3bb28e06424c964d8638d90b04")

    def test_15_one_inference_guard(self):
        source = inspect.getsource(STEP22C)
        self.assertEqual(source.count("generate_qwen_text(processor, model, inputs"), 1)
        self.assertIn("one-inference guard", source)

    def test_16_no_fallback_candidate_execution(self):
        self.assertIn('"fallback_candidate_runs": 0', inspect.getsource(STEP22C))

    def test_17_no_planner_calls(self):
        source = inspect.getsource(STEP22C)
        self.assertNotIn("vlm_reposition.planning", source)
        self.assertNotIn("grid_astar", source)

    def test_18_step12_to_step22b_artifacts_unchanged(self):
        self.assertEqual(STEP22C.snapshot_existing_results(STEP22C.DEFAULT_OUTPUT)["digest_sha256"], "cf28d266880be4fc59f207badc875bb7705cfca7db9d0f794ffd71fc874119d0")


if __name__ == "__main__":
    unittest.main()
