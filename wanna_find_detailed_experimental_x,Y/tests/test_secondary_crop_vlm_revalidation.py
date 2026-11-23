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
    "step22d", WORKSPACE / "scripts" / "run_secondary_crop_vlm_revalidation.py"
)
STEP22D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP22D)

from vlm_reposition.geometry import LegacyGeometry
from vlm_reposition.perception.qwen_grounding import (
    normalized_bbox_to_pixel_bbox,
    prepare_qwen_multi_image_inputs,
)
from vlm_reposition.perception.semantic_metric_grounding import (
    SemanticMetricGroundingError,
    annotate_metric_results,
    build_metric_observation,
    load_semantic_metric_config,
    make_method_d_candidate_prompt,
    parse_candidate_response,
)


class SecondaryCropVlmRevalidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_semantic_metric_config(STEP22D.CONFIG)
        cls.case, cls.frame = STEP22D.load_validated_case(cls.config)
        cls.prompt = make_method_d_candidate_prompt(STEP22D.IMAGE_ID, STEP22D.TRACK_ID, STEP22D.NORMALIZED_CANDIDATE_BBOX)
        qwen_config = json.loads(Path(cls.config["model_grounding_config"]).read_text())
        cls.geometry = LegacyGeometry.load(qwen_config["calibration_path"], qwen_config["legacy_geometry_path"])
        bounds = qwen_config["legacy_workspace_bounds_m"]
        cls.bounds = (*bounds["x"], *bounds["y"])

    def test_01_method_d_representative_identity(self):
        self.assertEqual((self.case["image_id"], self.case["track_id"]), ("image_014_photo_08", "track_039"))

    def test_02_full_source_image_correct(self):
        self.assertEqual(Path(self.case["source_image"]), Path("/home/binci/qwen_vision_test/photo_08.png"))

    def test_03_crop_provenance_correct(self):
        self.assertEqual(self.case["candidate_crop"]["source_bbox_px"], [963, 524, 1429, 705])
        self.assertEqual(self.case["candidate_crop"]["context_margin_ratio"], 0.15)

    def test_04_crop_source_candidate_correct(self):
        self.assertEqual(self.case["candidate_bbox_px"], [1017, 545, 1375, 684])

    def test_05_crop_is_semantic_conditioning_only(self):
        self.assertEqual(self.case["candidate_crop"]["coordinate_role"], "SEMANTIC_CONDITIONING_ONLY")

    def test_06_crop_is_not_metric_source(self):
        self.assertFalse(self.case["candidate_crop"]["metric_coordinate_source"])

    def test_07_full_image_dimensions_unchanged(self):
        image = cv2.imread(self.case["source_image"])
        crop = cv2.imread(str(STEP22D.CROP))
        self.assertEqual(image.shape[:2], (1133, 2016))
        self.assertEqual(crop.shape[:2], (181, 466))

    def test_08_candidate_metadata_normalized(self):
        self.assertEqual(self.case["candidate_bbox_normalized_1000"], [504, 481, 682, 604])

    def test_09_raw_bbox_excluded_from_prompt(self):
        self.assertNotIn("candidate_bbox_px", self.prompt)
        self.assertNotIn("[1017", self.prompt)

    def test_10_two_image_input_order_preserved(self):
        class Inputs:
            def to(self, device):
                return self
        class Processor:
            def apply_chat_template(inner, messages, **kwargs):
                inner.messages = messages
                return Inputs()
        class Model:
            device = "cpu"
        processor = Processor()
        prepare_qwen_multi_image_inputs(processor, Model(), ["full.png", "crop.png"], "prompt")
        content = processor.messages[0]["content"]
        self.assertEqual([item.get("image") for item in content[:2]], ["full.png", "crop.png"])
        self.assertEqual(content[2], {"type": "text", "text": "prompt"})

    def test_11_image_1_bbox_requirement_explicit(self):
        self.assertIn("IMAGE 1 coordinates, not IMAGE 2 crop coordinates", self.prompt)
        self.assertIn('bbox_reference_image must be exactly "IMAGE_1"', self.prompt)

    def test_12_crop_coordinate_output_not_remapped(self):
        item = {"track_id": "track_039", "status": "VISIBLE_ENTITY_FOUND", "visible_label": "racket", "semantic_type": "movable_object", "object_bbox_normalized_1000": [100, 100, 900, 900], "confidence_note": "crop-like"}
        raw = json.dumps({"image_id": STEP22D.IMAGE_ID, "candidate_results": [item]})
        with self.assertRaises(SemanticMetricGroundingError) as caught:
            parse_candidate_response(raw, STEP22D.IMAGE_ID, self.frame["confirmed_candidates"], False, "IMAGE_1")
        self.assertEqual(caught.exception.code, "BBOX_REFERENCE_AMBIGUOUS")

    def test_13_strict_bbox_parser_unchanged(self):
        item = {"track_id": "track_039", "status": "VISIBLE_ENTITY_FOUND", "visible_label": "racket", "semantic_type": "movable_object", "object_bbox_normalized_1000": [0.1, 0.2, 0.8, 0.9], "bbox_reference_image": "IMAGE_1", "confidence_note": "invalid"}
        with self.assertRaises(SemanticMetricGroundingError) as caught:
            parse_candidate_response(json.dumps({"image_id": STEP22D.IMAGE_ID, "candidate_results": [item]}), STEP22D.IMAGE_ID, self.frame["confirmed_candidates"], False, "IMAGE_1")
        self.assertEqual(caught.exception.code, "BBOX_INVALID")

    def test_14_normalized_to_raw_uses_full_image(self):
        self.assertEqual(normalized_bbox_to_pixel_bbox([500, 400, 600, 700], STEP22D.IMAGE_SIZE), (1008.0, 453.2, 1209.6, 793.1))

    def test_15_association_threshold_unchanged(self):
        self.assertEqual(self.config["candidate_association"], {"minimum_associated_iou": 0.05, "maximum_associated_center_distance_ratio": 0.08, "minimum_weak_iou": 0.001, "maximum_weak_center_distance_ratio": 0.15})

    def test_16_metric_only_on_associated(self):
        semantic = {"track_id": "track_039", "candidate_bbox_px": STEP22D.RAW_CANDIDATE_BBOX, "status": "VISIBLE_ENTITY_FOUND", "visible_label": "far object", "semantic_type": "movable_object", "object_bbox_normalized_1000": [0, 0, 10, 10], "confidence_note": "fixture"}
        result = build_metric_observation(self.frame, self.frame["confirmed_candidates"][0], semantic, self.frame["manifest_image"], STEP22D.IMAGE_SIZE, self.geometry, self.bounds, self.config["candidate_association"])
        self.assertNotEqual(result["association"]["status"], "ASSOCIATED")
        self.assertEqual(result["metric_grounding"]["status"], "NOT_RUN")
        image = cv2.imread(self.case["source_image"])
        self.assertEqual(annotate_metric_results(image, {"metric_observations": [result]}).shape, image.shape)

    def test_17_one_inference_guard(self):
        source = inspect.getsource(STEP22D)
        self.assertEqual(source.count("generate_qwen_text(processor, model, inputs"), 1)
        self.assertIn("one-inference guard", source)

    def test_18_no_retry_or_fallback(self):
        source = inspect.getsource(STEP22D)
        self.assertIn('"fallback_candidate_runs": 0', source)
        self.assertNotIn("for run", source)

    def test_19_no_planner_imports(self):
        source = inspect.getsource(STEP22D)
        self.assertNotIn("vlm_reposition.planning", source)
        self.assertNotIn("grid_astar", source)

    def test_20_step12_to_step22c_artifacts_unchanged(self):
        self.assertEqual(STEP22D.snapshot_protected(STEP22D.DEFAULT_OUTPUT)["digest_sha256"], "4d9e505794dcfe9f843debcc6d7ef6145c346d6e512b5c3d6174a245d60eb2c7")
        summary = STEP22D.STEP22C_RESULT / "step22c_summary.json"
        self.assertEqual(hashlib.sha256(summary.read_bytes()).hexdigest(), "84ffc08fb7d760cc7bc5304341469044f256ad4cad29c8229b1eecad6b6c09b7")


if __name__ == "__main__":
    unittest.main()
