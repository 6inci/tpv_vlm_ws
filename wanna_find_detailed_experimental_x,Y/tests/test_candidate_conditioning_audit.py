import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest

import cv2


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.perception import candidate_conditioning_audit as module
from vlm_reposition.perception.candidate_conditioning_audit import (
    CandidateConditioningAuditError,
    audit_prompt,
    candidate_metrics,
    load_audit_config,
    make_audit_visualizations,
    make_context_crop,
    normalize_half_open_bbox,
    sha256_file,
    validate_half_open_bbox,
)
from vlm_reposition.perception.semantic_metric_grounding import make_candidate_prompt


CONFIG = WORKSPACE / "config" / "candidate_conditioning_audit.json"
RESULT = WORKSPACE / "results" / "candidate_conditioning_audit"
PROTECTED = [
    WORKSPACE / "results" / name for name in (
        "workspace_state", "physical_geometry", "collision_geometry", "path_planning_baseline",
        "path_planning_stress", "dynamic_path_invalidation", "automatic_replanning",
        "dynamic_obstacle_policy", "live_scene_monitor", "offline_image_replay",
        "offline_semantic_metric_grounding", "offline_semantic_metric_grounding_representative",
    )
]


def hashes(roots):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for root in roots if root.exists() for path in sorted(root.rglob("*")) if path.is_file()}


def exhaustive_digest():
    root = WORKSPACE / "results" / "offline_semantic_metric_grounding"
    payload = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(WORKSPACE)}\n" for path in sorted(root.rglob("*")) if path.is_file())
    return hashlib.sha256(payload.encode()).hexdigest()


class CandidateConditioningAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = hashes(PROTECTED)
        cls.config = load_audit_config(CONFIG)
        cls.case = cls.config["representative_cases"][0]
        cls.image = cv2.imread("/home/binci/qwen_vision_test/photo_06.png")

    def test_01_representative_case_loading(self):
        self.assertEqual(len(self.config["representative_cases"]), 3)

    def test_02_image_dimension_verification(self):
        self.assertEqual(self.image.shape[:2], (1133, 2016))

    def test_03_raw_candidate_bbox_validation(self):
        self.assertEqual(validate_half_open_bbox([1, 2, 3, 4], (10, 10)), [1, 2, 3, 4])
        with self.assertRaises(CandidateConditioningAuditError):
            validate_half_open_bbox([1, 2, 11, 4], (10, 10))

    def test_04_half_open_bbox_convention_verification(self):
        self.assertEqual(validate_half_open_bbox([0, 0, 10, 10], (10, 10)), [0, 0, 10, 10])

    def test_05_boundary_touch_bbox_accepted(self):
        self.assertEqual(validate_half_open_bbox([5, 2, 10, 9], (10, 10)), [5, 2, 10, 9])

    def test_06_photo06_x2_equals_width_handling(self):
        self.assertEqual(validate_half_open_bbox(self.case["candidate_bbox_px"], (2016, 1133))[2], 2016)

    def test_07_raw_pixel_to_normalized_conversion(self):
        self.assertEqual(normalize_half_open_bbox([1430, 222, 2016, 1092], (2016, 1133)), [709, 196, 1000, 964])

    def test_08_deterministic_rounding(self):
        self.assertEqual(normalize_half_open_bbox([1, 1, 3, 3], (8, 8)), [125, 125, 375, 375])

    def test_09_normalized_bbox_within_range(self):
        self.assertTrue(all(0 <= value <= 1000 for value in normalize_half_open_bbox([0, 0, 2016, 1133], (2016, 1133))))

    def test_10_candidate_area_calculation(self):
        result = candidate_metrics(self.image, [699, 451, 745, 518], 1998)
        self.assertEqual(result["bbox_area_px"], 46 * 67)

    def test_11_border_touch_detection(self):
        self.assertTrue(candidate_metrics(self.image, self.case["candidate_bbox_px"], 1)["touches_image_border"])

    def test_12_overlay_preserves_original_dimensions(self):
        views = make_audit_visualizations(self.image, [699, 451, 745, 518], "track", [347, 398, 370, 457], self.config["visualization"])
        self.assertTrue(all(view.shape == self.image.shape for view in views.values()))

    def test_13_overlay_does_not_modify_source_artifact(self):
        path = Path("/home/binci/qwen_vision_test/photo_06.png")
        before = sha256_file(path)
        make_audit_visualizations(self.image, self.case["candidate_bbox_px"], "track", [709, 196, 1000, 964], self.config["visualization"])
        self.assertEqual(sha256_file(path), before)

    def test_14_crop_generator_correct_bbox(self):
        crop, metadata = make_context_crop(self.image, [100, 100, 200, 200], 0.1)
        self.assertEqual(metadata["source_bbox_px"], [90, 90, 210, 210])
        self.assertEqual(crop.shape[:2], (120, 120))

    def test_15_crop_not_metric_coordinate_source(self):
        _, metadata = make_context_crop(self.image, [100, 100, 200, 200], 0.1)
        self.assertFalse(metadata["metric_coordinate_source"])

    def test_16_reconstructed_prompt_extraction(self):
        candidate = {"track_id": self.case["track_id"], "bbox_px": self.case["candidate_bbox_px"]}
        result = audit_prompt(make_candidate_prompt(self.case["image_id"], [candidate]))
        self.assertEqual(result["classification"], "LIKELY_CONFUSING")

    def test_17_no_qwen_model_load(self):
        source = inspect.getsource(module) + (WORKSPACE / "scripts" / "audit_candidate_conditioning.py").read_text()
        self.assertNotIn("load_local_qwen", source)

    def test_18_no_generation_call(self):
        summary = json.loads((RESULT / "audit_summary.json").read_text())
        self.assertEqual(summary["qwen_inference_runs"], 0)
        self.assertFalse(summary["generation_called"])

    def test_19_no_geometry_modification(self):
        source = inspect.getsource(module) + (WORKSPACE / "scripts" / "audit_candidate_conditioning.py").read_text()
        self.assertNotIn("LegacyGeometry", source)
        self.assertNotIn("ground_normalized_bbox", source)

    def test_20_no_planner_imports(self):
        source = inspect.getsource(module) + (WORKSPACE / "scripts" / "audit_candidate_conditioning.py").read_text()
        self.assertNotIn("vlm_reposition.planning", source)

    def test_21_step22_results_unchanged(self):
        self.assertEqual(exhaustive_digest(), "1de4c7cb70351bfd51b4ab52e69e32548e6aecc9da3e109b324179e0498a32d9")

    def test_22_step12_to_step21_protected_artifacts_unchanged(self):
        self.assertEqual(hashes(PROTECTED), self.before)


if __name__ == "__main__":
    unittest.main()
