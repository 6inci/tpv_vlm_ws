#!/usr/bin/env python3
"""Audit STEP 22 candidate conditioning without loading or running Qwen."""

import argparse
import json
from pathlib import Path
import sys

import cv2


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.perception.candidate_conditioning_audit import (
    audit_prompt,
    candidate_metrics,
    load_audit_config,
    make_audit_visualizations,
    make_context_crop,
    normalize_half_open_bbox,
    raw_output_snapshot,
    sha256_file,
    validate_half_open_bbox,
)
from vlm_reposition.perception.image_geometry_audit import read_bgr
from vlm_reposition.perception.semantic_metric_grounding import make_candidate_prompt


DEFAULT_CONFIG = WORKSPACE / "config" / "candidate_conditioning_audit.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "candidate_conditioning_audit"
RAW_ROOTS = (
    WORKSPACE / "results" / "offline_semantic_metric_grounding",
    WORKSPACE / "results" / "offline_semantic_metric_grounding_representative",
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    config = load_audit_config(args.config)
    raw_before = raw_output_snapshot(RAW_ROOTS)
    prompt_dir = args.output_dir / "reconstructed_prompts"
    visual_dir = args.output_dir / "visualizations"
    crop_dir = args.output_dir / "candidate_crops"
    for directory in (prompt_dir, visual_dir, crop_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cases, prompt_records, coordinate_records, output_records = [], [], [], []
    for case in config["representative_cases"]:
        observation = json.loads(Path(case["observation_path"]).read_text())
        frame_result = json.loads(Path(case["frame_result_path"]).read_text())
        candidates = [item for item in observation["candidates"] if item["track_id"] == case["track_id"]]
        if len(candidates) != 1 or frame_result["image_id"] != case["image_id"]:
            raise ValueError(f"representative evidence mismatch: {case['case_id']}")
        candidate = candidates[0]
        if candidate["bbox_px"] != case["candidate_bbox_px"] or candidate["track_state"] != "CONFIRMED":
            raise ValueError(f"candidate identity mismatch: {case['case_id']}")

        source_path = Path(observation["source_image"])
        image = read_bgr(source_path)
        image_size = (image.shape[1], image.shape[0])
        bbox = validate_half_open_bbox(candidate["bbox_px"], image_size)
        normalized = normalize_half_open_bbox(bbox, image_size)
        metrics = candidate_metrics(image, bbox, candidate["area_px"])

        prompt = make_candidate_prompt(case["image_id"], [candidate])
        prompt_path = prompt_dir / f"{case['case_id']}_prompt.txt"
        prompt_path.write_text(prompt + "\n")
        prompt_result = audit_prompt(prompt)
        prompt_records.append({
            "case_id": case["case_id"], "prompt_path": str(prompt_path),
            "prompt_sha256": sha256_file(prompt_path), **prompt_result,
        })

        views = make_audit_visualizations(image, bbox, case["track_id"], normalized, config["visualization"])
        visualization_paths = {}
        for name, view in views.items():
            path = visual_dir / f"{case['case_id']}_{name}.png"
            if not cv2.imwrite(str(path), view):
                raise RuntimeError(f"could not write {path}")
            visualization_paths[name] = str(path)
        crop, crop_metadata = make_context_crop(image, bbox, config["candidate_crop"]["context_margin_ratio"])
        crop_path = crop_dir / f"{case['case_id']}_context.png"
        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError(f"could not write {crop_path}")

        semantic = frame_result["semantic_results"][0]
        raw_output_text = Path(case["raw_output_path"]).read_text()
        output_records.append({
            "case_id": case["case_id"], "raw_output_path": case["raw_output_path"],
            "raw_output_sha256": sha256_file(case["raw_output_path"]),
            "status": semantic["status"], "confidence_note": semantic["confidence_note"],
            "candidate_location_explanation_present": "bbox" in (semantic["confidence_note"] or "").lower(),
            "boundary_mentioned": "boundar" in (semantic["confidence_note"] or "").lower(),
            "raw_output_bytes": len(raw_output_text.encode()),
        })
        coordinate_records.append({
            "case_id": case["case_id"], "image_width": image_size[0], "image_height": image_size[1],
            "candidate_bbox_px": bbox, "candidate_bbox_normalized_1000": normalized,
            "coordinate_status": "AUDIT_DERIVED_COORDINATE", "rounding": "ROUND_HALF_UP",
        })
        cases.append({
            "case_id": case["case_id"], "image_id": case["image_id"], "source_image": str(source_path),
            "source_image_sha256": sha256_file(source_path), "image_size": list(image_size),
            "track_id": case["track_id"], "track_state": candidate["track_state"],
            "candidate_bbox_px": bbox, "candidate_bbox_normalized_1000": normalized,
            "visual_target": case["visual_target"], "candidate_shape_class": case["candidate_shape_class"],
            "metrics": metrics, "semantic_result": semantic,
            "visualizations": visualization_paths,
            "candidate_crop": {"path": str(crop_path), **crop_metadata},
        })

    bbox_convention = {
        "status": "CONFIRMED",
        "convention": "HALF_OPEN_[x1,y1,x2,y2)",
        "source": "scene_monitor.py detect_change_candidates",
        "construction": "[x, y, x + CC_STAT_WIDTH, y + CC_STAT_HEIGHT]",
        "valid_bounds": "0 <= x1 < x2 <= width; 0 <= y1 < y2 <= height",
        "photo06_x2_equals_width": "VALID_RIGHT_BOUNDARY_TOUCH",
        "prompt_convention_explicit": False,
    }
    representation_comparison = {
        "methods": [
            {"method": "A_RAW_PIXEL_COORDINATES_ONLY", "coordinate_ambiguity": "HIGH", "region_salience": "LOW", "full_scene_context": "HIGH", "original_geometry_preservation": "HIGH", "implementation_complexity": "LOW", "coordinate_confusion_risk": "HIGH", "parser_compatibility": "HIGH", "geometry_compatibility": "HIGH", "new_dependency": "NONE", "auditability": "MEDIUM"},
            {"method": "B_NORMALIZED_COORDINATES_ONLY", "coordinate_ambiguity": "LOW", "region_salience": "LOW", "full_scene_context": "HIGH", "original_geometry_preservation": "HIGH", "implementation_complexity": "LOW", "coordinate_confusion_risk": "LOW", "parser_compatibility": "HIGH", "geometry_compatibility": "HIGH", "new_dependency": "NONE", "auditability": "HIGH"},
            {"method": "C_FULL_IMAGE_OVERLAY_PLUS_NORMALIZED_COORDINATES", "coordinate_ambiguity": "LOW", "region_salience": "HIGH", "full_scene_context": "HIGH", "original_geometry_preservation": "HIGH", "implementation_complexity": "LOW", "coordinate_confusion_risk": "LOW", "parser_compatibility": "HIGH", "geometry_compatibility": "HIGH", "new_dependency": "NONE", "auditability": "HIGH"},
            {"method": "D_FULL_IMAGE_PLUS_CONTEXT_CROP", "coordinate_ambiguity": "MEDIUM", "region_salience": "HIGH", "full_scene_context": "HIGH", "original_geometry_preservation": "HIGH", "implementation_complexity": "MEDIUM", "coordinate_confusion_risk": "MEDIUM", "parser_compatibility": "HIGH", "geometry_compatibility": "HIGH", "new_dependency": "NONE", "auditability": "HIGH"}
        ],
        "performance_winner_claim": "NOT_EVALUATED_NO_INFERENCE",
    }
    recommendation = {
        "step": "STEP_22C_CANDIDATE_CONDITIONED_VLM_GROUNDING_REVALIDATION",
        "recommended_method": "C_FULL_IMAGE_OVERLAY_PLUS_NORMALIZED_COORDINATES",
        "semantic_input": "FULL_RESOLUTION_SAME_DOMAIN_IMAGE_WITH_DETERMINISTIC_VISIBLE_RECTANGLE_AND_TRACK_ID",
        "candidate_metadata": "NORMALIZED_INTEGER_0_1000_HALF_OPEN_DERIVED_FROM_RAW",
        "metric_coordinate_source": "VALIDATED_FULL_OBJECT_BBOX_IN_ORIGINAL_FULL_IMAGE_DOMAIN_ONLY",
        "rationale": ["removes raw-versus-normalized prompt ambiguity", "raises region salience without cropping or resizing", "retains full-scene context", "reuses current parser and geometry", "requires no dependency"],
        "fallback_to_evaluate_later": "D_FULL_IMAGE_PLUS_SECONDARY_CONTEXT_CROP",
        "actual_qwen_validation": "NOT_RUN_IN_STEP_22B",
    }

    write_json(args.output_dir / "representative_candidates.json", {"schema_version": "0.1", "cases": cases})
    write_json(args.output_dir / "coordinate_domain_audit.json", {"schema_version": "0.1", "input_domain": "RAW_IMAGE_PIXELS", "output_domain": "NORMALIZED_INTEGER_0_1000", "records": coordinate_records})
    write_json(args.output_dir / "bbox_convention_audit.json", bbox_convention)
    write_json(args.output_dir / "prompt_audit.json", {"schema_version": "0.1", "records": prompt_records, "overall": "LIKELY_CONFUSING"})
    write_json(args.output_dir / "qwen_output_audit.json", {"schema_version": "0.1", "records": output_records})
    write_json(args.output_dir / "exhaustive_comparison.json", {
        "schema_version": "0.1", "photo06_exhaustive": "VISIBLE_ENTITY_FOUND_PERSON_WITH_INVALID_0_1_BBOX",
        "photo06_representative": "NO_VISIBLE_ENTITY_BOUNDARY_NOTE",
        "confirmed_input_differences": ["multi-candidate versus single-candidate prompt", "annotated overlay versus original source image", "prompt version/schema wording"],
        "causal_explanation": "UNKNOWN",
    })
    write_json(args.output_dir / "representation_comparison.json", representation_comparison)
    write_json(args.output_dir / "recommended_step22c_design.json", recommendation)

    raw_after = raw_output_snapshot(RAW_ROOTS)
    summary = {
        "schema_version": "0.1", "step_result": "PASS", "audit_complete": True,
        "representative_cases": len(cases), "prompt_classification": "LIKELY_CONFUSING",
        "bbox_convention": "HALF_OPEN", "qwen_inference_runs": 0,
        "qwen_model_loaded": False, "generation_called": False,
        "raw_outputs_before": raw_before, "raw_outputs_after": raw_after,
        "raw_outputs_unchanged": raw_before == raw_after,
        "geometry_modified": False, "planner_modified": False,
        "recommended_method": recommendation["recommended_method"],
    }
    write_json(args.output_dir / "audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
