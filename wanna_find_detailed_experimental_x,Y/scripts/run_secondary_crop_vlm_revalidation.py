#!/usr/bin/env python3
"""Run the single STEP 22D Method D two-image Qwen revalidation case."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.geometry import LegacyGeometry
from vlm_reposition.perception.image_geometry_audit import read_bgr
from vlm_reposition.perception.qwen_grounding import (
    generate_qwen_text,
    load_local_qwen,
    prepare_qwen_multi_image_inputs,
)
from vlm_reposition.perception.semantic_metric_grounding import (
    SemanticMetricGroundingError,
    annotate_metric_results,
    load_semantic_metric_config,
    make_method_d_candidate_prompt,
    process_frame_response,
    select_triggered_frames,
)


IMAGE_ID = "image_014_photo_08"
TRACK_ID = "track_039"
RAW_CANDIDATE_BBOX = [1017, 545, 1375, 684]
NORMALIZED_CANDIDATE_BBOX = [504, 481, 682, 604]
IMAGE_SIZE = (2016, 1133)
CROP_BBOX = [963, 524, 1429, 705]
CROP_SIZE = (466, 181)
METHOD = "D_FULL_IMAGE_PLUS_SECONDARY_CONTEXT_CROP"
CONFIG = WORKSPACE / "config" / "offline_semantic_metric_grounding.json"
AUDIT_CASES = WORKSPACE / "results" / "candidate_conditioning_audit" / "representative_candidates.json"
STEP22C_RESULT = WORKSPACE / "results" / "candidate_conditioned_vlm_revalidation"
CROP = WORKSPACE / "results" / "candidate_conditioning_audit" / "candidate_crops" / "photo08_track039_context.png"
DEFAULT_OUTPUT = WORKSPACE / "results" / "candidate_conditioned_vlm_revalidation_method_d"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def snapshot_protected(excluded):
    later_results = (
        WORKSPACE / "results" / "physical_obstacle_geometry",
        WORKSPACE / "results" / "physical_measurement_registration",
        WORKSPACE / "results" / "navigation_obstacle_adapter",
        WORKSPACE / "results" / "navigation_dynamic_world_integration",
        WORKSPACE / "results" / "navigation_path_validity",
        WORKSPACE / "results" / "controlled_navigation_path_effect",
        WORKSPACE / "results" / "path_block_persistence_integration",
        WORKSPACE / "results" / "persistent_block_replanning_integration",
        WORKSPACE / "results" / "offline_navigation_chain_integration",
        WORKSPACE / "results" / "observed_blocking_acquisition",
    )
    files = sorted(
        path for path in (WORKSPACE / "results").rglob("*")
        if path.is_file() and excluded not in path.parents
        and all(later not in path.parents for later in later_results)
    )
    files += [
        WORKSPACE / "src" / "vlm_reposition" / "geometry" / "legacy.py",
        Path("/home/binci/qwen_vision_test/homography_12pt_undistorted.json"),
        Path("/home/binci/camera_calibration_1920x1080/calibration_result.json"),
    ]
    records = [(str(path), sha256_file(path)) for path in files]
    digest = hashlib.sha256("".join(f"{digest}  {path}\n" for path, digest in records).encode()).hexdigest()
    return {"file_count": len(records), "digest_sha256": digest}


def load_validated_case(config):
    cases = json.loads(AUDIT_CASES.read_text())["cases"]
    matches = [case for case in cases if case["image_id"] == IMAGE_ID and case["track_id"] == TRACK_ID]
    if len(matches) != 1:
        raise ValueError("STEP 22B representative identity mismatch")
    case = matches[0]
    crop = case["candidate_crop"]
    if (
        case["candidate_bbox_px"] != RAW_CANDIDATE_BBOX
        or case["candidate_bbox_normalized_1000"] != NORMALIZED_CANDIDATE_BBOX
        or Path(crop["path"]) != CROP
        or crop["source_bbox_px"] != CROP_BBOX
        or crop["context_margin_ratio"] != 0.15
        or crop["coordinate_role"] != "SEMANTIC_CONDITIONING_ONLY"
        or crop["metric_coordinate_source"] is not False
    ):
        raise ValueError("STEP 22B crop provenance mismatch")

    step22c = json.loads((STEP22C_RESULT / "step22c_summary.json").read_text())
    if step22c["qwen_inference_runs"] != 1 or step22c["step_result"] != "PARTIAL":
        raise ValueError("STEP 22C preserved result mismatch")

    manifest = json.loads(Path(config["input_replay_manifest"]).read_text())
    frames = [frame for frame in select_triggered_frames(manifest, config["scene_observation_directory"]) if frame["image_id"] == IMAGE_ID]
    if len(frames) != 1:
        raise ValueError("representative frame identity mismatch")
    candidates = [candidate for candidate in frames[0]["confirmed_candidates"] if candidate["track_id"] == TRACK_ID]
    if len(candidates) != 1 or candidates[0]["bbox_px"] != RAW_CANDIDATE_BBOX:
        raise ValueError("representative candidate identity mismatch")
    frames[0]["confirmed_candidates"] = candidates
    return case, frames[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    output = DEFAULT_OUTPUT
    resume = args.finalize_existing
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    if resume and (
        not (output / "qwen_raw_output.txt").is_file()
        or (output / "step22d_summary.json").exists()
    ):
        raise RuntimeError("finalize-existing requires one incomplete run with preserved raw output and no summary")

    protected_before = snapshot_protected(output)
    config = load_semantic_metric_config(CONFIG)
    qwen_config = json.loads(Path(config["model_grounding_config"]).read_text())
    case, frame = load_validated_case(config)
    source = Path(case["source_image"])
    source_hash_before, crop_hash_before = sha256_file(source), sha256_file(CROP)
    source_image, crop_image = read_bgr(source), read_bgr(CROP)
    if (source_image.shape[1], source_image.shape[0]) != IMAGE_SIZE:
        raise ValueError("IMAGE 1 must remain the original 2016x1133 full scene")
    if (crop_image.shape[1], crop_image.shape[0]) != CROP_SIZE:
        raise ValueError("IMAGE 2 crop dimensions do not match STEP 22B provenance")

    prompt = make_method_d_candidate_prompt(IMAGE_ID, TRACK_ID, NORMALIZED_CANDIDATE_BBOX)
    if str(RAW_CANDIDATE_BBOX[0]) in prompt or "candidate_bbox_px" in prompt:
        raise ValueError("raw candidate pixels leaked into the semantic prompt")

    visual_dir = output / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=resume)
    if not resume:
        (output / "method_d_prompt.txt").write_text(prompt + "\n")
        write_json(output / "method_d_input_metadata.json", {
            "schema_version": "0.1", "method": METHOD, "image_id": IMAGE_ID, "track_id": TRACK_ID,
            "model_input_order": ["IMAGE_1_FULL_SCENE", "IMAGE_2_CANDIDATE_DETAIL"],
            "candidate_region_in_image_1_normalized_1000": NORMALIZED_CANDIDATE_BBOX,
            "candidate_bbox_raw_px_provenance_only": RAW_CANDIDATE_BBOX,
            "output_bbox_required_reference": "IMAGE_1",
            "retry": "NOT_ALLOWED", "fallback_candidate": "NOT_ALLOWED",
        })
        write_json(output / "full_scene_reference.json", {
            "source_image_id": IMAGE_ID, "source_image_path": str(source), "sha256": source_hash_before,
            "width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1], "role": "ORIGINAL_FULL_SCENE_AND_METRIC_COORDINATE_DOMAIN",
            "resize": False, "crop": False, "warp": False, "undistort": False, "overlay": False,
        })
        write_json(output / "crop_provenance.json", {
            "source_image_id": IMAGE_ID, "source_image_path": str(source), "track_id": TRACK_ID,
            "source_candidate_bbox_px": RAW_CANDIDATE_BBOX, "crop_bbox_in_original_px": CROP_BBOX,
            "context_margin_policy": {"ratio": 0.15, "status": "DEVELOPMENT_PARAMETER"},
            "crop_path": str(CROP), "crop_sha256": crop_hash_before,
            "crop_width": CROP_SIZE[0], "crop_height": CROP_SIZE[1],
            "semantic_conditioning_only": True, "metric_coordinate_source": False,
            "stretch": False, "upscale": False, "neural_enhancement": False,
        })

    geometry = LegacyGeometry.load(qwen_config["calibration_path"], qwen_config["legacy_geometry_path"])
    bounds_config = qwen_config["legacy_workspace_bounds_m"]
    bounds = (*bounds_config["x"], *bounds_config["y"])
    raw_path = output / "qwen_raw_output.txt"
    inference_runs = 1 if resume else 0
    if resume:
        if sha256_file(raw_path) != "2d7069060d07a131eae60eb30882db48323db47807f34680227f0515b9fddf70":
            raise RuntimeError("preserved raw output hash mismatch")
        raw_text = raw_path.read_text()
        model_load_sec = inference_sec = None
        runtime_device = "cpu"
    else:
        try:
            processor, model, model_load_sec = load_local_qwen(qwen_config["model_id"])
            inputs = prepare_qwen_multi_image_inputs(processor, model, [source, CROP], prompt)
            if inference_runs != 0:
                raise RuntimeError("one-inference guard rejected a repeated generation")
            inference_runs += 1
            raw_text, inference_sec = generate_qwen_text(processor, model, inputs, config["max_new_tokens"])
            runtime_device = str(model.device)
        except Exception as exc:
            summary = {
                "schema_version": "0.1", "step_result": "FAIL", "reason": "QWEN_RUNTIME_FAILURE",
                "error": str(exc), "qwen_inference_runs": inference_runs, "fallback_candidate_runs": 0,
                "physical_geometry": "NOT_AVAILABLE", "planning_role": "NOT_ASSIGNED", "planner_injection": "NOT_RUN",
            }
            write_json(output / "step22d_summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            return 1
        raw_path.write_text(raw_text + "\n")
    try:
        result = process_frame_response(
            frame, raw_text, geometry, bounds, config["candidate_association"], IMAGE_SIZE,
            require_candidate_bbox_echo=False, required_bbox_reference_image="IMAGE_1",
        )
    except SemanticMetricGroundingError as exc:
        parsed = {"status": "REJECTED", "error_code": exc.code, "error": str(exc)}
        bbox_reference = {
            "status": "BBOX_REFERENCE_AMBIGUOUS" if exc.code == "BBOX_REFERENCE_AMBIGUOUS" else "NOT_EVALUATED",
            "required_reference": "IMAGE_1", "crop_bbox_remapped": False,
        }
        association = {"status": "NOT_EVALUATED"}
        metric = {"status": "NOT_RUN", "world_xy": None}
        semantic_status, observation = "PARSE_REJECTED", None
    else:
        semantic = result["semantic_results"][0]
        observation = result["metric_observations"][0]
        parsed = {"status": "ACCEPTED", "semantic_result": semantic}
        bbox_reference = {
            "status": "VALIDATED_IMAGE_1" if semantic["status"] == "VISIBLE_ENTITY_FOUND" else "NOT_EVALUATED",
            "required_reference": "IMAGE_1", "declared_reference": semantic.get("bbox_reference_image"),
            "crop_bbox_remapped": False, "full_image_dimensions": list(IMAGE_SIZE),
        }
        association = {
            **observation["association"], "candidate_bbox_raw_px": RAW_CANDIDATE_BBOX,
            "object_bbox_raw_px": None if observation["visual_grounding"] is None else observation["visual_grounding"]["bbox_raw_px"],
            "thresholds": config["candidate_association"],
        }
        metric = {
            **observation["metric_grounding"],
            "ground_reference": None if observation["visual_grounding"] is None else {
                "policy": "BOTTOM_CENTER", "meaning": "DEVELOPMENT_GROUND_REFERENCE",
                "raw_px": observation["visual_grounding"]["ground_reference_px"],
                "undistorted_px": observation["visual_grounding"].get("undistorted_ground_reference_px"),
            },
            "geometry_provenance": observation["source_image"]["geometry_provenance"],
            "coordinate_frame": observation["metric_grounding"]["coordinate_frame"],
            "crop_used_as_metric_source": False,
        }
        semantic_status = semantic["status"]

    write_json(output / "parsed_result.json", parsed)
    write_json(output / "bbox_reference_validation.json", bbox_reference)
    write_json(output / "association_result.json", association)
    write_json(output / "metric_result.json", metric)

    visualization_path = None
    if observation is not None and metric["status"] == "AVAILABLE":
        visual = annotate_metric_results(source_image, result)
        cv2.putText(visual, "Geometry: LEGACY_WORKSPACE_COORDINATE / development inference", (20, visual.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        visualization_path = visual_dir / "photo08_track039_metric_result.png"
        if not cv2.imwrite(str(visualization_path), visual):
            raise RuntimeError("could not write metric visualization")

    import torch
    from transformers import __version__ as transformers_version

    protected_after = snapshot_protected(output)
    source_unchanged = sha256_file(source) == source_hash_before
    crop_unchanged = sha256_file(CROP) == crop_hash_before
    passed = metric["status"] == "AVAILABLE" and association["status"] == "ASSOCIATED"
    summary = {
        "schema_version": "0.1", "step_result": "PASS" if passed else "PARTIAL",
        "step22_final_status": "PASS" if passed else "PARTIAL", "method": METHOD,
        "image_id": IMAGE_ID, "track_id": TRACK_ID, "semantic_status": semantic_status,
        "bbox_reference_status": bbox_reference["status"], "association_status": association["status"],
        "metric_status": metric["status"], "world_xy": metric.get("world_xy"),
        "qwen_raw_output": str(raw_path), "qwen_raw_output_sha256": sha256_file(raw_path),
        "visualization": None if visualization_path is None else str(visualization_path),
        "qwen_runtime": {
            "model": qwen_config["model_id"], "model_cache": qwen_config["model_cache_path"],
            "local_files_only": True, "do_sample": False, "device": runtime_device,
            "python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers_version,
            "cuda_available": torch.cuda.is_available(), "model_load_sec": model_load_sec, "inference_sec": inference_sec,
            "timing_status": "NOT_RECOVERED_AFTER_DETERMINISTIC_POSTPROCESS_RESUME" if resume else "MEASURED",
        },
        "qwen_inference_runs": inference_runs, "fallback_candidate_runs": 0,
        "postprocess_resumed_from_preserved_raw_output": resume,
        "generation_called_during_postprocess_resume": False if resume else True,
        "source_original_unmodified": source_unchanged, "crop_artifact_unmodified": crop_unchanged,
        "protected_artifacts_before": protected_before, "protected_artifacts_after": protected_after,
        "protected_artifacts_unchanged": protected_before == protected_after,
        "geometry_implementation_modified": False, "physical_geometry": "NOT_AVAILABLE",
        "planning_role": "NOT_ASSIGNED", "planner_injection": "NOT_RUN",
    }
    if inference_runs != 1 or not source_unchanged or not crop_unchanged or protected_before != protected_after:
        raise RuntimeError("STEP 22D execution guard failed")
    write_json(output / "step22d_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
