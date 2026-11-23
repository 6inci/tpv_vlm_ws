#!/usr/bin/env python3
"""Run actual Qwen semantics and metric point grounding on STEP 21 triggers."""

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
    bbox_iou,
    generate_qwen_text,
    load_local_qwen,
    prepare_qwen_inputs,
)
from vlm_reposition.perception.semantic_metric_grounding import (
    SemanticMetricGroundingError,
    annotate_candidates_for_qwen,
    annotate_metric_results,
    load_semantic_metric_config,
    make_candidate_prompt,
    process_frame_response,
    select_triggered_frames,
)


DEFAULT_CONFIG = WORKSPACE / "config" / "offline_semantic_metric_grounding.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "offline_semantic_metric_grounding"


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def slug(frame):
    return f"{frame['scene_observation']['replay_index']:03d}_{Path(frame['scene_observation']['source_image']).stem}"


def compare_runs(first, second):
    second_by_track = {item["scene_observation"]["track_id"]: item for item in second["metric_observations"]}
    comparisons = []
    for item in first["metric_observations"]:
        track_id = item["scene_observation"]["track_id"]
        other = second_by_track.get(track_id)
        bbox_iou_value = None
        if other and item["visual_grounding"] and other["visual_grounding"]:
            bbox_iou_value = bbox_iou(
                item["visual_grounding"]["bbox_normalized_1000"],
                other["visual_grounding"]["bbox_normalized_1000"],
            )
        comparisons.append({
            "track_id": track_id,
            "status_equal": other is not None and item["semantic"]["status"] == other["semantic"]["status"],
            "label_equal": other is not None and item["semantic"]["label"] == other["semantic"]["label"],
            "association_equal": other is not None and item["association"]["status"] == other["association"]["status"],
            "metric_status_equal": other is not None and item["metric_grounding"]["status"] == other["metric_grounding"]["status"],
            "bbox_iou": bbox_iou_value,
            "world_xy_equal": other is not None and item["metric_grounding"].get("world_xy") == other["metric_grounding"].get("world_xy"),
        })
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-id")
    parser.add_argument("--track-id")
    parser.add_argument("--runs", type=int)
    args = parser.parse_args()
    if not args.image_id or not args.track_id:
        parser.error("--image-id and --track-id are required for one representative candidate")
    if args.runs is not None and args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    config = load_semantic_metric_config(args.config)
    qwen_config = load_json(config["model_grounding_config"])
    manifest = load_json(config["input_replay_manifest"])
    frames = select_triggered_frames(manifest, config["scene_observation_directory"])
    frames = [frame for frame in frames if frame["image_id"] == args.image_id]
    if not frames:
        parser.error(f"unknown or untriggered --image-id: {args.image_id}")
    candidates = [item for item in frames[0]["confirmed_candidates"] if item["track_id"] == args.track_id]
    if not candidates:
        parser.error(f"unknown or unconfirmed --track-id for {args.image_id}: {args.track_id}")
    frames[0]["confirmed_candidates"] = candidates
    geometry = LegacyGeometry.load(qwen_config["calibration_path"], qwen_config["legacy_geometry_path"])
    bounds_config = qwen_config["legacy_workspace_bounds_m"]
    bounds = (*bounds_config["x"], *bounds_config["y"])

    processor, model, model_load_sec = load_local_qwen(qwen_config["model_id"])
    import torch
    from transformers import __version__ as transformers_version

    raw_dir = args.output_dir / "qwen_raw_outputs"
    frame_dir = args.output_dir / "frame_results"
    metric_dir = args.output_dir / "metric_observations"
    visualization_dir = args.output_dir / "visualizations"
    repeatability_dir = args.output_dir / "repeatability"
    qwen_input_dir = args.output_dir / "qwen_inputs"
    for directory in (raw_dir, frame_dir, metric_dir, visualization_dir, repeatability_dir, qwen_input_dir):
        directory.mkdir(parents=True, exist_ok=directory == raw_dir)

    all_runs = {}
    total_runs = []
    for frame_index, frame in enumerate(frames):
        run_count = args.runs if args.runs is not None else (
            config["repeatability_runs"] if frame_index < config["repeatability_image_count"] else 1
        )
        image = read_bgr(frame["scene_observation"]["source_image"])
        prompt_image = annotate_candidates_for_qwen(image, frame["confirmed_candidates"])
        overlay_path = qwen_input_dir / f"{slug(frame)}_{args.track_id}_candidate_overlay.png"
        if not cv2.imwrite(str(overlay_path), prompt_image):
            raise RuntimeError(f"could not write candidate overlay: {overlay_path}")
        input_path = Path(frame["scene_observation"]["source_image"])
        prompt = make_candidate_prompt(frame["image_id"], frame["confirmed_candidates"])
        inputs = prepare_qwen_inputs(processor, model, input_path, prompt)
        frame_runs = []
        for run_number in range(1, run_count + 1):
            raw_text, inference_sec = generate_qwen_text(
                processor, model, inputs, config["max_new_tokens"]
            )
            raw_path = raw_dir / f"{slug(frame)}_run{run_number:02d}.txt"
            raw_path.write_text(raw_text + "\n")
            try:
                result = process_frame_response(
                    frame, raw_text, geometry, bounds, config["candidate_association"],
                    (image.shape[1], image.shape[0]),
                )
            except SemanticMetricGroundingError as exc:
                result = {
                    "schema_version": "0.1",
                    "image_id": frame["image_id"],
                    "source_image": frame["scene_observation"]["source_image"],
                    "scene_state": frame["scene_observation"]["scene_state"],
                    "semantic_trigger": True,
                    "frame_semantic_status": "FRAME_SEMANTIC_PARSE_FAILED",
                    "error_code": exc.code,
                    "error": str(exc),
                    "semantic_results": [],
                    "metric_observations": [],
                }
            result.update({
                "run": run_number,
                "prompt_version": config["semantic_prompt_version"],
                "parser_policy": config["parser_policy"],
                "qwen_input_image": str(input_path),
                "candidate_overlay": str(overlay_path),
                "raw_output_path": str(raw_path),
                "raw_output_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "inference_sec": inference_sec,
            })
            write_json(frame_dir / f"{slug(frame)}_run{run_number:02d}.json", result)
            if run_number == 1:
                for candidate_index, observation in enumerate(result["metric_observations"], start=1):
                    write_json(metric_dir / f"{slug(frame)}_{candidate_index:02d}_{observation['scene_observation']['track_id']}.json", observation)
                if any(item["metric_grounding"]["status"] == "AVAILABLE" for item in result["metric_observations"]):
                    visual = annotate_metric_results(image, result)
                    if not cv2.imwrite(str(visualization_dir / f"{slug(frame)}_metric_points.png"), visual):
                        raise RuntimeError("could not write metric result visualization")
            frame_runs.append(result)
            total_runs.append(result)
            print(
                f"{frame['image_id']} run {run_number}: {result['frame_semantic_status']} "
                f"observations={len(result['metric_observations'])} inference={inference_sec:.3f}s",
                flush=True,
            )
        all_runs[frame["image_id"]] = frame_runs

    repeatability = []
    for image_id, runs in all_runs.items():
        if len(runs) < 2:
            continue
        comparison = {
            "image_id": image_id,
            "run_count": len(runs),
            "both_parsed": all(run["frame_semantic_status"] == "PARSED" for run in runs),
            "candidate_comparisons": compare_runs(runs[0], runs[1]) if all(run["frame_semantic_status"] == "PARSED" for run in runs[:2]) else [],
        }
        repeatability.append(comparison)
        write_json(repeatability_dir / f"{image_id}.json", comparison)

    primary = [runs[0] for runs in all_runs.values()]
    observations = [item for run in primary for item in run["metric_observations"]]
    semantic_counts = {
        status: sum(item["semantic"]["status"] == status for item in observations)
        for status in ("VISIBLE_ENTITY_FOUND", "NO_VISIBLE_ENTITY", "AMBIGUOUS")
    }
    association_counts = {
        status: sum(item["association"]["status"] == status for item in observations)
        for status in ("ASSOCIATED", "WEAK_ASSOCIATION", "NOT_ASSOCIATED")
    }
    metric_available = sum(item["metric_grounding"]["status"] == "AVAILABLE" for item in observations)
    parser_failures = sum(run["frame_semantic_status"] != "PARSED" for run in total_runs)
    summary = {
        "schema_version": "0.1",
        "step_result": "PASS" if parser_failures == 0 and metric_available > 0 else "PARTIAL",
        "software": "PASS",
        "semantic_metric_pipeline": "PASS" if parser_failures == 0 and metric_available > 0 else "PARTIAL",
        "input_group_id": manifest["group_id"],
        "trigger_policy": config["trigger_policy"],
        "ordering_policy": manifest["ordering_policy"],
        "triggered_frames": len(frames),
        "triggered_candidates": sum(len(frame["confirmed_candidates"]) for frame in frames),
        **semantic_counts,
        **association_counts,
        "METRIC_GROUNDING_AVAILABLE": metric_available,
        "METRIC_GROUNDING_NOT_ALLOWED": sum(item["metric_grounding"]["status"] == "METRIC_GROUNDING_NOT_ALLOWED" for item in observations),
        "INSIDE": sum(item["metric_grounding"].get("workspace_status") == "INSIDE" for item in observations),
        "OUTSIDE": sum(item["metric_grounding"].get("workspace_status") == "OUTSIDE" for item in observations),
        "parser_failures": parser_failures,
        "repeatability": repeatability,
        "qwen_runtime": {
            "model": qwen_config["model_id"],
            "model_cache": qwen_config["model_cache_path"],
            "local_files_only": True,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(model.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model_load_sec": model_load_sec,
            "inference_runs": len(total_runs),
            "inference_sec": [run["inference_sec"] for run in total_runs],
        },
        "geometry": {
            "coordinate_frame": geometry.coordinate_system,
            "unit": "m",
            "pipeline": "RAW_BOTTOM_CENTER_PX_TO_STORED_UNDISTORTION_TO_LEGACY_HOMOGRAPHY",
            "final_paper_accuracy_claim": False,
        },
        "physical_geometry": "NOT_AVAILABLE",
        "planning_role": "NOT_ASSIGNED",
        "planner_injection": "NOT_RUN",
        "a_star": "NOT_RUN",
        "replanning": "NOT_RUN",
    }
    write_json(args.output_dir / "semantic_metric_grounding_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
