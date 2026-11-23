#!/usr/bin/env python3
"""Run task-conditioned grounding for every partial-reference target."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.geometry import LegacyGeometry
from vlm_reposition.perception import (
    bbox_iou,
    generate_qwen_text,
    ground_target_result,
    load_local_qwen,
    load_partial_semantic_reference,
    normalize_semantic_label,
    prepare_qwen_inputs,
)


DEFAULT_CONFIG = WORKSPACE / "config" / "qwen_grounding.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "task_conditioned_grounding"


def load_json(path):
    return json.loads(Path(path).read_text())


def make_prompt(target_query):
    return f'''Locate the object described by the target query in this image.

Target query:
"{target_query}"

Return JSON only in this format:
{{"candidates":[{{"label":"specific visible object name","bbox_2d":[x1,y1,x2,y2]}}]}}

bbox_2d coordinates must be normalized from 0 to 1000.
If the target is absent, return {{"candidates":[]}}.
If multiple visually plausible instances match, return every candidate instead of choosing the first.
Do not estimate metric coordinates, describe the scene, generate robot actions, or explain your answer.'''


def file_slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")


def summarize_target(reference_entity, runs):
    found = [run for run in runs if run["target_status"] == "TARGET_FOUND"]
    labels = [run["detected_label"] for run in found]
    bbox_consistency = None
    world_variation = None
    if len(found) == len(runs):
        bbox_consistency = [
            bbox_iou(found[0]["bbox_normalized_0_1000"], run["bbox_normalized_0_1000"])
            for run in found
        ]
        if all(run["world_coordinate"] is not None for run in found):
            coordinates = np.asarray([
                [run["world_coordinate"]["x"], run["world_coordinate"]["y"]]
                for run in found
            ])
            world_variation = (coordinates.max(axis=0) - coordinates.min(axis=0)).tolist()
    return {
        "reference_id": reference_entity["reference_id"],
        "canonical_target": reference_entity["canonical_label"],
        "run_count": len(runs),
        "detection_count": len(found),
        "target_statuses": [run["target_status"] for run in runs],
        "detected_labels": labels,
        "label_consistent": (
            len(labels) == len(runs)
            and len({normalize_semantic_label(label) for label in labels}) == 1
        ),
        "bbox_iou_vs_run01": bbox_consistency,
        "reference_bbox_iou": [run.get("reference_bbox_iou") for run in runs],
        "position_statuses": [run["position_status"] for run in runs],
        "semantic_statuses": [run["semantic_status"] for run in runs],
        "world_xy_peak_to_peak_m": world_variation,
        "active_workspace_statuses": [run.get("active_workspace_status") for run in runs],
    }


def draw_targets(image, representatives):
    visual = image.copy()
    colors = [(0, 255, 0), (255, 200, 0), (255, 0, 255)]
    for color, result in zip(colors, representatives):
        if result["target_status"] != "TARGET_FOUND":
            continue
        x1, y1, x2, y2 = result["bbox_pixel_raw"]
        raw = result["raw_reference_pixel"]
        world = result["world_coordinate"]
        cv2.rectangle(visual, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
        cv2.circle(visual, (int(raw["u"]), int(raw["v"])), 8, (0, 0, 255), -1)
        coordinate = "X=NA Y=NA" if world is None else f"X={world['x']:.2f} Y={world['y']:.2f}"
        lines = [
            f"query={result['query']} label={result['detected_label']}",
            f"{coordinate} {result['position_status']}",
        ]
        text_y = max(20, int(y1) - 26)
        for line_number, text in enumerate(lines):
            cv2.putText(
                visual,
                text,
                (int(x1), text_y + line_number * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return visual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    config = load_json(args.config)
    image_path = args.image or Path(config["image_path"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {image_path}")
    image_size = (image.shape[1], image.shape[0])
    geometry = LegacyGeometry.load(config["calibration_path"], config["legacy_geometry_path"])
    geometry.validate_image(image)
    reference = load_partial_semantic_reference(config["semantic_reference_path"])
    bounds = config["legacy_workspace_bounds_m"]
    workspace_bounds = (*bounds["x"], *bounds["y"])
    iou_threshold = float(config["repeatability_iou_match_threshold"])
    broad_scene = load_json(config["semantic_scene_result_path"])
    broad_by_reference = {
        item["reference_id"]: item for item in broad_scene["reference_comparison"]
    }

    processor, model, model_load_sec = load_local_qwen(config["model_id"])
    all_runs = {}
    for reference_entity in reference["entities"]:
        query = reference_entity["canonical_label"]
        inputs = prepare_qwen_inputs(processor, model, image_path, make_prompt(query))
        target_runs = []
        for run_number in range(1, args.runs + 1):
            raw_output, inference_sec = generate_qwen_text(
                processor, model, inputs, max_new_tokens=240
            )
            result = ground_target_result(
                query,
                raw_output,
                image_size,
                geometry,
                workspace_bounds,
                reference_entity,
                iou_threshold,
            )
            result.update({
                "run": run_number,
                "raw_model_output": raw_output,
                "inference_sec": inference_sec,
            })
            target_runs.append(result)
            print(
                f"{query} run {run_number}: status={result['target_status']} "
                f"label={result.get('detected_label')} "
                f"iou={result.get('reference_bbox_iou')} "
                f"inference={inference_sec:.3f} s"
            )
        all_runs[reference_entity["reference_id"]] = target_runs

    target_summaries = [
        summarize_target(entity, all_runs[entity["reference_id"]])
        for entity in reference["entities"]
    ]
    passed = all(
        run["target_status"] == "TARGET_FOUND"
        and run["position_status"] == "REFERENCE_POSITION_MATCH"
        for target_runs in all_runs.values()
        for run in target_runs
    )
    alias_candidates = sorted({
        (run["query"], run["detected_label"])
        for target_runs in all_runs.values()
        for run in target_runs
        if run["semantic_status"] == "HUMAN_ALIAS_CANDIDATE"
    })
    broad_comparison = [
        {
            "reference_id": entity["reference_id"],
            "canonical_target": entity["canonical_label"],
            "broad_scene_detected": broad_by_reference[entity["reference_id"]]["matched_entity_id"] is not None,
            "broad_scene_result": broad_by_reference[entity["reference_id"]]["result"],
            "task_conditioned_detection_count": target_summaries[index]["detection_count"],
            "task_conditioned_all_position_matches": all(
                status == "REFERENCE_POSITION_MATCH"
                for status in target_summaries[index]["position_statuses"]
            ),
        }
        for index, entity in enumerate(reference["entities"])
    ]

    import torch
    from transformers import __version__ as transformers_version

    summary = {
        "result": "PASS" if passed else "PARTIAL",
        "verification_type": "Task-Conditioned Target Grounding Verification",
        "not_final_paper_accuracy": True,
        "created_at": datetime.now().astimezone().isoformat(),
        "image": str(image_path),
        "model": config["model_id"],
        "model_cache": config["model_cache_path"],
        "local_files_only": True,
        "model_load_sec": model_load_sec,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "reference_status": reference["reference_status"],
        "association_rule": f"reference bbox IoU >= {iou_threshold}; not an accuracy threshold",
        "target_summaries": target_summaries,
        "broad_vs_task_conditioned": broad_comparison,
        "human_alias_candidates": [
            {"query": query, "detected_label": label, "automatically_added": False}
            for query, label in alias_candidates
        ],
    }

    representatives = [all_runs[entity["reference_id"]][0] for entity in reference["entities"]]
    visual = draw_targets(image, representatives)
    args.output_dir.mkdir(parents=True)
    for reference_entity in reference["entities"]:
        stem = file_slug(reference_entity["canonical_label"])
        for run in all_runs[reference_entity["reference_id"]]:
            (args.output_dir / f"{stem}_run{run['run']:02d}.json").write_text(
                json.dumps(run, ensure_ascii=False, indent=2) + "\n"
            )
    (args.output_dir / "target_grounding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    visualization_path = args.output_dir / "photo_04_task_targets.png"
    if not cv2.imwrite(str(visualization_path), visual):
        raise RuntimeError(f"could not write visualization: {visualization_path}")

    print(f"model load: {model_load_sec:.3f} s")
    print(f"human alias candidates: {alias_candidates}")
    print(f"results: {args.output_dir}")
    print(summary["result"])


if __name__ == "__main__":
    main()
