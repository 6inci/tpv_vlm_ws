#!/usr/bin/env python3
"""Ground a Qwen multi-entity scene into the verified legacy workspace."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.geometry import LegacyGeometry
from vlm_reposition.perception import (
    bbox_iou,
    generate_qwen_text,
    ground_scene_entities,
    load_local_qwen,
    parse_scene_entities,
    prepare_qwen_inputs,
)


DEFAULT_CONFIG = WORKSPACE / "config" / "qwen_grounding.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "multi_entity_grounding"


def load_json(path):
    return json.loads(Path(path).read_text())


def make_prompt():
    return '''Inspect this workspace image and identify distinct physically relevant entities that may matter for mobile-robot task execution.

For each entity return:
- label
- semantic_type
- navigation_role
- bbox_2d

Allowed semantic_type values only:
movable_object, static_obstacle, person, furniture, workspace_element, unknown

Allowed navigation_role values only:
target_candidate, obstacle_candidate, context_only, unknown

Use bbox_2d coordinates normalized to the range 0 to 1000.
Return each distinct entity once.
Return JSON only in exactly this structure:
{"entities":[{"label":"name","semantic_type":"allowed_value","navigation_role":"allowed_value","bbox_2d":[x1,y1,x2,y2]}]}'''


def normalize_label(label):
    return " ".join(label.casefold().split())


def repeatability_summary(runs, iou_threshold):
    baseline = runs[0]["entities"]
    matches = [[entity] + [None] * (len(runs) - 1) for entity in baseline]
    unmatched_by_run = {"run01": []}

    for run_index, run in enumerate(runs[1:], start=1):
        used = set()
        for baseline_index, reference in enumerate(baseline):
            candidates = []
            for candidate_index, candidate in enumerate(run["entities"]):
                if candidate_index in used or normalize_label(candidate["label"]) != normalize_label(reference["label"]):
                    continue
                candidates.append((
                    bbox_iou(
                        reference["bbox_normalized_0_1000"],
                        candidate["bbox_normalized_0_1000"],
                    ),
                    candidate_index,
                    candidate,
                ))
            if candidates:
                overlap, candidate_index, candidate = max(candidates, key=lambda item: item[0])
                if overlap >= iou_threshold:
                    matches[baseline_index][run_index] = candidate
                    used.add(candidate_index)
        unmatched_by_run[f"run{run_index + 1:02d}"] = [
            entity["label"]
            for index, entity in enumerate(run["entities"])
            if index not in used
        ]

    common = []
    for group in matches:
        if any(entity is None for entity in group):
            continue
        worlds = [entity["world_coordinate"] for entity in group]
        world_variation = None
        if all(world is not None for world in worlds):
            coordinates = np.asarray([[world["x"], world["y"]] for world in worlds])
            world_variation = (coordinates.max(axis=0) - coordinates.min(axis=0)).tolist()
        common.append({
            "normalized_label": normalize_label(group[0]["label"]),
            "entity_ids_by_run": [entity["id"] for entity in group],
            "labels_by_run": [entity["label"] for entity in group],
            "semantic_types_by_run": [entity["semantic_type"] for entity in group],
            "navigation_roles_by_run": [entity["navigation_role"] for entity in group],
            "classification_consistent": (
                len({entity["semantic_type"] for entity in group}) == 1
                and len({entity["navigation_role"] for entity in group}) == 1
            ),
            "bbox_iou_vs_run01": [
                bbox_iou(
                    group[0]["bbox_normalized_0_1000"],
                    entity["bbox_normalized_0_1000"],
                )
                for entity in group
            ],
            "world_xy_peak_to_peak_m": world_variation,
        })

    missing_by_run = {}
    for run_index in range(1, len(runs)):
        missing_by_run[f"run{run_index + 1:02d}"] = [
            group[0]["label"] for group in matches if group[run_index] is None
        ]
    return {
        "matching_rule": "normalized label + maximum bbox IoU",
        "iou_match_threshold": iou_threshold,
        "entity_counts": [len(run["entities"]) for run in runs],
        "common_entity_count": len(common),
        "common_entities": common,
        "unmatched_entities_by_run": unmatched_by_run,
        "baseline_entities_missing_by_run": missing_by_run,
    }


def legacy_comparison(path, fresh_entities):
    legacy = load_json(path)
    comparisons = []
    for legacy_entity in legacy.get("objects", []):
        overlap, fresh = max(
            [(
                bbox_iou(legacy_entity["bbox_2d"], entity["bbox_normalized_0_1000"]),
                entity,
            )
            for entity in fresh_entities
            ],
            key=lambda item: item[0],
        )
        comparisons.append({
            "legacy_label": legacy_entity["label"],
            "legacy_bbox_normalized_0_1000": legacy_entity["bbox_2d"],
            "best_fresh_label_by_bbox_overlap": fresh["label"],
            "fresh_bbox_normalized_0_1000": fresh["bbox_normalized_0_1000"],
            "bbox_iou": overlap,
        })
    return {
        "source": str(path),
        "legacy_instruction": legacy.get("instruction"),
        "legacy_behavior": "The legacy prompt requested three named objects, not an open scene inventory.",
        "legacy_labels": [entity["label"] for entity in legacy.get("objects", [])],
        "fresh_labels": [entity["label"] for entity in fresh_entities],
        "spatial_overlap_comparison": comparisons,
        "not_accuracy_evaluation": True,
    }


def draw_scene(image, entities):
    visual = image.copy()
    colors = {
        "target_candidate": (0, 255, 0),
        "obstacle_candidate": (0, 165, 255),
        "context_only": (255, 200, 0),
        "unknown": (180, 180, 180),
    }
    for entity in entities:
        x1, y1, x2, y2 = entity["bbox_pixel_raw"]
        raw = entity["raw_reference_pixel"]
        color = colors[entity["navigation_role"]]
        cv2.rectangle(visual, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.circle(visual, (int(raw["u"]), int(raw["v"])), 6, (0, 0, 255), -1)
        world = entity["world_coordinate"]
        coordinate = "X=NA Y=NA" if world is None else f"X={world['x']:.2f} Y={world['y']:.2f}"
        lines = [
            f"{entity['id']} {entity['label']}",
            f"{entity['semantic_type']}/{entity['navigation_role']} {coordinate}",
        ]
        text_y = max(18, int(y1) - 24)
        for line_number, text in enumerate(lines):
            cv2.putText(
                visual,
                text,
                (int(x1), text_y + line_number * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
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
    bounds = config["legacy_workspace_bounds_m"]
    workspace_bounds = (*bounds["x"], *bounds["y"])

    processor, model, model_load_sec = load_local_qwen(config["model_id"])
    inputs = prepare_qwen_inputs(processor, model, image_path, make_prompt())
    runs = []
    for run_number in range(1, args.runs + 1):
        raw_output, inference_sec = generate_qwen_text(
            processor, model, inputs, max_new_tokens=1800
        )
        detections = parse_scene_entities(raw_output)
        entities = ground_scene_entities(detections, image_size, geometry, workspace_bounds)
        runs.append({
            "run": run_number,
            "image": str(image_path),
            "coordinate_frame": geometry.coordinate_system,
            "raw_model_output": raw_output,
            "inference_sec": inference_sec,
            "entities": entities,
        })
        print(
            f"run {run_number}: entities={len(entities)} "
            f"valid={sum(entity['geometry_status'] == 'VALID' for entity in entities)} "
            f"outside={sum(entity['geometry_status'] == 'OUTSIDE_LEGACY_WORKSPACE' for entity in entities)} "
            f"invalid={sum(entity['geometry_status'] == 'INVALID' for entity in entities)} "
            f"inference={inference_sec:.3f} s"
        )

    repeatability = repeatability_summary(
        runs, float(config["repeatability_iou_match_threshold"])
    )
    invalid_count = sum(
        entity["geometry_status"] == "INVALID"
        for run in runs
        for entity in run["entities"]
    )
    outside_count = sum(
        entity["geometry_status"] == "OUTSIDE_LEGACY_WORKSPACE"
        for run in runs
        for entity in run["entities"]
    )
    comparison = legacy_comparison(config["stored_legacy_result_path"], runs[0]["entities"])

    import torch
    from transformers import __version__ as transformers_version

    summary = {
        "result": "PARTIAL" if invalid_count or outside_count else "PASS",
        "verification_type": "Multi-Entity Semantic + Metric Scene Grounding Integration",
        "not_final_paper_accuracy": True,
        "created_at": datetime.now().astimezone().isoformat(),
        "image": str(image_path),
        "coordinate_frame": geometry.coordinate_system,
        "model": config["model_id"],
        "model_cache": config["model_cache_path"],
        "local_files_only": True,
        "model_load_sec": model_load_sec,
        "inference_sec": [run["inference_sec"] for run in runs],
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "semantic_type_policy": "Strict validation; unsupported values are rejected.",
        "navigation_role_policy": "Strict validation; unsupported values are rejected.",
        "geometry_status_policy": {
            "VALID": "finite X/Y within configured legacy workspace bounds",
            "OUTSIDE_LEGACY_WORKSPACE": "finite X/Y outside configured legacy workspace bounds; not clamped",
            "INVALID": "reference pixel could not be transformed",
        },
        "workspace_bounds_m": bounds,
        "geometry_status_counts_by_run": [
            {
                status: sum(entity["geometry_status"] == status for entity in run["entities"])
                for status in ("VALID", "OUTSIDE_LEGACY_WORKSPACE", "INVALID")
            }
            for run in runs
        ],
        "repeatability": repeatability,
        "legacy_comparison": comparison,
        "limitations": [
            "BBox bottom-center is only an approximate floor reference point.",
            "Obstacle physical footprints are not estimated.",
            "No occupancy map is generated.",
            "No semantic or position ground truth is available.",
        ],
    }

    visual = draw_scene(image, runs[0]["entities"])
    args.output_dir.mkdir(parents=True)
    for run in runs:
        (args.output_dir / f"scene_run{run['run']:02d}.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2) + "\n"
        )
    (args.output_dir / "scene_repeatability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    visualization_path = args.output_dir / "photo_04_scene_grounding.png"
    if not cv2.imwrite(str(visualization_path), visual):
        raise RuntimeError(f"could not write visualization: {visualization_path}")

    print(f"model load: {model_load_sec:.3f} s")
    print(f"entity counts: {repeatability['entity_counts']}")
    print(f"common entities: {repeatability['common_entity_count']}")
    print(f"results: {args.output_dir}")
    print(summary["result"])


if __name__ == "__main__":
    main()
