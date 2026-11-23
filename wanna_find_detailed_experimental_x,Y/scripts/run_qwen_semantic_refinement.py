#!/usr/bin/env python3
"""Verify specific semantic labels against the partial photo_04 reference."""

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
    apply_semantic_reference,
    bbox_iou,
    generate_qwen_text,
    ground_scene_entities,
    load_local_qwen,
    load_partial_semantic_reference,
    normalize_semantic_label,
    parse_semantic_entities,
    prepare_qwen_inputs,
)


DEFAULT_CONFIG = WORKSPACE / "config" / "qwen_grounding.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "semantic_refinement"


def load_json(path):
    return json.loads(Path(path).read_text())


def make_prompt():
    return '''Inspect this workspace image and identify distinct physical entities relevant to a mobile-robot workspace.

For each entity return exactly these fields:
1. label: Use a specific visible object name, such as "blue storage box", "office chair", or "person". Do not use ontology category names such as "workspace_element", "movable_object", "static_obstacle", "furniture", or "unknown" as the label when a more specific visible name can reasonably be determined. The label "person" is allowed for a visible person.
2. semantic_type: Choose exactly one of movable_object, static_obstacle, person, furniture, workspace_element, unknown.
3. bbox_2d: Use coordinates normalized to the range 0 to 1000.

Do not output navigation roles, target choices, collision roles, task decisions, or explanations.
Return each distinct entity once.
Return JSON only in exactly this structure:
{"entities":[{"label":"specific visible name","semantic_type":"allowed_value","bbox_2d":[x1,y1,x2,y2]}]}'''


def reference_repeatability(runs, reference):
    summaries = []
    for reference_entity in reference["entities"]:
        comparisons = [
            next(
                comparison
                for comparison in run["reference_comparison"]
                if comparison["reference_id"] == reference_entity["reference_id"]
            )
            for run in runs
        ]
        matched_entities = []
        for run, comparison in zip(runs, comparisons):
            matched_entities.append(next(
                (
                    entity for entity in run["entities"]
                    if entity["id"] == comparison["matched_entity_id"]
                ),
                None,
            ))

        bbox_consistency = None
        world_variation = None
        if all(entity is not None for entity in matched_entities):
            bbox_consistency = [
                bbox_iou(
                    matched_entities[0]["bbox_normalized_0_1000"],
                    entity["bbox_normalized_0_1000"],
                )
                for entity in matched_entities
            ]
            if all(entity["world_coordinate"] is not None for entity in matched_entities):
                worlds = np.asarray([
                    [entity["world_coordinate"]["x"], entity["world_coordinate"]["y"]]
                    for entity in matched_entities
                ])
                world_variation = (worlds.max(axis=0) - worlds.min(axis=0)).tolist()

        labels = [comparison["fresh_label"] for comparison in comparisons]
        semantic_types = [comparison.get("semantic_type") for comparison in comparisons]
        summaries.append({
            "reference_id": reference_entity["reference_id"],
            "canonical_label": reference_entity["canonical_label"],
            "fresh_labels_by_run": labels,
            "label_consistent": (
                None not in labels
                and len({normalize_semantic_label(label) for label in labels}) == 1
            ),
            "semantic_types_by_run": semantic_types,
            "semantic_type_consistent": (
                None not in semantic_types and len(set(semantic_types)) == 1
            ),
            "reference_bbox_iou_by_run": [comparison["bbox_iou"] for comparison in comparisons],
            "bbox_iou_vs_run01": bbox_consistency,
            "world_xy_peak_to_peak_m": world_variation,
            "results_by_run": [comparison["result"] for comparison in comparisons],
        })
    return summaries


def draw_semantic_scene(image, entities):
    visual = image.copy()
    for entity in entities:
        matched = entity["reference_review_status"] == "MATCHED_REFERENCE"
        if entity["semantic_label_status"] != "SPECIFIC":
            color = (0, 0, 255)
        elif matched:
            color = (0, 255, 0)
        elif entity["active_workspace_status"] == "INSIDE":
            color = (255, 255, 0)
        else:
            color = (160, 160, 160)
        x1, y1, x2, y2 = entity["bbox_pixel_raw"]
        raw = entity["raw_reference_pixel"]
        cv2.rectangle(visual, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.circle(visual, (int(raw["u"]), int(raw["v"])), 6, (0, 0, 255), -1)
        world = entity["world_coordinate"]
        coordinate = "X=NA Y=NA" if world is None else f"X={world['x']:.2f} Y={world['y']:.2f}"
        reference_mark = " REF" if matched else ""
        lines = [
            f"{entity['id']} {entity['label']}{reference_mark}",
            f"{entity['semantic_type']} {entity['active_workspace_status']} {coordinate}",
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
    reference = load_partial_semantic_reference(config["semantic_reference_path"])
    bounds = config["legacy_workspace_bounds_m"]
    workspace_bounds = (*bounds["x"], *bounds["y"])
    iou_threshold = float(config["repeatability_iou_match_threshold"])

    processor, model, model_load_sec = load_local_qwen(config["model_id"])
    inputs = prepare_qwen_inputs(processor, model, image_path, make_prompt())
    runs = []
    for run_number in range(1, args.runs + 1):
        raw_output, inference_sec = generate_qwen_text(
            processor, model, inputs, max_new_tokens=1800
        )
        detections = parse_semantic_entities(raw_output)
        grounded = ground_scene_entities(detections, image_size, geometry, workspace_bounds)
        entities, comparisons = apply_semantic_reference(
            grounded, reference, iou_threshold
        )
        runs.append({
            "run": run_number,
            "image": str(image_path),
            "coordinate_frame": geometry.coordinate_system,
            "raw_model_output": raw_output,
            "inference_sec": inference_sec,
            "entities": entities,
            "reference_comparison": comparisons,
        })
        print(
            f"run {run_number}: entities={len(entities)} "
            f"specific={sum(entity['semantic_label_status'] == 'SPECIFIC' for entity in entities)} "
            f"generic={sum(entity['semantic_label_status'] == 'GENERIC' for entity in entities)} "
            f"references_passed={sum(item['result'] == 'PASS' for item in comparisons)}/3 "
            f"inference={inference_sec:.3f} s"
        )

    repeatability = reference_repeatability(runs, reference)
    all_references_passed = all(
        comparison["result"] == "PASS"
        for run in runs
        for comparison in run["reference_comparison"]
    )
    invalid_count = sum(
        entity["active_workspace_status"] == "INVALID_GEOMETRY"
        for run in runs
        for entity in run["entities"]
    )
    result = "PASS" if all_references_passed and not invalid_count else "PARTIAL"

    comparison_output = {
        "reference_file": config["semantic_reference_path"],
        "reference_status": reference["reference_status"],
        "association_rule": f"maximum bbox IoU >= {iou_threshold}; not an accuracy threshold",
        "runs": [
            {"run": run["run"], "comparisons": run["reference_comparison"]}
            for run in runs
        ],
    }
    active_counts = [
        {
            status: sum(entity["active_workspace_status"] == status for entity in run["entities"])
            for status in ("INSIDE", "OUTSIDE", "INVALID_GEOMETRY")
        }
        for run in runs
    ]

    import torch
    from transformers import __version__ as transformers_version

    summary = {
        "result": result,
        "verification_type": "Semantic Label Specificity / Software Integration Verification",
        "not_final_paper_accuracy": True,
        "created_at": datetime.now().astimezone().isoformat(),
        "image": str(image_path),
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
        "architecture": {
            "vlm_outputs": ["specific semantic label", "semantic_type", "bbox_2d"],
            "vlm_does_not_output": [
                "navigation_role", "target selection", "collision role", "path role"
            ],
            "geometry_outputs": ["undistorted pixel", "metric X/Y"],
        },
        "prompt_contains_current_reference_answers": False,
        "reference_status": reference["reference_status"],
        "entity_counts": [len(run["entities"]) for run in runs],
        "label_status_counts_by_run": [
            {
                status: sum(entity["semantic_label_status"] == status for entity in run["entities"])
                for status in ("SPECIFIC", "GENERIC", "UNKNOWN")
            }
            for run in runs
        ],
        "active_workspace_counts_by_run": active_counts,
        "reference_repeatability": repeatability,
        "unreviewed_entity_counts": [
            sum(entity["reference_review_status"] == "UNREVIEWED" for entity in run["entities"])
            for run in runs
        ],
    }

    visual = draw_semantic_scene(image, runs[0]["entities"])
    args.output_dir.mkdir(parents=True)
    for run in runs:
        (args.output_dir / f"semantic_run{run['run']:02d}.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2) + "\n"
        )
    (args.output_dir / "semantic_reference_comparison.json").write_text(
        json.dumps(comparison_output, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "semantic_repeatability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    visualization_path = args.output_dir / "photo_04_semantic_refined.png"
    if not cv2.imwrite(str(visualization_path), visual):
        raise RuntimeError(f"could not write visualization: {visualization_path}")

    print(f"model load: {model_load_sec:.3f} s")
    print(f"entity counts: {summary['entity_counts']}")
    print(f"active workspace counts: {active_counts}")
    print(f"results: {args.output_dir}")
    print(result)


if __name__ == "__main__":
    main()
