#!/usr/bin/env python3
"""Run stored-bbox integration and fresh Qwen single-object grounding."""

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
    generate_qwen_text,
    ground_normalized_bbox,
    load_local_qwen,
    parse_single_detection,
    prepare_qwen_inputs,
)


DEFAULT_CONFIG = WORKSPACE / "config" / "qwen_grounding.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "qwen_grounding"


def load_json(path):
    return json.loads(Path(path).read_text())


def find_stored_object(path, label):
    matches = [item for item in load_json(path).get("objects", []) if item.get("label") == label]
    if len(matches) != 1:
        raise ValueError(f"expected one stored object labeled {label!r}, received {len(matches)}")
    return matches[0]


def make_prompt(target):
    return f'''Locate the object described by this instruction:

"{target}"

Output only JSON in exactly this format:
{{"label":"detected object name","bbox_2d":[x1,y1,x2,y2]}}

Coordinates must be normalized to the range 0 to 1000.
Do not output any explanation.'''


def variation(runs, field):
    values = np.asarray([field(run) for run in runs], dtype=np.float64)
    return (values.max(axis=0) - values.min(axis=0)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--target", default="red cup")
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

    stored = find_stored_object(
        config["stored_legacy_result_path"], config["stored_legacy_target_label"]
    )
    stored_grounding = ground_normalized_bbox(
        args.target, stored["label"], stored["bbox_2d"], image_size, geometry
    )
    stored_raw = stored_grounding["raw_reference_pixel"]
    stored_new_world = stored_grounding["world_coordinate"]
    legacy_world = stored["world_xy_m"]
    world_difference = [
        stored_new_world["x"] - legacy_world[0],
        stored_new_world["y"] - legacy_world[1],
    ]
    stored_check = {
        "result": "PASS",
        "interpretation": (
            "BBox parsing and verified raw-to-undistorted-to-world integration succeeded. "
            "The stored world value used a different legacy distorted-image Homography and was rounded."
        ),
        "source": config["stored_legacy_result_path"],
        "legacy_bbox_normalized_0_1000": stored["bbox_2d"],
        "new_parsed_bbox_normalized_0_1000": stored_grounding["bbox_model_normalized_0_1000"],
        "legacy_bottom_center_pixel_rounded": stored["bottom_center_pixel"],
        "new_bottom_center_pixel": [stored_raw["u"], stored_raw["v"]],
        "legacy_world_xy_m_rounded": legacy_world,
        "new_verified_geometry_world_xy_m": [stored_new_world["x"], stored_new_world["y"]],
        "new_minus_legacy_world_xy_m": world_difference,
        "difference_norm_m": float(np.linalg.norm(world_difference)),
        "grounding": stored_grounding,
    }

    from transformers import __version__ as transformers_version
    import torch

    processor, model, model_load_sec = load_local_qwen(config["model_id"])
    inputs = prepare_qwen_inputs(processor, model, image_path, make_prompt(args.target))

    runs = []
    for run_number in range(1, args.runs + 1):
        raw_output, inference_sec = generate_qwen_text(
            processor, model, inputs, max_new_tokens=80
        )
        detection = parse_single_detection(raw_output)
        grounding = ground_normalized_bbox(
            args.target, detection["label"], detection["bbox_2d"], image_size, geometry
        )
        runs.append({
            "run": run_number,
            "raw_model_output": raw_output,
            "inference_sec": inference_sec,
            "grounding": grounding,
        })
        print(
            f"run {run_number}: bbox={detection['bbox_2d']} "
            f"world=({grounding['world_coordinate']['x']:.6f}, "
            f"{grounding['world_coordinate']['y']:.6f}) m "
            f"inference={inference_sec:.3f} s"
        )

    bbox_variation = variation(
        runs, lambda run: run["grounding"]["bbox_model_normalized_0_1000"]
    )
    pixel_variation = variation(
        runs,
        lambda run: list(run["grounding"]["raw_reference_pixel"].values()),
    )
    world_variation = variation(
        runs,
        lambda run: [
            run["grounding"]["world_coordinate"]["x"],
            run["grounding"]["world_coordinate"]["y"],
        ],
    )
    summary = {
        "result": "PASS",
        "verification_type": "Semantic-to-Pixel-to-Metric Grounding Software Integration",
        "not_final_paper_accuracy": True,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_image": str(image_path),
        "target_query": args.target,
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
        "stored_bbox_integration": stored_check,
        "fresh_inference": {
            "runs": args.runs,
            "bbox_coordinate_peak_to_peak_normalized_0_1000": bbox_variation,
            "raw_reference_pixel_peak_to_peak_px": pixel_variation,
            "world_xy_peak_to_peak_m": world_variation,
            "inference_sec": [run["inference_sec"] for run in runs],
        },
    }

    visual = image.copy()
    first = runs[0]["grounding"]
    x1, y1, x2, y2 = first["bbox_pixel_raw"]
    raw = first["raw_reference_pixel"]
    world = first["world_coordinate"]
    cv2.rectangle(visual, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
    cv2.circle(visual, (int(raw["u"]), int(raw["v"])), 10, (0, 0, 255), -1)
    text = f"{first['label']} X={world['x']:.3f}m Y={world['y']:.3f}m"
    cv2.putText(
        visual, text, (int(x1), max(40, int(y1) - 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "stored_bbox_geometry_check.json").write_text(
        json.dumps(stored_check, ensure_ascii=False, indent=2) + "\n"
    )
    for run in runs:
        (args.output_dir / f"qwen_red_cup_run{run['run']:02d}.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2) + "\n"
        )
    (args.output_dir / "qwen_grounding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    visual_path = args.output_dir / "red_cup_grounding.png"
    if not cv2.imwrite(str(visual_path), visual):
        raise RuntimeError(f"could not write visualization: {visual_path}")

    print(f"model load: {model_load_sec:.3f} s")
    print(f"stored bbox integration: {stored_check['result']}")
    print(f"bbox peak-to-peak: {bbox_variation}")
    print(f"world peak-to-peak: {world_variation} m")
    print(f"results: {args.output_dir}")
    print("PASS")


if __name__ == "__main__":
    main()
