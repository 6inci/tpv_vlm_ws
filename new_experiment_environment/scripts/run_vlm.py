#!/usr/bin/env python3
"""Run one local Qwen scene-understanding pass and preserve its raw output."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys


REFERENCE_CODE = Path("/home/binci/vlm_ws/wanna_find_detailed_experimental_x,Y")
sys.path.insert(0, str(REFERENCE_CODE / "src"))

from vlm_reposition.perception.qwen_grounding import (  # noqa: E402
    generate_qwen_text,
    load_local_qwen,
    prepare_qwen_inputs,
)


PROMPT = '''Analyze this single oblique overhead image of an indoor industrial robotics/logistics test environment.

Return JSON only. Do not use markdown. Use integer image coordinates normalized to 0..1000.

Required schema:
{
  "scene_context": {
    "environment_type": "short label",
    "summary": "one concise sentence",
    "confidence": 0.0
  },
  "objects": [
    {
      "id": "obj_001",
      "label": "specific visible object",
      "category": "rack|box|pallet|workbench|truck|traffic_cone|manipulator|mobile_robot|cart|drum|fence|other",
      "bbox_2d": [x1,y1,x2,y2],
      "confidence": 0.0
    }
  ],
  "relations": [
    {
      "subject_id": "obj_001",
      "predicate": "near|left_of|right_of|in_front_of|behind|on|contains|inside|adjacent_to|bounded_by",
      "object_id": "obj_002",
      "confidence": 0.0
    }
  ],
  "semantic_regions": [
    {
      "id": "region_storage",
      "label": "Storage Area|Loading Area|Work Station|Material Area|No-Go Zone|Drop Zone|Traversable Area",
      "polygon": [[x1,y1],[x2,y2],[x3,y3]],
      "confidence": 0.0,
      "evidence_object_ids": ["obj_001"],
      "reasoning": "short evidence-based reason"
    }
  ],
  "traversable_area": {
    "polygon": [[x1,y1],[x2,y2],[x3,y3]],
    "confidence": 0.0,
    "reasoning": "visible open floor only"
  },
  "restricted_areas": [
    {"polygon": [[x1,y1],[x2,y2],[x3,y3]], "reason": "visible restriction evidence"}
  ],
  "unknown_areas": [
    {"polygon": [[x1,y1],[x2,y2],[x3,y3]], "reason": "occluded or ambiguous"}
  ]
}

Rules:
- Include each distinct major visible object once. Group tightly stacked identical boxes as one box_group if individual boxes are not needed.
- Bboxes and polygons must refer to the original full image, never a crop.
- Infer regions from combinations of objects and spatial relations, not from one object alone.
- Traversable area means visible open black floor that a mobile robot could approximately occupy in image space.
- Object footprints, regions, and traversability are approximate image-plane interpretations, not metric geometry.
- Do not invent hidden objects, physical dimensions, metric coordinates, or a path.'''


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "experiment_config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    image = Path(config["input_image"])
    output = Path(config["output_dir"])
    raw_path = output / "qwen_raw_output.txt"
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite {raw_path}")
    if not image.is_file():
        raise FileNotFoundError(image)
    if not Path(config["model_cache"]).is_dir():
        raise FileNotFoundError(config["model_cache"])

    output.mkdir(parents=True, exist_ok=True)
    (output / "vlm_prompt.txt").write_text(PROMPT + "\n")

    processor, model, load_seconds = load_local_qwen(config["model_id"])
    inputs = prepare_qwen_inputs(processor, model, image, PROMPT)
    raw, inference_seconds = generate_qwen_text(
        processor, model, inputs, int(config["max_new_tokens"])
    )
    raw_path.write_text(raw.rstrip() + "\n")

    import cv2
    import torch
    import transformers

    frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
    runtime = {
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "created_at": datetime.now().astimezone().isoformat(),
        "input_image": str(image),
        "input_sha256": sha256(image),
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
        "model_id": config["model_id"],
        "local_files_only": True,
        "device": str(model.device),
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "raw_output_sha256": sha256(raw_path),
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
