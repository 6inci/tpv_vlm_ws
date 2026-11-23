#!/usr/bin/env python3
"""Run a compact second Qwen pass for relations and image-plane regions."""

from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_vlm import generate_qwen_text, load_local_qwen, prepare_qwen_inputs

PROMPT = '''Analyze the industrial robotics image. Return JSON only, compact, under 700 tokens.
Coordinates are integer 0..1000 in the original image. Known object IDs:
obj_001 left rack; obj_002 right rack; obj_003 truck; obj_004 workbench;
obj_005..007 cones; obj_008..010 manipulators; obj_011 mobile robot;
obj_012 cart; obj_013 lower box pallet; obj_014 gray-block pallet;
obj_015 drum pallet; obj_016 right box pallet.
Schema:
{"relations":[{"subject_id":"obj_...","predicate":"near|left_of|right_of|in_front_of|behind|on|contains|inside|adjacent_to","object_id":"obj_...","confidence":0.0}],
"semantic_regions":[{"id":"region_storage|region_loading|region_work|region_material|region_robot_work","label":"Storage Area|Loading Area|Work Station|Material Area|No-Go Zone","polygon":[[x,y],[x,y],[x,y]],"confidence":0.0,"evidence_object_ids":["obj_..."],"reasoning":"brief"}],
"traversable_area":{"polygon":[[x,y],[x,y],[x,y]],"confidence":0.0,"reasoning":"brief"},
"restricted_areas":[{"polygon":[[x,y],[x,y],[x,y]],"reason":"brief"}],
"unknown_areas":[{"polygon":[[x,y],[x,y],[x,y]],"reason":"brief"}]}
Give at most 6 relations and exactly 5 semantic regions. Regions and traversability are approximate image-plane interpretations, not metric geometry or paths.'''


def main():
    config = json.loads((ROOT / "experiment_config.json").read_text())
    output = Path(config["output_dir"])
    raw_path = output / "qwen_semantics_raw_output.txt"
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite {raw_path}")
    (output / "vlm_semantics_prompt.txt").write_text(PROMPT + "\n")
    processor, model, load_seconds = load_local_qwen(config["model_id"])
    inputs = prepare_qwen_inputs(processor, model, Path(config["input_image"]), PROMPT)
    raw, inference_seconds = generate_qwen_text(processor, model, inputs, 700)
    raw_path.write_text(raw.rstrip() + "\n")
    metadata = {"created_at": datetime.now().astimezone().isoformat(), "device": str(model.device),
                "model_load_seconds": load_seconds, "inference_seconds": inference_seconds,
                "max_new_tokens": 700, "purpose": "relations_and_regions"}
    (output / "runtime_semantics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
