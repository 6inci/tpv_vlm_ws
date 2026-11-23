#!/usr/bin/env python3
"""Build deterministic unified states from saved STEP 07/08/09/11 outputs."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.world import build_state_from_saved_results, load_zones


DEFAULT_OUTPUT = WORKSPACE / "results" / "workspace_state"
DEFAULT_ZONES = WORKSPACE / "config" / "zones.json"
DEFAULT_GROUNDING_CONFIG = WORKSPACE / "config" / "qwen_grounding.json"
TASK_DIR = WORKSPACE / "results" / "task_parser_rework"
TARGET_DIR = WORKSPACE / "results" / "task_conditioned_grounding"

CASES = (
    ("red_cup_zone_a_state.json", "valid_ko_red_cup_zone_a_run01.json", "red_cup_run01.json"),
    ("white_bag_zone_b_state.json", "valid_ko_white_bag_zone_b_run01.json", "white_bag_run01.json"),
    ("trophy_zone_f_state.json", "valid_ko_trophy_zone_f_run01.json", "gold_trophy_like_object_run01.json"),
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def draw_metric_state(state, zones):
    image = np.full((860, 760, 3), 255, dtype=np.uint8)
    origin, scale = (100, 780), 400.0

    def pixel(point):
        return int(round(origin[0] + point[0] * scale)), int(round(origin[1] - point[1] * scale))

    selected = state["destination"]["zone_id"]
    for zone in zones["zones"]:
        polygon = np.asarray([pixel(point) for point in zone["polygon"]], dtype=np.int32)
        color = (170, 235, 170) if zone["id"] == selected else (235, 235, 235)
        cv2.fillPoly(image, [polygon], color)
        cv2.polylines(image, [polygon], True, (80, 80, 80), 2, cv2.LINE_AA)
        center = pixel(zone["center"])
        cv2.drawMarker(image, center, (0, 120, 0), cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(image, zone["id"], (center[0] - 30, center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    for entity in state["scene"]["entities"]:
        world = entity["world_coordinate"]
        if entity["active_workspace_status"] == "INSIDE":
            cv2.circle(image, pixel((world["x"], world["y"])), 4, (180, 80, 20), -1)

    target = state["target"]
    world = target["world_coordinate"]
    target_point = pixel((world["x"], world["y"]))
    cv2.circle(image, target_point, 9, (0, 0, 220), -1)
    cv2.putText(image, target["grounding_phrase"], (target_point[0] + 10, target_point[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 2, cv2.LINE_AA)

    cv2.rectangle(image, pixel((0.0, 1.8)), pixel((1.2, 0.0)), (0, 0, 0), 3)
    cv2.putText(image, "LEGACY_WORKSPACE_COORDINATE [m]", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "Zone center is not a robot goal pose. No path shown.", (100, 825), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES)
    parser.add_argument("--grounding-config", type=Path, default=DEFAULT_GROUNDING_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    grounding_config = json.loads(args.grounding_config.read_text())
    scene_path = Path(grounding_config["semantic_scene_result_path"])
    threshold = grounding_config["repeatability_iou_match_threshold"]
    states = []
    for filename, task_name, target_name in CASES:
        state = build_state_from_saved_results(
            TASK_DIR / task_name,
            TARGET_DIR / target_name,
            scene_path,
            args.zones,
            threshold,
            grounding_config["legacy_geometry_path"],
        )
        states.append((filename, state))

    args.output_dir.mkdir(parents=True)
    for filename, state in states:
        write_json(args.output_dir / filename, state)

    zones = load_zones(args.zones)
    visualization_path = args.output_dir / "workspace_task_state_metric.png"
    if not cv2.imwrite(str(visualization_path), draw_metric_state(states[0][1], zones)):
        raise RuntimeError(f"could not write {visualization_path}")

    summary = {
        "result": "PASS",
        "verification_type": "Unified Workspace / Task State Construction",
        "input_mode": "SAVED_RESULTS_ONLY",
        "state_count": len(states),
        "coordinate_frame": zones["coordinate_frame"],
        "unit": zones["unit"],
        "scene_entity_count": states[0][1]["scene"]["entity_count"],
        "states": [
            {
                "result_file": str((args.output_dir / filename).resolve()),
                "instruction": state["task"]["raw_instruction"],
                "target": state["target"]["grounding_phrase"],
                "target_workspace_status": state["target"]["active_workspace_status"],
                "destination_zone": state["destination"]["zone_id"],
                "scene_association": state["target"]["scene_association"],
                "state_validity": state["state_validity"],
                "feasibility": state["feasibility"]["status"],
                "planning": state["planning"]["status"],
            }
            for filename, state in states
        ],
        "visualization": str(visualization_path.resolve()),
        "new_inference_performed": False,
        "new_geometry_estimation_performed": False,
        "task_parsing_performed": False,
    }
    write_json(args.output_dir / "workspace_state_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
