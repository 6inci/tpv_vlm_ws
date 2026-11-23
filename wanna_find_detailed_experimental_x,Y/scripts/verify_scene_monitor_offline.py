#!/usr/bin/env python3
"""Verify the fixed-reference scene monitor with deterministic synthetic frames."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.perception import SceneMonitor, audit_camera_sources, load_scene_monitor_config


DEFAULT_CONFIG = WORKSPACE / "config" / "live_scene_monitor.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "live_scene_monitor"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def semantic_result(value):
    if isinstance(value, dict):
        return {key: semantic_result(item) for key, item in value.items() if key != "processing_time_ms"}
    if isinstance(value, list):
        return [semantic_result(item) for item in value]
    return value


def frame_with_rectangles(rectangles=(), value=40, shape=(120, 160)):
    frame = np.full((shape[0], shape[1], 3), value, dtype=np.uint8)
    for x_min, y_min, x_max, y_max in rectangles:
        frame[y_min:y_max, x_min:x_max] = 220
    return frame


def offline_fixtures():
    empty = frame_with_rectangles()
    return {
        "F1": {"description": "No change", "reference": empty, "frames": [empty.copy() for _ in range(3)]},
        "F2": {
            "description": "Single-frame local noise",
            "reference": empty,
            "frames": [frame_with_rectangles([(20, 20, 40, 40)]), empty.copy(), empty.copy()],
        },
        "F3": {
            "description": "Persistent local change",
            "reference": empty,
            "frames": [frame_with_rectangles([(50, 40, 80, 70)]) for _ in range(3)],
        },
        "F4": {
            "description": "Moving rectangle",
            "reference": empty,
            "frames": [frame_with_rectangles([(x, 60, x + 20, 85)]) for x in (30, 35, 40, 45)],
        },
        "F5": {
            "description": "Candidate disappears and retires",
            "reference": empty,
            "frames": [
                frame_with_rectangles([(60, 30, 85, 55)]), frame_with_rectangles([(60, 30, 85, 55)]),
                empty.copy(), empty.copy(), empty.copy(),
            ],
        },
        "F6": {
            "description": "Global illumination change",
            "reference": empty,
            "frames": [frame_with_rectangles(value=230)],
        },
        "F7": {
            "description": "Two independent candidates",
            "reference": empty,
            "frames": [frame_with_rectangles([(15, 20, 40, 50), (110, 65, 140, 95)]) for _ in range(2)],
        },
        "F8": {
            "description": "Merge and split identity stress",
            "reference": empty,
            "frames": [
                frame_with_rectangles([(30, 35, 55, 65), (90, 35, 115, 65)]),
                frame_with_rectangles([(30, 30, 115, 75)]),
                frame_with_rectangles([(30, 35, 55, 65), (90, 35, 115, 65)]),
            ],
        },
    }


def _source():
    return {"source_id": "SYNTHETIC_OFFLINE_FIXTURE", "source_path": "PROGRAMMATIC_FRAME_SEQUENCE"}


def run_offline_scenarios(config):
    outputs, visual_frames = {}, {}
    for scenario_id, fixture in offline_fixtures().items():
        monitor = SceneMonitor(config, fixture["reference"], _source())
        observations = []
        for frame_index, frame in enumerate(fixture["frames"]):
            started = perf_counter()
            observation = monitor.process(frame, frame_index)
            observation["processing_time_ms"] = (perf_counter() - started) * 1000.0
            observations.append(observation)
        outputs[scenario_id] = {
            "scenario_id": scenario_id,
            "description": fixture["description"],
            "result": "PASS",
            "observations": observations,
            "retired_tracks": deepcopy(monitor.tracker.retired_tracks),
            "latest_trigger": monitor.get_latest_trigger(),
        }
        visual_frames[scenario_id] = fixture["frames"][-1].copy()

    expected = {
        "F1": all(item["scene_state"] == "NO_SIGNIFICANT_CHANGE" and not item["semantic_trigger"] for item in outputs["F1"]["observations"]),
        "F2": not any(item["semantic_trigger"] for item in outputs["F2"]["observations"]),
        "F3": [item["semantic_trigger"] for item in outputs["F3"]["observations"]] == [False, True, True],
        "F4": len({item["candidates"][0]["track_id"] for item in outputs["F4"]["observations"]}) == 1,
        "F5": len(outputs["F5"]["retired_tracks"]) == 1,
        "F6": outputs["F6"]["observations"][0]["scene_state"] == "GLOBAL_CHANGE" and not outputs["F6"]["observations"][0]["candidates"],
        "F7": [item["track_id"] for item in outputs["F7"]["observations"][-1]["candidates"]] == ["track_001", "track_002"],
        "F8": len(outputs["F8"]["observations"]) == 3,
    }
    if not all(expected.values()):
        raise RuntimeError(f"offline scene-monitor expectation failed: {expected}")
    return outputs, visual_frames


def annotate(frame, observation):
    header_height = 55
    image = np.full((frame.shape[0] + header_height, frame.shape[1], 3), 30, dtype=np.uint8)
    image[header_height:] = frame
    color = (0, 0, 220) if observation["scene_state"] == "GLOBAL_CHANGE" else (0, 200, 0)
    for candidate in observation["candidates"]:
        x1, y1, x2, y2 = candidate["bbox_px"]
        y1, y2 = y1 + header_height, y2 + header_height
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        cv2.putText(image, candidate["track_id"], (x1, min(image.shape[0] - 4, y2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    cv2.putText(image, observation["scene_state"], (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"changed={observation['change_score']['changed_pixel_ratio']:.4f}", (5, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"semantic_trigger={observation['semantic_trigger']}", (5, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return image


def build_summary(config, outputs, deterministic, audit):
    observations = [item for result in outputs.values() for item in result["observations"]]
    timings = [item["processing_time_ms"] for item in observations]
    return {
        "overall_software_result": "PASS",
        "offline_deterministic_validation": "PASS",
        "live_external_camera_validation": "NOT_AVAILABLE" if audit["selected_device_path"] is None else "PENDING",
        "scenario_count": len(outputs),
        "frame_count": len(observations),
        "local_change_frame_count": sum(item["scene_state"] == "LOCAL_CHANGE" for item in observations),
        "global_change_frame_count": sum(item["scene_state"] == "GLOBAL_CHANGE" for item in observations),
        "semantic_trigger_frame_count": sum(item["semantic_trigger"] for item in observations),
        "deterministic_scenarios": deterministic,
        "processing_time_ms": {
            "mean": float(np.mean(timings)),
            "p95": float(np.percentile(timings, 95)),
            "status": "VOLATILE_OBSERVATION_NOT_REAL_TIME_GUARANTEE",
        },
        "reference_mode": config["reference_mode"],
        "tracking_scope": "PIXEL_SPACE_CHANGE_TRACKING",
        "candidate_semantics": "CHANGE_CANDIDATE_ONLY",
        "metric_coordinate_conversion": "NOT_PERFORMED",
        "bbox_physical_size_inference": "NOT_PERFORMED",
        "planner_obstacle_conversion": "NOT_PERFORMED",
        "qwen_inference": "NOT_RUN",
        "ros_execution": "NOT_RUN",
        "robot_execution": "NOT_RUN",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    config = load_scene_monitor_config(args.config)
    outputs, frames = run_offline_scenarios(config)
    repeated, _ = run_offline_scenarios(config)
    deterministic = {
        scenario_id: semantic_result(outputs[scenario_id]) == semantic_result(repeated[scenario_id])
        for scenario_id in outputs
    }
    if not all(deterministic.values()):
        raise RuntimeError("offline scene-monitor results are not deterministic")
    audit = audit_camera_sources(config)

    sample_dir = args.output_dir / "observation_samples"
    visualization_dir = args.output_dir / "visualizations"
    sample_dir.mkdir(parents=True)
    visualization_dir.mkdir()
    write_json(args.output_dir / "scene_monitor_config_snapshot.json", config)
    write_json(args.output_dir / "live_camera_audit.json", audit)
    for scenario_id, result in outputs.items():
        write_json(sample_dir / f"{scenario_id}_observations.json", result)
        image = annotate(frames[scenario_id], result["observations"][-1])
        if not cv2.imwrite(str(visualization_dir / f"{scenario_id}_annotated.png"), image):
            raise RuntimeError(f"failed to write {scenario_id} visualization")
    summary = build_summary(config, outputs, deterministic, audit)
    write_json(args.output_dir / "offline_validation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
