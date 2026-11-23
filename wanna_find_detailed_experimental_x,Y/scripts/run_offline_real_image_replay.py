#!/usr/bin/env python3
"""Replay compatible stored images through the unchanged STEP 20 scene monitor."""

import argparse
import json
from pathlib import Path
import sys

import cv2


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.perception.image_geometry_audit import (
    annotate_replay_frame,
    read_bgr,
    run_offline_replay,
)
from vlm_reposition.perception.scene_monitor import load_scene_monitor_config


DEFAULT_MANIFEST = WORKSPACE / "config" / "offline_real_image_replay.json"
DEFAULT_SCENE_CONFIG = WORKSPACE / "config" / "live_scene_monitor.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "offline_image_replay"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary_path = args.output_dir / "real_image_scene_monitor_summary.json"
    observation_dir = args.output_dir / "scene_monitor_observations"
    visualization_dir = args.output_dir / "scene_monitor_visualizations"
    if summary_path.exists() or observation_dir.exists() or visualization_dir.exists():
        raise FileExistsError("refusing to overwrite existing replay outputs")

    manifest = json.loads(args.manifest.read_text())
    scene_config = load_scene_monitor_config(args.scene_config)
    observations = run_offline_replay(manifest, scene_config)
    repeated = run_offline_replay(manifest, scene_config)
    deterministic = observations == repeated
    if not deterministic:
        raise RuntimeError("stored-image replay was not deterministic")

    observation_dir.mkdir(parents=True)
    visualization_dir.mkdir()
    for observation in observations:
        stem = Path(observation["source_image"]).stem
        write_json(observation_dir / f"replay_{observation['replay_index']:03d}_{stem}.json", observation)
        image = annotate_replay_frame(read_bgr(observation["source_image"]), observation)
        if not cv2.imwrite(str(visualization_dir / f"replay_{observation['replay_index']:03d}_{stem}.png"), image):
            raise RuntimeError(f"failed to write replay visualization: {stem}")

    summary = {
        "schema_version": "0.1",
        "software_result": "PASS",
        "dataset_usability_status": manifest["geometry_status"],
        "stored_image_replay_result": "PASS",
        "evidence_boundary": "OFFLINE STORED-IMAGE REPLAY IS NOT LIVE REAL-TIME EVIDENCE",
        "ordering_policy": manifest["ordering_policy"],
        "tracking_interpretation": "DEVELOPMENT_ONLY_UNKNOWN_TIME_GAPS",
        "scene_monitor_reuse": "EXISTING_STEP_20_SCENE_MONITOR_UNCHANGED",
        "deterministic_repeat": deterministic,
        "image_count": len(observations),
        "scene_state_counts": {
            state: sum(item["scene_state"] == state for item in observations)
            for state in ("NO_SIGNIFICANT_CHANGE", "LOCAL_CHANGE", "GLOBAL_CHANGE")
        },
        "semantic_trigger_count": sum(item["semantic_trigger"] for item in observations),
        "geometry_scene_cross_check": [{
            "source_image": item["source_image"],
            "geometry_compatibility_status": item["geometry_compatibility_status"],
            "scene_monitor_state": item["scene_state"],
        } for item in observations],
        "qwen_inference": "NOT_RUN",
        "metric_coordinate_generation": "NOT_PERFORMED",
        "physical_obstacle_geometry_generation": "NOT_PERFORMED",
        "planner_injection": "NOT_PERFORMED",
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
