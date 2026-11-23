#!/usr/bin/env python3
"""Run a bounded, headless scene-monitor smoke test on the exact research camera."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.perception import (
    CameraSource,
    CameraSourceError,
    SceneMonitor,
    SceneMonitorError,
    audit_camera_sources,
    load_scene_monitor_config,
    select_stable_reference,
)


DEFAULT_CONFIG = WORKSPACE / "config" / "live_scene_monitor.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "live_scene_monitor"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run_live_validation(config, audit, requested_frames=None, sample_directory=None):
    frame_limit = config["live_validation"]["maximum_frames"]
    if requested_frames is not None:
        if isinstance(requested_frames, bool) or not isinstance(requested_frames, int) or requested_frames <= 0:
            raise ValueError("requested_frames must be a positive integer")
        frame_limit = min(frame_limit, requested_frames)
    base = {
        "software_interface": "PASS",
        "live_external_camera_validation": None,
        "camera_audit_status": audit["audit_status"],
        "configured_device_path": audit["configured_device_path"],
        "selected_device_path": audit["selected_device_path"],
        "fallback_used": False,
        "frames_requested": frame_limit,
        "frames_captured": 0,
        "capture_failures": 0,
        "processing_frames": 0,
        "actual_capture_configuration": None,
        "reference": None,
        "mean_processing_time_ms": None,
        "p95_processing_time_ms": None,
        "local_change_frames": 0,
        "global_change_frames": 0,
        "maximum_active_track_count": 0,
        "semantic_trigger_count": 0,
        "performance_metric_status": "VOLATILE_OBSERVATION_NOT_REAL_TIME_GUARANTEE",
    }
    if audit["selected_device_path"] is None:
        base.update({
            "live_external_camera_validation": "NOT_AVAILABLE",
            "error_state": "CAMERA_NOT_FOUND",
            "reason": "The exact configured external research camera is unavailable; no built-in fallback was attempted.",
        })
        return base

    source = CameraSource(config, audit)
    timings = []
    try:
        base["actual_capture_configuration"] = source.open()
        for _ in range(config["warmup_frames"]):
            source.read()
            base["frames_captured"] += 1

        def reference_frames():
            for _ in range(config["maximum_reference_attempts"]):
                frame, _ = source.read()
                base["frames_captured"] += 1
                yield frame

        reference, reference_status = select_stable_reference(reference_frames(), config)
        base["reference"] = reference_status
        monitor = SceneMonitor(config, reference, {
            "source_id": config["camera_selector"]["required_identity_substring"],
            "source_path": audit["selected_device_path"],
        })
        saved = 0
        for frame_index in range(frame_limit):
            frame, _ = source.read()
            base["frames_captured"] += 1
            started = perf_counter()
            observation = monitor.process(frame, frame_index)
            timings.append((perf_counter() - started) * 1000.0)
            base["processing_frames"] += 1
            base["local_change_frames"] += observation["scene_state"] == "LOCAL_CHANGE"
            base["global_change_frames"] += observation["scene_state"] == "GLOBAL_CHANGE"
            base["maximum_active_track_count"] = max(base["maximum_active_track_count"], len(observation["active_tracks"]))
            base["semantic_trigger_count"] += observation["semantic_trigger"]
            if sample_directory is not None and saved < config["live_validation"]["maximum_saved_sample_frames"] and (
                observation["semantic_trigger"] or observation["scene_state"] == "GLOBAL_CHANGE"
            ):
                sample_directory.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(sample_directory / f"live_sample_{saved + 1:03d}.png"), frame)
                saved += 1
    except (CameraSourceError, SceneMonitorError) as exc:
        base["capture_failures"] += 1
        base["live_external_camera_validation"] = "FAIL"
        base["error_state"] = exc.code
        base["reason"] = str(exc)
    else:
        base["live_external_camera_validation"] = "PASS"
        base["error_state"] = None
        base["reason"] = None
    finally:
        source.close()
    if timings:
        base["mean_processing_time_ms"] = float(np.mean(timings))
        base["p95_processing_time_ms"] = float(np.percentile(timings, 95))
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "live_validation_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {summary_path}")
    config = load_scene_monitor_config(args.config)
    audit = audit_camera_sources(config)
    write_json(args.output_dir / "live_camera_audit.json", audit)
    summary = run_live_validation(
        config, audit, args.max_frames, args.output_dir / "observation_samples"
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
