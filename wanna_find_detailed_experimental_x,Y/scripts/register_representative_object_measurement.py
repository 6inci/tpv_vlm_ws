#!/usr/bin/env python3
"""Register a user measurement only when the STEP 23M contract is READY."""

import hashlib
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.world import (
    assess_measurement,
    build_measured_world_placement,
    register_ready_measurement,
)


ACTUAL_INPUT = WORKSPACE / "config" / "representative_object_measurement.json"
TEMPLATE = WORKSPACE / "config" / "representative_object_measurement_TEMPLATE.json"
REGISTRY = WORKSPACE / "config" / "measured_object_geometry_registry.json"
STEP23_CONFIG = WORKSPACE / "config" / "physical_obstacle_geometry.json"
STEP23_RESULT = WORKSPACE / "results" / "physical_obstacle_geometry" / "representative_geometry_status.json"
OUTPUT = WORKSPACE / "results" / "physical_measurement_registration"
PROTECTED_BASELINE = {
    "file_count": 461,
    "digest_sha256": "f1d56b83b69ded03146825556da0a7e72bea3428ab9df6f29ef955de84ade1c1",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_protected():
    later_results = (
        WORKSPACE / "results" / "navigation_obstacle_adapter",
        WORKSPACE / "results" / "navigation_dynamic_world_integration",
        WORKSPACE / "results" / "navigation_path_validity",
        WORKSPACE / "results" / "controlled_navigation_path_effect",
        WORKSPACE / "results" / "path_block_persistence_integration",
        WORKSPACE / "results" / "persistent_block_replanning_integration",
        WORKSPACE / "results" / "offline_navigation_chain_integration",
        WORKSPACE / "results" / "observed_blocking_acquisition",
    )
    files = sorted(
        path for path in (WORKSPACE / "results").rglob("*")
        if path.is_file() and OUTPUT not in path.parents
        and all(later not in path.parents for later in later_results)
    )
    files += [
        STEP23_CONFIG,
        WORKSPACE / "src" / "vlm_reposition" / "world" / "object_footprint.py",
        WORKSPACE / "src" / "vlm_reposition" / "geometry" / "legacy.py",
        Path("/home/binci/qwen_vision_test/homography_12pt_undistorted.json"),
        Path("/home/binci/camera_calibration_1920x1080/calibration_result.json"),
    ]
    records = [(str(path), sha256_file(path)) for path in files]
    digest = hashlib.sha256("".join(f"{digest}  {path}\n" for path, digest in records).encode()).hexdigest()
    return {"file_count": len(records), "digest_sha256": digest}


def load_representative_metric_point():
    result = read_json(STEP23_RESULT)
    metric = result["semantic_metric_input"]
    if (
        result["representative_physical_geometry"] != "NOT_READY_MEASUREMENT_REQUIRED"
        or metric["image_id"] != "image_014_photo_08"
        or metric["track_id"] != "track_039"
        or metric["semantic_label"] != "tennis racket"
        or metric["world_xy"] != [0.7646719813346863, 0.713485598564148]
        or metric["coordinate_frame"] != "LEGACY_WORKSPACE_COORDINATE"
    ):
        raise ValueError("STEP 23 representative metric evidence mismatch")
    return metric


def main():
    protected_before = snapshot_protected()
    if protected_before != PROTECTED_BASELINE:
        raise RuntimeError(f"protected STEP 12-23 artifact baseline mismatch: {protected_before}")
    if not TEMPLATE.is_file() or not REGISTRY.is_file():
        raise FileNotFoundError("measurement template and READY-only registry are required")

    metric = load_representative_metric_point()
    registry = read_json(REGISTRY)
    actual_input_exists = ACTUAL_INPUT.is_file()
    measurement = read_json(ACTUAL_INPUT) if actual_input_exists else None
    assessment = assess_measurement(measurement, ACTUAL_INPUT)
    registry_updated = False
    if assessment["status"] == "READY":
        registry = register_ready_measurement(registry, assessment)
        write_json(REGISTRY, registry)
        world_placement = build_measured_world_placement(assessment, metric)
        registry_updated = True
    else:
        world_placement = {"status": "NOT_RUN", "reason": assessment["status"]}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "existing_measurement_audit.json", {
        "searched_locations": [str(WORKSPACE), "/home/binci/qwen_vision_test"],
        "actual_measurement_input": str(ACTUAL_INPUT),
        "actual_measurement_input_exists": actual_input_exists,
        "accepted_source": "USER_PHYSICAL_MEASUREMENT",
        "explicit_user_measurement_found": actual_input_exists,
        "internet_or_default_product_dimensions_used": False,
        "vlm_or_image_bbox_dimensions_used": False,
    })
    write_json(OUTPUT / "measurement_request.json", {
        "object_id": "photo08_track039_tennis_racket",
        "semantic_label": "tennis racket",
        "status": assessment["status"],
        "template": str(TEMPLATE),
        "missing_fields": assessment["missing_fields"],
        "required_measurements": [
            "select one shape: CIRCLE, AXIS_ALIGNED_RECTANGLE, or POLYGON",
            "provide the selected shape dimensions or local polygon in mm, cm, or m",
            "measure the relation from the observed bottom-center ground reference to the physical footprint anchor",
            "provide a reproducible orientation reference and yaw for rectangle or polygon",
        ],
        "optional_measurements": ["measurement uncertainty with unit", "measurement notes"],
        "instructions": [
            "Copy the template to config/representative_object_measurement.json and replace nulls only with direct user measurements.",
            "Do not use internet specifications, image bbox size, semantic assumptions, VLM estimates, or STEP 14 synthetic values.",
            "Keep raw input units; canonical meter values are generated deterministically after validation.",
        ],
        "shape_checklist": {
            "AXIS_ALIGNED_RECTANGLE": ["total length", "maximum width", "anchor relation", "orientation reference and yaw"],
            "POLYGON": ["ground-footprint boundary points", "polygon unit", "anchor relation", "local reference axis and yaw"],
            "CIRCLE": ["radius", "center-to-observed-anchor relation"],
        },
        "optional_experiment_design_note": "A measured box, rectangular block, or cylinder would simplify future footprint and anchor validation; the current representative object is not changed automatically.",
    })

    protected_after = snapshot_protected()
    summary = {
        "schema_version": "0.1",
        "step_result": "PASS",
        "software": "PASS",
        "representative_object": "photo08_track039_tennis_racket",
        "measurement_input_status": assessment["status"],
        "representative_physical_geometry": "READY" if assessment["status"] == "READY" else "NOT_READY",
        "registry_updated": registry_updated,
        "registry_object_count": len(registry["objects"]),
        "world_placement": world_placement,
        "planning_geometry": {"status": "NOT_GENERATED"},
        "planner_injection": "NOT_RUN",
        "qwen_inference_runs": 0,
        "step17_call": "NOT_RUN",
        "step18_call": "NOT_RUN",
        "step19_call": "NOT_RUN",
        "astar": "NOT_RUN",
        "costmap": "NOT_RUN",
        "protected_artifacts_before": protected_before,
        "protected_artifacts_after": protected_after,
        "protected_artifacts_unchanged": protected_before == protected_after,
        "recommended_next_step": "USER_PHYSICAL_MEASUREMENT_INPUT" if assessment["status"] != "READY" else "SEPARATELY_APPROVED_STEP_24",
    }
    if not summary["protected_artifacts_unchanged"]:
        raise RuntimeError("protected STEP 12-23 artifacts changed")
    write_json(OUTPUT / "measurement_registration_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
