#!/usr/bin/env python3
"""Build STEP 23 physical-footprint policy and development-only evidence."""

import hashlib
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.world import (
    build_local_footprint,
    load_physical_obstacle_config,
    measurement_requirements,
    place_world_footprint,
    planning_readiness,
)


CONFIG = WORKSPACE / "config" / "physical_obstacle_geometry.json"
STEP22D = WORKSPACE / "results" / "candidate_conditioned_vlm_revalidation_method_d"
OUTPUT = WORKSPACE / "results" / "physical_obstacle_geometry"
PROTECTED_BASELINE = {
    "file_count": 453,
    "digest_sha256": "10e689387a1e5d6baeb38c989d236b524f3f755666ee3f2126419466fb25bfd8",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_protected():
    later_results = (
        WORKSPACE / "results" / "physical_measurement_registration",
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
        WORKSPACE / "src" / "vlm_reposition" / "geometry" / "legacy.py",
        Path("/home/binci/qwen_vision_test/homography_12pt_undistorted.json"),
        Path("/home/binci/camera_calibration_1920x1080/calibration_result.json"),
    ]
    records = [(str(path), sha256_file(path)) for path in files]
    digest = hashlib.sha256("".join(f"{digest}  {path}\n" for path, digest in records).encode()).hexdigest()
    return {"file_count": len(records), "digest_sha256": digest}


def load_representative_observation(config):
    configured = config["representative_observations"][0]
    summary = read_json(STEP22D / "step22d_summary.json")
    parsed = read_json(STEP22D / "parsed_result.json")
    metric = read_json(STEP22D / "metric_result.json")
    semantic = parsed["semantic_result"]
    expected = configured["metric_point"]
    if (
        summary["step_result"] != "PASS"
        or summary["image_id"] != configured["image_id"]
        or summary["track_id"] != configured["track_id"]
        or semantic["visible_label"] != configured["semantic_label"]
        or metric["status"] != "AVAILABLE"
        or metric["world_xy"] != expected["world_xy"]
        or metric["coordinate_frame"] != expected["coordinate_frame"]
        or metric["ground_reference"]["policy"] != "BOTTOM_CENTER"
        or metric["ground_reference"]["meaning"] != "DEVELOPMENT_GROUND_REFERENCE"
    ):
        raise ValueError("STEP 22D representative evidence mismatch")
    return {
        "image_id": configured["image_id"],
        "track_id": configured["track_id"],
        "semantic_label": semantic["visible_label"],
        "semantic_type": semantic["semantic_type"],
        "world_xy": metric["world_xy"],
        "coordinate_frame": metric["coordinate_frame"],
        "unit": metric["unit"],
        "ground_reference": metric["ground_reference"],
        "geometry_provenance": metric["geometry_provenance"],
        "qwen_bbox_normalized_1000": semantic["object_bbox_normalized_1000"],
        "conditioning_candidate_bbox_normalized_1000": [504, 481, 682, 604],
        "independent_object_extent": "NOT_EVALUATED",
    }


def main():
    protected_before = snapshot_protected()
    if protected_before != PROTECTED_BASELINE:
        raise RuntimeError(f"protected STEP 12-22D artifact baseline mismatch: {protected_before}")

    config = load_physical_obstacle_config(CONFIG)
    observation = load_representative_observation(config)
    representative = config["representative_observations"][0]["physical_geometry"]
    fixture = config["development_fixtures"][0]
    representative_readiness = planning_readiness(representative)
    fixture_result = place_world_footprint(fixture, fixture["fixture_world_anchor_xy"])
    requirements = measurement_requirements(representative)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "physical_geometry_policy.json", {
        "schema_version": config["schema_version"],
        "source_policy": config["source_policy"],
        "allowed_shape_types": config["allowed_shape_types"],
        "anchor_policy": config["anchor_policy"],
        "orientation_policy": config["orientation_policy"],
        "layer_policy": config["layer_policy"],
        "planar_ground_contact_requirement": "Only independently validated physical boundary points known to lie on the ground plane may use ground Homography; image bbox corners are prohibited.",
        "semantic_metric_point_is_physical_footprint": False,
        "vlm_bbox_used_for_metric_dimensions": False,
    })
    write_json(OUTPUT / "geometry_source_audit.json", {
        "object": "tennis racket",
        "searched_locations": ["/home/binci/vlm_ws", "/home/binci/qwen_vision_test"],
        "audit_mode": "READ_ONLY_REPOSITORY_DOC_CONFIG_RESULT_REVIEW",
        "explicit_physical_measurement_sources": [],
        "length_measurement": "NOT_FOUND",
        "width_measurement": "NOT_FOUND",
        "ground_footprint_measurement": "NOT_FOUND",
        "observed_point_to_footprint_anchor_measurement": "NOT_FOUND",
        "physical_geometry_source": "UNKNOWN",
        "internet_or_general_product_specification_used": False,
        "image_bbox_used_as_measurement": False,
        "semantic_label_used_as_measurement": False,
        "vlm_generated_dimensions_used": False,
    })
    write_json(OUTPUT / "representative_geometry_status.json", {
        "schema_version": "0.1",
        "semantic_metric_input": observation,
        "physical_geometry": representative,
        "metric_point_status": "AVAILABLE",
        "physical_dimensions": "UNKNOWN",
        "anchor_relation": "UNKNOWN",
        "physical_footprint": "NOT_AVAILABLE",
        "planning_geometry_readiness": representative_readiness,
        "representative_physical_geometry": "NOT_READY_MEASUREMENT_REQUIRED",
        "planning_ready": False,
        "planning_geometry": {"status": "NOT_GENERATED"},
        "planner_injection": "NOT_RUN",
    })
    write_json(OUTPUT / "development_fixture_result.json", {
        "schema_version": "0.1",
        "fixture_status": "DEVELOPMENT_ONLY",
        "not_actual_tennis_racket_geometry": True,
        "source_record": fixture,
        "local_footprint": build_local_footprint(fixture),
        "world_placement": fixture_result,
    })
    write_json(OUTPUT / "measurement_requirements.json", {
        "object": representative["semantic_label"],
        "required_before_real_planner_injection": requirements,
        "minimum_registration_policy": "Register only the selected 2D ground-contact shape, its minimum dimensions, the observation-anchor relation, and orientation reference when required.",
        "arbitrary_values_allowed": False,
    })

    protected_after = snapshot_protected()
    summary = {
        "schema_version": "0.1",
        "step_result": "PASS",
        "software": "PASS",
        "representative_semantic_metric_point": "AVAILABLE",
        "representative_physical_geometry": "NOT_READY_MEASUREMENT_REQUIRED",
        "development_fixture": "DEVELOPMENT_ONLY",
        "qwen_inference_runs": 0,
        "planner_injection": "NOT_RUN",
        "step17_call": "NOT_RUN",
        "step18_call": "NOT_RUN",
        "step19_call": "NOT_RUN",
        "astar": "NOT_RUN",
        "costmap": "NOT_RUN",
        "protected_artifacts_before": protected_before,
        "protected_artifacts_after": protected_after,
        "protected_artifacts_unchanged": protected_before == protected_after,
        "recommended_next_step": "STEP_23M_REPRESENTATIVE_OBJECT_PHYSICAL_MEASUREMENT_REGISTRATION",
    }
    if not summary["protected_artifacts_unchanged"]:
        raise RuntimeError("protected STEP 12-22D artifacts changed")
    write_json(OUTPUT / "planning_readiness_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
