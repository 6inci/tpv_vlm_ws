#!/usr/bin/env python3
"""Build the STEP 24N adapter artifact without executing a planner."""

import hashlib
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.planning import build_navigation_obstacle_observation


STEP22D = WORKSPACE / "results" / "candidate_conditioned_vlm_revalidation_method_d"
STEP23M = WORKSPACE / "results" / "physical_measurement_registration" / "measurement_registration_summary.json"
PLANNER_CONFIG = WORKSPACE / "config" / "planner_baseline.json"
GRID = WORKSPACE / "results" / "path_planning_baseline" / "binary_occupancy_grid.json"
OUTPUT = WORKSPACE / "results" / "navigation_obstacle_adapter"
PROTECTED_BASELINE = {
    "file_count": 467,
    "digest_sha256": "cc27887f2f5be0b762500545c86e078268b520aaec5db1007acadd37158e3f32",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_protected():
    later_results = (
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
        PLANNER_CONFIG,
        WORKSPACE / "config" / "physical_obstacle_geometry.json",
        WORKSPACE / "config" / "measured_object_geometry_registry.json",
        WORKSPACE / "src" / "vlm_reposition" / "geometry" / "legacy.py",
        WORKSPACE / "src" / "vlm_reposition" / "world" / "object_footprint.py",
        WORKSPACE / "src" / "vlm_reposition" / "world" / "measurement_registry.py",
        Path("/home/binci/qwen_vision_test/homography_12pt_undistorted.json"),
        Path("/home/binci/camera_calibration_1920x1080/calibration_result.json"),
    ]
    records = [(str(path), sha256_file(path)) for path in files]
    digest = hashlib.sha256("".join(f"{digest}  {path}\n" for path, digest in records).encode()).hexdigest()
    return {"file_count": len(records), "digest_sha256": digest}


def load_representative_observation():
    summary = read_json(STEP22D / "step22d_summary.json")
    parsed = read_json(STEP22D / "parsed_result.json")
    association = read_json(STEP22D / "association_result.json")
    metric = read_json(STEP22D / "metric_result.json")
    semantic = parsed["semantic_result"]
    if (
        summary["step_result"] != "PASS"
        or summary["image_id"] != "image_014_photo_08"
        or summary["track_id"] != "track_039"
        or parsed["status"] != "ACCEPTED"
        or semantic["track_id"] != summary["track_id"]
        or association["status"] != summary["association_status"]
        or metric["world_xy"] != summary["world_xy"]
    ):
        raise ValueError("STEP 22D representative evidence mismatch")
    return {
        "image_id": summary["image_id"],
        "track_id": summary["track_id"],
        "semantic": {
            "label": semantic["visible_label"],
            "type": semantic["semantic_type"],
            "status": semantic["status"],
        },
        "association": {"status": association["status"], "meaning": association["meaning"]},
        "metric": metric,
        "timestamp": {"status": "NOT_AVAILABLE"},
        "provenance": {
            "source_step": "STEP_22D",
            "source_directory": str(STEP22D),
            "metric_result_sha256": sha256_file(STEP22D / "metric_result.json"),
            "geometry_provenance": metric["geometry_provenance"],
            "world_xy_recomputed": False,
            "qwen_inference_runs": 0,
        },
    }


def planner_contract_audit(config, grid):
    return {
        "schema_version": "0.1",
        "step15": {
            "grid_resolution_m": config["grid_resolution_m"],
            "grid_resolution_matches_artifact": config["grid_resolution_m"] == grid["resolution_m"],
            "coordinate_frame": grid["coordinate_frame"],
            "occupied_cell_representation": "BINARY_GRID",
            "point_projection_helper": "vlm_reposition.planning.metric_to_grid",
            "robot_footprint_handling": "CONTINUOUS_COLLISION_GEOMETRY_BEFORE_GRID_DISCRETIZATION",
            "additional_inflation_m": grid["additional_inflation_m"],
            "safety_cost_radius_m": grid["safety_cost_radius_m"],
        },
        "step17": {
            "accepted_geometry": "DEVELOPMENT_SYNTHETIC_AXIS_ALIGNED_RECTANGLE",
            "point_observation_directly_accepted": False,
            "adapter_output_injected": False,
        },
        "step18": {"dependency": "STEP17_RECTANGLE_COLLISION_WORLD_AND_GRID", "called": False},
        "step19": {
            "identity_dependency": "OBSTACLE_ID_PLUS_OBSERVATION_INDEX_AND_LOGICAL_TIME",
            "track_id_bridge_defined": False,
            "called": False,
        },
        "step25n_boundary": "A separately approved bridge must map this observation into the STEP17 contract.",
    }


def main():
    protected_before = snapshot_protected()
    if protected_before != PROTECTED_BASELINE:
        raise RuntimeError(f"protected STEP 12-23M artifact baseline mismatch: {protected_before}")
    physical = read_json(STEP23M)
    if physical["measurement_input_status"] != "WAITING_FOR_USER_MEASUREMENT" or physical["representative_physical_geometry"] != "NOT_READY":
        raise RuntimeError("STEP 23M branch status changed")

    input_observation = load_representative_observation()
    config, grid = read_json(PLANNER_CONFIG), read_json(GRID)
    output = build_navigation_obstacle_observation(input_observation, config, grid)
    audit = planner_contract_audit(config, grid)
    protected_after = snapshot_protected()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "input_observation.json", input_observation)
    write_json(OUTPUT / "planner_contract_audit.json", audit)
    write_json(OUTPUT / "navigation_obstacle_observation.json", output)
    write_json(OUTPUT / "adapter_validation.json", {
        "status": "PASS",
        "gates": {
            "semantic_visible": True,
            "candidate_associated": True,
            "metric_available": True,
            "frame_valid": True,
            "workspace_inside": True,
            "world_xy_finite": True,
            "provenance_allowed": True,
        },
        "physical_measurement_required": False,
        "physical_size_inferred": False,
        "safety_margin_added": False,
        "grid_projection_deterministic": True,
    })
    summary = {
        "schema_version": "0.1",
        "step_result": "PASS",
        "software": "PASS",
        "representative_observation": output["observation_id"],
        "navigation_observation": "AVAILABLE",
        "navigation_role": output["navigation_role"],
        "navigation_representation": output["navigation_representation"]["type"],
        "navigation_geometry": "NAVIGATION_APPROXIMATION",
        "reference_point_world": output["reference_point_world"],
        "grid_index": output["navigation_representation"]["grid_index"],
        "physical_geometry": "NOT_READY",
        "step23m_measurement_input_status": physical["measurement_input_status"],
        "qwen_inference_runs": 0,
        "homography_recomputation": "NOT_RUN",
        "planner_injection": "NOT_RUN",
        "step17_call": "NOT_RUN",
        "step18_call": "NOT_RUN",
        "step19_call": "NOT_RUN",
        "astar": "NOT_RUN",
        "costmap": "NOT_RUN",
        "ros_nav2": "NOT_RUN",
        "protected_artifacts_before": protected_before,
        "protected_artifacts_after": protected_after,
        "protected_artifacts_unchanged": protected_before == protected_after,
        "recommended_next_step": "STEP_25N_APPROXIMATE_NAVIGATION_OBSTACLE_TO_EXISTING_DYNAMIC_WORLD_INTEGRATION",
    }
    if not summary["protected_artifacts_unchanged"]:
        raise RuntimeError("protected STEP 12-23M artifacts changed")
    write_json(OUTPUT / "step24n_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
