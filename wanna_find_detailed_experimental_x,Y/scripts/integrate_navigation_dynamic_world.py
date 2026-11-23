#!/usr/bin/env python3
"""Register the STEP 24N cell through STEP 17 without path evaluation or planning."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.planning import (
    apply_obstacle_event,
    build_dynamic_collision_world,
    build_grid_map,
    build_step17_dynamic_obstacle,
    dynamic_scenario_by_id,
    initial_obstacle_state,
    load_dynamic_config,
    load_planner_config,
)
from vlm_reposition.world import load_physical_geometry, load_zones


STEP24N = WORKSPACE / "results" / "navigation_obstacle_adapter"
STEP23M = WORKSPACE / "results" / "physical_measurement_registration" / "measurement_registration_summary.json"
DYNAMIC_CONFIG = WORKSPACE / "config" / "dynamic_obstacle_scenarios.json"
PLANNER_CONFIG = WORKSPACE / "config" / "planner_baseline.json"
GRID_ARTIFACT = WORKSPACE / "results" / "path_planning_baseline" / "binary_occupancy_grid.json"
ZONES = WORKSPACE / "config" / "zones.json"
PHYSICAL = WORKSPACE / "config" / "physical_geometry.json"
READINESS = WORKSPACE / "results" / "physical_geometry" / "physical_geometry_readiness.json"
OUTPUT = WORKSPACE / "results" / "navigation_dynamic_world_integration"
PROTECTED_BASELINE = {
    "file_count": 471,
    "digest_sha256": "c484140a6aa97463b953e5f9377490a40d49734f7aba393aa3b401beeb53d95a",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def snapshot_protected():
    later_results = (
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
        DYNAMIC_CONFIG,
        PLANNER_CONFIG,
        WORKSPACE / "config" / "static_obstacles.json",
        WORKSPACE / "config" / "measured_object_geometry_registry.json",
        WORKSPACE / "src" / "vlm_reposition" / "geometry" / "legacy.py",
        Path("/home/binci/qwen_vision_test/homography_12pt_undistorted.json"),
        Path("/home/binci/camera_calibration_1920x1080/calibration_result.json"),
    ]
    records = [(str(path), sha256_file(path)) for path in files]
    aggregate = hashlib.sha256("".join(f"{file_hash}  {path}\n" for path, file_hash in records).encode()).hexdigest()
    return {"file_count": len(records), "digest_sha256": aggregate}


def load_step24n_observation():
    summary = read_json(STEP24N / "step24n_summary.json")
    observation = read_json(STEP24N / "navigation_obstacle_observation.json")
    if (
        summary.get("step_result") != "PASS"
        or summary.get("navigation_observation") != "AVAILABLE"
        or summary.get("representative_observation") != observation.get("observation_id")
        or summary.get("grid_index") != observation.get("navigation_representation", {}).get("grid_index")
        or summary.get("reference_point_world") != observation.get("reference_point_world")
    ):
        raise ValueError("STEP 24N source artifacts are inconsistent")
    return observation


def occupancy_effect(before, after, source_cell):
    changed = [
        [row, col, before["occupancy"][row][col], after["occupancy"][row][col]]
        for row in range(before["rows"])
        for col in range(before["cols"])
        if before["occupancy"][row][col] != after["occupancy"][row][col]
    ]
    row, col = source_cell["row"], source_cell["col"]
    source_before, source_after = before["occupancy"][row][col], after["occupancy"][row][col]
    if source_before == source_after == 1 and changed:
        status = "SOURCE_CELL_UNCHANGED_ALREADY_OCCUPIED_WITH_ADDITIONAL_COLLISION_OCCUPANCY"
    elif not changed:
        status = "UNCHANGED_DUE_TO_EXISTING_OCCUPANCY"
    else:
        status = "OCCUPANCY_CHANGED"
    return {
        "status": status,
        "source_cell": deepcopy(source_cell),
        "source_cell_occupancy_before": source_before,
        "source_cell_occupancy_after": source_after,
        "source_cell_delta": source_after - source_before,
        "changed_cell_count": len(changed),
        "free_to_occupied_count": sum(item[2:] == [0, 1] for item in changed),
        "occupied_to_free_count": sum(item[2:] == [1, 0] for item in changed),
        "changed_cells": changed,
        "occupancy_sha256_before": digest(before["occupancy"]),
        "occupancy_sha256_after": digest(after["occupancy"]),
        "interpretation": "The source cell was already occupied. Any additional cells result from the existing robot-footprint collision transform applied to the exact cell rectangle, not inferred object size.",
    }


def main():
    protected_before = snapshot_protected()
    if protected_before != PROTECTED_BASELINE:
        raise RuntimeError(f"protected STEP 12-24N artifact baseline mismatch: {protected_before}")
    measurement = read_json(STEP23M)
    if measurement["measurement_input_status"] != "WAITING_FOR_USER_MEASUREMENT" or measurement["representative_physical_geometry"] != "NOT_READY":
        raise RuntimeError("STEP 23M physical branch changed")

    observation = load_step24n_observation()
    dynamic_config = load_dynamic_config(DYNAMIC_CONFIG)
    planner_config = load_planner_config(PLANNER_CONFIG)
    baseline_grid = read_json(GRID_ARTIFACT)
    zones = load_zones(ZONES)
    physical = load_physical_geometry(PHYSICAL)
    readiness = read_json(READINESS)["planning_geometry_readiness"]
    scenario = dynamic_scenario_by_id(dynamic_config, "D1")

    obstacle = build_step17_dynamic_obstacle(observation, baseline_grid, dynamic_config)
    event = {
        "event_id": f'step25n_add_{obstacle["id"]}',
        "event_index": 1,
        "event_index_semantics": "DETERMINISTIC_OFFLINE_ORDER_NOT_WALL_CLOCK_TIME",
        "operation": "ADD",
        "obstacle": obstacle,
    }
    state_before = initial_obstacle_state(scenario)
    world_before = build_dynamic_collision_world(dynamic_config, state_before, zones, physical, readiness)
    grid_before = build_grid_map(world_before, planner_config)
    if grid_before["occupancy"] != baseline_grid["occupancy"]:
        raise RuntimeError("STEP 17 initial occupancy does not match the STEP 15 baseline")
    state_after = apply_obstacle_event(state_before, event)
    world_after = build_dynamic_collision_world(dynamic_config, state_after, zones, physical, readiness)
    grid_after = build_grid_map(world_after, planner_config)

    registered = next(item for item in state_after["obstacles"] if item["id"] == obstacle["id"])
    static_before = [item for item in state_before["obstacles"] if item["state_role"] == "STATIC_OBSTACLE"]
    static_after = [item for item in state_after["obstacles"] if item["state_role"] == "STATIC_OBSTACLE"]
    effect = occupancy_effect(grid_before, grid_after, observation["navigation_representation"]["grid_index"])
    protected_after = snapshot_protected()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "input_navigation_observation.json", observation)
    write_json(OUTPUT / "step17_contract_audit.json", {
        "schema_version": "0.1",
        "obstacle_structure": "STATE_ROLE_PLUS_AXIS_ALIGNED_RECTANGLE",
        "required_fields": ["id", "state_role", "geometry_status", "shape", "bounds", "coordinate_frame", "unit", "source_type", "source_detail"],
        "obstacle_id_policy": "NON_EMPTY_UNIQUE_WITHIN_STATE",
        "rectangle_convention": "CLOSED_CONTACT_COLLISION_WITH_ORDERED_METRIC_AABB",
        "operations": ["ADD", "REMOVE", "MOVE"],
        "normal_update_api": "apply_obstacle_event",
        "collision_world_api": "build_dynamic_collision_world",
        "grid_rebuild_api": "build_grid_map",
        "existing_development_synthetic_behavior_preserved": True,
        "navigation_extension": "NAVIGATION_APPROXIMATION_WITH_NAVIGATION_DISCRETIZATION_CELL",
        "step18_dependency": "STEP17_STATE_COLLISION_WORLD_AND_GRID",
        "step19_dependency": "OBSTACLE_ID_OBSERVATION_INDEX_LOGICAL_TIME",
    })
    write_json(OUTPUT / "cell_geometry_bridge.json", {
        "source_reference_point_world": observation["reference_point_world"],
        "source_navigation_representation": observation["navigation_representation"],
        "bridge_geometry": obstacle["bridge_geometry"],
        "cell_bounds_m": obstacle["bounds"],
        "physical_geometry_used": False,
        "arbitrary_expansion_added": False,
    })
    write_json(OUTPUT / "dynamic_obstacle_record.json", registered)
    write_json(OUTPUT / "dynamic_world_before.json", {"obstacle_state": state_before, "collision_world": world_before})
    write_json(OUTPUT / "dynamic_world_after.json", {"obstacle_state": state_after, "collision_world": world_after})
    write_json(OUTPUT / "occupancy_effect.json", effect)
    summary = {
        "schema_version": "0.1",
        "step_result": "PASS",
        "software": "PASS",
        "source_step": "STEP_24N",
        "observation_id": observation["observation_id"],
        "dynamic_obstacle_id": obstacle["id"],
        "dynamic_obstacle": "REGISTERED",
        "world_revision_before": state_before["world_revision"],
        "world_revision_after": state_after["world_revision"],
        "bridge_geometry": "NAVIGATION_DISCRETIZATION_CELL",
        "physical_geometry": "NOT_READY_NOT_USED",
        "step23m_measurement_input_status": measurement["measurement_input_status"],
        "static_obstacles_unchanged": static_before == static_after,
        "source_cell_occupancy_effect": effect["status"],
        "source_cell_delta": effect["source_cell_delta"],
        "overall_changed_cell_count": effect["changed_cell_count"],
        "collision_world_rebuilt": True,
        "occupancy_grid_rebuilt": True,
        "grid_rebuild_scope": "STEP17_STATE_MATERIALIZATION_ONLY",
        "qwen_inference_runs": 0,
        "homography_recomputation": "NOT_RUN",
        "path_validation": "NOT_RUN",
        "step18_call": "NOT_RUN",
        "step19_call": "NOT_RUN",
        "astar": "NOT_RUN",
        "path_planning": "NOT_RUN",
        "replanning": "NOT_RUN",
        "ros_nav2": "NOT_RUN",
        "protected_artifacts_before": protected_before,
        "protected_artifacts_after": protected_after,
        "protected_artifacts_unchanged": protected_before == protected_after,
        "recommended_next_step": "SEPARATE_NAVIGATION_EFFECT_STEP_DYNAMIC_OBSERVATION_TO_EXISTING_PATH_VALIDITY",
    }
    if registered != obstacle or not summary["static_obstacles_unchanged"] or not summary["protected_artifacts_unchanged"]:
        raise RuntimeError("STEP 25N integration evidence failed")
    write_json(OUTPUT / "step25n_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
