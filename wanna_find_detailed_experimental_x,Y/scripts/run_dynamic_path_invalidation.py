#!/usr/bin/env python3
"""Run offline dynamic-obstacle events against one unchanged current path."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.planning import (
    apply_obstacle_event,
    build_dynamic_collision_world,
    build_grid_map,
    current_path_metrics,
    initial_obstacle_state,
    load_dynamic_config,
    load_planner_config,
    plan_path,
    validate_current_path,
)
from vlm_reposition.world import load_physical_geometry, load_zones


DEFAULT_SCENARIOS = WORKSPACE / "config" / "dynamic_obstacle_scenarios.json"
DEFAULT_PLANNER = WORKSPACE / "config" / "planner_baseline.json"
DEFAULT_ZONES = WORKSPACE / "config" / "zones.json"
DEFAULT_PHYSICAL = WORKSPACE / "config" / "physical_geometry.json"
DEFAULT_READINESS = WORKSPACE / "results" / "physical_geometry" / "physical_geometry_readiness.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "dynamic_path_invalidation"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def timed(function, *args):
    started = perf_counter()
    value = function(*args)
    return value, (perf_counter() - started) * 1000.0


def grid_summary(grid):
    return {
        "resolution_m": grid["resolution_m"],
        "rows": grid["rows"],
        "cols": grid["cols"],
        "free_cell_count": grid["free_cell_count"],
        "occupied_cell_count": grid["occupied_cell_count"],
        "additional_inflation_m": grid["additional_inflation_m"],
        "occupancy_sha256": digest(grid["occupancy"]),
        "safety_risk_sha256": digest(grid["safety_risk"]),
    }


def planner_config_for_scenario(base, scenario):
    config = deepcopy(base)
    config["development_start"] = deepcopy(scenario["initial_start"])
    config["development_goal"] = deepcopy(scenario["initial_goal"])
    return config


def revision_record(state, world, grid, validation, metrics):
    return {
        "world_revision": state["world_revision"],
        "obstacle_state": deepcopy(state["obstacles"]),
        "collision_obstacles": deepcopy(world["static_obstacles"]),
        "grid": grid_summary(grid),
        "path_validation": deepcopy(validation),
        "current_path_metrics": deepcopy(metrics),
    }


def run_scenario(config, scenario, zones, physical, readiness, base_planner):
    planner_config = planner_config_for_scenario(base_planner, scenario)
    state = initial_obstacle_state(scenario)
    world, initial_world_ms = timed(build_dynamic_collision_world, config, state, zones, physical, readiness)
    grid, initial_grid_ms = timed(build_grid_map, world, planner_config)
    current_path = plan_path(
        world, grid, planner_config, config["current_path_source_planner"],
        scenario["initial_start"], scenario["initial_goal"],
    )
    if current_path["status"] != "PATH_FOUND":
        raise RuntimeError(f"{scenario['scenario_id']} initial current path was not found")
    original_grid = deepcopy(current_path["path_grid"])
    original_world = deepcopy(current_path["path_world"])
    validation, initial_validation_ms = timed(
        validate_current_path, original_grid, original_world, world, grid
    )
    if validation["path_state"] != "VALID":
        raise RuntimeError(f"{scenario['scenario_id']} initial current path is not valid")
    metrics = current_path_metrics(original_grid, original_world, world, grid, "VALID")
    revisions = [revision_record(state, world, grid, validation, metrics)]
    contexts = [(world, grid, validation)]
    events = []

    for event in scenario["events"]:
        before_state = state
        before_validation = validation
        before_metrics = metrics
        before_grid = grid
        state = apply_obstacle_event(state, event)
        world, world_ms = timed(build_dynamic_collision_world, config, state, zones, physical, readiness)
        grid, grid_ms = timed(build_grid_map, world, planner_config)
        validation, validation_ms = timed(
            validate_current_path, original_grid, original_world, world, grid
        )
        metrics = current_path_metrics(original_grid, original_world, world, grid, validation["path_state"])
        if validation["path_state"] != event["expected_path_state"]:
            raise RuntimeError(f"{event['event_id']} expected {event['expected_path_state']} but got {validation['path_state']}")
        events.append({
            "event_id": event["event_id"],
            "event_index": event["event_index"],
            "operation": event["operation"],
            "obstacle_id": event["obstacle"]["id"],
            "world_revision_before": before_state["world_revision"],
            "world_revision_after": state["world_revision"],
            "path_state_before": before_validation["path_state"],
            "path_state_after": validation["path_state"],
            "reason": validation["reason"],
            "blocking_obstacle_ids": validation["blocking_obstacle_ids"],
            "first_blocked_waypoint_index": validation["first_blocked_waypoint_index"],
            "first_blocked_edge_index": validation["first_blocked_edge_index"],
            "replan_required": validation["replan_required"],
            "minimum_clearance_before_m": before_metrics["minimum_clearance_m"],
            "minimum_clearance_after_m": metrics["minimum_clearance_m"],
            "accumulated_safety_risk_before": before_metrics["accumulated_safety_risk"],
            "accumulated_safety_risk_after": metrics["accumulated_safety_risk"],
            "accumulated_safety_risk_after_status": metrics["accumulated_safety_risk_status"],
            "occupancy_sha256_before": grid_summary(before_grid)["occupancy_sha256"],
            "occupancy_sha256_after": grid_summary(grid)["occupancy_sha256"],
            "safety_risk_sha256_before": grid_summary(before_grid)["safety_risk_sha256"],
            "safety_risk_sha256_after": grid_summary(grid)["safety_risk_sha256"],
            "collision_world_rebuild_time_ms": world_ms,
            "grid_rebuild_time_ms": grid_ms,
            "path_validation_time_ms": validation_ms,
            "performance_metric_status": "VOLATILE_OBSERVATION_NOT_BENCHMARK",
        })
        revisions.append(revision_record(state, world, grid, validation, metrics))
        contexts.append((world, grid, validation))

    unchanged = original_grid == current_path["path_grid"] and original_world == current_path["path_world"]
    if not unchanged:
        raise RuntimeError("current path changed during event processing")
    return {
        "scenario_id": scenario["scenario_id"],
        "description": scenario["description"],
        "result": "PASS",
        "geometry_status": config["geometry_status"],
        "current_path_source_planner": config["current_path_source_planner"],
        "initial_planner_run_count": 1,
        "post_event_planner_run_count": 0,
        "automatic_replanning": "NOT_RUN",
        "current_path_unchanged_across_events": unchanged,
        "current_path_sha256": digest({"path_grid": original_grid, "path_world": original_world}),
        "current_path": current_path,
        "initial_performance": {
            "collision_world_rebuild_time_ms": initial_world_ms,
            "grid_rebuild_time_ms": initial_grid_ms,
            "path_validation_time_ms": initial_validation_ms,
            "performance_metric_status": "VOLATILE_OBSERVATION_NOT_BENCHMARK",
        },
        "events": events,
        "revisions": revisions,
    }, contexts


def semantic_result(value):
    if isinstance(value, dict):
        return {
            key: semantic_result(item)
            for key, item in value.items()
            if not key.endswith("_time_ms") and key != "planning_time_ms"
        }
    if isinstance(value, list):
        return [semantic_result(item) for item in value]
    return value


def draw_revision(scenario_result, revision, world, validation):
    image = np.full((880, 960, 3), 255, dtype=np.uint8)
    origin, scale = (80, 800), 400.0

    def pixel(point):
        return int(origin[0] + point[0] * scale), int(origin[1] - point[1] * scale)

    def rectangle(bounds):
        return np.asarray([
            pixel((bounds["x_min"], bounds["y_min"])), pixel((bounds["x_max"], bounds["y_min"])),
            pixel((bounds["x_max"], bounds["y_max"])), pixel((bounds["x_min"], bounds["y_max"])),
        ], dtype=np.int32)

    cv2.polylines(image, [rectangle(world["workspace"]["raw_bounds"])], True, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.polylines(image, [rectangle(world["workspace"]["valid_robot_center_bounds"])], True, (0, 150, 0), 2, cv2.LINE_AA)
    overlay = image.copy()
    for obstacle in world["static_obstacles"]:
        color = (180, 220, 255) if obstacle["state_role"] == "DYNAMIC_OBSTACLE" else (215, 215, 215)
        cv2.fillPoly(overlay, [rectangle(obstacle["collision_bounds"])], color)
    image = cv2.addWeighted(overlay, 0.62, image, 0.38, 0)
    for obstacle in world["static_obstacles"]:
        color = (0, 0, 210) if obstacle["state_role"] == "DYNAMIC_OBSTACLE" else (90, 90, 90)
        cv2.polylines(image, [rectangle(obstacle["collision_bounds"])], True, color, 3, cv2.LINE_AA)
        cv2.polylines(image, [rectangle(obstacle["raw_bounds"])], True, (0, 140, 255), 2, cv2.LINE_AA)
    path = scenario_result["current_path"]["path_world"]
    cv2.polylines(image, [np.asarray([pixel(point) for point in path], dtype=np.int32)], False, (180, 20, 180), 3, cv2.LINE_AA)
    start, goal = pixel(path[0]), pixel(path[-1])
    cv2.circle(image, start, 7, (0, 160, 0), -1, cv2.LINE_AA)
    cv2.circle(image, goal, 7, (0, 0, 0), -1, cv2.LINE_AA)
    waypoint_index = validation["first_blocked_waypoint_index"]
    edge_index = validation["first_blocked_edge_index"]
    if waypoint_index is not None:
        cv2.drawMarker(image, pixel(path[waypoint_index]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 4)
    elif edge_index is not None:
        a, b = path[edge_index], path[edge_index + 1]
        cv2.drawMarker(image, pixel(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 4)
    state_color = (0, 140, 0) if validation["path_state"] == "VALID" else (0, 0, 210)
    cv2.putText(image, f"{scenario_result['scenario_id']} revision {revision}", (575, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, validation["path_state"], (575, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2, cv2.LINE_AA)
    cv2.putText(image, f"reason: {validation['reason'] or 'NONE'}", (575, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "MAGENTA: unchanged path", (575, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 20, 180), 1, cv2.LINE_AA)
    cv2.putText(image, "RED fill/line: dynamic expanded", (575, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 0, 190), 1, cv2.LINE_AA)
    cv2.putText(image, "GRAY: static expanded", (575, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(image, "ORANGE: raw obstacle", (575, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 120, 220), 1, cv2.LINE_AA)
    cv2.putText(image, "DEVELOPMENT_SYNTHETIC", (575, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "X/Y [m]", (80, 840), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def write_visualizations(directory, outputs, contexts):
    paths = []
    for scenario_id, result in outputs.items():
        for revision, (world, _, validation) in enumerate(contexts[scenario_id]):
            path = directory / f"{scenario_id}_revision_{revision}.png"
            if not cv2.imwrite(str(path), draw_revision(result, revision, world, validation)):
                raise RuntimeError(f"could not write {path}")
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--planner", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES)
    parser.add_argument("--physical", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    config = load_dynamic_config(args.scenarios)
    planner = load_planner_config(args.planner)
    zones = load_zones(args.zones)
    physical = load_physical_geometry(args.physical)
    readiness = read_json(args.readiness)["planning_geometry_readiness"]
    outputs, contexts, deterministic = {}, {}, {}
    for scenario in config["scenarios"]:
        result, local_contexts = run_scenario(config, scenario, zones, physical, readiness, planner)
        repeated, _ = run_scenario(config, scenario, zones, physical, readiness, planner)
        equal = semantic_result(result) == semantic_result(repeated)
        if not equal:
            raise RuntimeError(f"{scenario['scenario_id']} event sequence is not deterministic")
        outputs[scenario["scenario_id"]] = result
        contexts[scenario["scenario_id"]] = local_contexts
        deterministic[scenario["scenario_id"]] = equal

    scenario_dir = args.output_dir / "scenario_results"
    visualization_dir = args.output_dir / "visualizations"
    scenario_dir.mkdir(parents=True)
    visualization_dir.mkdir()
    for scenario_id, result in outputs.items():
        write_json(scenario_dir / f"{scenario_id}.json", result)
    visualization_paths = write_visualizations(visualization_dir, outputs, contexts)

    event_results = [event for result in outputs.values() for event in result["events"]]
    summary = {
        "result": "PASS",
        "verification_type": "Offline Dynamic State Update + Current Path Invalidation",
        "sources": {
            "dynamic_scenarios": str(args.scenarios.resolve()),
            "planner_config": str(args.planner.resolve()),
        },
        "scenario_count": len(outputs),
        "event_count": len(event_results),
        "initial_planner_run_count": len(outputs),
        "post_event_planner_run_count": 0,
        "expected_result_match_count": len(event_results),
        "final_path_states": {key: value["events"][-1]["path_state_after"] for key, value in outputs.items()},
        "deterministic_event_sequences": deterministic,
        "current_paths_unchanged": all(value["current_path_unchanged_across_events"] for value in outputs.values()),
        "hard_invalidation_policy": "COLLISION_ONLY",
        "soft_safety_cost_change_policy": "DOES_NOT_INVALIDATE_PATH",
        "collision_world_source": "RAW_METRIC_OBSTACLE_STATE",
        "full_grid_rebuild_after_each_event": True,
        "additional_inflation_m": 0.0,
        "performance_metric_status": "VOLATILE_OBSERVATION_NOT_BENCHMARK",
        "automatic_replanning": "NOT_RUN",
        "live_camera": "NOT_RUN",
        "aruco": "NOT_RUN",
        "real_robot_execution": "NOT_RUN",
        "manipulation": "NOT_RUN",
        "visualizations": [str(path.resolve()) for path in visualization_paths],
    }
    write_json(args.output_dir / "dynamic_path_invalidation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
