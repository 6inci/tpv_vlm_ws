#!/usr/bin/env python3
"""Run deterministic offline automatic-replanning scenarios."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from run_dynamic_path_invalidation import digest, write_json
from vlm_reposition.planning import (
    build_dynamic_collision_world,
    build_grid_map,
    create_active_path,
    current_path_metrics,
    evaluate_event_and_replan,
    initial_replanning_state,
    load_planner_config,
    load_replanning_config,
    plan_path,
    validate_current_path,
    validate_replanned_path,
)
from vlm_reposition.world import load_physical_geometry, load_zones


DEFAULT_SCENARIOS = WORKSPACE / "config" / "automatic_replanning_scenarios.json"
DEFAULT_PLANNER = WORKSPACE / "config" / "planner_baseline.json"
DEFAULT_ZONES = WORKSPACE / "config" / "zones.json"
DEFAULT_PHYSICAL = WORKSPACE / "config" / "physical_geometry.json"
DEFAULT_READINESS = WORKSPACE / "results" / "physical_geometry" / "physical_geometry_readiness.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "automatic_replanning"


def semantic_result(value):
    if isinstance(value, dict):
        return {
            key: semantic_result(item)
            for key, item in value.items()
            if not key.endswith("_ms") and key != "planning_time_ms"
        }
    if isinstance(value, list):
        return [semantic_result(item) for item in value]
    return value


def read_json(path):
    return json.loads(Path(path).read_text())


def grid_record(grid, world_revision):
    return {
        "world_revision": world_revision,
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
    config["development_goal"] = deepcopy(scenario["goal"])
    return config


def run_scenario(config, scenario, zones, physical, readiness, base_planner):
    planner_config = planner_config_for_scenario(base_planner, scenario)
    state = initial_replanning_state(scenario)
    world = build_dynamic_collision_world(config, state, zones, physical, readiness)
    grid = build_grid_map(world, planner_config)
    planned = plan_path(
        world, grid, planner_config, config["planner"], scenario["initial_start"], scenario["goal"]
    )
    if planned["status"] != "PATH_FOUND":
        raise RuntimeError(f"{scenario['scenario_id']} initial path was not found")
    initial_validation = validate_replanned_path(
        planned,
        {"x": planned["path_world"][0][0], "y": planned["path_world"][0][1]},
        scenario["goal"], world, grid,
    )
    if initial_validation["status"] != "VALID":
        raise RuntimeError(f"{scenario['scenario_id']} initial path validation failed")
    active = create_active_path(planned, scenario["scenario_id"], 0, 0, scenario["goal"])
    initial_active = deepcopy(active)
    history = []
    progress_by_revision = {}
    events = []
    revisions = [{
        "world_revision": 0,
        "obstacle_state": deepcopy(state["obstacles"]),
        "grid": grid_record(grid, 0),
    }]
    contexts = [{"world": world, "old_active": deepcopy(active), "new_active": None, "event": None}]

    for event in scenario["events"]:
        if active is None:
            raise RuntimeError(f"{scenario['scenario_id']} has an event after STOP")
        old_active = deepcopy(active)
        index = event["robot_waypoint_index"]
        old_remaining_grid = old_active["path_grid"][index:]
        old_remaining_world = old_active["path_world"][index:]
        before_validation = validate_current_path(old_remaining_grid, old_remaining_world, world, grid)
        before_metrics = current_path_metrics(
            old_remaining_grid, old_remaining_world, world, grid, before_validation["path_state"]
        )
        previous_index = progress_by_revision.get(old_active["path_revision"])
        state, world, grid, active, history, result = evaluate_event_and_replan(
            config, scenario, planner_config, zones, physical, readiness,
            state, old_active, history, event, previous_index,
        )
        progress_by_revision[old_active["path_revision"]] = index
        result.update({
            "before_event_path_state": before_validation["path_state"],
            "minimum_clearance_before_event_m": before_metrics["minimum_clearance_m"],
            "accumulated_safety_risk_before_event": before_metrics["accumulated_safety_risk"],
            "grid": grid_record(grid, state["world_revision"]),
        })
        actual_revision = active["path_revision"] if active is not None else old_active["path_revision"]
        expected = (
            result["old_path_state"] == event["expected_old_path_state"]
            and result["replanning"]["status"] == event["expected_replan_status"]
            and result["system_action"] == event["expected_system_action"]
            and actual_revision == event["expected_active_path_revision"]
        )
        if not expected:
            raise RuntimeError(f"{event['event_id']} did not match its expected state/action/revision")
        events.append(result)
        revisions.append({
            "world_revision": state["world_revision"],
            "obstacle_state": deepcopy(state["obstacles"]),
            "grid": grid_record(grid, state["world_revision"]),
        })
        contexts.append({
            "world": world,
            "old_active": old_active,
            "new_active": deepcopy(active),
            "event": result,
        })

    path_records = deepcopy(history)
    if active is not None:
        path_records.append(deepcopy(active))
    return {
        "scenario_id": scenario["scenario_id"],
        "description": scenario["description"],
        "result": "PASS",
        "geometry_status": config["geometry_status"],
        "planner": config["planner"],
        "initial_planner_run_count": 1,
        "planner_runs_after_events": sum(item["planner_runs_after_event"] for item in events),
        "initial_active_path": initial_active,
        "initial_path_validation": initial_validation,
        "events": events,
        "path_records": path_records,
        "final_active_path": deepcopy(active),
        "world_revisions": revisions,
        "live_perception": "NOT_RUN",
        "robot_execution": "NOT_RUN",
    }, contexts


def draw_revision(scenario_result, context, revision):
    image = np.full((900, 1050, 3), 255, dtype=np.uint8)
    origin, scale = (80, 820), 400.0

    def pixel(point):
        return int(origin[0] + point[0] * scale), int(origin[1] - point[1] * scale)

    def rectangle(bounds):
        return np.asarray([
            pixel((bounds["x_min"], bounds["y_min"])), pixel((bounds["x_max"], bounds["y_min"])),
            pixel((bounds["x_max"], bounds["y_max"])), pixel((bounds["x_min"], bounds["y_max"])),
        ], dtype=np.int32)

    world = context["world"]
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

    old_path = context["old_active"]["path_world"]
    event = context["event"]
    if event is None:
        cv2.polylines(image, [np.asarray([pixel(point) for point in old_path], dtype=np.int32)], False, (220, 80, 20), 3, cv2.LINE_AA)
        status, action = "INITIAL ACTIVE", "NONE"
        current = old_path[0]
    else:
        index = event["current_robot_index_on_old_path"]
        prefix, remaining = old_path[:index + 1], old_path[index:]
        cv2.polylines(image, [np.asarray([pixel(point) for point in prefix], dtype=np.int32)], False, (0, 150, 0), 4, cv2.LINE_AA)
        cv2.polylines(image, [np.asarray([pixel(point) for point in remaining], dtype=np.int32)], False, (180, 20, 180), 3, cv2.LINE_AA)
        current = [event["current_robot_pose"]["x"], event["current_robot_pose"]["y"]]
        if context["new_active"] is not None and event["system_action"] == "SWITCH_TO_REPLANNED_PATH":
            points = context["new_active"]["path_world"]
            cv2.polylines(image, [np.asarray([pixel(point) for point in points], dtype=np.int32)], False, (220, 80, 20), 4, cv2.LINE_AA)
        status, action = event["replanning"]["status"], event["system_action"]
    cv2.drawMarker(image, pixel(current), (0, 120, 0), cv2.MARKER_DIAMOND, 18, 3)
    goal = scenario_result["initial_active_path"]["goal_pose"]["cell_center_m"]
    cv2.circle(image, pixel(goal), 8, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(image, f"{scenario_result['scenario_id']} revision {revision}", (575, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, status, (575, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 2, cv2.LINE_AA)
    cv2.putText(image, action, (575, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "GREEN: Executed Prefix", (575, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "MAGENTA: Old Remaining Path", (575, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 20, 180), 1, cv2.LINE_AA)
    cv2.putText(image, "BLUE: New Active Path", (575, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 80, 20), 1, cv2.LINE_AA)
    cv2.putText(image, "DIAMOND: Current Robot Pose", (575, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 100, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "BLACK DOT: Goal", (575, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "RED/ORANGE: Blocking/Raw Obstacle", (575, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 180), 1, cv2.LINE_AA)
    cv2.putText(image, "DEVELOPMENT_SYNTHETIC", (575, 820), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def write_visualizations(directory, outputs, contexts):
    paths = []
    for scenario_id, result in outputs.items():
        for revision, context in enumerate(contexts[scenario_id]):
            path = directory / f"{scenario_id}_revision_{revision}.png"
            if not cv2.imwrite(str(path), draw_revision(result, context, revision)):
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

    config = load_replanning_config(args.scenarios)
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
            raise RuntimeError(f"{scenario['scenario_id']} is not semantically deterministic")
        outputs[scenario["scenario_id"]] = result
        contexts[scenario["scenario_id"]] = local_contexts
        deterministic[scenario["scenario_id"]] = equal

    scenario_dir = args.output_dir / "scenario_results"
    visualization_dir = args.output_dir / "visualizations"
    scenario_dir.mkdir(parents=True)
    visualization_dir.mkdir()
    names = {
        "R1": "R1_valid_keep.json", "R2": "R2_add_blocker_replan.json",
        "R3": "R3_move_blocker_replan.json", "R4": "R4_no_path_stop.json",
        "R5": "R5_soft_cost_keep.json", "R6": "R6_multiple_replans.json",
        "R7": "R7_remove_after_replan.json",
    }
    for scenario_id, result in outputs.items():
        write_json(scenario_dir / names[scenario_id], result)
    visualization_paths = write_visualizations(visualization_dir, outputs, contexts)

    events = [event for result in outputs.values() for event in result["events"]]
    summary = {
        "result": "PASS",
        "verification_type": "Event-Driven Offline Automatic Replanning Baseline",
        "sources": {"scenarios": str(args.scenarios.resolve()), "planner": str(args.planner.resolve())},
        "scenario_count": len(outputs),
        "event_count": len(events),
        "initial_planner_run_count": len(outputs),
        "post_event_planner_run_count": sum(item["planner_runs_after_event"] for item in events),
        "replan_success_count": sum(item["replanning"]["status"] == "REPLAN_SUCCESS" for item in events),
        "replan_failed_count": sum(item["replanning"]["status"] == "REPLAN_FAILED" for item in events),
        "valid_keep_count": sum(item["system_action"] == "CONTINUE_CURRENT_PATH" for item in events),
        "stop_required_count": sum(item["stop_required"] for item in events),
        "deterministic_scenarios": deterministic,
        "planner": config["planner"],
        "replan_start_policy": "CURRENT_ACTIVE_PATH_WAYPOINT",
        "goal_policy": "RETAIN_ORIGINAL_MISSION_GOAL",
        "trigger_policy": "HARD_PATH_INVALIDATION_ONLY",
        "valid_path_policy": "KEEP_CURRENT_ACTIVE_PATH",
        "removed_blocker_policy": "NO_AUTOMATIC_OLD_PATH_RESTORATION",
        "partial_path_activation": "FORBIDDEN",
        "additional_inflation_m": 0.0,
        "performance_metric_status": "VOLATILE_OBSERVATION_NOT_BENCHMARK",
        "live_perception": "NOT_RUN",
        "ros_nav2_execution": "NOT_RUN",
        "real_robot_execution": "NOT_RUN",
        "manipulation": "NOT_RUN",
        "visualizations": [str(path.resolve()) for path in visualization_paths],
    }
    write_json(args.output_dir / "automatic_replanning_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
