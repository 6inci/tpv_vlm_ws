#!/usr/bin/env python3
"""Run deterministic offline obstacle-persistence policy scenarios."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from run_dynamic_path_invalidation import write_json
from vlm_reposition.planning import (
    evaluate_policy_observation,
    initialize_policy_runtime,
    load_obstacle_policy_config,
    load_planner_config,
)
from vlm_reposition.world import load_physical_geometry, load_zones


DEFAULT_POLICY = WORKSPACE / "config" / "dynamic_obstacle_policy.json"
DEFAULT_PLANNER = WORKSPACE / "config" / "planner_baseline.json"
DEFAULT_ZONES = WORKSPACE / "config" / "zones.json"
DEFAULT_PHYSICAL = WORKSPACE / "config" / "physical_geometry.json"
DEFAULT_READINESS = WORKSPACE / "results" / "physical_geometry" / "physical_geometry_readiness.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "dynamic_obstacle_policy"


def read_json(path):
    return json.loads(Path(path).read_text())


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


def planner_config_for_scenario(base, scenario):
    config = deepcopy(base)
    config["development_start"] = deepcopy(scenario["start"])
    config["development_goal"] = deepcopy(scenario["goal"])
    return config


def _episode_metrics(runtime):
    presence = [
        item
        for tracker in runtime["trackers"].values()
        for item in tracker["presence_confirmation_history"]
    ]
    clear = [
        item
        for tracker in runtime["trackers"].values()
        for item in tracker["clear_confirmation_history"]
    ]
    return {
        "time_to_presence_confirmation_s": [item["time_to_confirmation_s"] for item in presence],
        "time_to_clear_confirmation_s": [item["time_to_confirmation_s"] for item in clear],
        "stop_count": runtime["stop_count"],
        "wait_observation_count": runtime["wait_observation_count"],
        "planner_run_count": runtime["planner_run_count"],
    }


def run_scenario(config, scenario, zones, physical, readiness, base_planner):
    planner_config = planner_config_for_scenario(base_planner, scenario)
    runtime = initialize_policy_runtime(config, scenario, planner_config, zones, physical, readiness)
    initial_active = deepcopy(runtime["initial_active_path"])
    observations, blocking_world = [], None
    for observation in scenario["observations"]:
        runtime, result = evaluate_policy_observation(
            config, scenario, planner_config, zones, physical, readiness, runtime, observation
        )
        if result["navigation_state"] != observation["expected_navigation_state"] or result["action"] != observation["expected_action"]:
            raise RuntimeError(f"{scenario['scenario_id']} observation {observation['observation_index']} did not match expectation")
        observations.append(result)
        if blocking_world is None and result["path_state"] == "BLOCKED":
            blocking_world = deepcopy(runtime["world"])

    path_records = deepcopy(runtime["path_history"])
    if runtime["active_path"] is not None:
        path_records.append(deepcopy(runtime["active_path"]))
    result = {
        "scenario_id": scenario["scenario_id"],
        "description": scenario["description"],
        "result": "PASS",
        "temporal_policy": deepcopy(config["parameters"]),
        "initial_active_path": initial_active,
        "initial_path_validation": deepcopy(runtime["initial_path_validation"]),
        "observations": observations,
        "obstacle_trackers": deepcopy(runtime["trackers"]),
        "blocking_episodes": deepcopy(runtime["episodes"]),
        "temporal_metrics": _episode_metrics(runtime),
        "path_records": path_records,
        "final_active_path": deepcopy(runtime["active_path"]),
        "final_confirmed_obstacle_state": deepcopy(runtime["obstacle_state"]),
        "final_navigation_state": runtime["navigation_state"],
        "live_perception": "NOT_RUN",
        "robot_execution": "NOT_RUN",
    }
    context = {
        "blocking_world": blocking_world or deepcopy(runtime["world"]),
        "final_world": deepcopy(runtime["world"]),
    }
    return result, context


def draw_timeline(result):
    rows = result["observations"]
    width, row_height = 1500, 48
    image = np.full((120 + row_height * len(rows), width, 3), 255, dtype=np.uint8)
    columns = [(25, "time"), (120, "raw"), (330, "confirmed"), (535, "path"), (650, "navigation state"), (990, "policy action"), (1290, "replan")]
    cv2.putText(image, f"{result['scenario_id']} temporal policy timeline", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    for x, label in columns:
        cv2.putText(image, label, (x, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 50, 50), 1, cv2.LINE_AA)
    for row_index, item in enumerate(rows):
        y = 120 + row_index * row_height
        if row_index % 2:
            cv2.rectangle(image, (10, y - 29), (width - 10, y + 14), (246, 246, 246), -1)
        values = [
            f"{item['logical_time_s']:.1f}",
            ",".join(item["raw_detected_obstacle_ids"]) or "-",
            ",".join(item["confirmed_obstacle_ids"]) or "-",
            item["path_state"], item["navigation_state"], item["action"], item["replan_status"],
        ]
        for (x, _), value in zip(columns, values):
            color = (0, 0, 190) if value in {"BLOCKED", "STOP_REQUIRED", "STOP"} else (20, 20, 20)
            cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
    return image


def draw_spatial(result, context):
    image = np.full((900, 1050, 3), 255, dtype=np.uint8)
    origin, scale = (80, 820), 400.0

    def pixel(point):
        return int(origin[0] + point[0] * scale), int(origin[1] - point[1] * scale)

    def polygon(bounds):
        return np.asarray([
            pixel((bounds["x_min"], bounds["y_min"])), pixel((bounds["x_max"], bounds["y_min"])),
            pixel((bounds["x_max"], bounds["y_max"])), pixel((bounds["x_min"], bounds["y_max"])),
        ], dtype=np.int32)

    world = context["blocking_world"]
    cv2.polylines(image, [polygon(world["workspace"]["raw_bounds"])], True, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.polylines(image, [polygon(world["workspace"]["valid_robot_center_bounds"])], True, (0, 150, 0), 2, cv2.LINE_AA)
    overlay = image.copy()
    for obstacle in world["static_obstacles"]:
        color = (170, 215, 255) if obstacle["state_role"] == "DYNAMIC_OBSTACLE" else (215, 215, 215)
        cv2.fillPoly(overlay, [polygon(obstacle["collision_bounds"])], color)
    image = cv2.addWeighted(overlay, 0.65, image, 0.35, 0)
    for obstacle in world["static_obstacles"]:
        color = (0, 0, 205) if obstacle["state_role"] == "DYNAMIC_OBSTACLE" else (90, 90, 90)
        cv2.polylines(image, [polygon(obstacle["collision_bounds"])], True, color, 3, cv2.LINE_AA)
        cv2.polylines(image, [polygon(obstacle["raw_bounds"])], True, (0, 140, 255), 2, cv2.LINE_AA)

    initial = result["initial_active_path"]["path_world"]
    stop = next((item for item in result["observations"] if item["navigation_state"] == "STOPPED_FOR_OBSTACLE"), None)
    stop_index = 0 if stop is None else stop["current_robot_pose"]["waypoint_index"]
    cv2.polylines(image, [np.asarray([pixel(point) for point in initial], dtype=np.int32)], False, (180, 20, 180), 3, cv2.LINE_AA)
    cv2.polylines(image, [np.asarray([pixel(point) for point in initial[:stop_index + 1]], dtype=np.int32)], False, (0, 150, 0), 4, cv2.LINE_AA)
    final = result["final_active_path"]
    if final is not None and final["path_revision"] > 0:
        cv2.polylines(image, [np.asarray([pixel(point) for point in final["path_world"]], dtype=np.int32)], False, (220, 80, 20), 4, cv2.LINE_AA)
    pose = initial[stop_index]
    cv2.drawMarker(image, pixel(pose), (0, 120, 0), cv2.MARKER_DIAMOND, 18, 3)
    goal = result["initial_active_path"]["goal_pose"]["cell_center_m"]
    cv2.circle(image, pixel(goal), 8, (0, 0, 0), -1, cv2.LINE_AA)
    final_revision = None if final is None else final["path_revision"]
    cv2.putText(image, f"{result['scenario_id']} policy spatial result", (575, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"final state: {result['final_navigation_state']}", (575, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(image, f"active revision: {final_revision}", (575, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(image, "green executed | magenta original | blue replanned", (575, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    return image


def write_visualizations(directory, outputs, contexts):
    paths = []
    for scenario_id, result in outputs.items():
        path = directory / f"{scenario_id}_timeline.png"
        if not cv2.imwrite(str(path), draw_timeline(result)):
            raise RuntimeError(f"failed to write {path}")
        paths.append(path)
    for scenario_id in ("P1", "P2", "P3", "P10"):
        path = directory / f"{scenario_id}_spatial.png"
        if not cv2.imwrite(str(path), draw_spatial(outputs[scenario_id], contexts[scenario_id])):
            raise RuntimeError(f"failed to write {path}")
        paths.append(path)
    return paths


def build_summary(config, outputs, deterministic, policy_path, planner_path, visualization_paths):
    episodes = [episode for result in outputs.values() for episode in result["blocking_episodes"]]
    observations = [item for result in outputs.values() for item in result["observations"]]
    return {
        "result": "PASS",
        "verification_type": "DEVELOPMENT TEMPORAL POLICY — deterministic offline synthetic simulation",
        "sources": {"policy": str(policy_path.resolve()), "planner": str(planner_path.resolve())},
        "parameters": deepcopy(config["parameters"]),
        "scenario_count": len(outputs),
        "observation_count": len(observations),
        "temporary_episode_count": sum(item["classification"] == "TEMPORARY_BLOCKING" for item in episodes),
        "persistent_episode_count": sum(item["classification"] == "PERSISTENT_BLOCKING" for item in episodes),
        "temporary_resumed_count": sum(item["action"] == "RESUME_CURRENT_PATH" for item in observations),
        "persistent_replanned_count": sum(item["replan_triggered"] for item in episodes),
        "unnecessary_replan_count": sum(item["planner_runs_this_observation"] for item in observations if item["path_state"] == "VALID"),
        "failed_replan_count": sum(item["replan_status"] == "REPLAN_FAILED" for item in episodes),
        "planner_runs": sum(result["temporal_metrics"]["planner_run_count"] for result in outputs.values()),
        "stop_count": sum(result["temporal_metrics"]["stop_count"] for result in outputs.values()),
        "wait_observation_count": sum(result["temporal_metrics"]["wait_observation_count"] for result in outputs.values()),
        "deterministic_scenarios": deterministic,
        "replan_trigger_policy": "CONFIRMED_HARD_BLOCKAGE_DURATION_GTE_THRESHOLD",
        "temporary_clear_policy": "REVALIDATE_AND_RESUME_EXISTING_ACTIVE_PATH",
        "replan_start_policy": "STOPPED_CURRENT_ACTIVE_PATH_WAYPOINT",
        "goal_policy": "RETAIN_ORIGINAL_MISSION_GOAL",
        "confirmed_world_source": "DEBOUNCED_CONFIRMED_OBSTACLES_ONLY",
        "costmap_update_policy": "FULL_REBUILD_ON_CONFIRMED_WORLD_CHANGE",
        "additional_inflation_m": 0.0,
        "stop_semantics": "STOP is an offline navigation-policy decision and not a hardware safety stop.",
        "performance_metric_status": "VOLATILE_OBSERVATION_NOT_BENCHMARK",
        "live_perception": "NOT_RUN",
        "ros_nav2_execution": "NOT_RUN",
        "real_robot_execution": "NOT_RUN",
        "manipulation": "NOT_RUN",
        "visualizations": [str(path.resolve()) for path in visualization_paths],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--planner", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES)
    parser.add_argument("--physical", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    config = load_obstacle_policy_config(args.policy)
    planner = load_planner_config(args.planner)
    zones = load_zones(args.zones)
    physical = load_physical_geometry(args.physical)
    readiness = read_json(args.readiness)["planning_geometry_readiness"]
    outputs, contexts, deterministic = {}, {}, {}
    for scenario in config["scenarios"]:
        result, context = run_scenario(config, scenario, zones, physical, readiness, planner)
        repeated, _ = run_scenario(config, scenario, zones, physical, readiness, planner)
        deterministic[scenario["scenario_id"]] = semantic_result(result) == semantic_result(repeated)
        if not deterministic[scenario["scenario_id"]]:
            raise RuntimeError(f"{scenario['scenario_id']} is not semantically deterministic")
        outputs[scenario["scenario_id"]], contexts[scenario["scenario_id"]] = result, context

    scenario_dir = args.output_dir / "scenario_results"
    visualization_dir = args.output_dir / "visualizations"
    scenario_dir.mkdir(parents=True)
    visualization_dir.mkdir()
    names = {
        "P1": "P1_temporary_clear.json", "P2": "P2_persistent_replan.json",
        "P3": "P3_persistent_no_path.json", "P4": "P4_detection_flicker.json",
        "P5": "P5_persistent_off_path.json", "P6": "P6_soft_risk_only.json",
        "P7": "P7_clear_before_threshold.json", "P8": "P8_threshold_persistent.json",
        "P9": "P9_other_blocker_remains.json", "P10": "P10_clear_after_replan.json",
        "P11": "P11_sequential_episodes.json",
    }
    for scenario_id, result in outputs.items():
        write_json(scenario_dir / names[scenario_id], result)
    visualization_paths = write_visualizations(visualization_dir, outputs, contexts)
    summary = build_summary(config, outputs, deterministic, args.policy, args.planner, visualization_paths)
    write_json(args.output_dir / "dynamic_obstacle_policy_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
