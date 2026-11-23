"""Deterministic logical-time obstacle persistence policy."""

from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path

from .dynamic_state import (
    _validate_obstacle,
    apply_obstacle_event,
    build_dynamic_collision_world,
    current_path_metrics,
    validate_current_path,
)
from .grid_astar import build_grid_map, plan_path
from .replanning import (
    create_active_path,
    current_robot_pose,
    evaluate_event_and_replan,
    validate_replanned_path,
)


class ObstaclePolicyError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ObstaclePolicyError("NON_FINITE_VALUE", f"{name} must be finite")
    return float(value)


def validate_obstacle_policy_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != "0.1":
        raise ObstaclePolicyError("INVALID_SCHEMA", "policy schema_version must be 0.1")
    required = {
        "coordinate_frame": "LEGACY_WORKSPACE_COORDINATE",
        "unit": "m",
        "mode": "ROBOT_ONLY_TRANSLATIONAL_BASELINE",
        "fixed_yaw_rad": 0.0,
        "geometry_status": "DEVELOPMENT_SYNTHETIC",
        "planner": "SAFETY_COST_ASTAR",
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ObstaclePolicyError("INVALID_SCHEMA", "policy metadata is invalid")
    if not isinstance(config.get("source_detail"), str) or not config["source_detail"]:
        raise ObstaclePolicyError("MISSING_PROVENANCE", "policy provenance is required")
    bounds = config.get("workspace_bounds")
    if not isinstance(bounds, dict):
        raise ObstaclePolicyError("INVALID_SCHEMA", "workspace_bounds is required")
    parsed = {key: _finite(bounds.get(key), f"workspace_bounds.{key}") for key in ("x_min", "x_max", "y_min", "y_max")}
    if parsed["x_min"] >= parsed["x_max"] or parsed["y_min"] >= parsed["y_max"]:
        raise ObstaclePolicyError("INVALID_SCHEMA", "workspace bounds must be ordered")

    parameters = config.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("parameter_status") != "DEVELOPMENT_PARAMETER":
        raise ObstaclePolicyError("INVALID_POLICY_PARAMETERS", "development policy parameters are required")
    interval = _finite(parameters.get("observation_interval_s"), "observation_interval_s")
    threshold = _finite(parameters.get("persistence_threshold_s"), "persistence_threshold_s")
    tolerance = _finite(parameters.get("comparison_tolerance_s"), "comparison_tolerance_s")
    for key in ("presence_confirmation_count", "absence_confirmation_count"):
        value = parameters.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ObstaclePolicyError("INVALID_POLICY_PARAMETERS", f"{key} must be a positive integer")
    if interval <= 0 or threshold < 0 or tolerance < 0:
        raise ObstaclePolicyError("INVALID_POLICY_PARAMETERS", "interval must be positive; threshold/tolerance non-negative")

    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ObstaclePolicyError("INVALID_SCHEMA", "scenarios must be non-empty")
    scenario_ids = set()
    allowed_states = {"MOVING", "STOPPED_FOR_OBSTACLE", "WAITING_FOR_CLEARANCE", "MOVING_REPLANNED_PATH", "STOP_REQUIRED"}
    allowed_actions = {"CONTINUE_CURRENT_PATH", "STOP_AND_WAIT", "RESUME_CURRENT_PATH", "SWITCH_TO_REPLANNED_PATH", "STOP"}
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id") if isinstance(scenario, dict) else None
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in scenario_ids:
            raise ObstaclePolicyError("INVALID_SCENARIO_ID", "scenario IDs must be non-empty and unique")
        scenario_ids.add(scenario_id)
        if not isinstance(scenario.get("description"), str) or not scenario["description"]:
            raise ObstaclePolicyError("INVALID_SCHEMA", f"{scenario_id} description is required")
        for point_name in ("start", "goal"):
            point = scenario.get(point_name)
            if not isinstance(point, dict) or point.get("status") != "DEVELOPMENT_DEFINED":
                raise ObstaclePolicyError("INVALID_ENDPOINT", f"{scenario_id}.{point_name} is invalid")
            _finite(point.get("x"), f"{scenario_id}.{point_name}.x")
            _finite(point.get("y"), f"{scenario_id}.{point_name}.y")
        initial = scenario.get("initial_obstacles")
        catalog = scenario.get("obstacle_catalog")
        if not isinstance(initial, list) or not initial or not isinstance(catalog, list) or not catalog:
            raise ObstaclePolicyError("INVALID_SCHEMA", f"{scenario_id} obstacle state/catalog is invalid")
        initial_ids, catalog_ids = set(), set()
        for obstacle in initial:
            _validate_obstacle(obstacle, config, scenario_id)
            if obstacle["id"] in initial_ids:
                raise ObstaclePolicyError("DUPLICATE_OBSTACLE_ID", f"duplicate initial obstacle {obstacle['id']}")
            initial_ids.add(obstacle["id"])
        for obstacle in catalog:
            _validate_obstacle(obstacle, config, scenario_id)
            if obstacle["state_role"] != "DYNAMIC_OBSTACLE" or obstacle["id"] in catalog_ids or obstacle["id"] in initial_ids:
                raise ObstaclePolicyError("INVALID_OBSTACLE_CATALOG", f"invalid catalog obstacle {obstacle['id']}")
            catalog_ids.add(obstacle["id"])
        observations = scenario.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ObstaclePolicyError("INVALID_SCHEMA", f"{scenario_id}.observations must be non-empty")
        for index, observation in enumerate(observations):
            if not isinstance(observation, dict) or observation.get("observation_index") != index:
                raise ObstaclePolicyError("INVALID_OBSERVATION_ORDER", f"{scenario_id} observation indices must start at zero")
            logical_time = _finite(observation.get("logical_time_s"), f"{scenario_id}.logical_time_s")
            if not math.isclose(logical_time, index * interval, abs_tol=tolerance, rel_tol=0.0):
                raise ObstaclePolicyError("INVALID_LOGICAL_TIME", f"{scenario_id} logical times must follow observation_interval_s")
            progress = observation.get("robot_waypoint_index")
            if isinstance(progress, bool) or not isinstance(progress, int) or progress < 0:
                raise ObstaclePolicyError("INVALID_ROBOT_PROGRESS", f"{scenario_id} robot progress is invalid")
            detected = observation.get("detected_obstacle_ids")
            if not isinstance(detected, list) or len(detected) != len(set(detected)) or any(item not in catalog_ids for item in detected):
                raise ObstaclePolicyError("INVALID_DETECTION_SET", f"{scenario_id} detection IDs are invalid")
            if observation.get("expected_navigation_state") not in allowed_states or observation.get("expected_action") not in allowed_actions:
                raise ObstaclePolicyError("INVALID_EXPECTED_RESULT", f"{scenario_id} expected state/action is invalid")
    required_ids = {f"P{index}" for index in range(1, 11)}
    if not required_ids.issubset(scenario_ids) or not scenario_ids.issubset(required_ids | {"P11"}):
        raise ObstaclePolicyError("INVALID_SCENARIO_SET", "P1-P10 are required and only optional P11 is allowed")
    return config


def load_obstacle_policy_config(path):
    return validate_obstacle_policy_config(json.loads(Path(path).read_text()))


def policy_scenario_by_id(config, scenario_id):
    return next(item for item in config["scenarios"] if item["scenario_id"] == scenario_id)


def _new_tracker(obstacle_id):
    return {
        "obstacle_id": obstacle_id,
        "state": "UNSEEN",
        "confirmed_present": False,
        "consecutive_presence": 0,
        "consecutive_absence": 0,
        "first_detection_time_s": None,
        "first_absence_time_s": None,
        "presence_confirmation_history": [],
        "clear_confirmation_history": [],
    }


def update_confirmed_obstacles(config, scenario, trackers, obstacle_state, observation):
    parameters = config["parameters"]
    logical_time = observation["logical_time_s"]
    detected = set(observation["detected_obstacle_ids"])
    catalog = {item["id"]: item for item in scenario["obstacle_catalog"]}
    changes = []
    for obstacle_id in sorted(catalog):
        tracker = trackers.setdefault(obstacle_id, _new_tracker(obstacle_id))
        if obstacle_id in detected:
            if tracker["consecutive_presence"] == 0:
                tracker["first_detection_time_s"] = logical_time
            tracker["consecutive_presence"] += 1
            tracker["consecutive_absence"] = 0
            tracker["first_absence_time_s"] = None
            if not tracker["confirmed_present"] and tracker["consecutive_presence"] >= parameters["presence_confirmation_count"]:
                tracker["confirmed_present"] = True
                tracker["state"] = "CONFIRMED_PRESENT"
                tracker["presence_confirmation_history"].append({
                    "first_detection_time_s": tracker["first_detection_time_s"],
                    "confirmation_time_s": logical_time,
                    "time_to_confirmation_s": round(logical_time - tracker["first_detection_time_s"], 10),
                })
                event = {"operation": "ADD", "obstacle": deepcopy(catalog[obstacle_id])}
                obstacle_state = apply_obstacle_event(obstacle_state, event)
                changes.append({"operation": "ADD", "obstacle_id": obstacle_id})
            elif tracker["confirmed_present"]:
                tracker["state"] = "CONFIRMED_PRESENT"
            else:
                tracker["state"] = "PRESENCE_PENDING"
        else:
            tracker["consecutive_presence"] = 0
            tracker["first_detection_time_s"] = None
            if tracker["confirmed_present"]:
                if tracker["consecutive_absence"] == 0:
                    tracker["first_absence_time_s"] = logical_time
                tracker["consecutive_absence"] += 1
                if tracker["consecutive_absence"] >= parameters["absence_confirmation_count"]:
                    tracker["confirmed_present"] = False
                    tracker["state"] = "CLEARED"
                    tracker["clear_confirmation_history"].append({
                        "first_absence_time_s": tracker["first_absence_time_s"],
                        "confirmation_time_s": logical_time,
                        "time_to_confirmation_s": round(logical_time - tracker["first_absence_time_s"], 10),
                    })
                    event = {"operation": "REMOVE", "obstacle": {"id": obstacle_id}}
                    obstacle_state = apply_obstacle_event(obstacle_state, event)
                    changes.append({"operation": "REMOVE", "obstacle_id": obstacle_id})
                else:
                    tracker["state"] = "ABSENCE_PENDING"
            else:
                tracker["consecutive_absence"] = 0
                tracker["state"] = "CLEARED" if tracker["clear_confirmation_history"] else "UNSEEN"
    return trackers, obstacle_state, changes


def initialize_policy_runtime(config, scenario, planner_config, zones, physical, readiness):
    obstacle_state = {"world_revision": 0, "obstacles": deepcopy(scenario["initial_obstacles"])}
    world = build_dynamic_collision_world(config, obstacle_state, zones, physical, readiness)
    grid = build_grid_map(world, planner_config)
    planned = plan_path(world, grid, planner_config, config["planner"], scenario["start"], scenario["goal"])
    if planned["status"] != "PATH_FOUND":
        raise RuntimeError(f"{scenario['scenario_id']} initial path was not found")
    validation = validate_replanned_path(
        planned, {"x": planned["path_world"][0][0], "y": planned["path_world"][0][1]},
        scenario["goal"], world, grid,
    )
    if validation["status"] != "VALID":
        raise RuntimeError(f"{scenario['scenario_id']} initial path validation failed")
    active = create_active_path(planned, scenario["scenario_id"], 0, 0, scenario["goal"])
    return {
        "obstacle_state": obstacle_state,
        "world": world,
        "grid": grid,
        "active_path": active,
        "initial_active_path": deepcopy(active),
        "path_history": [],
        "trackers": {item["id"]: _new_tracker(item["id"]) for item in scenario["obstacle_catalog"]},
        "episodes": [],
        "active_episode_index": None,
        "navigation_state": "MOVING",
        "stopped_pose": None,
        "progress_by_revision": {},
        "planner_run_count": 0,
        "stop_count": 0,
        "wait_observation_count": 0,
        "initial_path_validation": validation,
    }


def _episode(runtime):
    index = runtime["active_episode_index"]
    return None if index is None else runtime["episodes"][index]


def evaluate_policy_observation(
    config, scenario, planner_config, zones, physical, readiness, runtime, observation,
    execute_replanning=True,
):
    trackers, obstacle_state, changes = update_confirmed_obstacles(
        config, scenario, runtime["trackers"], runtime["obstacle_state"], observation
    )
    runtime["trackers"] = trackers
    runtime["obstacle_state"] = obstacle_state
    if changes:
        runtime["world"] = build_dynamic_collision_world(config, obstacle_state, zones, physical, readiness)
        runtime["grid"] = build_grid_map(runtime["world"], planner_config)
    active = runtime["active_path"]
    if active is None:
        raise ObstaclePolicyError("MISSION_STOPPED", "observations cannot continue after STOP_REQUIRED")
    index = observation["robot_waypoint_index"]
    episode = _episode(runtime)
    if runtime["navigation_state"] in {"STOPPED_FOR_OBSTACLE", "WAITING_FOR_CLEARANCE"}:
        if runtime["stopped_pose"] is None or index != runtime["stopped_pose"]["waypoint_index"] or active["path_revision"] != runtime["stopped_pose"]["path_revision"]:
            raise ObstaclePolicyError("ROBOT_MOVED_DURING_WAIT", "robot progress must remain fixed while waiting")
    previous = runtime["progress_by_revision"].get(active["path_revision"])
    pose = current_robot_pose(active, index, previous)
    runtime["progress_by_revision"][active["path_revision"]] = index
    remaining_grid = active["path_grid"][index:]
    remaining_world = active["path_world"][index:]
    validation = validate_current_path(remaining_grid, remaining_world, runtime["world"], runtime["grid"])
    metrics = current_path_metrics(remaining_grid, remaining_world, runtime["world"], runtime["grid"], validation["path_state"])
    logical_time = observation["logical_time_s"]
    action = "CONTINUE_CURRENT_PATH"
    replan_result = None
    observation_episode = episode

    if validation["path_state"] == "VALID":
        if episode is not None and not episode["replan_triggered"]:
            episode["classification"] = "TEMPORARY_BLOCKING"
            episode["clear_observation_index"] = observation["observation_index"]
            episode["clear_time_s"] = logical_time
            episode["blocking_duration_s"] = round(logical_time - episode["blocking_since_s"], 10)
            episode["path_revision_after"] = active["path_revision"]
            action = "RESUME_CURRENT_PATH"
            observation_episode = episode
            runtime["active_episode_index"] = None
            runtime["stopped_pose"] = None
            runtime["navigation_state"] = "MOVING" if active["path_revision"] == 0 else "MOVING_REPLANNED_PATH"
        else:
            if episode is not None and episode["replan_triggered"]:
                confirmed = {key for key, value in trackers.items() if value["confirmed_present"]}
                if not confirmed.intersection(episode["blocking_obstacle_ids"]):
                    episode["clear_observation_index"] = observation["observation_index"]
                    episode["clear_time_s"] = logical_time
                    observation_episode = episode
                    runtime["active_episode_index"] = None
            runtime["navigation_state"] = "MOVING" if active["path_revision"] == 0 else "MOVING_REPLANNED_PATH"
    else:
        if episode is None or episode["replan_triggered"]:
            episode = {
                "blocking_episode_id": f"episode_{len(runtime['episodes']) + 1:03d}",
                "start_observation_index": observation["observation_index"],
                "blocking_since_s": logical_time,
                "blocking_obstacle_ids": deepcopy(validation["blocking_obstacle_ids"]),
                "classification": "TEMPORARY_BLOCKING",
                "persistent_classification_time_s": None,
                "clear_observation_index": None,
                "clear_time_s": None,
                "blocking_duration_s": 0.0,
                "replan_triggered": False,
                "replan_ready": False,
                "replan_trigger_time_s": None,
                "replan_status": "NOT_RUN",
                "planner_run_count": 0,
                "stop_count": 1,
                "wait_observation_count": 0,
                "path_revision_before": active["path_revision"],
                "path_revision_after": active["path_revision"],
            }
            runtime["episodes"].append(episode)
            observation_episode = episode
            runtime["active_episode_index"] = len(runtime["episodes"]) - 1
            runtime["stopped_pose"] = deepcopy(pose)
            runtime["stop_count"] += 1
            runtime["navigation_state"] = "STOPPED_FOR_OBSTACLE"
            action = "STOP_AND_WAIT"
        else:
            duration = round(logical_time - episode["blocking_since_s"], 10)
            episode["blocking_duration_s"] = duration
            episode["blocking_obstacle_ids"] = deepcopy(validation["blocking_obstacle_ids"])
            threshold = config["parameters"]["persistence_threshold_s"]
            tolerance = config["parameters"]["comparison_tolerance_s"]
            if duration + tolerance >= threshold and not episode["replan_triggered"] and not episode["replan_ready"]:
                episode["classification"] = "PERSISTENT_BLOCKING"
                episode["persistent_classification_time_s"] = logical_time
                episode["replan_ready"] = True
                episode["replan_trigger_time_s"] = logical_time
                if execute_replanning:
                    episode["replan_triggered"] = True
                    trigger = {
                        "event_id": f"{scenario['scenario_id']}_persistence_{episode['blocking_episode_id']}",
                        "event_index": observation["observation_index"],
                        "operation": "PERSISTENCE_TRIGGER",
                        "robot_waypoint_index": runtime["stopped_pose"]["waypoint_index"],
                    }
                    state, world, grid, new_active, history, replan_result = evaluate_event_and_replan(
                        config, scenario, planner_config, zones, physical, readiness,
                        obstacle_state, active, runtime["path_history"], trigger,
                        runtime["stopped_pose"]["waypoint_index"], apply_obstacle_update=False,
                    )
                    runtime["obstacle_state"], runtime["world"], runtime["grid"] = state, world, grid
                    runtime["active_path"], runtime["path_history"] = new_active, history
                    runtime["planner_run_count"] += replan_result["planner_runs_after_event"]
                    episode["planner_run_count"] = replan_result["planner_runs_after_event"]
                    episode["replan_status"] = replan_result["replanning"]["status"]
                    episode["path_revision_after"] = active["path_revision"] if new_active is None else new_active["path_revision"]
                    observation_episode = episode
                    if new_active is None:
                        runtime["navigation_state"] = "STOP_REQUIRED"
                        action = "STOP"
                    else:
                        runtime["navigation_state"] = "MOVING_REPLANNED_PATH"
                        runtime["stopped_pose"] = None
                        action = "SWITCH_TO_REPLANNED_PATH"
                else:
                    episode["replan_status"] = "REPLAN_REQUIRED"
                    runtime["navigation_state"] = "WAITING_FOR_CLEARANCE"
                    action = "STOP_AND_WAIT"
                    observation_episode = episode
            else:
                episode["wait_observation_count"] += 1
                runtime["wait_observation_count"] += 1
                runtime["navigation_state"] = "WAITING_FOR_CLEARANCE"
                action = "STOP_AND_WAIT"

    confirmed_ids = sorted(key for key, value in trackers.items() if value["confirmed_present"])
    current_episode = observation_episode or _episode(runtime)
    result = {
        "observation_index": observation["observation_index"],
        "logical_time_s": logical_time,
        "raw_detected_obstacle_ids": sorted(observation["detected_obstacle_ids"]),
        "temporal_obstacle_states": {
            key: {
                "state": value["state"],
                "confirmed_present": value["confirmed_present"],
                "consecutive_presence": value["consecutive_presence"],
                "consecutive_absence": value["consecutive_absence"],
            }
            for key, value in sorted(trackers.items())
        },
        "confirmed_obstacle_ids": confirmed_ids,
        "confirmed_world_changed": bool(changes),
        "confirmed_world_changes": changes,
        "world_revision": runtime["obstacle_state"]["world_revision"],
        "path_state": validation["path_state"],
        "path_validation": validation,
        "minimum_clearance_m": metrics["minimum_clearance_m"],
        "accumulated_safety_risk": metrics["accumulated_safety_risk"],
        "current_robot_pose": pose,
        "navigation_state": runtime["navigation_state"],
        "action": action,
        "blocking_episode_id": None if current_episode is None else current_episode["blocking_episode_id"],
        "blocking_since_s": None if current_episode is None else current_episode["blocking_since_s"],
        "blocking_duration_s": None if current_episode is None else current_episode["blocking_duration_s"],
        "classification": None if current_episode is None else current_episode["classification"],
        "planner_runs_this_observation": 0 if replan_result is None else replan_result["planner_runs_after_event"],
        "replan_status": (
            current_episode["replan_status"]
            if replan_result is None and current_episode is not None
            else "NOT_RUN" if replan_result is None
            else replan_result["replanning"]["status"]
        ),
        "replan_result": replan_result,
        "active_path_revision": None if runtime["active_path"] is None else runtime["active_path"]["path_revision"],
    }
    return runtime, result
