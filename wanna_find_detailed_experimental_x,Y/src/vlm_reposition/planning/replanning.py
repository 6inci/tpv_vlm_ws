"""Offline event-driven replanning on top of the STEP 17 dynamic world."""

from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path
from time import perf_counter

from .collision_geometry import check_robot_center
from .dynamic_state import (
    _validate_obstacle,
    apply_obstacle_event,
    build_dynamic_collision_world,
    current_path_metrics,
    validate_current_path,
)
from .grid_astar import build_grid_map, metric_to_grid, path_length_m, plan_path


class ReplanningError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ReplanningError("NON_FINITE_VALUE", f"{name} must be finite")
    return float(value)


def validate_replanning_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != "0.1":
        raise ReplanningError("INVALID_SCHEMA", "replanning schema_version must be 0.1")
    required = {
        "coordinate_frame": "LEGACY_WORKSPACE_COORDINATE",
        "unit": "m",
        "mode": "ROBOT_ONLY_TRANSLATIONAL_BASELINE",
        "fixed_yaw_rad": 0.0,
        "geometry_status": "DEVELOPMENT_SYNTHETIC",
        "planner": "SAFETY_COST_ASTAR",
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ReplanningError("INVALID_SCHEMA", "replanning metadata is invalid")
    if not isinstance(config.get("source_detail"), str) or not config["source_detail"]:
        raise ReplanningError("MISSING_PROVENANCE", "replanning provenance is required")
    bounds = config.get("workspace_bounds")
    if not isinstance(bounds, dict):
        raise ReplanningError("INVALID_SCHEMA", "workspace_bounds is required")
    parsed_bounds = {key: _finite(bounds.get(key), f"workspace_bounds.{key}") for key in ("x_min", "x_max", "y_min", "y_max")}
    if parsed_bounds["x_min"] >= parsed_bounds["x_max"] or parsed_bounds["y_min"] >= parsed_bounds["y_max"]:
        raise ReplanningError("INVALID_SCHEMA", "workspace bounds must be ordered")

    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ReplanningError("INVALID_SCHEMA", "scenarios must be a non-empty list")
    scenario_ids = set()
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id") if isinstance(scenario, dict) else None
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in scenario_ids:
            raise ReplanningError("INVALID_SCENARIO_ID", "scenario IDs must be non-empty and unique")
        scenario_ids.add(scenario_id)
        if not isinstance(scenario.get("description"), str) or not scenario["description"]:
            raise ReplanningError("INVALID_SCHEMA", f"{scenario_id} description is required")
        for point_name in ("initial_start", "goal"):
            point = scenario.get(point_name)
            if not isinstance(point, dict) or point.get("status") != "DEVELOPMENT_DEFINED":
                raise ReplanningError("INVALID_ENDPOINT", f"{scenario_id}.{point_name} is invalid")
            _finite(point.get("x"), f"{scenario_id}.{point_name}.x")
            _finite(point.get("y"), f"{scenario_id}.{point_name}.y")
        obstacles = scenario.get("initial_obstacles")
        if not isinstance(obstacles, list) or not obstacles:
            raise ReplanningError("INVALID_SCHEMA", f"{scenario_id}.initial_obstacles must be non-empty")
        known_ids = set()
        for obstacle in obstacles:
            _validate_obstacle(obstacle, config, scenario_id)
            if obstacle["id"] in known_ids:
                raise ReplanningError("DUPLICATE_OBSTACLE_ID", f"duplicate obstacle ID {obstacle['id']}")
            known_ids.add(obstacle["id"])
        events = scenario.get("events")
        if not isinstance(events, list) or not events:
            raise ReplanningError("INVALID_SCHEMA", f"{scenario_id}.events must be non-empty")
        event_ids = set()
        for expected_index, event in enumerate(events, start=1):
            if not isinstance(event, dict) or event.get("event_index") != expected_index:
                raise ReplanningError("INVALID_EVENT_ORDER", f"{scenario_id} event indices must be contiguous from 1")
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id or event_id in event_ids:
                raise ReplanningError("INVALID_EVENT_ID", f"{scenario_id} event IDs must be non-empty and unique")
            event_ids.add(event_id)
            operation = event.get("operation")
            if operation not in {"ADD", "REMOVE", "MOVE"}:
                raise ReplanningError("UNSUPPORTED_OPERATION", f"unsupported operation {operation}")
            obstacle = event.get("obstacle")
            obstacle_id = obstacle.get("id") if isinstance(obstacle, dict) else None
            if operation == "REMOVE":
                if obstacle_id not in known_ids:
                    raise ReplanningError("UNKNOWN_OBSTACLE_ID", f"cannot REMOVE unknown obstacle {obstacle_id}")
                if not isinstance(obstacle.get("source_type"), str) or not obstacle["source_type"] or not isinstance(obstacle.get("source_detail"), str) or not obstacle["source_detail"]:
                    raise ReplanningError("MISSING_PROVENANCE", f"{event_id} REMOVE provenance is required")
                known_ids.remove(obstacle_id)
            else:
                _validate_obstacle(obstacle, config, event_id)
                if obstacle["state_role"] != "DYNAMIC_OBSTACLE":
                    raise ReplanningError("INVALID_OBSTACLE_ROLE", f"{event_id} must update a dynamic obstacle")
                if operation == "ADD" and obstacle_id in known_ids:
                    raise ReplanningError("DUPLICATE_OBSTACLE_ID", f"cannot ADD existing obstacle {obstacle_id}")
                if operation == "MOVE" and obstacle_id not in known_ids:
                    raise ReplanningError("UNKNOWN_OBSTACLE_ID", f"cannot MOVE unknown obstacle {obstacle_id}")
                if operation == "ADD":
                    known_ids.add(obstacle_id)
            progress = event.get("robot_waypoint_index")
            if isinstance(progress, bool) or not isinstance(progress, int) or progress < 0:
                raise ReplanningError("INVALID_ROBOT_PROGRESS", f"{event_id} robot_waypoint_index must be non-negative")
            if event.get("expected_old_path_state") not in {"VALID", "BLOCKED"}:
                raise ReplanningError("INVALID_EXPECTED_RESULT", f"{event_id} old path expectation is invalid")
            if event.get("expected_replan_status") not in {"NOT_RUN", "REPLAN_SUCCESS", "REPLAN_FAILED"}:
                raise ReplanningError("INVALID_EXPECTED_RESULT", f"{event_id} replan expectation is invalid")
            if event.get("expected_system_action") not in {"CONTINUE_CURRENT_PATH", "SWITCH_TO_REPLANNED_PATH", "STOP"}:
                raise ReplanningError("INVALID_EXPECTED_RESULT", f"{event_id} action expectation is invalid")
            revision = event.get("expected_active_path_revision")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise ReplanningError("INVALID_EXPECTED_RESULT", f"{event_id} expected path revision is invalid")
    if scenario_ids != {f"R{index}" for index in range(1, 8)}:
        raise ReplanningError("INVALID_SCENARIO_SET", "the seven required R1-R7 scenarios must be present")
    return config


def load_replanning_config(path):
    return validate_replanning_config(json.loads(Path(path).read_text()))


def replanning_scenario_by_id(config, scenario_id):
    return next(item for item in config["scenarios"] if item["scenario_id"] == scenario_id)


def initial_replanning_state(scenario):
    return {"world_revision": 0, "obstacles": deepcopy(scenario["initial_obstacles"])}


def create_active_path(planner_result, scenario_id, path_revision, world_revision, mission_goal, lineage=None):
    if planner_result.get("status") != "PATH_FOUND":
        raise ReplanningError("PATH_NOT_FOUND", "only a validated PATH_FOUND result can become active")
    return {
        "path_id": f"{scenario_id}_path_r{path_revision}",
        "path_revision": path_revision,
        "source_planner": planner_result["planner"],
        "world_revision_created": world_revision,
        "start_pose": {
            "input_m": deepcopy(planner_result["start_input_m"]),
            "grid": deepcopy(planner_result["start_grid"]),
            "cell_center_m": deepcopy(planner_result["start_cell_center_m"]),
        },
        "goal_pose": {
            "mission_input_m": [float(mission_goal["x"]), float(mission_goal["y"])],
            "grid": deepcopy(planner_result["goal_grid"]),
            "cell_center_m": deepcopy(planner_result["goal_cell_center_m"]),
        },
        "path_grid": deepcopy(planner_result["path_grid"]),
        "path_world": deepcopy(planner_result["path_world"]),
        "path_length_m": planner_result["path_length_m"],
        "minimum_clearance_m": planner_result["minimum_clearance_m"],
        "accumulated_safety_risk": planner_result["accumulated_safety_risk"],
        "waypoint_count": planner_result["number_of_waypoints"],
        "status": "ACTIVE",
        "lineage": deepcopy(lineage),
    }


def current_robot_pose(active_path, waypoint_index, previous_index=None):
    if isinstance(waypoint_index, bool) or not isinstance(waypoint_index, int) or not 0 <= waypoint_index < len(active_path["path_world"]):
        raise ReplanningError("INVALID_ROBOT_PROGRESS", "robot waypoint index is outside the active path")
    if previous_index is not None and waypoint_index < previous_index:
        raise ReplanningError("BACKWARD_ROBOT_PROGRESS", "robot progress cannot move backward on one path revision")
    point = active_path["path_world"][waypoint_index]
    return {
        "path_id": active_path["path_id"],
        "path_revision": active_path["path_revision"],
        "waypoint_index": waypoint_index,
        "grid": deepcopy(active_path["path_grid"][waypoint_index]),
        "x": point[0],
        "y": point[1],
    }


def _endpoint_status(point, collision_world, grid, label):
    query = check_robot_center(point["x"], point["y"], collision_world)
    if query["status"] != "COLLISION_FREE":
        return {"valid": False, "reason": label, "blocking_obstacle_ids": [query["obstacle_id"] or "WORKSPACE_BOUNDARY"]}
    cell = metric_to_grid(point["x"], point["y"], grid)
    if grid["occupancy"][cell[0]][cell[1]]:
        return {"valid": False, "reason": label, "blocking_obstacle_ids": ["GRID_OCCUPANCY"]}
    return {"valid": True, "reason": None, "blocking_obstacle_ids": [], "grid": cell}


def validate_replanned_path(planner_result, current_pose, mission_goal, collision_world, grid):
    if planner_result.get("status") != "PATH_FOUND":
        return {"status": "INVALID", "reason": "PATH_NOT_FOUND"}
    validation = validate_current_path(
        planner_result["path_grid"], planner_result["path_world"], collision_world, grid
    )
    current_cell = metric_to_grid(current_pose["x"], current_pose["y"], grid)
    goal_cell = metric_to_grid(mission_goal["x"], mission_goal["y"], grid)
    checks = {
        "path_state_valid": validation["path_state"] == "VALID",
        "start_cell_matches_current_pose": planner_result["start_grid"] == current_cell,
        "first_waypoint_matches_current_pose_mapping": planner_result["path_grid"][0] == current_cell,
        "goal_cell_matches_original_goal": planner_result["goal_grid"] == goal_cell,
        "final_waypoint_matches_original_goal_mapping": planner_result["path_grid"][-1] == goal_cell,
    }
    return {
        "status": "VALID" if all(checks.values()) else "INVALID",
        "checks": checks,
        "path_validation": validation,
    }


def _timed(function, *args):
    started = perf_counter()
    value = function(*args)
    return value, (perf_counter() - started) * 1000.0


def evaluate_event_and_replan(config, scenario, planner_config, zones, physical, readiness,
                              obstacle_state, active_path, path_history, event, previous_progress_index=None,
                              apply_obstacle_update=True):
    pipeline_started = perf_counter()
    pose = current_robot_pose(active_path, event["robot_waypoint_index"], previous_progress_index)
    old_revision = active_path["path_revision"]
    updated_state = apply_obstacle_event(obstacle_state, event) if apply_obstacle_update else deepcopy(obstacle_state)
    world, world_ms = _timed(build_dynamic_collision_world, config, updated_state, zones, physical, readiness)
    grid, grid_ms = _timed(build_grid_map, world, planner_config)
    pose_status = _endpoint_status(pose, world, grid, "ROBOT_POSE_INVALID")
    goal_point = {"x": scenario["goal"]["x"], "y": scenario["goal"]["y"]}
    goal_status = _endpoint_status(goal_point, world, grid, "GOAL_BLOCKED")
    executed_prefix = active_path["path_world"][:pose["waypoint_index"] + 1]
    remaining_grid = active_path["path_grid"][pose["waypoint_index"]:]
    remaining_world = active_path["path_world"][pose["waypoint_index"]:]
    common = {
        "event_id": event["event_id"],
        "trigger_event_id": event["event_id"],
        "event_index": event["event_index"],
        "operation": event["operation"],
        "world_revision_before": obstacle_state["world_revision"],
        "world_revision_after": updated_state["world_revision"],
        "trigger_world_revision": updated_state["world_revision"],
        "current_robot_pose": pose,
        "goal_pose": deepcopy(scenario["goal"]),
        "current_robot_index_on_old_path": pose["waypoint_index"],
        "executed_prefix_length_m": path_length_m(executed_prefix),
        "old_remaining_length_m": path_length_m(remaining_world),
        "old_path_revision": old_revision,
        "collision_world_rebuild_ms": world_ms,
        "grid_rebuild_ms": grid_ms,
        "costmap_world_revision": updated_state["world_revision"],
        "additional_inflation_m": grid["additional_inflation_m"],
        "performance_metric_status": "VOLATILE_OBSERVATION_NOT_BENCHMARK",
    }

    endpoint_failure = None if pose_status["valid"] and goal_status["valid"] else (pose_status if not pose_status["valid"] else goal_status)
    if endpoint_failure:
        old = deepcopy(active_path)
        old["status"] = endpoint_failure["reason"]
        history = deepcopy(path_history) + [old]
        common.update({
            "old_path_state": "NOT_EVALUATED",
            "current_path_validation": None,
            "old_path_minimum_clearance_m": None,
            "blocking_reason": endpoint_failure["reason"],
            "blocking_obstacle_ids": endpoint_failure["blocking_obstacle_ids"],
            "first_blocked_waypoint": None,
            "first_blocked_edge": None,
            "state_trace": ["STOP_REQUIRED"],
            "replanning": {"required": False, "status": "NOT_RUN", "planner": config["planner"], "planner_result": None, "replan_planning_time_ms": None},
            "old_active_path": {"path_id": old["path_id"], "path_revision": old_revision, "status": old["status"]},
            "new_active_path": None,
            "new_path_length_m": None,
            "new_path_minimum_clearance_m": None,
            "replanned_path_validation": None,
            "replanned_path_validation_ms": None,
            "planner_runs_after_event": 0,
            "stop_required": True,
            "system_action": "STOP",
            "total_replanning_pipeline_ms": (perf_counter() - pipeline_started) * 1000.0,
        })
        return updated_state, world, grid, None, history, common

    validation, validation_ms = _timed(
        validate_current_path, remaining_grid, remaining_world, world, grid
    )
    old_metrics = current_path_metrics(remaining_grid, remaining_world, world, grid, validation["path_state"])
    common.update({
        "old_path_state": validation["path_state"],
        "current_path_validation": validation,
        "current_path_validation_ms": validation_ms,
        "old_path_minimum_clearance_m": old_metrics["minimum_clearance_m"],
        "old_path_accumulated_safety_risk": old_metrics["accumulated_safety_risk"],
        "blocking_reason": validation["reason"],
        "blocking_obstacle_ids": validation["blocking_obstacle_ids"],
        "first_blocked_waypoint": None if validation["first_blocked_waypoint_index"] is None else validation["first_blocked_waypoint_index"] + pose["waypoint_index"],
        "first_blocked_edge": None if validation["first_blocked_edge_index"] is None else validation["first_blocked_edge_index"] + pose["waypoint_index"],
    })
    if validation["path_state"] == "VALID":
        common.update({
            "state_trace": ["PATH_VALID"],
            "replanning": {"required": False, "status": "NOT_RUN", "planner": config["planner"], "planner_result": None, "replan_planning_time_ms": None},
            "old_active_path": {"path_id": active_path["path_id"], "path_revision": old_revision, "status": "ACTIVE"},
            "new_active_path": None,
            "active_path_after_event": {"path_id": active_path["path_id"], "path_revision": old_revision, "status": "ACTIVE"},
            "path_retention_reason": "ACTIVE_PATH_REMAINS_VALID",
            "new_path_length_m": None,
            "new_path_minimum_clearance_m": None,
            "replanned_path_validation": None,
            "replanned_path_validation_ms": None,
            "planner_runs_after_event": 0,
            "stop_required": False,
            "system_action": "CONTINUE_CURRENT_PATH",
            "total_replanning_pipeline_ms": (perf_counter() - pipeline_started) * 1000.0,
        })
        return updated_state, world, grid, active_path, deepcopy(path_history), common

    start = {"x": pose["x"], "y": pose["y"], "status": "DEVELOPMENT_DEFINED"}
    replanned = plan_path(world, grid, planner_config, config["planner"], start, scenario["goal"])
    replanning = {
        "required": True,
        "status": None,
        "planner": config["planner"],
        "planner_result": replanned["status"],
        "replan_planning_time_ms": replanned["planning_time_ms"],
        "nodes_expanded": replanned["nodes_expanded"],
        "new_path_length_m": replanned["path_length_m"],
        "minimum_clearance_m": replanned["minimum_clearance_m"],
        "accumulated_safety_risk": replanned["accumulated_safety_risk"],
        "waypoint_count": replanned["number_of_waypoints"],
    }
    if replanned["status"] == "NO_PATH":
        old = deepcopy(active_path)
        old["status"] = "BLOCKED"
        history = deepcopy(path_history) + [old]
        replanning["status"] = "REPLAN_FAILED"
        common.update({
            "state_trace": ["REPLAN_REQUIRED", "REPLANNING", "REPLAN_FAILED", "STOP_REQUIRED"],
            "replanning": replanning,
            "old_active_path": {"path_id": old["path_id"], "path_revision": old_revision, "status": "BLOCKED"},
            "new_active_path": None,
            "new_path_length_m": None,
            "new_path_minimum_clearance_m": None,
            "replanned_path_validation": None,
            "replanned_path_validation_ms": None,
            "planner_runs_after_event": 1,
            "stop_required": True,
            "system_action": "STOP",
            "total_replanning_pipeline_ms": (perf_counter() - pipeline_started) * 1000.0,
        })
        return updated_state, world, grid, None, history, common

    replanned_validation, replanned_validation_ms = _timed(
        validate_replanned_path, replanned, pose, scenario["goal"], world, grid
    )
    if replanned_validation["status"] != "VALID":
        raise RuntimeError("A* returned a path that failed independent replanning validation")
    lineage = {
        "previous_path_revision": old_revision,
        "new_path_revision": old_revision + 1,
        "replan_trigger_world_revision": updated_state["world_revision"],
        "trigger_event_id": event["event_id"],
        "trigger_reason": validation["reason"],
        "blocking_obstacle_ids": deepcopy(validation["blocking_obstacle_ids"]),
        "replan_start_pose": deepcopy(pose),
        "goal_pose": deepcopy(scenario["goal"]),
        "planner": config["planner"],
    }
    old = deepcopy(active_path)
    old["status"] = "SUPERSEDED"
    history = deepcopy(path_history) + [old]
    new_active = create_active_path(
        replanned, scenario["scenario_id"], old_revision + 1,
        updated_state["world_revision"], scenario["goal"], lineage,
    )
    replanning["status"] = "REPLAN_SUCCESS"
    common.update({
        "state_trace": ["REPLAN_REQUIRED", "REPLANNING", "REPLAN_SUCCESS"],
        "replanning": replanning,
        "old_active_path": {"path_id": old["path_id"], "path_revision": old_revision, "status": "SUPERSEDED"},
        "new_active_path": {"path_id": new_active["path_id"], "path_revision": new_active["path_revision"], "status": "ACTIVE"},
        "active_path_after_event": {"path_id": new_active["path_id"], "path_revision": new_active["path_revision"], "status": "ACTIVE"},
        "path_retention_reason": None,
        "new_path_length_m": new_active["path_length_m"],
        "new_path_minimum_clearance_m": new_active["minimum_clearance_m"],
        "replanned_path_validation": replanned_validation,
        "replanned_path_validation_ms": replanned_validation_ms,
        "planner_runs_after_event": 1,
        "stop_required": False,
        "system_action": "SWITCH_TO_REPLANNED_PATH",
        "lineage": lineage,
        "total_replanning_pipeline_ms": (perf_counter() - pipeline_started) * 1000.0,
    })
    return updated_state, world, grid, new_active, history, common
