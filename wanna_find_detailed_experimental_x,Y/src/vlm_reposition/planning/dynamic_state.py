"""Deterministic offline obstacle events and current-path invalidation."""

from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path

from .collision_geometry import build_collision_world, check_robot_center
from .grid_astar import accumulated_safety_risk, clearance_at, grid_to_metric


class DynamicStateError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise DynamicStateError("NON_FINITE_GEOMETRY", f"{name} must be finite")
    return float(value)


def _validate_obstacle(obstacle, config, name):
    if not isinstance(obstacle, dict):
        raise DynamicStateError("INVALID_OBSTACLE", f"{name} must be an object")
    obstacle_id = obstacle.get("id")
    if not isinstance(obstacle_id, str) or not obstacle_id:
        raise DynamicStateError("INVALID_OBSTACLE_ID", f"{name}.id must be non-empty")
    if obstacle.get("state_role") not in {"STATIC_OBSTACLE", "DYNAMIC_OBSTACLE"}:
        raise DynamicStateError("INVALID_OBSTACLE_ROLE", f"{obstacle_id} has invalid state_role")
    geometry_status = obstacle.get("geometry_status")
    if geometry_status not in {"DEVELOPMENT_SYNTHETIC", "NAVIGATION_APPROXIMATION"}:
        raise DynamicStateError("UNSUPPORTED_GEOMETRY_STATUS", f"{obstacle_id} has unsupported geometry status")
    if geometry_status == "NAVIGATION_APPROXIMATION" and (
        obstacle.get("state_role") != "DYNAMIC_OBSTACLE"
        or obstacle.get("bridge_geometry") != "NAVIGATION_DISCRETIZATION_CELL"
        or obstacle.get("representation_source") != "SINGLE_OCCUPIED_CELL"
        or obstacle.get("physical_geometry_status") != "NOT_READY"
        or obstacle.get("source_type") != "STEP24N_NAVIGATION_APPROXIMATION"
    ):
        raise DynamicStateError("INVALID_NAVIGATION_APPROXIMATION", f"{obstacle_id} navigation provenance is invalid")
    if obstacle.get("shape") != "axis_aligned_rectangle":
        raise DynamicStateError("INVALID_RECTANGLE", f"{obstacle_id} must be an axis-aligned rectangle")
    if obstacle.get("coordinate_frame") != config["coordinate_frame"]:
        raise DynamicStateError("FRAME_MISMATCH", f"{obstacle_id} frame mismatch")
    if obstacle.get("unit") != config["unit"]:
        raise DynamicStateError("UNIT_MISMATCH", f"{obstacle_id} unit mismatch")
    if not isinstance(obstacle.get("source_type"), str) or not obstacle["source_type"] or not isinstance(obstacle.get("source_detail"), str) or not obstacle["source_detail"]:
        raise DynamicStateError("MISSING_PROVENANCE", f"{obstacle_id} provenance is required")
    bounds = obstacle.get("bounds")
    if not isinstance(bounds, dict):
        raise DynamicStateError("INVALID_RECTANGLE", f"{obstacle_id}.bounds must be an object")
    parsed = {key: _finite(bounds.get(key), f"{obstacle_id}.bounds.{key}") for key in ("x_min", "x_max", "y_min", "y_max")}
    if parsed["x_min"] >= parsed["x_max"] or parsed["y_min"] >= parsed["y_max"]:
        raise DynamicStateError("INVALID_RECTANGLE", f"{obstacle_id} bounds must be ordered")
    workspace = config["workspace_bounds"]
    if parsed["x_min"] < workspace["x_min"] or parsed["x_max"] > workspace["x_max"] or parsed["y_min"] < workspace["y_min"] or parsed["y_max"] > workspace["y_max"]:
        raise DynamicStateError("OBSTACLE_OUTSIDE_WORKSPACE", f"{obstacle_id} is outside workspace")
    return obstacle


def validate_dynamic_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != "0.1":
        raise DynamicStateError("INVALID_SCHEMA", "dynamic scenario schema_version must be 0.1")
    required = {
        "coordinate_frame": "LEGACY_WORKSPACE_COORDINATE",
        "unit": "m",
        "mode": "ROBOT_ONLY_TRANSLATIONAL_BASELINE",
        "fixed_yaw_rad": 0.0,
        "geometry_status": "DEVELOPMENT_SYNTHETIC",
        "current_path_source_planner": "SAFETY_COST_ASTAR",
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise DynamicStateError("INVALID_SCHEMA", "dynamic scenario metadata is invalid")
    if not isinstance(config.get("source_detail"), str) or not config["source_detail"]:
        raise DynamicStateError("MISSING_PROVENANCE", "dynamic scenario provenance is required")
    bounds = config.get("workspace_bounds")
    if not isinstance(bounds, dict):
        raise DynamicStateError("INVALID_SCHEMA", "workspace_bounds is required")
    for key in ("x_min", "x_max", "y_min", "y_max"):
        _finite(bounds.get(key), f"workspace_bounds.{key}")
    if bounds["x_min"] >= bounds["x_max"] or bounds["y_min"] >= bounds["y_max"]:
        raise DynamicStateError("INVALID_SCHEMA", "workspace bounds must be ordered")

    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise DynamicStateError("INVALID_SCHEMA", "scenarios must be a non-empty list")
    scenario_ids = set()
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id") if isinstance(scenario, dict) else None
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in scenario_ids:
            raise DynamicStateError("INVALID_SCENARIO_ID", "scenario IDs must be non-empty and unique")
        scenario_ids.add(scenario_id)
        if not isinstance(scenario.get("description"), str) or not scenario["description"]:
            raise DynamicStateError("INVALID_SCHEMA", f"{scenario_id} description is required")
        for point_name in ("initial_start", "initial_goal"):
            point = scenario.get(point_name)
            if not isinstance(point, dict) or point.get("status") != "DEVELOPMENT_DEFINED":
                raise DynamicStateError("INVALID_ENDPOINT", f"{scenario_id}.{point_name} is invalid")
            _finite(point.get("x"), f"{scenario_id}.{point_name}.x")
            _finite(point.get("y"), f"{scenario_id}.{point_name}.y")
        ids = set()
        static = scenario.get("initial_static_obstacles")
        dynamic = scenario.get("initial_dynamic_obstacles", [])
        if not isinstance(static, list) or not static or not isinstance(dynamic, list):
            raise DynamicStateError("INVALID_SCHEMA", f"{scenario_id} initial obstacles are invalid")
        for obstacle in static + dynamic:
            _validate_obstacle(obstacle, config, scenario_id)
            if obstacle["id"] in ids:
                raise DynamicStateError("DUPLICATE_OBSTACLE_ID", f"duplicate obstacle ID {obstacle['id']}")
            ids.add(obstacle["id"])
        if any(item["state_role"] != "STATIC_OBSTACLE" for item in static) or any(item["state_role"] != "DYNAMIC_OBSTACLE" for item in dynamic):
            raise DynamicStateError("INVALID_OBSTACLE_ROLE", f"{scenario_id} initial obstacle roles do not match their state lists")
        events = scenario.get("events")
        if not isinstance(events, list) or not events:
            raise DynamicStateError("INVALID_SCHEMA", f"{scenario_id} events must be non-empty")
        event_ids = set()
        known_ids = set(ids)
        for expected_index, event in enumerate(events, start=1):
            if not isinstance(event, dict) or event.get("event_index") != expected_index:
                raise DynamicStateError("INVALID_EVENT_ORDER", f"{scenario_id} event indices must start at 1 and be contiguous")
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id or event_id in event_ids:
                raise DynamicStateError("INVALID_EVENT_ID", f"{scenario_id} event IDs must be non-empty and unique")
            event_ids.add(event_id)
            operation = event.get("operation")
            if operation not in {"ADD", "REMOVE", "MOVE"}:
                raise DynamicStateError("UNSUPPORTED_OPERATION", f"unsupported operation {operation}")
            if event.get("expected_path_state") not in {"VALID", "BLOCKED"}:
                raise DynamicStateError("INVALID_EXPECTED_PATH_STATE", f"{event_id} expected state is invalid")
            obstacle = event.get("obstacle")
            if operation == "REMOVE":
                obstacle_id = obstacle.get("id") if isinstance(obstacle, dict) else None
                if not isinstance(obstacle_id, str) or not obstacle_id:
                    raise DynamicStateError("INVALID_OBSTACLE_ID", f"{event_id} REMOVE requires an obstacle ID")
                if not isinstance(obstacle.get("source_type"), str) or not obstacle["source_type"] or not isinstance(obstacle.get("source_detail"), str) or not obstacle["source_detail"]:
                    raise DynamicStateError("MISSING_PROVENANCE", f"{event_id} REMOVE provenance is required")
                if obstacle_id not in known_ids:
                    raise DynamicStateError("UNKNOWN_OBSTACLE_ID", f"cannot REMOVE unknown obstacle {obstacle_id}")
                known_ids.remove(obstacle_id)
            else:
                _validate_obstacle(obstacle, config, event_id)
                if obstacle["state_role"] != "DYNAMIC_OBSTACLE":
                    raise DynamicStateError("INVALID_OBSTACLE_ROLE", f"{event_id} must update a dynamic obstacle")
                obstacle_id = obstacle["id"]
                if operation == "ADD" and obstacle_id in known_ids:
                    raise DynamicStateError("DUPLICATE_OBSTACLE_ID", f"cannot ADD existing obstacle {obstacle_id}")
                if operation == "MOVE" and obstacle_id not in known_ids:
                    raise DynamicStateError("UNKNOWN_OBSTACLE_ID", f"cannot MOVE unknown obstacle {obstacle_id}")
                if operation == "ADD":
                    known_ids.add(obstacle_id)
    required_ids = {f"D{index}" for index in range(1, 7)}
    if scenario_ids != required_ids:
        raise DynamicStateError("INVALID_SCENARIO_SET", "the six required D1-D6 scenarios must be present")
    return config


def load_dynamic_config(path):
    return validate_dynamic_config(json.loads(Path(path).read_text()))


def scenario_by_id(config, scenario_id):
    return next(item for item in config["scenarios"] if item["scenario_id"] == scenario_id)


def initial_obstacle_state(scenario):
    obstacles = deepcopy(scenario["initial_static_obstacles"] + scenario.get("initial_dynamic_obstacles", []))
    return {"world_revision": 0, "obstacles": obstacles}


def apply_obstacle_event(state, event):
    updated = deepcopy(state)
    obstacles = updated["obstacles"]
    ids = {item["id"]: index for index, item in enumerate(obstacles)}
    operation = event.get("operation")
    obstacle = event.get("obstacle")
    obstacle_id = obstacle.get("id") if isinstance(obstacle, dict) else None
    if operation == "ADD":
        if obstacle_id in ids:
            raise DynamicStateError("DUPLICATE_OBSTACLE_ID", f"cannot ADD existing obstacle {obstacle_id}")
        obstacles.append(deepcopy(obstacle))
    elif operation == "REMOVE":
        if obstacle_id not in ids:
            raise DynamicStateError("UNKNOWN_OBSTACLE_ID", f"cannot REMOVE unknown obstacle {obstacle_id}")
        obstacles.pop(ids[obstacle_id])
    elif operation == "MOVE":
        if obstacle_id not in ids:
            raise DynamicStateError("UNKNOWN_OBSTACLE_ID", f"cannot MOVE unknown obstacle {obstacle_id}")
        obstacles[ids[obstacle_id]] = deepcopy(obstacle)
    else:
        raise DynamicStateError("UNSUPPORTED_OPERATION", f"unsupported operation {operation}")
    revision = state.get("world_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise DynamicStateError("INVALID_WORLD_REVISION", "world_revision must be a non-negative integer")
    updated["world_revision"] = revision + 1
    return updated


def obstacle_config_from_state(config, state):
    return {
        "schema_version": "0.1",
        "coordinate_frame": config["coordinate_frame"],
        "unit": config["unit"],
        "scenario_type": "ROBOT_ONLY_STATIC_PLANNING_DEVELOPMENT",
        "mode": config["mode"],
        "fixed_yaw_rad": config["fixed_yaw_rad"],
        "touching_policy": "COLLISION",
        "safety_margin_m": 0.0,
        "obstacles": [
            {**deepcopy(obstacle), "planning_role": "STATIC_OBSTACLE"}
            for obstacle in state["obstacles"]
        ],
    }


def build_dynamic_collision_world(config, state, zones, physical, readiness):
    obstacle_config = obstacle_config_from_state(config, state)
    world = build_collision_world(zones, physical, obstacle_config, readiness)
    roles = {item["id"]: item["state_role"] for item in state["obstacles"]}
    for obstacle in world["static_obstacles"]:
        obstacle["state_role"] = roles[obstacle["id"]]
    world["world_revision"] = state["world_revision"]
    return world


def _segment_intersects_bounds(a, b, bounds):
    """Liang-Barsky intersection, with boundary contact treated as collision."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    low, high = 0.0, 1.0
    for p, q in ((-dx, a[0] - bounds["x_min"]), (dx, bounds["x_max"] - a[0]),
                 (-dy, a[1] - bounds["y_min"]), (dy, bounds["y_max"] - a[1])):
        if p == 0:
            if q < 0:
                return False
            continue
        ratio = q / p
        if p < 0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
        if low > high:
            return False
    return True


def _blocked(reason, revision, waypoint_index, edge_index, obstacle_ids):
    return {
        "path_state": "BLOCKED",
        "replan_required": True,
        "world_revision": revision,
        "first_blocked_waypoint_index": waypoint_index,
        "first_blocked_edge_index": edge_index,
        "reason": reason,
        "blocking_obstacle_ids": sorted(set(obstacle_ids)),
    }


def validate_current_path(path_grid, path_world, collision_world, grid):
    revision = collision_world.get("world_revision")
    if not path_grid or len(path_grid) != len(path_world):
        raise DynamicStateError("INVALID_CURRENT_PATH", "grid and world paths must be non-empty and aligned")
    for index, (cell, point) in enumerate(zip(path_grid, path_world)):
        if not isinstance(cell, (list, tuple)) or len(cell) != 2 or not all(isinstance(value, int) and not isinstance(value, bool) for value in cell):
            raise DynamicStateError("INVALID_CURRENT_PATH", f"path cell {index} is invalid")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise DynamicStateError("INVALID_CURRENT_PATH", f"path point {index} is invalid")
        _finite(point[0], f"path point {index}.x")
        _finite(point[1], f"path point {index}.y")

    def check_waypoint(index):
        cell, point = path_grid[index], path_world[index]
        x, y = point
        if not (0 <= cell[0] < grid["rows"] and 0 <= cell[1] < grid["cols"]):
            return _blocked("WORKSPACE_INVALID", revision, index, None if index == 0 else index - 1, ["WORKSPACE_BOUNDARY"])
        query = check_robot_center(x, y, collision_world)
        if query["status"] != "COLLISION_FREE":
            provenance = [query["obstacle_id"]] if query["obstacle_id"] else ["WORKSPACE_BOUNDARY"]
            reason = "WORKSPACE_INVALID" if query["reason"] == "WORKSPACE_BOUNDARY" else "WAYPOINT_COLLISION"
            return _blocked(reason, revision, index, None if index == 0 else index - 1, provenance)
        if grid["occupancy"][cell[0]][cell[1]]:
            return _blocked("WAYPOINT_COLLISION", revision, index, None if index == 0 else index - 1, ["GRID_OCCUPANCY"])
        return None

    invalid = check_waypoint(0)
    if invalid:
        return invalid
    for edge_index, (current_cell, next_cell, a, b) in enumerate(zip(path_grid, path_grid[1:], path_world, path_world[1:])):
        dr, dc = next_cell[0] - current_cell[0], next_cell[1] - current_cell[1]
        if (dr, dc) == (0, 0) or abs(dr) > 1 or abs(dc) > 1:
            return _blocked("EDGE_BLOCKED", revision, None, edge_index, ["INVALID_GRID_TRANSITION"])
        invalid = check_waypoint(edge_index + 1)
        if invalid:
            return invalid
        intersecting = [
            obstacle["id"] for obstacle in collision_world["static_obstacles"]
            if _segment_intersects_bounds(a, b, obstacle["collision_bounds"])
        ]
        if intersecting:
            return _blocked("EDGE_BLOCKED", revision, None, edge_index, intersecting)
        if dr and dc:
            corner_cells = ((current_cell[0] + dr, current_cell[1]), (current_cell[0], current_cell[1] + dc))
            blocked_cells = [cell for cell in corner_cells if grid["occupancy"][cell[0]][cell[1]]]
            if blocked_cells:
                provenance = []
                for row, col in blocked_cells:
                    query = check_robot_center(*grid_to_metric(row, col, grid), collision_world)
                    provenance.append(query["obstacle_id"] or "WORKSPACE_BOUNDARY")
                return _blocked("CORNER_CUT_BLOCKED", revision, None, edge_index, provenance)
    return {
        "path_state": "VALID",
        "replan_required": False,
        "world_revision": revision,
        "first_blocked_waypoint_index": None,
        "first_blocked_edge_index": None,
        "reason": None,
        "blocking_obstacle_ids": [],
    }


def current_path_metrics(path_grid, path_world, collision_world, grid, path_state):
    minimum_clearance = min(clearance_at(x, y, collision_world) for x, y in path_world)
    if path_state == "BLOCKED":
        return {
            "minimum_clearance_m": minimum_clearance,
            "accumulated_safety_risk": None,
            "accumulated_safety_risk_status": "NOT_APPLICABLE_PATH_BLOCKED",
        }
    return {
        "minimum_clearance_m": minimum_clearance,
        "accumulated_safety_risk": accumulated_safety_risk(path_grid, path_world, grid),
        "accumulated_safety_risk_status": "AVAILABLE",
    }
