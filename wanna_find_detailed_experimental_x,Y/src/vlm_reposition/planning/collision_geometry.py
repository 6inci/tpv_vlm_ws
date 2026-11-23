"""Continuous collision geometry for the fixed-yaw robot-only baseline."""

from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path

from vlm_reposition.world.physical_geometry import validate_physical_geometry
from vlm_reposition.world.zones import validate_zones


class CollisionGeometryError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise CollisionGeometryError("NON_FINITE_GEOMETRY", f"{name} must be finite")
    return float(value)


def _bounds(value, name):
    if not isinstance(value, dict):
        raise CollisionGeometryError("INVALID_RECTANGLE", f"{name} must be an object")
    result = {key: _finite(value.get(key), f"{name}.{key}") for key in ("x_min", "x_max", "y_min", "y_max")}
    if result["x_min"] >= result["x_max"] or result["y_min"] >= result["y_max"]:
        raise CollisionGeometryError("INVALID_RECTANGLE", f"{name} bounds must be ordered")
    return result


def validate_static_obstacles(config):
    if not isinstance(config, dict) or config.get("schema_version") != "0.1":
        raise CollisionGeometryError("INVALID_SCHEMA", "static obstacle schema_version must be 0.1")
    if config.get("coordinate_frame") != "LEGACY_WORKSPACE_COORDINATE":
        raise CollisionGeometryError("FRAME_MISMATCH", "static obstacles must use LEGACY_WORKSPACE_COORDINATE")
    if config.get("unit") != "m":
        raise CollisionGeometryError("UNIT_MISMATCH", "static obstacles must use meters")
    if config.get("scenario_type") != "ROBOT_ONLY_STATIC_PLANNING_DEVELOPMENT":
        raise CollisionGeometryError("INVALID_SCHEMA", "unexpected static obstacle scenario type")
    if config.get("mode") != "ROBOT_ONLY_TRANSLATIONAL_BASELINE" or config.get("fixed_yaw_rad") != 0.0:
        raise CollisionGeometryError("INVALID_BASELINE_MODE", "baseline must use fixed yaw 0 rad")
    if config.get("touching_policy") != "COLLISION":
        raise CollisionGeometryError("INVALID_TOUCHING_POLICY", "touching policy must be COLLISION")
    if config.get("safety_margin_m") != 0.0:
        raise CollisionGeometryError("SAFETY_MARGIN_NOT_ALLOWED", "STEP 14 safety margin must be zero")

    obstacles = config.get("obstacles")
    if not isinstance(obstacles, list) or not obstacles:
        raise CollisionGeometryError("INVALID_SCHEMA", "obstacles must be a non-empty list")
    ids = set()
    for index, obstacle in enumerate(obstacles, start=1):
        if not isinstance(obstacle, dict):
            raise CollisionGeometryError("INVALID_SCHEMA", f"obstacle {index} must be an object")
        obstacle_id = obstacle.get("id")
        if not isinstance(obstacle_id, str) or not obstacle_id or obstacle_id in ids:
            raise CollisionGeometryError("INVALID_SCHEMA", f"obstacle {index} has invalid or duplicate ID")
        ids.add(obstacle_id)
        geometry_status = obstacle.get("geometry_status")
        if geometry_status not in {"DEVELOPMENT_SYNTHETIC", "NAVIGATION_APPROXIMATION"}:
            raise CollisionGeometryError("UNSUPPORTED_GEOMETRY_STATUS", f"{obstacle_id} has unsupported geometry status")
        if geometry_status == "NAVIGATION_APPROXIMATION" and (
            obstacle.get("state_role") != "DYNAMIC_OBSTACLE"
            or obstacle.get("bridge_geometry") != "NAVIGATION_DISCRETIZATION_CELL"
            or obstacle.get("representation_source") != "SINGLE_OCCUPIED_CELL"
            or obstacle.get("physical_geometry_status") != "NOT_READY"
            or obstacle.get("source_type") != "STEP24N_NAVIGATION_APPROXIMATION"
        ):
            raise CollisionGeometryError("INVALID_NAVIGATION_APPROXIMATION", f"{obstacle_id} navigation provenance is invalid")
        if obstacle.get("planning_role") != "STATIC_OBSTACLE":
            raise CollisionGeometryError("UNSUPPORTED_PLANNING_ROLE", f"{obstacle_id} must be a STATIC_OBSTACLE")
        if obstacle.get("shape") != "axis_aligned_rectangle":
            raise CollisionGeometryError("UNSUPPORTED_GEOMETRY_SHAPE", f"{obstacle_id} must be axis aligned")
        if obstacle.get("coordinate_frame") != config["coordinate_frame"]:
            raise CollisionGeometryError("FRAME_MISMATCH", f"{obstacle_id} frame mismatch")
        if obstacle.get("unit") != config["unit"]:
            raise CollisionGeometryError("UNIT_MISMATCH", f"{obstacle_id} unit mismatch")
        if not isinstance(obstacle.get("source_type"), str) or not obstacle["source_type"] or not isinstance(obstacle.get("source_detail"), str) or not obstacle["source_detail"]:
            raise CollisionGeometryError("MISSING_PROVENANCE", f"{obstacle_id} must record source provenance")
        _bounds(obstacle.get("bounds"), f"{obstacle_id}.bounds")
    return config


def load_static_obstacles(path):
    return validate_static_obstacles(json.loads(Path(path).read_text()))


def robot_half_extents(robot):
    if not isinstance(robot, dict) or robot.get("geometry_status") != "CONFIRMED":
        raise CollisionGeometryError("ROBOT_FOOTPRINT_NOT_READY", "robot footprint must be CONFIRMED")
    if robot.get("reference_frame") != "base_link":
        raise CollisionGeometryError("ROBOT_FRAME_MISMATCH", "robot footprint must use base_link")
    convention = robot.get("axis_convention", {})
    if convention.get("x_positive") != "forward" or convention.get("y_positive") != "left":
        raise CollisionGeometryError("ROBOT_AXIS_MISMATCH", "robot axes must be +x forward and +y left")
    points = robot.get("source_footprint_m")
    if not isinstance(points, list) or len(points) != 4:
        raise CollisionGeometryError("INVALID_ROBOT_FOOTPRINT", "robot footprint must have four points")
    parsed = []
    for index, point in enumerate(points, start=1):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise CollisionGeometryError("INVALID_ROBOT_FOOTPRINT", f"robot point {index} is invalid")
        parsed.append((_finite(point[0], f"robot point {index} x"), _finite(point[1], f"robot point {index} y")))
    xs, ys = [point[0] for point in parsed], [point[1] for point in parsed]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if not math.isclose(xmin, -xmax) or not math.isclose(ymin, -ymax) or set(parsed) != {
        (xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)
    }:
        raise CollisionGeometryError("INVALID_ROBOT_FOOTPRINT", "robot footprint must be an origin-centered axis-aligned rectangle")
    if not math.isclose(2 * xmax, robot.get("length_m")) or not math.isclose(2 * ymax, robot.get("width_m")):
        raise CollisionGeometryError("ROBOT_FOOTPRINT_SOURCE_MISMATCH", "robot polygon and extents disagree")
    return {"x": xmax, "y": ymax}


def compute_valid_robot_center_bounds(workspace_bounds, half_extents):
    workspace = _bounds(workspace_bounds, "workspace")
    half_x = _finite(half_extents.get("x"), "robot half x")
    half_y = _finite(half_extents.get("y"), "robot half y")
    if half_x <= 0 or half_y <= 0:
        raise CollisionGeometryError("INVALID_ROBOT_FOOTPRINT", "robot half extents must be positive")
    center = {
        "x_min": workspace["x_min"] + half_x,
        "x_max": workspace["x_max"] - half_x,
        "y_min": workspace["y_min"] + half_y,
        "y_max": workspace["y_max"] - half_y,
    }
    if center["x_min"] >= center["x_max"] or center["y_min"] >= center["y_max"]:
        raise CollisionGeometryError("WORKSPACE_TOO_SMALL", "workspace is too small for the robot footprint")
    return center


def expand_axis_aligned_obstacle(bounds, half_extents):
    raw = _bounds(bounds, "obstacle")
    half_x = _finite(half_extents.get("x"), "robot half x")
    half_y = _finite(half_extents.get("y"), "robot half y")
    if half_x <= 0 or half_y <= 0:
        raise CollisionGeometryError("INVALID_ROBOT_FOOTPRINT", "robot half extents must be positive")
    expanded = {
        "x_min": raw["x_min"] - half_x,
        "x_max": raw["x_max"] + half_x,
        "y_min": raw["y_min"] - half_y,
        "y_max": raw["y_max"] + half_y,
    }
    return _bounds(expanded, "expanded obstacle")


def build_collision_world(workspace_config, physical_config, obstacle_config, manipulation_readiness):
    validate_zones(deepcopy(workspace_config))
    validate_physical_geometry(deepcopy(physical_config))
    validate_static_obstacles(deepcopy(obstacle_config))
    if workspace_config["coordinate_frame"] != obstacle_config["coordinate_frame"]:
        raise CollisionGeometryError("FRAME_MISMATCH", "workspace and obstacle frames differ")
    if workspace_config["unit"] != obstacle_config["unit"] or physical_config["coordinate_units"] != obstacle_config["unit"]:
        raise CollisionGeometryError("UNIT_MISMATCH", "workspace, robot and obstacle units differ")
    if manipulation_readiness not in {"READY", "PARTIAL", "NOT_READY"}:
        raise CollisionGeometryError("INVALID_READINESS", "manipulation readiness is invalid")

    xbounds, ybounds = workspace_config["workspace_bounds"]["x"], workspace_config["workspace_bounds"]["y"]
    workspace = {"x_min": xbounds[0], "x_max": xbounds[1], "y_min": ybounds[0], "y_max": ybounds[1]}
    workspace = _bounds(workspace, "workspace")
    half_extents = robot_half_extents(physical_config["robot"])
    valid_center = compute_valid_robot_center_bounds(workspace, half_extents)
    obstacles = []
    for obstacle in obstacle_config["obstacles"]:
        raw = _bounds(obstacle["bounds"], f"{obstacle['id']}.bounds")
        if raw["x_min"] < workspace["x_min"] or raw["x_max"] > workspace["x_max"] or raw["y_min"] < workspace["y_min"] or raw["y_max"] > workspace["y_max"]:
            raise CollisionGeometryError("OBSTACLE_OUTSIDE_WORKSPACE", f"{obstacle['id']} is outside workspace bounds")
        record = {
            "id": obstacle["id"],
            "geometry_status": obstacle["geometry_status"],
            "planning_role": obstacle["planning_role"],
            "shape": obstacle["shape"],
            "raw_bounds": raw,
            "collision_bounds": expand_axis_aligned_obstacle(raw, half_extents),
            "source_type": obstacle["source_type"],
            "source_detail": obstacle["source_detail"],
        }
        for key in (
            "bridge_geometry", "representation_source", "physical_geometry_status",
            "reference_point_world", "navigation_cell", "observation_identity", "timestamp", "provenance",
        ):
            if key in obstacle:
                record[key] = deepcopy(obstacle[key])
        obstacles.append(record)

    return {
        "schema_version": "0.1",
        "mode": obstacle_config["mode"],
        "coordinate_frame": obstacle_config["coordinate_frame"],
        "unit": obstacle_config["unit"],
        "touching_policy": "COLLISION",
        "safety_margin_m": 0.0,
        "robot": {
            "reference_frame": physical_config["robot"]["reference_frame"],
            "axis_convention": deepcopy(physical_config["robot"]["axis_convention"]),
            "footprint_m": deepcopy(physical_config["robot"]["source_footprint_m"]),
            "half_extents_m": half_extents,
            "fixed_yaw_rad": 0.0,
            "geometry_status": physical_config["robot"]["geometry_status"],
            "source_path": physical_config["robot"]["source_path"],
        },
        "workspace": {"raw_bounds": workspace, "valid_robot_center_bounds": valid_center},
        "static_obstacles": obstacles,
        "readiness": {
            "overall_manipulation_geometry_readiness": manipulation_readiness,
            "robot_only_static_planning_geometry_readiness": "READY",
        },
        "collision_query": {"status": "AVAILABLE", "representation": "CONTINUOUS_METRIC_RECTANGLES"},
        "path_planning": {"status": "NOT_RUN"},
    }


def check_robot_center(x, y, collision_world):
    x, y = _finite(x, "robot center x"), _finite(y, "robot center y")
    center = collision_world["workspace"]["valid_robot_center_bounds"]
    if x <= center["x_min"] or x >= center["x_max"] or y <= center["y_min"] or y >= center["y_max"]:
        return {"status": "COLLISION", "reason": "WORKSPACE_BOUNDARY", "obstacle_id": None}
    for obstacle in collision_world["static_obstacles"]:
        bounds = obstacle["collision_bounds"]
        if bounds["x_min"] <= x <= bounds["x_max"] and bounds["y_min"] <= y <= bounds["y_max"]:
            return {"status": "COLLISION", "reason": "STATIC_OBSTACLE", "obstacle_id": obstacle["id"]}
    return {"status": "COLLISION_FREE", "reason": None, "obstacle_id": None}


def is_robot_center_valid(x, y, collision_world):
    return check_robot_center(x, y, collision_world)["status"] == "COLLISION_FREE"
