"""Validate a semantic-metric point and project it onto the existing planner grid."""

import math
from numbers import Real

from .grid_astar import grid_to_metric, metric_to_grid, validate_planner_config


class NavigationObstacleAdapterError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _require(value, expected, code, message):
    if value != expected:
        raise NavigationObstacleAdapterError(code, message)


def validate_semantic_metric_observation(observation):
    if not isinstance(observation, dict):
        raise NavigationObstacleAdapterError("MALFORMED_OBSERVATION", "observation must be an object")
    for key in ("image_id", "track_id"):
        if not isinstance(observation.get(key), str) or not observation[key]:
            raise NavigationObstacleAdapterError("MALFORMED_OBSERVATION", f"{key} is required")

    semantic = observation.get("semantic")
    association = observation.get("association")
    metric = observation.get("metric")
    if not all(isinstance(value, dict) for value in (semantic, association, metric)):
        raise NavigationObstacleAdapterError("MALFORMED_OBSERVATION", "semantic, association, and metric are required")
    _require(semantic.get("status"), "VISIBLE_ENTITY_FOUND", "SEMANTIC_NOT_VISIBLE", "visible semantic entity is required")
    if not isinstance(semantic.get("label"), str) or not semantic["label"]:
        raise NavigationObstacleAdapterError("MALFORMED_OBSERVATION", "semantic label is required")
    if not isinstance(semantic.get("type"), str) or not semantic["type"]:
        raise NavigationObstacleAdapterError("MALFORMED_OBSERVATION", "semantic type is required")
    _require(association.get("status"), "ASSOCIATED", "NOT_ASSOCIATED", "candidate association is required")
    _require(metric.get("status"), "AVAILABLE", "METRIC_NOT_AVAILABLE", "metric point is required")
    _require(metric.get("coordinate_frame"), "LEGACY_WORKSPACE_COORDINATE", "FRAME_MISMATCH", "unexpected world frame")
    _require(metric.get("unit"), "m", "UNIT_MISMATCH", "world point must use meters")
    _require(metric.get("workspace_status"), "INSIDE", "OUTSIDE_WORKSPACE", "world point must be inside workspace")
    _require(metric.get("pipeline"), "RAW_BOTTOM_CENTER_PX_TO_STORED_UNDISTORTION_TO_LEGACY_HOMOGRAPHY", "UNSUPPORTED_PROVENANCE", "unexpected metric pipeline")
    _require(metric.get("crop_used_as_metric_source"), False, "UNSUPPORTED_PROVENANCE", "crop cannot source the metric point")
    ground = metric.get("ground_reference")
    if not isinstance(ground, dict):
        raise NavigationObstacleAdapterError("UNSUPPORTED_PROVENANCE", "ground reference is required")
    _require(ground.get("policy"), "BOTTOM_CENTER", "UNSUPPORTED_PROVENANCE", "unexpected ground-reference policy")
    _require(ground.get("meaning"), "DEVELOPMENT_GROUND_REFERENCE", "UNSUPPORTED_PROVENANCE", "unexpected ground-reference meaning")

    point = metric.get("world_xy")
    if not isinstance(point, list) or len(point) != 2:
        raise NavigationObstacleAdapterError("WORLD_POINT_MISSING", "world_xy must contain two coordinates")
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in point):
        raise NavigationObstacleAdapterError("WORLD_POINT_NON_FINITE", "world_xy must be finite")
    return observation


def build_navigation_obstacle_observation(observation, planner_config, grid):
    validate_semantic_metric_observation(observation)
    validate_planner_config(planner_config)
    if not isinstance(grid, dict):
        raise NavigationObstacleAdapterError("INVALID_GRID", "grid is required")
    for key in ("coordinate_frame", "unit"):
        if grid.get(key) != planner_config[key]:
            raise NavigationObstacleAdapterError("GRID_CONTRACT_MISMATCH", f"grid {key} mismatch")
    if grid.get("resolution_m") != planner_config["grid_resolution_m"]:
        raise NavigationObstacleAdapterError("GRID_CONTRACT_MISMATCH", "grid resolution mismatch")

    x, y = observation["metric"]["world_xy"]
    row, col = metric_to_grid(x, y, grid)
    timestamp = observation.get("timestamp", {"status": "NOT_AVAILABLE"})
    if timestamp != {"status": "NOT_AVAILABLE"}:
        raise NavigationObstacleAdapterError("UNSUPPORTED_TIMESTAMP", "offline source has no validated timestamp")
    semantic = observation["semantic"]
    metric = observation["metric"]
    return {
        "schema_version": "0.1",
        "observation_id": f'{observation["image_id"]}_{observation["track_id"]}',
        "source_image_id": observation["image_id"],
        "track_id": observation["track_id"],
        "semantic": semantic.copy(),
        "navigation_role": "OBSTACLE_CANDIDATE",
        "reference_point_world": {
            "xy_m": [float(x), float(y)],
            "frame": metric["coordinate_frame"],
            "reference_type": "BBOX_BOTTOM_CENTER_DEVELOPMENT_GROUND_REFERENCE",
            "workspace_status": metric["workspace_status"],
        },
        "navigation_representation": {
            "type": "SINGLE_OCCUPIED_CELL",
            "source_representation": "NAVIGATION_POINT_OBSTACLE",
            "geometry_status": "NAVIGATION_APPROXIMATION",
            "physical_geometry_source": "NOT_USED",
            "grid_index": {"row": row, "col": col},
            "grid_cell_center_m": grid_to_metric(row, col, grid),
            "grid_resolution_m": grid["resolution_m"],
            "discretization_source": "PLANNER_GRID_RESOLUTION",
            "baseline_cell_occupancy": grid["occupancy"][row][col],
            "projection_only": True,
            "occupancy_grid_modified": False,
        },
        "physical_geometry": {"status": "NOT_READY", "required_for_adapter": False},
        "timestamp": timestamp,
        "evidence": {
            "semantic": semantic["status"],
            "association": observation["association"]["status"],
            "metric": metric["status"],
            "numeric_confidence": "NOT_PROVIDED",
        },
        "provenance": observation["provenance"].copy(),
        "planner_injection": "NOT_RUN",
    }
