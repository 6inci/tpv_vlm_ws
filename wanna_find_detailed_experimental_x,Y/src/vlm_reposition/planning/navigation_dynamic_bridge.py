"""Bridge a STEP 24N containing cell into the existing STEP 17 obstacle contract."""

from copy import deepcopy
import math
from numbers import Real

from .dynamic_state import _validate_obstacle
from .grid_astar import grid_to_metric, metric_to_grid


class NavigationDynamicBridgeError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def grid_cell_bounds(row, col, grid):
    x, y = grid_to_metric(row, col, grid)
    half = grid["resolution_m"] / 2
    return {"x_min": x - half, "x_max": x + half, "y_min": y - half, "y_max": y + half}


def build_step17_dynamic_obstacle(observation, grid, dynamic_config):
    if not isinstance(observation, dict) or observation.get("schema_version") != "0.1":
        raise NavigationDynamicBridgeError("MALFORMED_OBSERVATION", "STEP 24N observation is invalid")
    if observation.get("navigation_role") != "OBSTACLE_CANDIDATE":
        raise NavigationDynamicBridgeError("INVALID_NAVIGATION_ROLE", "observation must be an obstacle candidate")
    representation = observation.get("navigation_representation")
    if not isinstance(representation, dict) or representation.get("type") != "SINGLE_OCCUPIED_CELL":
        raise NavigationDynamicBridgeError("INVALID_REPRESENTATION", "single occupied cell is required")
    if (
        representation.get("source_representation") != "NAVIGATION_POINT_OBSTACLE"
        or representation.get("geometry_status") != "NAVIGATION_APPROXIMATION"
        or representation.get("physical_geometry_source") != "NOT_USED"
        or representation.get("discretization_source") != "PLANNER_GRID_RESOLUTION"
        or representation.get("occupancy_grid_modified") is not False
    ):
        raise NavigationDynamicBridgeError("INVALID_PROVENANCE", "navigation approximation provenance is invalid")
    if observation.get("physical_geometry") != {"status": "NOT_READY", "required_for_adapter": False}:
        raise NavigationDynamicBridgeError("INVALID_PHYSICAL_GEOMETRY", "physical geometry must remain NOT_READY")
    if observation.get("timestamp") != {"status": "NOT_AVAILABLE"}:
        raise NavigationDynamicBridgeError("INVALID_TIMESTAMP", "offline observation timestamp must remain unavailable")

    reference = observation.get("reference_point_world")
    cell = representation.get("grid_index")
    if not isinstance(reference, dict) or not isinstance(cell, dict):
        raise NavigationDynamicBridgeError("MALFORMED_OBSERVATION", "reference point and grid cell are required")
    point = reference.get("xy_m")
    if (
        not isinstance(point, list) or len(point) != 2
        or any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in point)
    ):
        raise NavigationDynamicBridgeError("INVALID_REFERENCE_POINT", "reference point must be finite")
    if reference.get("frame") != dynamic_config["coordinate_frame"] or grid.get("coordinate_frame") != dynamic_config["coordinate_frame"]:
        raise NavigationDynamicBridgeError("FRAME_MISMATCH", "observation, grid, and STEP 17 frame must match")
    if grid.get("unit") != dynamic_config["unit"] or representation.get("grid_resolution_m") != grid.get("resolution_m"):
        raise NavigationDynamicBridgeError("GRID_METADATA_MISMATCH", "grid metadata does not match STEP 17")
    row, col = cell.get("row"), cell.get("col")
    if [row, col] != metric_to_grid(point[0], point[1], grid):
        raise NavigationDynamicBridgeError("GRID_CELL_MISMATCH", "reference point is not in the recorded cell")
    if representation.get("grid_cell_center_m") != grid_to_metric(row, col, grid):
        raise NavigationDynamicBridgeError("GRID_CELL_MISMATCH", "recorded cell center does not match the grid")
    if representation.get("baseline_cell_occupancy") != grid["occupancy"][row][col]:
        raise NavigationDynamicBridgeError("GRID_CELL_MISMATCH", "recorded baseline occupancy does not match the grid")

    obstacle = {
        "id": f'dynamic_{observation["observation_id"]}',
        "state_role": "DYNAMIC_OBSTACLE",
        "geometry_status": "NAVIGATION_APPROXIMATION",
        "shape": "axis_aligned_rectangle",
        "bounds": grid_cell_bounds(row, col, grid),
        "coordinate_frame": dynamic_config["coordinate_frame"],
        "unit": dynamic_config["unit"],
        "source_type": "STEP24N_NAVIGATION_APPROXIMATION",
        "source_detail": "Exact metric bounds of the validated STEP 24N containing grid cell; not physical object geometry.",
        "bridge_geometry": "NAVIGATION_DISCRETIZATION_CELL",
        "representation_source": "SINGLE_OCCUPIED_CELL",
        "physical_geometry_status": "NOT_READY",
        "reference_point_world": deepcopy(reference),
        "navigation_cell": {
            "row": row,
            "col": col,
            "center_m": deepcopy(representation["grid_cell_center_m"]),
            "resolution_m": grid["resolution_m"],
            "bounds_m": grid_cell_bounds(row, col, grid),
        },
        "observation_identity": {
            "source_image_id": observation["source_image_id"],
            "track_id": observation["track_id"],
            "identity_scope": "OFFLINE_TEMPORAL_OBSERVATION",
            "persistent_physical_object_identity": "NOT_CLAIMED",
        },
        "timestamp": deepcopy(observation["timestamp"]),
        "provenance": {
            "source_step": "STEP_24N",
            "source_observation_id": observation["observation_id"],
            "source_provenance": deepcopy(observation["provenance"]),
            "reference_point_recomputed": False,
            "physical_geometry_used": False,
        },
    }
    _validate_obstacle(obstacle, dynamic_config, "STEP25N bridge")
    return obstacle
