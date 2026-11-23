"""Deterministic binary grid, safety cost layer, and 8-connected A*."""

import heapq
import itertools
import json
import math
from numbers import Real
from pathlib import Path
from time import perf_counter

from .collision_geometry import check_robot_center, is_robot_center_valid


NEIGHBORS = (
    ("N", 1, 0),
    ("NE", 1, 1),
    ("E", 0, 1),
    ("SE", -1, 1),
    ("S", -1, 0),
    ("SW", -1, -1),
    ("W", 0, -1),
    ("NW", 1, -1),
)


class PlannerError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _finite(value, code, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise PlannerError(code, f"{name} must be finite")
    return float(value)


def validate_planner_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != "0.1":
        raise PlannerError("INVALID_PLANNER_CONFIG", "planner schema_version must be 0.1")
    required = {
        "coordinate_frame": "LEGACY_WORKSPACE_COORDINATE",
        "unit": "m",
        "mode": "ROBOT_ONLY_TRANSLATIONAL_BASELINE",
        "connectivity": 8,
        "fixed_yaw_rad": 0.0,
        "parameter_status": "DEVELOPMENT_PARAMETER",
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise PlannerError("INVALID_PLANNER_CONFIG", "planner baseline metadata is invalid")
    if config.get("neighbor_order") != [item[0] for item in NEIGHBORS]:
        raise PlannerError("INVALID_PLANNER_CONFIG", "neighbor order must be deterministic")
    for key in ("grid_resolution_m", "safety_cost_radius_m"):
        if _finite(config.get(key), "INVALID_PLANNER_CONFIG", key) <= 0:
            raise PlannerError("INVALID_PLANNER_CONFIG", f"{key} must be positive")
    if _finite(config.get("safety_weight"), "INVALID_PLANNER_CONFIG", "safety_weight") < 0:
        raise PlannerError("INVALID_PLANNER_CONFIG", "safety_weight must be non-negative")
    if not isinstance(config.get("source_detail"), str) or not config["source_detail"]:
        raise PlannerError("INVALID_PLANNER_CONFIG", "development parameter provenance is required")
    for name in ("development_start", "development_goal"):
        point = config.get(name)
        if not isinstance(point, dict) or point.get("status") != "DEVELOPMENT_DEFINED":
            raise PlannerError("INVALID_PLANNER_CONFIG", f"{name} must be DEVELOPMENT_DEFINED")
        _finite(point.get("x"), "INVALID_PLANNER_CONFIG", f"{name}.x")
        _finite(point.get("y"), "INVALID_PLANNER_CONFIG", f"{name}.y")
    return config


def load_planner_config(path):
    return validate_planner_config(json.loads(Path(path).read_text()))


def _validate_world(world, config):
    if world.get("mode") != config["mode"] or world.get("robot", {}).get("fixed_yaw_rad") != config["fixed_yaw_rad"]:
        raise PlannerError("MODE_MISMATCH", "collision world and planner mode differ")
    if world.get("coordinate_frame") != config["coordinate_frame"]:
        raise PlannerError("FRAME_MISMATCH", "collision world and planner frame differ")
    if world.get("unit") != config["unit"]:
        raise PlannerError("UNIT_MISMATCH", "collision world and planner unit differ")
    if world.get("collision_query", {}).get("status") != "AVAILABLE":
        raise PlannerError("COLLISION_QUERY_UNAVAILABLE", "STEP 14 collision query is unavailable")
    if world.get("safety_margin_m") != 0.0:
        raise PlannerError("UNEXPECTED_COLLISION_MARGIN", "STEP 14 collision world must have zero added margin")


def grid_to_metric(row, col, grid):
    if isinstance(row, bool) or isinstance(col, bool) or not isinstance(row, int) or not isinstance(col, int):
        raise PlannerError("INVALID_GRID_INDEX", "grid index must be integer")
    if not (0 <= row < grid["rows"] and 0 <= col < grid["cols"]):
        raise PlannerError("INVALID_GRID_INDEX", "grid index is outside the map")
    resolution = grid["resolution_m"]
    return [
        grid["origin_m"][0] + (col + 0.5) * resolution,
        grid["origin_m"][1] + (row + 0.5) * resolution,
    ]


def metric_to_grid(x, y, grid):
    x = _finite(x, "POINT_NON_FINITE", "x")
    y = _finite(y, "POINT_NON_FINITE", "y")
    resolution = grid["resolution_m"]
    col = math.floor((x - grid["origin_m"][0]) / resolution)
    row = math.floor((y - grid["origin_m"][1]) / resolution)
    if not (0 <= row < grid["rows"] and 0 <= col < grid["cols"]):
        raise PlannerError("POINT_OUTSIDE_GRID", "metric point is outside the grid")
    return [row, col]


def clearance_at(x, y, collision_world):
    center = collision_world["workspace"]["valid_robot_center_bounds"]
    distances = [x - center["x_min"], center["x_max"] - x, y - center["y_min"], center["y_max"] - y]
    for obstacle in collision_world["static_obstacles"]:
        bounds = obstacle["collision_bounds"]
        dx = max(bounds["x_min"] - x, 0.0, x - bounds["x_max"])
        dy = max(bounds["y_min"] - y, 0.0, y - bounds["y_max"])
        distances.append(math.hypot(dx, dy))
    return max(0.0, min(distances))


def safety_risk_from_clearance(clearance_m, radius_m):
    clearance_m = _finite(clearance_m, "INVALID_CLEARANCE", "clearance")
    radius_m = _finite(radius_m, "INVALID_SAFETY_RADIUS", "safety radius")
    if clearance_m < 0 or radius_m <= 0:
        raise PlannerError("INVALID_CLEARANCE", "clearance must be non-negative and radius positive")
    return max(0.0, 1.0 - clearance_m / radius_m)


def build_grid_map(collision_world, config):
    validate_planner_config(config)
    _validate_world(collision_world, config)
    bounds = collision_world["workspace"]["raw_bounds"]
    resolution = config["grid_resolution_m"]
    cols_float = (bounds["x_max"] - bounds["x_min"]) / resolution
    rows_float = (bounds["y_max"] - bounds["y_min"]) / resolution
    cols, rows = round(cols_float), round(rows_float)
    if not math.isclose(cols_float, cols) or not math.isclose(rows_float, rows):
        raise PlannerError("GRID_RESOLUTION_MISMATCH", "resolution must divide the raw workspace")
    grid = {
        "schema_version": "0.1",
        "coordinate_frame": collision_world["coordinate_frame"],
        "unit": collision_world["unit"],
        "resolution_m": resolution,
        "rows": rows,
        "cols": cols,
        "origin_m": [bounds["x_min"], bounds["y_min"]],
        "cell_center_policy": "HALF_OPEN_CONTAINING_CELL_CENTER",
        "source_collision_representation": collision_world["collision_query"]["representation"],
        "additional_inflation_m": 0.0,
        "safety_cost_radius_m": config["safety_cost_radius_m"],
    }
    occupancy, risks = [], []
    free_count = 0
    for row in range(rows):
        occupied_row, risk_row = [], []
        for col in range(cols):
            x, y = grid_to_metric(row, col, grid)
            occupied = not is_robot_center_valid(x, y, collision_world)
            occupied_row.append(1 if occupied else 0)
            if occupied:
                risk_row.append(None)
            else:
                free_count += 1
                risk_row.append(safety_risk_from_clearance(
                    clearance_at(x, y, collision_world), config["safety_cost_radius_m"]
                ))
        occupancy.append(occupied_row)
        risks.append(risk_row)
    grid["occupancy"] = occupancy
    grid["safety_risk"] = risks
    grid["free_cell_count"] = free_count
    grid["occupied_cell_count"] = rows * cols - free_count
    return grid


def neighbors(cell, grid):
    row, col = cell
    result = []
    for name, dr, dc in NEIGHBORS:
        next_row, next_col = row + dr, col + dc
        if not (0 <= next_row < grid["rows"] and 0 <= next_col < grid["cols"]):
            continue
        if grid["occupancy"][next_row][next_col]:
            continue
        if dr and dc and (grid["occupancy"][row + dr][col] or grid["occupancy"][row][col + dc]):
            continue
        distance = grid["resolution_m"] * (math.sqrt(2.0) if dr and dc else 1.0)
        result.append(((next_row, next_col), distance, name))
    return result


def astar_grid(grid, start_cell, goal_cell, planner, safety_weight):
    if planner not in {"DISTANCE_ONLY_ASTAR", "SAFETY_COST_ASTAR"}:
        raise PlannerError("UNSUPPORTED_PLANNER", "unknown planner")
    start_cell, goal_cell = tuple(start_cell), tuple(goal_cell)
    for label, cell in (("START", start_cell), ("GOAL", goal_cell)):
        row, col = cell
        if not (0 <= row < grid["rows"] and 0 <= col < grid["cols"]) or grid["occupancy"][row][col]:
            raise PlannerError(f"{label}_IN_COLLISION", f"{label.lower()} cell is not traversable")
    weight = _finite(safety_weight, "INVALID_SAFETY_WEIGHT", "safety weight")
    if weight < 0:
        raise PlannerError("INVALID_SAFETY_WEIGHT", "safety weight must be non-negative")

    def heuristic(cell):
        return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1]) * grid["resolution_m"]

    sequence = itertools.count()
    open_heap = [(heuristic(start_cell), heuristic(start_cell), next(sequence), start_cell)]
    costs = {start_cell: 0.0}
    parents = {}
    closed = set()
    nodes_expanded = 0
    while open_heap:
        _, _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1
        if current == goal_cell:
            path = [current]
            while current in parents:
                current = parents[current]
                path.append(current)
            path.reverse()
            return {
                "status": "PATH_FOUND",
                "path_grid": [list(cell) for cell in path],
                "nodes_expanded": nodes_expanded,
                "total_cost": costs[goal_cell],
            }
        for next_cell, distance, _ in neighbors(current, grid):
            risk = grid["safety_risk"][next_cell[0]][next_cell[1]] or 0.0
            edge_cost = distance
            if planner == "SAFETY_COST_ASTAR":
                edge_cost += weight * risk * distance
            candidate = costs[current] + edge_cost
            if candidate < costs.get(next_cell, math.inf) - 1e-15:
                costs[next_cell] = candidate
                parents[next_cell] = current
                h = heuristic(next_cell)
                heapq.heappush(open_heap, (candidate + h, h, next(sequence), next_cell))
    return {"status": "NO_PATH", "path_grid": [], "nodes_expanded": nodes_expanded, "total_cost": None}


def _validate_endpoint(label, point, collision_world, grid):
    x = _finite(point.get("x"), f"{label}_NON_FINITE", f"{label.lower()} x")
    y = _finite(point.get("y"), f"{label}_NON_FINITE", f"{label.lower()} y")
    bounds = collision_world["workspace"]["valid_robot_center_bounds"]
    if x <= bounds["x_min"] or x >= bounds["x_max"] or y <= bounds["y_min"] or y >= bounds["y_max"]:
        raise PlannerError(f"{label}_OUTSIDE_WORKSPACE", f"{label.lower()} is outside valid robot-center workspace")
    if check_robot_center(x, y, collision_world)["status"] != "COLLISION_FREE":
        raise PlannerError(f"{label}_IN_COLLISION", f"{label.lower()} is in collision")
    cell = metric_to_grid(x, y, grid)
    if grid["occupancy"][cell[0]][cell[1]]:
        raise PlannerError(f"{label}_IN_COLLISION", f"mapped {label.lower()} cell is occupied")
    return cell


def path_length_m(path_world):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path_world, path_world[1:]))


def accumulated_safety_risk(path_grid, path_world, grid):
    return sum(
        grid["safety_risk"][cell[0]][cell[1]] * math.hypot(b[0] - a[0], b[1] - a[1])
        for cell, a, b in zip(path_grid[1:], path_world, path_world[1:])
    )


def validate_grid_path(path_grid, grid):
    if not path_grid:
        return False
    for cell in path_grid:
        if grid["occupancy"][cell[0]][cell[1]]:
            return False
    for current, following in zip(path_grid, path_grid[1:]):
        if tuple(following) not in {item[0] for item in neighbors(tuple(current), grid)}:
            return False
    return True


def plan_path(collision_world, grid, config, planner, start=None, goal=None):
    start = config["development_start"] if start is None else start
    goal = config["development_goal"] if goal is None else goal
    start_cell = _validate_endpoint("START", start, collision_world, grid)
    goal_cell = _validate_endpoint("GOAL", goal, collision_world, grid)
    started = perf_counter()
    search = astar_grid(grid, start_cell, goal_cell, planner, config["safety_weight"])
    planning_time_ms = (perf_counter() - started) * 1000.0
    result = {
        "status": search["status"],
        "path_found": search["status"] == "PATH_FOUND",
        "planner": planner,
        "start_input_m": [float(start["x"]), float(start["y"])],
        "goal_input_m": [float(goal["x"]), float(goal["y"])],
        "start_grid": start_cell,
        "goal_grid": goal_cell,
        "start_cell_center_m": grid_to_metric(*start_cell, grid),
        "goal_cell_center_m": grid_to_metric(*goal_cell, grid),
        "mapping_policy": grid["cell_center_policy"],
        "path_grid": search["path_grid"],
        "path_world": [],
        "planning_time_ms": planning_time_ms,
        "nodes_expanded": search["nodes_expanded"],
        "path_length_m": None,
        "minimum_clearance_m": None,
        "accumulated_safety_risk": None,
        "number_of_waypoints": 0,
        "total_objective_cost": search["total_cost"],
        "collision_free": False,
    }
    if search["status"] == "NO_PATH":
        return result
    path_world = [grid_to_metric(row, col, grid) for row, col in search["path_grid"]]
    if not validate_grid_path(search["path_grid"], grid):
        raise RuntimeError("A* returned an invalid path")
    result.update({
        "path_world": path_world,
        "path_length_m": path_length_m(path_world),
        "minimum_clearance_m": min(clearance_at(x, y, collision_world) for x, y in path_world),
        "accumulated_safety_risk": accumulated_safety_risk(search["path_grid"], path_world, grid),
        "number_of_waypoints": len(path_world),
        "collision_free": True,
    })
    return result


def semantic_path_result(result):
    return {key: value for key, value in result.items() if key != "planning_time_ms"}
