#!/usr/bin/env python3
"""Run deterministic static-planner stress scenarios and parameter sweeps."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.planning import (
    build_collision_world,
    build_grid_map,
    is_robot_center_valid,
    load_planner_config,
    plan_path,
    semantic_path_result,
    validate_planner_config,
    validate_static_obstacles,
)
from vlm_reposition.world import load_physical_geometry, load_zones


DEFAULT_SCENARIOS = WORKSPACE / "config" / "planner_stress_scenarios.json"
DEFAULT_PLANNER = WORKSPACE / "config" / "planner_baseline.json"
DEFAULT_ZONES = WORKSPACE / "config" / "zones.json"
DEFAULT_PHYSICAL = WORKSPACE / "config" / "physical_geometry.json"
DEFAULT_READINESS = WORKSPACE / "results" / "physical_geometry" / "physical_geometry_readiness.json"
DEFAULT_BASELINE_WORLD = WORKSPACE / "results" / "collision_geometry" / "static_collision_world.json"
DEFAULT_BASELINE_DISTANCE = WORKSPACE / "results" / "path_planning_baseline" / "distance_only_astar.json"
DEFAULT_BASELINE_SAFETY = WORKSPACE / "results" / "path_planning_baseline" / "safety_cost_astar.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "path_planning_stress"


class StressConfigError(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def scenario_obstacle_config(config, scenario):
    return {
        "schema_version": "0.1",
        "coordinate_frame": config["coordinate_frame"],
        "unit": config["unit"],
        "scenario_type": "ROBOT_ONLY_STATIC_PLANNING_DEVELOPMENT",
        "mode": config["mode"],
        "fixed_yaw_rad": config["fixed_yaw_rad"],
        "touching_policy": "COLLISION",
        "safety_margin_m": 0.0,
        "obstacles": deepcopy(scenario["static_obstacles"]),
    }


def validate_stress_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != "0.1":
        raise StressConfigError("stress scenario schema_version must be 0.1")
    required = {
        "coordinate_frame": "LEGACY_WORKSPACE_COORDINATE",
        "unit": "m",
        "mode": "ROBOT_ONLY_TRANSLATIONAL_BASELINE",
        "fixed_yaw_rad": 0.0,
        "geometry_status": "DEVELOPMENT_SYNTHETIC",
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise StressConfigError("stress scenario metadata is invalid")
    if not isinstance(config.get("source_detail"), str) or not config["source_detail"]:
        raise StressConfigError("stress scenario provenance is required")
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise StressConfigError("scenarios must be a non-empty list")
    ids = set()
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id") if isinstance(scenario, dict) else None
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in ids:
            raise StressConfigError("scenario ID must be non-empty and unique")
        ids.add(scenario_id)
        if not isinstance(scenario.get("description"), str) or not scenario["description"]:
            raise StressConfigError(f"{scenario_id} description is required")
        if scenario.get("geometry_status") != "DEVELOPMENT_SYNTHETIC":
            raise StressConfigError(f"{scenario_id} must be DEVELOPMENT_SYNTHETIC")
        if scenario.get("expected_high_level_result") not in {"PATH_EXISTS", "NO_PATH_EXPECTED"}:
            raise StressConfigError(f"{scenario_id} expected result is invalid")
        for endpoint in ("start", "goal"):
            point = scenario.get(endpoint)
            if not isinstance(point, dict) or point.get("status") != "DEVELOPMENT_DEFINED":
                raise StressConfigError(f"{scenario_id} {endpoint} must be DEVELOPMENT_DEFINED")
            for axis in ("x", "y"):
                value = point.get(axis)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise StressConfigError(f"{scenario_id} {endpoint}.{axis} must be finite")
        try:
            validate_static_obstacles(scenario_obstacle_config(config, scenario))
        except ValueError as error:
            raise StressConfigError(f"{scenario_id}: {error}") from error
    required_ids = {
        "S1_baseline", "S2_shifted_left", "S3_two_obstacles", "S4_narrow_passage",
        "S5_completely_blocked", "S6_alternate_endpoints",
    }
    if ids != required_ids:
        raise StressConfigError("the six required stress scenarios must be present")
    return config


def load_stress_config(path):
    return validate_stress_config(read_json(path))


def scenario_by_id(config, scenario_id):
    return next(scenario for scenario in config["scenarios"] if scenario["scenario_id"] == scenario_id)


def planner_config_for_scenario(base_config, scenario):
    config = deepcopy(base_config)
    config["development_start"] = deepcopy(scenario["start"])
    config["development_goal"] = deepcopy(scenario["goal"])
    return validate_planner_config(config)


def occupancy_digest(grid):
    payload = json.dumps(grid["occupancy"], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def risk_digest(grid):
    payload = json.dumps(grid["safety_risk"], separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def independent_path_validation(result, world, grid):
    if result["status"] == "NO_PATH":
        valid = not result["path_grid"] and not result["path_world"] and not result["path_found"]
        return {
            "status": "NO_PATH_CONFIRMED" if valid else "INVALID_NO_PATH_RESULT",
            "no_partial_path": valid,
            "collision_free": None,
            "valid_adjacency": None,
            "corner_cutting_absent": None,
        }
    cells, points = result["path_grid"], result["path_world"]
    occupancy_free = len(cells) == len(points) and all(grid["occupancy"][row][col] == 0 for row, col in cells)
    continuous_free = all(is_robot_center_valid(x, y, world) for x, y in points)
    bounds = world["workspace"]["valid_robot_center_bounds"]
    inside_bounds = all(
        bounds["x_min"] < x < bounds["x_max"] and bounds["y_min"] < y < bounds["y_max"]
        for x, y in points
    )
    valid_adjacency = True
    no_corner_cutting = True
    for (row, col), (next_row, next_col) in zip(cells, cells[1:]):
        dr, dc = next_row - row, next_col - col
        if (dr, dc) == (0, 0) or abs(dr) > 1 or abs(dc) > 1:
            valid_adjacency = False
        if dr and dc and (grid["occupancy"][row + dr][col] or grid["occupancy"][row][col + dc]):
            no_corner_cutting = False
    valid = occupancy_free and continuous_free and inside_bounds and valid_adjacency and no_corner_cutting
    return {
        "status": "PATH_VALID" if valid else "PATH_INVALID",
        "waypoints_free": occupancy_free,
        "continuous_collision_query_free": continuous_free,
        "inside_valid_robot_center_workspace": inside_bounds,
        "valid_adjacency": valid_adjacency,
        "corner_cutting_absent": no_corner_cutting,
        "collision_free": valid,
    }


def build_scenario_world(stress_config, scenario, zones, physical, readiness):
    return build_collision_world(
        zones,
        physical,
        scenario_obstacle_config(stress_config, scenario),
        readiness,
    )


def run_scenario(stress_config, scenario, zones, physical, readiness, base_planner_config):
    world = build_scenario_world(stress_config, scenario, zones, physical, readiness)
    planner_config = planner_config_for_scenario(base_planner_config, scenario)
    grid = build_grid_map(world, planner_config)
    distance = plan_path(world, grid, planner_config, "DISTANCE_ONLY_ASTAR")
    safety = plan_path(world, grid, planner_config, "SAFETY_COST_ASTAR")
    validations = {
        "distance_only_astar": independent_path_validation(distance, world, grid),
        "safety_cost_astar": independent_path_validation(safety, world, grid),
    }
    expected_status = "PATH_FOUND" if scenario["expected_high_level_result"] == "PATH_EXISTS" else "NO_PATH"
    if distance["status"] != expected_status or safety["status"] != expected_status:
        raise RuntimeError(f"{scenario['scenario_id']} did not match its high-level expected result")
    if any(value["status"] not in {"PATH_VALID", "NO_PATH_CONFIRMED"} for value in validations.values()):
        raise RuntimeError(f"{scenario['scenario_id']} independent validation failed")

    rebuilt_grid = build_grid_map(world, planner_config)
    rebuilt_distance = plan_path(world, rebuilt_grid, planner_config, "DISTANCE_ONLY_ASTAR")
    rebuilt_safety = plan_path(world, rebuilt_grid, planner_config, "SAFETY_COST_ASTAR")
    deterministic = {
        "occupancy_identical": grid["occupancy"] == rebuilt_grid["occupancy"],
        "safety_risk_identical": grid["safety_risk"] == rebuilt_grid["safety_risk"],
        "distance_path_semantic_identical": semantic_path_result(distance) == semantic_path_result(rebuilt_distance),
        "safety_path_semantic_identical": semantic_path_result(safety) == semantic_path_result(rebuilt_safety),
    }
    if not all(deterministic.values()):
        raise RuntimeError(f"{scenario['scenario_id']} is not deterministic")

    result = {
        "scenario_id": scenario["scenario_id"],
        "description": scenario["description"],
        "geometry_status": scenario["geometry_status"],
        "expected_high_level_result": scenario["expected_high_level_result"],
        "result": "PASS",
        "start": deepcopy(scenario["start"]),
        "goal": deepcopy(scenario["goal"]),
        "static_obstacles": deepcopy(world["static_obstacles"]),
        "grid": {
            "resolution_m": grid["resolution_m"],
            "rows": grid["rows"],
            "cols": grid["cols"],
            "cell_count": grid["rows"] * grid["cols"],
            "free_cell_count": grid["free_cell_count"],
            "occupied_cell_count": grid["occupied_cell_count"],
            "additional_inflation_m": grid["additional_inflation_m"],
            "occupancy_sha256": occupancy_digest(grid),
            "safety_risk_sha256": risk_digest(grid),
        },
        "distance_only_astar": distance,
        "safety_cost_astar": safety,
        "independent_validation": validations,
        "determinism": deterministic,
    }
    return result, world, grid, planner_config


def metric_record(result):
    return {key: result[key] for key in (
        "status", "path_found", "path_length_m", "minimum_clearance_m", "accumulated_safety_risk",
        "nodes_expanded", "number_of_waypoints", "planning_time_ms", "total_objective_cost", "collision_free",
    )}


def run_resolution_sweep(world, scenario, base_config):
    entries = []
    for resolution in (0.01, 0.02, 0.04):
        config = planner_config_for_scenario(base_config, scenario)
        config["grid_resolution_m"] = resolution
        grid = build_grid_map(world, config)
        distance = plan_path(world, grid, config, "DISTANCE_ONLY_ASTAR")
        safety = plan_path(world, grid, config, "SAFETY_COST_ASTAR")
        if independent_path_validation(distance, world, grid)["status"] != "PATH_VALID" or independent_path_validation(safety, world, grid)["status"] != "PATH_VALID":
            raise RuntimeError(f"resolution {resolution} produced an invalid path")
        entries.append({
            "grid_resolution_m": resolution,
            "parameter_status": "DEVELOPMENT_PARAMETER",
            "grid_cell_count": grid["rows"] * grid["cols"],
            "free_cell_count": grid["free_cell_count"],
            "occupied_cell_count": grid["occupied_cell_count"],
            "additional_inflation_m": grid["additional_inflation_m"],
            "collision_bounds": [deepcopy(item["collision_bounds"]) for item in world["static_obstacles"]],
            "distance_only_astar": metric_record(distance),
            "safety_cost_astar": metric_record(safety),
        })
    return {"sweep": "GRID_RESOLUTION", "one_factor_at_a_time": True, "entries": entries}


def run_safety_radius_sweep(world, scenario, base_config):
    entries = []
    for radius in (0.10, 0.20, 0.30):
        config = planner_config_for_scenario(base_config, scenario)
        config["safety_cost_radius_m"] = radius
        grid = build_grid_map(world, config)
        result = plan_path(world, grid, config, "SAFETY_COST_ASTAR")
        if independent_path_validation(result, world, grid)["status"] != "PATH_VALID":
            raise RuntimeError(f"safety radius {radius} produced an invalid path")
        entries.append({
            "safety_cost_radius_m": radius,
            "parameter_status": "DEVELOPMENT_PARAMETER",
            "occupancy_sha256": occupancy_digest(grid),
            "additional_inflation_m": grid["additional_inflation_m"],
            "safety_cost_astar": metric_record(result),
        })
    if len({entry["occupancy_sha256"] for entry in entries}) != 1:
        raise RuntimeError("safety radius changed hard occupancy")
    return {"sweep": "SAFETY_RADIUS", "one_factor_at_a_time": True, "entries": entries}


def run_safety_weight_sweep(world, scenario, base_config):
    config = planner_config_for_scenario(base_config, scenario)
    grid = build_grid_map(world, config)
    digest = occupancy_digest(grid)
    entries = []
    for weight in (0.0, 0.5, 1.0, 2.0, 4.0):
        weighted_config = deepcopy(config)
        weighted_config["safety_weight"] = weight
        result = plan_path(world, grid, weighted_config, "SAFETY_COST_ASTAR")
        if independent_path_validation(result, world, grid)["status"] != "PATH_VALID":
            raise RuntimeError(f"safety weight {weight} produced an invalid path")
        entries.append({
            "safety_weight": weight,
            "parameter_status": "DEVELOPMENT_PARAMETER",
            "occupancy_sha256": digest,
            "additional_inflation_m": grid["additional_inflation_m"],
            "safety_cost_astar": metric_record(result),
        })
    distance = plan_path(world, grid, config, "DISTANCE_ONLY_ASTAR")
    if not math.isclose(entries[0]["safety_cost_astar"]["total_objective_cost"], distance["total_objective_cost"]):
        raise RuntimeError("zero safety weight is not distance-objective equivalent")
    return {
        "sweep": "SAFETY_WEIGHT",
        "one_factor_at_a_time": True,
        "zero_weight_distance_objective_equivalent": True,
        "entries": entries,
    }


def draw_scenario(result, world):
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
        cv2.fillPoly(overlay, [rectangle(obstacle["collision_bounds"])], (205, 205, 255))
    image = cv2.addWeighted(overlay, 0.55, image, 0.45, 0)
    for obstacle in world["static_obstacles"]:
        cv2.polylines(image, [rectangle(obstacle["collision_bounds"])], True, (0, 0, 210), 3, cv2.LINE_AA)
        cv2.polylines(image, [rectangle(obstacle["raw_bounds"])], True, (0, 140, 255), 3, cv2.LINE_AA)
    distance, safety = result["distance_only_astar"], result["safety_cost_astar"]
    if distance["path_found"]:
        cv2.polylines(image, [np.asarray([pixel(point) for point in distance["path_world"]], dtype=np.int32)], False, (220, 80, 20), 3, cv2.LINE_AA)
    if safety["path_found"]:
        cv2.polylines(image, [np.asarray([pixel(point) for point in safety["path_world"]], dtype=np.int32)], False, (180, 20, 180), 3, cv2.LINE_AA)
    start = pixel((result["start"]["x"], result["start"]["y"]))
    goal = pixel((result["goal"]["x"], result["goal"]["y"]))
    cv2.circle(image, start, 7, (0, 160, 0), -1, cv2.LINE_AA)
    cv2.circle(image, goal, 7, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(image, result["scenario_id"], (580, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, result["expected_high_level_result"], (580, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "BLUE: distance-only", (580, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 80, 20), 2, cv2.LINE_AA)
    cv2.putText(image, "MAGENTA: safety-cost", (580, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 20, 180), 2, cv2.LINE_AA)
    cv2.putText(image, "ORANGE: raw obstacle", (580, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 120, 220), 1, cv2.LINE_AA)
    cv2.putText(image, "RED: expanded collision", (580, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 190), 1, cv2.LINE_AA)
    if not distance["path_found"]:
        cv2.putText(image, "NO_PATH confirmed", (580, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 190), 2, cv2.LINE_AA)
    cv2.putText(image, "DEVELOPMENT_SYNTHETIC", (580, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "X/Y [m]", (80, 840), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def draw_line_chart(title, x_values, series, x_label, y_label):
    image = np.full((620, 900, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 100, 830, 80, 500
    values = [value for _, items, _ in series for value in items]
    minimum, maximum = min(values), max(values)
    if math.isclose(minimum, maximum):
        delta = max(abs(minimum) * 0.25, 0.01)
        minimum -= delta
        maximum += delta
    padding = (maximum - minimum) * 0.08
    minimum, maximum = minimum - padding, maximum + padding
    cv2.line(image, (left, bottom), (right, bottom), (0, 0, 0), 2)
    cv2.line(image, (left, bottom), (left, top), (0, 0, 0), 2)

    def point(index, value):
        x = left + int(index * (right - left) / max(1, len(x_values) - 1))
        y = bottom - int((value - minimum) * (bottom - top) / (maximum - minimum))
        return x, y

    for label, items, color in series:
        points = np.asarray([point(index, value) for index, value in enumerate(items)], dtype=np.int32)
        cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)
        for coordinate in points:
            cv2.circle(image, tuple(coordinate), 5, color, -1, cv2.LINE_AA)
        legend_y = 115 + 30 * [item[0] for item in series].index(label)
        cv2.putText(image, label, (640, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    for index, value in enumerate(x_values):
        x, _ = point(index, minimum)
        cv2.putText(image, str(value), (x - 14, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
    cv2.putText(image, title, (60, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, x_label, (390, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.putText(image, y_label, (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
    cv2.putText(image, f"min={minimum + padding:.6g} max={maximum - padding:.6g}", (105, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
    return image


def write_visualizations(directory, scenario_outputs, contexts, resolution, radius, weight):
    paths = []
    for scenario_id, result in scenario_outputs.items():
        suffix = "path_comparison" if result["distance_only_astar"]["path_found"] else "overview"
        path = directory / f"{scenario_id}_{suffix}.png"
        if not cv2.imwrite(str(path), draw_scenario(result, contexts[scenario_id][0])):
            raise RuntimeError(f"could not write {path}")
        paths.append(path)

    colors = {"distance": (220, 80, 20), "safety": (180, 20, 180)}
    resolution_entries = resolution["entries"]
    resolution_specs = (
        ("planning_time", "planning_time_ms", "Planning time [ms]"),
        ("nodes_expanded", "nodes_expanded", "Nodes expanded"),
        ("path_length", "path_length_m", "Path length [m]"),
    )
    for filename, key, label in resolution_specs:
        chart = draw_line_chart(
            f"Resolution sensitivity: {label}",
            [entry["grid_resolution_m"] for entry in resolution_entries],
            [
                ("distance-only", [entry["distance_only_astar"][key] for entry in resolution_entries], colors["distance"]),
                ("safety-cost", [entry["safety_cost_astar"][key] for entry in resolution_entries], colors["safety"]),
            ],
            "Grid resolution [m]", label,
        )
        path = directory / f"resolution_{filename}.png"
        if not cv2.imwrite(str(path), chart):
            raise RuntimeError(f"could not write {path}")
        paths.append(path)

    for sweep, prefix, x_key, x_label in (
        (radius, "safety_radius", "safety_cost_radius_m", "Safety radius [m]"),
        (weight, "safety_weight", "safety_weight", "Safety weight"),
    ):
        for filename, key, label in (
            ("path_length", "path_length_m", "Path length [m]"),
            ("minimum_clearance", "minimum_clearance_m", "Minimum clearance [m]"),
            ("accumulated_risk", "accumulated_safety_risk", "Accumulated safety risk"),
        ):
            entries = sweep["entries"]
            chart = draw_line_chart(
                f"{x_label} sensitivity: {label}",
                [entry[x_key] for entry in entries],
                [("safety-cost", [entry["safety_cost_astar"][key] for entry in entries], colors["safety"])],
                x_label, label,
            )
            path = directory / f"{prefix}_{filename}.png"
            if not cv2.imwrite(str(path), chart):
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
    parser.add_argument("--baseline-world", type=Path, default=DEFAULT_BASELINE_WORLD)
    parser.add_argument("--baseline-distance", type=Path, default=DEFAULT_BASELINE_DISTANCE)
    parser.add_argument("--baseline-safety", type=Path, default=DEFAULT_BASELINE_SAFETY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    stress_config = load_stress_config(args.scenarios)
    planner_config = load_planner_config(args.planner)
    zones = load_zones(args.zones)
    physical = load_physical_geometry(args.physical)
    readiness = read_json(args.readiness)["planning_geometry_readiness"]
    scenario_outputs, contexts = {}, {}
    for scenario in stress_config["scenarios"]:
        result, world, grid, local_config = run_scenario(
            stress_config, scenario, zones, physical, readiness, planner_config
        )
        scenario_outputs[scenario["scenario_id"]] = result
        contexts[scenario["scenario_id"]] = (world, grid, local_config)

    s1 = scenario_outputs["S1_baseline"]
    s1_world = contexts["S1_baseline"][0]
    baseline_regression = {
        "collision_world_identical": s1_world == read_json(args.baseline_world),
        "distance_path_semantic_identical": semantic_path_result(s1["distance_only_astar"]) == semantic_path_result(read_json(args.baseline_distance)),
        "safety_path_semantic_identical": semantic_path_result(s1["safety_cost_astar"]) == semantic_path_result(read_json(args.baseline_safety)),
    }
    if not all(baseline_regression.values()):
        raise RuntimeError("S1 did not reproduce the STEP 15 baseline")

    s1_scenario = scenario_by_id(stress_config, "S1_baseline")
    resolution = run_resolution_sweep(s1_world, s1_scenario, planner_config)
    radius = run_safety_radius_sweep(s1_world, s1_scenario, planner_config)
    weight = run_safety_weight_sweep(s1_world, s1_scenario, planner_config)

    scenario_dir = args.output_dir / "scenario_results"
    sensitivity_dir = args.output_dir / "parameter_sensitivity"
    visualization_dir = args.output_dir / "visualizations"
    scenario_dir.mkdir(parents=True)
    sensitivity_dir.mkdir()
    visualization_dir.mkdir()
    for scenario_id, result in scenario_outputs.items():
        write_json(scenario_dir / f"{scenario_id}.json", result)
    write_json(sensitivity_dir / "resolution_sweep.json", resolution)
    write_json(sensitivity_dir / "safety_radius_sweep.json", radius)
    write_json(sensitivity_dir / "safety_weight_sweep.json", weight)
    visualization_paths = write_visualizations(
        visualization_dir, scenario_outputs, contexts, resolution, radius, weight
    )

    path_expected = sum(result["expected_high_level_result"] == "PATH_EXISTS" for result in scenario_outputs.values())
    no_path_expected = len(scenario_outputs) - path_expected
    path_found_results = [
        planner
        for result in scenario_outputs.values()
        for planner in (result["distance_only_astar"], result["safety_cost_astar"])
        if planner["status"] == "PATH_FOUND"
    ]
    no_path_results = [
        planner
        for result in scenario_outputs.values()
        for planner in (result["distance_only_astar"], result["safety_cost_astar"])
        if planner["status"] == "NO_PATH"
    ]
    summary = {
        "result": "PASS",
        "verification_type": "Static Path Stress Test + Parameter Sensitivity",
        "sources": {
            "stress_scenarios": str(args.scenarios.resolve()),
            "planner_config": str(args.planner.resolve()),
            "step15_collision_world": str(args.baseline_world.resolve()),
            "step15_distance_path": str(args.baseline_distance.resolve()),
            "step15_safety_path": str(args.baseline_safety.resolve()),
        },
        "scenario_count": len(scenario_outputs),
        "path_exists_expected_count": path_expected,
        "no_path_expected_count": no_path_expected,
        "planner_run_count": len(scenario_outputs) * 2,
        "planner_success_count": len(path_found_results) + len(no_path_results),
        "path_found_count": len(path_found_results),
        "no_path_confirmation_count": len(no_path_results),
        "collision_free_validation_count": sum(result["collision_free"] for result in path_found_results),
        "parameter_sweep_setting_count": len(resolution["entries"]) + len(radius["entries"]) + len(weight["entries"]),
        "parameter_sweep_planner_run_count": len(resolution["entries"]) * 2 + len(radius["entries"]) + len(weight["entries"]),
        "baseline_regression": baseline_regression,
        "scenario_results": {
            scenario_id: {
                "expected": result["expected_high_level_result"],
                "distance_status": result["distance_only_astar"]["status"],
                "safety_status": result["safety_cost_astar"]["status"],
                "deterministic": all(result["determinism"].values()),
            }
            for scenario_id, result in scenario_outputs.items()
        },
        "hard_occupancy_unchanged_by_safety_radius": len({entry["occupancy_sha256"] for entry in radius["entries"]}) == 1,
        "hard_occupancy_unchanged_by_safety_weight": len({entry["occupancy_sha256"] for entry in weight["entries"]}) == 1,
        "additional_inflation_m": 0.0,
        "parameter_optimization": "NOT_RUN",
        "path_smoothing": "NOT_RUN",
        "dynamic_replanning": "NOT_RUN",
        "real_robot_execution": "NOT_RUN",
        "manipulation": "NOT_RUN",
        "visualizations": [str(path.resolve()) for path in visualization_paths],
    }
    write_json(args.output_dir / "stress_test_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
