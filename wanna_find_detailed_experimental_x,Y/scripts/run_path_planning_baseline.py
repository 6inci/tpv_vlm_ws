#!/usr/bin/env python3
"""Run and visualize the offline STEP 15 A* planner comparison."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.planning import (
    build_grid_map,
    load_planner_config,
    metric_to_grid,
    plan_path,
    semantic_path_result,
)


DEFAULT_COLLISION_WORLD = WORKSPACE / "results" / "collision_geometry" / "static_collision_world.json"
DEFAULT_CONFIG = WORKSPACE / "config" / "planner_baseline.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "path_planning_baseline"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _grid_canvas(grid, start_cell, goal_cell, kind):
    cell_pixels = 8
    occupancy = np.asarray(grid["occupancy"], dtype=np.uint8)
    if kind == "occupancy":
        plot = np.full((*occupancy.shape, 3), 255, dtype=np.uint8)
        plot[occupancy == 1] = (35, 35, 35)
        title = "BINARY OCCUPANCY GRID"
    else:
        risks = np.asarray([[0.0 if value is None else value for value in row] for row in grid["safety_risk"]])
        plot = cv2.applyColorMap(np.rint(risks * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        plot[occupancy == 1] = (0, 0, 0)
        title = "OFFLINE DEVELOPMENT SAFETY COST LAYER"
    plot = cv2.resize(np.flipud(plot), (grid["cols"] * cell_pixels, grid["rows"] * cell_pixels), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((880, 820, 3), 255, dtype=np.uint8)
    x0, y0 = 70, 80
    canvas[y0:y0 + plot.shape[0], x0:x0 + plot.shape[1]] = plot
    cv2.rectangle(canvas, (x0, y0), (x0 + plot.shape[1], y0 + plot.shape[0]), (0, 0, 0), 2)

    def pixel(cell):
        row, col = cell
        return int(x0 + (col + 0.5) * cell_pixels), int(y0 + (grid["rows"] - row - 0.5) * cell_pixels)

    cv2.circle(canvas, pixel(start_cell), 7, (0, 180, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, pixel(goal_cell), 7, (220, 40, 220), -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "START", (580, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 150, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "GOAL", (580, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 20, 180), 2, cv2.LINE_AA)
    resolution_label = f"X/Y [m]  resolution={grid['resolution_m']:.2f} m"
    cv2.putText(canvas, resolution_label, (70, 835), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
    if kind == "occupancy":
        cv2.putText(canvas, "WHITE: FREE", (580, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, "BLACK: OCCUPIED", (580, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        gradient = np.arange(255, -1, -1, dtype=np.uint8).reshape(256, 1)
        colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
        colorbar = cv2.resize(colorbar, (28, 256), interpolation=cv2.INTER_NEAREST)
        canvas[300:556, 600:628] = colorbar
        cv2.putText(canvas, "1.0", (640, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.putText(canvas, "0.0", (640, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.putText(canvas, "Normalized Safety Risk", (565, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.putText(canvas, "BLACK: BLOCKED", (580, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return canvas


def _path_comparison(collision_world, distance_path, safety_path):
    image = np.full((880, 950, 3), 255, dtype=np.uint8)
    origin, scale = (80, 800), 400.0

    def pixel(point):
        return int(origin[0] + point[0] * scale), int(origin[1] - point[1] * scale)

    def rectangle(bounds):
        return np.asarray([
            pixel((bounds["x_min"], bounds["y_min"])),
            pixel((bounds["x_max"], bounds["y_min"])),
            pixel((bounds["x_max"], bounds["y_max"])),
            pixel((bounds["x_min"], bounds["y_max"])),
        ], dtype=np.int32)

    cv2.polylines(image, [rectangle(collision_world["workspace"]["raw_bounds"])], True, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.polylines(image, [rectangle(collision_world["workspace"]["valid_robot_center_bounds"])], True, (0, 150, 0), 2, cv2.LINE_AA)
    overlay = image.copy()
    for obstacle in collision_world["static_obstacles"]:
        cv2.fillPoly(overlay, [rectangle(obstacle["collision_bounds"])], (205, 205, 255))
    image = cv2.addWeighted(overlay, 0.55, image, 0.45, 0)
    for obstacle in collision_world["static_obstacles"]:
        cv2.polylines(image, [rectangle(obstacle["collision_bounds"])], True, (0, 0, 210), 3, cv2.LINE_AA)
        cv2.polylines(image, [rectangle(obstacle["raw_bounds"])], True, (0, 140, 255), 3, cv2.LINE_AA)
    distance_pixels = np.asarray([pixel(point) for point in distance_path["path_world"]], dtype=np.int32)
    safety_pixels = np.asarray([pixel(point) for point in safety_path["path_world"]], dtype=np.int32)
    cv2.polylines(image, [distance_pixels], False, (220, 80, 20), 3, cv2.LINE_AA)
    cv2.polylines(image, [safety_pixels], False, (180, 20, 180), 3, cv2.LINE_AA)
    start = pixel(distance_path["start_cell_center_m"])
    goal = pixel(distance_path["goal_cell_center_m"])
    cv2.circle(image, start, 7, (0, 160, 0), -1, cv2.LINE_AA)
    cv2.circle(image, goal, 7, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(image, "8-CONNECTED A* PATH COMPARISON", (575, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "BLUE: distance-only", (575, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 80, 20), 2, cv2.LINE_AA)
    cv2.putText(image, "MAGENTA: safety-cost", (575, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 20, 180), 2, cv2.LINE_AA)
    cv2.putText(image, "ORANGE: raw obstacle", (575, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 220), 1, cv2.LINE_AA)
    cv2.putText(image, "RED: expanded collision", (575, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 190), 1, cv2.LINE_AA)
    cv2.putText(image, "No smoothing / fixed yaw 0", (575, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "X/Y [m]", (80, 840), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def _planner_metrics(result):
    return {key: result[key] for key in (
        "path_found", "path_length_m", "planning_time_ms", "minimum_clearance_m",
        "accumulated_safety_risk", "nodes_expanded", "number_of_waypoints", "total_objective_cost",
    )}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-world", type=Path, default=DEFAULT_COLLISION_WORLD)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    collision_world = read_json(args.collision_world)
    config = load_planner_config(args.config)
    grid = build_grid_map(collision_world, config)
    distance_path = plan_path(collision_world, grid, config, "DISTANCE_ONLY_ASTAR")
    safety_path = plan_path(collision_world, grid, config, "SAFETY_COST_ASTAR")
    if not distance_path["collision_free"] or not safety_path["collision_free"]:
        raise RuntimeError("planner returned a colliding path")
    rebuilt_grid = build_grid_map(collision_world, config)
    rebuilt_distance = plan_path(collision_world, rebuilt_grid, config, "DISTANCE_ONLY_ASTAR")
    rebuilt_safety = plan_path(collision_world, rebuilt_grid, config, "SAFETY_COST_ASTAR")
    if grid != rebuilt_grid or semantic_path_result(distance_path) != semantic_path_result(rebuilt_distance) or semantic_path_result(safety_path) != semantic_path_result(rebuilt_safety):
        raise RuntimeError("semantic planner reconstruction is not deterministic")

    args.output_dir.mkdir(parents=True)
    occupancy_artifact = {key: value for key, value in grid.items() if key != "safety_risk"}
    safety_artifact = {
        "schema_version": "0.1",
        "coordinate_frame": grid["coordinate_frame"],
        "unit": grid["unit"],
        "rows": grid["rows"],
        "cols": grid["cols"],
        "resolution_m": grid["resolution_m"],
        "safety_cost_radius_m": grid["safety_cost_radius_m"],
        "risk_definition": "max(0, 1 - clearance / safety_cost_radius_m)",
        "clearance_reference": "ROBOT_EXPANDED_COLLISION_OBSTACLES_AND_VALID_ROBOT_CENTER_BOUNDARY",
        "occupied_value": None,
        "safety_risk": grid["safety_risk"],
    }
    write_json(args.output_dir / "binary_occupancy_grid.json", occupancy_artifact)
    write_json(args.output_dir / "safety_cost_layer.json", safety_artifact)
    write_json(args.output_dir / "distance_only_astar.json", distance_path)
    write_json(args.output_dir / "safety_cost_astar.json", safety_path)

    start_cell = metric_to_grid(config["development_start"]["x"], config["development_start"]["y"], grid)
    goal_cell = metric_to_grid(config["development_goal"]["x"], config["development_goal"]["y"], grid)
    images = {
        "binary_occupancy_grid": args.output_dir / "binary_occupancy_grid.png",
        "safety_costmap": args.output_dir / "safety_costmap.png",
        "astar_path_comparison": args.output_dir / "astar_path_comparison.png",
    }
    rendered = {
        images["binary_occupancy_grid"]: _grid_canvas(grid, start_cell, goal_cell, "occupancy"),
        images["safety_costmap"]: _grid_canvas(grid, start_cell, goal_cell, "safety"),
        images["astar_path_comparison"]: _path_comparison(collision_world, distance_path, safety_path),
    }
    for path, image in rendered.items():
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"could not write {path}")

    summary = {
        "result": "PASS",
        "verification_type": "Binary Occupancy Grid + Safety Cost Layer + 8-Connected A* Baseline",
        "sources": {
            "collision_world": str(args.collision_world.resolve()),
            "planner_config": str(args.config.resolve()),
        },
        "mode": config["mode"],
        "fixed_yaw_rad": config["fixed_yaw_rad"],
        "parameter_status": config["parameter_status"],
        "grid": {
            "resolution_m": grid["resolution_m"],
            "rows": grid["rows"],
            "cols": grid["cols"],
            "free_cell_count": grid["free_cell_count"],
            "occupied_cell_count": grid["occupied_cell_count"],
            "additional_inflation_m": grid["additional_inflation_m"],
        },
        "safety_cost": {
            "radius_m": config["safety_cost_radius_m"],
            "weight": config["safety_weight"],
            "representation": "OFFLINE_DEVELOPMENT_SAFETY_COST_LAYER",
            "nav2_costmap2d": False,
        },
        "start": distance_path["start_input_m"],
        "goal": distance_path["goal_input_m"],
        "start_cell_center_m": distance_path["start_cell_center_m"],
        "goal_cell_center_m": distance_path["goal_cell_center_m"],
        "mapping_policy": grid["cell_center_policy"],
        "neighbor_order": config["neighbor_order"],
        "distance_only_astar": _planner_metrics(distance_path),
        "safety_cost_astar": _planner_metrics(safety_path),
        "both_paths_collision_free": True,
        "semantic_reconstruction_identical": True,
        "path_smoothing": "NOT_RUN",
        "dynamic_replanning": "NOT_RUN",
        "real_robot_execution": "NOT_RUN",
        "manipulation": "NOT_RUN",
        "visualizations": {key: str(path.resolve()) for key, path in images.items()},
    }
    write_json(args.output_dir / "planner_comparison_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
