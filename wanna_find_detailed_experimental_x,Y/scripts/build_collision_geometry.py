#!/usr/bin/env python3
"""Build and visualize deterministic robot-only static collision geometry."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.planning import build_collision_world, check_robot_center, load_static_obstacles
from vlm_reposition.world import load_physical_geometry, load_zones


DEFAULT_ZONES = WORKSPACE / "config" / "zones.json"
DEFAULT_PHYSICAL = WORKSPACE / "config" / "physical_geometry.json"
DEFAULT_OBSTACLES = WORKSPACE / "config" / "static_obstacles.json"
DEFAULT_READINESS = WORKSPACE / "results" / "physical_geometry" / "physical_geometry_readiness.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "collision_geometry"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def bounds_polygon(bounds, pixel):
    return np.asarray([
        pixel((bounds["x_min"], bounds["y_min"])),
        pixel((bounds["x_max"], bounds["y_min"])),
        pixel((bounds["x_max"], bounds["y_max"])),
        pixel((bounds["x_min"], bounds["y_max"])),
    ], dtype=np.int32)


def draw_collision_world(world):
    image = np.full((900, 1150, 3), 255, dtype=np.uint8)
    origin, scale = (90, 800), 380.0

    def pixel(point):
        return int(origin[0] + point[0] * scale), int(origin[1] - point[1] * scale)

    raw_workspace = bounds_polygon(world["workspace"]["raw_bounds"], pixel)
    valid_center = bounds_polygon(world["workspace"]["valid_robot_center_bounds"], pixel)
    cv2.polylines(image, [raw_workspace], True, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.polylines(image, [valid_center], True, (0, 150, 0), 3, cv2.LINE_AA)

    overlay = image.copy()
    for obstacle in world["static_obstacles"]:
        expanded = bounds_polygon(obstacle["collision_bounds"], pixel)
        cv2.fillPoly(overlay, [expanded], (190, 190, 255))
    image = cv2.addWeighted(overlay, 0.55, image, 0.45, 0)
    for obstacle in world["static_obstacles"]:
        raw = bounds_polygon(obstacle["raw_bounds"], pixel)
        expanded = bounds_polygon(obstacle["collision_bounds"], pixel)
        cv2.polylines(image, [expanded], True, (0, 0, 210), 3, cv2.LINE_AA)
        cv2.polylines(image, [raw], True, (0, 140, 255), 3, cv2.LINE_AA)
        anchor = pixel((obstacle["raw_bounds"]["x_min"], obstacle["raw_bounds"]["y_max"]))
        cv2.putText(image, obstacle["id"], (anchor[0] + 5, anchor[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 180), 1, cv2.LINE_AA)

    cv2.putText(image, "RAW WORKSPACE", (90, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "GREEN: valid robot-center boundary", (600, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 130, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "ORANGE: raw synthetic obstacle", (600, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 120, 220), 2, cv2.LINE_AA)
    cv2.putText(image, "RED: robot-expanded collision obstacle", (600, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 190), 2, cv2.LINE_AA)

    inset_origin, inset_scale = (850, 285), 700.0
    footprint = np.asarray([
        [int(inset_origin[0] + x * inset_scale), int(inset_origin[1] - y * inset_scale)]
        for x, y in world["robot"]["footprint_m"]
    ], dtype=np.int32)
    cv2.fillPoly(image, [footprint], (215, 235, 255))
    cv2.polylines(image, [footprint], True, (0, 90, 180), 3, cv2.LINE_AA)
    cv2.arrowedLine(image, inset_origin, (1035, inset_origin[1]), (0, 0, 0), 2, tipLength=0.08)
    cv2.arrowedLine(image, inset_origin, (inset_origin[0], 100), (0, 0, 0), 2, tipLength=0.08)
    cv2.putText(image, "ROBOT FOOTPRINT / base_link", (670, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "+x", (1040, inset_origin[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(image, "+y", (inset_origin[0] + 8, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(image, "fixed yaw = 0 rad", (760, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 90, 180), 2, cv2.LINE_AA)
    cv2.putText(image, "No padding / no inflation / no path", (600, 850), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "X/Y [m]", (90, 840), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES)
    parser.add_argument("--physical", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--obstacles", type=Path, default=DEFAULT_OBSTACLES)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    zones = load_zones(args.zones)
    physical = load_physical_geometry(args.physical)
    obstacles = load_static_obstacles(args.obstacles)
    manipulation_readiness = read_json(args.readiness)["planning_geometry_readiness"]
    world = build_collision_world(zones, physical, obstacles, manipulation_readiness)

    expanded = world["static_obstacles"][0]["collision_bounds"]
    checks = {
        "inside_collision_obstacle": check_robot_center(0.5, 0.8, world),
        "outside_collision_obstacle": check_robot_center(0.9, 0.9, world),
        "touching_collision_obstacle": check_robot_center(expanded["x_min"], 0.8, world),
        "touching_workspace_boundary": check_robot_center(
            world["workspace"]["valid_robot_center_bounds"]["x_min"], 0.3, world
        ),
    }
    if [value["status"] for value in checks.values()] != [
        "COLLISION", "COLLISION_FREE", "COLLISION", "COLLISION"
    ]:
        raise RuntimeError("representative collision checks failed")

    args.output_dir.mkdir(parents=True)
    world_path = args.output_dir / "static_collision_world.json"
    write_json(world_path, world)
    visualization_path = args.output_dir / "static_collision_geometry_metric.png"
    if not cv2.imwrite(str(visualization_path), draw_collision_world(world)):
        raise RuntimeError(f"could not write {visualization_path}")
    summary = {
        "result": "PASS",
        "verification_type": "Static Obstacle Representation + Robot Collision Geometry",
        "sources": {
            "workspace": str(args.zones.resolve()),
            "robot_footprint": str(args.physical.resolve()),
            "static_obstacles": str(args.obstacles.resolve()),
            "manipulation_readiness": str(args.readiness.resolve()),
        },
        "mode": world["mode"],
        "fixed_yaw_rad": world["robot"]["fixed_yaw_rad"],
        "robot_half_extents_m": world["robot"]["half_extents_m"],
        "valid_robot_center_bounds": world["workspace"]["valid_robot_center_bounds"],
        "collision_obstacle_bounds": expanded,
        "touching_policy": world["touching_policy"],
        "safety_margin_m": world["safety_margin_m"],
        "representative_checks": checks,
        "overall_manipulation_geometry_readiness": manipulation_readiness,
        "robot_only_static_planning_geometry_readiness": "READY",
        "continuous_metric_representation": True,
        "occupancy_grid_generated": False,
        "path_planning": "NOT_RUN",
        "astar_implemented": False,
        "dynamic_replanning_implemented": False,
        "visualization": str(visualization_path.resolve()),
    }
    write_json(args.output_dir / "static_collision_geometry_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
