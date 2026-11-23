#!/usr/bin/env python3
"""Audit physical geometry and build planning-relevant representation states."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.world import build_physical_planning_state, load_physical_geometry


DEFAULT_CONFIG = WORKSPACE / "config" / "physical_geometry.json"
DEFAULT_STATE_DIR = WORKSPACE / "results" / "workspace_state"
DEFAULT_OUTPUT = WORKSPACE / "results" / "physical_geometry"
MECANUM_ROOT = Path("/home/binci/mecanum_ros2")

CASES = (
    ("red_cup_zone_a_state.json", "red_cup_zone_a_physical_state.json"),
    ("white_bag_zone_b_state.json", "white_bag_zone_b_physical_state.json"),
    ("trophy_zone_f_state.json", "trophy_zone_f_physical_state.json"),
)

NEEDED_MEASUREMENTS = [
    "gripper front extension from base_link origin",
    "gripper maximum outer collision width in open and closed states",
    "gripper grasp/holding reference offset",
    "red cup ground-plane footprint diameter or width/length",
    "white bag footprint in the defined experimental loading state",
    "gold trophy-like object ground-plane footprint width/length",
    "metric polygons for only the chairs/desks selected as static planning obstacles",
]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw_metric(config, workspace_state, physical_state):
    image = np.full((820, 1200, 3), 255, dtype=np.uint8)

    local_origin, local_scale = (270, 410), 1100.0
    footprint = physical_state["physical_geometry"]["robot"]["footprint_m"]
    local_polygon = np.asarray([
        [int(local_origin[0] + x * local_scale), int(local_origin[1] - y * local_scale)]
        for x, y in footprint
    ], dtype=np.int32)
    cv2.fillPoly(image, [local_polygon], (210, 235, 255))
    cv2.polylines(image, [local_polygon], True, (0, 90, 180), 3, cv2.LINE_AA)
    cv2.arrowedLine(image, local_origin, (470, local_origin[1]), (0, 0, 0), 2, tipLength=0.08)
    cv2.arrowedLine(image, local_origin, (local_origin[0], 180), (0, 0, 0), 2, tipLength=0.08)
    cv2.putText(image, "+x forward", (475, local_origin[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "+y left", (local_origin[0] + 8, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "CONFIRMED ROBOT FOOTPRINT / base_link", (55, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "0.290 m x 0.246 m", (170, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 90, 180), 2, cv2.LINE_AA)
    cv2.putText(image, "Robot pose unavailable: local-frame panel only", (65, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    bounds = workspace_state["workspace"]["bounds"]
    workspace_origin, workspace_scale = (720, 720), 340.0

    def pixel(point):
        return int(workspace_origin[0] + point[0] * workspace_scale), int(workspace_origin[1] - point[1] * workspace_scale)

    top_left = pixel((bounds["x_min"], bounds["y_max"]))
    bottom_right = pixel((bounds["x_max"], bounds["y_min"]))
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 3)
    colors = {
        "TASK_TARGET": (0, 0, 220),
        "DYNAMIC_ENTITY": (0, 140, 255),
        "STATIC_OBSTACLE_CANDIDATE": (180, 80, 20),
        "UNASSIGNED": (120, 120, 120),
    }
    for entity in physical_state["physical_geometry"]["planning_entities"]:
        world = entity["world_coordinate"]
        if entity["active_workspace_status"] != "INSIDE":
            continue
        point = pixel((world["x"], world["y"]))
        cv2.circle(image, point, 6, colors[entity["planning_role"]], -1)
        cv2.putText(image, entity["scene_entity_id"], (point[0] + 7, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colors[entity["planning_role"]], 1, cv2.LINE_AA)
    cv2.putText(image, "WORKSPACE REFERENCE POINTS ONLY", (730, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "No obstacle footprints / no robot pose / no path", (720, 765), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(image, (600, 30), (600, 790), (180, 180, 180), 2)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    config = load_physical_geometry(args.config)
    outputs = []
    states = []
    for state_name, output_name in CASES:
        state_path = args.state_dir / state_name
        workspace_state = read_json(state_path)
        physical_state = build_physical_planning_state(workspace_state, config, state_path)
        states.append((workspace_state, physical_state))
        outputs.append((output_name, physical_state))

    args.output_dir.mkdir(parents=True)
    for output_name, state in outputs:
        write_json(args.output_dir / output_name, state)

    nav2_path = MECANUM_ROOT / "src/mecanum_bringup/config/nav2_params.yaml"
    urdf_path = MECANUM_ROOT / "src/mecanum_bringup/urdf/mecanum_robot.urdf.xacro"
    readme_path = MECANUM_ROOT / "README.md"
    audit = {
        "result": "PASS",
        "audit_type": "PHYSICAL_GEOMETRY_SOURCE_AUDIT",
        "searched_roots": [str(WORKSPACE), "/home/binci/qwen_vision_test", str(MECANUM_ROOT)],
        "confirmed": [
            {
                "item": "robot planning footprint",
                "value_m": config["robot"]["source_footprint_m"],
                "extent_m": {"length": config["robot"]["length_m"], "width": config["robot"]["width_m"]},
                "reference_frame": "base_link",
                "source": str(nav2_path),
                "source_sha256": sha256(nav2_path),
                "evidence_scope": "existing recorded Nav2 planning footprint; not freshly remeasured in STEP 13",
            },
            {
                "item": "mecanum wheel/CAD reference geometry",
                "wheel_radius_m": 0.04,
                "wheel_center_front_rear_m": 0.192091,
                "wheel_center_left_right_m": 0.207001,
                "source": str(urdf_path),
                "source_sha256": sha256(urdf_path),
                "planning_use": "NOT_USED_TO_REPLACE_THE_RECORDED_FOOTPRINT",
            },
            {
                "item": "URDF simplified chassis collision box",
                "value_m": {"length": 0.25, "width": 0.20, "height": 0.08},
                "source": str(urdf_path),
                "source_sha256": sha256(urdf_path),
                "planning_use": "NOT_SELECTED_AS_FULL_ROBOT_FOOTPRINT",
                "reason": "URDF labels it as a retained simplified chassis collision shape; Nav2 provides the larger explicit planning footprint",
            },
        ],
        "inferred": [],
        "unknown": [
            "gripper physical footprint, front extension, opening and holding offset",
            "red cup physical footprint",
            "white bag physical footprint and loading-state convention",
            "gold trophy-like object physical footprint",
            "physical polygons for selected chair/desk static obstacle candidates",
        ],
        "excluded_from_physical_geometry": [
            {
                "item": "Nav2 footprint_padding 0.01 m",
                "reason": "planning padding is not a measured physical dimension and inflation is outside STEP 13",
            },
            {
                "item": "Nav2 inflation_radius 0.55 m",
                "reason": "costmap policy is not physical geometry and inflation is outside STEP 13",
            },
            {
                "item": "Qwen image bboxes",
                "reason": "perspective image bounds are not physical footprints",
            },
        ],
        "additional_sources": [
            {"path": str(readme_path), "sha256": sha256(readme_path)},
            {"path": "/home/binci/vlm_ws/docs/RESEARCH_SCOPE.md"},
            {"path": "/home/binci/vlm_ws/results/workspace_state/"},
        ],
        "planning_consequence": "robot geometry is READY; gripper, target objects and static obstacle footprints are NOT_READY",
        "needed_measurements": NEEDED_MEASUREMENTS,
        "reference_repository_modified": False,
    }
    write_json(args.output_dir / "physical_geometry_audit.json", audit)

    readiness = {
        "step_result": "PASS",
        "planning_geometry_readiness": "PARTIAL",
        "representation_status": "VALID",
        "robot_geometry_readiness": "READY",
        "gripper_geometry_readiness": "NOT_READY",
        "object_geometry_readiness": {
            name: "NOT_READY" for name in config["movable_objects"]
        },
        "static_obstacle_geometry_readiness": "NOT_READY",
        "physical_state_results": [
            {
                "result_file": str((args.output_dir / output_name).resolve()),
                "target": state["workspace_task_summary"]["target"],
                "overall_readiness": state["planning_geometry_readiness"]["overall"],
                "collision_checking": state["collision_checking"]["status"],
                "planning": state["planning"]["status"],
            }
            for (output_name, state) in outputs
        ],
        "needed_measurements": NEEDED_MEASUREMENTS,
    }
    write_json(args.output_dir / "physical_geometry_readiness.json", readiness)

    visualization_path = args.output_dir / "physical_geometry_metric.png"
    if not cv2.imwrite(str(visualization_path), draw_metric(config, *states[0])):
        raise RuntimeError(f"could not write {visualization_path}")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
