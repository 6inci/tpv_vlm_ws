#!/usr/bin/env python3
"""Validate offline metric goal zones and create deterministic visualizations."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.geometry import LegacyGeometry
from vlm_reposition.world import (
    find_zone_for_point,
    load_zones,
    resolve_zone_alias,
    world_to_undistorted_pixel,
)


DEFAULT_ZONES = WORKSPACE / "config" / "zones.json"
DEFAULT_GROUNDING_CONFIG = WORKSPACE / "config" / "qwen_grounding.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "goal_zones"
COLORS = [
    (255, 180, 80),
    (80, 180, 255),
    (120, 220, 120),
    (220, 120, 220),
    (220, 220, 100),
    (100, 150, 240),
]


def load_json(path):
    return json.loads(Path(path).read_text())


def load_target_membership(result_dir, zones):
    records = []
    result_paths = sorted(Path(result_dir).glob("*_run01.json"))
    if len(result_paths) != 3:
        raise RuntimeError(f"expected three STEP 08 run01 results, found {len(result_paths)}")
    for path in result_paths:
        result = load_json(path)
        world = result.get("world_coordinate")
        if result.get("target_status") != "TARGET_FOUND" or not isinstance(world, dict):
            raise RuntimeError(f"STEP 08 result lacks a grounded target: {path}")
        membership = find_zone_for_point(world["x"], world["y"], zones)
        records.append({
            "target": result["query"],
            "source_result": str(path),
            "world_x": world["x"],
            "world_y": world["y"],
            "unit": world["unit"],
            "coordinate_frame": world["frame"],
            "active_workspace_status": result["active_workspace_status"],
            "zone_result": membership["status"],
            "zone_id": membership["zone_id"],
            "boundary_zone_ids": membership["zone_ids"] if membership["status"] == "BOUNDARY" else [],
        })
    return records


def projected_polygon(zone, homography):
    return np.asarray(
        [world_to_undistorted_pixel(x, y, homography) for x, y in zone["polygon"]],
        dtype=np.int32,
    )


def draw_undistorted_zones(image, zones, geometry, targets):
    visual = image.copy()
    overlay = image.copy()
    for color, zone in zip(COLORS, zones["zones"]):
        polygon = projected_polygon(zone, geometry.homography)
        cv2.fillPoly(overlay, [polygon], color)
    visual = cv2.addWeighted(overlay, 0.20, visual, 0.80, 0)

    for color, zone in zip(COLORS, zones["zones"]):
        polygon = projected_polygon(zone, geometry.homography)
        center = world_to_undistorted_pixel(*zone["center"], geometry.homography)
        cv2.polylines(visual, [polygon], True, color, 4, cv2.LINE_AA)
        cv2.circle(visual, tuple(int(round(value)) for value in center), 7, color, -1)
        cv2.putText(
            visual,
            zone["id"],
            (int(center[0]) + 10, int(center[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    for target in targets:
        pixel = world_to_undistorted_pixel(
            target["world_x"], target["world_y"], geometry.homography
        )
        point = tuple(int(round(value)) for value in pixel)
        outside = target["zone_result"] == "OUTSIDE_WORKSPACE"
        color = (0, 0, 255) if outside else (0, 255, 0)
        suffix = " OUTSIDE ACTIVE WORKSPACE" if outside else f" {target['zone_id']}"
        cv2.circle(visual, point, 9, color, -1)
        cv2.putText(
            visual,
            target["target"] + suffix,
            (point[0] + 12, point[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        visual,
        zones["coordinate_frame"] + " / OFFLINE DEVELOPMENT ZONES",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return visual


def draw_metric_diagram(zones, targets):
    image = np.full((860, 1100, 3), 255, dtype=np.uint8)
    origin = (100, 780)
    scale = 400.0

    def pixel(point):
        return int(round(origin[0] + point[0] * scale)), int(
            round(origin[1] - point[1] * scale)
        )

    overlay = image.copy()
    for color, zone in zip(COLORS, zones["zones"]):
        polygon = np.asarray([pixel(point) for point in zone["polygon"]], dtype=np.int32)
        cv2.fillPoly(overlay, [polygon], color)
    image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)

    for color, zone in zip(COLORS, zones["zones"]):
        polygon = np.asarray([pixel(point) for point in zone["polygon"]], dtype=np.int32)
        center = pixel(zone["center"])
        cv2.polylines(image, [polygon], True, color, 3, cv2.LINE_AA)
        cv2.circle(image, center, 6, color, -1)
        cv2.putText(
            image,
            zone["id"],
            (center[0] - 35, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.arrowedLine(image, origin, (980, origin[1]), (0, 0, 0), 2, tipLength=0.03)
    cv2.arrowedLine(image, origin, (origin[0], 35), (0, 0, 0), 2, tipLength=0.03)
    for value in (0.0, 0.6, 1.2):
        point = pixel((value, 0.0))
        cv2.line(image, (point[0], point[1] - 5), (point[0], point[1] + 5), (0, 0, 0), 2)
        cv2.putText(image, f"{value:.1f}", (point[0] - 15, point[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    for value in (0.0, 0.6, 1.2, 1.8):
        point = pixel((0.0, value))
        cv2.line(image, (point[0] - 5, point[1]), (point[0] + 5, point[1]), (0, 0, 0), 2)
        cv2.putText(image, f"{value:.1f}", (point[0] - 45, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(image, "X [m]", (995, origin[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(image, "Y [m]", (35, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    for target in targets:
        point = pixel((target["world_x"], target["world_y"]))
        outside = target["zone_result"] == "OUTSIDE_WORKSPACE"
        color = (0, 0, 255) if outside else (0, 120, 0)
        cv2.circle(image, point, 8, color, -1)
        cv2.putText(
            image,
            target["target"] + (" OUTSIDE" if outside else ""),
            (point[0] + 10, point[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        "OFFLINE_DEVELOPMENT_ZONE_SET / LEGACY_WORKSPACE_COORDINATE",
        (100, 825),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES)
    parser.add_argument("--grounding-config", type=Path, default=DEFAULT_GROUNDING_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")

    zones = load_zones(args.zones)
    grounding_config = load_json(args.grounding_config)
    geometry = LegacyGeometry.load(
        grounding_config["calibration_path"], grounding_config["legacy_geometry_path"]
    )
    image = cv2.imread(grounding_config["undistorted_image_path"], cv2.IMREAD_COLOR)
    geometry.validate_image(image)
    targets = load_target_membership(grounding_config["task_grounding_result_dir"], zones)

    expected_ids = {f"zone_{letter}" for letter in "ABCDEF"}
    actual_ids = {zone["id"] for zone in zones["zones"]}
    if actual_ids != expected_ids:
        raise RuntimeError(f"expected zone_A through zone_F, received {sorted(actual_ids)}")
    alias_checks = [
        {"input": alias, "result": resolve_zone_alias(alias, zones), "expected": zone["id"]}
        for zone in zones["zones"]
        for alias in zone["aliases"]
    ]
    if any(check["result"] != check["expected"] for check in alias_checks):
        raise RuntimeError("zone alias verification failed")
    if any(
        target["active_workspace_status"] == "OUTSIDE"
        and target["zone_result"] != "OUTSIDE_WORKSPACE"
        for target in targets
    ):
        raise RuntimeError("an outside target was incorrectly assigned to a zone")

    undistorted_visual = draw_undistorted_zones(image, zones, geometry, targets)
    metric_visual = draw_metric_diagram(zones, targets)
    args.output_dir.mkdir(parents=True)
    undistorted_path = args.output_dir / "photo_04_undistorted_zones.png"
    metric_path = args.output_dir / "goal_zones_metric.png"
    if not cv2.imwrite(str(undistorted_path), undistorted_visual):
        raise RuntimeError(f"could not write {undistorted_path}")
    if not cv2.imwrite(str(metric_path), metric_visual):
        raise RuntimeError(f"could not write {metric_path}")

    membership_path = args.output_dir / "target_zone_membership.json"
    membership_path.write_text(json.dumps(targets, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "result": "PASS",
        "verification_type": "Offline Metric Goal Zone Definition + Verification",
        "created_at": datetime.now().astimezone().isoformat(),
        "zone_config": str(args.zones),
        "coordinate_frame": zones["coordinate_frame"],
        "unit": zones["unit"],
        "zone_status": zones["status"],
        "workspace_bounds": zones["workspace_bounds"],
        "grid_spacing": zones["grid_spacing"],
        "zone_count": len(zones["zones"]),
        "validation": {
            "unique_zone_ids": True,
            "finite_polygons": True,
            "polygons_within_workspace": True,
            "centers_strictly_inside": True,
            "duplicate_aliases": False,
            "interior_overlap_m2": 0.0,
            "shared_edges_allowed": True,
        },
        "boundary_policy": "Return BOUNDARY with every touching zone ID; never select one silently.",
        "alias_checks": alias_checks,
        "unknown_alias_checks": [
            {"input": text, "result": resolve_zone_alias(text, zones)}
            for text in ("저쪽", "왼쪽", "저기", "가까운 곳")
        ],
        "target_membership": targets,
        "inverse_homography_usage": "visualization only",
        "undistorted_visualization": str(undistorted_path),
        "metric_visualization": str(metric_path),
    }
    (args.output_dir / "zone_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"result": summary["result"], "target_membership": targets}, ensure_ascii=False, indent=2))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
