"""Deterministic metric goal-zone loading and lookup."""

import json
import math
from numbers import Real
from pathlib import Path

import cv2
import numpy as np


EPSILON = 1e-9


class ZoneValidationError(ValueError):
    """Raised when zone configuration or lookup input is invalid."""


def _finite_number(value):
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


def _point(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(
        _finite_number(coordinate) for coordinate in value
    ):
        raise ZoneValidationError(f"{name} must be a finite [x, y] point")
    return float(value[0]), float(value[1])


def _normalize_alias(text):
    if not isinstance(text, str) or not text.strip():
        raise ZoneValidationError("zone alias must be non-empty text")
    return " ".join(text.casefold().split())


def point_in_zone(x, y, zone):
    point = _point([x, y], "query point")
    polygon = [_point(vertex, "polygon vertex") for vertex in zone["polygon"]]

    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (point[0] - start[0]) * (end[1] - start[1]) - (
            point[1] - start[1]
        ) * (end[0] - start[0])
        if abs(cross) <= EPSILON and min(start[0], end[0]) - EPSILON <= point[0] <= max(
            start[0], end[0]
        ) + EPSILON and min(start[1], end[1]) - EPSILON <= point[1] <= max(
            start[1], end[1]
        ) + EPSILON:
            return "BOUNDARY"

    inside = False
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing_x = start[0] + (point[1] - start[1]) * (end[0] - start[0]) / (
                end[1] - start[1]
            )
            if point[0] < crossing_x:
                inside = not inside
    return "INSIDE" if inside else "OUTSIDE"


def validate_zones(config):
    if config.get("coordinate_frame") != "LEGACY_WORKSPACE_COORDINATE":
        raise ZoneValidationError("coordinate_frame must be LEGACY_WORKSPACE_COORDINATE")
    if config.get("unit") != "m":
        raise ZoneValidationError("zone coordinates must use unit m")
    if config.get("status") != "OFFLINE_DEVELOPMENT_ZONE_SET":
        raise ZoneValidationError("status must be OFFLINE_DEVELOPMENT_ZONE_SET")
    if not _finite_number(config.get("grid_spacing")) or config["grid_spacing"] <= 0:
        raise ZoneValidationError("grid_spacing must be finite and positive")

    bounds = config.get("workspace_bounds", {})
    xmin, xmax = _point(bounds.get("x"), "workspace x bounds")
    ymin, ymax = _point(bounds.get("y"), "workspace y bounds")
    if xmin >= xmax or ymin >= ymax:
        raise ZoneValidationError("workspace bounds must be ordered")

    zones = config.get("zones")
    if not isinstance(zones, list) or not zones:
        raise ZoneValidationError("zones must be a non-empty list")
    ids = set()
    alias_owner = {}
    polygons = []
    for index, zone in enumerate(zones, start=1):
        if not isinstance(zone, dict):
            raise ZoneValidationError(f"zone {index} must be an object")
        zone_id = zone.get("id")
        if not isinstance(zone_id, str) or not zone_id or zone_id in ids:
            raise ZoneValidationError(f"zone {index} has an invalid or duplicate ID")
        ids.add(zone_id)
        if zone.get("status") != "OFFLINE_DEVELOPMENT_ZONE" or not zone.get(
            "development_only"
        ) or zone.get("physical_validation") is not False or zone.get(
            "planning_validation"
        ) is not False:
            raise ZoneValidationError(f"{zone_id} must remain an unvalidated development zone")

        polygon = zone.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ZoneValidationError(f"{zone_id} polygon must contain at least three vertices")
        polygon = [_point(vertex, f"{zone_id} polygon vertex") for vertex in polygon]
        if any(not (xmin <= x <= xmax and ymin <= y <= ymax) for x, y in polygon):
            raise ZoneValidationError(f"{zone_id} polygon is outside workspace bounds")
        contour = np.asarray(polygon, dtype=np.float32)
        if not cv2.isContourConvex(contour):
            raise ZoneValidationError(f"{zone_id} polygon must be convex")
        polygons.append((zone_id, contour))

        center = _point(zone.get("center"), f"{zone_id} center")
        if point_in_zone(*center, zone) != "INSIDE":
            raise ZoneValidationError(f"{zone_id} center must be strictly inside its polygon")

        aliases = zone.get("aliases")
        if not isinstance(aliases, list) or not aliases:
            raise ZoneValidationError(f"{zone_id} aliases must be a non-empty list")
        for alias in [zone_id, *aliases]:
            normalized = _normalize_alias(alias)
            if normalized in alias_owner:
                raise ZoneValidationError(
                    f"duplicate zone alias {alias!r}: {alias_owner[normalized]} and {zone_id}"
                )
            alias_owner[normalized] = zone_id

    for index, (first_id, first) in enumerate(polygons):
        for second_id, second in polygons[index + 1 :]:
            area, _ = cv2.intersectConvexConvex(first, second)
            if area > EPSILON:
                raise ZoneValidationError(
                    f"zone interiors overlap: {first_id} and {second_id} ({area} m^2)"
                )
    return config


def load_zones(path):
    return validate_zones(json.loads(Path(path).read_text()))


def get_zone(zones, zone_id):
    for zone in zones["zones"]:
        if zone["id"] == zone_id:
            return zone
    raise ZoneValidationError(f"unknown zone ID: {zone_id}")


def get_zone_center(zones, zone_id):
    return tuple(get_zone(zones, zone_id)["center"])


def resolve_zone_alias(text, zones):
    normalized = _normalize_alias(text)
    for zone in zones["zones"]:
        if normalized in {
            _normalize_alias(alias) for alias in [zone["id"], *zone["aliases"]]
        }:
            return zone["id"]
    return "ZONE_NOT_RESOLVED"


def find_zone_for_point(x, y, zones):
    x, y = _point([x, y], "query point")
    xmin, xmax = zones["workspace_bounds"]["x"]
    ymin, ymax = zones["workspace_bounds"]["y"]
    if not (xmin - EPSILON <= x <= xmax + EPSILON and ymin - EPSILON <= y <= ymax + EPSILON):
        return {"status": "OUTSIDE_WORKSPACE", "zone_id": None, "zone_ids": []}

    statuses = [(zone["id"], point_in_zone(x, y, zone)) for zone in zones["zones"]]
    boundaries = [zone_id for zone_id, status in statuses if status == "BOUNDARY"]
    if boundaries:
        return {"status": "BOUNDARY", "zone_id": None, "zone_ids": boundaries}
    inside = [zone_id for zone_id, status in statuses if status == "INSIDE"]
    if len(inside) == 1:
        return {"status": "ZONE_FOUND", "zone_id": inside[0], "zone_ids": inside}
    if len(inside) > 1:
        raise ZoneValidationError(f"point is inside overlapping zones: {inside}")
    return {"status": "ZONE_NOT_FOUND", "zone_id": None, "zone_ids": []}


def world_to_undistorted_pixel(x, y, homography):
    x, y = _point([x, y], "world point")
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ZoneValidationError("homography must be a finite 3x3 matrix")
    try:
        transformed = np.linalg.inv(matrix) @ np.asarray([x, y, 1.0])
    except np.linalg.LinAlgError as exc:
        raise ZoneValidationError("homography is singular") from exc
    if not np.isfinite(transformed).all() or abs(transformed[2]) <= EPSILON:
        raise ZoneValidationError("inverse homography produced an invalid pixel")
    return float(transformed[0] / transformed[2]), float(transformed[1] / transformed[2])
