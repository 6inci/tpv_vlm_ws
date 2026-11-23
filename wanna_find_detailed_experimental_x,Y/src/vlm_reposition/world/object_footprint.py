"""Physical obstacle footprint contract before any planning representation."""

from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path


GEOMETRY_SOURCES = (
    "MEASURED_PHYSICAL_GEOMETRY",
    "KNOWN_EXPERIMENT_OBJECT_GEOMETRY",
    "PLANAR_GROUND_CONTACT_GEOMETRY",
    "DEVELOPMENT_SYNTHETIC",
    "UNKNOWN",
)
SHAPE_TYPES = frozenset({"CIRCLE", "AXIS_ALIGNED_RECTANGLE", "POLYGON", "UNKNOWN"})
ANCHOR_STATUSES = frozenset({
    "MEASURED_FOOTPRINT_CENTER",
    "MEASURED_CONTACT_REFERENCE",
    "VALIDATED_GROUND_REFERENCE",
    "DEVELOPMENT_ASSUMED_CENTER",
    "UNKNOWN",
})
PROHIBITED_SOURCES = frozenset({"IMAGE_BBOX", "VLM_BBOX", "SEMANTIC_LABEL", "VLM_GENERATED_DIMENSIONS"})


class ObjectFootprintError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ObjectFootprintError("INVALID_GEOMETRY", f"{name} must be finite")
    return float(value)


def _positive(value, name):
    value = _finite(value, name)
    if value <= 0:
        raise ObjectFootprintError("INVALID_DIMENSION", f"{name} must be positive")
    return value


def _point(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ObjectFootprintError("INVALID_GEOMETRY", f"{name} must be [x,y]")
    return [_finite(value[0], f"{name}.x"), _finite(value[1], f"{name}.y")]


def _polygon(value):
    if not isinstance(value, list) or len(value) < 3:
        raise ObjectFootprintError("INVALID_POLYGON", "polygon must contain at least three points")
    points = [_point(point, f"polygon[{index}]") for index, point in enumerate(value)]
    if len({tuple(point) for point in points}) < 3:
        raise ObjectFootprintError("INVALID_POLYGON", "polygon must contain three distinct points")
    area2 = sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )
    if math.isclose(area2, 0.0):
        raise ObjectFootprintError("INVALID_POLYGON", "polygon area must be non-zero")
    return points


def rectangle_local_footprint(length_m, width_m):
    half_x = _positive(length_m, "length_m") / 2
    half_y = _positive(width_m, "width_m") / 2
    return [[-half_x, -half_y], [half_x, -half_y], [half_x, half_y], [-half_x, half_y]]


def _validate_source(record):
    source = record.get("geometry_source")
    if source in PROHIBITED_SOURCES:
        raise ObjectFootprintError("PROHIBITED_GEOMETRY_SOURCE", f"{source} is not physical geometry evidence")
    if source not in GEOMETRY_SOURCES:
        raise ObjectFootprintError("INVALID_GEOMETRY_SOURCE", "unsupported physical geometry source")
    expected_status = (
        "UNKNOWN" if source == "UNKNOWN"
        else "DEVELOPMENT_SYNTHETIC" if source == "DEVELOPMENT_SYNTHETIC"
        else "CONFIRMED"
    )
    if record.get("source_status") != expected_status:
        raise ObjectFootprintError("SOURCE_STATUS_MISMATCH", "source status does not match geometry source")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("source_path") or not provenance.get("source_detail"):
        raise ObjectFootprintError("MISSING_PROVENANCE", "geometry source requires path and detail")
    required_source_types = {
        "MEASURED_PHYSICAL_GEOMETRY": "PHYSICAL_MEASUREMENT",
        "KNOWN_EXPERIMENT_OBJECT_GEOMETRY": "VERIFIED_EXPERIMENT_OBJECT_GEOMETRY",
        "PLANAR_GROUND_CONTACT_GEOMETRY": "VALIDATED_PLANAR_GROUND_CONTACT_GEOMETRY",
        "DEVELOPMENT_SYNTHETIC": "STEP14_EXISTING_DEVELOPMENT_SYNTHETIC",
        "UNKNOWN": "REPOSITORY_AUDIT_NO_MEASUREMENT",
    }
    if provenance.get("source_type") != required_source_types[source]:
        raise ObjectFootprintError("SOURCE_PROVENANCE_MISMATCH", "provenance type does not match geometry source")
    if source == "PLANAR_GROUND_CONTACT_GEOMETRY" and provenance.get("all_boundary_points_validated_on_ground_plane") is not True:
        raise ObjectFootprintError("UNVALIDATED_GROUND_CONTACT", "planar source requires validated ground-plane boundary points")


def _validate_anchor(record):
    anchor = record.get("anchor")
    if not isinstance(anchor, dict) or not isinstance(anchor.get("type"), str):
        raise ObjectFootprintError("INVALID_ANCHOR", "anchor definition is required")
    status = anchor.get("status")
    if status not in ANCHOR_STATUSES:
        raise ObjectFootprintError("INVALID_ANCHOR_STATUS", "unsupported anchor status")
    offset = anchor.get("offset_local_m")
    if status == "UNKNOWN":
        if offset is not None:
            raise ObjectFootprintError("UNKNOWN_ANCHOR_HAS_OFFSET", "unknown anchor cannot have an offset")
    else:
        _point(offset, "anchor.offset_local_m")
    if status == "DEVELOPMENT_ASSUMED_CENTER" and record.get("geometry_source") != "DEVELOPMENT_SYNTHETIC":
        raise ObjectFootprintError("DEVELOPMENT_ANCHOR_SOURCE_MISMATCH", "assumed center is development-only")


def _validate_orientation(record):
    orientation = record.get("orientation")
    if not isinstance(orientation, dict) or orientation.get("status") not in {"NOT_REQUIRED", "KNOWN", "UNKNOWN"}:
        raise ObjectFootprintError("INVALID_ORIENTATION", "orientation status is invalid")
    required = record.get("orientation_required") is True
    if required and orientation["status"] == "KNOWN":
        _finite(orientation.get("yaw_rad"), "orientation.yaw_rad")
    elif orientation.get("yaw_rad") is not None:
        raise ObjectFootprintError("INVALID_ORIENTATION", "yaw is allowed only for KNOWN orientation")
    if not required and orientation["status"] != "NOT_REQUIRED":
        raise ObjectFootprintError("INVALID_ORIENTATION", "orientation must be NOT_REQUIRED for this shape")


def validate_geometry_record(record):
    if not isinstance(record, dict) or not record.get("object_id") or not record.get("semantic_label"):
        raise ObjectFootprintError("INVALID_SCHEMA", "geometry record requires object identity")
    _validate_source(record)
    shape = record.get("shape_type")
    if shape not in SHAPE_TYPES:
        raise ObjectFootprintError("INVALID_SHAPE_TYPE", "unsupported footprint shape")
    _validate_anchor(record)
    _validate_orientation(record)
    if not isinstance(record.get("uncertainty"), dict):
        raise ObjectFootprintError("INVALID_UNCERTAINTY", "uncertainty must remain an explicit separate layer")
    if record.get("planning_ready") is not False:
        raise ObjectFootprintError("PLANNER_INJECTION_NOT_ALLOWED", "STEP 23 records cannot be planner-ready")
    if "safety_margin_m" in record or "robot_inflation_m" in record:
        raise ObjectFootprintError("PHYSICAL_LAYER_CONTAMINATION", "safety and robot inflation are not physical dimensions")
    if record.get("planning_geometry") != {"status": "NOT_GENERATED"}:
        raise ObjectFootprintError("PLANNING_GEOMETRY_NOT_SEPARATE", "planning geometry must remain NOT_GENERATED")

    if record["geometry_source"] == "UNKNOWN":
        if shape != "UNKNOWN" or record.get("dimensions_m") is not None or record.get("physical_footprint_local") is not None:
            raise ObjectFootprintError("UNKNOWN_SOURCE_HAS_GEOMETRY", "unknown source cannot create physical geometry")
        return record
    if shape == "UNKNOWN":
        raise ObjectFootprintError("MISSING_DIMENSIONS", "known source requires a footprint shape")

    local = build_local_footprint(record, validate=False)
    supplied = record.get("physical_footprint_local")
    if supplied is not None and supplied != local:
        raise ObjectFootprintError("FOOTPRINT_SOURCE_MISMATCH", "stored local footprint differs from source dimensions")
    return record


def validate_physical_obstacle_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != "0.1":
        raise ObjectFootprintError("INVALID_SCHEMA", "physical obstacle geometry schema_version must be 0.1")
    if config.get("coordinate_frame") != "LEGACY_WORKSPACE_COORDINATE" or config.get("unit") != "m":
        raise ObjectFootprintError("FRAME_OR_UNIT_MISMATCH", "physical obstacle geometry must use legacy workspace meters")
    policy = config.get("source_policy", {})
    if policy.get("priority") != list(GEOMETRY_SOURCES):
        raise ObjectFootprintError("INVALID_SOURCE_PRIORITY", "geometry source priority is invalid")
    if set(config.get("allowed_shape_types", [])) != SHAPE_TYPES:
        raise ObjectFootprintError("INVALID_SHAPE_POLICY", "shape policy is invalid")
    records = config.get("representative_observations")
    fixtures = config.get("development_fixtures")
    if not isinstance(records, list) or not records or not isinstance(fixtures, list) or not fixtures:
        raise ObjectFootprintError("INVALID_SCHEMA", "representative observations and fixtures are required")
    for item in records:
        validate_geometry_record(item["physical_geometry"])
    for record in fixtures:
        validate_geometry_record(record)
    return config


def load_physical_obstacle_config(path):
    return validate_physical_obstacle_config(json.loads(Path(path).read_text()))


def build_local_footprint(record, validate=True):
    if validate:
        _validate_source(record)
    if record.get("geometry_source") == "UNKNOWN":
        raise ObjectFootprintError("UNKNOWN_SOURCE", "physical footprint cannot be created from UNKNOWN")
    shape = record.get("shape_type")
    dimensions = record.get("dimensions_m")
    if shape == "CIRCLE":
        if not isinstance(dimensions, dict):
            raise ObjectFootprintError("MISSING_DIMENSIONS", "circle diameter is required")
        diameter = _positive(dimensions.get("diameter_m"), "diameter_m")
        return {"center_xy_m": [0.0, 0.0], "radius_m": diameter / 2}
    if shape == "AXIS_ALIGNED_RECTANGLE":
        if not isinstance(dimensions, dict):
            raise ObjectFootprintError("MISSING_DIMENSIONS", "rectangle dimensions are required")
        return rectangle_local_footprint(dimensions.get("length_m"), dimensions.get("width_m"))
    if shape == "POLYGON":
        return _polygon(record.get("physical_footprint_local"))
    raise ObjectFootprintError("INVALID_SHAPE_TYPE", "known geometry requires a supported shape")


def planning_readiness(record):
    validate_geometry_record(record)
    if record["geometry_source"] == "UNKNOWN":
        return "NOT_READY_UNKNOWN_SOURCE"
    if record["anchor"]["status"] == "UNKNOWN":
        return "NOT_READY_MISSING_ANCHOR"
    if record.get("orientation_required") and record["orientation"]["status"] == "UNKNOWN":
        return "NOT_READY_MISSING_ORIENTATION"
    if record["geometry_source"] == "DEVELOPMENT_SYNTHETIC":
        return "DEVELOPMENT_ONLY"
    return "READY"


def place_world_footprint(record, anchor_world_xy, coordinate_frame="LEGACY_WORKSPACE_COORDINATE"):
    readiness = planning_readiness(record)
    if readiness not in {"READY", "DEVELOPMENT_ONLY"}:
        raise ObjectFootprintError(readiness, "geometry is not ready for world placement")
    anchor = _point(anchor_world_xy, "anchor_world_xy")
    offset = _point(record["anchor"]["offset_local_m"], "anchor.offset_local_m")
    origin = [anchor[0] + offset[0], anchor[1] + offset[1]]
    local = build_local_footprint(record)
    yaw = record["orientation"].get("yaw_rad") or 0.0

    if record["shape_type"] == "CIRCLE":
        radius = local["radius_m"]
        world = {"center_xy_m": origin, "radius_m": radius}
        bounds = {"x_min": origin[0] - radius, "x_max": origin[0] + radius,
                  "y_min": origin[1] - radius, "y_max": origin[1] + radius}
    else:
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        points = [
            [origin[0] + x * cos_yaw - y * sin_yaw, origin[1] + x * sin_yaw + y * cos_yaw]
            for x, y in local
        ]
        world = points
        bounds = {"x_min": min(p[0] for p in points), "x_max": max(p[0] for p in points),
                  "y_min": min(p[1] for p in points), "y_max": max(p[1] for p in points)}
    return {
        "object_id": record["object_id"],
        "geometry_source": record["geometry_source"],
        "planning_geometry_readiness": readiness,
        "shape_type": record["shape_type"],
        "coordinate_frame": coordinate_frame,
        "unit": "m",
        "anchor_world_xy": anchor,
        "physical_footprint_local": deepcopy(local),
        "physical_footprint_world": world,
        "physical_bounds_world_m": bounds,
        "planning_geometry": {"status": "NOT_GENERATED"},
    }


def measurement_requirements(record):
    if record["geometry_source"] != "UNKNOWN":
        return []
    return [
        "ground footprint dimensions (length and width, diameter, or measured polygon)",
        "anchor relation between the observed bottom-center ground-reference point and the footprint",
        "orientation reference if the selected footprint is orientation-sensitive",
    ]
