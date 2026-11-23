"""User physical-measurement intake for the STEP 23 footprint contract."""

from copy import deepcopy
import math
from numbers import Real

from .object_footprint import (
    ObjectFootprintError,
    place_world_footprint,
    planning_readiness,
    validate_geometry_record,
)


UNIT_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0}
MEASUREMENT_STATUSES = frozenset({
    "READY",
    "WAITING_FOR_USER_MEASUREMENT",
    "NOT_READY_MISSING_DIMENSIONS",
    "NOT_READY_MISSING_ANCHOR",
    "NOT_READY_MISSING_ORIENTATION",
    "INVALID_MEASUREMENT",
})
REAL_ANCHOR_STATUSES = frozenset({
    "MEASURED_FOOTPRINT_CENTER",
    "MEASURED_CONTACT_REFERENCE",
    "VALIDATED_GROUND_REFERENCE",
})


class MeasurementRegistrationError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def to_m(value, unit):
    if unit not in UNIT_TO_M:
        raise MeasurementRegistrationError("INVALID_UNIT", "unit must be mm, cm, or m")
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise MeasurementRegistrationError("INVALID_VALUE", "measurement must be finite")
    return float(value) * UNIT_TO_M[unit]


def _positive_m(value, unit, name):
    result = to_m(value, unit)
    if result <= 0:
        raise MeasurementRegistrationError("INVALID_DIMENSION", f"{name} must be positive")
    return result


def _point_m(value, unit, name):
    if not isinstance(value, list) or len(value) != 2:
        raise MeasurementRegistrationError("INVALID_POINT", f"{name} must be [x,y]")
    return [to_m(value[0], unit), to_m(value[1], unit)]


def _result(status, missing=None, **extra):
    return {"status": status, "missing_fields": missing or [], **extra}


def _canonical_shape(measurement):
    shape = measurement.get("shape_type")
    if shape not in {"CIRCLE", "AXIS_ALIGNED_RECTANGLE", "POLYGON"}:
        return None, None, None
    if shape == "POLYGON":
        points = measurement.get("polygon_local")
        unit = measurement.get("polygon_unit")
        if points is None or unit is None:
            return shape, None, None
        if not isinstance(points, list):
            raise MeasurementRegistrationError("INVALID_POLYGON", "polygon_local must be a list")
        return shape, None, [_point_m(point, unit, "polygon vertex") for point in points]

    dimensions = measurement.get("dimensions")
    if not isinstance(dimensions, dict) or dimensions.get("unit") is None:
        return shape, None, None
    unit = dimensions["unit"]
    if shape == "CIRCLE":
        if dimensions.get("radius") is None:
            return shape, None, None
        radius = _positive_m(dimensions["radius"], unit, "radius")
        return shape, {"diameter_m": 2 * radius}, None
    if dimensions.get("length") is None or dimensions.get("width") is None:
        return shape, None, None
    return shape, {
        "length_m": _positive_m(dimensions["length"], unit, "length"),
        "width_m": _positive_m(dimensions["width"], unit, "width"),
    }, None


def assess_measurement(measurement, source_path="config/representative_object_measurement.json"):
    if measurement is None:
        return _result("WAITING_FOR_USER_MEASUREMENT", ["shape_type", "dimensions_or_polygon", "anchor", "orientation_if_required"])
    if not isinstance(measurement, dict):
        return _result("INVALID_MEASUREMENT", error_code="INVALID_SCHEMA", error="measurement must be an object", raw_input=deepcopy(measurement))
    if measurement.get("measurement_status") == "USER_MEASUREMENT_REQUIRED":
        return _result("WAITING_FOR_USER_MEASUREMENT", ["shape_type", "dimensions_or_polygon", "anchor", "orientation_if_required"])
    raw = deepcopy(measurement)
    try:
        if not isinstance(measurement, dict) or measurement.get("schema_version") != "0.1":
            raise MeasurementRegistrationError("INVALID_SCHEMA", "measurement schema_version must be 0.1")
        if not measurement.get("object_id") or not measurement.get("semantic_label"):
            raise MeasurementRegistrationError("INVALID_SCHEMA", "object identity is required")
        if measurement.get("measurement_status") != "USER_MEASUREMENT_PROVIDED":
            raise MeasurementRegistrationError("INVALID_STATUS", "actual input must declare USER_MEASUREMENT_PROVIDED")
        if measurement.get("geometry_source") != "MEASURED_PHYSICAL_GEOMETRY":
            raise MeasurementRegistrationError("INVALID_SOURCE", "actual input must be a physical measurement")

        shape, dimensions_m, polygon_m = _canonical_shape(measurement)
        if shape is None:
            return _result("NOT_READY_MISSING_DIMENSIONS", ["shape_type", "dimensions_or_polygon"], raw_input=raw)
        if dimensions_m is None and polygon_m is None:
            return _result("NOT_READY_MISSING_DIMENSIONS", ["dimensions_or_polygon"], raw_input=raw)

        anchor = measurement.get("anchor")
        if not isinstance(anchor, dict) or anchor.get("type") in {None, "UNKNOWN"}:
            return _result("NOT_READY_MISSING_ANCHOR", ["anchor.type", "anchor.offset_local", "anchor.unit", "anchor.measurement_method"], raw_input=raw)
        if anchor.get("type") not in REAL_ANCHOR_STATUSES:
            raise MeasurementRegistrationError("INVALID_ANCHOR", "actual measurement requires a measured or validated anchor")
        if anchor.get("offset_local") is None or anchor.get("unit") is None or not anchor.get("measurement_method"):
            return _result("NOT_READY_MISSING_ANCHOR", ["anchor.offset_local", "anchor.unit", "anchor.measurement_method"], raw_input=raw)
        anchor_offset_m = _point_m(anchor["offset_local"], anchor["unit"], "anchor.offset_local")

        orientation_required = shape != "CIRCLE"
        orientation = measurement.get("orientation")
        if orientation_required and (
            not isinstance(orientation, dict)
            or orientation.get("status") != "KNOWN"
            or orientation.get("value_rad") is None
            or not orientation.get("reference_definition")
        ):
            return _result("NOT_READY_MISSING_ORIENTATION", ["orientation.status", "orientation.reference_definition", "orientation.value_rad"], raw_input=raw)
        if orientation_required:
            yaw_rad = orientation["value_rad"]
            if isinstance(yaw_rad, bool) or not isinstance(yaw_rad, Real) or not math.isfinite(yaw_rad):
                raise MeasurementRegistrationError("INVALID_ORIENTATION", "orientation value_rad must be finite")
            yaw_rad = float(yaw_rad)
            orientation_record = {"status": "KNOWN", "yaw_rad": yaw_rad}
        else:
            orientation_record = {"status": "NOT_REQUIRED", "yaw_rad": None}

        uncertainty = measurement.get("measurement_uncertainty")
        uncertainty_m = "UNKNOWN"
        if uncertainty is not None:
            if not isinstance(uncertainty, dict) or uncertainty.get("value") is None or uncertainty.get("unit") is None:
                raise MeasurementRegistrationError("INVALID_UNCERTAINTY", "uncertainty requires value and unit")
            uncertainty_m = to_m(uncertainty["value"], uncertainty["unit"])
            if uncertainty_m < 0:
                raise MeasurementRegistrationError("INVALID_UNCERTAINTY", "uncertainty cannot be negative")

        canonical = {
            "object_id": measurement["object_id"],
            "semantic_label": measurement["semantic_label"],
            "geometry_source": "MEASURED_PHYSICAL_GEOMETRY",
            "source_status": "CONFIRMED",
            "shape_type": shape,
            "dimensions_m": dimensions_m,
            "anchor": {"type": anchor["type"], "status": anchor["type"], "offset_local_m": anchor_offset_m},
            "orientation_required": orientation_required,
            "orientation": orientation_record,
            "physical_footprint_local": polygon_m,
            "uncertainty": {"measurement_uncertainty_m": uncertainty_m},
            "planning_ready": False,
            "planning_geometry": {"status": "NOT_GENERATED"},
            "provenance": {
                "source_type": "PHYSICAL_MEASUREMENT",
                "source": "USER_PHYSICAL_MEASUREMENT",
                "source_path": str(source_path),
                "source_detail": anchor["measurement_method"],
                "raw_input": raw,
                "input_unit": measurement.get("polygon_unit") if shape == "POLYGON" else measurement["dimensions"]["unit"],
                "canonical_metric_values": {
                    "dimensions_m": deepcopy(dimensions_m),
                    "polygon_local_m": deepcopy(polygon_m),
                    "anchor_offset_local_m": anchor_offset_m,
                    "orientation_value_rad": orientation_record["yaw_rad"],
                    "measurement_uncertainty_m": uncertainty_m,
                },
                "validation_status": "READY",
            },
        }
        validate_geometry_record(canonical)
        if planning_readiness(canonical) != "READY":
            raise MeasurementRegistrationError("READINESS_MISMATCH", "validated measurement did not become READY")
        return _result("READY", physical_geometry=canonical, raw_input=raw)
    except (MeasurementRegistrationError, ObjectFootprintError) as exc:
        return _result("INVALID_MEASUREMENT", error_code=exc.code, error=str(exc), raw_input=raw)


def register_ready_measurement(registry, assessment):
    if not isinstance(registry, dict) or registry.get("schema_version") != "0.1" or not isinstance(registry.get("objects"), list):
        raise MeasurementRegistrationError("INVALID_REGISTRY", "registry schema is invalid")
    if assessment.get("status") != "READY" or "physical_geometry" not in assessment:
        raise MeasurementRegistrationError("MEASUREMENT_NOT_READY", "only READY measurements can enter the registry")
    result = deepcopy(registry)
    object_id = assessment["physical_geometry"]["object_id"]
    if any(item.get("object_id") == object_id for item in result["objects"]):
        raise MeasurementRegistrationError("DUPLICATE_OBJECT", "object is already registered")
    result["objects"].append(deepcopy(assessment["physical_geometry"]))
    result["registry_status"] = "HAS_READY_MEASUREMENTS"
    return result


def build_measured_world_placement(assessment, metric_observation):
    if assessment.get("status") != "READY":
        return {"status": "NOT_RUN", "reason": assessment.get("status")}
    if metric_observation.get("world_xy") is None or metric_observation.get("coordinate_frame") != "LEGACY_WORKSPACE_COORDINATE":
        raise MeasurementRegistrationError("INVALID_METRIC_OBSERVATION", "validated STEP 22D metric point is required")
    result = place_world_footprint(
        assessment["physical_geometry"],
        metric_observation["world_xy"],
        metric_observation["coordinate_frame"],
    )
    return {"status": "AVAILABLE", **result}
