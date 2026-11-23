"""Deterministic unified workspace/task state construction."""

from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path

from vlm_reposition.perception import bbox_iou, validate_normalized_bbox

from .zones import ZoneValidationError, get_zone, load_zones


class StateValidationError(ValueError):
    """Raised when validated inputs cannot form a unified state."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


def _finite_point(value, code, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_finite(v) for v in value):
        raise StateValidationError(code, f"{name} must be a finite [x, y] point")
    return [float(value[0]), float(value[1])]


def _workspace_bounds(config):
    bounds = config.get("workspace_bounds", {})
    x = _finite_point(bounds.get("x"), "INVALID_WORKSPACE_BOUNDS", "workspace x bounds")
    y = _finite_point(bounds.get("y"), "INVALID_WORKSPACE_BOUNDS", "workspace y bounds")
    if x[0] >= x[1] or y[0] >= y[1]:
        raise StateValidationError("INVALID_WORKSPACE_BOUNDS", "workspace bounds must be ordered")
    return {"x_min": x[0], "x_max": x[1], "y_min": y[0], "y_max": y[1], "unit": config.get("unit")}


def _world_coordinate(value, frame, unit, name):
    if not isinstance(value, dict) or not _finite(value.get("x")) or not _finite(value.get("y")):
        raise StateValidationError("NON_FINITE_COORDINATE", f"{name} world coordinate must be finite")
    if value.get("frame") != frame:
        raise StateValidationError("FRAME_MISMATCH", f"{name} frame does not match workspace frame")
    if value.get("unit") != unit:
        raise StateValidationError("UNIT_MISMATCH", f"{name} unit does not match workspace unit")
    return deepcopy(value)


def _destination(destination_zone, task_zone_id, frame, unit):
    if not isinstance(destination_zone, dict) or destination_zone.get("id") != task_zone_id:
        raise StateValidationError("DESTINATION_NOT_RESOLVED", f"destination zone does not match {task_zone_id!r}")
    polygon = destination_zone.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise StateValidationError("INVALID_DESTINATION_POLYGON", "destination polygon needs at least three vertices")
    polygon = [_finite_point(point, "INVALID_DESTINATION_POLYGON", "destination polygon vertex") for point in polygon]
    area = abs(sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(polygon, polygon[1:] + polygon[:1])
    )) / 2
    if area == 0:
        raise StateValidationError("INVALID_DESTINATION_POLYGON", "destination polygon area must be positive")
    return {
        "surface_phrase": None,
        "zone_id": task_zone_id,
        "center": _finite_point(destination_zone.get("center"), "INVALID_DESTINATION_POLYGON", "destination center"),
        "polygon": polygon,
        "unit": unit,
        "frame": frame,
        "development_status": destination_zone.get("status"),
        "development_only": destination_zone.get("development_only"),
        "physical_validation": destination_zone.get("physical_validation"),
        "planning_validation": destination_zone.get("planning_validation"),
    }


def _scene_entities(entities, frame, unit):
    if not isinstance(entities, list):
        raise StateValidationError("INVALID_SCENE", "scene entities must be a list")
    result = []
    for index, entity in enumerate(entities, start=1):
        if not isinstance(entity, dict):
            raise StateValidationError("INVALID_SCENE", f"scene entity {index} must be an object")
        try:
            bbox = list(validate_normalized_bbox(entity.get("bbox_normalized_0_1000")))
        except ValueError as exc:
            raise StateValidationError("INVALID_SCENE", f"scene entity {index} has invalid bbox") from exc
        world = entity.get("world_coordinate")
        item = {
            "id": entity.get("id"),
            "label": entity.get("label"),
            "semantic_type": entity.get("semantic_type"),
            "bbox_normalized_0_1000": bbox,
            "reference_point_type": entity.get("reference_point_type"),
            "world_coordinate": _world_coordinate(world, frame, unit, f"scene entity {index}"),
            "geometry_status": entity.get("geometry_status"),
            "semantic_label_status": entity.get("semantic_label_status"),
            "active_workspace_status": entity.get("active_workspace_status"),
            "reference_review_status": entity.get("reference_review_status"),
        }
        if not all(isinstance(item[key], str) and item[key] for key in ("id", "label", "semantic_type")):
            raise StateValidationError("INVALID_SCENE", f"scene entity {index} identity is invalid")
        if "matched_reference_id" in entity:
            item["matched_reference_id"] = entity["matched_reference_id"]
        result.append(item)
    return result


def _association(target_bbox, entities, threshold):
    if not _finite(threshold) or not 0 <= threshold <= 1:
        raise StateValidationError("INVALID_ASSOCIATION_THRESHOLD", "association threshold must be within 0..1")
    candidates = [(bbox_iou(target_bbox, entity["bbox_normalized_0_1000"]), entity["id"]) for entity in entities]
    overlap, entity_id = max(candidates, default=(0.0, None))
    matched = overlap >= threshold
    return {
        "status": "ASSOCIATED_BY_BBOX_IOU" if matched else "NOT_ASSOCIATED",
        "scene_entity_id": entity_id if matched else None,
        "bbox_iou": overlap,
        "threshold": float(threshold),
        "meaning": "SAME_VISUAL_ENTITY_CANDIDATE",
    }


def build_workspace_task_state(
    validated_task,
    target_grounding,
    destination_zone,
    scene_entities,
    workspace_config,
    provenance=None,
    association_iou_threshold=0.5,
):
    """Combine already-validated saved outputs without new inference or geometry."""
    if not isinstance(validated_task, dict) or validated_task.get("status") != "VALID_TASK":
        status = validated_task.get("status") if isinstance(validated_task, dict) else None
        raise StateValidationError("INVALID_TASK", f"task status must be VALID_TASK, received {status!r}")
    if not isinstance(target_grounding, dict) or target_grounding.get("target_status") != "TARGET_FOUND":
        status = target_grounding.get("target_status") if isinstance(target_grounding, dict) else None
        raise StateValidationError("TARGET_NOT_FOUND", f"target status must be TARGET_FOUND, received {status!r}")

    frame = workspace_config.get("coordinate_frame")
    unit = workspace_config.get("unit")
    if not isinstance(frame, str) or not frame:
        raise StateValidationError("FRAME_MISMATCH", "workspace coordinate frame is missing")
    if unit != "m":
        raise StateValidationError("UNIT_MISMATCH", "workspace metric unit must be m")
    bounds = _workspace_bounds(workspace_config)

    target_task = validated_task.get("target", {})
    grounding_phrase = target_task.get("grounding_phrase")
    if target_grounding.get("query") != grounding_phrase:
        raise StateValidationError("TARGET_MISMATCH", "target grounding query does not match validated target")
    target_bbox = list(validate_normalized_bbox(target_grounding.get("bbox_normalized_0_1000")))
    target_world = _world_coordinate(target_grounding.get("world_coordinate"), frame, unit, "target")
    entities = _scene_entities(scene_entities, frame, unit)
    association = _association(target_bbox, entities, association_iou_threshold)

    destination_task = validated_task.get("destination", {})
    destination = _destination(destination_zone, destination_task.get("zone_id"), frame, unit)
    destination["surface_phrase"] = destination_task.get("surface_phrase")

    return {
        "schema_version": "0.1",
        "state_type": "OFFLINE_DEVELOPMENT_WORKSPACE_TASK_STATE",
        "state_validity": "VALID_REPRESENTATION",
        "execution_readiness": "NOT_EVALUATED",
        "coordinate_frame": frame,
        "task": {
            "status": "VALID_TASK",
            "raw_instruction": validated_task.get("raw_instruction"),
            "action": {
                "surface_phrase": validated_task.get("action", {}).get("surface_phrase"),
                "canonical": validated_task.get("action", {}).get("canonical"),
            },
        },
        "workspace": {"source_status": workspace_config.get("status"), "bounds": bounds},
        "target": {
            "surface_phrase": target_task.get("surface_phrase"),
            "grounding_phrase": grounding_phrase,
            "canonicalization_status": target_task.get("canonicalization_status"),
            "grounding_status": "TARGET_FOUND",
            "detected_label": target_grounding.get("detected_label"),
            "semantic_label_status": target_grounding.get("semantic_label_status"),
            "bbox_normalized_0_1000": target_bbox,
            "reference_point_type": target_grounding.get("reference_point_type"),
            "world_coordinate": target_world,
            "active_workspace_status": target_grounding.get("active_workspace_status"),
            "scene_association": association,
        },
        "destination": destination,
        "scene": {"status": "AVAILABLE", "entity_count": len(entities), "entities": entities},
        "robot": {"status": "NOT_AVAILABLE"},
        "feasibility": {"status": "NOT_EVALUATED"},
        "planning": {"status": "NOT_RUN"},
        "provenance": deepcopy(provenance) if provenance is not None else {},
    }


def build_state_from_saved_results(
    task_path,
    target_grounding_path,
    scene_path,
    zones_path,
    association_iou_threshold=0.5,
    geometry_source=None,
):
    """Load existing result/config JSON files and build one unified state."""
    paths = [Path(path).resolve() for path in (task_path, target_grounding_path, scene_path, zones_path)]
    task_record, target, scene = (json.loads(path.read_text()) for path in paths[:3])
    zones = load_zones(paths[3])
    if scene.get("coordinate_frame") != zones["coordinate_frame"]:
        raise StateValidationError("FRAME_MISMATCH", "scene result frame does not match zone config")
    task = task_record.get("validated", task_record)
    zone_id = task.get("destination", {}).get("zone_id")
    try:
        zone = get_zone(zones, zone_id)
    except ZoneValidationError as exc:
        raise StateValidationError("DESTINATION_NOT_RESOLVED", str(exc)) from exc
    scene_summary_path = paths[2].parent / "semantic_repeatability_summary.json"
    scene_source_status = (
        json.loads(scene_summary_path.read_text()).get("result")
        if scene_summary_path.is_file()
        else "NOT_RECORDED"
    )
    provenance = {
        "task_source": str(paths[0]),
        "target_grounding_source": str(paths[1]),
        "scene_source": str(paths[2]),
        "zone_config_source": str(paths[3]),
        "geometry_source": str(Path(geometry_source).resolve()) if geometry_source else None,
        "scene_verification_source": str(scene_summary_path) if scene_summary_path.is_file() else None,
        "source_statuses": {
            "task": task.get("status"),
            "target_grounding": target.get("target_status"),
            "destination": zones.get("status"),
            "scene": scene_source_status,
        },
        "source_steps": {
            "task": "STEP_11_TASK_PARSER_REWORK",
            "target_grounding": "STEP_08_TASK_CONDITIONED_GROUNDING",
            "destination": "STEP_09_GOAL_ZONE_CONFIG",
            "scene": "STEP_07_SEMANTIC_REFINEMENT",
            "geometry": "STEP_04_LEGACY_GEOMETRY",
        },
    }
    return build_workspace_task_state(
        task,
        target,
        zone,
        scene.get("entities"),
        zones,
        provenance,
        association_iou_threshold,
    )
