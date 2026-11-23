"""Validated physical-geometry representation without collision checking."""

from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path


GEOMETRY_STATUSES = frozenset({"CONFIRMED", "UNKNOWN", "DEVELOPMENT_SYNTHETIC", "INFERRED"})
PLANNING_ROLES = frozenset({
    "UNASSIGNED",
    "STATIC_OBSTACLE_CANDIDATE",
    "DYNAMIC_ENTITY",
    "TASK_TARGET",
    "IGNORE_FOR_PLANNING",
})
PLANNER_USABLE_STATUSES = frozenset({"CONFIRMED", "DEVELOPMENT_SYNTHETIC"})
DIMENSION_FIELDS = (
    "length_m",
    "width_m",
    "diameter_m",
    "front_extension_m",
    "open_width_m",
    "closed_width_m",
    "holding_offset_m",
)


class PhysicalGeometryError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise PhysicalGeometryError("INVALID_DIMENSION", f"{name} must be finite and positive")
    return float(value)


def rectangle_footprint(length_m, width_m):
    length, width = _positive(length_m, "length_m"), _positive(width_m, "width_m")
    half_length, half_width = length / 2, width / 2
    return [
        [half_length, half_width],
        [half_length, -half_width],
        [-half_length, -half_width],
        [-half_length, half_width],
    ]


def _validate_component(component, name):
    if not isinstance(component, dict):
        raise PhysicalGeometryError("INVALID_SCHEMA", f"{name} must be an object")
    status = component.get("geometry_status")
    if status not in GEOMETRY_STATUSES:
        raise PhysicalGeometryError("INVALID_GEOMETRY_STATUS", f"{name} has unsupported geometry status")
    source = component.get("source_path")
    if not isinstance(source, str) or not source:
        raise PhysicalGeometryError("MISSING_PROVENANCE", f"{name} must record a source path")

    present = {field: component.get(field) for field in DIMENSION_FIELDS if field in component}
    if status == "UNKNOWN":
        if component.get("footprint_type") is not None or any(value is not None for value in present.values()):
            raise PhysicalGeometryError("UNKNOWN_HAS_DIMENSION", f"{name} UNKNOWN geometry must not contain dimensions")
        return
    if component.get("footprint_type") not in {None, "rectangle"}:
        raise PhysicalGeometryError("UNSUPPORTED_FOOTPRINT_SHAPE", f"{name} has an unsupported footprint shape")
    if status in PLANNER_USABLE_STATUSES:
        if component.get("footprint_type") != "rectangle":
            raise PhysicalGeometryError("UNSUPPORTED_FOOTPRINT_SHAPE", f"{name} confirmed geometry must be rectangular")
        generated = rectangle_footprint(component.get("length_m"), component.get("width_m"))
        source_footprint = component.get("source_footprint_m")
        if source_footprint is not None and generated != source_footprint:
            raise PhysicalGeometryError("FOOTPRINT_SOURCE_MISMATCH", f"{name} dimensions do not match source footprint")
        return
    for field, value in present.items():
        if value is not None:
            _positive(value, f"{name}.{field}")


def validate_physical_geometry(config):
    if not isinstance(config, dict) or config.get("schema_version") != "0.1":
        raise PhysicalGeometryError("INVALID_SCHEMA", "physical geometry schema_version must be 0.1")
    if config.get("coordinate_units") != "m":
        raise PhysicalGeometryError("INVALID_UNIT", "physical geometry units must be m")
    _validate_component(config.get("robot"), "robot")
    _validate_component(config.get("gripper"), "gripper")
    objects = config.get("movable_objects")
    if not isinstance(objects, dict) or not objects:
        raise PhysicalGeometryError("INVALID_SCHEMA", "movable_objects must be a non-empty object")
    for name, component in objects.items():
        _validate_component(component, f"movable_objects.{name}")
    policy = config.get("planning_role_policy", {})
    semantic_roles = policy.get("semantic_type_roles") if isinstance(policy, dict) else None
    if not isinstance(policy, dict) or not isinstance(semantic_roles, dict):
        raise PhysicalGeometryError("INVALID_SCHEMA", "planning role policy must be an object")
    roles = [policy.get("task_target_role"), policy.get("default_role"), *semantic_roles.values()]
    if any(role not in PLANNING_ROLES for role in roles):
        raise PhysicalGeometryError("INVALID_PLANNING_ROLE", "planning role policy contains an unsupported role")
    return config


def load_physical_geometry(path):
    return validate_physical_geometry(json.loads(Path(path).read_text()))


def _component_output(component):
    result = deepcopy(component)
    usable = component["geometry_status"] in PLANNER_USABLE_STATUSES
    result["geometry_readiness"] = "READY" if usable else "NOT_READY"
    result["footprint_m"] = (
        rectangle_footprint(component["length_m"], component["width_m"])
        if usable
        else None
    )
    return result


def _planning_entities(workspace_state, policy):
    target_id = workspace_state["target"]["scene_association"]["scene_entity_id"]
    entities = []
    for entity in workspace_state["scene"]["entities"]:
        role = (
            policy["task_target_role"]
            if entity["id"] == target_id
            else policy["semantic_type_roles"].get(entity["semantic_type"], policy["default_role"])
        )
        entities.append({
            "scene_entity_id": entity["id"],
            "label": entity["label"],
            "semantic_type": entity["semantic_type"],
            "planning_role": role,
            "world_coordinate": deepcopy(entity["world_coordinate"]),
            "active_workspace_status": entity["active_workspace_status"],
            "geometry_status": "UNKNOWN",
            "footprint_m": None,
            "geometry_readiness": "NOT_READY",
            "source": "STEP_07_REFERENCE_POINT_ONLY",
        })
    return entities


def build_physical_planning_state(workspace_state, physical_config, workspace_state_source=None):
    validate_physical_geometry(physical_config)
    if not isinstance(workspace_state, dict) or workspace_state.get("state_validity") != "VALID_REPRESENTATION":
        raise PhysicalGeometryError("INVALID_WORKSPACE_STATE", "STEP 12 state must be VALID_REPRESENTATION")
    target_name = workspace_state.get("target", {}).get("grounding_phrase")
    if target_name not in physical_config["movable_objects"]:
        raise PhysicalGeometryError("TARGET_GEOMETRY_NOT_CONFIGURED", f"no geometry record for {target_name!r}")

    robot = _component_output(physical_config["robot"])
    gripper = _component_output(physical_config["gripper"])
    target_object = _component_output(physical_config["movable_objects"][target_name])
    entities = _planning_entities(workspace_state, physical_config["planning_role_policy"])
    obstacle_candidates = [entity for entity in entities if entity["planning_role"] == "STATIC_OBSTACLE_CANDIDATE"]
    obstacle_ready = bool(obstacle_candidates) and all(
        entity["geometry_readiness"] == "READY" for entity in obstacle_candidates
    )
    readiness = {
        "robot": robot["geometry_readiness"],
        "gripper": gripper["geometry_readiness"],
        "target_object": target_object["geometry_readiness"],
        "static_obstacles": "READY" if obstacle_ready else "NOT_READY",
    }
    readiness["overall"] = (
        "READY"
        if all(value == "READY" for value in readiness.values())
        else "PARTIAL" if readiness["robot"] == "READY" else "NOT_READY"
    )
    return {
        "schema_version": "0.1",
        "state_type": "OFFLINE_DEVELOPMENT_PHYSICAL_PLANNING_STATE",
        "representation_status": "VALID",
        "workspace_task_state_source": str(Path(workspace_state_source).resolve()) if workspace_state_source else None,
        "workspace_task_summary": {
            "instruction": workspace_state["task"]["raw_instruction"],
            "target": target_name,
            "destination_zone": workspace_state["destination"]["zone_id"],
            "coordinate_frame": workspace_state["coordinate_frame"],
        },
        "physical_geometry": {
            "robot": robot,
            "gripper": gripper,
            "target_object": target_object,
            "planning_entities": entities,
        },
        "planning_geometry_readiness": readiness,
        "collision_checking": {"status": "NOT_RUN"},
        "planning": {"status": "NOT_RUN"},
    }
