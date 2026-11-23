"""Data contracts: coordinate-frame names, a small JSON-Schema-subset validator, and the
schemas of every structure exchanged between modules or written to a run directory."""
import json

# ---------------------------------------------------------------- coordinate frames
RAW_PX = "raw_image_px"                  # decoded input frame, distorted, origin top-left, x right, y down
UNDISTORTED_PX = "undistorted_image_px"  # after cv2.undistort with the configured intrinsics
ROI_PX = "roi_local_px"                  # zoomed ROI crop; raw = offset + roi / scale
GROUND_XY = "ground_map_xy"              # metric floor plane (reserved: needs a validated homography)
GRID_RC = "grid_row_col"                 # planning grid cell (row, col); cell centre = (c+0.5, r+0.5)*cell_px
FRAMES = (RAW_PX, UNDISTORTED_PX, ROI_PX, GROUND_XY, GRID_RC)

MAP_MODE_IMAGE = "IMAGE_SPACE_DEMO"
MAP_MODE_GROUND = "CALIBRATED_GROUND_MAP"

# Knowledge provenance attached to relations / regions / poses.
OBSERVED = "observed_by_model"           # produced by a model or algorithm on this frame
INFERRED = "inferred"                    # semantic inference (VLM), not verified
MANUAL = "manual_prior"                  # supplied by a person / prior map
MANUAL_TEST = "manual_test_data"         # IMAGE_DEMO inputs, never counted as model output

STEP_STATUSES = ("PENDING", "RUNNING", "OK", "UNCERTAIN", "FAILED", "SKIPPED", "NOT_RUN", "CANCELLED")


class SchemaError(ValueError):
    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


def validate(instance, schema):
    errors = []
    _validate(instance, schema, "$", errors)
    if errors:
        raise SchemaError(errors)
    return instance


def _is_type(x, t):
    if t == "integer":
        return isinstance(x, int) and not isinstance(x, bool)
    if t == "number":
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if t == "boolean":
        return isinstance(x, bool)
    return isinstance(x, {"object": dict, "array": list, "string": str, "null": type(None)}[t])


def _validate(x, s, path, errors):
    t = s.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_is_type(x, tt) for tt in types):
            errors.append(f"{path}: expected {t}, got {type(x).__name__}")
            return
    if "enum" in s and x not in s["enum"]:
        errors.append(f"{path}: {x!r} not in {s['enum']}")
    if isinstance(x, dict):
        for k in s.get("required", []):
            if k not in x:
                errors.append(f"{path}: missing required field '{k}'")
        props = s.get("properties", {})
        if s.get("additionalProperties") is False:
            errors.extend(f"{path}: unexpected field '{k}'" for k in x if k not in props)
        for k, sub in props.items():
            if k in x:
                _validate(x[k], sub, f"{path}.{k}", errors)
    elif isinstance(x, list):
        if len(x) < s.get("minItems", 0):
            errors.append(f"{path}: needs at least {s['minItems']} items")
        if "maxItems" in s and len(x) > s["maxItems"]:
            errors.append(f"{path}: allows at most {s['maxItems']} items")
        if "items" in s:
            for i, item in enumerate(x):
                _validate(item, s["items"], f"{path}[{i}]", errors)
    elif isinstance(x, str):
        if len(x) < s.get("minLength", 0):
            errors.append(f"{path}: shorter than {s['minLength']}")
    elif _is_type(x, "number"):
        if "minimum" in s and x < s["minimum"]:
            errors.append(f"{path}: {x} < minimum {s['minimum']}")
        if "maximum" in s and x > s["maximum"]:
            errors.append(f"{path}: {x} > maximum {s['maximum']}")


def parse_json_object(raw):
    """Strict JSON-object extraction from a model reply: optional ``` fence, nothing else."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) == 2 else ""
        text = text.rsplit("```", 1)[0]
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise SchemaError(["$: expected a JSON object"])
    return obj


# ---------------------------------------------------------------- schemas
_STR = {"type": "string"}
_STR_OR_NULL = {"type": ["string", "null"]}
_STR_LIST = {"type": "array", "items": _STR}
_XY = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
_XYXY = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
_FRAME = {"type": "string", "enum": list(FRAMES)}

TASK_SCHEMA = {  # structured task handed to the orchestrator
    "type": "object",
    "required": ["task_type", "target", "destination", "actions", "parser"],
    "properties": {
        "task_type": {"type": "string", "enum": ["transport_and_place", "navigate_to_region"]},
        "target": {"type": ["object", "null"], "required": ["class", "color", "source_reference", "source_relation"],
                   "properties": {"class": _STR, "color": _STR_OR_NULL, "source_reference": _STR_OR_NULL,
                                  "source_relation": _STR_OR_NULL}},
        "destination": {"type": "object", "required": ["region_type", "reference_object"],
                        "properties": {"region_type": _STR, "reference_object": _STR_OR_NULL}},
        "actions": {"type": "array", "items": _STR, "minItems": 1},
        "parser": {"type": "object", "required": ["backend", "raw_response"]},
    },
}

DETECTION_SCHEMA = {
    "type": "object",
    "required": ["object_id", "class_name", "bbox_xyxy", "coordinate_frame", "score", "image_frame_id",
                 "model_name", "model_version", "prompt", "source", "knowledge"],
    "properties": {
        "object_id": _STR, "class_name": _STR, "bbox_xyxy": _XYXY, "coordinate_frame": _FRAME,
        "score": {"type": ["number", "null"]}, "image_frame_id": {"type": ["integer", "string"]},
        "model_name": _STR_OR_NULL, "model_version": _STR_OR_NULL, "prompt": _STR_OR_NULL, "source": _STR,
        "knowledge": {"type": "string", "enum": [OBSERVED, MANUAL_TEST]},
    },
}

RELATION_SCHEMA = {
    "type": "object",
    "required": ["relation_id", "type", "subject", "object", "method", "evidence", "status", "verified", "knowledge"],
    "properties": {"relation_id": _STR, "type": _STR, "subject": _STR, "object": _STR, "method": _STR,
                   "evidence": {"type": ["object", "array", "string"]},
                   "status": {"type": "string", "enum": ["COMPUTED", "INFERRED", "UNVERIFIED", "MANUAL"]},
                   "verified": {"type": "boolean"},
                   "knowledge": {"type": "string", "enum": [OBSERVED, INFERRED, MANUAL, MANUAL_TEST]}},
}

REGION_SCHEMA = {
    "type": "object",
    "required": ["region_id", "region_type", "polygon_xy", "coordinate_frame", "reference_objects", "evidence",
                 "verification_status", "source"],
    "properties": {"region_id": _STR, "region_type": _STR,
                   "polygon_xy": {"type": "array", "items": _XY, "minItems": 3}, "coordinate_frame": _FRAME,
                   "reference_objects": _STR_LIST, "evidence": {"type": "array"},
                   "verification_status": {"type": "string", "enum": ["unverified", "verified", "manual_unconfirmed"]},
                   "source": _STR},
}

GOAL_POSE_SCHEMA = {
    "type": "object",
    "required": ["purpose", "xy", "cell", "heading_deg", "coordinate_frame", "anchor", "path_cost", "metric_valid"],
    "properties": {"purpose": {"type": "string", "enum": ["pickup_approach", "dropoff_approach"]}, "xy": _XY,
                   "cell": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                   "heading_deg": {"type": ["number", "null"]}, "coordinate_frame": _FRAME, "anchor": _STR,
                   "path_cost": {"type": "number"}, "metric_valid": {"type": "boolean"}},
}

PATH_SCHEMA = {
    "type": "object",
    "required": ["path_id", "status", "coordinate_frame", "planner", "waypoints_xy", "raw_cells", "length_px"],
    "properties": {"path_id": {"type": "string", "enum": ["A_start_to_pickup", "B_pickup_to_dropoff", "A_start_to_goal"]},
                   "status": _STR, "coordinate_frame": _FRAME, "planner": _STR,
                   "waypoints_xy": {"type": "array", "items": _XY}, "raw_cells": {"type": "array"},
                   "length_px": {"type": ["number", "null"]}},
}

HANDOFF_SCHEMA = {
    "type": "object",
    "required": ["action", "target_object_id", "target_class", "target_color", "approach_pose", "coordinate_frame",
                 "status", "hardware_executable", "mode"],
    "properties": {"action": {"type": "string", "enum": ["PICKUP_HANDOFF", "PLACEMENT_HANDOFF", "NAVIGATION_GOAL"]},
                   "target_object_id": _STR_OR_NULL, "target_class": _STR_OR_NULL, "target_color": _STR_OR_NULL,
                   "approach_pose": {"type": ["object", "null"]}, "coordinate_frame": {"type": ["string", "null"]},
                   "status": _STR, "hardware_executable": {"type": "boolean"},
                   "mode": {"type": "string", "enum": ["simulation"]}},
}

RESULT_SCHEMA = {
    "type": "object",
    "required": ["run_id", "analysis_mode", "map_mode", "command", "frame", "steps", "overall_status", "task",
                 "detections", "scene_graph", "region_grounding", "goals", "paths", "validation", "handoff",
                 "evaluation_record"],
}
