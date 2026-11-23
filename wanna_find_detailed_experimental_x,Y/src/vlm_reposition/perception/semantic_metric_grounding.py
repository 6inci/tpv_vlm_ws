"""Candidate-conditioned semantics and legacy metric point grounding."""

import json
import math
from numbers import Real
from pathlib import Path

import cv2

from vlm_reposition.geometry import GeometryCompatibilityError
from .qwen_grounding import (
    SEMANTIC_TYPES,
    _parse_json_object,
    bbox_bottom_center,
    ground_normalized_bbox,
    normalized_bbox_to_pixel_bbox,
    validate_normalized_bbox,
)


SEMANTIC_STATUSES = {"VISIBLE_ENTITY_FOUND", "NO_VISIBLE_ENTITY", "AMBIGUOUS"}
ALLOWED_METRIC_POLICIES = {"REFERENCE_ALLOWED", "DEVELOPMENT_INFERRED_ALLOWED"}
DEVELOPMENT_PROVENANCE = (
    "Image-registration-based development inference only; not final physical camera-pose validation."
)


class SemanticMetricGroundingError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def load_semantic_metric_config(path):
    config = json.loads(Path(path).read_text())
    required = (
        "model_grounding_config", "input_replay_manifest", "scene_observation_directory",
        "trigger_policy", "semantic_prompt_version", "parser_policy", "candidate_association",
        "ground_reference_policy", "metric_geometry_policy", "max_new_tokens",
        "repeatability_image_count", "repeatability_runs", "parameter_status",
    )
    if config.get("schema_version") != "0.1" or any(key not in config for key in required):
        raise SemanticMetricGroundingError("INVALID_CONFIG", "invalid semantic metric grounding config")
    if config["trigger_policy"] != "SEMANTIC_TRIGGER_TRUE_CONFIRMED_CANDIDATES_ONLY":
        raise SemanticMetricGroundingError("INVALID_CONFIG", "unsupported trigger policy")
    if config["ground_reference_policy"] != "BOTTOM_CENTER":
        raise SemanticMetricGroundingError("INVALID_CONFIG", "ground reference must be BOTTOM_CENTER")
    if config["parameter_status"] != "DEVELOPMENT_PARAMETER":
        raise SemanticMetricGroundingError("INVALID_CONFIG", "association thresholds must be development parameters")
    return config


def select_triggered_frames(manifest, observation_directory):
    allowed = {str(Path(item["path"]).resolve()): item for item in manifest["images"]}
    selected = []
    for path in sorted(Path(observation_directory).glob("*.json")):
        observation = json.loads(path.read_text())
        source = str(Path(observation["source_image"]).resolve())
        manifest_item = allowed.get(source)
        if manifest_item is None:
            raise SemanticMetricGroundingError("IMAGE_NOT_ALLOWED", f"observation image is not in replay manifest: {source}")
        if not observation["semantic_trigger"]:
            continue
        if manifest_item["metric_transform_policy"] not in ALLOWED_METRIC_POLICIES:
            raise SemanticMetricGroundingError("METRIC_POLICY_NOT_ALLOWED", f"triggered image policy blocks metric path: {source}")
        candidates = [item for item in observation["candidates"] if item.get("track_state") == "CONFIRMED"]
        if not candidates:
            raise SemanticMetricGroundingError("NO_CONFIRMED_CANDIDATE", f"trigger has no confirmed candidate: {source}")
        selected.append({
            "image_id": manifest_item["image_id"],
            "manifest_image": manifest_item,
            "scene_observation": observation,
            "confirmed_candidates": candidates,
        })
    return selected


def candidate_prompt_payload(candidates):
    return [{
        "track_id": item["track_id"],
        "candidate_bbox_px": item["bbox_px"],
    } for item in candidates]


def make_candidate_prompt(image_id, candidates):
    if len(candidates) != 1:
        raise SemanticMetricGroundingError(
            "SINGLE_CANDIDATE_REQUIRED", "exactly one representative candidate is required per inference"
        )
    candidate = candidate_prompt_payload(candidates)[0]
    return f'''You are given one original full-resolution workspace scene image. The candidate is a pixel-space change relative to a fixed reference. It is not an object detection.

image_id: {image_id}
track_id: {candidate["track_id"]}
candidate_bbox_px: {json.dumps(candidate["candidate_bbox_px"], separators=(",", ":"))}

Determine whether a currently visible physical entity occupies or overlaps this candidate. A change may instead be an old-object ghost, shadow, background/exposure change, or ambiguous fragment.

Return JSON only with exactly one result for the supplied track_id:
{{"image_id":"{image_id}","candidate_results":[{{"track_id":"track_id copied exactly","candidate_bbox_px":[x1,y1,x2,y2],"status":"VISIBLE_ENTITY_FOUND|NO_VISIBLE_ENTITY|AMBIGUOUS","visible_label":"specific visible name or null","semantic_type":"movable_object|static_obstacle|person|furniture|workspace_element|unknown","object_bbox_normalized_1000":[x1,y1,x2,y2] or null,"confidence_note":"short note"}}]}}

For VISIBLE_ENTITY_FOUND, return the full visible physical entity bbox, not the change candidate bbox. object_bbox_normalized_1000 must contain INTEGER coordinates in the normalized 0-1000 full-image domain. Valid example: [125,240,680,910]. Invalid: [0.125,0.240,0.680,0.910]. candidate_bbox_px is raw pixel metadata: copy it only into candidate_bbox_px and never copy raw pixel values into object_bbox_normalized_1000. Do not repair or rescale an invalid domain. For NO_VISIBLE_ENTITY or AMBIGUOUS, use null bbox. Preserve track_id and candidate_bbox_px exactly. Do not estimate metric coordinates, physical dimensions, navigation roles, paths, robot actions, or obstacle geometry.'''


def make_method_c_candidate_prompt(image_id, track_id, candidate_bbox_normalized_1000):
    bbox = candidate_bbox_normalized_1000
    if not isinstance(track_id, str) or not track_id or (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(type(value) is not int for value in bbox)
    ):
        raise SemanticMetricGroundingError("INVALID_METHOD_C_INPUT", "invalid Method C candidate identity or bbox")
    try:
        validate_normalized_bbox(bbox)
    except ValueError as exc:
        raise SemanticMetricGroundingError("INVALID_METHOD_C_INPUT", str(exc)) from exc
    bbox_text = json.dumps(list(bbox), separators=(",", ":"))
    return f'''You are viewing one full-resolution scene image. A single candidate region is visibly highlighted in the image.

image_id: {image_id}
track_id: {track_id}
candidate_region_normalized_1000: {bbox_text}

Candidate bbox format is [x_min,y_min,x_max,y_max]. Coordinates are normalized INTEGER values from 0 to 1000, where the top-left is [0,0] and the bottom-right boundary is [1000,1000]. The highlighted region is only an attention cue from a scene-change detector. It is NOT necessarily the full object bounding box.

Determine whether a currently visible physical object overlaps or occupies the highlighted region. If one exists, return VISIBLE_ENTITY_FOUND, a concise visible label, a coarse semantic type, and the FULL visible-object bounding box. Do not return the highlighted rectangle merely because it was supplied.

Return JSON only with exactly one result for the supplied track_id:
{{"image_id":"{image_id}","candidate_results":[{{"track_id":"{track_id}","status":"VISIBLE_ENTITY_FOUND|NO_VISIBLE_ENTITY|AMBIGUOUS","visible_label":"specific visible name or null","semantic_type":"movable_object|static_obstacle|person|furniture|workspace_element|unknown","object_bbox_normalized_1000":[x_min,y_min,x_max,y_max] or null,"confidence_note":"short note"}}]}}

For VISIBLE_ENTITY_FOUND, object_bbox_normalized_1000 must contain exactly four INTEGER coordinates in the normalized 0-1000 full-image domain. Valid example: [120,250,680,900]. Invalid: [0.12,0.25,0.68,0.90]. For NO_VISIBLE_ENTITY or AMBIGUOUS, use null bbox. If uncertain, return AMBIGUOUS. Do not return metric coordinates, physical size, a navigation role, a path, a robot action, or obstacle geometry.'''


def make_method_d_candidate_prompt(image_id, track_id, candidate_bbox_normalized_1000):
    bbox = candidate_bbox_normalized_1000
    if not isinstance(track_id, str) or not track_id or (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(type(value) is not int for value in bbox)
    ):
        raise SemanticMetricGroundingError("INVALID_METHOD_D_INPUT", "invalid Method D candidate identity or bbox")
    try:
        validate_normalized_bbox(bbox)
    except ValueError as exc:
        raise SemanticMetricGroundingError("INVALID_METHOD_D_INPUT", str(exc)) from exc
    bbox_text = json.dumps(list(bbox), separators=(",", ":"))
    return f'''You are given two images in this exact order.

IMAGE 1 is the original full-resolution scene.
IMAGE 2 is a crop taken directly from IMAGE 1 around candidate {track_id}. It is provided only to make the candidate easier to visually inspect.

image_id: {image_id}
track_id: {track_id}
candidate_region_in_image_1_normalized_1000: {bbox_text}

Candidate bbox format is [x_min,y_min,x_max,y_max]. Coordinates are normalized INTEGER values from 0 to 1000, where IMAGE 1 top-left is [0,0] and bottom-right boundary is [1000,1000].

First determine whether IMAGE 2 contains a currently visible physical object corresponding to {track_id}. If yes, identify it and locate the SAME object in IMAGE 1. Return the FULL visible-object bounding box in IMAGE 1 coordinates, not IMAGE 2 crop coordinates.

Return JSON only with exactly one result for the supplied track_id:
{{"image_id":"{image_id}","candidate_results":[{{"track_id":"{track_id}","status":"VISIBLE_ENTITY_FOUND|NO_VISIBLE_ENTITY|AMBIGUOUS","visible_label":"specific visible name or null","semantic_type":"movable_object|static_obstacle|person|furniture|workspace_element|unknown","object_bbox_normalized_1000":[x_min,y_min,x_max,y_max] or null,"bbox_reference_image":"IMAGE_1","confidence_note":"short note"}}]}}

For VISIBLE_ENTITY_FOUND, object_bbox_normalized_1000 must contain exactly four INTEGER coordinates in the normalized 0-1000 IMAGE 1 full-scene domain and bbox_reference_image must be exactly "IMAGE_1". Do NOT return IMAGE 2 crop coordinates. Valid example: [120,250,680,900]. Invalid: [0.12,0.25,0.68,0.90]. For NO_VISIBLE_ENTITY or AMBIGUOUS, use null bbox. If uncertain, return AMBIGUOUS. Do not return metric coordinates, physical size, a navigation role, a path, a robot action, or obstacle geometry.'''


def _validate_pixel_bbox(bbox, name="bbox"):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise SemanticMetricGroundingError("INVALID_PIXEL_BBOX", f"{name} must contain four coordinates")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in bbox):
        raise SemanticMetricGroundingError("INVALID_PIXEL_BBOX", f"{name} coordinates must be numeric")
    values = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in values) or values[0] >= values[2] or values[1] >= values[3]:
        raise SemanticMetricGroundingError("INVALID_PIXEL_BBOX", f"{name} must be finite and ordered")
    return values


def parse_candidate_response(
    raw_text, image_id, candidates, require_candidate_bbox_echo=True,
    required_bbox_reference_image=None,
):
    try:
        parsed = _parse_json_object(raw_text)
    except ValueError as exc:
        raise SemanticMetricGroundingError("FRAME_SEMANTIC_PARSE_FAILED", str(exc)) from exc
    if parsed.get("image_id") != image_id or not isinstance(parsed.get("candidate_results"), list):
        raise SemanticMetricGroundingError("FRAME_SEMANTIC_PARSE_FAILED", "image_id or candidate_results is invalid")
    expected = {item["track_id"]: item for item in candidates}
    results, seen = {}, set()
    for index, item in enumerate(parsed["candidate_results"], start=1):
        if not isinstance(item, dict):
            raise SemanticMetricGroundingError("FRAME_SEMANTIC_PARSE_FAILED", f"result {index} is not an object")
        track_id = item.get("track_id")
        if track_id in seen:
            raise SemanticMetricGroundingError("DUPLICATE_CANDIDATE_ID", f"duplicate candidate ID: {track_id}")
        if track_id not in expected:
            raise SemanticMetricGroundingError("UNKNOWN_CANDIDATE_ID", f"unknown candidate ID: {track_id}")
        seen.add(track_id)
        expected_bbox = list(expected[track_id]["bbox_px"])
        if require_candidate_bbox_echo and item.get("candidate_bbox_px") != expected_bbox:
            raise SemanticMetricGroundingError("CANDIDATE_BBOX_MISMATCH", f"candidate bbox changed for {track_id}")
        status = item.get("status")
        if status not in SEMANTIC_STATUSES:
            raise SemanticMetricGroundingError("INVALID_SEMANTIC_STATUS", f"invalid semantic status for {track_id}")
        label, semantic_type = item.get("visible_label"), item.get("semantic_type")
        bbox = item.get("object_bbox_normalized_1000")
        if status == "VISIBLE_ENTITY_FOUND":
            if not isinstance(label, str) or not label.strip() or semantic_type not in SEMANTIC_TYPES:
                raise SemanticMetricGroundingError("INVALID_VISIBLE_ENTITY", f"invalid label/type for {track_id}")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or any(type(value) is not int for value in bbox):
                raise SemanticMetricGroundingError("BBOX_INVALID", f"visible bbox must use integer 0-1000 coordinates for {track_id}")
            try:
                bbox = list(validate_normalized_bbox(bbox))
            except ValueError as exc:
                raise SemanticMetricGroundingError("BBOX_INVALID", f"invalid visible bbox for {track_id}: {exc}") from exc
            if required_bbox_reference_image is not None and item.get("bbox_reference_image") != required_bbox_reference_image:
                raise SemanticMetricGroundingError(
                    "BBOX_REFERENCE_AMBIGUOUS",
                    f"visible bbox for {track_id} must explicitly reference {required_bbox_reference_image}",
                )
            label = label.strip()
        else:
            if bbox is not None:
                raise SemanticMetricGroundingError("BBOX_INVALID", f"non-visible result must use null bbox for {track_id}")
            label = label.strip() if isinstance(label, str) and label.strip() else None
            semantic_type = semantic_type if semantic_type in SEMANTIC_TYPES else "unknown"
        note = item.get("confidence_note")
        if note is not None and not isinstance(note, str):
            raise SemanticMetricGroundingError("FRAME_SEMANTIC_PARSE_FAILED", f"confidence_note is invalid for {track_id}")
        results[track_id] = {
            "track_id": track_id,
            "candidate_bbox_px": expected_bbox,
            "status": status,
            "visible_label": label,
            "semantic_type": semantic_type,
            "object_bbox_normalized_1000": bbox,
            "confidence_note": note,
        }
        if required_bbox_reference_image is not None:
            results[track_id]["bbox_reference_image"] = item.get("bbox_reference_image")
    missing = set(expected) - seen
    if missing:
        raise SemanticMetricGroundingError("MISSING_CANDIDATE_ID", f"missing candidate IDs: {sorted(missing)}")
    return [results[item["track_id"]] for item in candidates]


def bbox_association(candidate_bbox, entity_bbox, image_size, thresholds):
    candidate = _validate_pixel_bbox(candidate_bbox, "candidate_bbox")
    entity = _validate_pixel_bbox(entity_bbox, "entity_bbox")
    ax1, ay1, ax2, ay2 = candidate
    bx1, by1, bx2, by2 = entity
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    iou = 0.0 if union == 0 else intersection / union
    candidate_center = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)
    entity_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)
    candidate_center_inside = bx1 <= candidate_center[0] <= bx2 and by1 <= candidate_center[1] <= by2
    entity_center_inside = ax1 <= entity_center[0] <= ax2 and ay1 <= entity_center[1] <= ay2
    distance = math.hypot(candidate_center[0] - entity_center[0], candidate_center[1] - entity_center[1])
    diagonal = math.hypot(*image_size)
    distance_ratio = distance / diagonal
    associated = (
        iou >= thresholds["minimum_associated_iou"]
        or candidate_center_inside
        or entity_center_inside
        or (intersection > 0 and distance_ratio <= thresholds["maximum_associated_center_distance_ratio"])
    )
    weak = (
        iou >= thresholds["minimum_weak_iou"]
        or intersection > 0
        or distance_ratio <= thresholds["maximum_weak_center_distance_ratio"]
    )
    return {
        "status": "ASSOCIATED" if associated else ("WEAK_ASSOCIATION" if weak else "NOT_ASSOCIATED"),
        "meaning": "SAME_VISIBLE_ENTITY_CANDIDATE_NOT_IDENTITY_PROOF",
        "iou": iou,
        "intersection_area_px": intersection,
        "candidate_centroid_inside_entity": candidate_center_inside,
        "entity_center_inside_candidate": entity_center_inside,
        "center_distance_px": distance,
        "center_distance_image_diagonal_ratio": distance_ratio,
    }


def workspace_status(world_xy, bounds):
    xmin, xmax, ymin, ymax = bounds
    return "INSIDE" if xmin <= world_xy[0] <= xmax and ymin <= world_xy[1] <= ymax else "OUTSIDE"


def build_metric_observation(frame, candidate, semantic, manifest_image, image_size, geometry, bounds, thresholds):
    observation = {
        "schema_version": "0.1",
        "observation_type": "OFFLINE_SEMANTIC_METRIC_POINT_OBSERVATION",
        "source_image": {
            "image_id": frame["image_id"],
            "path": frame["scene_observation"]["source_image"],
            "width": image_size[0],
            "height": image_size[1],
            "geometry_evidence": manifest_image["compatibility_status"],
            "metric_transform_policy": manifest_image["metric_transform_policy"],
            "geometry_provenance": DEVELOPMENT_PROVENANCE if manifest_image["metric_transform_policy"] == "DEVELOPMENT_INFERRED_ALLOWED" else "LEGACY_REFERENCE_VERIFIED",
        },
        "scene_observation": {
            "track_id": candidate["track_id"],
            "candidate_bbox_px": candidate["bbox_px"],
            "candidate_type": "CHANGE_CANDIDATE_NOT_OBJECT_DETECTION",
        },
        "semantic": {
            "status": semantic["status"],
            "label": semantic["visible_label"],
            "semantic_type": semantic["semantic_type"],
            "planning_role": "NOT_ASSIGNED",
            "confidence_note": semantic["confidence_note"],
        },
        "association": {"status": "NOT_EVALUATED"},
        "visual_grounding": None,
        "metric_grounding": {
            "status": "NOT_RUN",
            "coordinate_frame": None,
            "unit": None,
            "world_xy": None,
            "workspace_status": "NOT_EVALUATED",
        },
        "physical_geometry": {"status": "NOT_AVAILABLE"},
        "planning": {"role": "NOT_ASSIGNED", "obstacle_injection": "NOT_RUN"},
    }
    if semantic["status"] != "VISIBLE_ENTITY_FOUND":
        return observation
    raw_bbox = normalized_bbox_to_pixel_bbox(semantic["object_bbox_normalized_1000"], image_size)
    if not (0 <= raw_bbox[0] < raw_bbox[2] <= image_size[0] and 0 <= raw_bbox[1] < raw_bbox[3] <= image_size[1]):
        observation["metric_grounding"]["status"] = "BBOX_INVALID"
        return observation
    association = bbox_association(candidate["bbox_px"], raw_bbox, image_size, thresholds)
    observation["association"] = association
    ground_reference = bbox_bottom_center(raw_bbox)
    observation["visual_grounding"] = {
        "bbox_normalized_1000": semantic["object_bbox_normalized_1000"],
        "bbox_raw_px": list(raw_bbox),
        "ground_reference_policy": "BOTTOM_CENTER",
        "ground_reference_px": list(ground_reference),
    }
    if association["status"] != "ASSOCIATED":
        return observation
    if manifest_image["metric_transform_policy"] not in ALLOWED_METRIC_POLICIES:
        observation["metric_grounding"]["status"] = "METRIC_GROUNDING_NOT_ALLOWED"
        return observation
    if tuple(image_size) != geometry.image_size:
        observation["metric_grounding"]["status"] = "METRIC_GROUNDING_NOT_ALLOWED"
        return observation
    try:
        grounded = ground_normalized_bbox(
            "candidate-associated visible entity",
            semantic["visible_label"],
            semantic["object_bbox_normalized_1000"],
            image_size,
            geometry,
        )
    except (GeometryCompatibilityError, ValueError) as exc:
        observation["metric_grounding"].update({"status": "ERROR", "error": str(exc)})
        return observation
    world = grounded["world_coordinate"]
    observation["visual_grounding"]["undistorted_ground_reference_px"] = [
        grounded["undistorted_reference_pixel"]["u"],
        grounded["undistorted_reference_pixel"]["v"],
    ]
    observation["metric_grounding"] = {
        "status": "AVAILABLE",
        "coordinate_frame": world["frame"],
        "unit": world["unit"],
        "world_xy": [world["x"], world["y"]],
        "workspace_status": workspace_status((world["x"], world["y"]), bounds),
        "pipeline": "RAW_BOTTOM_CENTER_PX_TO_STORED_UNDISTORTION_TO_LEGACY_HOMOGRAPHY",
    }
    return observation


def process_frame_response(
    frame, raw_text, geometry, bounds, thresholds, image_size=None,
    require_candidate_bbox_echo=True, required_bbox_reference_image=None,
):
    image_size = tuple(image_size or geometry.image_size)
    semantics = parse_candidate_response(
        raw_text, frame["image_id"], frame["confirmed_candidates"],
        require_candidate_bbox_echo, required_bbox_reference_image,
    )
    by_track = {item["track_id"]: item for item in semantics}
    observations = [
        build_metric_observation(
            frame, candidate, by_track[candidate["track_id"]], frame["manifest_image"],
            image_size, geometry, bounds, thresholds,
        )
        for candidate in frame["confirmed_candidates"]
    ]
    return {
        "schema_version": "0.1",
        "image_id": frame["image_id"],
        "source_image": frame["scene_observation"]["source_image"],
        "scene_state": frame["scene_observation"]["scene_state"],
        "semantic_trigger": frame["scene_observation"]["semantic_trigger"],
        "frame_semantic_status": "PARSED",
        "semantic_results": semantics,
        "metric_observations": observations,
    }


def annotate_candidates_for_qwen(image, candidates):
    visual = image.copy()
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["bbox_px"]
        cv2.rectangle(visual, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 3)
        cv2.putText(visual, candidate["track_id"], (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return visual


def annotate_metric_results(image, result):
    visual = image.copy()
    for observation in result["metric_observations"]:
        x1, y1, x2, y2 = observation["scene_observation"]["candidate_bbox_px"]
        cv2.rectangle(visual, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
        grounding = observation["visual_grounding"]
        if grounding is None:
            continue
        bx1, by1, bx2, by2 = grounding["bbox_raw_px"]
        cv2.rectangle(visual, (int(bx1), int(by1)), (int(bx2), int(by2)), (255, 160, 0), 3)
        gx, gy = grounding["ground_reference_px"]
        cv2.circle(visual, (int(gx), min(image.shape[0] - 1, int(gy))), 7, (0, 0, 255), -1)
        metric = observation["metric_grounding"]
        world_text = "X/Y=NA" if metric["status"] != "AVAILABLE" else f"X={metric['world_xy'][0]:.3f} Y={metric['world_xy'][1]:.3f}"
        label = observation["semantic"]["label"] or observation["semantic"]["status"]
        text = f"{observation['scene_observation']['track_id']} {label} {world_text}"
        cv2.putText(visual, text, (int(bx1), max(20, int(by1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1, cv2.LINE_AA)
    return visual
