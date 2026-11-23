"""Validated Qwen bbox conversion and legacy metric grounding."""

import json
import math
from numbers import Real
from pathlib import Path
import time

from vlm_reposition.geometry import GeometryCompatibilityError


SEMANTIC_TYPES = frozenset({
    "movable_object",
    "static_obstacle",
    "person",
    "furniture",
    "workspace_element",
    "unknown",
})
NAVIGATION_ROLES = frozenset({
    "target_candidate",
    "obstacle_candidate",
    "context_only",
    "unknown",
})
GENERIC_LABELS = frozenset({
    "movable object",
    "static obstacle",
    "furniture",
    "workspace element",
    "unknown",
    "object",
    "thing",
})


class GroundingValidationError(ValueError):
    """Raised when model output cannot be grounded unambiguously."""


def validate_normalized_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise GroundingValidationError("bbox_2d must contain exactly four coordinates")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in bbox):
        raise GroundingValidationError("bbox_2d coordinates must be numbers")
    bbox = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) and 0 <= value <= 1000 for value in bbox):
        raise GroundingValidationError("bbox_2d coordinates must be finite and within 0..1000")
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        raise GroundingValidationError("bbox_2d must satisfy x1 < x2 and y1 < y2")
    return bbox


def _parse_json_object(model_output):
    if not isinstance(model_output, str) or not model_output.strip():
        raise GroundingValidationError("model output must be non-empty text")

    decoder = json.JSONDecoder()
    parsed = None
    for index, character in enumerate(model_output):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(model_output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    if parsed is None:
        raise GroundingValidationError("model output does not contain a JSON object")
    return parsed


def parse_single_detection(model_output):
    parsed = _parse_json_object(model_output)

    if "objects" in parsed:
        objects = parsed["objects"]
        if not isinstance(objects, list) or len(objects) != 1:
            count = len(objects) if isinstance(objects, list) else "invalid"
            raise GroundingValidationError(f"expected one target candidate, received {count}")
        parsed = objects[0]
    if not isinstance(parsed, dict):
        raise GroundingValidationError("detection must be a JSON object")

    label = parsed.get("label")
    if not isinstance(label, str) or not label.strip():
        raise GroundingValidationError("detection label must be non-empty text")
    return {"label": label.strip(), "bbox_2d": list(validate_normalized_bbox(parsed.get("bbox_2d")))}


def parse_scene_entities(model_output):
    parsed = _parse_json_object(model_output)
    entities = parsed.get("entities")
    if not isinstance(entities, list) or not entities:
        raise GroundingValidationError("entities must be a non-empty list")

    validated = []
    for index, entity in enumerate(entities, start=1):
        if not isinstance(entity, dict):
            raise GroundingValidationError(f"entity {index} must be a JSON object")
        label = entity.get("label")
        if not isinstance(label, str) or not label.strip():
            raise GroundingValidationError(f"entity {index} label must be non-empty text")
        semantic_type = entity.get("semantic_type")
        if semantic_type not in SEMANTIC_TYPES:
            raise GroundingValidationError(
                f"entity {index} semantic_type {semantic_type!r} is not allowed"
            )
        navigation_role = entity.get("navigation_role")
        if navigation_role not in NAVIGATION_ROLES:
            raise GroundingValidationError(
                f"entity {index} navigation_role {navigation_role!r} is not allowed"
            )
        validated.append({
            "label": label.strip(),
            "semantic_type": semantic_type,
            "navigation_role": navigation_role,
            "bbox_2d": list(validate_normalized_bbox(entity.get("bbox_2d"))),
        })
    return validated


def parse_semantic_entities(model_output):
    parsed = _parse_json_object(model_output)
    entities = parsed.get("entities")
    if not isinstance(entities, list) or not entities:
        raise GroundingValidationError("entities must be a non-empty list")

    validated = []
    for index, entity in enumerate(entities, start=1):
        if not isinstance(entity, dict):
            raise GroundingValidationError(f"entity {index} must be a JSON object")
        label = entity.get("label")
        if not isinstance(label, str) or not label.strip():
            raise GroundingValidationError(f"entity {index} label must be non-empty text")
        semantic_type = entity.get("semantic_type")
        if semantic_type not in SEMANTIC_TYPES:
            raise GroundingValidationError(
                f"entity {index} semantic_type {semantic_type!r} is not allowed"
            )
        validated.append({
            "label": label.strip(),
            "semantic_type": semantic_type,
            "bbox_2d": list(validate_normalized_bbox(entity.get("bbox_2d"))),
        })
    return validated


def parse_target_candidates(model_output):
    parsed = _parse_json_object(model_output)
    if "candidates" in parsed:
        candidates = parsed["candidates"]
    elif "objects" in parsed:
        candidates = parsed["objects"]
    elif "label" in parsed or "bbox_2d" in parsed:
        candidates = [parsed]
    else:
        raise GroundingValidationError("target output must contain candidates")
    if not isinstance(candidates, list):
        raise GroundingValidationError("target candidates must be a list")

    validated = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise GroundingValidationError(f"target candidate {index} must be a JSON object")
        label = candidate.get("label")
        if not isinstance(label, str) or not label.strip():
            raise GroundingValidationError(f"target candidate {index} label must be non-empty text")
        validated.append({
            "label": label.strip(),
            "bbox_2d": list(validate_normalized_bbox(candidate.get("bbox_2d"))),
        })
    return validated


def normalized_bbox_to_pixel_bbox(bbox, image_size):
    x1, y1, x2, y2 = validate_normalized_bbox(bbox)
    if (
        not isinstance(image_size, (list, tuple))
        or len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, Real) for value in image_size)
    ):
        raise GroundingValidationError("image_size must be (width, height)")
    width, height = (float(value) for value in image_size)
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise GroundingValidationError("image dimensions must be finite and positive")
    return (
        x1 * width / 1000,
        y1 * height / 1000,
        x2 * width / 1000,
        y2 * height / 1000,
    )


def bbox_bottom_center(pixel_bbox):
    if not isinstance(pixel_bbox, (list, tuple)) or len(pixel_bbox) != 4:
        raise GroundingValidationError("pixel bbox must contain exactly four coordinates")
    x1, y1, x2, y2 = (float(value) for value in pixel_bbox)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
        raise GroundingValidationError("pixel bbox must be finite and ordered")
    return (x1 + x2) / 2, y2


def ground_normalized_bbox(target_query, label, bbox, image_size, geometry):
    if tuple(image_size) != geometry.image_size:
        raise GroundingValidationError(
            f"Expected {geometry.image_size[0]}x{geometry.image_size[1]} legacy reference image, "
            f"received {image_size[0]}x{image_size[1]}."
        )
    model_bbox = validate_normalized_bbox(bbox)
    pixel_bbox = normalized_bbox_to_pixel_bbox(model_bbox, image_size)
    raw_pixel = bbox_bottom_center(pixel_bbox)
    undistorted_pixel = geometry.raw_pixel_to_undistorted_pixel(*raw_pixel)
    world = geometry.undistorted_pixel_to_world(*undistorted_pixel)
    return {
        "target_query": target_query,
        "label": label,
        "bbox_model_normalized_0_1000": list(model_bbox),
        "bbox_pixel_raw": list(pixel_bbox),
        "reference_point_type": "bbox_bottom_center",
        "raw_reference_pixel": {"u": raw_pixel[0], "v": raw_pixel[1]},
        "undistorted_reference_pixel": {"u": undistorted_pixel[0], "v": undistorted_pixel[1]},
        "world_coordinate": {
            "x": world[0],
            "y": world[1],
            "unit": "m",
            "frame": geometry.coordinate_system,
        },
    }


def ground_scene_entities(entities, image_size, geometry, workspace_bounds):
    if tuple(image_size) != geometry.image_size:
        raise GroundingValidationError(
            f"Expected {geometry.image_size[0]}x{geometry.image_size[1]} legacy reference image, "
            f"received {image_size[0]}x{image_size[1]}."
        )
    if len(workspace_bounds) != 4:
        raise GroundingValidationError("workspace_bounds must be (xmin, xmax, ymin, ymax)")
    xmin, xmax, ymin, ymax = (float(value) for value in workspace_bounds)
    if not all(math.isfinite(value) for value in (xmin, xmax, ymin, ymax)) or xmin > xmax or ymin > ymax:
        raise GroundingValidationError("workspace bounds must be finite and ordered")

    grounded_entities = []
    for index, entity in enumerate(entities, start=1):
        bbox = validate_normalized_bbox(entity["bbox_2d"])
        pixel_bbox = normalized_bbox_to_pixel_bbox(bbox, image_size)
        raw_pixel = bbox_bottom_center(pixel_bbox)
        result = {
            "id": f"entity_{index:03d}",
            "label": entity["label"],
            "semantic_type": entity["semantic_type"],
            "bbox_normalized_0_1000": list(bbox),
            "bbox_pixel_raw": list(pixel_bbox),
            "reference_point_type": "bbox_bottom_center",
            "raw_reference_pixel": {"u": raw_pixel[0], "v": raw_pixel[1]},
        }
        if "navigation_role" in entity:
            result["navigation_role"] = entity["navigation_role"]
        try:
            grounded = ground_normalized_bbox(
                "workspace scene", entity["label"], bbox, image_size, geometry
            )
        except GeometryCompatibilityError as exc:
            result.update({
                "undistorted_reference_pixel": None,
                "world_coordinate": None,
                "geometry_status": "INVALID",
                "geometry_error": str(exc),
            })
        else:
            world = grounded["world_coordinate"]
            result.update({
                "undistorted_reference_pixel": grounded["undistorted_reference_pixel"],
                "world_coordinate": world,
                "geometry_status": (
                    "VALID"
                    if xmin <= world["x"] <= xmax and ymin <= world["y"] <= ymax
                    else "OUTSIDE_LEGACY_WORKSPACE"
                ),
            })
        grounded_entities.append(result)
    return grounded_entities


def bbox_iou(first, second):
    ax1, ay1, ax2, ay2 = validate_normalized_bbox(first)
    bx1, by1, bx2, by2 = validate_normalized_bbox(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union


def normalize_semantic_label(label):
    return " ".join(label.casefold().replace("_", " ").replace("-", " ").split())


def semantic_label_status(label):
    normalized = normalize_semantic_label(label)
    if normalized == "unknown":
        return "UNKNOWN"
    return "GENERIC" if normalized in GENERIC_LABELS else "SPECIFIC"


def load_partial_semantic_reference(path):
    reference = json.loads(Path(path).read_text())
    if reference.get("reference_status") != "PARTIAL_SEMANTIC_REFERENCE":
        raise GroundingValidationError("reference_status must be PARTIAL_SEMANTIC_REFERENCE")
    entities = reference.get("entities")
    if not isinstance(entities, list) or not entities:
        raise GroundingValidationError("semantic reference entities must be a non-empty list")
    seen_ids = set()
    for index, entity in enumerate(entities, start=1):
        reference_id = entity.get("reference_id")
        canonical = entity.get("canonical_label")
        aliases = entity.get("aliases")
        if not isinstance(reference_id, str) or not reference_id or reference_id in seen_ids:
            raise GroundingValidationError(f"reference entity {index} has an invalid or duplicate ID")
        if not isinstance(canonical, str) or not canonical.strip():
            raise GroundingValidationError(f"reference entity {index} canonical label is invalid")
        if not isinstance(aliases, list) or not aliases or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise GroundingValidationError(f"reference entity {index} aliases are invalid")
        validate_normalized_bbox(entity.get("bbox_normalized_0_1000"))
        seen_ids.add(reference_id)
    return reference


def apply_semantic_reference(entities, reference, iou_threshold):
    if not isinstance(iou_threshold, Real) or not 0 <= iou_threshold <= 1:
        raise GroundingValidationError("IoU threshold must be within 0..1")
    enriched = []
    for entity in entities:
        item = dict(entity)
        item["semantic_label_status"] = semantic_label_status(item["label"])
        item["active_workspace_status"] = {
            "VALID": "INSIDE",
            "OUTSIDE_LEGACY_WORKSPACE": "OUTSIDE",
            "INVALID": "INVALID_GEOMETRY",
        }[item["geometry_status"]]
        item["reference_review_status"] = "UNREVIEWED"
        enriched.append(item)

    matches = []
    used = set()
    for reference_entity in reference["entities"]:
        candidates = [
            (
                bbox_iou(
                    reference_entity["bbox_normalized_0_1000"],
                    entity["bbox_normalized_0_1000"],
                ),
                index,
            )
            for index, entity in enumerate(enriched)
            if index not in used
        ]
        if not candidates:
            matches.append({
                "reference_id": reference_entity["reference_id"],
                "canonical_label": reference_entity["canonical_label"],
                "matched_entity_id": None,
                "fresh_label": None,
                "bbox_iou": 0.0,
                "alias_match": False,
                "semantic_label_status": None,
                "result": "NO_BBOX_MATCH",
            })
            continue
        overlap, index = max(candidates, key=lambda item: item[0])
        if overlap < iou_threshold:
            matches.append({
                "reference_id": reference_entity["reference_id"],
                "canonical_label": reference_entity["canonical_label"],
                "matched_entity_id": None,
                "fresh_label": None,
                "bbox_iou": overlap,
                "alias_match": False,
                "semantic_label_status": None,
                "result": "NO_BBOX_MATCH",
            })
            continue

        used.add(index)
        entity = enriched[index]
        entity["reference_review_status"] = "MATCHED_REFERENCE"
        entity["matched_reference_id"] = reference_entity["reference_id"]
        aliases = {
            normalize_semantic_label(reference_entity["canonical_label"]),
            *(normalize_semantic_label(alias) for alias in reference_entity["aliases"]),
        }
        alias_match = normalize_semantic_label(entity["label"]) in aliases
        status = entity["semantic_label_status"]
        result = "PASS" if alias_match and status == "SPECIFIC" else (
            "GENERIC_LABEL" if status != "SPECIFIC" else "LABEL_MISMATCH"
        )
        matches.append({
            "reference_id": reference_entity["reference_id"],
            "canonical_label": reference_entity["canonical_label"],
            "matched_entity_id": entity["id"],
            "fresh_label": entity["label"],
            "semantic_type": entity["semantic_type"],
            "bbox_iou": overlap,
            "alias_match": alias_match,
            "semantic_label_status": status,
            "result": result,
        })
    return enriched, matches


def ground_target_result(
    target_query,
    model_output,
    image_size,
    geometry,
    workspace_bounds,
    reference_entity=None,
    iou_threshold=0.5,
):
    try:
        candidates = parse_target_candidates(model_output)
    except GroundingValidationError as exc:
        return {
            "query": target_query,
            "target_status": "INVALID_OUTPUT",
            "error": str(exc),
            "position_status": "NOT_EVALUATED",
            "semantic_status": "NOT_EVALUATED",
        }
    if not candidates:
        return {
            "query": target_query,
            "target_status": "TARGET_NOT_FOUND",
            "candidates": [],
            "position_status": "NOT_EVALUATED",
            "semantic_status": "NOT_EVALUATED",
        }
    if len(candidates) > 1:
        return {
            "query": target_query,
            "target_status": "TARGET_AMBIGUOUS",
            "candidates": candidates,
            "position_status": "NOT_EVALUATED",
            "semantic_status": "NOT_EVALUATED",
        }

    candidate = candidates[0]
    result = {
        "query": target_query,
        "target_status": "TARGET_FOUND",
        "detected_label": candidate["label"],
        "bbox_normalized_0_1000": candidate["bbox_2d"],
        "semantic_label_status": semantic_label_status(candidate["label"]),
    }
    try:
        grounded = ground_normalized_bbox(
            target_query, candidate["label"], candidate["bbox_2d"], image_size, geometry
        )
    except GeometryCompatibilityError as exc:
        pixel_bbox = normalized_bbox_to_pixel_bbox(candidate["bbox_2d"], image_size)
        raw_pixel = bbox_bottom_center(pixel_bbox)
        result.update({
            "bbox_pixel_raw": list(pixel_bbox),
            "reference_point_type": "bbox_bottom_center",
            "raw_reference_pixel": {"u": raw_pixel[0], "v": raw_pixel[1]},
            "undistorted_reference_pixel": None,
            "world_coordinate": None,
            "active_workspace_status": "INVALID_GEOMETRY",
            "geometry_error": str(exc),
        })
    else:
        if len(workspace_bounds) != 4:
            raise GroundingValidationError("workspace_bounds must be (xmin, xmax, ymin, ymax)")
        xmin, xmax, ymin, ymax = (float(value) for value in workspace_bounds)
        if not all(math.isfinite(value) for value in (xmin, xmax, ymin, ymax)) or xmin > xmax or ymin > ymax:
            raise GroundingValidationError("workspace bounds must be finite and ordered")
        world = grounded["world_coordinate"]
        result.update({
            "bbox_pixel_raw": grounded["bbox_pixel_raw"],
            "reference_point_type": grounded["reference_point_type"],
            "raw_reference_pixel": grounded["raw_reference_pixel"],
            "undistorted_reference_pixel": grounded["undistorted_reference_pixel"],
            "world_coordinate": world,
            "active_workspace_status": (
                "INSIDE"
                if xmin <= world["x"] <= xmax and ymin <= world["y"] <= ymax
                else "OUTSIDE"
            ),
        })

    if reference_entity is None:
        result.update({
            "reference_bbox_iou": None,
            "position_status": "NOT_EVALUATED",
            "semantic_status": "UNREVIEWED",
        })
        return result

    overlap = bbox_iou(
        reference_entity["bbox_normalized_0_1000"], candidate["bbox_2d"]
    )
    position_status = (
        "REFERENCE_POSITION_MATCH"
        if overlap >= iou_threshold
        else "REFERENCE_POSITION_MISMATCH"
    )
    aliases = {
        normalize_semantic_label(reference_entity["canonical_label"]),
        *(normalize_semantic_label(alias) for alias in reference_entity["aliases"]),
    }
    alias_match = normalize_semantic_label(candidate["label"]) in aliases
    label_status = result["semantic_label_status"]
    if alias_match:
        semantic_status = "REFERENCE_ALIAS_MATCH"
    elif position_status == "REFERENCE_POSITION_MATCH" and label_status == "SPECIFIC":
        semantic_status = "HUMAN_ALIAS_CANDIDATE"
    elif label_status != "SPECIFIC":
        semantic_status = "GENERIC_LABEL"
    else:
        semantic_status = "LABEL_MISMATCH"
    result.update({
        "reference_id": reference_entity["reference_id"],
        "reference_canonical_label": reference_entity["canonical_label"],
        "reference_bbox_iou": overlap,
        "position_status": position_status,
        "alias_match": alias_match,
        "semantic_status": semantic_status,
    })
    return result


def load_local_qwen(model_id):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype="auto", device_map="auto", local_files_only=True
    )
    model.eval()
    return processor, model, time.perf_counter() - started


def prepare_qwen_inputs(processor, model, image_path, prompt):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ],
    }]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


def prepare_qwen_multi_image_inputs(processor, model, image_paths, prompt):
    if not isinstance(image_paths, (list, tuple)) or len(image_paths) < 2:
        raise GroundingValidationError("multi-image input requires at least two ordered image paths")
    messages = [{
        "role": "user",
        "content": [
            *({"type": "image", "image": str(path)} for path in image_paths),
            {"type": "text", "text": prompt},
        ],
    }]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


def generate_qwen_text(processor, model, inputs, max_new_tokens):
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_sec = time.perf_counter() - started
    output = generated[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(output, skip_special_tokens=True), inference_sec
