"""Deterministic, inference-free audit helpers for STEP 22 candidate conditioning."""

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


class CandidateConditioningAuditError(ValueError):
    pass


def load_audit_config(path):
    config = json.loads(Path(path).read_text())
    required = {
        "representative_cases", "coordinate_conversion_policy", "visualization",
        "candidate_crop", "future_conditioning_prototypes", "parameter_status",
    }
    if config.get("schema_version") != "0.1" or not required <= config.keys():
        raise CandidateConditioningAuditError("invalid candidate-conditioning audit config")
    if len(config["representative_cases"]) != 3 or config["parameter_status"] != "DEVELOPMENT_PARAMETER":
        raise CandidateConditioningAuditError("audit requires three representative development cases")
    return config


def validate_half_open_bbox(bbox, image_size):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or any(type(value) is not int for value in bbox):
        raise CandidateConditioningAuditError("bbox must contain four integer coordinates")
    width, height = image_size
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise CandidateConditioningAuditError("bbox must be valid in the half-open image domain")
    return list(bbox)


def normalize_half_open_bbox(bbox, image_size):
    values = validate_half_open_bbox(bbox, image_size)
    width, height = image_size
    extents = (width, height, width, height)
    return [int(math.floor(value / extent * 1000 + 0.5)) for value, extent in zip(values, extents)]


def candidate_metrics(image, bbox, component_area_px):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = validate_half_open_bbox(bbox, (width, height))
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    bbox_area = (x2 - x1) * (y2 - y1)
    return {
        "width_px": x2 - x1,
        "height_px": y2 - y1,
        "bbox_area_px": bbox_area,
        "bbox_area_ratio": bbox_area / (width * height),
        "component_area_px": int(component_area_px),
        "component_area_ratio": int(component_area_px) / (width * height),
        "center_px": [(x1 + x2) / 2, (y1 + y2) / 2],
        "border_distance_px": {"left": x1, "top": y1, "right": width - x2, "bottom": height - y2},
        "touches_image_border": any((x1 == 0, y1 == 0, x2 == width, y2 == height)),
        "mean_edge_magnitude": float(np.mean(cv2.magnitude(gx, gy))),
        "local_contrast_stddev": float(np.std(gray)),
    }


def make_audit_visualizations(image, bbox, track_id, normalized_bbox, settings):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = validate_half_open_bbox(bbox, (width, height))
    color = tuple(settings["rectangle_bgr"])
    highlight_color = tuple(settings["highlight_bgr"])
    thickness = settings["rectangle_thickness"]
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    rectangle = image.copy()
    cv2.rectangle(rectangle, (x1, y1), (x2 - 1, y2 - 1), color, thickness)

    highlight = image.copy()
    layer = image.copy()
    cv2.rectangle(layer, (x1, y1), (x2 - 1, y2 - 1), highlight_color, -1)
    cv2.addWeighted(layer, settings["highlight_alpha"], highlight, 1 - settings["highlight_alpha"], 0, highlight)

    center_view = image.copy()
    cv2.circle(center_view, center, settings["center_radius"], (0, 0, 255), -1)

    combined = highlight.copy()
    cv2.rectangle(combined, (x1, y1), (x2 - 1, y2 - 1), color, thickness)
    cv2.circle(combined, center, settings["center_radius"], (0, 0, 255), -1)
    label = f"{track_id} raw={list(bbox)} norm={normalized_bbox}"
    cv2.putText(combined, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return {"raw_rectangle": rectangle, "highlight": highlight, "center": center_view, "combined": combined}


def make_context_crop(image, bbox, margin_ratio):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = validate_half_open_bbox(bbox, (width, height))
    margin_x = int(math.ceil((x2 - x1) * margin_ratio))
    margin_y = int(math.ceil((y2 - y1) * margin_ratio))
    crop_bbox = [max(0, x1 - margin_x), max(0, y1 - margin_y), min(width, x2 + margin_x), min(height, y2 + margin_y)]
    cx1, cy1, cx2, cy2 = crop_bbox
    return image[cy1:cy2, cx1:cx2].copy(), {
        "source_bbox_px": crop_bbox,
        "context_margin_ratio": margin_ratio,
        "coordinate_role": "SEMANTIC_CONDITIONING_ONLY",
        "metric_coordinate_source": False,
    }


def audit_prompt(prompt):
    checks = {
        "candidate_raw_pixels_explicit": "raw pixel metadata" in prompt and "pixel-space change" in prompt,
        "image_dimensions_explicit": "image_width" in prompt or "image_height" in prompt,
        "bbox_order_explicit": "x_min" in prompt and "y_min" in prompt and "x_max" in prompt and "y_max" in prompt,
        "half_open_convention_explicit": "half-open" in prompt or "exclusive" in prompt,
        "output_normalized_0_1000_explicit": "normalized 0-1000" in prompt,
        "full_object_bbox_explicit": "full visible physical entity bbox" in prompt,
    }
    classification = "CLEAR" if all(checks.values()) else (
        "LIKELY_CONFUSING" if not checks["image_dimensions_explicit"] and not checks["half_open_convention_explicit"] else "AMBIGUOUS"
    )
    return {"classification": classification, "checks": checks}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_output_snapshot(roots):
    files = sorted(path for root in roots for path in Path(root).rglob("qwen_raw_outputs/*") if path.is_file())
    records = [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in files]
    digest = hashlib.sha256("".join(f"{item['sha256']}  {item['path']}\n" for item in records).encode()).hexdigest()
    return {"count": len(records), "digest_sha256": digest, "files": records}
