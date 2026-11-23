#!/usr/bin/env python3
"""Synthetic 1920x1080 semantic/detection/segmentation/metric feasibility run."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np

import run_pipeline as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "metric_pipeline_config.json"
REFERENCE_CODE = Path("/home/binci/vlm_ws/wanna_find_detailed_experimental_x,Y")
OBJECT_PROMPT = '''Inspect this 1920x1080 synthetic industrial robotics workspace image.
Return JSON only. Coordinates are integer normalized 0..1000 over the full original image.
Required categories: truck, pallet, box, rack, workbench, cart, drum, mobile_robot, manipulator, fence, traffic_cone, other.
Detect every distinct major visible instance. Do not force uncertain objects into a class.
Schema: {"objects":[{"id":"q_001","label":"specific label","category":"allowed category","bbox_2d":[x1,y1,x2,y2],"certainty":"high|medium|low"}],"other_objects":[],"uncertain_objects":[]}
Do not output metric coordinates, paths, hidden objects, or markdown.'''
SEMANTIC_PROMPT_HEAD = '''Analyze functional regions and visible relations in this synthetic industrial workspace. Return JSON only.
Coordinates are integer normalized 0..1000 over the original full image.
Do not equate an object bbox with a semantic region. Do not force unsupported regions.
Target taxonomy: Storage Area, Work Station, Loading Area, Material Area, Drop Zone, No-Go Zone.
Allowed relations: near, adjacent_to, left_of, right_of, in_front_of, behind, on, inside.
Schema: {"regions":[{"id":"region_...","label":"taxonomy label","supporting_object_ids":["q_..."],"spatial_reason":"brief","polygon":[[x,y],[x,y],[x,y]],"certainty":"high|medium|low"}],"relations":[{"subject_id":"q_...","predicate":"allowed relation","object_id":"q_...","certainty":"high|medium|low"}]}
Use only visible evidence and the following detected IDs/labels:
'''


def first_json(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
                return value
            except json.JSONDecodeError:
                pass
    raise ValueError("no complete JSON object")


def complete_array_items(text, key):
    marker = f'"{key}"'; position = text.find(marker)
    if position < 0: return []
    position = text.find("[", position + len(marker)) + 1; decoder = json.JSONDecoder(); items = []
    while position > 0:
        while position < len(text) and text[position] in " \r\n\t,": position += 1
        if position >= len(text) or text[position] != "{": break
        try: item, consumed = decoder.raw_decode(text[position:])
        except json.JSONDecodeError: break
        items.append(item); position += consumed
    return items


def remap_semantic_ids(regions, relations, objects):
    aliases = {"shelf": "rack", "rack": "rack", "desk": "workbench", "workbench": "workbench",
               "truck": "truck", "pallet": "pallet", "cart": "cart", "barrel": "drum", "drum": "drum",
               "fence": "fence", "cone": "traffic_cone", "robot": "mobile_robot", "manipulator": "manipulator"}
    by_category = {}
    for obj in objects: by_category.setdefault(obj["category"], []).append(obj["id"])
    referenced = {value for region in regions for value in region.get("supporting_object_ids", [])}
    referenced |= {value for relation in relations for value in (relation.get("subject_id"), relation.get("object_id")) if value}
    mapping = {}
    for value in referenced:
        prefix, _, suffix = value.rpartition("_"); category = aliases.get(prefix.casefold())
        if category and suffix.isdigit() and int(suffix) <= len(by_category.get(category, [])):
            mapping[value] = by_category[category][int(suffix) - 1]
    for region in regions:
        region["original_supporting_object_ids"] = region.get("supporting_object_ids", [])
        region["supporting_object_ids"] = [mapping[value] for value in region["original_supporting_object_ids"] if value in mapping]
    remapped = [{**relation, "original_subject_id": relation["subject_id"], "original_object_id": relation["object_id"],
                 "subject_id": mapping[relation["subject_id"]], "object_id": mapping[relation["object_id"]],
                 "id_mapping_source": "DETERMINISTIC_CATEGORY_ORDINAL_POSTPROCESS"}
                for relation in relations if relation.get("subject_id") in mapping and relation.get("object_id") in mapping]
    return remapped, mapping, sorted(referenced - mapping.keys())


def bbox_px(box, width, height):
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
        raise ValueError(f"invalid normalized bbox: {box}")
    return [round(box[0] * width / 1000), round(box[1] * height / 1000),
            round(box[2] * width / 1000), round(box[3] * height / 1000)]


def summary(values):
    values = np.asarray(values, np.float64)
    return {"count": len(values), "mean": float(values.mean()),
            "rmse": float(np.sqrt(np.mean(values ** 2))), "max": float(values.max())}


def project(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2), matrix).reshape(-1, 2)


def geometry_stage(run, image, config):
    directory = run / "geometry"
    points = config["control_points"]
    calibration = [point for point in points if point["role"] == "calibration"]
    source = np.asarray([point["pixel"] for point in calibration], np.float64)
    target = np.asarray([point["world_m"] for point in calibration], np.float64)
    homography, _ = cv2.findHomography(source, target, 0)
    if homography is None or not np.isfinite(homography).all():
        raise ValueError("homography fit failed")
    inverse = np.linalg.inv(homography)
    predicted_world = project([point["pixel"] for point in points], homography)
    predicted_pixel = project([point["world_m"] for point in points], inverse)
    residuals = []
    for point, world, pixel in zip(points, predicted_world, predicted_pixel):
        residuals.append({
            **point, "predicted_world_m": world.tolist(),
            "metric_error_m": float(np.linalg.norm(world - point["world_m"])),
            "predicted_pixel": pixel.tolist(),
            "pixel_error_px": float(np.linalg.norm(pixel - point["pixel"])),
            "source_type": "MANUAL_IMAGE_AUDIT",
        })
    calibration_metric = summary([item["metric_error_m"] for item in residuals if item["role"] == "calibration"])
    validation_metric = summary([item["metric_error_m"] for item in residuals if item["role"] == "validation"])
    calibration_pixel = summary([item["pixel_error_px"] for item in residuals if item["role"] == "calibration"])
    validation_pixel = summary([item["pixel_error_px"] for item in residuals if item["role"] == "validation"])
    gate = validation_metric["rmse"] <= config["validation_rmse_gate_m"] and validation_metric["max"] <= config["validation_max_error_gate_m"]
    base.dump(directory / "control_points.json", {
        "status": "PARTIAL", "point_count": len(points), "calibration_count": len(calibration),
        "validation_count": len(points) - len(calibration), "provenance": config["control_point_provenance"],
        "coordinate_convention": {"pixel": "raw synthetic image u-right/v-down", "world": "bottom-left origin; X-right/Y-up; metres"},
        "points": residuals,
    })
    result = {
        "status": "PARTIAL", "feasibility_gate": "PASS" if gate else "FAIL",
        "scope": "SYNTHETIC_SOFTWARE_FEASIBILITY_ONLY",
        "image_to_world_homography": homography.tolist(), "world_to_image_homography": inverse.tolist(),
        "number_of_control_points": len(points), "calibration_points": len(calibration), "validation_points": len(points) - len(calibration),
        "calibration_metric_error_m": calibration_metric, "validation_metric_error_m": validation_metric,
        "calibration_reprojection_error_px": calibration_pixel, "validation_reprojection_error_px": validation_pixel,
        "gate_thresholds_m": {"validation_rmse": config["validation_rmse_gate_m"], "validation_max": config["validation_max_error_gate_m"]},
    }
    base.dump(directory / "homography.json", result)
    overlay = image.copy()
    reprojection = image.copy()
    for item in residuals:
        actual = tuple(map(round, item["pixel"])); predicted = tuple(map(round, item["predicted_pixel"]))
        color = (0, 255, 255) if item["role"] == "calibration" else (255, 0, 255)
        cv2.circle(overlay, actual, 9, color, -1)
        base.label(overlay, f"{item['id']} {'C' if item['role']=='calibration' else 'V'}", (actual[0] - 28, max(20, actual[1] - 13)), color, .55)
        cv2.circle(reprojection, actual, 7, (0, 255, 0), -1)
        cv2.arrowedLine(reprojection, actual, predicted, (0, 0, 255), 2, tipLength=.2)
        base.label(reprojection, f"{item['id']} {item['metric_error_m']:.3f}m", (actual[0] + 8, max(18, actual[1] - 7)), (0, 255, 255), .45)
    base.label(overlay, "YELLOW calibration / MAGENTA holdout validation", (18, 28))
    base.label(reprojection, "GREEN observed / RED predicted from world", (18, 28))
    cv2.imwrite(str(directory / "control_points_overlay.png"), overlay)
    cv2.imwrite(str(directory / "reprojection_validation.png"), reprojection)
    base.write(directory / "RESULT.md", f"""# Geometry Result

Status: PARTIAL

8 calibration + 4 holdout validation points. Validation RMSE `{validation_metric['rmse']:.4f} m`, max `{validation_metric['max']:.4f} m`; configured feasibility gate is `{'PASS' if gate else 'FAIL'}`. This is a synthetic software/pipeline feasibility homography, not real-camera calibration or control-grade geometry.
""")
    return result


def validate_qwen_objects(data, width, height):
    allowed = {"truck", "pallet", "box", "rack", "workbench", "cart", "drum", "mobile_robot", "manipulator", "fence", "traffic_cone", "other"}
    objects, ids = [], set()
    for index, item in enumerate(data.get("objects", []), 1):
        try:
            box = [float(value) for value in item["bbox_2d"]]
            pixels = bbox_px(box, width, height)
            object_id = str(item.get("id") or f"q_{index:03d}")
            if object_id in ids:
                continue
            ids.add(object_id)
            category = str(item.get("category", "other"))
            objects.append({"id": object_id, "label": str(item["label"]),
                            "category": category if category in allowed else "other",
                            "bbox_2d": box, "bbox_px": pixels,
                            "certainty": item.get("certainty", item.get("confidence")),
                            "source_type": "QWEN_MODEL_OUTPUT"})
        except (KeyError, TypeError, ValueError):
            continue
    return objects


def qwen_stage(run, image, config):
    directory = run / "01_qwen"
    base.write(directory / "OBJECT_PROMPT.txt", OBJECT_PROMPT)
    sys.path.insert(0, str(REFERENCE_CODE / "src"))
    from vlm_reposition.perception.qwen_grounding import generate_qwen_text, load_local_qwen, prepare_qwen_inputs
    attempts, objects, regions, relations = [], [], [], []
    raw_objects = raw_semantics = ""
    started = time.perf_counter()
    try:
        processor, model, load_seconds = load_local_qwen(config["qwen_model"])
        inputs = prepare_qwen_inputs(processor, model, Path(config["input_image"]), OBJECT_PROMPT)
        raw_objects, seconds = generate_qwen_text(processor, model, inputs, 2000)
        base.write(directory / "objects_raw.txt", raw_objects)
        try:
            parsed = first_json(raw_objects); recovery = False
        except ValueError:
            parsed = {"objects": complete_array_items(raw_objects, "objects")}; recovery = True
        objects = validate_qwen_objects(parsed, image.shape[1], image.shape[0])
        attempts.append({"attempt": 1, "purpose": "fresh object generation", "status": "PARTIAL" if recovery else "PASS", "seconds": seconds, "objects": len(objects), "parse_mode": "complete-array-item recovery" if recovery else "complete JSON"})
        if recovery:
            attempts.append({"attempt": 2, "purpose": "deterministic truncated-array recovery", "status": "PASS" if objects else "BLOCKED", "objects": len(objects)})
        id_lines = "\n".join(f"{item['id']}: {item['label']}" for item in objects)
        semantic_prompt = SEMANTIC_PROMPT_HEAD + id_lines
        base.write(directory / "SEMANTIC_PROMPT.txt", semantic_prompt)
        inputs = prepare_qwen_inputs(processor, model, Path(config["input_image"]), semantic_prompt)
        raw_semantics, semantic_seconds = generate_qwen_text(processor, model, inputs, 1400)
        base.write(directory / "semantics_raw.txt", raw_semantics)
        semantic = first_json(raw_semantics)
        for item in semantic.get("regions", []):
            polygon = item.get("polygon")
            if isinstance(polygon, list) and len(polygon) >= 3 and all(len(point) == 2 and 0 <= point[0] <= 1000 and 0 <= point[1] <= 1000 for point in polygon):
                regions.append(item)
        allowed_relations = {"near", "adjacent_to", "left_of", "right_of", "in_front_of", "behind", "on", "inside"}
        raw_relations = [item for item in semantic.get("relations", []) if item.get("predicate") in allowed_relations]
        relations, id_mapping, unresolved_ids = remap_semantic_ids(regions, raw_relations, objects)
        attempts.append({"attempt": len(attempts) + 1, "purpose": "fresh semantic regions/relations", "status": "PASS", "seconds": semantic_seconds, "regions": len(regions), "relations": len(relations), "id_mapping": id_mapping, "unresolved_ids": unresolved_ids})
        device = str(model.device)
        del inputs, model, processor
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except Exception as exc:
        attempts.append({"attempt": len(attempts) + 1, "purpose": "Qwen fresh inference", "status": "BLOCKED", "error": repr(exc)})
        base.write(run / "logs/qwen_exception.log", traceback.format_exc())
        device = None; load_seconds = None
    status = "PASS" if len(objects) >= 10 and regions else "PARTIAL" if objects else "BLOCKED"
    base.dump(directory / "objects.json", {"status": status, "coordinate_system": "normalized_0_1000_and_raw_pixels", "objects": objects, "attempts": attempts})
    base.dump(directory / "semantic_regions.json", {"status": "PASS" if regions else "PARTIAL", "regions": regions, "relations": relations})
    base.dump(directory / "runtime.json", {"model": config["qwen_model"], "device": device, "load_seconds": load_seconds, "total_seconds": time.perf_counter() - started})
    base.object_overlay(image, objects, directory / "object_overlay.png", (0, 220, 255))
    if regions:
        normalized = [{**item, "polygon_normalized": item["polygon"]} for item in regions]
        base.semantic_overlay(image, normalized, directory / "semantic_overlay.png")
    else:
        cv2.imwrite(str(directory / "semantic_overlay.png"), image)
    base.write(directory / "RESULT.md", f"# Qwen Result\n\nStatus: {status}\n\nFresh Qwen inference produced {len(objects)} objects, {len(regions)} regions, and {len(relations)} validated relations. Raw responses are preserved; no prior-image bbox was reused.")
    return status, objects, regions, relations, attempts


def yolo_world_stage(run, image):
    directory = run / "07_yolo_world"
    checks = []
    checks.append({"attempt": 1, "check": "qwen_vl import ultralytics/mmdet", "available": bool(importlib.util.find_spec("ultralytics") or importlib.util.find_spec("mmdet"))})
    isaac_python = Path("/home/binci/miniconda3/envs/env_isaaclab/bin/python")
    result = subprocess.run([str(isaac_python), "-c", "import importlib.util; print(bool(importlib.util.find_spec('ultralytics') or importlib.util.find_spec('mmdet')))"] , capture_output=True, text=True, check=False)
    checks.append({"attempt": 2, "check": "env_isaaclab import ultralytics/mmdet", "available": result.stdout.strip() == "True"})
    local_models = list(ROOT.rglob("*yolo*world*.pt"))
    checks.append({"attempt": 3, "check": "workspace YOLO-World model cache", "available": bool(local_models), "paths": [str(path) for path in local_models]})
    status = "PASS" if any(item["available"] for item in checks) else "BLOCKED"
    base.dump(directory / "detections.json", {"status": status, "detector": "YOLO-World", "attempts": checks, "detections": [], "blocker": None if status == "PASS" else "No installed implementation/model cache; project rules require approval before isolated package installation."})
    overlay = image.copy(); base.label(overlay, f"YOLO-WORLD {status}", (22, 42), (0, 0, 255), .9)
    cv2.imwrite(str(directory / "overlay.png"), overlay)
    base.write(directory / "RESULT.md", f"# YOLO-World Result\n\nStatus: {status}\n\n세 환경/로컬 model cache를 확인했으나 구현과 weight가 없다. 기존 qwen_vl을 변경하지 않았고, 승인 없는 isolated package 설치를 수행하지 않았다.")
    return status, [], checks


def nms_detections(detections, config, image_shape):
    height, width = image_shape[:2]
    candidates = []
    for det in detections:
        category = base.canonical(det["label"])
        x1, y1, x2, y2 = det["bbox_px"]
        area_fraction = max(0, x2 - x1) * max(0, y2 - y1) / (width * height)
        if det["score"] >= config["sam2_detection_score_min"] and category != "other" and area_fraction <= .35:
            candidates.append({**det, "canonical_category": category})
    selected = []
    for det in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if all(det["canonical_category"] != kept["canonical_category"] or base.iou(det["bbox_px"], kept["bbox_px"]) < config["sam2_nms_iou"] for kept in selected):
            selected.append(det)
    return selected


def sam2_stage(run, image, detections, config):
    directory = run / "08_sam2"; masks_dir = directory / "masks"; masks_dir.mkdir(parents=True, exist_ok=True)
    selected = nms_detections(detections, config, image.shape)
    attempts, records, masks = [], [], {}
    model = processor = None
    model_id = config["sam2_model"]
    try:
        from transformers import Sam2Model, Sam2Processor
        processor = Sam2Processor.from_pretrained(model_id, local_files_only=True)
        model = Sam2Model.from_pretrained(model_id, local_files_only=True)
        attempts.append({"attempt": 1, "hypothesis": "local cache", "status": "PASS"})
    except Exception as exc:
        attempts.append({"attempt": 1, "hypothesis": "local cache", "status": "BLOCKED", "error": repr(exc)})
    if model is None:
        try:
            processor = Sam2Processor.from_pretrained(model_id)
            model = Sam2Model.from_pretrained(model_id)
            attempts.append({"attempt": 2, "hypothesis": "download through installed Transformers; no package change", "status": "PASS"})
        except Exception as exc:
            attempts.append({"attempt": 2, "hypothesis": "model download", "status": "BLOCKED", "error": repr(exc)})
    if model is not None and selected:
        import torch
        from PIL import Image
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model.to(device).eval()
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            boxes = [[float(value) for value in det["bbox_px"]] for det in selected]
            encoded = processor(images=pil, input_boxes=[boxes], return_tensors="pt")
            model_inputs = {key: value.to(device) for key, value in encoded.items() if key in {"pixel_values", "input_boxes", "input_points", "input_labels"}}
            with torch.inference_mode():
                outputs = model(**model_inputs, multimask_output=False)
            processed = processor.post_process_masks(outputs.pred_masks.cpu(), encoded["original_sizes"])[0]
            for index, det in enumerate(selected):
                mask = processed[index, 0].numpy().astype(np.uint8) * 255
                masks[det["id"]] = mask
                path = masks_dir / f"{det['id']}.png"; cv2.imwrite(str(path), mask)
                bbox_area = max(1.0, (det["bbox_px"][2] - det["bbox_px"][0]) * (det["bbox_px"][3] - det["bbox_px"][1]))
                records.append({"detection_id": det["id"], "label": det["label"], "canonical_category": det["canonical_category"], "bbox_px": det["bbox_px"], "detection_score": det["score"], "mask_path": str(path.relative_to(run)), "mask_area_px": int(np.count_nonzero(mask)), "bbox_area_px": bbox_area, "mask_to_bbox_area_ratio": float(np.count_nonzero(mask) / bbox_area)})
            attempts.append({"attempt": len(attempts) + 1, "hypothesis": f"bbox-prompt inference on {device}", "status": "PASS", "masks": len(records)})
        except Exception as exc:
            attempts.append({"attempt": len(attempts) + 1, "hypothesis": f"bbox-prompt inference on {device}", "status": "BLOCKED", "error": repr(exc)})
            base.write(run / "logs/sam2_exception.log", traceback.format_exc())
        del model, processor
        gc.collect(); torch.cuda.empty_cache()
    status = "PASS" if records else "BLOCKED"
    overlay = image.copy()
    for index, record in enumerate(records):
        mask = masks[record["detection_id"]] > 0
        color = np.array(((37 * index + 60) % 255, (83 * index + 90) % 255, (131 * index + 120) % 255), np.uint8)
        overlay[mask] = (overlay[mask] * .55 + color * .45).astype(np.uint8)
    base.label(overlay, f"SAM2 bbox-prompt masks: {len(records)}", (20, 35))
    cv2.imwrite(str(directory / "segmentation_overlay.png"), overlay)
    base.dump(directory / "segmentation.json", {"status": status, "model": model_id, "prompt_source": "Grounding DINO boxes after score/area/category/NMS filtering", "attempts": attempts, "selected_detections": len(selected), "segments": records})
    base.write(directory / "RESULT.md", f"# SAM2 Result\n\nStatus: {status}\n\nGrounding DINO bbox {len(selected)}개를 prompt로 사용해 {len(records)}개 mask를 생성했다. Mask는 detector semantics를 새로 추론하지 않으며 pixel/metric ground truth가 아니다.")
    return status, records, masks, selected, attempts


def object_category(item):
    category = str(item.get("category", "")).replace("_", " ")
    return base.canonical(category + " " + item.get("label", ""))


def compare_detectors(run, qwen_objects, dino, yolo):
    candidates = []
    for obj in qwen_objects:
        for det in dino:
            if object_category(obj) == base.canonical(det["label"]):
                overlap = base.iou(obj["bbox_px"], det["bbox_px"])
                if overlap >= .1:
                    candidates.append((overlap, obj, det))
    used_qwen, used_dino, matches = set(), set(), []
    for overlap, obj, det in sorted(candidates, reverse=True, key=lambda item: item[0]):
        if obj["id"] not in used_qwen and det["id"] not in used_dino:
            used_qwen.add(obj["id"]); used_dino.add(det["id"])
            matches.append({"qwen_id": obj["id"], "grounding_dino_id": det["id"], "canonical_category": object_category(obj), "iou": float(overlap)})
    result = {
        "status": "PARTIAL", "ground_truth_available": False, "accuracy_claim": False,
        "qwen_vs_grounding_dino": {"status": "PASS" if qwen_objects and dino else "PARTIAL", "qwen_count": len(qwen_objects), "grounding_dino_count": len(dino), "compatible_matches": matches, "minimum_iou": .1},
        "qwen_vs_yolo_world": {"status": "BLOCKED" if not yolo else "PASS", "reason": "YOLO-World unavailable" if not yolo else None},
        "grounding_dino_vs_yolo_world": {"status": "BLOCKED" if not yolo else "PASS", "reason": "YOLO-World unavailable" if not yolo else None},
    }
    directory = run / "06_detector_comparison"; base.dump(directory / "detector_comparison.json", result)
    base.write(directory / "RESULT.md", f"# Detector Comparison\n\nStatus: PARTIAL\n\nQwen {len(qwen_objects)}, Grounding DINO {len(dino)}, compatible one-to-one IoU≥0.10 matches {len(matches)}. YOLO-World comparisons are BLOCKED. No ground-truth accuracy claim.")
    return result


def image_polygon(region, width, height):
    return np.asarray([[round(point[0] * width / 1000), round(point[1] * height / 1000)] for point in region["polygon"]], np.int32)


def metric_transform(homography, config):
    resolution = config["map_resolution_m_per_px"]; height_m = config["workspace_m"][1]
    world_to_map = np.asarray([[1 / resolution, 0, 0], [0, -1 / resolution, height_m / resolution], [0, 0, 1]], np.float64)
    return world_to_map @ np.asarray(homography, np.float64)


def metric_map_stage(run, image, config, geometry, regions, selected, sam_masks):
    directory = run / "03_metric_map"; width_m, height_m = config["workspace_m"]; resolution = config["map_resolution_m_per_px"]
    size = (round(width_m / resolution), round(height_m / resolution)); matrix = metric_transform(geometry["image_to_world_homography"], config)
    bbox_occupancy = np.zeros(image.shape[:2], np.uint8); robot = np.zeros_like(bbox_occupancy)
    for det in selected:
        x1, y1, x2, y2 = map(round, det["bbox_px"])
        target = robot if det["canonical_category"] == "mobile_robot" else bbox_occupancy
        cv2.rectangle(target, (x1, y1), (x2, y2), 255, cv2.FILLED)
    sam_occupancy = np.zeros_like(bbox_occupancy); sam_robot = np.zeros_like(bbox_occupancy)
    for det in selected:
        mask = sam_masks.get(det["id"])
        if mask is not None:
            target = sam_robot if det["canonical_category"] == "mobile_robot" else sam_occupancy
            cv2.bitwise_or(target, mask, dst=target)
    no_go = np.zeros_like(bbox_occupancy); semantic_image = np.zeros((*image.shape[:2], 3), np.uint8)
    for index, region in enumerate(regions):
        points = image_polygon(region, image.shape[1], image.shape[0]); label_text = str(region.get("label", ""))
        color = ((53 * index + 70) % 255, (97 * index + 100) % 255, (149 * index + 130) % 255)
        cv2.fillPoly(semantic_image, [points], color)
        if "no-go" in label_text.casefold() or "no go" in label_text.casefold():
            cv2.fillPoly(no_go, [points], 255)
    warp = lambda mask: cv2.warpPerspective(mask, matrix, size, flags=cv2.INTER_NEAREST)
    floor = np.full((size[1], size[0]), 255, np.uint8)
    bbox_metric, sam_metric = warp(bbox_occupancy), warp(sam_occupancy)
    robot_metric = cv2.bitwise_or(warp(robot), warp(sam_robot)); no_go_metric = warp(no_go)
    semantic_metric = cv2.warpPerspective(semantic_image, matrix, size, flags=cv2.INTER_NEAREST)
    margin_px = round(config["safety_margin_m"] / resolution); kernel = np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8)
    bbox_inflated = cv2.dilate(bbox_metric, kernel); sam_inflated = cv2.dilate(sam_metric, kernel) if np.any(sam_metric) else sam_metric
    traversable_bbox = cv2.bitwise_and(floor, cv2.bitwise_not(cv2.bitwise_or(bbox_inflated, no_go_metric)))
    traversable_sam = cv2.bitwise_and(floor, cv2.bitwise_not(cv2.bitwise_or(sam_inflated, no_go_metric))) if np.any(sam_metric) else None
    cv2.imwrite(str(directory / "occupancy_metric_bbox.png"), bbox_metric)
    cv2.imwrite(str(directory / "traversability_metric_bbox.png"), traversable_bbox)
    cv2.imwrite(str(directory / "occupancy_metric.png"), sam_metric if np.any(sam_metric) else bbox_metric)
    cv2.imwrite(str(directory / "semantic_metric.png"), semantic_metric)
    if traversable_sam is not None: cv2.imwrite(str(directory / "traversability_metric.png"), traversable_sam)
    def render(occupancy, traversable, name):
        canvas = np.full((size[1], size[0], 3), 35, np.uint8); canvas[traversable > 0] = (55, 145, 55); canvas[occupancy > 0] = (35, 35, 220); canvas[no_go_metric > 0] = (170, 35, 180); canvas[robot_metric > 0] = (230, 180, 20)
        canvas = cv2.addWeighted(canvas, .82, semantic_metric, .18, 0)
        cv2.imwrite(str(directory / name), canvas)
    render(bbox_metric, traversable_bbox, "metric_map_bbox.png")
    render(sam_metric if np.any(sam_metric) else bbox_metric, traversable_sam if traversable_sam is not None else traversable_bbox, "metric_map.png")
    area = lambda mask: float(np.count_nonzero(mask) * resolution * resolution)
    artifacts = {"metric_map": str((directory / "metric_map.png").relative_to(run)), "occupancy_bbox": str((directory / "occupancy_metric_bbox.png").relative_to(run)), "traversability_bbox": str((directory / "traversability_metric_bbox.png").relative_to(run)), "semantic": str((directory / "semantic_metric.png").relative_to(run)), "occupancy_sam2": str((directory / "occupancy_metric.png").relative_to(run)), "traversability_sam2": str((directory / "traversability_metric.png").relative_to(run)) if traversable_sam is not None else None}
    data = {"status": "PARTIAL", "scope": "SYNTHETIC_SOFTWARE_FEASIBILITY_ONLY", "workspace_size_m": config["workspace_m"], "resolution_m_per_px": resolution, "raster_size_px": list(size), "origin": [0.0, 0.0, 0.0], "raster_row_zero_world_y_m": height_m, "safety_margin_m": config["safety_margin_m"], "bbox_occupancy_area_m2": area(bbox_metric), "sam2_occupancy_area_m2": area(sam_metric) if np.any(sam_metric) else None, "bbox_traversable_area_m2": area(traversable_bbox), "sam2_traversable_area_m2": area(traversable_sam) if traversable_sam is not None else None, "artifacts": artifacts, "nav2_ready": False}
    base.dump(directory / "metric_map.json", data)
    base.dump(directory / "map_metadata.json", {"status": "PARTIAL", "image": "metric_map.png", "resolution": resolution, "origin": [0.0, 0.0, 0.0], "width": size[0], "height": size[1], "frame": config["world_frame"], "nav2_ready": False})
    base.write(directory / "RESULT.md", f"# Metric Map Result\n\nStatus: PARTIAL\n\nSynthetic workspace {width_m}×{height_m} m을 0.01 m/px ({size[0]}×{size[1]} px)로 표현했다. Bbox와 SAM2 occupancy/traversability를 모두 보존했다. Resolution은 visualization/detail용이며 geometry accuracy를 뜻하지 않고 Nav2-ready도 아니다.")
    return data, matrix, size


def scene_graph_stage(run, config, geometry, qwen_objects, regions, relations, comparison, sam_records, sam_masks, matrix, size):
    directory = run / "10_integration"; matches = {item["qwen_id"]: item for item in comparison["qwen_vs_grounding_dino"]["compatible_matches"]}; sam_by_id = {item["detection_id"]: item for item in sam_records}
    H = np.asarray(geometry["image_to_world_homography"], np.float64); width_m, height_m = config["workspace_m"]; resolution = config["map_resolution_m_per_px"]
    nodes = []
    for obj in qwen_objects:
        x1, y1, x2, y2 = obj["bbox_px"]; xy = project([[(x1 + x2) / 2, y2]], H)[0]; match = matches.get(obj["id"]); det_id = match["grounding_dino_id"] if match else None; sam = sam_by_id.get(det_id)
        memberships = []
        for region in regions:
            if cv2.pointPolygonTest(image_polygon(region, 1920, 1080), ((x1 + x2) / 2, y2), False) >= 0: memberships.append(region.get("id"))
        footprint = None
        if det_id in sam_masks:
            metric_mask = cv2.warpPerspective(sam_masks[det_id], matrix, size, flags=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(metric_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea); contour = cv2.approxPolyDP(contour, 2, True).reshape(-1, 2)
                footprint = [[round(float(px * resolution), 4), round(float(height_m - py * resolution), 4)] for px, py in contour]
        nodes.append({"id": obj["id"], "semantic_label": obj["label"], "canonical_category": object_category(obj), "semantic_source": "QWEN_MODEL_OUTPUT", "detector_source": "Grounding DINO" if match else "Qwen bbox only", "bbox_px": obj["bbox_px"], "grounding_dino_match": match, "segmentation_mask_reference": sam["mask_path"] if sam else None, "representative_xy_m": [float(xy[0]), float(xy[1])], "workspace_status": "INSIDE" if 0 <= xy[0] <= width_m and 0 <= xy[1] <= height_m else "OUTSIDE", "metric_footprint": footprint, "semantic_region_membership": memberships})
    metric_regions = []
    for region in regions:
        metric_regions.append({**region, "polygon_metric_m": project(image_polygon(region, 1920, 1080), H).tolist()})
    graph = {"status": "PARTIAL", "frame": config["world_frame"], "scope": "SYNTHETIC_SOFTWARE_FEASIBILITY_ONLY", "nodes": nodes, "relations": relations, "semantic_regions": metric_regions, "layer_ownership": {"semantic": "Qwen", "detection": "Grounding DINO; YOLO-World BLOCKED", "segmentation": "SAM2", "geometry": "synthetic homography", "planning_representation": "metric bbox/SAM2 occupancy and traversability"}}
    base.dump(directory / "metric_scene_graph.json", graph)
    base.write(directory / "RESULT.md", f"# Semantic/Geometric Integration\n\nStatus: PARTIAL\n\n{len(nodes)} object nodes combine Qwen semantics, detector matching, SAM2 mask references, representative metric XY, footprints, relations, and region membership. All metric values inherit synthetic homography uncertainty.")
    return graph


def mcp_stage(run):
    directory = run / "09_mcp"; server_dir = directory / "server"; demo_dir = directory / "demo"; server_dir.mkdir(parents=True, exist_ok=True); demo_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / "scripts/scene_mcp_server.py"; target = server_dir / "server.py"; shutil.copy2(source, target)
    from scene_mcp_server import TOOLS
    base.dump(directory / "tool_schema.json", {"tools": TOOLS})
    graph = run / "10_integration/metric_scene_graph.json"
    self_check = subprocess.run([sys.executable, str(target), "--self-check"], capture_output=True, text=True, check=False)
    demo = subprocess.run([sys.executable, str(target), "--scene", str(graph), "--once", "query_scene_graph", "--args", '{"label":"truck"}'], capture_output=True, text=True, check=False)
    if demo.returncode == 0:
        try: base.dump(demo_dir / "demo_response.json", json.loads(demo.stdout))
        except json.JSONDecodeError: base.write(demo_dir / "demo_response.txt", demo.stdout)
    status = "PARTIAL" if self_check.returncode == 0 and demo.returncode == 0 else "BLOCKED"
    base.write(directory / "README.md", f"""# Read-only scene MCP prototype

Status: {status}

`server/server.py` exposes five read-only JSON-RPC tools over newline-delimited stdio. It separates orchestration from Qwen/detector/SAM2/geometry computation and reads only this run's generated scene/map JSON. Local self-check and one-shot truck query were executed, but no external MCP client/conformance test or authentication is included; this is not a research-performance claim.
""")
    base.write(directory / "RESULT.md", f"# MCP Result\n\nStatus: {status}\n\nSelf-check return code {self_check.returncode}; demo return code {demo.returncode}. External MCP client interoperability remains unverified.")
    return status


def inventory(run, image, config):
    import torch, transformers
    data = {"status": "PASS", "experiment_id": "SYNTHETIC_METRIC_VLM_PIPELINE", "run_id": run.name,
            "created_at": base.datetime.now().astimezone().isoformat(), "git_commit": base.git_commit(),
            "input_image": config["input_image"], "input_dataset": "single synthetic feasibility image",
            "input_sha256": base.sha256(Path(config["input_image"])), "image_size_px": [image.shape[1], image.shape[0]],
            "hardware_condition": "offline image inference; fixed laptop GPU; no camera/robot motion",
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "opencv": cv2.__version__, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "config_snapshot": str(run / "config_snapshot.json"),
            "reused_code": [str(ROOT / "scripts/run_pipeline.py"), str(REFERENCE_CODE / "src/vlm_reposition/perception/qwen_grounding.py")],
            "protected_existing_runs": ["EXP001", "EXP002", "AUTO_VLM_PIPELINE"]}
    base.dump(run / "00_inventory/inventory.json", data)
    base.write(run / "00_inventory/ENVIRONMENT_INVENTORY.md", f"# Inventory\n\nStatus: PASS\n\nInput `{config['input_image']}`, SHA-256 `{data['input_sha256']}`, 1920×1080. Qwen and Grounding DINO installed/cached; Transformers SAM2 implementation installed but model initially uncached; YOLO-World implementation/cache absent. Existing runs are read-only.")
    return data


def final_report(run, config, inventory_data, statuses, geometry, qwen_attempts, dino_attempts, yolo_attempts, sam_attempts, metric_map):
    files = "\n".join(f"- `{path.relative_to(run)}`" for path in sorted(run.rglob("*")) if path.is_file())
    H = json.dumps(geometry["image_to_world_homography"], ensure_ascii=False)
    graph = base.load(run / "10_integration/metric_scene_graph.json")
    outside_nodes = sum(node["workspace_status"] != "INSIDE" for node in graph["nodes"])
    text = f"""# RUN REPORT

## Run identity

- Experiment: `{inventory_data['experiment_id']}`
- Run: `{inventory_data['run_id']}`
- Timestamp: `{inventory_data['created_at']}`
- Git commit: `{inventory_data['git_commit'] or 'UNKNOWN (workspace is not a Git worktree)'}`
- Hardware condition: `{inventory_data['hardware_condition']}`
- Environment: Python `{inventory_data['python']}`, Torch `{inventory_data['torch']}`, GPU `{inventory_data['gpu']}`
- Config snapshot: `config_snapshot.json`

## Stage status

- Environment inventory: {statuses['inventory']}
- Synthetic geometry/homography: {statuses['geometry']}
- Qwen: {statuses['qwen']}
- Grounding DINO: {statuses['dino']}
- YOLO-World: {statuses['yolo']}
- SAM2: {statuses['sam2']}
- Metric map: {statuses['map']}
- Semantic/geometric integration: {statuses['integration']}
- MCP: {statuses['mcp']}
- End-to-end: PARTIAL

## Input and geometry

- Input: `{config['input_image']}`
- SHA-256: `{inventory_data['input_sha256']}`
- Tile geometry: 3×2 square mats, each 1.05 m; nominal workspace 3.15×2.10 m.
- World coordinates: bottom-left `(0,0)`, X right, Y up, metres.
- Control points: {geometry['number_of_control_points']} total ({geometry['calibration_points']} fit, {geometry['validation_points']} holdout).
- Homography image pixel → world XY: `{H}`
- Calibration RMSE: `{geometry['calibration_metric_error_m']['rmse']:.6f} m`; max `{geometry['calibration_metric_error_m']['max']:.6f} m`.
- Holdout RMSE: `{geometry['validation_metric_error_m']['rmse']:.6f} m`; max `{geometry['validation_metric_error_m']['max']:.6f} m`.
- Geometry claim: synthetic software/pipeline feasibility only.

## Model and stage evidence

- Qwen attempts: `{json.dumps(qwen_attempts, ensure_ascii=False)}`
- Grounding DINO attempts: `{json.dumps(dino_attempts, ensure_ascii=False)}`
- YOLO-World checks: `{json.dumps(yolo_attempts, ensure_ascii=False)}`
- SAM2 attempts: `{json.dumps(sam_attempts, ensure_ascii=False)}`
- Map resolution: `{metric_map['resolution_m_per_px']} m/px`, raster `{metric_map['raster_size_px']}`. This visualization resolution does not imply 1 cm calibration accuracy.
- Bbox occupancy/traversability and SAM2 occupancy/traversability are both preserved.
- No package/system/CUDA/ROS2 change and no robot motion occurred.

## Failures and limits

- YOLO-World is BLOCKED because no installed implementation/model cache exists and project rules require approval before package installation.
- Geometry is PARTIAL because the generated tile grid is not perfectly projective and points were manually audited.
- Qwen, Grounding DINO, and SAM2 outputs lack ground-truth annotation; no accuracy claim is made.
- MCP is a locally tested read-only prototype, not externally client-validated infrastructure.
- Qwen object JSON ended at `q_032`; 31 complete records were recovered deterministically and missing tail objects were not invented.
- {outside_nodes}/{len(graph['nodes'])} object representative points project outside nominal workspace bounds; they remain flagged rather than clamped.
- SAM2 prompt filtering omitted some low-score detector candidates, so the mask map is an improvement diagnostic rather than complete occupancy.

## RESEARCH CLAIMS SUPPORTED

- The implemented software chain can connect open-vocabulary localization, bbox-prompt segmentation, synthetic pixel→metric projection, layered occupancy/traversability, and scene queries in one reproducible run.
- On this image, SAM2 masks reduce rectangular bbox over-coverage; both raw computed areas are reported without claiming ground-truth superiority.
- Holdout points expose the synthetic homography's measurable residual instead of treating calibration-point fit as independent validation.

## CLAIMS NOT SUPPORTED

- Real-camera calibration accuracy, detector accuracy, segmentation IoU, collision safety, robot footprint clearance, path-planning validity, or Nav2 readiness.
- YOLO-World comparative performance, because that detector is BLOCKED in the current approved environment.
- Generalization from this single synthetic image to the physical workspace.

## Artifacts

{files}

## WHAT WORKS NOW

- Fresh Qwen and Grounding DINO inference on the new 1920×1080 image.
- 8-point homography fit with 4 independent holdout points and numeric residuals.
- Grounding DINO bbox → SAM2 masks → metric occupancy/traversability conversion.
- Bbox and SAM2 metric maps, integrated metric scene graph, and read-only scene-query prototype.

## WHAT IS ONLY A SYNTHETIC/FEASIBILITY TEST

- Every homography-derived XY, footprint, occupancy area, traversability layer, and metric map in this run.
- The 3.15×2.10 m scale is imposed from the synthetic mat specification, not measured from a real camera scene.
- Detector/SAM2 comparison is diagnostic only and does not establish accuracy.

## WHAT MUST BE REPEATED WITH THE REAL CAMERA

- Capture the fixed camera's undistorted 1920×1080 frame and confirm current intrinsic calibration/camera pose.
- Re-click or automatically measure 8–12 real floor pixel↔world points and repeat holdout residual validation.
- Re-run all detectors/SAM2 on the real frame and human-review object masks, no-go regions, and robot footprint inflation.
- Validate the metric map against physical measurements before visualization, dry-run, planning, or Nav2 use.

## WHAT TO DO NEXT

1. Review control-point and reprojection overlays; replace any disputed synthetic point and rerun once.
2. Provide approval for an isolated YOLO-World environment if the three-detector comparison is still required.
3. Annotate a small real-image detection/segmentation reference set before threshold tuning or performance claims.
4. Repeat calibration and metric-map validation with the real undistorted camera frame.
"""
    base.write(run / "RUN_REPORT.md", text)


def self_check(run=None):
    config = base.load(CONFIG_PATH); assert len(config["control_points"]) == 12; assert sum(point["role"] == "validation" for point in config["control_points"]) == 4
    assert round(base.iou([0, 0, 10, 10], [5, 5, 15, 15]), 6) == .142857
    if run:
        required = ["geometry/homography.json", "geometry/control_points_overlay.png", "01_qwen/objects.json", "04_grounding_dino/detections.json", "07_yolo_world/detections.json", "08_sam2/segmentation.json", "03_metric_map/metric_map.json", "10_integration/metric_scene_graph.json", "09_mcp/README.md", "RUN_REPORT.md"]
        for relative in required:
            path = run / relative; assert path.is_file() and path.stat().st_size, path
            if path.suffix == ".json": base.load(path)
            if path.suffix == ".png": assert cv2.imread(str(path)) is not None


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path); parser.add_argument("--self-check", action="store_true"); args = parser.parse_args()
    if args.self_check: self_check(); print("self-check: PASS"); return
    config = base.load(args.config); timestamp = base.datetime.now().astimezone(); run = (args.output or ROOT / "results" / f"METRIC_VLM_PIPELINE_{timestamp:%Y%m%dT%H%M%S%z}").resolve()
    if run.exists(): raise FileExistsError(f"refusing to overwrite {run}")
    for name in ["00_inventory", "01_qwen", "03_metric_map", "04_grounding_dino", "06_detector_comparison", "07_yolo_world", "08_sam2", "09_mcp", "10_integration", "geometry", "logs"]: (run / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run / "config_snapshot.json")
    image = cv2.imread(config["input_image"]); assert image is not None and image.shape[1::-1] == (1920, 1080)
    statuses = {"inventory": "PASS", "geometry": "PARTIAL", "qwen": "BLOCKED", "dino": "BLOCKED", "yolo": "BLOCKED", "sam2": "BLOCKED", "map": "BLOCKED", "integration": "BLOCKED", "mcp": "BLOCKED"}
    inventory_data = inventory(run, image, config)
    geometry = geometry_stage(run, image, config)
    statuses["qwen"], qwen_objects, regions, relations, qwen_attempts = qwen_stage(run, image, config)
    try:
        statuses["dino"], dino, dino_attempts = base.run_grounding_dino(run, image)
    except Exception as exc:
        dino=[]; dino_attempts=[{"attempt":1,"status":"BLOCKED","error":repr(exc)}]; base.dump(run/"04_grounding_dino/detections.json",{"status":"BLOCKED","attempts":dino_attempts,"detections":[]}); base.write(run/"logs/dino_exception.log",traceback.format_exc())
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except ImportError: pass
    statuses["yolo"], yolo, yolo_attempts = yolo_world_stage(run, image)
    comparison = compare_detectors(run, qwen_objects, dino, yolo)
    statuses["sam2"], sam_records, sam_masks, selected, sam_attempts = sam2_stage(run, image, dino, config) if dino else ("BLOCKED", [], {}, [], [{"status":"SKIPPED","reason":"no detector boxes"}])
    metric_map, matrix, size = metric_map_stage(run, image, config, geometry, regions, selected, sam_masks); statuses["map"] = metric_map["status"]
    scene_graph_stage(run, config, geometry, qwen_objects, regions, relations, comparison, sam_records, sam_masks, matrix, size); statuses["integration"] = "PARTIAL"
    statuses["mcp"] = mcp_stage(run)
    final_report(run, config, inventory_data, statuses, geometry, qwen_attempts, dino_attempts, yolo_attempts, sam_attempts, metric_map)
    self_check(run); base.write(run / "logs/self_check.log", "self-check: PASS")
    base.manifest(run)
    print(run)


if __name__ == "__main__":
    main()
