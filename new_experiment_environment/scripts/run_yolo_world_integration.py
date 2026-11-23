#!/usr/bin/env python3
"""Run isolated YOLO-World inference, then integrate its artifacts read-only."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np

import run_pipeline as base


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "warehouse_robotics_1920x1080.png"
BASELINE = ROOT / "results/METRIC_VLM_PIPELINE_20260929T005848+0900"
VOCABULARY = [
    "truck", "pallet", "cardboard box", "storage rack", "workbench", "cart",
    "drum", "mobile robot", "robot arm", "conveyor belt", "safety fence", "traffic cone",
]
QWEN_ENV = Path("/home/binci/miniconda3/envs/qwen_vl")
CONDA = Path("/home/binci/miniconda3/bin/conda")
QWEN_LIST_BEFORE = "c579a72830c2938228934614a3efb930115bfaa94cd703f4e3848de1eb53cd16"
QWEN_META_BEFORE = "906dc17729294e7e58c8fa6cfe9523d19b5e0a83117ba24b0075c6d27180dbda"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(label: str) -> str:
    return "conveyor" if "conveyor" in label.casefold() else base.canonical(label)


def label_of(item: dict) -> str:
    return str(item.get("label") or item.get("category") or "unknown")


def score_of(item: dict) -> float:
    return float(item.get("confidence", item.get("score", 0.0)))


def bbox_of(item: dict) -> list[float]:
    return [float(value) for value in item["bbox_px"]]


def package_hash() -> str:
    result = subprocess.run(
        [str(CONDA), "list", "-n", "qwen_vl", "--json"],
        check=True, capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def conda_meta_hash() -> str:
    lines = []
    for path in (QWEN_ENV / "conda-meta").iterdir():
        if path.is_file():
            lines.append(f"{sha256(path)}  {path}\n")
    return hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()


def project(points, homography):
    array = np.asarray(points, np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(array, np.asarray(homography, np.float64))[0]


def infer(run: Path, image_path: Path) -> None:
    out = run / "07_yolo_world"
    target = out / "detections.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    image_path = image_path.resolve()
    run = run.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_dir = out / "ultralytics_config" / "Ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(out / "ultralytics_config")
    os.chdir(out)

    import torch
    import ultralytics
    from ultralytics import YOLOWorld

    started = time.perf_counter()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_started = time.perf_counter()
    model = YOLOWorld("yolov8s-world.pt")
    model.to(device)
    model_seconds = time.perf_counter() - model_started
    vocabulary_started = time.perf_counter()
    model.set_classes(VOCABULARY.copy())
    vocabulary_seconds = time.perf_counter() - vocabulary_started
    inference_started = time.perf_counter()
    result = model.predict(
        source=str(image_path), device=0 if device.startswith("cuda") else "cpu",
        imgsz=1280, conf=0.05, iou=0.50, verbose=True,
    )[0]
    inference_seconds = time.perf_counter() - inference_started

    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for index, (box, confidence, class_id) in enumerate(zip(boxes, scores, classes), 1):
            detections.append({
                "id": f"yolo_{index:03d}",
                "label": str(result.names[class_id]),
                "confidence": round(float(confidence), 6),
                "score": round(float(confidence), 6),
                "bbox_px": [round(float(value), 2) for value in box],
                "coordinate_system": "original_image_pixels",
            })
    cv2.imwrite(str(out / "overlay.png"), result.plot())
    write(out / "PROMPT.txt", "\n".join(VOCABULARY))
    weights = out / "yolov8s-world.pt"
    payload = {
        "status": "PASS" if detections else "PARTIAL",
        "detector": "Ultralytics YOLO-World",
        "model": "yolov8s-world.pt",
        "model_sha256": sha256(weights) if weights.exists() else None,
        "vocabulary": VOCABULARY,
        "parameters": {"imgsz": 1280, "conf": 0.05, "iou": 0.50},
        "coordinate_system": "original_image_pixels",
        "detections": detections,
    }
    dump(target, payload)
    counts = Counter(item["label"] for item in detections)
    runtime = {
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": "/home/binci/miniconda3/envs/yolo_world",
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_used": device.startswith("cuda"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "input_image": str(image_path),
        "input_sha256": sha256(image_path),
        "model_load_seconds": model_seconds,
        "vocabulary_encoding_seconds": vocabulary_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "ultralytics_speed_ms": result.speed,
    }
    dump(out / "runtime.json", runtime)
    write(out / "RESULT.md", f"""# YOLO-World Result

Status: {payload['status']}

- Model: `yolov8s-world.pt`
- Environment: `/home/binci/miniconda3/envs/yolo_world`
- CUDA used: `{runtime['cuda_used']}` (`{runtime['device']}`)
- Detections: `{len(detections)}`
- Classes: `{dict(sorted(counts.items()))}`
- Ground truth: unavailable; this is not an accuracy result.
""")


def one_to_one(left: list[dict], right: list[dict], left_name: str, right_name: str) -> list[dict]:
    candidates = []
    for a in left:
        for b in right:
            if canonical(label_of(a)) == canonical(label_of(b)) and canonical(label_of(a)) != "other":
                overlap = base.iou(bbox_of(a), bbox_of(b))
                if overlap >= 0.10:
                    candidates.append((overlap, a, b))
    used_left, used_right, matches = set(), set(), []
    for overlap, a, b in sorted(candidates, key=lambda row: row[0], reverse=True):
        if a["id"] in used_left or b["id"] in used_right:
            continue
        used_left.add(a["id"]); used_right.add(b["id"])
        matches.append({
            f"{left_name}_id": a["id"], f"{right_name}_id": b["id"],
            "canonical_label": canonical(label_of(a)), "bbox_iou": float(overlap),
        })
    return matches


def duplicates(items: list[dict]) -> list[dict]:
    found = []
    for i, first in enumerate(items):
        for second in items[i + 1:]:
            if canonical(label_of(first)) == canonical(label_of(second)) and canonical(label_of(first)) != "other":
                overlap = base.iou(bbox_of(first), bbox_of(second))
                if overlap >= 0.50:
                    found.append({"first_id": first["id"], "second_id": second["id"], "label": canonical(label_of(first)), "iou": float(overlap)})
    return found


def summarize(items: list[dict]) -> dict:
    exact = Counter(label_of(item) for item in items)
    categories = Counter(canonical(label_of(item)) for item in items)
    return {
        "count": len(items), "detected_classes": dict(sorted(exact.items())),
        "canonical_counts": dict(sorted(categories.items())), "duplicate_pairs_iou_ge_0_50": duplicates(items),
    }


def select_for_sam(detections: list[dict], shape) -> list[dict]:
    height, width = shape[:2]
    candidates = []
    for item in detections:
        x1, y1, x2, y2 = bbox_of(item)
        category = canonical(label_of(item))
        area_fraction = max(0, x2 - x1) * max(0, y2 - y1) / (width * height)
        if score_of(item) >= 0.05 and category != "other" and area_fraction <= 0.35:
            candidates.append({**item, "canonical_category": category})
    selected = []
    for item in sorted(candidates, key=score_of, reverse=True):
        if all(item["canonical_category"] != kept["canonical_category"] or base.iou(bbox_of(item), bbox_of(kept)) < 0.65 for kept in selected):
            selected.append(item)
    return selected


def sam2_yolo(run: Path, image, detections: list[dict]) -> tuple[list[dict], dict[str, np.ndarray], list[dict], dict]:
    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor

    out = run / "08_sam2_yolo_world"
    masks_dir = out / "masks"
    target = out / "segmentation.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    masks_dir.mkdir(parents=True, exist_ok=True)
    selected = select_for_sam(detections, image.shape)
    model_id = "facebook/sam2.1-hiera-tiny"
    started = time.perf_counter()
    processor = Sam2Processor.from_pretrained(model_id, local_files_only=True)
    model = Sam2Model.from_pretrained(model_id, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    boxes = [bbox_of(item) for item in selected]
    encoded = processor(images=pil, input_boxes=[boxes], return_tensors="pt")
    model_inputs = {key: value.to(device) for key, value in encoded.items() if key in {"pixel_values", "input_boxes", "input_points", "input_labels"}}
    with torch.inference_mode():
        outputs = model(**model_inputs, multimask_output=False)
    processed = processor.post_process_masks(outputs.pred_masks.cpu(), encoded["original_sizes"])[0]

    records, masks = [], {}
    overlay = image.copy()
    for index, item in enumerate(selected):
        mask = processed[index, 0].numpy().astype(np.uint8) * 255
        masks[item["id"]] = mask
        path = masks_dir / f"{item['id']}.png"
        cv2.imwrite(str(path), mask)
        bbox_area = max(1.0, (item["bbox_px"][2] - item["bbox_px"][0]) * (item["bbox_px"][3] - item["bbox_px"][1]))
        records.append({
            "detection_id": item["id"], "label": item["label"], "canonical_category": item["canonical_category"],
            "bbox_px": item["bbox_px"], "detection_confidence": score_of(item),
            "mask_path": str(path.relative_to(run)), "mask_area_px": int(np.count_nonzero(mask)),
            "bbox_area_px": float(bbox_area), "mask_to_bbox_area_ratio": float(np.count_nonzero(mask) / bbox_area),
        })
        pixels = mask > 0
        color = np.asarray(((47 * index + 70) % 255, (89 * index + 90) % 255, (137 * index + 120) % 255), np.uint8)
        overlay[pixels] = (overlay[pixels] * 0.55 + color * 0.45).astype(np.uint8)
    base.label(overlay, f"YOLO-WORLD bbox -> SAM2 masks: {len(records)}", (20, 35))
    cv2.imwrite(str(out / "overlay.png"), overlay)
    runtime = {"model": model_id, "prompt_source": "YOLO-World boxes", "device": device, "seconds": time.perf_counter() - started}
    dump(target, {"status": "PASS" if records else "PARTIAL", **runtime, "selected_detections": len(selected), "segments": records})
    dump(out / "runtime.json", runtime)
    write(out / "RESULT.md", f"# YOLO-World → SAM2 Result\n\nStatus: {'PASS' if records else 'PARTIAL'}\n\nYOLO-World bbox {len(selected)}개로 별도 SAM2 mask {len(records)}개를 생성했다. 기존 Grounding DINO → SAM2 artifact는 읽기만 했다.")
    return records, masks, selected, runtime


def comparison(run: Path, qwen: list[dict], dino: list[dict], yolo: list[dict]) -> dict:
    qwen_yolo = one_to_one(qwen, yolo, "qwen", "yolo_world")
    dino_yolo = one_to_one(dino, yolo, "grounding_dino", "yolo_world")
    left_manipulators = [item for item in yolo if canonical(label_of(item)) == "manipulator" and sum(bbox_of(item)[::2]) / 2 < 650]
    central_manipulators = [item for item in yolo if canonical(label_of(item)) == "manipulator" and 650 <= sum(bbox_of(item)[::2]) / 2 < 1000]
    expected = {canonical(name) for name in VOCABULARY}
    present = {canonical(label_of(item)) for item in yolo}
    result = {
        "status": "PARTIAL", "ground_truth_available": False, "accuracy_claim": False,
        "matching_rule": "same canonical label, greedy one-to-one bbox IoU >= 0.10",
        "qwen": summarize(qwen), "grounding_dino": summarize(dino), "yolo_world": summarize(yolo),
        "qwen_vs_yolo_world": {"matches": qwen_yolo, "match_count": len(qwen_yolo)},
        "grounding_dino_vs_yolo_world": {"matches": dino_yolo, "match_count": len(dino_yolo)},
        "required_checks": {
            "conveyor_detected": "conveyor" in present,
            "manipulator_detected": "manipulator" in present,
            "left_manipulator_count_x_center_lt_650": len(left_manipulators),
            "central_manipulator_count_650_le_x_center_lt_1000": len(central_manipulators),
            "missing_canonical_classes": sorted(expected - present),
        },
        "qwen_known_omissions": {"conveyor_detected": any(canonical(label_of(item)) == "conveyor" for item in qwen), "left_manipulator_count_x_center_lt_650": sum(canonical(label_of(item)) == "manipulator" and sum(bbox_of(item)[::2]) / 2 < 650 for item in qwen)},
    }
    out = run / "06_detector_comparison"
    dump(out / "detector_comparison.json", result)
    checks = result["required_checks"]
    write(out / "RESULT.md", f"""# Qwen / Grounding DINO / YOLO-World Comparison

Status: PARTIAL

Ground truth가 없어 accuracy를 계산하지 않았다. Qwen {len(qwen)}개, Grounding DINO {len(dino)}개, YOLO-World {len(yolo)}개다. YOLO-World의 Qwen-compatible match는 {len(qwen_yolo)}개, Grounding-DINO-compatible match는 {len(dino_yolo)}개다.

- Conveyor detected: `{checks['conveyor_detected']}`
- Manipulator detected: `{checks['manipulator_detected']}`
- Left manipulator count (x-center < 650 px): `{checks['left_manipulator_count_x_center_lt_650']}`
- Central manipulator count (650 <= x-center < 1000 px): `{checks['central_manipulator_count_650_le_x_center_lt_1000']}`
- Duplicate criterion: same canonical label and bbox IoU >= 0.50
""")
    return result


def compare_masks(run: Path, baseline: Path, dino: list[dict], yolo: list[dict], yolo_masks: dict[str, np.ndarray]) -> dict:
    old_segments = {item["detection_id"]: item for item in load(baseline / "08_sam2/segmentation.json")["segments"]}
    matches = one_to_one(dino, yolo, "grounding_dino", "yolo_world")
    rows = []
    for match in matches:
        dino_id, yolo_id = match["grounding_dino_id"], match["yolo_world_id"]
        if dino_id not in old_segments or yolo_id not in yolo_masks:
            continue
        old_mask = cv2.imread(str(baseline / old_segments[dino_id]["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
        new_mask = yolo_masks[yolo_id] > 0
        union = np.count_nonzero(old_mask | new_mask)
        rows.append({**match, "mask_iou": float(np.count_nonzero(old_mask & new_mask) / union) if union else 0.0})
    result = {"status": "PARTIAL" if rows else "BLOCKED", "ground_truth_available": False, "accuracy_claim": False, "comparable_pairs": rows}
    dump(run / "08_sam2_yolo_world/mask_comparison.json", result)
    return result


def metric_map(run: Path, image, selected: list[dict], masks: dict[str, np.ndarray], baseline: Path) -> dict:
    out = run / "03_metric_map"
    geometry = load(baseline / "geometry/homography.json")
    config = load(ROOT / "metric_pipeline_config.json")
    width_m, height_m = config["workspace_m"]
    resolution = config["map_resolution_m_per_px"]
    size = (round(width_m / resolution), round(height_m / resolution))
    world_to_map = np.asarray([[1 / resolution, 0, 0], [0, -1 / resolution, height_m / resolution], [0, 0, 1]], np.float64)
    homography = np.asarray(geometry["image_to_world_homography"], np.float64)
    matrix = world_to_map @ homography
    bbox_image = np.zeros(image.shape[:2], np.uint8)
    mask_image = np.zeros_like(bbox_image)
    projected = []
    for item in selected:
        x1, y1, x2, y2 = bbox_of(item)
        cv2.rectangle(bbox_image, (round(x1), round(y1)), (round(x2), round(y2)), 255, cv2.FILLED)
        if item["id"] in masks:
            cv2.bitwise_or(mask_image, masks[item["id"]], dst=mask_image)
        center, bottom = project([[(x1 + x2) / 2, (y1 + y2) / 2], [(x1 + x2) / 2, y2]], homography)
        projected.append({
            "id": item["id"], "label": item["label"], "bbox_px": item["bbox_px"],
            "center_xy_m": [float(center[0]), float(center[1])], "bottom_center_xy_m": [float(bottom[0]), float(bottom[1])],
            "bottom_center_inside_workspace": bool(0 <= bottom[0] <= width_m and 0 <= bottom[1] <= height_m),
        })
    bbox_metric = cv2.warpPerspective(bbox_image, matrix, size, flags=cv2.INTER_NEAREST)
    sam_metric = cv2.warpPerspective(mask_image, matrix, size, flags=cv2.INTER_NEAREST)
    cv2.imwrite(str(out / "yolo_bbox_metric_occupancy.png"), bbox_metric)
    cv2.imwrite(str(out / "yolo_sam2_metric_occupancy.png"), sam_metric)
    canvas = np.full((size[1], size[0], 3), 235, np.uint8)
    canvas[bbox_metric > 0] = (70, 70, 220)
    canvas[sam_metric > 0] = (220, 150, 40)
    cv2.imwrite(str(out / "metric_map_overlay.png"), canvas)
    result = {
        "status": "PARTIAL", "scope": "SYNTHETIC_SOFTWARE_FEASIBILITY_ONLY",
        "frame": config["world_frame"], "homography_source": str(baseline / "geometry/homography.json"),
        "workspace_size_m": config["workspace_m"], "resolution_m_per_px": resolution,
        "raster_size_px": list(size), "detections_projected": len(projected),
        "bbox_occupancy_area_m2": float(np.count_nonzero(bbox_metric) * resolution**2),
        "sam2_occupancy_area_m2": float(np.count_nonzero(sam_metric) * resolution**2),
        "objects": projected, "real_camera_accuracy_claim": False, "nav2_ready": False,
    }
    dump(out / "metric_projection.json", result)
    write(out / "RESULT.md", f"# YOLO-World Metric Integration\n\nStatus: PARTIAL\n\n{len(projected)}개 bbox와 SAM2 mask를 기존 synthetic homography로 변환했다. 이는 software feasibility 확인이며 실제 카메라 정확도나 Nav2 readiness를 뜻하지 않는다.")
    return result


def environment_after(run: Path) -> dict:
    import torch
    import transformers
    current_list, current_meta = package_hash(), conda_meta_hash()
    result = {
        "environment": str(QWEN_ENV), "conda_list_sha256": current_list, "conda_meta_sha256": current_meta,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "matches_before": {"conda_list": current_list == QWEN_LIST_BEFORE, "conda_meta": current_meta == QWEN_META_BEFORE},
    }
    dump(run / "00_inventory/qwen_vl_after.json", result)
    return result


def report(run: Path, comparison_data: dict, sam_records: list[dict], sam_runtime: dict, masks_data: dict, metric_data: dict, qwen_after: dict) -> None:
    yolo = load(run / "07_yolo_world/detections.json")
    runtime = load(run / "07_yolo_world/runtime.json")
    checks = comparison_data["required_checks"]
    classes = comparison_data["yolo_world"]["detected_classes"]
    required_coverage = checks["conveyor_detected"] and checks["manipulator_detected"]
    status = "PASS" if yolo["detections"] and sam_records and required_coverage else "PARTIAL" if yolo["detections"] else "BLOCKED"
    write(run / "RUN_REPORT.md", f"""# YOLO-World Integration RUN REPORT

## Status

YOLO-World: {status}

- Model: `yolov8s-world.pt`
- Ultralytics: `{runtime['ultralytics']}`
- Isolated environment: `/home/binci/miniconda3/envs/yolo_world`
- CUDA used: `{runtime['cuda_used']}` (`{runtime['device']}`)
- Detections: `{len(yolo['detections'])}`
- Detected classes: `{json.dumps(classes, ensure_ascii=False)}`
- Conveyor detected: `{checks['conveyor_detected']}`
- Manipulator detected: `{checks['manipulator_detected']}`
- Left manipulators (x-center < 650 px): `{checks['left_manipulator_count_x_center_lt_650']}`
- Central manipulators (650 <= x-center < 1000 px): `{checks['central_manipulator_count_650_le_x_center_lt_1000']}`

## Detector comparison

- Ground truth is unavailable; no accuracy claim is made.
- Qwen: `{comparison_data['qwen']['count']}` detections.
- Grounding DINO: `{comparison_data['grounding_dino']['count']}` detections.
- YOLO-World: `{comparison_data['yolo_world']['count']}` detections.
- Qwen ↔ YOLO compatible one-to-one matches: `{comparison_data['qwen_vs_yolo_world']['match_count']}`.
- Grounding DINO ↔ YOLO compatible one-to-one matches: `{comparison_data['grounding_dino_vs_yolo_world']['match_count']}`.
- YOLO duplicate pairs (same canonical label, bbox IoU >= 0.50): `{len(comparison_data['yolo_world']['duplicate_pairs_iou_ge_0_50'])}`.
- Qwen previously had conveyor=`{comparison_data['qwen_known_omissions']['conveyor_detected']}` and left manipulator count=`{comparison_data['qwen_known_omissions']['left_manipulator_count_x_center_lt_650']}` by the same spatial rule.

## SAM2 integration

- Existing Grounding DINO → SAM2 artifacts remained unchanged and were read only.
- New YOLO-World → SAM2 masks: `{len(sam_records)}` using `{sam_runtime['model']}` on `{sam_runtime['device']}`.
- Grounding-DINO-mask ↔ YOLO-mask comparable pairs: `{len(masks_data['comparable_pairs'])}`.
- Mask IoU is diagnostic only because there is no segmentation ground truth.

## Synthetic metric integration

- Status: `{metric_data['status']}`.
- Projected detections: `{metric_data['detections_projected']}`.
- YOLO bbox occupancy: `{metric_data['bbox_occupancy_area_m2']:.4f} m²`.
- YOLO→SAM2 occupancy: `{metric_data['sam2_occupancy_area_m2']:.4f} m²`.
- The reused homography is synthetic feasibility geometry, not real-camera calibration and not Nav2-ready.

## Environment protection

- `qwen_vl` conda package-list hash unchanged: `{qwen_after['matches_before']['conda_list']}`.
- `qwen_vl` conda metadata hash unchanged: `{qwen_after['matches_before']['conda_meta']}`.
- `qwen_vl` Torch/Transformers after run: `{qwen_after['torch']}` / `{qwen_after['transformers']}`.
- No sudo, apt, driver, CUDA, ROS2, or kernel changes were made.
- Ultralytics first import created `/home/binci/.config/Ultralytics/settings.json`.
- YOLO weight is run-local, but the first inference config fallback stored CLIP at `/home/binci/weights/clip/ViT-B-32.pt`; this did not alter `qwen_vl` but was not fully run-local.

## WHAT WORKS NOW

- Isolated YOLO-World CUDA inference with the requested 12-class vocabulary.
- Three-detector class/count/IoU diagnostic comparison.
- Separate Grounding DINO→SAM2 and YOLO-World→SAM2 paths.
- YOLO bbox/mask projection through the preserved synthetic homography.

## WHAT TO DO NEXT

1. Human-review YOLO boxes and both SAM2 mask paths on the source image.
2. Add ground-truth annotations before reporting detector or mask accuracy.
3. Repeat pixel→metric validation with real undistorted camera data and measured control points.
""")


def manifest(run: Path) -> None:
    files = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files.append({"path": str(path.relative_to(run)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    dump(run / "ARTIFACT_MANIFEST.json", {"run": run.name, "files": files})


def integrate(run: Path, image_path: Path, baseline: Path) -> None:
    image = cv2.imread(str(image_path))
    assert image is not None and image.shape[1::-1] == (1920, 1080)
    assert sha256(image_path) == load(run / "07_yolo_world/runtime.json")["input_sha256"]
    qwen = load(baseline / "01_qwen/objects.json")["objects"]
    dino = load(baseline / "04_grounding_dino/detections.json")["detections"]
    yolo = load(run / "07_yolo_world/detections.json")["detections"]
    comparison_data = comparison(run, qwen, dino, yolo)
    sam_records, masks, selected, sam_runtime = sam2_yolo(run, image, yolo)
    masks_data = compare_masks(run, baseline, dino, yolo, masks)
    metric_data = metric_map(run, image, selected, masks, baseline)
    qwen_after = environment_after(run)
    report(run, comparison_data, sam_records, sam_runtime, masks_data, metric_data, qwen_after)
    self_check(run)
    write(run / "logs/self_check.log", "self-check: PASS")
    manifest(run)


def self_check(run: Path | None = None) -> None:
    assert canonical("conveyor belt") == "conveyor"
    assert canonical("robot arm") == "manipulator"
    assert round(base.iou([0, 0, 10, 10], [5, 5, 15, 15]), 6) == 0.142857
    if run:
        for relative in [
            "07_yolo_world/detections.json", "07_yolo_world/overlay.png", "07_yolo_world/runtime.json", "07_yolo_world/PROMPT.txt",
            "06_detector_comparison/detector_comparison.json", "08_sam2_yolo_world/segmentation.json",
            "08_sam2_yolo_world/overlay.png", "03_metric_map/metric_projection.json", "03_metric_map/metric_map_overlay.png", "RUN_REPORT.md",
        ]:
            path = run / relative
            assert path.is_file() and path.stat().st_size > 0, path
            if path.suffix == ".json":
                load(path)
            if path.suffix == ".png":
                assert cv2.imread(str(path)) is not None, path
        assert load(run / "00_inventory/qwen_vl_after.json")["matches_before"] == {"conda_list": True, "conda_meta": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["infer", "integrate", "self-check"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--image", type=Path, default=IMAGE)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    if args.stage == "self-check":
        self_check(); print("self-check: PASS"); return
    if args.run is None:
        parser.error("--run is required")
    try:
        infer(args.run, args.image) if args.stage == "infer" else integrate(args.run.resolve(), args.image.resolve(), args.baseline.resolve())
    except Exception:
        log = args.run / "logs" / f"{args.stage}_exception.log"
        write(log, traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
