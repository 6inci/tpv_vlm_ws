#!/usr/bin/env python3
"""Run semantic-grounding evaluation and controlled detector diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "warehouse_robotics_1920x1080.png"
BASELINE = ROOT / "results/METRIC_VLM_PIPELINE_20260929T005848+0900"
YOLO_BASELINE = ROOT / "results/YOLO_WORLD_INTEGRATION_20260929T012518+0900"
QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
YOLO_MODEL = YOLO_BASELINE / "07_yolo_world/yolov8s-world.pt"
REFERENCE_CODE = Path("/home/binci/vlm_ws/wanna_find_detailed_experimental_x,Y/src")
WORKSPACE_M = [3.15, 2.10]

TRIALS = [
    {
        "trial_id": "canonical",
        "labels": ["manipulator", "conveyor", "mobile robot", "pallet"],
    },
    {
        "trial_id": "common_synonyms",
        "labels": ["robot arm", "robotic arm", "conveyor belt", "industrial conveyor", "AGV", "robot vehicle", "wooden pallet", "industrial pallet"],
    },
    {
        "trial_id": "extended_synonyms",
        "labels": ["industrial robot", "industrial robot arm", "robotic manipulator", "belt conveyor", "AMR", "mobile robot platform", "cargo pallet"],
    },
]

FAMILY_TERMS = {
    "manipulator": ["manipulator", "robot arm", "robotic arm", "industrial robot", "industrial robot arm", "robotic manipulator"],
    "conveyor": ["conveyor", "conveyor belt", "belt conveyor", "industrial conveyor"],
    "mobile_robot": ["mobile robot", "agv", "amr", "robot vehicle", "mobile robot platform"],
    "pallet": ["pallet", "wooden pallet", "cargo pallet", "industrial pallet"],
}

TEST_CASES = [
    {"task_id": "A01", "task_type": "TYPE_A", "instruction": "트럭 앞쪽의 이동 가능한 공간으로 이동해.", "required_relation": "in_front_of"},
    {"task_id": "A02", "task_type": "TYPE_A", "instruction": "카트 오른쪽의 이동 가능한 공간으로 이동해.", "required_relation": "right_of"},
    {"task_id": "B01", "task_type": "TYPE_B", "instruction": "트럭과 가장 가까운 팔레트 사이의 이동 가능한 공간으로 이동해.", "required_relation": "between"},
    {"task_id": "B02", "task_type": "TYPE_B", "instruction": "카트 옆 드럼통 근처의 이동 가능한 공간으로 이동해.", "required_relation": "near|adjacent_to"},
    {"task_id": "B03", "task_type": "TYPE_B", "instruction": "두 선반 사이의 통로를 찾아 이동해.", "required_relation": "between"},
    {"task_id": "B04", "task_type": "TYPE_B", "instruction": "트럭의 왼쪽이면서 가까운 팔레트 앞쪽인 빈 공간으로 이동해.", "required_relation": "left_of|in_front_of"},
    {"task_id": "B05", "task_type": "TYPE_B", "instruction": "워크벤치 뒤쪽의 이동 가능한 공간으로 이동해.", "required_relation": "behind"},
    {"task_id": "C01", "task_type": "TYPE_C", "instruction": "트럭 옆 하역 작업이 가능한 공간으로 이동해.", "candidate_region": "Loading Area"},
    {"task_id": "C02", "task_type": "TYPE_C", "instruction": "부품 보관 구역의 이동 가능한 공간으로 이동해.", "candidate_region": "Storage Area"},
    {"task_id": "C03", "task_type": "TYPE_C", "instruction": "로봇 작업 구역 앞으로 이동해.", "candidate_region": "Work Area"},
    {"task_id": "C04", "task_type": "TYPE_C", "instruction": "자재 보관 구역의 이동 가능한 공간으로 이동해.", "candidate_region": "Material Area"},
    {"task_id": "D01", "task_type": "TYPE_D", "instruction": "트럭에서 가장 가까운 팔레트 옆의 이동 가능한 공간으로 이동해.", "required_relation": "nearest|adjacent_to"},
    {"task_id": "D02", "task_type": "TYPE_D", "instruction": "컨베이어 옆 매니퓰레이터 앞의 이동 가능한 공간으로 이동해.", "required_relation": "adjacent_to|in_front_of"},
    {"task_id": "E01", "task_type": "TYPE_E", "instruction": "지게차 충전 구역으로 이동해.", "insufficient_evidence_reason": "forklift/charging evidence not established"},
    {"task_id": "E02", "task_type": "TYPE_E", "instruction": "아무 작업 구역으로 이동해.", "insufficient_evidence_reason": "multiple functional work candidates"},
]


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


def tree_hash(path: Path) -> str:
    lines = [f"{sha256(item)}  {item.name}\n" for item in path.iterdir() if item.is_file()]
    return hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()


def first_json(text: str):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
                return value
            except json.JSONDecodeError:
                pass
    raise ValueError("no complete JSON object")


def complete_array_items(text: str, key: str) -> list[dict]:
    marker = f'"{key}"'
    position = text.find(marker)
    if position < 0:
        return []
    position = text.find("[", position + len(marker)) + 1
    decoder, items = json.JSONDecoder(), []
    while position > 0:
        while position < len(text) and text[position] in " \r\n\t,":
            position += 1
        if position >= len(text) or text[position] != "{":
            break
        try:
            item, consumed = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            break
        items.append(item)
        position += consumed
    return items


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def family(label: str) -> str | None:
    normalized = " ".join(str(label).casefold().replace("_", " ").replace("-", " ").split())
    for name, terms in FAMILY_TERMS.items():
        if any(term in normalized for term in terms):
            return name
    return None


def label_image(image, text, xy, color=(255, 255, 255), scale=.48):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def project_metric(pixel: list[float], homography: list[list[float]]) -> dict:
    projected = cv2.perspectiveTransform(np.asarray(pixel, np.float64).reshape(1, 1, 2), np.asarray(homography, np.float64))[0, 0]
    x, y = map(float, projected)
    return {
        "x": x,
        "y": y,
        "unit": "m",
        "frame": "synthetic_workspace_bottom_left_xy_m",
        "synthetic_only": True,
        "outside_workspace": not (0 <= x <= WORKSPACE_M[0] and 0 <= y <= WORKSPACE_M[1]),
    }


def test_case_payload(cases=TEST_CASES) -> str:
    return json.dumps([
        {key: item[key] for key in ("task_id", "task_type", "instruction")}
        for item in cases
    ], ensure_ascii=False, separators=(",", ":"))


def prepare(run: Path) -> None:
    if run.exists():
        raise FileExistsError(f"refusing to overwrite {run}")
    directories = [
        "00_inventory", "01_test_cases", "02_structured_context",
        "03_end_to_end/qwen", "03_end_to_end/grounding_dino", "03_end_to_end/yolo_world",
        "03_end_to_end/diagnostic_augmented", "04_semantic_regions/overlays",
        "05_target_grounding/overlays", "06_error_analysis", "07_human_validation",
        "08_future_experiments", "detector_diagnostics/overlays", "logs",
    ]
    for name in directories:
        (run / name).mkdir(parents=True)
    required = [
        IMAGE,
        BASELINE / "01_qwen/objects.json",
        BASELINE / "01_qwen/semantic_regions.json",
        BASELINE / "04_grounding_dino/detections.json",
        BASELINE / "geometry/homography.json",
        BASELINE / "10_integration/metric_scene_graph.json",
        YOLO_BASELINE / "07_yolo_world/detections.json",
        YOLO_MODEL,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    qwen_meta = Path("/home/binci/miniconda3/envs/qwen_vl/conda-meta")
    yolo_meta = Path("/home/binci/miniconda3/envs/yolo_world/conda-meta")
    inventory = {
        "created_at": datetime.now().astimezone().isoformat(),
        "input_image": str(IMAGE),
        "input_sha256": sha256(IMAGE),
        "image_size_px": [1920, 1080],
        "existing_artifacts_read_only": True,
        "sources": {"metric_baseline": str(BASELINE), "yolo_baseline": str(YOLO_BASELINE)},
        "models": {"qwen": QWEN_MODEL, "grounding_dino": DINO_MODEL, "yolo_world": str(YOLO_MODEL)},
        "environment_hash_before": {"qwen_vl_conda_meta": tree_hash(qwen_meta), "yolo_world_conda_meta": tree_hash(yolo_meta)},
        "ground_truth": "NOT_AVAILABLE",
    }
    dump(run / "00_inventory/inventory.json", inventory)
    write(run / "00_inventory/ENVIRONMENT_INVENTORY.md", f"""# Environment Inventory

Status: PASS

- Input: `{IMAGE}` (1920×1080, SHA-256 `{inventory['input_sha256']}`)
- Qwen: `{QWEN_MODEL}` in existing `qwen_vl`
- Grounding DINO: `{DINO_MODEL}` in existing `qwen_vl`
- YOLO-World: existing `{YOLO_MODEL.name}` in existing `yolo_world`
- Existing runs are read-only references; no environment/package change is authorized.
- Human-reviewed semantic ground truth: NOT_AVAILABLE
""")
    dump(run / "01_test_cases/test_cases.json", {
        "status": "PASS", "ground_truth_status": "NOT_AVAILABLE",
        "coordinate_system": "original_image_pixels", "test_cases": TEST_CASES,
    })
    dump(run / "run_config.json", {
        "input_image": str(IMAGE), "qwen_model": QWEN_MODEL, "dino_model": DINO_MODEL,
        "yolo_model": str(YOLO_MODEL), "vocabulary_trials": TRIALS,
        "diagnostic_thresholds": {"yolo_confidence": .02, "yolo_iou": .50, "dino_box": .15, "dino_text": .15, "reference_iou": .30, "wrong_label_iou": .50},
        "synthetic_geometry": str(BASELINE / "geometry/homography.json"),
    })
    write(run / "RUN_PLAN.md", """# Run Plan

1. Reuse prior Qwen/DINO/YOLO/SAM2/geometry artifacts read-only.
2. Run three controlled synonym trials in YOLO-World and Grounding DINO.
3. Run Qwen semantic grounding with structured, Qwen, DINO, YOLO, and diagnostic-augmented contexts.
4. Project target pixels through the existing synthetic-only homography.
5. Produce overlays, failure separation, human-validation material, future plans, and the final report.
""")


def detector_context(items: list[dict]) -> list[dict]:
    compact = []
    for item in items:
        compact.append({
            "id": item.get("id"),
            "label": item.get("label") or item.get("category"),
            "bbox_px": item.get("bbox_px"),
            "confidence": item.get("confidence", item.get("score", item.get("certainty"))),
        })
    return compact


def reasoning_prompt(source: str, context, cases=TEST_CASES) -> str:
    return f"""Evaluate natural-language semantic grounding in this 1920x1080 synthetic workspace image.
Input path: {source}.
Object detection is evidence, not the final answer. Distinguish object bbox, functional semantic region, obstacle/no-go, and final robot goal.
Use combinations of objects and explicit spatial relations for Loading/Storage/Work/Material areas. Use ONLY object identities present in CONTEXT as supporting_objects. Do not invent missing evidence.
For insufficient or ambiguous evidence return ABSTAIN or AMBIGUOUS and target_pixel null.
For ANSWERED items, target_pixel must be visible usable floor in original pixel coordinates [x,y], not inside an object bbox. polygon_px is an approximate semantic-region boundary, not an object box.
Do not output chain-of-thought. Output only compact JSON with this schema:
{{"predictions":[{{"task_id":"A01","status":"ANSWERED|ABSTAIN|AMBIGUOUS|FAILED","predicted_semantic_region":"label or null","supporting_objects":["context IDs"],"supporting_relations":[{{"subject_id":"id","predicate":"left_of|right_of|in_front_of|behind|near|between|adjacent_to","object_id":"id"}}],"region_geometry":{{"polygon_px":[[x,y],[x,y],[x,y]]}},"target_pixel":[x,y],"support_level":"HIGH|MEDIUM|LOW","evidence_summary":"brief observable evidence"}}]}}
CONTEXT={json.dumps(context, ensure_ascii=False, separators=(",", ":"))}
TASKS={test_case_payload(cases)}
"""


def validate_predictions(raw: str, source: str, cases: list[dict], homography) -> tuple[list[dict], str]:
    try:
        parsed = first_json(raw)
        predictions, parse_mode = parsed.get("predictions", []), "complete_json"
    except ValueError:
        predictions, parse_mode = complete_array_items(raw, "predictions"), "complete_array_item_recovery"
    by_id, allowed_status = {}, {"ANSWERED", "ABSTAIN", "AMBIGUOUS", "FAILED"}
    valid_ids = {item["task_id"] for item in cases}
    for item in predictions:
        task_id = str(item.get("task_id", ""))
        if task_id not in valid_ids or task_id in by_id:
            continue
        status = item.get("status") if item.get("status") in allowed_status else "FAILED"
        target = item.get("target_pixel")
        errors = []
        if not (isinstance(target, list) and len(target) == 2 and all(isinstance(value, (int, float)) for value in target) and 0 <= target[0] < 1920 and 0 <= target[1] < 1080):
            if target is not None:
                errors.append("invalid_target_pixel")
            target = None
        polygon = item.get("region_geometry", {}).get("polygon_px")
        if not (isinstance(polygon, list) and len(polygon) >= 3 and all(isinstance(point, list) and len(point) == 2 for point in polygon)):
            polygon = None
        by_id[task_id] = {
            "task_id": task_id,
            "instruction": next(case["instruction"] for case in cases if case["task_id"] == task_id),
            "task_type": next(case["task_type"] for case in cases if case["task_id"] == task_id),
            "status": status,
            "predicted_semantic_region": item.get("predicted_semantic_region"),
            "supporting_objects": item.get("supporting_objects", []) if isinstance(item.get("supporting_objects", []), list) else [],
            "supporting_relations": item.get("supporting_relations", []) if isinstance(item.get("supporting_relations", []), list) else [],
            "region_geometry": {"polygon_px": polygon, "coordinate_system": "original_image_pixels"},
            "target_pixel": target,
            "target_metric_xy": project_metric(target, homography) if target else None,
            "support_level": item.get("support_level"),
            "evidence_summary": item.get("evidence_summary"),
            "source_path": source,
            "validation_errors": errors,
        }
    for case in cases:
        if case["task_id"] not in by_id:
            by_id[case["task_id"]] = {
                "task_id": case["task_id"], "instruction": case["instruction"], "task_type": case["task_type"],
                "status": "FAILED", "predicted_semantic_region": None, "supporting_objects": [], "supporting_relations": [],
                "region_geometry": {"polygon_px": None, "coordinate_system": "original_image_pixels"},
                "target_pixel": None, "target_metric_xy": None, "support_level": None,
                "evidence_summary": "No complete model item was recovered.", "source_path": source,
                "validation_errors": ["missing_prediction"],
            }
    return [by_id[item["task_id"]] for item in cases], parse_mode


def qwen_stage(run: Path) -> None:
    sys.path.insert(0, str(REFERENCE_CODE))
    from vlm_reposition.perception.qwen_grounding import generate_qwen_text, load_local_qwen, prepare_qwen_inputs

    qwen_objects = load(BASELINE / "01_qwen/objects.json")["objects"]
    semantic = load(BASELINE / "01_qwen/semantic_regions.json")
    dino = load(BASELINE / "04_grounding_dino/detections.json")["detections"]
    yolo = load(YOLO_BASELINE / "07_yolo_world/detections.json")["detections"]
    homography = load(BASELINE / "geometry/homography.json")["image_to_world_homography"]
    dino_trials = load(run / "detector_diagnostics/vocabulary_trials_dino.json")["trials"]
    yolo_trials = load(run / "detector_diagnostics/vocabulary_trials_yolo.json")["trials"]
    diagnostic = []
    for detector, trials in (("dino_diag", dino_trials), ("yolo_diag", yolo_trials)):
        for trial in trials:
            for item in trial["detections"]:
                if family(item["label"]) in {"manipulator", "conveyor"}:
                    diagnostic.append({**item, "id": f"{detector}_{trial['trial_id']}_{item['id']}", "detector": detector, "trial_id": trial["trial_id"]})
    end_to_end_cases = [item for item in TEST_CASES if item["task_id"] in {"C01", "C03", "D02", "E01"}]
    contexts = [
        ("STRUCTURED_CONTEXT", run / "02_structured_context", {
            "objects": detector_context(qwen_objects), "semantic_regions": semantic["regions"], "spatial_relations": semantic["relations"],
            "warning": "human-reviewable model-derived context; not an oracle or ground truth",
        }, TEST_CASES, 5000),
        ("QWEN_OBJECT_OUTPUT", run / "03_end_to_end/qwen", {"detections": detector_context(qwen_objects)}, end_to_end_cases, 1600),
        ("GROUNDING_DINO_SAM2", run / "03_end_to_end/grounding_dino", {"detections": detector_context(dino), "sam2_masks_available": 39}, end_to_end_cases, 1600),
        ("YOLO_WORLD_SAM2", run / "03_end_to_end/yolo_world", {"detections": detector_context(yolo), "sam2_masks_available": 38}, end_to_end_cases, 1600),
        ("DIAGNOSTIC_AUGMENTED", run / "03_end_to_end/diagnostic_augmented", {"controlled_trial_detections": detector_context(diagnostic)}, [item for item in TEST_CASES if item["task_id"] in {"C03", "D02"}], 1600),
    ]
    processor, model, load_seconds = load_local_qwen(QWEN_MODEL)
    for source, directory, context, cases, max_tokens in contexts:
        if (directory / "predictions.json").is_file():
            continue
        started = time.perf_counter()
        prompt = reasoning_prompt(source, context, cases)
        write(directory / "PROMPT.txt", prompt)
        inputs = prepare_qwen_inputs(processor, model, IMAGE, prompt)
        raw, inference_seconds = generate_qwen_text(processor, model, inputs, max_tokens)
        write(directory / "MODEL_RAW.txt", raw)
        predictions, parse_mode = validate_predictions(raw, source, cases, homography)
        status = "PASS" if all(item["status"] != "FAILED" for item in predictions) else "PARTIAL" if any(item["status"] != "FAILED" for item in predictions) else "BLOCKED"
        dump(directory / "predictions.json", {
            "status": status, "source_path": source, "model": QWEN_MODEL, "ground_truth_available": False,
            "parse_mode": parse_mode, "coordinate_system": "original_image_pixels",
            "metric_projection_scope": "SYNTHETIC_SOFTWARE_FEASIBILITY_ONLY", "predictions": predictions,
        })
        dump(directory / "runtime.json", {
            "model": QWEN_MODEL, "device": str(model.device), "model_load_seconds_shared": load_seconds,
            "inference_seconds": inference_seconds, "total_stage_seconds": time.perf_counter() - started,
            "max_new_tokens": max_tokens, "python": platform.python_version(),
        })
        write(directory / "RESULT.md", f"# {source} Result\n\nStatus: {status}\n\nRecovered {sum(item['status'] != 'FAILED' for item in predictions)}/{len(predictions)} structured predictions. Ground truth is unavailable; no semantic accuracy is claimed.")
        del inputs
        gc.collect()
    del model, processor
    import torch
    torch.cuda.empty_cache()


def yolo_diagnostic(run: Path) -> None:
    out = run / "detector_diagnostics"
    os.environ["YOLO_CONFIG_DIR"] = str(out / "yolo_config")
    from ultralytics import YOLOWorld
    import torch
    import ultralytics

    image = cv2.imread(str(IMAGE))
    model = YOLOWorld(str(YOLO_MODEL))
    device = 0 if torch.cuda.is_available() else "cpu"
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    trials, started = [], time.perf_counter()
    for trial in TRIALS:
        model.set_classes(trial["labels"].copy())
        result = model.predict(source=str(IMAGE), device=device, imgsz=1280, conf=.02, iou=.50, max_det=100, verbose=False)[0]
        detections = []
        if result.boxes is not None:
            for index, (box, score, class_id) in enumerate(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)), 1):
                detections.append({
                    "id": f"y_{index:03d}", "label": str(result.names[class_id]), "confidence": round(float(score), 6),
                    "bbox_px": [round(float(value), 2) for value in box], "coordinate_system": "original_image_pixels",
                })
        overlay = image.copy()
        for item in detections:
            x1, y1, x2, y2 = map(round, item["bbox_px"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 210, 255), 2)
            label_image(overlay, f"{item['label']} {item['confidence']:.2f}", (x1, max(18, y1 - 5)), (0, 210, 255))
        cv2.imwrite(str(out / "overlays" / f"yolo_{trial['trial_id']}.png"), overlay)
        trials.append({**trial, "detections": detections, "detection_count": len(detections)})
    dump(out / "vocabulary_trials_yolo.json", {
        "status": "PASS", "detector": "Ultralytics YOLO-World", "model": str(YOLO_MODEL),
        "parameters": {"imgsz": 1280, "confidence": .02, "iou": .50, "max_det": 100}, "trials": trials,
        "runtime": {"seconds": time.perf_counter() - started, "cuda_used": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__, "ultralytics": ultralytics.__version__},
    })


def dino_diagnostic(run: Path) -> None:
    from PIL import Image
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    out = run / "detector_diagnostics"
    bgr = cv2.imread(str(IMAGE))
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    processor = AutoProcessor.from_pretrained(DINO_MODEL, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    trials, started = [], time.perf_counter()
    for trial in TRIALS:
        prompt = ". ".join(trial["labels"]) + "."
        inputs = processor(images=pil, text=prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(outputs, inputs["input_ids"], threshold=.15, text_threshold=.15, target_sizes=[pil.size[::-1]])[0]
        labels = result.get("text_labels") or result.get("labels")
        detections = []
        for index, (score, box, label) in enumerate(zip(result["scores"].cpu().tolist(), result["boxes"].cpu().tolist(), labels), 1):
            detections.append({
                "id": f"d_{index:03d}", "label": str(label), "confidence": round(float(score), 6),
                "bbox_px": [round(float(value), 2) for value in box], "coordinate_system": "original_image_pixels",
            })
        overlay = bgr.copy()
        for item in detections:
            x1, y1, x2, y2 = map(round, item["bbox_px"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 255, 80), 2)
            label_image(overlay, f"{item['label']} {item['confidence']:.2f}", (x1, max(18, y1 - 5)), (80, 255, 80))
        cv2.imwrite(str(out / "overlays" / f"dino_{trial['trial_id']}.png"), overlay)
        trials.append({**trial, "prompt": prompt, "detections": detections, "detection_count": len(detections)})
        del inputs, outputs
    dump(out / "vocabulary_trials_dino.json", {
        "status": "PASS", "detector": "Grounding DINO", "model": DINO_MODEL,
        "parameters": {"box_threshold": .15, "text_threshold": .15}, "trials": trials,
        "runtime": {"seconds": time.perf_counter() - started, "cuda_used": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__},
    })


def label_trial_summary(detector: str, trials: list[dict]) -> list[dict]:
    rows = []
    for trial in trials:
        for tested_label in trial["labels"]:
            matches = [item for item in trial["detections"] if tested_label.casefold() in item["label"].casefold() or item["label"].casefold() in tested_label.casefold()]
            rows.append({
                "detector": detector, "trial_id": trial["trial_id"], "tested_label": tested_label,
                "detection": bool(matches), "count": len(matches),
                "best_confidence": max((item["confidence"] for item in matches), default=None),
                "bboxes_px": [item["bbox_px"] for item in matches],
            })
    return rows


def diagnostic_for(name: str, references: list[dict], dino_trials, yolo_trials, baseline_dino, baseline_yolo) -> dict:
    successful, all_family_detections = [], []
    for detector, trials in (("Grounding DINO", dino_trials), ("YOLO-World", yolo_trials)):
        for trial in trials:
            for item in trial["detections"]:
                if family(item["label"]) != name:
                    continue
                candidate = {**item, "detector": detector, "trial_id": trial["trial_id"]}
                if references:
                    candidate["reference_ious"] = [{"reference_id": ref["id"], "iou": iou(item["bbox_px"], ref["bbox_px"])} for ref in references]
                    candidate["best_reference_iou"] = max(value["iou"] for value in candidate["reference_ious"])
                    if candidate["best_reference_iou"] >= .30:
                        successful.append(candidate)
                all_family_detections.append(candidate)
    wrong = []
    if references:
        for detector, items in (("Grounding DINO baseline", baseline_dino), ("YOLO-World baseline", baseline_yolo)):
            for item in items:
                best = max(iou(item["bbox_px"], ref["bbox_px"]) for ref in references)
                if best >= .50 and family(item.get("label", "")) != name:
                    wrong.append({"detector": detector, "id": item["id"], "label": item["label"], "confidence": item.get("confidence", item.get("score")), "bbox_px": item["bbox_px"], "best_iou": best})
    best = max(successful, key=lambda item: item.get("best_reference_iou", 0), default=None)
    trial_hits = {(item["detector"], item["trial_id"]) for item in successful}
    if not references:
        diagnosis = "AMBIGUOUS" if all_family_detections else "NOT_LOCALIZED"
    elif successful and len(trial_hits) < 6:
        diagnosis = "VOCABULARY_DEPENDENT"
    elif successful:
        diagnosis = "AMBIGUOUS"
    elif wrong:
        diagnosis = "WRONG_CLASS"
    else:
        diagnosis = "NOT_LOCALIZED"
    return {
        "reference_object": name,
        "reference_status": "QWEN_MODEL_OUTPUT_NOT_GROUND_TRUTH" if references else "NO_REVIEWED_REFERENCE_REGION",
        "expected_regions": references or None,
        "tested_labels": FAMILY_TERMS[name],
        "successful_labels": sorted({item["label"] for item in successful}),
        "wrong_labels_at_same_location": wrong,
        "best_iou_detection": best,
        "all_family_detections_without_reference_filter": all_family_detections,
        "diagnosis": diagnosis,
    }


def cross_detector(diagnostics: dict, dino_trials, yolo_trials) -> dict:
    families = {}
    for name in FAMILY_TERMS:
        dino = [{**item, "trial_id": trial["trial_id"]} for trial in dino_trials for item in trial["detections"] if family(item["label"]) == name]
        yolo = [{**item, "trial_id": trial["trial_id"]} for trial in yolo_trials for item in trial["detections"] if family(item["label"]) == name]
        pairs = [{"dino_id": a["id"], "dino_label": a["label"], "dino_trial": a["trial_id"], "yolo_id": b["id"], "yolo_label": b["label"], "yolo_trial": b["trial_id"], "bbox_iou": iou(a["bbox_px"], b["bbox_px"])} for a in dino for b in yolo]
        families[name] = {
            "grounding_dino_detections": len(dino), "yolo_world_detections": len(yolo),
            "best_cross_detector_match": max(pairs, key=lambda item: item["bbox_iou"], default=None),
            "diagnosis": diagnostics[name]["diagnosis"],
        }
    return {
        "ground_truth_available": False,
        "comparison_scope": "detection behavior on this image; confidence scales are detector-specific and not directly comparable",
        "same_vocabulary_trials": [item["labels"] for item in TRIALS],
        "families": families,
    }


def prediction_metrics(paths: dict[str, Path]) -> tuple[dict, dict[str, list[dict]]]:
    outputs, metrics = {}, {}
    for name, path in paths.items():
        predictions = load(path)["predictions"]
        outputs[name] = predictions
        statuses = Counter(item["status"] for item in predictions)
        regions = Counter(str(item["predicted_semantic_region"]) for item in predictions if item["predicted_semantic_region"])
        metrics[name] = {
            "total": len(predictions), "statuses": dict(statuses),
            "supporting_object_references": sum(len(item["supporting_objects"]) for item in predictions),
            "relation_references": sum(len(item["supporting_relations"]) for item in predictions),
            "semantic_region_distribution": dict(regions),
        }
    common_ids = sorted(set.intersection(*(set(item["task_id"] for item in values) for values in outputs.values())))
    agreement = []
    for task_id in common_ids:
        values = {name: next(item for item in predictions if item["task_id"] == task_id) for name, predictions in outputs.items()}
        labels = [str(item["predicted_semantic_region"]) for item in values.values()]
        agreement.append({"task_id": task_id, "all_sources_same_region_label": len(set(labels)) == 1, "predictions": {name: {"status": item["status"], "region": item["predicted_semantic_region"], "target_pixel": item["target_pixel"]} for name, item in values.items()}})
    return {"per_source": metrics, "cross_source_agreement": agreement}, outputs


def draw_summaries(run: Path, structured: list[dict]) -> None:
    image = cv2.imread(str(IMAGE))
    colors = {"TYPE_A": (255, 180, 0), "TYPE_B": (0, 220, 255), "TYPE_C": (100, 255, 100), "TYPE_D": (255, 100, 255), "TYPE_E": (80, 80, 255)}
    for task_type in colors:
        out = image.copy()
        for item in structured:
            if item["task_type"] != task_type:
                continue
            polygon = item["region_geometry"]["polygon_px"]
            if polygon:
                cv2.polylines(out, [np.asarray(polygon, np.int32)], True, colors[task_type], 2)
            if item["target_pixel"]:
                point = tuple(map(round, item["target_pixel"]))
                cv2.drawMarker(out, point, colors[task_type], cv2.MARKER_CROSS, 24, 3)
                label_image(out, item["task_id"], (point[0] + 8, point[1] - 8), colors[task_type], .6)
        label_image(out, f"{task_type} structured-context predictions / NOT GROUND TRUTH", (18, 30), colors[task_type], .65)
        destination = run / ("04_semantic_regions/overlays" if task_type == "TYPE_C" else "05_target_grounding/overlays") / f"{task_type}_summary.png"
        cv2.imwrite(str(destination), out)


def finalize(run: Path) -> None:
    dino_data = load(run / "detector_diagnostics/vocabulary_trials_dino.json")
    yolo_data = load(run / "detector_diagnostics/vocabulary_trials_yolo.json")
    dino_trials, yolo_trials = dino_data["trials"], yolo_data["trials"]
    baseline_dino = load(BASELINE / "04_grounding_dino/detections.json")["detections"]
    baseline_yolo = load(YOLO_BASELINE / "07_yolo_world/detections.json")["detections"]
    qwen_objects = load(BASELINE / "01_qwen/objects.json")["objects"]
    refs = {
        "manipulator": [item for item in qwen_objects if item["id"] in {"q_030", "q_031"}],
        "conveyor": [],
        "mobile_robot": [item for item in qwen_objects if item["id"] == "q_029"],
        "pallet": [item for item in qwen_objects if item["category"] == "pallet"],
    }
    diagnostics = {name: diagnostic_for(name, refs[name], dino_trials, yolo_trials, baseline_dino, baseline_yolo) for name in FAMILY_TERMS}
    label_rows = label_trial_summary("Grounding DINO", dino_trials) + label_trial_summary("YOLO-World", yolo_trials)
    dump(run / "detector_diagnostics/vocabulary_trials.json", {
        "status": "PASS", "controlled_trial_count": 3, "all_synonyms_tested": True,
        "confidence_warning": "Detector confidence scales are not directly comparable.",
        "label_results": label_rows, "source_files": ["vocabulary_trials_dino.json", "vocabulary_trials_yolo.json"],
    })
    for name, result in diagnostics.items():
        dump(run / f"detector_diagnostics/{name}_diagnostic.json", result)
    comparison = cross_detector(diagnostics, dino_trials, yolo_trials)
    dump(run / "detector_diagnostics/detector_cross_comparison.json", comparison)

    paths = {
        "structured_context": run / "02_structured_context/predictions.json",
        "qwen": run / "03_end_to_end/qwen/predictions.json",
        "grounding_dino": run / "03_end_to_end/grounding_dino/predictions.json",
        "yolo_world": run / "03_end_to_end/yolo_world/predictions.json",
    }
    metrics, outputs = prediction_metrics(paths)
    augmented = load(run / "03_end_to_end/diagnostic_augmented/predictions.json")["predictions"]
    work_area = {name: [item for item in values if item["task_id"] in {"C03", "D02"}] for name, values in outputs.items()}
    work_area["diagnostic_augmented"] = augmented
    metrics["ground_truth_blocked_metrics"] = [
        "Spatial Relation Accuracy", "Semantic Region Classification Accuracy", "Semantic Region IoU",
        "Instruction Grounding Success Rate", "Goal-in-Region Rate", "Abstention Precision", "Abstention Recall",
    ]
    dump(run / "04_semantic_regions/predictions.json", {
        "ground_truth_available": False,
        "predictions": {name: [item for item in values if item["task_type"] == "TYPE_C"] for name, values in outputs.items()},
    })
    dump(run / "05_target_grounding/target_predictions.json", {
        "coordinate_systems": {"target_pixel": "original_image_pixels", "target_metric_xy": "synthetic_workspace_bottom_left_xy_m"},
        "synthetic_only": True,
        "predictions": {name: [item for item in values if item["target_pixel"] is not None] for name, values in outputs.items()},
    })
    draw_summaries(run, outputs["structured_context"])

    yolo_work = work_area["yolo_world"]
    augmented_work = work_area["diagnostic_augmented"]
    error_analysis = {
        "status": "PARTIAL",
        "ground_truth_available": False,
        "taxonomy": ["DETECTION_FAILURE", "CLASSIFICATION_FAILURE", "RELATION_FAILURE", "SEMANTIC_INFERENCE_FAILURE", "TARGET_GENERATION_FAILURE", "AMBIGUITY_HANDLING_FAILURE"],
        "validated_errors": [
            {"reference": "q_030", "category": "CLASSIFICATION_FAILURE", "evidence": "Baseline YOLO cart overlaps Qwen manipulator at IoU 0.9311; location was found under the wrong label."},
            {"reference": "q_029", "category": "CLASSIFICATION_FAILURE", "evidence": "Baseline YOLO truck overlaps Qwen mobile_robot at IoU 0.9231; location was found under the wrong label."},
        ],
        "detector_diagnostics": {name: value["diagnosis"] for name, value in diagnostics.items()},
        "work_area_predictions_by_source": work_area,
        "diagnostic_augmentation_changed_work_area_output": None,
        "diagnostic_augmentation_comparison_status": "BLOCKED_BY_QWEN_SOURCE_SPECIFIC_GENERATION",
        "semantic_or_relation_error_claims": "BLOCKED_BY_GROUND_TRUTH",
        "quantitative_without_ground_truth": metrics,
    }
    dump(run / "06_error_analysis/error_analysis.json", error_analysis)
    write(run / "06_error_analysis/RESULT.md", """# Error Analysis

Status: PARTIAL

Detector localization/classification behavior is separated from semantic reasoning. The q_030→cart and q_029→truck high-IoU cases are classification failures rather than localization failures. Relation, semantic-region, target, and abstention correctness remain BLOCKED_BY_GROUND_TRUTH.
""")

    validation_rows = []
    for case in TEST_CASES:
        validation_rows.append({
            "task_id": case["task_id"], "instruction": case["instruction"],
            "candidate_region_for_review": case.get("candidate_region"),
            "predictions": {name: next(item for item in values if item["task_id"] == case["task_id"]) for name, values in outputs.items()},
            "human_selection": None,
            "allowed_selections": ["CORRECT", "WRONG_REGION", "BOUNDARY_ADJUST", "WRONG_RELATION", "WRONG_OBJECT", "AMBIGUOUS", "SHOULD_ABSTAIN", "UNSURE"],
            "human_notes": "",
        })
    dump(run / "07_human_validation/semantic_ground_truth_template.json", {
        "status": "AWAITING_HUMAN_REVIEW", "ground_truth_created": False, "items": validation_rows,
    })
    lines = ["# Human Semantic Validation", "", "Status: AWAITING_HUMAN_REVIEW", "", "This form does not define ground truth automatically. Review the JSON predictions and overlays, then select one allowed label per task.", "", "| Task | Instruction | Candidate | Human selection | Notes |", "|---|---|---|---|---|"]
    for case in TEST_CASES:
        lines.append(f"| {case['task_id']} | {case['instruction']} | {case.get('candidate_region', '—')} |  |  |")
    write(run / "07_human_validation/HUMAN_SEMANTIC_VALIDATION.md", "\n".join(lines))

    write(run / "08_future_experiments/FUTURE_SEMANTIC_SCENE_PLAN.md", """# Future Semantic Scene Plan

| Scene | Objects | Change | Held constant | Shortcut tested | Instructions | Required annotation |
|---|---|---|---|---|---|---|
| A baseline | truck, racks, pallets, workbench, robot arms, conveyor, cart, drums, mobile robot | none | camera/lighting/config | baseline | current TYPE A–E set | reviewed objects, relations, regions, masks, targets |
| B rearranged | same as A | exchange Loading/Storage/Work/Material positions | object identities, camera | absolute-position shortcut | same TYPE C/D wording | new boxes, relations, region polygons, valid targets |
| C monochrome | same geometry | recolor or use white printed objects | layout/camera | color shortcut | same TYPE A–D wording | identity and region review |
| D partial cues | truck without pallet; manipulator without conveyor | remove one semantic cue | camera/remaining objects | evidence reduction | Loading/Work Area requests | presence/absence and expected abstention |
| E ambiguous | repeated pallets/racks/drums | add equally plausible instances | camera/classes | relation disambiguation | nearest/between instructions | instance IDs, relations, intended referent |
""")
    write(run / "08_future_experiments/ABLATION_PLAN.md", """# Ablation Plan

No ablation performance value is reported in this run.

1. Object only vs object+spatial relation vs object+relation+semantic reasoning.
2. Qwen-only vs Grounding DINO/SAM2-assisted vs YOLO-World/SAM2-assisted.
3. Color-rich vs monochrome.
4. Original vs rearranged layout.
5. Full vs partial semantic cues.

Each requires a frozen test set and human-reviewed objects, relations, region boundaries, abstention labels, and valid target regions.
""")

    write(run / "detector_diagnostics/RESULT.md", f"""# Detector Diagnostic

Status: PASS (behavior characterization; no accuracy claim)

- Controlled vocabulary combinations: 3
- Manipulator diagnosis: `{diagnostics['manipulator']['diagnosis']}`
- Conveyor diagnosis: `{diagnostics['conveyor']['diagnosis']}`
- Mobile robot diagnosis: `{diagnostics['mobile_robot']['diagnosis']}`
- Pallet diagnosis: `{diagnostics['pallet']['diagnosis']}`
- q_030/cart and q_029/truck are high-IoU wrong-class localizations, not failures to localize the positions.
- Conveyor has no reviewed reference bbox; detections are cross-detector behavior only and remain human-reviewable.
""")

    inventory = load(run / "00_inventory/inventory.json")
    qwen_meta = tree_hash(Path("/home/binci/miniconda3/envs/qwen_vl/conda-meta"))
    yolo_meta = tree_hash(Path("/home/binci/miniconda3/envs/yolo_world/conda-meta"))
    inventory["environment_hash_after"] = {"qwen_vl_conda_meta": qwen_meta, "yolo_world_conda_meta": yolo_meta}
    inventory["environment_unchanged"] = {
        "qwen_vl": qwen_meta == inventory["environment_hash_before"]["qwen_vl_conda_meta"],
        "yolo_world": yolo_meta == inventory["environment_hash_before"]["yolo_world_conda_meta"],
    }
    dump(run / "00_inventory/inventory.json", inventory)

    source_lines = []
    for name, value in metrics["per_source"].items():
        source_lines.append(f"- {name}: {value['statuses']} / regions {value['semantic_region_distribution']}")
    diagnosis_lines = [f"- {name}: {value['diagnosis']}" for name, value in diagnostics.items()]
    report = f"""# Semantic Reasoning Run Report

Overall status: PARTIAL

## 1. Research question

Can a VLM use objects, spatial relations, and functional context from a third-person workspace image to ground a natural-language instruction to a semantic region and a traversable target, while abstaining when evidence is insufficient?

## 2. Input image

`{IMAGE}` (1920×1080, SHA-256 `{sha256(IMAGE)}`)

## 3. Models and environments

- Qwen semantic grounding: `{QWEN_MODEL}` in unchanged `qwen_vl`; actual image-conditioned inference.
- Grounding DINO diagnostic: `{DINO_MODEL}` in unchanged `qwen_vl`; CUDA.
- YOLO-World diagnostic: `{YOLO_MODEL.name}` in unchanged `yolo_world`; CUDA.
- Existing DINO/SAM2 and YOLO/SAM2 masks are referenced read-only; segmentation was not redundantly rerun.

## 4. Test cases

15 tasks: TYPE_A 2, TYPE_B 5, TYPE_C 4, TYPE_D 2, TYPE_E 2. Object bbox, semantic region, obstacle/no-go, and robot goal are kept separate.

## 5. Actual inference

Qwen completed the 15-task STRUCTURED_CONTEXT inference. Source-specific Qwen generation was attempted with 15-task, 8-task, and 4-task scopes but returned no complete first response and is BLOCKED after three bounded attempts. Three controlled vocabulary trials completed independently in both detectors.

## 6. Detector Diagnostic

{chr(10).join(diagnosis_lines)}

The q_030 location was previously localized as `cart` by YOLO at IoU 0.9311, and q_029 as `truck` at IoU 0.9231. These are WRONG_CLASS_LOCALIZATION / CLASSIFICATION_FAILURE cases. Conveyor comparison now uses the same three vocabulary groups in both detectors; because no reviewed conveyor bbox exists, behavior is reported without accuracy or superiority claims.

Work Area detector evidence is stored in `06_error_analysis/error_analysis.json`; the requested before/after Qwen semantic comparison is BLOCKED because source-specific generation did not complete.

## 7. Structured-context and end-to-end results

{chr(10).join(source_lines)}

STRUCTURED_CONTEXT is model-derived and human-reviewable, not an oracle. End-to-end paths restrict supporting object identities to each detector source.

## 8. Semantic-region predictions and target grounding

TYPE_C predictions are in `04_semantic_regions/predictions.json`. Pixel targets and deterministic synthetic metric projections are in `05_target_grounding/target_predictions.json`. Outside-workspace projections are flagged and never clamped.

## 9. Detector source effects

Cross-source semantic agreement is BLOCKED because the source-specific Qwen paths did not complete. Detector evidence differences are still recorded on this image without accuracy or superiority claims.

## 10. Failure taxonomy

Confirmed detector errors use DETECTION_FAILURE or CLASSIFICATION_FAILURE. RELATION_FAILURE, SEMANTIC_INFERENCE_FAILURE, TARGET_GENERATION_FAILURE, and AMBIGUITY_HANDLING_FAILURE require human review and remain BLOCKED_BY_GROUND_TRUTH.

## 11. Ground truth

Human-reviewed semantic ground truth is unavailable. The Qwen reference boxes and semantic regions are model outputs, not ground truth.

## 12. Claims currently supported

- The 15-task structured-context path and both detectors' three vocabulary trials execute and leave raw/structured artifacts; source-specific Qwen paths are explicitly BLOCKED.
- High-IoU wrong-class localization is separable from failure to localize.
- Pixel targets can be projected through the retained synthetic-only homography.
- Source-dependent semantic outputs and abstention behavior can be counted without calling them correct.

## 13. Claims not supported

No semantic accuracy, relation accuracy, region IoU, instruction success rate, goal-in-region rate, abstention precision/recall, detector superiority, or real-camera metric accuracy is claimed.

## 14. Human validation

Complete `07_human_validation/HUMAN_SEMANTIC_VALIDATION.md` or the JSON template using only the listed review labels. Conveyor location and every predicted region polygon/target require review.

## 15. Real-camera repetition

Capture an undistorted fixed-camera frame, measure floor correspondences with holdouts, annotate objects/relations/regions/masks/valid targets, then rerun with frozen prompts and thresholds before any dry-run or robot motion.

## 16. Next priorities

1. Human-review all 15 tasks and the conveyor candidate detections.
2. Recompute blocked semantic and abstention metrics from reviewed annotations.
3. Capture Scenes B–E to test position, color, cue-removal, and ambiguity shortcuts.
4. Repeat homography and target validation with measured real-camera geometry.
"""
    write(run / "SEMANTIC_REASONING_RUN_REPORT.md", report)
    manifest = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            manifest.append({"path": str(path.relative_to(run)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    dump(run / "ARTIFACT_MANIFEST.json", {"run": run.name, "files": manifest})
    self_check(run)
    write(run / "logs/self_check.log", "self-check: PASS")


def self_check(run: Path | None = None) -> None:
    assert round(iou([0, 0, 10, 10], [5, 5, 15, 15]), 6) == .142857
    assert family("industrial robot arm") == "manipulator"
    assert family("AMR") == "mobile_robot"
    assert len(TEST_CASES) == 15 and len(TRIALS) == 3
    if run:
        required = [
            "01_test_cases/test_cases.json", "02_structured_context/predictions.json",
            "03_end_to_end/qwen/predictions.json", "03_end_to_end/grounding_dino/predictions.json",
            "03_end_to_end/yolo_world/predictions.json", "04_semantic_regions/predictions.json",
            "05_target_grounding/target_predictions.json", "06_error_analysis/error_analysis.json",
            "07_human_validation/semantic_ground_truth_template.json", "08_future_experiments/FUTURE_SEMANTIC_SCENE_PLAN.md",
            "detector_diagnostics/vocabulary_trials.json", "detector_diagnostics/manipulator_diagnostic.json",
            "detector_diagnostics/conveyor_diagnostic.json", "detector_diagnostics/mobile_robot_diagnostic.json",
            "detector_diagnostics/pallet_diagnostic.json", "detector_diagnostics/detector_cross_comparison.json",
            "detector_diagnostics/RESULT.md", "SEMANTIC_REASONING_RUN_REPORT.md",
        ]
        for relative in required:
            path = run / relative
            assert path.is_file() and path.stat().st_size > 0, path
            if path.suffix == ".json":
                load(path)
        inventory = load(run / "00_inventory/inventory.json")
        assert all(inventory["environment_unchanged"].values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "yolo-diagnostic", "dino-diagnostic", "qwen", "finalize", "self-check"])
    parser.add_argument("--run", type=Path, required=False)
    args = parser.parse_args()
    if args.stage == "self-check" and args.run is None:
        self_check(); print("self-check: PASS"); return
    if args.run is None:
        parser.error("--run is required")
    run = args.run.resolve()
    stages = {
        "prepare": prepare, "yolo-diagnostic": yolo_diagnostic,
        "dino-diagnostic": dino_diagnostic, "qwen": qwen_stage,
        "finalize": finalize, "self-check": self_check,
    }
    try:
        stages[args.stage](run)
        print(f"{args.stage}: PASS\n{run}")
    except Exception:
        run.mkdir(parents=True, exist_ok=True)
        write(run / "logs" / f"{args.stage}_exception.log", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
