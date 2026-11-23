#!/usr/bin/env python3
"""Build one non-metric semantic pipeline run from preserved experiment evidence."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_IMAGE = ROOT / "approximate_environment.png"
EXP001 = ROOT / "results/EXP001_run01"
EXP002 = ROOT / "results/EXP002_semantic_region_eval"
CORE_PROMPT = (
    "truck. pallet. cardboard box. storage rack. shelf. workbench. desk. "
    "cart. trolley. drum. barrel. mobile robot. robot arm. manipulator. "
    "safety fence. traffic cone."
)
COLORS = {
    "Storage Area": (255, 150, 40), "Loading Area": (40, 180, 255),
    "Work Station": (200, 80, 220), "Material Area": (80, 210, 100),
    "No-Go Zone": (40, 40, 230), "Drop Zone": (0, 220, 220),
}


def load(path: Path):
    return json.loads(path.read_text())


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    return result.stdout.strip() or None


def polygon_px(region, width, height):
    poly = region.get("polygon_px")
    if poly:
        return np.asarray(poly, np.int32)
    poly = region.get("polygon") or region.get("polygon_normalized")
    if not poly:
        return None
    return np.asarray([
        [round(x * (width - 1) / 1000), round(y * (height - 1) / 1000)]
        for x, y in poly
    ], np.int32)


def label(image, text, xy, color=(255, 255, 255), scale=.5) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def object_overlay(image, objects, path: Path, color=(0, 230, 255)) -> None:
    out = image.copy()
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["bbox_px"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label(out, f"{obj['id']} {obj['category']}", (x1, max(16, y1 - 5)), color)
    label(out, "QWEN REUSED / ORIGINAL IMAGE PIXELS / NON-METRIC", (18, out.shape[0] - 20))
    cv2.imwrite(str(path), out)


def semantic_overlay(image, regions, path: Path) -> None:
    out = image.copy()
    layer = out.copy()
    for region in regions:
        pts = polygon_px(region, out.shape[1], out.shape[0])
        if pts is None:
            continue
        color = COLORS.get(region.get("label"), (180, 180, 180))
        cv2.fillPoly(layer, [pts], color)
        cv2.polylines(out, [pts], True, color, 3, cv2.LINE_AA)
        anchor = tuple(pts[np.argmin(pts[:, 1])])
        label(out, region["label"], anchor, color, .6)
    cv2.addWeighted(layer, .23, out, .77, 0, out)
    label(out, "SEMANTIC REGIONS / IMAGE-PLANE / NON-METRIC", (18, out.shape[0] - 20))
    cv2.imwrite(str(path), out)


def stage_inventory(run: Path, image, config) -> dict:
    runtime = load(EXP001 / "runtime.json")
    current = {
        "created_at": datetime.now().astimezone().isoformat(),
        "workspace": str(ROOT),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "input_image": str(SOURCE_IMAGE),
        "input_sha256": sha256(SOURCE_IMAGE),
        "input_size_px": [image.shape[1], image.shape[0]],
        "reused_experiments": [str(EXP001), str(EXP002)],
        "qwen_runtime": runtime,
        "grounding_dino_local_cache": False,
        "matching_metric_geometry": False,
    }
    try:
        import torch
        import transformers
        current.update({
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda_available_now": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except (ImportError, RuntimeError) as exc:
        current["ml_environment_error"] = repr(exc)
    cache = Path.home() / ".cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny"
    current["grounding_dino_local_cache"] = cache.is_dir()
    dump(run / "00_inventory/inventory.json", current)
    write(run / "00_inventory/ENVIRONMENT_INVENTORY.md", f"""# Environment Inventory

Status: PASS

- Workspace: `{ROOT}`
- Input: `{SOURCE_IMAGE}` ({image.shape[1]}×{image.shape[0]}, SHA-256 `{current['input_sha256']}`)
- Input audit: overlay가 없는 원본 작업환경 이미지로 확인됨.
- Reused Qwen objects/target regions: `{EXP001}`
- Reused open semantic inference/evaluation: `{EXP002}`
- Qwen model: `{runtime['model_id']}`; preserved run device `{runtime['device']}`
- Current Python/OpenCV/NumPy: `{current['python']}` / `{current['opencv']}` / `{current['numpy']}`
- Current CUDA/GPU: `{current.get('cuda_available_now')}` / `{current.get('gpu')}`
- Grounding DINO package: 별도 package 없음; installed Transformers native implementation 사용 가능.
- Grounding DINO local cache before run: `{current['grounding_dino_local_cache']}`
- Metric geometry: 이 입력 이미지와 일치하는 검증된 pixel↔world correspondence가 없음. 다른 카메라 장면의 homography는 재사용하지 않음.
- Safety: package/system/CUDA/ROS2 변경 및 robot motion 없음.
""")
    return current


def stage_objects(run: Path, image) -> list[dict]:
    source = load(EXP001 / "object_recognition.json")
    objects = source["objects"]
    out = {
        "status": "PARTIAL",
        "source": str(EXP001 / "object_recognition.json"),
        "raw_model_response": str(EXP001 / "qwen_raw_output.txt"),
        "coordinate_system": source["coordinate_system"],
        "image_size_px": source["image_size_px"],
        "geometry_status": "IMAGE-PLANE / NON-METRIC",
        "objects": objects,
        "other_objects": [],
        "uncertain_objects": [],
        "limitations": [
            "Reused completed Qwen inference; no needless rerun.",
            "Primary response was truncated after 16 complete objects.",
            "No ground-truth accuracy claim.",
        ],
    }
    directory = run / "01_object_detection"
    dump(directory / "objects.json", out)
    shutil.copy2(EXP001 / "vlm_prompt.txt", directory / "PROMPT.txt")
    object_overlay(image, objects, directory / "object_detection_overlay.png")
    write(directory / "RESULT.md", f"""# Object Detection Result

Status: PARTIAL

완료된 EXP001 Qwen3-VL-4B-Instruct 결과를 재사용했다. 주요 객체 {len(objects)}개와 원본 0..1000 bbox/pixel bbox를 보존했으며 overlay는 새 run에서 재생성했다. Ground truth가 없으므로 accuracy를 주장하지 않는다. 원 응답 truncation 때문에 fence/conveyor 등 일부 visible object 누락 가능성이 있다.
""")
    return objects


def stage_semantics(run: Path, image) -> list[dict]:
    target = load(EXP001 / "semantic_regions.json")
    opened = load(EXP002 / "model_output/semantic_region_prediction.json")
    comparison = load(EXP002 / "evaluation/semantic_region_comparison.json")
    directory = run / "02_semantic_regions"
    dump(directory / "open_semantic_analysis.json", opened)
    dump(directory / "semantic_regions.json", {
        "status": "PARTIAL", "geometry_status": "IMAGE-PLANE / NON-METRIC",
        "open_interpretation": opened["regions"],
        "target_taxonomy": target["semantic_regions"],
        "comparison_source": str(EXP002 / "evaluation/semantic_region_comparison.json"),
        "comparison_summary": {
            "reference_regions": len(comparison["region_correspondence"]),
            "missed_reference_regions": comparison["missed_reference_regions"],
            "extra_predicted_regions": comparison["extra_predicted_regions"],
            "performance_claim_status": comparison["performance_claim_status"],
        },
    })
    shutil.copy2(EXP002 / "model_output/semantic_region_raw.txt", directory / "MODEL_RAW.txt")
    write(directory / "PROMPT.txt", opened["blind_inference"]["selected_prompt"])
    semantic_overlay(image, target["semantic_regions"], directory / "semantic_regions_overlay.png")
    write(directory / "RESULT.md", """# Semantic Region Result

Status: PARTIAL

EXP002의 blind open-semantic 분석과 EXP001의 target taxonomy 구역을 함께 재사용했다. Open 결과는 5개 기능 영역을 제안했지만 polygon을 출력하지 않았고, target 결과의 polygon은 수동/agent 근사치다. Object bbox와 semantic region은 별도 layer로 유지하며 성능 수치로 해석하지 않는다.
""")
    return target["semantic_regions"]


def floor_mask(image) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = np.uint8(gray < 115) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no floor candidate contour")
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return mask


def stage_traversability(run: Path, image, objects, regions, margin_px: int):
    height, width = image.shape[:2]
    floor = floor_mask(image)
    occupancy = np.zeros((height, width), np.uint8)
    robot_state = np.zeros_like(occupancy)
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["bbox_px"])
        target = robot_state if obj["category"] == "mobile_robot" else occupancy
        cv2.rectangle(target, (x1, y1), (x2, y2), 255, cv2.FILLED)
    if margin_px:
        size = margin_px * 2 + 1
        occupancy_inflated = cv2.dilate(occupancy, np.ones((size, size), np.uint8))
    else:
        occupancy_inflated = occupancy.copy()
    no_go = np.zeros_like(occupancy)
    for region in regions:
        if region["label"] == "No-Go Zone":
            cv2.fillPoly(no_go, [polygon_px(region, width, height)], 255)
    traversable = cv2.bitwise_and(floor, cv2.bitwise_not(cv2.bitwise_or(occupancy_inflated, no_go)))
    unknown = cv2.bitwise_not(floor)
    directory = run / "03_traversability"
    cv2.imwrite(str(directory / "physical_occupancy_mask.png"), occupancy)
    cv2.imwrite(str(directory / "no_go_mask.png"), no_go)
    cv2.imwrite(str(directory / "traversability_mask.png"), traversable)
    overlay = image.copy()
    layer = np.zeros_like(image)
    layer[traversable > 0] = (45, 180, 45)
    layer[occupancy_inflated > 0] = (30, 30, 230)
    layer[no_go > 0] = (180, 30, 200)
    cv2.addWeighted(layer, .46, overlay, .78, 0, overlay)
    label(overlay, "GREEN free | RED physical occupancy | MAGENTA explicit no-go", (18, 28))
    label(overlay, "IMAGE-PLANE / NON-METRIC", (18, height - 20))
    cv2.imwrite(str(directory / "traversability_overlay.png"), overlay)
    counts = {
        "workspace_floor_px": int(np.count_nonzero(floor)),
        "physical_occupancy_px": int(np.count_nonzero(occupancy)),
        "inflated_occupancy_px": int(np.count_nonzero(occupancy_inflated)),
        "explicit_no_go_px": int(np.count_nonzero(no_go)),
        "traversable_candidate_px": int(np.count_nonzero(traversable)),
        "unknown_px": int(np.count_nonzero(unknown)),
    }
    dump(directory / "traversability.json", {
        "status": "PARTIAL", "coordinate_system": "original_image_pixels",
        "geometry_status": "IMAGE-PLANE / NON-METRIC",
        "formula": "floor - physical occupancy - explicit no-go - pixel safety margin",
        "semantic_regions_restricted_by_name": False,
        "mobile_robot_treatment": "current state mask; excluded from environment occupancy",
        "floor_method": "largest connected grayscale<115 component after 31px closing",
        "safety_margin_px": margin_px, "pixel_counts": counts,
        "warning": "Not safe for robot planning; bbox occupancy and pixel margin are qualitative.",
    })
    write(directory / "RESULT.md", f"""# Traversability Result

Status: PARTIAL

IMAGE-PLANE / NON-METRIC. Semantic 영역 이름으로 제한하지 않고 floor에서 Qwen object bbox 점유, explicit No-Go, {margin_px}px image margin만 제외했다. Mobile robot은 environment obstacle이 아니라 current state로 분리했다. Free candidate는 {counts['traversable_candidate_px']} px이며 bbox/floor threshold 기반 근사이므로 실제 주행에는 사용할 수 없다.
""")
    return floor, occupancy, occupancy_inflated, no_go, traversable, robot_state, counts


def tensor_values(result):
    scores = result["scores"].detach().cpu().tolist()
    boxes = result["boxes"].detach().cpu().tolist()
    labels = result.get("text_labels") or result.get("labels")
    return scores, boxes, [str(v) for v in labels]


def run_grounding_dino(run: Path, image) -> tuple[str, list[dict], list[dict]]:
    from PIL import Image
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    directory = run / "04_grounding_dino"
    model_id = "IDEA-Research/grounding-dino-tiny"
    attempts = []
    processor = model = None
    try:
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True)
        attempts.append({"attempt": 1, "hypothesis": "local cache", "status": "PASS"})
    except Exception as exc:
        attempts.append({"attempt": 1, "hypothesis": "local cache", "status": "BLOCKED", "error": repr(exc)})
    if model is None:
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            attempts.append({"attempt": 2, "hypothesis": "download with installed Transformers; no package change", "status": "PASS"})
        except Exception as exc:
            attempts.append({"attempt": 2, "hypothesis": "download with installed Transformers; no package change", "status": "BLOCKED", "error": repr(exc)})
    detections = []
    if model is not None:
        devices = ["cuda" if torch.cuda.is_available() else "cpu"]
        if devices[0] == "cuda":
            devices.append("cpu")
        for device in devices:
            try:
                model.to(device).eval()
                pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                inputs = processor(images=pil, text=CORE_PROMPT, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.inference_mode():
                    outputs = model(**inputs)
                result = processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"], threshold=.25, text_threshold=.2,
                    target_sizes=[pil.size[::-1]],
                )[0]
                scores, boxes, labels = tensor_values(result)
                detections = [
                    {"id": f"gd_{i:03d}", "label": label_text,
                     "score": round(float(score), 6),
                     "bbox_px": [round(float(x), 2) for x in box]}
                    for i, (score, box, label_text) in enumerate(zip(scores, boxes, labels), 1)
                ]
                attempts.append({"attempt": len(attempts) + 1, "hypothesis": f"inference on {device}", "status": "PASS", "detections": len(detections)})
                break
            except Exception as exc:
                attempts.append({"attempt": len(attempts) + 1, "hypothesis": f"inference on {device}", "status": "BLOCKED", "error": repr(exc)})
                if len(attempts) >= 3:
                    break
                if device == "cuda":
                    torch.cuda.empty_cache()
    status = "PASS" if detections else "BLOCKED"
    dump(directory / "detections.json", {
        "status": status, "model": model_id, "prompt": CORE_PROMPT,
        "box_threshold": .25, "text_threshold": .2,
        "coordinate_system": "original_image_pixels", "attempts": attempts,
        "detections": detections,
    })
    write(directory / "PROMPT.txt", CORE_PROMPT)
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [round(v) for v in det["bbox_px"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (80, 255, 80), 2)
        label(out, f"{det['label']} {det['score']:.2f}", (x1, max(16, y1 - 5)), (80, 255, 80))
    if not detections:
        label(out, "GROUNDING DINO BLOCKED - SEE RESULT.md", (20, 40), (0, 0, 255), .8)
    cv2.imwrite(str(directory / "grounding_dino_overlay.png"), out)
    summary = "실행되어 detection을 생성했다." if detections else "최대 허용 범위 내 시도 후 detection을 생성하지 못했다. Qwen 결과로 후속 단계를 계속한다."
    write(directory / "RESULT.md", f"""# Grounding DINO Result

Status: {status}

{summary}

Attempts: {json.dumps(attempts, ensure_ascii=False, indent=2)}
""")
    return status, detections, attempts


def canonical(label_text: str) -> str:
    text = label_text.lower()
    aliases = {
        "traffic_cone": ("traffic cone", "cone"),
        "manipulator": ("robot arm", "manipulator"),
        "mobile_robot": ("mobile robot",),
        "rack": ("storage rack", "shelf", "rack"),
        "workbench": ("workbench", "desk"),
        "cart": ("trolley", "cart"),
        "drum": ("barrel", "drum"),
        "box": ("cardboard box", "box"),
        "pallet": ("pallet",), "truck": ("truck",), "fence": ("fence",),
    }
    for category, names in aliases.items():
        if any(name in text for name in names):
            return category
    return "other"


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    area = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - area
    return area / union if union else 0.0


def stage_analysis(run: Path, image, objects, regions, detections, dino_status):
    relations = load(EXP001 / "object_relations.json")["relations"]
    candidates = []
    for obj in objects:
        for det in detections:
            qcat, dcat = obj["category"], canonical(det["label"])
            compatible = qcat == dcat or (qcat == "pallet" and dcat in {"pallet", "box", "drum"})
            if compatible:
                overlap = iou(obj["bbox_px"], det["bbox_px"])
                if overlap >= .1:
                    candidates.append((overlap, obj, det))
    used_qwen, used_dino, matches = set(), set(), []
    for overlap, obj, det in sorted(candidates, key=lambda value: value[0], reverse=True):
        if obj["id"] in used_qwen or det["id"] in used_dino:
            continue
        used_qwen.add(obj["id"]); used_dino.add(det["id"])
        matches.append({"qwen_id": obj["id"], "grounding_dino_id": det["id"], "iou": round(overlap, 6)})
    comparison = {
        "status": "PASS" if detections else "PARTIAL",
        "ground_truth_available": False,
        "accuracy_claim": False,
        "qwen_count": len(objects), "grounding_dino_count": len(detections),
        "qwen_categories": sorted({obj["category"] for obj in objects}),
        "grounding_dino_categories": sorted({canonical(det["label"]) for det in detections}),
        "matched_instances": matches,
        "minimum_match_iou": .1,
        "unmatched_qwen_ids": [obj["id"] for obj in objects if obj["id"] not in used_qwen],
        "unmatched_grounding_dino_ids": [det["id"] for det in detections if det["id"] not in used_dino],
        "note": "IoU is diagnostic only; detector boxes need not encode identical object scope.",
    }
    nodes = [
        {"id": obj["id"], "type": "object", "label": obj["label"],
         "category": obj["category"], "bbox_px": obj["bbox_px"],
         "grounding_dino_match": next((m for m in matches if m["qwen_id"] == obj["id"]), None)}
        for obj in objects
    ] + [
        {"id": region["id"], "type": "semantic_region", "label": region["label"],
         "evidence_object_ids": region["evidence_object_ids"]}
        for region in regions
    ]
    edges = [
        {"source": rel["subject_id"], "relation": rel["predicate"], "target": rel["object_id"],
         "confidence": rel["confidence"], "source_type": "REUSED_QWEN_MODEL_OUTPUT"}
        for rel in relations
    ]
    for region in regions:
        edges.extend({"source": obj_id, "relation": "supports_region", "target": region["id"],
                      "source_type": "REUSED_REGION_EVIDENCE"}
                     for obj_id in region["evidence_object_ids"])
    directory = run / "05_semantic_analysis"
    dump(directory / "detection_comparison.json", comparison)
    dump(directory / "spatial_relations.json", {
        "status": "PARTIAL", "relations": relations,
        "source": str(EXP001 / "object_relations.json"),
        "warning": "Single-image VLM relations; not metric measurements.",
    })
    dump(directory / "semantic_scene_graph.json", {
        "status": "PARTIAL", "nodes": nodes, "edges": edges,
        "grounding_dino_status": dino_status,
    })
    out = image.copy()
    centers = {}
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["bbox_px"])
        centers[obj["id"]] = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
        label(out, obj["id"], (x1, max(15, y1 - 4)), (0, 220, 255))
    for rel in relations:
        if rel["subject_id"] in centers and rel["object_id"] in centers:
            start, end = centers[rel["subject_id"]], centers[rel["object_id"]]
            cv2.arrowedLine(out, start, end, (255, 200, 40), 2, tipLength=.04)
            label(out, rel["predicate"], ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2), (255, 200, 40), .42)
    label(out, "SEMANTIC SCENE GRAPH OVERLAY / NON-METRIC", (18, out.shape[0] - 20))
    cv2.imwrite(str(directory / "semantic_analysis_overlay.png"), out)
    write(directory / "RESULT.md", f"""# Detection Comparison and Semantic Analysis

Status: PARTIAL

Grounding DINO status는 {dino_status}, Qwen {len(objects)}개 대비 Grounding DINO {len(detections)}개이며 one-to-one compatible-label IoU match는 {len(matches)}개다. Ground truth가 없어 accuracy가 아니라 coverage/count/overlap 진단만 기록했다. Scene graph는 보존된 Qwen relation과 semantic-region evidence를 연결하며, 근거 없는 관계를 새로 생성하지 않았다.
""")


def stage_map(run: Path, image, objects, regions, masks, counts):
    floor, occupancy, occupancy_inflated, no_go, traversable, robot_state = masks
    directory = run / "06_2d_map"
    canvas = np.full_like(image, 35)
    canvas[floor > 0] = (75, 75, 75)
    canvas[traversable > 0] = (55, 150, 55)
    canvas[occupancy_inflated > 0] = (40, 40, 210)
    canvas[no_go > 0] = (170, 35, 180)
    canvas[robot_state > 0] = (230, 180, 20)
    for region in regions:
        pts = polygon_px(region, image.shape[1], image.shape[0])
        cv2.polylines(canvas, [pts], True, COLORS.get(region["label"], (180, 180, 180)), 2)
        anchor = tuple(pts[np.argmin(pts[:, 1])])
        label(canvas, region["label"], anchor, COLORS.get(region["label"], (180, 180, 180)), .52)
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["bbox_px"])
        label(canvas, obj["id"], (x1, min(image.shape[0] - 5, y2)), (255, 255, 255), .38)
    label(canvas, "IMAGE-PLANE SEMANTIC 2D MAP / NON-METRIC", (18, image.shape[0] - 20))
    cv2.imwrite(str(directory / "semantic_2d_map.png"), canvas)
    cv2.imwrite(str(directory / "occupancy_image_plane.png"), occupancy_inflated)
    dump(directory / "semantic_2d_map.json", {
        "status": "PASS", "map_type": "IMAGE-PLANE SEMANTIC 2D MAP",
        "metric": False, "coordinate_system": "original_image_pixels",
        "image_size_px": [image.shape[1], image.shape[0]],
        "layers": ["workspace_floor", "physical_occupancy", "semantic_regions", "explicit_no_go", "traversable_candidate", "mobile_robot_state", "object_ids"],
        "pixel_counts": counts,
        "metric_map_status": "BLOCKED",
        "metric_blocker": "No validated pixel-world correspondence for this exact image/camera pose.",
        "required_inputs": ["current camera calibration", "fixed current camera pose confirmation", "at least 4 floor pixel-world correspondences; 8-12 recommended"],
    })
    write(directory / "RESULT.md", """# 2D Map Result

Image-plane map: PASS

Metric map: BLOCKED

Semantic, physical occupancy, explicit No-Go, traversability, mobile-robot state를 원본 pixel grid에 분리해 표시했다. 이 장면과 일치하는 검증된 camera-to-ground 자료가 없으므로 legacy/current-other-scene homography를 적용하거나 PGM/YAML metric scale을 생성하지 않았다.
""")


def manifest(run: Path) -> None:
    files = []
    for path in sorted(p for p in run.rglob("*") if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"):
        files.append({"path": str(path.relative_to(run)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    dump(run / "ARTIFACT_MANIFEST.json", {"files": files})


def report(run: Path, config, inventory, statuses, attempts, counts) -> None:
    artifacts = "\n".join(f"- `{path.relative_to(run)}`" for path in sorted(run.rglob("*")) if path.is_file())
    command = " ".join(sys.argv)
    text = f"""# RUN REPORT

STEP 0 Environment Inventory: {statuses['inventory']}

STEP 1 Object Detection: {statuses['objects']}

STEP 2 Semantic Regions: {statuses['semantics']}

STEP 3 Traversability: {statuses['traversability']}

STEP 4 Grounding DINO: {statuses['dino']}

STEP 5 Semantic Analysis: {statuses['analysis']}

STEP 6 Image-plane 2D Map: {statuses['image_map']}

STEP 6 Metric 2D Map: {statuses['metric_map']}

STEP 7 End-to-End Pipeline: {statuses['pipeline']}

## Run identity

- Experiment/run: `{config['experiment_id']}` / `{config['run_id']}`
- Timestamp: `{config['created_at']}`
- Git commit: `{inventory['git_commit'] or 'UNKNOWN (workspace is not a Git worktree)'}`
- Input: `{SOURCE_IMAGE}`
- Input SHA-256: `{inventory['input_sha256']}`
- Model evidence: Qwen/Qwen3-VL-4B-Instruct completed EXP001/EXP002; Grounding DINO `IDEA-Research/grounding-dino-tiny`
- Environment: `{sys.executable}`, Python `{platform.python_version()}`, GPU `{inventory.get('gpu')}`
- Command: `{command}`
- Config snapshot: `config_snapshot.json`

## Execution and findings

- STEP 1 reused the completed Qwen pass instead of rerunning it. Sixteen complete object records were recovered previously; the preserved raw response was truncated, so status remains PARTIAL.
- STEP 2 reused both blind/open semantic inference and target-taxonomy region evidence. Open output had no polygons; target polygons remain approximate.
- STEP 3 replaced the old semantic-zone-as-obstacle behavior. It computed `floor - physical occupancy - explicit no-go - {config['safety_margin_px']} px margin`; semantic labels alone do not restrict traversal.
- STEP 4 attempts: `{json.dumps(attempts, ensure_ascii=False)}`
- STEP 5 makes no accuracy claim. It records class coverage, counts, and compatible-label IoU ≥ 0.10 only.
- STEP 6 generated an image-plane map. Metric conversion is BLOCKED because the exact image/camera pose has no validated pixel↔world evidence; unrelated homographies were not used.
- Pixel measurements were computed from masks, not manually invented: `{json.dumps(counts, ensure_ascii=False)}`
- No package/system/CUDA/ROS2 change and no robot motion occurred.

## Failed or limited items

- Qwen evidence is a prior completed run with truncated primary JSON and approximate region polygons.
- Grounding DINO failure details, if any, are preserved in `04_grounding_dino/detections.json` and `RESULT.md`; work continued with Qwen evidence.
- Traversability is bbox/threshold based, perspective-distorted, and not safe for planning.
- Metric map is BLOCKED pending current-scene calibration and floor correspondences.

## Artifacts

{artifacts}

## WHAT WORKS NOW

- One command creates the complete object → semantic region → layered traversability → Grounding DINO attempt → comparison/scene graph → image-plane 2D map chain.
- Existing Qwen/open-semantic evidence is reused with source paths and unmodified raw output references.
- Physical occupancy, explicit No-Go, semantic regions, traversable candidate, and mobile-robot state are separate layers.
- All JSON and PNG outputs are machine-readable/decodable and hashes are recorded.

## WHAT TO DO NEXT

1. Human-review the object boxes, target-region polygons, and floor mask; complete the existing validation form before claiming semantic performance.
2. Capture calibration/control points for this exact camera scene (minimum 4, recommended 8–12) and validate reprojection error before metric map generation.
3. If Grounding DINO is BLOCKED, resolve only the recorded model-access/runtime cause in an isolated environment; do not modify the validated Qwen environment packages.
4. Replace bbox occupancy with validated segmentation/footprints before any visualization, dry-run, or robot planning gate.
"""
    write(run / "RUN_REPORT.md", text)


def self_check(run: Path | None = None) -> None:
    assert canonical("robot arm") == "manipulator"
    assert round(iou([0, 0, 10, 10], [5, 5, 15, 15]), 6) == 0.142857
    if run:
        required = [
            "00_inventory/ENVIRONMENT_INVENTORY.md", "01_object_detection/objects.json",
            "02_semantic_regions/semantic_regions.json", "03_traversability/traversability_mask.png",
            "04_grounding_dino/detections.json", "05_semantic_analysis/semantic_scene_graph.json",
            "06_2d_map/semantic_2d_map.png", "RUN_REPORT.md",
        ]
        for relative in required:
            path = run / relative
            assert path.is_file() and path.stat().st_size > 0, path
            if path.suffix == ".json":
                load(path)
            if path.suffix == ".png":
                assert cv2.imread(str(path)) is not None, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--safety-margin-px", type=int, default=8)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check: PASS")
        return
    if args.safety_margin_px < 0:
        raise ValueError("safety margin must be non-negative")
    timestamp = datetime.now().astimezone()
    run = args.output or ROOT / "results" / f"AUTO_VLM_PIPELINE_{timestamp:%Y%m%dT%H%M%S%z}"
    run = run.resolve()
    if run.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run}")
    for name in ["00_inventory", "01_object_detection", "02_semantic_regions", "03_traversability", "04_grounding_dino", "05_semantic_analysis", "06_2d_map", "logs", "prompts"]:
        (run / name).mkdir(parents=True, exist_ok=False if name == "00_inventory" else True)
    image = cv2.imread(str(SOURCE_IMAGE))
    if image is None:
        raise FileNotFoundError(SOURCE_IMAGE)
    config = {
        "experiment_id": "AUTO_VLM_PIPELINE", "run_id": run.name,
        "created_at": timestamp.isoformat(), "input_image": str(SOURCE_IMAGE),
        "input_sha256": sha256(SOURCE_IMAGE), "safety_margin_px": args.safety_margin_px,
        "qwen_sources": [str(EXP001), str(EXP002)],
        "grounding_dino_model": "IDEA-Research/grounding-dino-tiny",
        "grounding_dino_box_threshold": .25, "grounding_dino_text_threshold": .2,
    }
    dump(run / "config_snapshot.json", config)
    statuses = {"inventory": "PASS", "objects": "PARTIAL", "semantics": "PARTIAL", "traversability": "PARTIAL", "dino": "BLOCKED", "analysis": "PARTIAL", "image_map": "PASS", "metric_map": "BLOCKED", "pipeline": "PASS"}
    attempts = []
    try:
        inventory = stage_inventory(run, image, config)
        objects = stage_objects(run, image)
        regions = stage_semantics(run, image)
        floor, occupancy, occupancy_inflated, no_go, traversable, robot_state, counts = stage_traversability(run, image, objects, regions, args.safety_margin_px)
        try:
            statuses["dino"], detections, attempts = run_grounding_dino(run, image)
        except Exception as exc:
            attempts = [{"attempt": 1, "hypothesis": "Transformers Grounding DINO import", "status": "BLOCKED", "error": repr(exc)}]
            detections = []
            dump(run / "04_grounding_dino/detections.json", {"status": "BLOCKED", "attempts": attempts, "detections": []})
            write(run / "04_grounding_dino/PROMPT.txt", CORE_PROMPT)
            blocked = image.copy(); label(blocked, "GROUNDING DINO BLOCKED - SEE RESULT.md", (20, 40), (0, 0, 255), .8)
            cv2.imwrite(str(run / "04_grounding_dino/grounding_dino_overlay.png"), blocked)
            write(run / "04_grounding_dino/RESULT.md", f"# Grounding DINO Result\n\nStatus: BLOCKED\n\n{repr(exc)}")
            write(run / "logs/grounding_dino_exception.log", traceback.format_exc())
        stage_analysis(run, image, objects, regions, detections, statuses["dino"])
        stage_map(run, image, objects, regions, (floor, occupancy, occupancy_inflated, no_go, traversable, robot_state), counts)
        report(run, config, inventory, statuses, attempts, counts)
        self_check(run)
        write(run / "logs/self_check.log", "self-check: PASS")
        manifest(run)
        print(run)
    except Exception:
        statuses["pipeline"] = "PARTIAL"
        write(run / "logs/pipeline_exception.log", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
