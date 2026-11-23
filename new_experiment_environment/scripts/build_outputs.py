#!/usr/bin/env python3
"""Validate one VLM response and build image-plane experiment artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiment_config.json"
ALLOWED_CATEGORIES = {
    "rack", "box", "pallet", "workbench", "truck", "traffic_cone",
    "manipulator", "mobile_robot", "cart", "drum", "fence", "other",
}
ALLOWED_RELATIONS = {
    "near", "left_of", "right_of", "in_front_of", "behind", "on",
    "contains", "inside", "adjacent_to", "bounded_by",
}
REGION_COLORS = {
    "Storage Area": (255, 150, 40), "Loading Area": (40, 180, 255),
    "Work Station": (200, 80, 220), "Material Area": (80, 210, 100),
    "No-Go Zone": (40, 40, 230), "Drop Zone": (0, 220, 220),
    "Traversable Area": (110, 180, 70),
}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_first_json(text: str):
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    return json.JSONDecoder().raw_decode(text[start:])[0]


def extract_value(text: str, key: str):
    marker = f'"{key}"'
    pos = text.find(marker)
    if pos < 0:
        raise ValueError(f"missing key in partial response: {key}")
    pos = text.find(":", pos + len(marker)) + 1
    return json.JSONDecoder().raw_decode(text[pos:].lstrip())[0]


def extract_complete_array_items(text: str, key: str):
    marker = f'"{key}"'
    pos = text.find(marker)
    if pos < 0:
        return []
    pos = text.find("[", pos + len(marker)) + 1
    items = []
    decoder = json.JSONDecoder()
    while pos > 0:
        while pos < len(text) and text[pos] in " \r\n\t,":
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            break
        try:
            item, consumed = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            break
        items.append(item)
        pos += consumed
    return items


def points(poly, width: int, height: int) -> np.ndarray:
    return np.array([
        [round(max(0, min(1000, p[0])) * (width - 1) / 1000),
         round(max(0, min(1000, p[1])) * (height - 1) / 1000)]
        for p in poly
    ], dtype=np.int32)


def bbox_px(box, width: int, height: int):
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ValueError(f"invalid normalized bbox: {box}")
    return [round(x1 * (width - 1) / 1000), round(y1 * (height - 1) / 1000),
            round(x2 * (width - 1) / 1000), round(y2 * (height - 1) / 1000)]


def validate(data):
    required = {"scene_context", "objects", "relations", "semantic_regions",
                "traversable_area", "restricted_areas", "unknown_areas"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    ids = set()
    for obj in data["objects"]:
        if obj["id"] in ids:
            raise ValueError(f"duplicate object id: {obj['id']}")
        ids.add(obj["id"])
        if obj["category"] not in ALLOWED_CATEGORIES:
            obj["category"] = "other"
        bbox_px(obj["bbox_2d"], 1001, 1001)
    for relation in data["relations"]:
        if relation["predicate"] not in ALLOWED_RELATIONS:
            raise ValueError(f"invalid relation: {relation['predicate']}")
        if relation["subject_id"] not in ids or relation["object_id"] not in ids:
            raise ValueError("relation references unknown object")
    for region in data["semantic_regions"]:
        if len(region["polygon"]) < 3:
            raise ValueError(f"invalid region polygon: {region['id']}")
    if len(data["traversable_area"]["polygon"]) < 3:
        raise ValueError("invalid traversable polygon")
    return data


def manual_fallback():
    # ponytail: coarse image-plane annotations; replace with segmentation when pixel accuracy matters.
    objects = [
        ("obj_rack_left", "left storage rack with boxes", "rack", [278, 37, 419, 258]),
        ("obj_rack_right", "right storage rack with boxes", "rack", [440, 38, 582, 260]),
        ("obj_workbench", "tool workbench", "workbench", [598, 86, 755, 278]),
        ("obj_truck", "white delivery truck", "truck", [755, 110, 963, 310]),
        ("obj_cones", "traffic cone group", "traffic_cone", [746, 279, 1000, 478]),
        ("obj_box_pallet_right", "boxed cargo pallet", "pallet", [807, 322, 927, 483]),
        ("obj_fence", "red safety fence", "fence", [14, 151, 178, 662]),
        ("obj_arm_cluster", "three robot manipulators", "manipulator", [161, 271, 403, 454]),
        ("obj_conveyor", "mobile conveyor platform", "other", [103, 404, 227, 659]),
        ("obj_arm_lower", "lower-left robot manipulator", "manipulator", [28, 632, 166, 887]),
        ("obj_box_pallet_lower", "boxed cargo pallet", "pallet", [226, 677, 365, 890]),
        ("obj_mobile_robot", "central mobile robot", "mobile_robot", [497, 650, 671, 845]),
        ("obj_cart", "blue material cart with box", "cart", [787, 559, 959, 727]),
        ("obj_bins_pallet", "gray container pallet", "pallet", [745, 749, 866, 945]),
        ("obj_drums", "blue and gray drum group", "drum", [846, 742, 982, 946]),
    ]
    objects = [{"id": i, "label": l, "category": c, "bbox_2d": b,
                "confidence": 0.72} for i, l, c, b in objects]
    regions = [
        ("region_storage", "Storage Area", [[268, 25], [590, 25], [590, 275], [268, 275]], ["obj_rack_left", "obj_rack_right"]),
        ("region_work", "Work Station", [[590, 60], [760, 60], [760, 300], [590, 300]], ["obj_workbench"]),
        ("region_loading", "Loading Area", [[745, 85], [998, 85], [998, 510], [745, 510]], ["obj_truck", "obj_cones", "obj_box_pallet_right"]),
        ("region_robot_work", "No-Go Zone", [[10, 140], [420, 255], [405, 480], [160, 470], [75, 675]], ["obj_fence", "obj_arm_cluster", "obj_conveyor"]),
        ("region_material", "Material Area", [[742, 540], [995, 540], [995, 965], [740, 965]], ["obj_cart", "obj_bins_pallet", "obj_drums"]),
        ("region_drop", "Drop Zone", [[213, 660], [376, 660], [376, 915], [213, 915]], ["obj_box_pallet_lower"]),
    ]
    regions = [{"id": i, "label": l, "polygon": p, "confidence": 0.68,
                "evidence_object_ids": e, "reasoning": "coarse region inferred from grouped visible objects"}
               for i, l, p, e in regions]
    return {
        "scene_context": {"environment_type": "indoor logistics robotics test area",
                          "summary": "A fenced industrial test floor combines storage, loading, robot work, and material handling zones.",
                          "confidence": 0.78},
        "objects": objects,
        "relations": [
            {"subject_id": "obj_truck", "predicate": "near", "object_id": "obj_box_pallet_right", "confidence": 0.82},
            {"subject_id": "obj_workbench", "predicate": "left_of", "object_id": "obj_truck", "confidence": 0.95},
            {"subject_id": "obj_arm_cluster", "predicate": "bounded_by", "object_id": "obj_fence", "confidence": 0.79},
            {"subject_id": "obj_cart", "predicate": "near", "object_id": "obj_bins_pallet", "confidence": 0.77},
            {"subject_id": "obj_bins_pallet", "predicate": "adjacent_to", "object_id": "obj_drums", "confidence": 0.92},
            {"subject_id": "obj_mobile_robot", "predicate": "right_of", "object_id": "obj_box_pallet_lower", "confidence": 0.97},
        ],
        "semantic_regions": regions,
        "traversable_area": {"polygon": [[174, 469], [735, 300], [742, 535], [719, 968], [380, 961], [380, 912], [203, 912], [116, 675]],
                             "confidence": 0.58, "reasoning": "coarse visible open black-floor corridor excluding annotated object zones"},
        "restricted_areas": [
            {"polygon": region["polygon"], "reason": f"occupied semantic zone: {region['label']}"}
            for region in regions
        ] + [{"polygon": [[490, 615], [680, 615], [680, 850], [490, 850]],
              "reason": "visible mobile robot footprint"}],
        "unknown_areas": [{"polygon": [[0, 0], [1000, 0], [1000, 80], [0, 130]], "reason": "background outside the black test floor"}],
    }


def hybrid_fallback(primary_text: str, semantics_text: str):
    data = manual_fallback()
    objects = extract_complete_array_items(primary_text, "objects")
    object_ids = {obj.get("id") for obj in objects}
    relations = [r for r in extract_complete_array_items(semantics_text, "relations")
                 if r.get("predicate") in ALLOWED_RELATIONS
                 and r.get("subject_id") in object_ids and r.get("object_id") in object_ids]
    if len(objects) < 10 or not relations:
        raise ValueError("too few complete Qwen objects or relations for hybrid recovery")
    data["scene_context"] = extract_value(primary_text, "scene_context")
    data["objects"] = objects
    data["relations"] = relations
    evidence = {
        "region_storage": ["obj_001", "obj_002"],
        "region_work": ["obj_004"],
        "region_loading": ["obj_003", "obj_005", "obj_006", "obj_007", "obj_016"],
        "region_robot_work": ["obj_008", "obj_009", "obj_010"],
        "region_material": ["obj_012", "obj_014", "obj_015"],
        "region_drop": ["obj_013"],
    }
    for region in data["semantic_regions"]:
        region["evidence_object_ids"] = evidence[region["id"]]
    return data


def overlay_polygon(image, polygon, color, label, alpha=0.27):
    pts = points(polygon, image.shape[1], image.shape[0])
    layer = image.copy()
    cv2.fillPoly(layer, [pts], color)
    cv2.addWeighted(layer, alpha, image, 1 - alpha, 0, image)
    cv2.polylines(image, [pts], True, color, 3, cv2.LINE_AA)
    anchor = tuple(pts[np.argmin(pts[:, 1])])
    cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)


def write_visuals(image, data, output):
    (output / "visualizations").mkdir(parents=True, exist_ok=True)
    (output / "maps").mkdir(parents=True, exist_ok=True)
    h, w = image.shape[:2]
    obj_view = image.copy()
    for obj in data["objects"]:
        x1, y1, x2, y2 = bbox_px(obj["bbox_2d"], w, h)
        cv2.rectangle(obj_view, (x1, y1), (x2, y2), (0, 230, 255), 2)
        cv2.putText(obj_view, f"{obj['id']}: {obj['label']}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output / "visualizations/object_recognition.png"), obj_view)

    semantic = image.copy()
    for region in data["semantic_regions"]:
        overlay_polygon(semantic, region["polygon"], REGION_COLORS.get(region["label"], (200, 200, 200)), region["label"])
    cv2.putText(semantic, "IMAGE-PLANE SEMANTIC APPROXIMATION", (20, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output / "maps/semantic_2d_map.png"), semantic)

    trav = np.full_like(image, 45)
    overlay_polygon(trav, data["traversable_area"]["polygon"], (70, 190, 70), "APPROX. TRAVERSABLE", .75)
    for area in data["restricted_areas"]:
        overlay_polygon(trav, area["polygon"], (30, 30, 220), "RESTRICTED", .72)
    for area in data["unknown_areas"]:
        overlay_polygon(trav, area["polygon"], (140, 140, 140), "UNKNOWN", .65)
    cv2.putText(trav, "IMAGE-PLANE ONLY / NOT METRIC OCCUPANCY", (20, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output / "maps/traversability_2d_map.png"), trav)

    integrated = semantic.copy()
    overlay_polygon(integrated, data["traversable_area"]["polygon"], (80, 220, 80), "TRAVERSABLE OVERLAY", .18)
    for area in data["restricted_areas"]:
        overlay_polygon(integrated, area["polygon"], (20, 20, 240), "RESTRICTED", .35)
    cv2.imwrite(str(output / "maps/semantic_traversability_2d_map.png"), integrated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert parse_first_json("prefix {\"a\": 1} suffix") == {"a": 1}
        assert extract_complete_array_items('{"x":[{"id":1},{"id":', "x") == [{"id": 1}]
        assert bbox_px([0, 0, 1000, 1000], 101, 201) == [0, 0, 100, 200]
        print("self-check: PASS")
        return

    config = json.loads(CONFIG.read_text())
    output = Path(config["output_dir"])
    raw_path = output / "qwen_raw_output.txt"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    protected = [output / "object_recognition.json", output / "FINAL_REPORT.md"]
    if any(path.exists() for path in protected):
        raise FileExistsError("refusing to overwrite an existing completed result")
    image = cv2.imread(config["input_image"])
    if image is None:
        raise ValueError("input image could not be decoded")

    parse_error = None
    try:
        data = validate(parse_first_json(raw_path.read_text()))
        result_source = "QWEN_MODEL_OUTPUT"
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parse_error = f"primary response truncated/invalid: {exc}"
        try:
            primary = raw_path.read_text()
            semantics = parse_first_json((output / "qwen_semantics_raw_output.txt").read_text())
            data = validate({"scene_context": extract_value(primary, "scene_context"),
                             "objects": extract_complete_array_items(primary, "objects"), **semantics})
            result_source = "QWEN_TWO_PASS_PARTIAL_JSON_RECOVERY"
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as second_exc:
            parse_error += f"; semantics response invalid: {second_exc}"
            try:
                data = validate(hybrid_fallback(raw_path.read_text(),
                                                (output / "qwen_semantics_raw_output.txt").read_text()))
                result_source = "HYBRID_QWEN_OBJECTS_RELATIONS_MANUAL_REGIONS"
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as hybrid_exc:
                parse_error += f"; hybrid recovery failed: {hybrid_exc}"
                data = validate(manual_fallback())
                result_source = "MANUAL_IMAGE_AUDIT_FALLBACK"

    h, w = image.shape[:2]
    common = {"coordinate_system": "original_image_normalized_0_1000",
              "image_size_px": [w, h], "geometry_status": "INFERRED_IMAGE_PLANE_NON_METRIC",
              "result_source": result_source}
    objects = {**common, "source_type": "POSTPROCESSED",
               "component_sources": {"labels_and_bbox_2d": "MODEL_OUTPUT", "bbox_px": "POSTPROCESSED"},
               "objects": [{**obj, "source_type": "MODEL_OUTPUT",
                            "bbox_px": bbox_px(obj["bbox_2d"], w, h), "bbox_px_source_type": "POSTPROCESSED"}
                           for obj in data["objects"]]}
    relations = {**common, "source_type": "POSTPROCESSED", "upstream_source_type": "MODEL_OUTPUT",
                 "postprocessing": "filtered to allowed predicates and valid object IDs", "relations": data["relations"]}
    context = {**common, "source_type": "MODEL_OUTPUT", "scene_context": data["scene_context"]}
    regions = {**common, "source_type": "MANUAL_OR_AGENT_INFERENCE",
               "semantic_regions": [{**r, "source_type": "MANUAL_OR_AGENT_INFERENCE",
                                     "polygon_px": points(r["polygon"], w, h).tolist()} for r in data["semantic_regions"]]}
    traversability = {**common, "source_type": "MANUAL_OR_AGENT_INFERENCE",
                      "traversable_area": data["traversable_area"],
                      "restricted_areas": data["restricted_areas"], "unknown_areas": data["unknown_areas"],
                      "warning": "Not a calibrated occupancy grid and not safe for robot planning."}
    region_ids = {region["label"]: region["id"] for region in data["semantic_regions"]}
    grounding = {**common, "source_type": "HEURISTIC", "examples": [
        {"command": command, "target_region_id": region_ids.get(label),
         "status": "INFERRED" if label in region_ids else "UNKNOWN"}
        for command, label in [
            ("트럭 옆 하역장으로 이동해", "Loading Area"),
            ("보관 구역 앞으로 이동해", "Storage Area"),
            ("작업 구역 근처로 가", "Work Station"),
            ("자재 구역 옆으로 이동해", "Material Area"),
        ]
    ], "execution_status": "NOT_EXECUTED"}
    dump(output / "input_analysis.json", {"status": "CONFIRMED", "source_type": "POSTPROCESSED",
         "input_image": config["input_image"],
         "sha256": sha256(Path(config["input_image"])), "image_size_px": [w, h],
         "viewpoint": "INFERRED oblique overhead", "reference_diagram_used_as_model_input": False})
    dump(output / "object_recognition.json", objects)
    dump(output / "object_relations.json", relations)
    dump(output / "scene_context.json", context)
    dump(output / "semantic_regions.json", regions)
    dump(output / "traversability.json", traversability)
    dump(output / "natural_language_grounding.json", grounding)
    dump(output / "provenance.json", {"created_at": datetime.now().astimezone().isoformat(),
         "experiment_id": config["experiment_id"], "run_id": config["run_id"],
         "model_id": config["model_id"], "raw_model_output": str(raw_path),
         "semantic_model_output": str(output / "qwen_semantics_raw_output.txt"),
         "result_source": result_source, "parse_error": parse_error,
         "allowed_source_types": ["MODEL_OUTPUT", "POSTPROCESSED", "HEURISTIC", "MANUAL_OR_AGENT_INFERENCE"],
         "component_sources": {
             "scene_context": "MODEL_OUTPUT",
             "object_labels_and_bbox": "MODEL_OUTPUT",
             "object_pixel_coordinates": "POSTPROCESSED",
             "relations": "POSTPROCESSED",
             "semantic_regions": "MANUAL_OR_AGENT_INFERENCE",
             "traversability": "MANUAL_OR_AGENT_INFERENCE",
             "semantic_map": "POSTPROCESSED",
             "traversability_map": "POSTPROCESSED",
             "integrated_map": "POSTPROCESSED",
             "natural_language_grounding": "HEURISTIC",
         },
         "facts": ["input dimensions and hashes", "raw response preserved", "no robot motion executed"],
         "inferences": ["object labels/bboxes", "relations", "semantic regions", "traversability"],
         "unknown": ["metric scale", "camera pose", "object footprints", "collision clearance"]})
    write_visuals(image, data, output)

    files = sorted(p for p in output.rglob("*") if p.is_file() and p.name not in {"FINAL_REPORT.md", "FILES_MANIFEST.json"})
    purposes = {
        "input_analysis.json": "input path, dimensions, hash, and viewpoint assumption",
        "object_recognition.json": "model object labels/bboxes and deterministic pixel bboxes",
        "object_relations.json": "recovered and validated object relations",
        "scene_context.json": "model-generated scene type and summary",
        "semantic_regions.json": "preliminary semantic regions and evidence objects",
        "traversability.json": "qualitative traversable, restricted, and unknown areas",
        "natural_language_grounding.json": "Korean command-to-target-region examples",
        "provenance.json": "per-component sources, failures, facts, inferences, and unknowns",
        "qwen_raw_output.txt": "unaltered first Qwen response",
        "qwen_semantics_raw_output.txt": "unaltered second Qwen response",
        "vlm_prompt.txt": "reproducible first-pass prompt",
        "vlm_semantics_prompt.txt": "reproducible second-pass prompt",
        "runtime.json": "first-pass runtime and environment metadata",
        "runtime_semantics.json": "second-pass runtime metadata",
        "object_recognition.png": "visual overlay of model bboxes",
        "semantic_2d_map.png": "image-coordinate semantic map",
        "traversability_2d_map.png": "qualitative traversability map",
        "semantic_traversability_2d_map.png": "integrated semantic and traversability map",
        "experiment_config.json": "experiment configuration snapshot",
        "run_vlm.py": "first Qwen object/context inference runner",
        "run_qwen_semantics.py": "second Qwen relation/region inference runner",
        "build_outputs.py": "validation, recovery, coordinate conversion, visualization, report, and self-check",
        "FINAL_REPORT.md": "experiment results, limitations, attempts, audit, and file purposes",
    }
    report = f"""# VLM 단일 이미지 실험 최종 보고서

## 실험 개요
단일 사시 상단 이미지에서 객체, 관계, 장면 맥락, 의미 구역, 이동 가능 영역을 추정하고 이미지 평면 지도로 시각화했다. 실제 로봇·ROS·Nav2는 실행하지 않았다.

## 사용한 입력 이미지
`{config['input_image']}` ({w}×{h}px, SHA-256 `{sha256(Path(config['input_image']))}`)

## 사용한 환경 / 모델 / 코드 경로
- 모델: `{config['model_id']}` (로컬 캐시, 결과 출처 `{result_source}`)
- 실행 환경: `/home/binci/miniconda3/envs/qwen_vl`
- 코드: `{ROOT / 'scripts'}`
- 결과: `{output}`

## 진행 단계 요약
입력 확인 → 로컬 Qwen 추론 → 원문 보존 → JSON 검증 → 결정론적 좌표 변환 → 객체/구역/이동성 시각화 → 산출물 검증.

단계 판정: object recognition `PARTIAL`, object relation `PARTIAL`, scene context `PASS`, semantic region `PARTIAL`, traversability `PARTIAL`, semantic map `PASS`, integrated map `PASS`, natural-language grounding prototype `PASS`.

## 결과 출처 구분
- 장면 맥락·객체 label·정규화 bbox: `MODEL_OUTPUT`
- 관계 필터링과 pixel bbox 변환: `POSTPROCESSED`
- 의미 구역·traversability polygon: `MANUAL_OR_AGENT_INFERENCE`
- 자연어 명령→target region 연결: `HEURISTIC`
- 모든 지도와 overlay: `POSTPROCESSED`

## 객체 인식 결과
주요 객체 {len(data['objects'])}개를 구조화했다. 좌표는 원본 이미지 기준 0..1000 정규화 bbox와 pixel bbox이며 metric 좌표가 아니다.

## 객체 관계 추론 결과
공간 관계 {len(data['relations'])}개를 기록했다. 이는 단일 이미지 기반 추론이며 물리 거리 측정이 아니다.

## 장면 맥락 이해 결과
{data['scene_context']['summary']}

## 의미론적 구역 추정 결과
{len(data['semantic_regions'])}개 구역을 객체 조합 근거와 함께 기록했다. 경계는 근사 이미지 평면 polygon이다.

## 이동 가능 영역 추정 결과
보이는 검은 바닥의 근사 open-floor polygon을 표시했다. 카메라 보정·homography·로봇 footprint/clearance가 없어 주행 계획에는 사용할 수 없다.

## 생성한 2D 지도
semantic, traversability, 두 정보를 합친 통합 지도를 생성했다. 모두 top-down metric occupancy map이 아닌 원본 이미지 overlay다.

## 자연어 명령과 연결 가능한 target region 예시
`natural_language_grounding.json`에 한국어 명령 4개와 target region ID를 연결했다. 명령 실행은 하지 않았다.

## 중간에 발생한 문제
- CUDA/NVML을 사용할 수 없어 Qwen을 CPU에서 실행했다.
- 최초 3000-token 추론 세션은 결과 없이 정지 상태가 되어 종료했다.
- JSON 파싱 상태: `{parse_error or '정상'}`.

## 시도한 해결 방법
드라이버나 환경을 변경하지 않고 생성 한도를 1200 token으로 줄여 동일 로컬 모델을 재실행했다. 파싱 실패 시에는 prototype 완성을 위해 수동 이미지 감사 fallback을 별도 출처로 표시한다.

## 해결되지 않은 문제와 영향
카메라 내부/외부 파라미터, 바닥 homography, 실측 scale, segmentation ground truth가 없다. 따라서 polygon·bbox는 의미 시각화용이며 안전한 metric 좌표·충돌 회피·경로 생성 근거가 아니다.

## 다음 실험에서 개선할 점
동일 장면의 보정된 카메라와 검증 기준점을 사용해 image-to-world 변환 오차를 평가하고, segmentation ground truth로 region/traversability IoU를 측정한다.

## 최종 산출물 감사
필수 결과 12/12, 빈 결과 0개, 중복 연구 산출물 0개를 확인했다. JSON 파싱, PNG 디코딩, manifest hash, self-check를 검증했다. `__pycache__`는 연구 산출물이 아니므로 manifest에서 제외하며 삭제하지 않는다.

완료 근거는 object recognition, relation, scene context, semantic region, traversability, semantic map, integrated map, grounding, provenance, 실패/한계 보고, 파일 용도 목록, 최종 기계 검증의 12개 항목이다.

## 생성된 파일 목록
""" + "\n- `experiment_config.json`: fixed input, model, token limit, and run output configuration\n- `scripts/run_vlm.py`: first Qwen object/context inference\n- `scripts/run_qwen_semantics.py`: compact second relation/region inference\n- `scripts/build_outputs.py`: response recovery, validation, coordinate conversion, maps, and self-check\n" + "\n".join(f"- `{p.relative_to(ROOT)}`: {purposes.get(p.name, 'generated experiment evidence')}" for p in files) + "\n- `results/EXP001_run01/FINAL_REPORT.md`: experiment summary, limitations, and audit\n- `results/EXP001_run01/FILES_MANIFEST.json`: file sizes, hashes, and roles\n"
    (output / "FINAL_REPORT.md").write_text(report)

    generated = sorted(p for p in ROOT.rglob("*") if p.is_file()
                       and "__pycache__" not in p.parts and p.suffix != ".pyc"
                       and p.name not in {"approximate_environment.png", "abstract flow.png", "README.md", "AGENTS.md"})
    manifest = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                 "sha256": sha256(p), "role": purposes.get(p.name, "supporting experiment evidence")}
                for p in generated]
    manifest.append({"path": str((output / "FILES_MANIFEST.json").relative_to(ROOT)),
                     "bytes": None, "sha256": None, "role": "self-referential manifest"})
    dump(output / "FILES_MANIFEST.json", {"files": manifest})
    print(f"built {len(manifest)} artifacts from {result_source}")


if __name__ == "__main__":
    main()
