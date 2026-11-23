"""Perception: Qwen3-VL scene analysis, Grounding DINO candidates, Qwen3-VL crop verification,
Qwen grounding fallback, and IMAGE_DEMO manual objects.

Every detector box is a *candidate*. A candidate becomes an accepted object only when the
crop verification answer is "yes". Raw model replies are kept in every record.
"""
import json
from pathlib import Path

import cv2

from ..config import sha256_text
from ..contracts import (DETECTION_SCHEMA, MANUAL_TEST, OBSERVED, RAW_PX, SchemaError, parse_json_object,
                         validate)
from ..geometry.coordinate_frames import box_area, iou

SCENE_SCHEMA = {
    "type": "object",
    "required": ["scene_type", "observed_equipment", "transport_objects", "task_relevant_objects", "evidence",
                 "uncertain"],
    "properties": {"scene_type": {"type": "string", "minLength": 1},
                   **{k: {"type": "array", "items": {"type": "string"}} for k in
                      ("observed_equipment", "transport_objects", "task_relevant_objects", "evidence", "uncertain")}},
}
VERIFY_SCHEMA = {
    "type": "object", "required": ["answer"],
    "properties": {"answer": {"type": "string", "enum": ["yes", "no", "uncertain"]},
                   "observed_color": {"type": ["string", "null"]}, "observed_object": {"type": ["string", "null"]},
                   "reason": {"type": ["string", "null"]}},
}
GROUNDING_SCHEMA = {
    "type": "object", "required": ["objects"],
    "properties": {"objects": {"type": "array", "items": {
        "type": "object", "required": ["bbox_2d"],
        "properties": {"bbox_2d": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1000},
                                   "minItems": 4, "maxItems": 4}}}}},
}
SCORE_NOTE = "detector score; not a calibrated probability"


# ------------------------------------------------------------------ Qwen helpers
def qwen_json(client, mcfg, prompt_key, prompt_text, image_path, max_new_tokens, schema):
    """Run Qwen, parse + validate JSON. Returns (obj | None, record). Never raises on bad JSON."""
    client.ensure_loaded("qwen")
    r = client.call("qwen_generate", prompt=prompt_text, image_path=None if image_path is None else str(image_path),
                    max_new_tokens=max_new_tokens)
    rec = {"model": r["model"], "commit_hash": r["commit_hash"], "prompt_id": mcfg["prompts"][prompt_key]["id"],
           "prompt_sha256": sha256_text(prompt_text), "image_path": r["image_path"], "raw_response": r["text"],
           "latency_s": r["latency_s"], "n_output_tokens": r["n_output_tokens"],
           "hit_max_new_tokens": r["hit_max_new_tokens"], "peak_vram_gib": r.get("peak_allocated_gib"),
           "error": None}
    try:
        obj = validate(parse_json_object(r["text"]), schema)
    except (json.JSONDecodeError, SchemaError) as e:
        rec["error"] = f"INVALID_RESPONSE: {e}"
        return None, rec
    return obj, rec


def analyze_scene(client, mcfg, image_path, task_summary):
    prompt = mcfg["prompts"]["scene_analysis"]["text"].replace("{task_summary}", task_summary)
    obj, rec = qwen_json(client, mcfg, "scene_analysis", prompt, image_path,
                         mcfg["qwen"]["max_new_tokens"]["scene_analysis"], SCENE_SCHEMA)
    return {"status": "OK" if obj else "FAILED", "output": obj, "call": rec,
            "note": "free-text model description; not ground truth"}


# ------------------------------------------------------------------ detector candidates
def nms(cands, thr):
    kept = []
    for c in sorted(cands, key=lambda c: -(c["score"] if c["score"] is not None else -1)):
        if all(iou(c["bbox_xyxy"], k["bbox_xyxy"]) < thr for k in kept):
            kept.append(c)
    return kept


def dino_candidates(client, mcfg, image_path, image_size, prompts):
    d = mcfg["grounding_dino"]
    client.ensure_loaded("dino")
    r = client.call("dino_detect", image_path=str(image_path), prompts=list(prompts),
                    box_threshold=d["box_threshold"], text_threshold=d["text_threshold"])
    w, h = image_size
    cands = []
    for res in r["results"]:
        for b, s, ph in zip(res["boxes_xyxy"], res["scores"], res["phrases"]):
            b = [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])]
            if box_area(b) <= 0 or box_area(b) > d["max_area_frac"] * w * h:
                continue
            cands.append({"bbox_xyxy": b, "score": s, "prompt": res["prompt"], "phrase": ph,
                          "source": "grounding_dino"})
    return nms(cands, d["nms_iou"]), r


# ------------------------------------------------------------------ crop verification
def make_verify_crop(image, box, vcfg, out_path):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    pad = max(vcfg["verify_crop_pad_ratio"] * max(x2 - x1, y2 - y1), vcfg["verify_crop_min_px"] / 2)
    cx1, cy1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    cx2, cy2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
    crop = image[cy1:cy2, cx1:cx2].copy()
    scale = max(1.0, vcfg["verify_crop_upscale_px"] / max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    p1 = (round((x1 - cx1) * scale), round((y1 - cy1) * scale))
    p2 = (round((x2 - cx1) * scale), round((y2 - cy1) * scale))
    cv2.rectangle(crop, p1, p2, (0, 255, 0), 2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return {"path": str(out_path), "crop_xyxy_raw": [cx1, cy1, cx2, cy2], "scale": round(scale, 4),
            "marker": "green rectangle = candidate box"}


def verify_box(client, mcfg, image, box, description, crop_path):
    crop = make_verify_crop(image, box, mcfg["target_inspection"], crop_path)
    prompt = mcfg["prompts"]["verify_object"]["text"].replace("{description}", description)
    obj, rec = qwen_json(client, mcfg, "verify_object", prompt, crop["path"],
                         mcfg["qwen"]["max_new_tokens"]["verify"], VERIFY_SCHEMA)
    return {"status": obj["answer"] if obj else "invalid_response", "answer": obj, "description": description,
            "crop": crop, "call": rec}


def qwen_ground_boxes(client, mcfg, image_path, image_size, query):
    prompt = mcfg["prompts"]["grounding_fallback"]["text"].replace("{query}", query)
    obj, rec = qwen_json(client, mcfg, "grounding_fallback", prompt, image_path,
                         mcfg["qwen"]["max_new_tokens"]["grounding"], GROUNDING_SCHEMA)
    w, h = image_size
    boxes = []
    for o in (obj or {}).get("objects", []):
        b = o["bbox_2d"]
        if b[0] < b[2] and b[1] < b[3]:
            boxes.append({"bbox_xyxy": [b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h],
                          "bbox_norm1000": b, "score": None, "prompt": query, "source": "qwen3_vl_grounding"})
    return boxes, rec


# ------------------------------------------------------------------ detection records
def make_detection(object_id, class_name, cand, frame_id, model_name, model_version, knowledge, **extra):
    det = {"object_id": object_id, "class_name": class_name,
           "bbox_xyxy": [round(float(v), 2) for v in cand["bbox_xyxy"]], "coordinate_frame": RAW_PX,
           "score": cand.get("score"), "score_note": SCORE_NOTE if cand.get("score") is not None else None,
           "image_frame_id": frame_id, "model_name": model_name, "model_version": model_version,
           "prompt": cand.get("prompt"), "source": cand["source"], "knowledge": knowledge, **extra}
    return validate(det, DETECTION_SCHEMA)


def detect_reference_objects(client, mcfg, image, image_path, frame_id, classes, extra_candidate_classes,
                             crop_dir, emit=lambda *a, **k: None):
    """classes: verified classes (conveyor/truck/mobile_robot). extra_candidate_classes: detector-only
    candidate lists computed while the detector is loaded (e.g. red_cube, verified later)."""
    d = mcfg["grounding_dino"]
    h, w = image.shape[:2]
    raw, cand_by_class = {}, {}
    for cls in list(classes) + list(extra_candidate_classes):       # phase 1: detector (GPU: DINO)
        cands, r = dino_candidates(client, mcfg, image_path, (w, h), d["class_prompts"][cls])
        cand_by_class[cls], raw[cls] = cands, r
        emit("info", f"detector {cls}: {len(cands)} candidates")
    dino_info = client.loaded.get("dino", {})
    objects, fallbacks = [], {}
    for cls in classes:                                                # phase 2: verification (GPU: Qwen)
        top = cand_by_class[cls][:d["verify_top_k"][cls]]
        desc = mcfg["verification"]["descriptions"][cls]
        for i, c in enumerate(top):
            c["verification"] = verify_box(client, mcfg, image, c["bbox_xyxy"], desc,
                                           Path(crop_dir) / f"verify_{cls}_{i + 1}.png")
            emit("info", f"verify {cls} cand {i + 1}: {c['verification']['status']}")
        for c in cand_by_class[cls][len(top):]:
            c["verification"] = {"status": "not_run", "reason": f"outside verify_top_k={len(top)}"}
        if not any(c["verification"]["status"] == "yes" for c in top) and cls in d["qwen_fallback_classes"]:
            fb, rec = qwen_ground_boxes(client, mcfg, image_path, (w, h), d["class_prompts"][cls][0])
            fallbacks[cls] = rec
            for i, c in enumerate(fb):
                c["verification"] = verify_box(client, mcfg, image, c["bbox_xyxy"], desc,
                                               Path(crop_dir) / f"verify_{cls}_fallback_{i + 1}.png")
            cand_by_class[cls] = cand_by_class[cls] + fb
            emit("info", f"{cls}: no verified detector candidate -> Qwen grounding fallback ({len(fb)} boxes)")
        n_acc = 0
        for k, c in enumerate(cand_by_class[cls]):
            accepted = c["verification"]["status"] == "yes"
            n_acc += accepted
            oid = f"{cls}_{n_acc}" if accepted else f"{cls}_cand{k + 1}"
            is_dino = c["source"] == "grounding_dino"
            model = dino_info.get("model_id") if is_dino else c["verification"].get("call", {}).get("model")
            version = dino_info.get("commit_hash") if is_dino else c["verification"].get("call", {}).get("commit_hash")
            objects.append(make_detection(oid, cls, c, frame_id, model, version, OBSERVED, accepted=accepted,
                                          verification=c["verification"], phrase=c.get("phrase")))
    return {"objects": objects, "unverified_candidates": {c: cand_by_class[c] for c in extra_candidate_classes},
            "detector_raw": raw, "qwen_fallback_calls": fallbacks,
            "detector": {"model": dino_info.get("model_id"), "commit_hash": dino_info.get("commit_hash"),
                         "box_threshold": d["box_threshold"], "text_threshold": d["text_threshold"],
                         "max_area_frac": d["max_area_frac"], "nms_iou": d["nms_iou"]}}


def manual_objects(demo_inputs, frame_id):
    """IMAGE_DEMO: boxes typed in by a person. Marked manual_test_data, never counted as detections."""
    out = []
    counters = {}
    for o in demo_inputs.get("objects", []):
        counters[o["class_name"]] = counters.get(o["class_name"], 0) + 1
        out.append(make_detection(f"{o['class_name']}_{counters[o['class_name']]}", o["class_name"],
                                  {"bbox_xyxy": o["bbox_xyxy"], "score": None, "prompt": None,
                                   "source": "manual_test_data"}, frame_id, None, None, MANUAL_TEST,
                                  accepted=True, verification={"status": "not_run", "reason": "IMAGE_DEMO manual input"},
                                  provenance=o.get("provenance")))
    return out
