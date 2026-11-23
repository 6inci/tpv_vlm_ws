#!/usr/bin/env python3
"""Task-conditioned placement target discovery and grounding on E1 (Exp 1-5, see config.yaml).

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
  $PY experiments/placement_target_grounding/placement_target.py --dry-run   # crops, prompts, offline checks only; NO model call
  $PY experiments/placement_target_grounding/placement_target.py             # Qwen (2 + at most 1 calls) and Grounding DINO, sequential

Order inside one run: Exp 1 Qwen -> Exp 2 DINO -> Exp 3 selection (+ at most one Qwen comparison call) -> Exp 4 route (offline) ->
evaluate. evaluation_reference.yaml is opened only in evaluate(), after every model output and decision has been written to disk.
Reused unchanged: v0.2 ModelWorkerClient / model worker, parse_json_object, validate, iou, box_area; task_route_integration build_context,
layers, draw_path; task_plan_prototype RectNavMap / approach rule; semantic_navigation_integration leg_to_region / leg_record.
"""
import argparse
import hashlib
import json
import shutil
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent / "task_route_integration"))
sys.path.insert(0, str(HERE.parent / "semantic_navigation_integration"))
import route_integration as RI  # noqa: E402
import run_integration as SN  # noqa: E402
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file, sha256_text  # noqa: E402
from semantic_task_nav.contracts import SchemaError, parse_json_object, validate  # noqa: E402
from semantic_task_nav.geometry.coordinate_frames import box_area, iou  # noqa: E402

PT, FN = RI.PT, RI.FN
_S, _SN = {"type": "string"}, {"type": ["string", "null"]}
OPEN_SCHEMA = {"type": "object", "required": ["candidates", "selected", "decision"],
               "properties": {"candidates": {"type": "array", "items": {"type": "object", "required": ["name_en", "suitability"],
                                                                        "properties": {"name_en": _S, "name_ko": _SN, "location_in_image": _SN, "reason": _SN,
                                                                                       "suitability": {"type": "string", "enum": ["suitable", "possible", "unsuitable"]},
                                                                                       "bbox_2d": {"type": ["array", "null"]}}}},
                              "selected": _SN, "decision": {"type": "string", "enum": ["selected", "cannot_determine"]}}}
COMPARE_SCHEMA = {"type": "object", "required": ["candidates", "best"],
                  "properties": {"candidates": {"type": "array", "items": {"type": "object", "required": ["id", "suitable"],
                                                                           "properties": {"id": {"type": "number"}, "observed_object": _SN, "reason": _SN,
                                                                                          "suitable": {"type": "string", "enum": ["yes", "no", "uncertain"]}}}},
                                 "best": {"type": ["number", "null"]}, "reason": _SN}}
DECISIONS = ("SELECTED", "AMBIGUOUS", "NOT_FOUND")
CAND_STATES = ("DETECTED", "SEMANTICALLY_RELEVANT", "SELECTED", "VERIFY_REQUIRED")


# ================================================================================ pure functions (tested offline)
def zone_crop_rect(area_box, img_wh, zc):
    """Area box expanded by max(ratio * longer side, min_px) on every side, clipped. Integer pixel rectangle [x0, y0, x1, y1)."""
    w, h = img_wh
    m = max(zc["margin_ratio"] * max(area_box[2] - area_box[0], area_box[3] - area_box[1]), zc["margin_min_px"])
    return [int(max(0, np.floor(area_box[0] - m))), int(max(0, np.floor(area_box[1] - m))), int(min(w, np.ceil(area_box[2] + m))), int(min(h, np.ceil(area_box[3] + m)))]


def parse_reply(text, schema):
    try:
        return validate(parse_json_object(text), schema), None
    except (json.JSONDecodeError, SchemaError) as e:
        return None, f"INVALID_RESPONSE: {e}"


def qwen_box_to_image(b, rect):
    """Qwen bbox_2d (0-1000, relative to the image it saw) -> image px. rect = the image region shown (full image or crop); scaling cancels."""
    if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)) or not (b[0] < b[2] and b[1] < b[3]):
        return None
    x0, y0, x1, y1 = rect
    return [round(x0 + b[0] / 1000 * (x1 - x0), 1), round(y0 + b[1] / 1000 * (y1 - y0), 1), round(x0 + b[2] / 1000 * (x1 - x0), 1), round(y0 + b[3] / 1000 * (y1 - y0), 1)]


def norm_phrase(s, max_words):
    words = str(s).lower().strip().rstrip(".").split()
    return " ".join(words[:max_words])


def dino_phrases(responses, max_words):
    """responses: {cond: parsed obj or None}. -> ordered phrases and their origins (which Qwen condition named them, how)."""
    origins = {}
    for cond, obj in responses.items():
        if not obj:
            continue
        for i, c in enumerate(obj["candidates"]):
            p = norm_phrase(c["name_en"], max_words)
            if p:
                origins.setdefault(p, []).append({"condition": cond, "candidate_index": i, "suitability": c["suitability"], "selected": False})
        if obj.get("selected"):
            p = norm_phrase(obj["selected"], max_words)
            if p:
                hit = [o for o in origins.get(p, []) if o["condition"] == cond]
                for o in hit:
                    o["selected"] = obj["decision"] == "selected"
                if not hit:
                    origins.setdefault(p, []).append({"condition": cond, "candidate_index": None, "suitability": None, "selected": obj["decision"] == "selected"})
    return list(origins), origins


def detections(raw, cond, rect, scale, max_area_frac):
    """Worker dino_detect output (boxes in the searched image) -> records in image px. Nothing is dropped: oversize boxes get a status."""
    sw, sh = raw["image_size"]
    out = []
    for res in raw["results"]:
        for b, s, ph in zip(res["boxes_xyxy"], res["scores"], res["phrases"]):
            bb = [max(0.0, b[0]), max(0.0, b[1]), min(float(sw), b[2]), min(float(sh), b[3])]
            img = [round(rect[0] + bb[0] / scale, 2), round(rect[1] + bb[1] / scale, 2), round(rect[0] + bb[2] / scale, 2), round(rect[1] + bb[3] / scale, 2)]
            out.append({"condition": cond, "prompt": res["prompt"].rstrip("."), "label": ph, "score": s, "bbox_searched": [round(v, 2) for v in bb], "bbox_xyxy": img,
                        "status": "FILTERED_OVERSIZE" if box_area(bb) > max_area_frac * sw * sh else ("EMPTY" if box_area(bb) <= 0 else "KEPT")})
    out.sort(key=lambda d: -d["score"])
    for k, d in enumerate(out, 1):
        d["rank_in_condition"] = k
        d["id"] = f"{cond}_{k}"
    return out


def merge(dets, thr):
    """Greedy IoU-only clustering of KEPT boxes over all conditions (highest score first). No containment rule."""
    clusters = []
    for d in sorted([d for d in dets if d["status"] == "KEPT"], key=lambda d: -d["score"]):
        for c in clusters:
            if iou(d["bbox_xyxy"], c["bbox_xyxy"]) >= thr:
                c["members"].append(d)
                d["merged_into"] = c["id"]
                break
        else:
            c = {"id": f"m{len(clusters) + 1}", "bbox_xyxy": d["bbox_xyxy"], "score": d["score"], "members": [d]}
            d["merged_into"] = c["id"]
            clusters.append(c)
    for k, c in enumerate(clusters, 1):
        c["rank"] = k
        c["phrases"] = sorted({m["prompt"] for m in c["members"]})
        c["conditions"] = sorted({m["condition"] for m in c["members"]})
        c["member_ids"] = [m["id"] for m in c["members"]]
        del c["members"]
    return clusters


def centre_in(box, rect):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return rect[0] <= cx < rect[2] and rect[1] <= cy < rect[3]


def select(merged, origins, responses, qwen_boxes, ctx_rect, e3, max_words):
    """Exp 3 decision BEFORE any comparison call. Returns the decision record; 'needs_comparison' lists the shown candidates."""
    rel_suit = set(e3["relevant_suitability"])
    for c in merged:
        org = [o for p in c["phrases"] for o in origins.get(p, [])]
        c["in_context_roi"] = centre_in(c["bbox_xyxy"], ctx_rect)
        c["qwen_origins"] = org
        c["states"] = ["DETECTED"]
        if c["in_context_roi"] and any(o["suitability"] in rel_suit or o["selected"] for o in org):
            c["states"].append("SEMANTICALLY_RELEVANT")
    S = [c for c in merged if "SEMANTICALLY_RELEVANT" in c["states"]]
    sel_names = {cond: norm_phrase(o["selected"], max_words) for cond, o in responses.items() if o and o["decision"] == "selected" and o.get("selected")}
    rec = {"selected_names_by_condition": sel_names, "n_detected": len(merged), "n_relevant": len(S), "relevant_ids": [c["id"] for c in S],
           "status": None, "selected": None, "rule": None, "needs_comparison": None}
    if len(set(sel_names.values())) > 1:
        rec["rule"] = "1: conditions A and B selected different names -> comparison"
    elif sel_names:
        N = next(iter(sel_names.values()))
        SN_ = [c for c in S if N in c["phrases"]]
        qb = [b for b in qwen_boxes.get(N, []) if b]
        linked = [c for c in SN_ if any(iou(c["bbox_xyxy"], b) >= e3["qwen_bbox_link_iou"] for b in qb)]
        rec.update({"name": N, "S_N": [c["id"] for c in SN_], "qwen_boxes_for_name": qb, "linked_by_qwen_box": [c["id"] for c in linked]})
        if len(SN_) == 1 and len(linked) == 1:
            rec.update({"status": "SELECTED", "selected": SN_[0]["id"], "rule": "2: one detected box for the selected name, linked by the Qwen box"})
            return rec
        rec["rule"] = f"2 not met (|S_N| = {len(SN_)}, linked = {len(linked)}) -> comparison"
    else:
        rec["rule"] = "no name selected by Qwen -> comparison of the relevant candidates"
    if S:
        shown = sorted(S, key=lambda c: -c["score"])[:e3["comparison"]["max_shown"]]
        rec["needs_comparison"] = [c["id"] for c in shown]
        rec["not_shown_relevant"] = [c["id"] for c in S if c not in shown]
        return rec
    rec.update({"status": "NOT_FOUND" if sel_names else "AMBIGUOUS",
                "rule": "4: Qwen named a selection but no relevant box was detected" if sel_names else "4: Qwen could not determine and nothing relevant was detected"})
    return rec


def apply_comparison(rec, cmp_obj, err):
    shown = rec["needs_comparison"]
    rec["comparison"] = {"parsed": cmp_obj, "error": err}
    if cmp_obj is None:
        rec.update({"status": "AMBIGUOUS", "rule": rec["rule"] + "; comparison reply invalid"})
        return rec
    best = cmp_obj.get("best")
    verdict = {int(c["id"]): c for c in cmp_obj["candidates"] if isinstance(c.get("id"), (int, float))}
    if best is not None and float(best).is_integer() and 1 <= int(best) <= len(shown) and verdict.get(int(best), {}).get("suitable") == "yes":
        rec.update({"status": "SELECTED", "selected": shown[int(best) - 1], "rule": rec["rule"] + f"; comparison best = {int(best)} (suitable yes)"})
    else:
        rec.update({"status": "AMBIGUOUS", "rule": rec["rule"] + f"; comparison best = {best} (not a suitable shown number)"})
    return rec


# ================================================================================ drawing helpers
def _t(img, text, org, col=(255, 255, 255), sc=0.5):
    PT._t(img, text, org, col, sc)


def panel(h, lines, w=760):
    return RI.panel(h, lines, w)


def crop_view(img, rect, scale):
    x0, y0, x1, y1 = rect
    return cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def to_crop(b, rect, scale):
    return [int(round((b[0] - rect[0]) * scale)), int(round((b[1] - rect[1]) * scale)), int(round((b[2] - rect[0]) * scale)), int(round((b[3] - rect[1]) * scale))]


def comparison_image(img, rect, scale, cands, path):
    im = crop_view(img, rect, scale)
    for k, c in enumerate(cands, 1):
        b = to_crop(c["bbox_xyxy"], rect, scale)
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(str(k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x, y = b[0], max(th + 4, b[1] - 4)
        cv2.rectangle(im, (x, y - th - 4), (x + tw + 6, y + 2), (0, 0, 0), -1)
        cv2.putText(im, str(k), (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), im)


# ================================================================================ main
def qwen_call(client, prompt, image_path, max_new_tokens, stage, prompt_id, log):
    client.ensure_loaded("qwen")
    tw = time.perf_counter()
    r = client.call("qwen_generate", prompt=prompt, image_path=str(image_path), max_new_tokens=max_new_tokens)
    log.append({"stage": stage, "model": r["model"], "commit_hash": r["commit_hash"], "prompt_id": prompt_id, "prompt_sha256": sha256_text(prompt),
                "image_path": str(image_path), "image_sha256": sha256_file(image_path), "image_size": r["image_size"], "n_input_tokens": r["n_input_tokens"],
                "n_output_tokens": r["n_output_tokens"], "hit_max_new_tokens": r["hit_max_new_tokens"], "latency_s": r["latency_s"],
                "wall_s": round(time.perf_counter() - tw, 2), "peak_vram_gib": r.get("peak_allocated_gib"), "raw_response": r["text"]})
    return r["text"]


def dino_call(client, image_path, phrases, thr, stage, log):
    client.ensure_loaded("dino")
    tw = time.perf_counter()
    r = client.call("dino_detect", image_path=str(image_path), prompts=list(phrases), box_threshold=thr["box"], text_threshold=thr["text"])
    log.append({"stage": stage, "model": r["model"], "commit_hash": r["commit_hash"], "image_path": str(image_path), "image_sha256": sha256_file(image_path),
                "prompts": list(phrases), "thresholds": thr, "n_boxes": sum(len(x["boxes_xyxy"]) for x in r["results"]), "wall_s": round(time.perf_counter() - tw, 2),
                "peak_vram_gib": r.get("peak_allocated_gib")})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg, rcfg, gcfg, v02 = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8")), RI.load_cfg(), PT.load_cfg(), load_config()
    mcfg = v02["models"]
    assert cfg["command"] == rcfg["command"] and (HERE / cfg["route_config"]).resolve() == (RI.HERE / "config.yaml").resolve()
    assert cfg["exp2"]["thresholds"] == {"box": mcfg["grounding_dino"]["box_threshold"], "text": mcfg["grounding_dino"]["text_threshold"]}
    assert cfg["exp2"]["max_area_frac"] == mcfg["grounding_dino"]["max_area_frac"] and mcfg["qwen"]["do_sample"] is False
    img_path = Path(rcfg["image"])
    assert sha256_file(img_path) == rcfg["image_sha256"], "image differs from the declared one"
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    H, W = img.shape[:2]
    fp_before = code_fingerprint()["sha256"]
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_dryrun" if args.dry_run else "_ptg"))
    out.mkdir(parents=True, exist_ok=False)
    for d in ("inputs", "prompts", "qwen", "dino", "figures"):
        (out / d).mkdir()
    (out / "config_snapshot.yaml").write_text((HERE / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    ctx = RI.build_context(rcfg, gcfg, v02, img)
    assert ctx["ft"]["accepted"]

    # ---------------------------------------------------------------- inputs (offline)
    zc = cfg["exp1"]["zone_crop"]
    area_box = [round(v, 1) for v in ctx["st"]["qwen_box_px"]]
    assert ctx["st"]["task_analysis"]["destination_area_id"] == zc["source_area_id"]
    rect = zone_crop_rect(area_box, (W, H), zc)
    full_p, crop2_p, crop1_p = out / "inputs/full_E1.png", out / "inputs/zone_crop_x2.png", out / "inputs/zone_crop_x1.png"
    shutil.copy2(img_path, full_p)
    cv2.imwrite(str(crop2_p), crop_view(img, rect, zc["upscale"]))
    cv2.imwrite(str(crop1_p), crop_view(img, rect, 1.0))
    cut = []
    for o in ctx["st"]["objects"]:                              # stored objects whose box crosses the crop border (would be cut)
        b = o["bbox_xyxy"]
        inside_any = b[2] > rect[0] and b[0] < rect[2] and b[3] > rect[1] and b[1] < rect[3]
        fully = b[0] >= rect[0] and b[1] >= rect[1] and b[2] <= rect[2] and b[3] <= rect[3]
        if inside_any and not fully:
            cut.append(o["object_id"])
    prompts = {"exp1": cfg["exp1"]["prompt"], "exp3_comparison": cfg["exp3"]["comparison"]["prompt"]}
    for k, p in prompts.items():
        (out / "prompts" / f"{k}.txt").write_text(p, encoding="utf-8")
    inputs = {"image": {"path": str(img_path), "sha256": rcfg["image_sha256"], "size": [W, H], "frame": "undistorted E1 px"},
              "zone_crop": {"source": f"stored Qwen L2 box of {zc['source_area_id']} (run {ctx['st']['qwen_run'].name})", "area_box_px": area_box, "rect_px": rect,
                            "upscale": zc["upscale"], "x2_size": list(cv2.imread(str(crop2_p)).shape[1::-1]), "stored_objects_cut_by_border": cut,
                            "sha256": {"x2": sha256_file(crop2_p), "x1": sha256_file(crop1_p)}},
              "prompt_sha256": {k: sha256_text(p) for k, p in prompts.items()}}
    (out / "inputs/inputs.json").write_text(json.dumps(inputs, indent=1, ensure_ascii=False), encoding="utf-8")
    print("zone crop", rect, "cut by border:", cut)
    if args.dry_run:
        print("dry run written", out)
        return out

    # ---------------------------------------------------------------- live: Exp 1 -> Exp 2 -> Exp 3 (comparison)
    from semantic_task_nav.perception.model_client import ModelWorkerClient
    client = ModelWorkerClient(mcfg, out / "worker_logs")
    calls, t0 = [], time.time()
    images = {"full": (full_p, [0, 0, W, H], 1.0), "zone_crop": (crop2_p, rect, zc["upscale"]), "zone_crop_x1": (crop1_p, rect, 1.0)}
    e1 = {}
    try:
        env = client.start()
        for cond, c in cfg["exp1"]["conditions"].items():
            p, r_, _ = images[c["image"]]
            raw = qwen_call(client, cfg["exp1"]["prompt"], p, cfg["exp1"]["max_new_tokens"], f"exp1_{cond}", cfg["exp1"]["prompt_id"], calls)
            (out / "qwen" / f"exp1_{cond}_raw.txt").write_text(raw, encoding="utf-8")
            obj, err = parse_reply(raw, OPEN_SCHEMA)
            for cand in (obj or {}).get("candidates", []):
                cand["bbox_image_px"] = qwen_box_to_image(cand.get("bbox_2d"), r_)
            e1[cond] = {"image": str(p), "image_rect_px": r_, "parsed": obj, "error": err}
            print("exp1", cond, "error" if err else (obj["decision"], obj.get("selected"), [c["name_en"] for c in obj["candidates"]]))
        (out / "exp1_open_reasoning.json").write_text(json.dumps(e1, indent=1, ensure_ascii=False), encoding="utf-8")

        responses = {k: v["parsed"] for k, v in e1.items()}
        phrases, origins = dino_phrases(responses, cfg["exp2"]["phrase_max_words"])
        qwen_boxes = {}
        for cond, v in e1.items():
            for cand in (v["parsed"] or {}).get("candidates", []):
                qwen_boxes.setdefault(norm_phrase(cand["name_en"], cfg["exp2"]["phrase_max_words"]), []).append(cand.get("bbox_image_px"))
        dets, diag = [], {}
        if phrases:
            for cond, c in cfg["exp2"]["conditions"].items():
                p, r_, sc = images[c["image"]]
                raw = dino_call(client, p, phrases, cfg["exp2"]["thresholds"], f"exp2_{cond}", calls)
                (out / "dino" / f"exp2_{cond}_raw.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")
                dets += detections(raw, cond, r_, sc, cfg["exp2"]["max_area_frac"])
        for dname, dc in cfg["exp2"]["diagnostics"].items():
            ph = phrases if dc["phrases"] == "from_exp1" else dc["phrases"]
            diag[dname] = {"phrases": ph, "thresholds": dc["thresholds"], "detections": []}
            if not ph:
                continue
            for cond in dc["conditions"]:
                p, r_, sc = images[cfg["exp2"]["conditions"][cond]["image"]]
                raw = dino_call(client, p, ph, dc["thresholds"], f"diag_{dname}_{cond}", calls)
                (out / "dino" / f"diag_{dname}_{cond}_raw.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")
                diag[dname]["detections"] += detections(raw, f"{dname}_{cond}", r_, sc, cfg["exp2"]["max_area_frac"])
            diag[dname]["merged"] = merge(diag[dname]["detections"], cfg["exp2"]["cross_condition_merge_iou"])
        merged = merge(dets, cfg["exp2"]["cross_condition_merge_iou"])
        for c in merged:
            c["in_context_roi"] = centre_in(c["bbox_xyxy"], rect)
        cross = {"in_roi_found_in": {}}
        for c in merged:
            if c["in_context_roi"]:
                key = "+".join(c["conditions"])
                cross["in_roi_found_in"][key] = cross["in_roi_found_in"].get(key, 0) + 1
        exp2 = {"phrases": phrases, "phrase_origins": origins, "thresholds": cfg["exp2"]["thresholds"], "n_detections": len(dets),
                "by_status": {s: sum(d["status"] == s for d in dets) for s in ("KEPT", "FILTERED_OVERSIZE", "EMPTY")}, "detections": dets, "merged_candidates": merged,
                "cross_condition": cross, "diagnostics": diag, "rank_limit": "none (every box above threshold is stored with its rank)"}
        (out / "exp2_candidates.json").write_text(json.dumps(exp2, indent=1, ensure_ascii=False), encoding="utf-8")
        print("exp2 phrases", phrases, "detections", len(dets), "merged", len(merged), "in ROI", sum(c["in_context_roi"] for c in merged))

        dec = select(merged, origins, responses, qwen_boxes, rect, cfg["exp3"], cfg["exp2"]["phrase_max_words"])
        if dec["needs_comparison"]:
            byid = {c["id"]: c for c in merged}
            shown = [byid[i] for i in dec["needs_comparison"]]
            cp = out / "inputs/comparison_numbered_x2.png"
            comparison_image(img, rect, zc["upscale"], shown, cp)
            raw = qwen_call(client, cfg["exp3"]["comparison"]["prompt"], cp, cfg["exp3"]["comparison"]["max_new_tokens"], "exp3_comparison", cfg["exp3"]["comparison"]["prompt_id"], calls)
            (out / "qwen/exp3_comparison_raw.txt").write_text(raw, encoding="utf-8")
            obj, err = parse_reply(raw, COMPARE_SCHEMA)
            dec = apply_comparison(dec, obj, err)
            dec["comparison"]["shown_numbering"] = {str(k): i for k, i in enumerate(dec["needs_comparison"], 1)}
        print("exp3", dec["status"], dec["selected"], dec["rule"])
    finally:
        client.close()
    worker_env = client.env_info
    (out / "model_calls.json").write_text(json.dumps({"calls": calls, "worker_env": worker_env, "swaps": client.swaps, "total_wall_s": round(time.time() - t0, 1),
                                                       "n_calls": {"qwen": sum(c["stage"].startswith("exp1") or c["stage"].startswith("exp3") for c in calls),
                                                                   "dino": sum(c["stage"].startswith(("exp2", "diag")) for c in calls)}}, indent=1, ensure_ascii=False), encoding="utf-8")
    for c in merged:
        if c["id"] == dec["selected"]:
            c["states"] += ["SELECTED", "VERIFY_REQUIRED"]
    exp3 = {"decision": dec, "zone_choice": {"area_id": zc["source_area_id"], "source": "stored Qwen L2 (zone level)", "note": "choosing the zone is NOT choosing the object inside it"},
            "candidates": [{k: c[k] for k in ("id", "rank", "bbox_xyxy", "score", "phrases", "conditions", "in_context_roi", "states", "qwen_origins")} for c in merged]}
    (out / "exp3_selection.json").write_text(json.dumps(exp3, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------- Exp 4: grounding + route B (offline)
    fr, s, pad, m = ctx["fr"], ctx["fr"].s, ctx["fr"].pad, ctx["ft"]["footprint_margin_cm"]
    L, Wd = map(float, cfg["exp4"]["robot_cm"])
    navs = {h: FN.RectNavMap(ctx["spec_bev"], fr.bev_size, ctx["pcfg"], FN.footprint_dims_cm(L, Wd, m, h), s) for h in (0, 90)}
    sn_plan = PT.jload((HERE / cfg["semnav_run"]).resolve() / "integrated_task_plan.json")
    pA = sn_plan["plans"][0]
    start_cm, old_B = pA["path_A"]["goal_cm"], pA["path_B_to_placement"]
    start_bev = fr.cm_to_bev([start_cm])[0].tolist()
    old_rep = SN.leg_record(ctx, SN.leg_to_region(ctx, gcfg, navs, 0, start_bev, ctx["place_candidate_cm"], "placement_floor", L, Wd))

    def ground_and_route(box, tag):
        poly = fr.img_to_cm(PT.box_poly(box))
        bc = fr.img_to_cm([[(box[0] + box[2]) / 2, box[3]]])[0]
        cc = fr.img_to_cm([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]])[0]
        zones = [z["id"] for layer in ("traversable", "static_obstacles", "no_go") for z in ctx["spec_cm"][layer]
                 if cv2.pointPolygonTest(np.float32(z["polygon"]), (float(bc[0]), float(bc[1])), False) >= 0]
        legrec = SN.leg_record(ctx, SN.leg_to_region(ctx, gcfg, navs, 0, start_bev, [poly.tolist()], tag, L, Wd))
        g = legrec.get("goal_cm")
        pt_gap = None if not g else round(FN.body_gap_cm([[bc[0] - .5, bc[1] - .5], [bc[0] + .5, bc[1] - .5], [bc[0] + .5, bc[1] + .5], [bc[0] - .5, bc[1] + .5]], g,
                                                         FN.footprint_dims_cm(L, Wd, 0, legrec["heading_deg"]), ctx["size_cm"], pad, 8), 1)
        gaps = [SN.CM.Raster(fr.bev_size, s, pad).gap(poly.tolist(), z["polygon"]) for z in ctx["spec_cm"]["traversable"]]
        dz = min((g_ for g_ in gaps if g_ is not None), default=None)
        return {"bbox_xyxy": box, "floor_polygon_cm": np.round(poly, 1).tolist(), "floor_contact_cm": [round(float(v), 1) for v in bc],
                "floor_contact_rule": cfg["exp4"]["floor_contact"], "box_centre_projection_cm": [round(float(v), 1) for v in cc],
                "projection_spread_cm": round(float(np.hypot(*(cc - bc))), 1), "zones_containing_contact": zones or ["NONE (outside every DRAFT zone: not drivable)"],
                "gap_polygon_to_drivable_zone_cm": None if dz is None else round(float(dz), 1), "route_B": legrec, "body_gap_to_contact_point_cm": pt_gap}
    exp4 = {"start_of_B_cm": start_cm, "old_B_stored": {k: old_B.get(k) for k in ("goal_cm", "length_cm", "gap_cm", "heading_deg")},
            "old_B_recomputed": {k: old_rep.get(k) for k in ("path_B_feasible", "goal_cm", "length_cm", "gap_cm", "heading_deg")},
            "old_B_reproduced": old_rep.get("goal_cm") == old_B.get("goal_cm") and old_rep.get("length_cm") == old_B.get("length_cm"), "new": None,
            "status": "NOT_RUN (no SELECTED target)"}
    if dec["status"] == "SELECTED":
        selbox = next(c for c in merged if c["id"] == dec["selected"])["bbox_xyxy"]
        exp4["new"] = ground_and_route(selbox, "selected_placement_target")
        r = exp4["new"]["route_B"]
        exp4["status"] = "ROUTE_B_GENERATED" if r["path_B_feasible"] else "NO_ROUTE_B"
        exp4["claim"] = "route to the nearest reachable pose of the selected target; placing the cube there is NOT verified"
    (out / "exp4_route_B.json").write_text(json.dumps(exp4, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("exp4", exp4["status"], exp4["old_B_reproduced"], (exp4["new"] or {}).get("route_B", {}).get("goal_cm"), (exp4["new"] or {}).get("route_B", {}).get("length_cm"))

    # ---------------------------------------------------------------- pipeline figures (no reference drawn)
    figures(img, rect, zc["upscale"], e1, merged, dec, exp4, ctx, out)

    # ---------------------------------------------------------------- Exp 5: evaluation (reference read only here)
    ev = evaluate(out, e1, merged, diag, dec, exp4, ground_and_route, img, rect, zc["upscale"])
    (out / "exp5_evaluation.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    summary = {"experiment_id": cfg["experiment_id"], "created": datetime.now().isoformat(timespec="seconds"), "claims": cfg["claims"],
               "model_calls": json.loads((out / "model_calls.json").read_text())["n_calls"], "exp1": {k: (v["parsed"] or {}).get("selected") if v["parsed"] else v["error"] for k, v in e1.items()},
               "exp3": {k: dec[k] for k in ("status", "selected", "rule")}, "exp4": exp4["status"], "evaluation": ev["stages"],
               "v02_code_before": fp_before, "v02_code_after": code_fingerprint()["sha256"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "manifest.json").write_text(json.dumps({"files_sha256": files, "code_sha256": {p.name: sha256_file(p) for p in sorted(HERE.glob("*.py"))}}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    return out


def evaluate(out, e1, merged, diag, dec, exp4, ground_and_route, img, rect, scale):
    """The ONLY place that reads evaluation_reference.yaml. Every input here was written to disk before this call."""
    ref = yaml.safe_load((HERE / "evaluation_reference.yaml").read_text(encoding="utf-8"))
    shutil.copy2(HERE / "evaluation_reference.yaml", out / "evaluation_reference_snapshot.yaml")
    rb, mt = ref["reference_target"]["bbox_xyxy_px"], ref["match"]
    rc = ((rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2)

    def grade(v):
        return "MATCH" if v >= mt["iou_match"] else "PARTIAL" if v >= mt["iou_partial"] else "NO"

    def qwen_locates(b):
        if not b:
            return False
        inside = b[0] <= rc[0] <= b[2] and b[1] <= rc[1] <= b[3]
        return iou(b, rb) >= mt["iou_partial"] or (inside and box_area(b) <= mt["qwen_box_max_area_ratio"] * box_area(rb))
    disc = {}
    for cond, v in e1.items():
        rows = []
        for c in (v["parsed"] or {}).get("candidates", []):
            name = f"{c.get('name_en', '')} {c.get('name_ko') or ''}".lower()
            rows.append({"name_en": c["name_en"], "name_ko": c.get("name_ko"), "suitability": c["suitability"],
                         "type_named": any(k in name for k in mt["type_keywords"]), "alt_named": any(k in name for k in mt["alt_type_keywords"]),
                         "bbox_image_px": c.get("bbox_image_px"), "iou_with_reference": None if not c.get("bbox_image_px") else round(iou(c["bbox_image_px"], rb), 3),
                         "locates_reference": qwen_locates(c.get("bbox_image_px")),
                         "is_selected_name": bool(v["parsed"]["selected"]) and norm_phrase(c["name_en"], 99) == norm_phrase(v["parsed"]["selected"], 99)})
        loc = [r for r in rows if r["locates_reference"]]
        disc[cond] = {"candidates": rows, "parse_error": v["error"],
                      "status": ("DISCOVERED_LOCATED" if loc else "NAMED_TYPE_NOT_LOCATED" if any(r["type_named"] for r in rows) else
                                 "ALT_NAME_NOT_LOCATED (manual review)" if any(r["alt_named"] for r in rows) else "NOT_DISCOVERED"),
                      "reference_selected_by_qwen": any(r["locates_reference"] and r["is_selected_name"] for r in rows)}
    best = max(merged, key=lambda c: iou(c["bbox_xyxy"], rb), default=None)
    bi = iou(best["bbox_xyxy"], rb) if best else 0.0
    det = {"best_candidate": best and {k: best[k] for k in ("id", "rank", "bbox_xyxy", "score", "phrases", "conditions", "states")}, "best_iou": round(bi, 3), "grade": grade(bi)}
    dd = {}
    for name, d in diag.items():
        bm = max(d.get("merged", []), key=lambda c: iou(c["bbox_xyxy"], rb), default=None)
        dd[name] = {"best_iou": round(iou(bm["bbox_xyxy"], rb), 3) if bm else 0.0, "best": bm and {k: bm[k] for k in ("id", "rank", "bbox_xyxy", "score", "phrases", "conditions")},
                    "n_merged": len(d.get("merged", [])), "note": "diagnostic: never used by the selection" + ("; contains the reference type -> not open-ended" if name == "VOCABULARY" else "")}
    sel = next((c for c in merged if c["id"] == dec["selected"]), None)
    si = iou(sel["bbox_xyxy"], rb) if sel else 0.0
    selection = {"decision": dec["status"], "selected": dec["selected"], "iou_with_reference": round(si, 3), "grade": grade(si) if sel else "NO_SELECTION",
                 "reference_candidate_states": best["states"] if best and grade(bi) != "NO" else None}
    oracle = ground_and_route(rb, "reference_target_oracle")
    ground = None
    if exp4["new"]:
        ground = {"iou_with_reference": round(si, 3), "floor_contact_distance_to_reference_cm": round(float(np.hypot(exp4["new"]["floor_contact_cm"][0] - oracle["floor_contact_cm"][0],
                                                                                                                        exp4["new"]["floor_contact_cm"][1] - oracle["floor_contact_cm"][1])), 1)}
    stages = {"Discovery": {k: v["status"] for k, v in disc.items()},
              "Detection": det["grade"] + (f" ({det['best_candidate']['id']}, rank {det['best_candidate']['rank']}, phrases {det['best_candidate']['phrases']})" if det["best_candidate"] and det["grade"] != "NO" else ""),
              "Selection": f"{selection['decision']} -> {selection['grade']}",
              "Grounding": "NOT_RUN" if not ground else ("MATCH" if si >= mt["iou_match"] else "WRONG_OBJECT" if si < mt["iou_partial"] else "PARTIAL") + f" (IoU {round(si, 3)}, contact offset {ground['floor_contact_distance_to_reference_cm']} cm)",
              "Navigation": exp4["status"]}
    fail = ("DISCOVERY: the model did not name or locate the reference" if not any(v["status"].startswith(("DISCOVERED", "NAMED")) for v in disc.values()) and det["grade"] == "NO" else
            "DETECTION: named but no matching box" if det["grade"] == "NO" else
            "SELECTION: detected but not selected" if selection["grade"] in ("NO", "NO_SELECTION") else
            "GROUNDING: selected box only partly on the reference" if selection["grade"] == "PARTIAL" else
            "NAVIGATION: no route B" if exp4["status"] != "ROUTE_B_GENERATED" else "NONE up to route B (placement itself unverified)")
    # evaluation figure (reference drawn here only)
    im = crop_view(img, rect, scale)
    b = to_crop(rb, rect, scale)
    PT._outline(im, lambda p: np.round(np.asarray(p).reshape(-1, 2)).astype(np.int32), CM_box(b), (0, 0, 255), 3, dashed=True)
    _t(im, "REFERENCE (AI-drawn from user description)", (b[0] - 120, b[3] + 28), (0, 0, 255), 0.6)
    if sel:
        sb = to_crop(sel["bbox_xyxy"], rect, scale)
        cv2.rectangle(im, tuple(sb[:2]), tuple(sb[2:]), (0, 255, 255), 2)
        _t(im, f"SELECTED {sel['id']} IoU {si:.2f}", (sb[0], sb[1] - 8), (0, 255, 255), 0.6)
    if best and best is not sel and bi > 0:
        bb = to_crop(best["bbox_xyxy"], rect, scale)
        cv2.rectangle(im, tuple(bb[:2]), tuple(bb[2:]), (255, 255, 0), 2)
        _t(im, f"best detected {best['id']} IoU {bi:.2f}", (bb[0], bb[3] + 50), (255, 255, 0), 0.6)
    for cond, v in e1.items():
        for c in (v["parsed"] or {}).get("candidates", []):
            if c.get("bbox_image_px"):
                q = to_crop(c["bbox_image_px"], rect, scale)
                cv2.rectangle(im, tuple(q[:2]), tuple(q[2:]), (255, 0, 255) if cond.startswith("A") else (255, 160, 40), 1)
    lines = ["# EVALUATION (reference read only at this stage)", f"reference box {rb} px: {ref['reference_target']['status']}"] + [f"{k}: {v}" for k, v in stages.items()] + \
            ["", f"first failing stage: {fail}", "", "red dashed = reference; yellow = selected; cyan = best-IoU detected candidate; magenta/blue thin = Qwen boxes (A full / B crop)"]
    cv2.imwrite(str(out / "figures/05_evaluation.png"), np.hstack([im, panel(im.shape[0], lines, 820)]))
    return {"reference": ref, "discovery": disc, "detection": det, "detection_diagnostics": dd, "selection": selection, "grounding": ground,
            "navigation": {"status": exp4["status"], "route": exp4["new"] and exp4["new"]["route_B"]},
            "reference_oracle_route_DIAGNOSTIC": {"note": "route to the reference box itself (evaluation only; shows whether navigation would work if the reference were chosen)", **oracle},
            "stages": stages, "first_failing_stage": fail, "rates": "NOT COMPUTED (one photo)"}


def CM_box(b):
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]


def figures(img, rect, scale, e1, merged, dec, exp4, ctx, out):
    fd = out / "figures"
    # 1: Exp 1 Qwen boxes, full image and crop
    full = img.copy()
    crop = crop_view(img, rect, scale)
    lines = ["# EXP 1 - open-ended placement reasoning (Qwen raw answers in qwen/)"]
    for cond, v in e1.items():
        col = (255, 0, 255) if cond.startswith("A") else (255, 160, 40)
        p = v["parsed"]
        lines.append(f"#{cond}: decision {p and p['decision']}, selected {p and p.get('selected')}" if p else f"#{cond}: {v['error']}")
        for i, c in enumerate((p or {}).get("candidates", []), 1):
            lines.append(f"{i}. {c['name_en']} [{c['suitability']}] box {c.get('bbox_image_px')}")
            b = c.get("bbox_image_px")
            if b:
                cv2.rectangle(full, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 3 if cond.startswith("A") else 2)
                _t(full, f"{cond[0]}{i} {c['name_en'][:28]}", (int(b[0]), int(b[1]) - 6), col, 0.55)
                q = to_crop(b, rect, scale)
                cv2.rectangle(crop, tuple(q[:2]), tuple(q[2:]), col, 2)
                _t(crop, f"{cond[0]}{i}", (q[0], q[1] - 6), col, 0.6)
    cv2.rectangle(full, (rect[0], rect[1]), (rect[2] - 1, rect[3] - 1), (255, 255, 255), 2)
    _t(full, "zone crop (stored area_5 box + margin)", (rect[0] + 6, rect[3] - 10))
    h = full.shape[0]
    cr = cv2.resize(crop, (round(crop.shape[1] * h / crop.shape[0]), h))
    cv2.imwrite(str(fd / "01_exp1_qwen_open.png"), np.hstack([full, cr, panel(h, lines, 760)]))
    # 2: DINO merged candidates in the zone crop
    im = crop_view(img, rect, scale)
    inroi = [c for c in merged if c["in_context_roi"]]
    for c in inroi:
        b = to_crop(c["bbox_xyxy"], rect, scale)
        col = (0, 255, 0) if len(c["conditions"]) > 1 else (0, 200, 255) if c["conditions"] == ["FULL"] else (255, 200, 0)
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), col, 1)
        _t(im, f"{c['id']}", (b[0], b[1] - 4), col, 0.45)
    lines = ["# EXP 2 - Grounding DINO candidates (merged over FULL / ROI / ROI_X2, IoU only)", f"{len(merged)} merged candidates, {len(inroi)} with centre in the zone crop",
             "green = found in >1 condition; orange = FULL only; blue = ROI / ROI_X2 only", ""]
    lines += [f"{c['id']} r{c['rank']} {c['score']:.2f} {','.join(c['conditions'])}: {'/'.join(c['phrases'])[:60]}" for c in sorted(inroi, key=lambda c: c["rank"])[:40]]
    cv2.imwrite(str(fd / "02_exp2_dino_candidates.png"), np.hstack([im, panel(im.shape[0], lines, 760)]))
    # 3: selection
    im = crop_view(img, rect, scale)
    for c in inroi:
        b = to_crop(c["bbox_xyxy"], rect, scale)
        st = c["states"]
        col = (0, 255, 255) if "SELECTED" in st else (0, 255, 0) if "SEMANTICALLY_RELEVANT" in st else (120, 120, 120)
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), col, 3 if "SELECTED" in st else 1)
        if "SEMANTICALLY_RELEVANT" in st:
            _t(im, c["id"], (b[0], b[1] - 4), col, 0.5)
    lines = ["# EXP 3 - semantic target selection", f"decision: {dec['status']} {dec['selected'] or ''}", f"rule: {dec['rule']}", f"Qwen selected names: {dec['selected_names_by_condition']}",
             f"relevant (DETECTED + Qwen suitable/possible + in zone): {dec['relevant_ids']}", "yellow = SELECTED (VERIFY_REQUIRED); green = SEMANTICALLY_RELEVANT; grey = DETECTED only"]
    if dec.get("comparison"):
        p = dec["comparison"]["parsed"]
        lines += ["", "#comparison call (numbered boxes, inputs/comparison_numbered_x2.png)", f"numbering: {dec['comparison'].get('shown_numbering')}"]
        lines += [f"{c['id']}: {c['suitable']} - {c.get('observed_object')}: {c.get('reason')}" for c in (p or {}).get("candidates", [])] + [f"best = {(p or {}).get('best')}: {(p or {}).get('reason')}"]
    cv2.imwrite(str(fd / "03_exp3_selection.png"), np.hstack([im, panel(im.shape[0], lines, 760)]))
    # 4: route comparison
    mp = PT.Mapper("photo", ctx["fr"])
    im = img.copy()
    RI.layers(im, mp, ctx, text=False)
    PT._fill(im, mp, ctx["place_candidate_cm"], (0, 255, 0), 0.3)
    lines = ["# EXP 4 - route B: earlier target (bay free floor) vs selected placement target", f"start of B = pick-up approach {exp4['start_of_B_cm']} cm (heading 0)",
             f"earlier B (free floor): goal {exp4['old_B_recomputed']['goal_cm']} cm, {exp4['old_B_recomputed']['length_cm']} cm, gap {exp4['old_B_recomputed']['gap_cm']} cm (reproduced: {exp4['old_B_reproduced']})"]
    if exp4["new"]:
        n, r = exp4["new"], exp4["new"]["route_B"]
        PT._outline(im, mp, n["floor_polygon_cm"], (0, 255, 255), 2)
        cv2.drawMarker(im, tuple(mp(n["floor_contact_cm"])[0]), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        if r["path_B_feasible"] and len(r["waypoints_cm"]) > 1:
            RI.draw_path(im, mp, {k: r[k] for k in ("waypoints_cm", "footprint_extent_xy_cm", "goal_cm")}, (0, 200, 255), "N", 15, coords=False)
        lines += [f"NEW B (selected target): goal {r.get('goal_cm')} cm, {r.get('length_cm')} cm, gap to box {r.get('gap_cm')} cm, to contact point {n['body_gap_to_contact_point_cm']} cm",
                  f"target contact {n['floor_contact_cm']} cm in {n['zones_containing_contact']}; box-centre projection {n['projection_spread_cm']} cm away (height parallax)",
                  f"target polygon gap to the drivable zone: {n['gap_polygon_to_drivable_zone_cm']} cm"]
    else:
        lines.append(f"NEW B: {exp4['status']}")
    lines += ["", "orange = new path B; green fill = earlier free-floor target; yellow = selected target and its floor contact", "plan only: placement NOT verified; no zone was relaxed"]
    cv2.imwrite(str(fd / "04_exp4_route_comparison.png"), np.hstack([im, panel(im.shape[0], lines, 900)]))


if __name__ == "__main__":
    main()
