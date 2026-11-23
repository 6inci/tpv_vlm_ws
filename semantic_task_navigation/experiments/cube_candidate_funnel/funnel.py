#!/usr/bin/env python3
"""Cube candidate funnel (offline; no model call, no GPU).

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
  $PY experiments/cube_candidate_funnel/funnel.py

For each of 8 photos of one fixed layout (only the red cube moves): at which stage of the v0.2 target search (conveyor ROI -> HSV/DINO
candidates -> merge/rank -> Qwen crop gate) is the cube candidate lost? v0.2 functions are called unchanged (make_roi, zoom_roi,
hsv_candidates, merge_candidates, check_alignment). Stored v0.2 model records (DINO, Qwen gate) exist for E1 only and are read, never re-run.
The cube position labels are an AI visual review (config.yaml), not ground truth: the output counts where a candidate is lost and
computes no accuracy. Nothing under src/ or config/ is modified.
"""
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file  # noqa: E402
from semantic_task_nav.geometry.coordinate_frames import RoiTransform, box_area, iou  # noqa: E402
from semantic_task_nav.navigation.map_builder import check_alignment  # noqa: E402
from semantic_task_nav.perception.target_inspection import hsv_candidates, make_roi, merge_candidates, zoom_roi  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
KEPT, DROPPED, NOT_REACHED, NOT_AVAILABLE = "KEPT", "DROPPED", "NOT_REACHED", "NOT_AVAILABLE"
STAGES = ("roi_search_area", "candidate_generation", "rank_within_verify_limit", "qwen_gate", "v02_decision")


def load_cfg():
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------------- matching and stages
def label_centre(label):
    return (label[0] + label[2]) / 2, (label[1] + label[3]) / 2


def keeps_label(cand_box, label, rule):
    """The candidate box keeps the label when the label centre is inside it (tolerance) or IoU >= iou_min."""
    cx, cy = label_centre(label)
    t = rule["centre_tolerance_px"]
    inside = cand_box[0] - t <= cx <= cand_box[2] + t and cand_box[1] - t <= cy <= cand_box[3] + t
    return bool(inside or iou(cand_box, label) >= rule["iou_min"])


def find_label_candidate(cands, label, rule):
    """Index of the tightest candidate that keeps the label, or None."""
    hits = [i for i, c in enumerate(cands) if keeps_label(c["bbox_xyxy"], label, rule)]
    return min(hits, key=lambda i: box_area(cands[i]["bbox_xyxy"])) if hits else None


def roi_stage(label, roi):
    x0, y0, x1, y1 = roi.crop_xyxy
    cx, cy = label_centre(label)
    inside = x0 <= cx < x1 and y0 <= cy < y1
    dx, dy = max(x0 - cx, 0, cx - x1), max(y0 - cy, 0, cy - y1)
    return {"status": KEPT if inside else DROPPED, "roi_crop_xyxy_raw": roi.crop_xyxy, "roi_scale": round(roi.scale, 4),
            "label_centre": [round(cx, 1), round(cy, 1)],
            "distance_outside_roi_px": 0.0 if inside else round(math.hypot(dx, dy), 1),
            "margin_inside_roi_px": round(min(cx - x0, x1 - cx, cy - y0, y1 - cy), 1) if inside else None}


def hsv_roi_candidates(image, roi, tcfg):
    """v0.2 colour/shape candidates found inside one ROI (raw_image_px boxes)."""
    return hsv_candidates(zoom_roi(image, roi), roi, tcfg)


def hsv_only_rank(cands, label, rule, tcfg):
    """Rank of the label candidate after v0.2 merge ordering when ONLY colour candidates exist (no detector score).
    Ordering is then decided by solidity alone, which is a weak ordering: see the report."""
    merged = merge_candidates([dict(c) for c in cands], tcfg["merge_iou"])
    idx = find_label_candidate(merged, label, rule)
    return (None if idx is None else idx + 1), len(merged)


def gate_record(v):
    """The stored Qwen crop check of one candidate, as v0.2 wrote it (answer is a dict; the raw model reply is kept)."""
    a, call = v.get("answer") or {}, v.get("call") or {}
    return {"status": v["status"], "answer": a.get("answer"), "observed_color": a.get("observed_color"), "observed_object": a.get("observed_object"),
            "reason": a.get("reason") or v.get("reason"), "crop": v.get("crop"), "prompt_id": call.get("prompt_id"),
            "prompt_sha256": call.get("prompt_sha256"), "model": call.get("model"), "raw_response": call.get("raw_response")}


def stored_candidate_record(ti, label, rule):
    """What the stored v0.2 run recorded for the candidate that keeps the label (E1 only)."""
    idx = find_label_candidate(ti["candidates"], label, rule)
    base = {"n_candidates": ti["n_candidates"], "n_verified": ti["n_verified"], "status": ti["status"]}
    if idx is None:
        return {**base, "candidate": None}
    c = ti["candidates"][idx]
    kinds = {}
    for s in c["sources"]:
        kinds[s] = kinds.get(s, 0) + 1
    return {**base, "candidate": {"candidate_id": c["candidate_id"], "bbox_xyxy": c["bbox_xyxy"], "rank": idx + 1,
                                  "sources": kinds, "detector_score": c.get("score"), "gate": gate_record(c["verification"])}}


def photo_record(pid, label, image, roi, tcfg, rule, stored_ti=None):
    """The funnel of one photo. Stage outcomes: KEPT / DROPPED / NOT_REACHED (an earlier stage already dropped the label) /
    NOT_AVAILABLE (needs a model run that is not stored for this photo)."""
    topk = tcfg["max_candidates_to_verify"]
    st = {}
    st["roi_search_area"] = roi_stage(label, roi)
    hsv = hsv_roi_candidates(image, roi, tcfg)
    hsv_idx = find_label_candidate(hsv, label, rule)
    in_roi = st["roi_search_area"]["status"] == KEPT
    sources = ["hsv_color_shape"]
    st["candidate_generation"] = {"n_hsv_candidates_in_roi": len(hsv), "label_in_hsv_candidates": hsv_idx is not None,
                                  "sources_available": sources}
    rank, n_merged, basis = None, None, "HSV_ONLY (no stored detector run for this photo)"
    stored_rec = None
    if stored_ti is not None:
        stored_rec = stored_candidate_record(stored_ti, label, rule)
        sources = ["hsv_color_shape", "grounding_dino_roi", "grounding_dino_full_image (stored)"]
        c = stored_rec["candidate"]
        rank, n_merged, basis = (c["rank"] if c else None), stored_rec["n_candidates"], "STORED_MERGE (v0.2 run: HSV + DINO)"
        st["candidate_generation"].update(sources_available=sources, stored_candidate=c)
        st["candidate_generation"]["status"] = KEPT if c else DROPPED
    else:
        st["candidate_generation"]["status"] = KEPT if hsv_idx is not None else DROPPED
        if hsv_idx is not None:
            rank, n_merged = hsv_only_rank(hsv, label, rule, tcfg)
    if not in_roi:                       # v0.2 only looks inside the ROI: nothing downstream can see the label
        st["candidate_generation"]["status"] = NOT_REACHED
        st["rank_within_verify_limit"] = {"status": NOT_REACHED}
    else:
        r = {"rank": rank, "n_candidates": n_merged, "limit": topk, "basis": basis}
        r["status"] = DROPPED if rank is None or rank > topk else KEPT
        st["rank_within_verify_limit"] = r
    if stored_rec is None:
        st["qwen_gate"] = {"status": NOT_AVAILABLE if in_roi else NOT_REACHED,
                           "reason": "no stored Qwen gate run for this photo; it is not re-run here (offline step)" if in_roi else None}
        st["v02_decision"] = {"status": NOT_AVAILABLE if in_roi else NOT_REACHED}
    else:
        c = stored_rec["candidate"]
        gate = c["gate"] if c else None
        st["qwen_gate"] = ({"status": NOT_REACHED} if gate is None else
                           {"status": KEPT if gate["status"] == "yes" else DROPPED if gate["status"] == "no" else NOT_REACHED,
                            "stored": gate})
        st["v02_decision"] = {"status": KEPT if stored_rec["status"] == "TARGET_SELECTED" else DROPPED, "stored_status": stored_rec["status"]}
    first = next((s for s in STAGES if st[s]["status"] == DROPPED), None)
    return {"photo_id": pid, "label_box": label, "stages": st,
            "first_dropped_stage": first or "NONE (stages after the last available one were not run for this photo)"}


def whole_image_counterfactual(image, label, roi, rule, tcfg):
    """Same v0.2 colour/shape candidates over the whole frame (no conveyor ROI)."""
    h, w = image.shape[:2]
    full = RoiTransform([0, 0, w, h], 1.0)
    cands = hsv_roi_candidates(image, full, tcfg)
    rank, n = hsv_only_rank(cands, label, rule, tcfg)
    x0, y0, x1, y1 = roi.crop_xyxy
    n_in = sum(1 for c in cands if x0 <= (c["bbox_xyxy"][0] + c["bbox_xyxy"][2]) / 2 < x1 and y0 <= (c["bbox_xyxy"][1] + c["bbox_xyxy"][3]) / 2 < y1)
    return {"n_candidates": len(cands), "n_inside_v02_roi": n_in, "n_outside_v02_roi": len(cands) - n_in,
            "label_in_candidates": rank is not None, "label_rank_hsv_only": rank, "label_within_verify_limit": rank is not None and rank <= tcfg["max_candidates_to_verify"],
            "ordering_note": "HSV-only ordering (solidity); the stored v0.2 ordering uses detector agreement and score, which exist for E1 only"}


def reproduction_check(image, roi, tcfg, stored_ti, repeat_ti, tol):
    """E1 pin: the recomputed ROI and colour candidates equal what the stored v0.2 run recorded."""
    rec = {"roi_transform_equal": roi.crop_xyxy == stored_ti["rois"][0]["transform"]["crop_xyxy_raw"]
                                  and abs(roi.scale - stored_ti["rois"][0]["transform"]["scale"]) < 1e-9}
    mine = sorted([[round(float(v), 2) for v in c["bbox_xyxy"]] for c in hsv_roi_candidates(image, roi, tcfg)])
    theirs = sorted([m["bbox_xyxy"] for c in stored_ti["candidates"] for m in c["member_boxes"] if m["source"] == "hsv_color_shape"])
    dev = max((abs(a - b) for x, y in zip(mine, theirs) for a, b in zip(x, y)), default=0.0)
    rec.update(n_hsv_recomputed=len(mine), n_hsv_stored=len(theirs), max_box_deviation_px=round(dev, 3),
               hsv_equal=len(mine) == len(theirs) and dev <= tol)
    key = lambda ti: [(c["bbox_xyxy"], c["sources"], c["verification"].get("status"), c["verification"].get("answer")) for c in ti["candidates"]]  # noqa: E731
    rec["stored_repeat_run_identical"] = key(stored_ti) == key(repeat_ti)
    return rec


def other_run_record(run_dir, rid):
    ti = json.loads((Path(run_dir) / "analysis/target_inspection.json").read_text(encoding="utf-8"))
    acc = [(i, c) for i, c in enumerate(ti["candidates"]) if c["verification"]["status"] == "yes"]
    out = {"id": rid, "run": str(run_dir), "status": ti["status"], "n_candidates": ti["n_candidates"], "n_verified": ti["n_verified"],
           "roi_crop_xyxy_raw": ti["rois"][0]["transform"]["crop_xyxy_raw"], "label": "none (not one of the 8 photos; no AI label made)"}
    out["accepted"] = [{"candidate_id": c["candidate_id"], "rank": i + 1, "bbox_xyxy": c["bbox_xyxy"], "sources": len(c["sources"]),
                        "gate": gate_record(c["verification"])} for i, c in acc]
    return out


# ----------------------------------------------------------------------------------------------- figures
def _t(img, text, org, color=(255, 255, 255), scale=0.5):
    text = str(text).encode("ascii", "replace").decode()
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(max(0, min(org[0], img.shape[1] - tw - 2))), int(max(th + 3, min(org[1], img.shape[0] - bl - 2)))
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + bl), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def _rect(img, b, color, thick=2, pad=0):
    cv2.rectangle(img, (int(round(b[0] - pad)), int(round(b[1] - pad))), (int(round(b[2] + pad)), int(round(b[3] + pad))), (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.rectangle(img, (int(round(b[0] - pad)), int(round(b[1] - pad))), (int(round(b[2] + pad)), int(round(b[3] + pad))), color, thick, cv2.LINE_AA)


def render_labels(photos, images, rois):
    """01: per photo, the whole frame with the v0.2 ROI and the AI label, plus a 3x crop of the label (for human checking)."""
    s, rows = 0.3, []
    for p in photos:
        img = images[p["photo_id"]]
        b = p["label_box"]
        cx, cy = label_centre(b)
        full = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = rois[p["photo_id"]].crop_xyxy
        cv2.rectangle(full, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)), (255, 160, 40), 2, cv2.LINE_AA)
        cv2.circle(full, (int(cx * s), int(cy * s)), 11, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(full, (int(cx * s), int(cy * s)), 11, (255, 255, 255), 2, cv2.LINE_AA)
        half = 48
        crop = img[max(0, int(cy) - half):int(cy) + half, max(0, int(cx) - half):int(cx) + half].copy()
        ox, oy = max(0, int(cx) - half), max(0, int(cy) - half)
        crop = cv2.resize(crop, None, fx=3.4, fy=3.4, interpolation=cv2.INTER_CUBIC)
        _rect(crop, [(b[0] - ox - 6) * 3.4, (b[1] - oy - 6) * 3.4, (b[2] - ox + 6) * 3.4, (b[3] - oy + 6) * 3.4], (255, 255, 255), 1)
        crop = cv2.resize(crop, (full.shape[0], full.shape[0]))
        panel = np.hstack([full, np.full((full.shape[0], 6, 3), 24, np.uint8), crop])
        first = p["first_dropped_stage"]
        _t(panel, f"{p['photo_id']}  label {b}  roi: {p['stages']['roi_search_area']['status']}", (6, 18), (0, 255, 255))
        _t(panel, f"first dropped stage: {first if first in STAGES else 'none available'}", (6, 40), (255, 255, 255))
        rows.append(panel)
    cols = [np.vstack(rows[i::2]) for i in range(2)]
    head = np.full((34, cols[0].shape[1] * 2 + 10, 3), 24, np.uint8)
    _t(head, "AI PRELIMINARY labels (not human ground truth). Blue = v0.2 conveyor ROI (E1 box as proxy for the other photos). Circle = label.", (6, 22), (200, 200, 200), 0.5)
    return np.vstack([head, np.hstack([cols[0], np.full((cols[0].shape[0], 10, 3), 24, np.uint8), cols[1]])])


def render_e1(image, stored_ti, label, roi):
    """02: the 11 stored v0.2 candidates of E1 with the Qwen gate answers."""
    crop = [100, 0, 900, 720]
    img = image.copy()
    x0, y0, x1, y1 = roi.crop_xyxy
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 160, 40), 2, cv2.LINE_AA)
    for c in stored_ti["candidates"]:
        v = c["verification"]
        hit = keeps_label(c["bbox_xyxy"], label, {"centre_tolerance_px": 2.0, "iou_min": 0.3}) and box_area(c["bbox_xyxy"]) < 1000
        col = (0, 255, 255) if hit else ((90, 90, 255) if v["status"] == "no" else (170, 170, 170))
        _rect(img, c["bbox_xyxy"], col, 3 if hit else 1)
    out = img[crop[1]:crop[3], crop[0]:crop[2]]
    sc = 1.4
    out = cv2.resize(out, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    for c in stored_ti["candidates"]:
        b = c["bbox_xyxy"]
        v = c["verification"]
        ans = gate_record(v)["observed_object"] if v["status"] == "no" else v["status"]
        _t(out, f"{c['candidate_id'][11:]}:{ans}", ((b[0] - crop[0]) * sc, (b[1] - crop[1]) * sc - 4), (255, 255, 255), 0.42)
    pn = np.full((78, out.shape[1], 3), 24, np.uint8)
    _t(pn, "E1, stored v0.2 run: yellow = candidate that keeps the AI label (cube), red = Qwen gate 'no', gray = not verified", (6, 22), (200, 200, 200), 0.5)
    _t(pn, f"status {stored_ti['status']}: {stored_ti['n_candidates']} candidates, {stored_ti['n_verified']} verified, 0 accepted. blue line = ROI right edge x={x1}", (6, 46), (200, 200, 200), 0.5)
    _t(pn, "text = candidate number : what the Qwen gate said it saw (answer 'no') or 'not_run'", (6, 68), (200, 200, 200), 0.5)
    return np.vstack([pn, out])


def render_gate_crops(items):
    """03: the stored Qwen crops of the cube candidate in three photos (E1: rejected; DEV_REF, REF_P01: accepted)."""
    tiles = []
    for title, path in items:
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (int(im.shape[1] * 448 / im.shape[0]), 448)) if im.shape[0] != 448 else im
        _t(im, title, (6, 20), (0, 255, 255), 0.5)
        tiles.append(im)
    return np.hstack([np.hstack([t, np.full((t.shape[0], 6, 3), 24, np.uint8)]) for t in tiles])


# ----------------------------------------------------------------------------------------------- main
def main():
    cfg, v02 = load_cfg(), load_config()
    tcfg, rule = v02["models"]["target_inspection"], cfg["match"]
    fp_before = code_fingerprint()
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_funnel")
    (out / "figures").mkdir(parents=True)
    img_dir = Path(cfg["image_dir"])
    stored = lambda rel: (HERE / rel).resolve()  # noqa: E731
    e1_dir = stored(cfg["stored"]["e1_run"])
    e1_ti = json.loads((e1_dir / "analysis/target_inspection.json").read_text(encoding="utf-8"))
    rep_ti = json.loads((stored(cfg["stored"]["e1_repeat_run"]) / "analysis/target_inspection.json").read_text(encoding="utf-8"))
    det = json.loads((e1_dir / "analysis/detections.json").read_text(encoding="utf-8"))
    anchor = next(o for o in det["objects"] if o["object_id"] == cfg["roi_anchor_object_id"] and o["accepted"])

    images, rois, photos, hashes = {}, {}, [], {}
    ref_path = img_dir / next(i["file"] for i in cfg["images"] if i["id"] == cfg["reference_photo_id"])
    for im in cfg["images"]:
        path = img_dir / im["file"]
        assert sha256_file(path) == im["sha256"], f"{im['file']} differs from the declared image"
        images[im["id"]] = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        hashes[im["id"]] = im["sha256"]
    h, w = images[cfg["reference_photo_id"]].shape[:2]
    shared_roi = make_roi(anchor["bbox_xyxy"], (w, h), v02["models"]["target_inspection"])     # v0.2 rule applied to the stored E1 conveyor box
    align = {}
    for im in cfg["images"]:
        pid = im["id"]
        rois[pid] = shared_roi
        align[pid] = ({"status": "REFERENCE"} if pid == cfg["reference_photo_id"] else
                      check_alignment({"reference_image": str(ref_path), "alignment_check": cfg["alignment"]}, images[pid], lambda p: Path(p)))
        rec = photo_record(pid, cfg["labels"][pid], images[pid], shared_roi, tcfg, rule, e1_ti if pid == cfg["reference_photo_id"] else None)
        rec.update(file=im["file"], sha256=im["sha256"],
                   roi_source="stored v0.2 conveyor_1 box of the E1 run -> v0.2 make_roi" + ("" if pid == cfg["reference_photo_id"] else " (PROXY: no stored conveyor box for this photo)"),
                   camera_alignment_vs_E1=align[pid], whole_image_hsv=whole_image_counterfactual(images[pid], cfg["labels"][pid], shared_roi, rule, tcfg))
        photos.append(rec)
    repro = reproduction_check(images[cfg["reference_photo_id"]], shared_roi, tcfg, e1_ti, rep_ti, cfg["reproduce"]["box_tolerance_px"])
    others = [other_run_record(stored(o["run"]), o["id"]) for o in cfg["stored"]["other_runs"]]

    counts = {}
    for p in photos:
        counts[p["first_dropped_stage"]] = counts.get(p["first_dropped_stage"], 0) + 1
    summary = {"n_photos": len(photos), "first_dropped_stage_counts": counts,
               "whole_image_hsv_label_in_candidates": sum(p["whole_image_hsv"]["label_in_candidates"] for p in photos),
               "whole_image_hsv_candidate_counts": [p["whole_image_hsv"]["n_candidates"] for p in photos],
               "whole_image_hsv_label_ranks_hsv_only": [p["whole_image_hsv"]["label_rank_hsv_only"] for p in photos],
               "label_distance_outside_roi_px": {p["photo_id"]: p["stages"]["roi_search_area"]["distance_outside_roi_px"] for p in photos},
               "stored_roi_right_edges_px": {"E1": e1_ti["rois"][0]["transform"]["crop_xyxy_raw"][2],
                                            **{o["id"]: o["roi_crop_xyxy_raw"][2] for o in others}},
               "note": "counts of where an AI-labelled candidate is lost; NOT accuracy or recall (no human ground truth)"}

    cv2.imwrite(str(out / "figures/01_labels_and_roi.png"), render_labels(photos, images, rois))
    e1_id = cfg["reference_photo_id"]
    cv2.imwrite(str(out / "figures/02_E1_stored_candidates.png"), render_e1(images[e1_id], e1_ti, cfg["labels"][e1_id], shared_roi))
    c1 = next(p for p in photos if p["photo_id"] == e1_id)["stages"]["candidate_generation"]["stored_candidate"]
    crops = [("E1 cube candidate: gate 'no' (round)", e1_dir / c1["gate"]["crop"]["path"])]
    for o in others:
        a = o["accepted"][0]
        crops.append((f"{o['id']} accepted candidate (rank {a['rank']}): gate 'yes'", Path(o["run"]) / a["gate"]["crop"]["path"]))
    cv2.imwrite(str(out / "figures/03_gate_crops.png"), render_gate_crops(crops))

    fp_after = code_fingerprint()
    result = {"experiment_id": cfg["experiment_id"], "created": datetime.now().isoformat(timespec="seconds"), "scope": cfg["scope"],
              "claims": {"accuracy_or_recall": "NOT_COMPUTED (no human ground truth)", "labels": cfg["label_status"],
                         "model_calls_in_this_run": 0, "stored_model_records_used": "E1 (DINO + Qwen gate), DEV_REF, REF_P01 (Qwen gate)"},
              "v02_target_inspection_params": tcfg, "match_rule": rule, "reproduction_E1": repro, "photos": photos,
              "stored_other_runs": others, "summary": summary,
              "provenance": {"v02_code_before": fp_before["sha256"], "v02_code_after": fp_after["sha256"], "v02_code_unchanged": fp_before["sha256"] == fp_after["sha256"],
                             "image_sha256": hashes, "config_sha256": sha256_file(HERE / "config.yaml"),
                             "stored_e1_run": str(e1_dir), "stored_conveyor_box_xyxy": anchor["bbox_xyxy"],
                             "cv2": cv2.__version__, "numpy": np.__version__}}
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"wrote {out}")
    print("reproduction E1:", json.dumps(repro))
    for p in photos:
        s = p["stages"]
        print(f"{p['photo_id']}: " + " | ".join(f"{k}={s[k]['status']}" for k in STAGES) + f" -> first drop: {p['first_dropped_stage']}"
              f" | whole-image HSV: n={p['whole_image_hsv']['n_candidates']} rank={p['whole_image_hsv']['label_rank_hsv_only']}")
    print("alignment:", {k: v.get("status") for k, v in align.items()}, "| v0.2 code unchanged:", result["provenance"]["v02_code_unchanged"])
    return result


if __name__ == "__main__":
    main()
