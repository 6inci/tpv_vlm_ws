#!/usr/bin/env python3
"""Stage 1 of the candidate retention policy (offline; no model call).

  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python experiments/candidate_retention_policy/retention.py

Reused unchanged: v0.2 hsv_candidates / merge_candidates / make_roi (through cube_candidate_funnel/funnel.py), the stored E1 DINO boxes and Qwen answers.
New: whole-image search, ROI as an ordering key, a verification budget that records (not deletes) what it leaves out, verification states.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent / "cube_candidate_funnel"))
import funnel as F  # noqa: E402
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file  # noqa: E402
from semantic_task_nav.geometry.coordinate_frames import RoiTransform, box_area, iou  # noqa: E402
from semantic_task_nav.perception.target_inspection import merge_candidates  # noqa: E402


def load_cfg():
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def inside(box, crop):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return crop[0] <= cx < crop[2] and crop[1] <= cy < crop[3]


def stored_dino(e1_run):
    """Stored Grounding DINO boxes of the v0.2 E1 run: whole-image cube queries and the ROI boxes kept as candidate members."""
    det = json.loads((e1_run / "analysis/detections.json").read_text(encoding="utf-8"))
    ti = json.loads((e1_run / "analysis/target_inspection.json").read_text(encoding="utf-8"))
    full = [{"bbox_xyxy": d["bbox_xyxy"], "score": d["score"], "prompt": d.get("prompt"), "source": "grounding_dino_full_image"} for d in det["unverified_candidates"]["red_cube"]]
    roi = [{"bbox_xyxy": m["bbox_xyxy"], "score": m["score"], "prompt": None, "source": "grounding_dino_roi"}
           for c in ti["candidates"] for m in c["member_boxes"] if m["source"] == "grounding_dino_roi"]
    return full, roi, ti


def build_candidates(image, tcfg, roi_crop, dino=()):
    """Whole-image HSV (+ stored DINO) -> v0.2 merge -> policy order. The label is never an input here."""
    h, w = image.shape[:2]
    hsv = F.hsv_roi_candidates(image, RoiTransform([0, 0, w, h], 1.0), tcfg)
    # A detector box larger than the v0.2 cube size limit is not a cube-sized candidate. With the v0.2 containment rule it would act as a BRIDGE and
    # fuse distinct small candidates into one (seen on E1: a 483 x 242 px box swallowed two). Such boxes are kept as their own records, never merged.
    big = [dict(c) for c in dino if box_area(c["bbox_xyxy"]) > tcfg["candidate_max_area_raw_px"]]
    small = [dict(c) for c in dino if box_area(c["bbox_xyxy"]) <= tcfg["candidate_max_area_raw_px"]]
    merged = merge_candidates([dict(c) for c in hsv] + small, tcfg["merge_iou"])
    for m in merged:
        m["oversize_detector_box"] = False
    for c in big:
        merged.append({**c, "sources": [c["source"]], "members": [c], "oversize_detector_box": True})
    for m in merged:
        m["source_types"] = sorted(set(m["sources"]))
        m["inside_roi"] = bool(inside(m["bbox_xyxy"], roi_crop))
    merged.sort(key=lambda m: (m["oversize_detector_box"], -len(m["source_types"]), not m["inside_roi"], -(m["score"] or 0.0), -(m.get("shape_stats") or {}).get("solidity", 0)))
    return merged


def apply_budget(merged, budget):
    return [{"rank": i + 1, "bbox_xyxy": [round(float(v), 2) for v in m["bbox_xyxy"]], "score": m["score"], "source_types": m["source_types"],
             "n_members": len(m["members"]), "inside_roi": m["inside_roi"], "oversize_detector_box": m["oversize_detector_box"], "shape_stats": m.get("shape_stats"),
             "retention": "SENT_TO_VERIFICATION" if i < budget else "BEYOND_BUDGET (recorded, not deleted)"} for i, m in enumerate(merged)]


def state_of(stored_verification, source_types, target_color):
    """Declared rule (config.states). Qwen yes is VLM_ACCEPTED, never CONFIRMED; detector agreement is correlated evidence."""
    if stored_verification is None or stored_verification.get("status") not in ("yes", "no"):
        return "PENDING_VERIFICATION", None
    ans = stored_verification.get("answer") or {}
    if stored_verification["status"] == "yes":
        return "VLM_ACCEPTED", None
    disagree = len(source_types) >= 2
    note = "Qwen 'no' vs %d detector source types pointing at the box (correlated evidence)" % len(source_types) if disagree else None
    if disagree and str(ans.get("observed_color", "")).lower() == target_color:
        return "PROBABLE", note
    return "REJECTED", note


def photo_decision(states):
    live = [s for s in states if s in ("VLM_ACCEPTED", "PROBABLE")]
    if any(s == "PENDING_VERIFICATION" for s in states):
        return "INCOMPLETE (candidates still PENDING_VERIFICATION)" if len(live) <= 1 else "AMBIGUOUS"
    return "AMBIGUOUS" if len(live) > 1 else live[0] if live else "TARGET_NOT_FOUND"


def attach_e1_states(cands, ti, target_color):
    for c in cands:
        best = max(ti["candidates"], key=lambda s: iou(c["bbox_xyxy"], s["bbox_xyxy"]))
        v = best["verification"] if iou(c["bbox_xyxy"], best["bbox_xyxy"]) >= 0.5 else None
        c["state"], c["disagreement"] = state_of(v, c["source_types"], target_color)
        c["stored_qwen"] = None if v is None or v.get("status") not in ("yes", "no") else {"candidate_id": best["candidate_id"], "status": v["status"], **{k: (v.get("answer") or {}).get(k) for k in ("observed_color", "observed_object", "reason")}}


def render(photos, images, roi_crop, labels, out):
    tiles = []
    for p in photos:
        img = images[p["photo_id"]].copy()
        cv2.rectangle(img, tuple(roi_crop[:2]), tuple(roi_crop[2:]), (255, 200, 0), 2)
        for c in p["candidates"]:
            col = (0, 220, 0) if c["retention"].startswith("SENT") else (120, 120, 120)
            F._rect(img, c["bbox_xyxy"], col, 2, pad=6)
            F._t(img, str(c["rank"]), (c["bbox_xyxy"][0] - 8, c["bbox_xyxy"][1] - 10), col, 0.7)
        F._rect(img, labels[p["photo_id"]], (0, 0, 255), 2, pad=16)
        F._t(img, "%s  label cand. rank %s/%s  v0.2: %s" % (p["photo_id"], p["label"]["rank"], p["n_candidates"], p["v02_baseline_first_dropped_stage"]), (10, 34), (255, 255, 255), 1.0)
        tiles.append(cv2.resize(img, (960, 540)))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(str(out / "01_retained_candidates.png"), np.vstack(rows))
    return ["01_retained_candidates.png"]


def main():
    cfg, fcfg, tcfg = load_cfg(), yaml.safe_load((HERE / load_cfg()["funnel_config"]).read_text(encoding="utf-8")), load_config()["models"]["target_inspection"]
    pol = cfg["policy"]
    fdir = (HERE / cfg["funnel_config"]).resolve().parent
    e1_run = (fdir / fcfg["stored"]["e1_run"]).resolve()
    baseline = {p["photo_id"]: p["first_dropped_stage"] for p in json.loads(((HERE / cfg["funnel_run"]).resolve() / "result.json").read_text(encoding="utf-8"))["photos"]}
    fp_before = code_fingerprint()["sha256"]
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_retention")
    out.mkdir(parents=True, exist_ok=False)
    full, roi_boxes, ti = stored_dino(e1_run)
    det = json.loads((e1_run / "analysis/detections.json").read_text(encoding="utf-8"))
    anchor = next(o for o in det["objects"] if o["object_id"] == fcfg["roi_anchor_object_id"])
    photos, images = [], {}
    for im in fcfg["images"]:
        path = Path(fcfg["image_dir"]) / im["file"]
        assert sha256_file(path) == im["sha256"], f"{im['file']} differs from the declared image"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        images[im["id"]] = image
        roi_crop = F.make_roi(anchor["bbox_xyxy"], (image.shape[1], image.shape[0]), tcfg).crop_xyxy
        is_e1 = im["id"] == fcfg["reference_photo_id"]
        merged = build_candidates(image, tcfg, roi_crop, (full + roi_boxes) if is_e1 else ())
        cands = apply_budget(merged, pol["verify_budget"])
        label = fcfg["labels"][im["id"]]                                # used only AFTER the candidates exist
        idx = F.find_label_candidate(cands, label, fcfg["match"])
        if is_e1:
            attach_e1_states(cands, ti, cfg["states"]["target_color"])
        else:
            for c in cands:
                c["state"], c["disagreement"], c["stored_qwen"] = "PENDING_VERIFICATION", None, None
        lab = {"status": "AI_PRELIMINARY_REVIEW", "in_candidates": idx is not None, "rank": None if idx is None else idx + 1,
               "reaches_verification": idx is not None and idx < pol["verify_budget"],
               "within_v02_limit_of_%d" % pol["v02_verify_limit_for_comparison"]: idx is not None and idx < pol["v02_verify_limit_for_comparison"],
               "inside_roi": None if idx is None else cands[idx]["inside_roi"], "state": None if idx is None else cands[idx]["state"]}
        photos.append({"photo_id": im["id"], "dino_records": "STORED (E1 v0.2 run)" if is_e1 else "NOT_AVAILABLE", "n_candidates": len(cands),
                       "n_sent_to_verification": sum(c["retention"].startswith("SENT") for c in cands), "n_beyond_budget": sum(c["retention"].startswith("BEYOND") for c in cands),
                       "n_inside_roi": sum(c["inside_roi"] for c in cands), "label": lab, "v02_baseline_first_dropped_stage": baseline[im["id"]],
                       "photo_decision_from_stored_answers": photo_decision([c["state"] for c in cands]) if is_e1 else "NOT_AVAILABLE (no Qwen answers)", "candidates": cands})
    figs = render(photos, images, roi_crop, fcfg["labels"], out)
    e1 = next(p for p in photos if p["photo_id"] == fcfg["reference_photo_id"])
    summary = {"photos": len(photos), "label_candidate_reaches_verification": sum(p["label"]["reaches_verification"] for p in photos),
               "label_candidate_within_v02_limit": sum(p["label"]["within_v02_limit_of_%d" % pol["v02_verify_limit_for_comparison"]] for p in photos),
               "v02_baseline_not_dropped_before_gate": sum(1 for p in photos if p["v02_baseline_first_dropped_stage"] != "roi_search_area"),
               "candidates_per_photo": [p["n_candidates"] for p in photos], "candidates_beyond_budget": sum(p["n_beyond_budget"] for p in photos),
               "e1_states": {s: sum(c["state"] == s for c in e1["candidates"]) for s in ("VLM_ACCEPTED", "PROBABLE", "REJECTED", "PENDING_VERIFICATION")},
               "e1_photo_decision": e1["photo_decision_from_stored_answers"]}
    result = {"experiment_id": cfg["experiment_id"], "created": datetime.now().isoformat(timespec="seconds"),
              "claims": {"accuracy_or_recall": "NOT_COMPUTED (labels are an AI preliminary review, 8 photos of one session)", "model_calls_in_this_run": 0,
                         "counts_mean": "how many AI-labelled cube candidates reach the verification stage under this policy"},
              "policy": pol, "states_rule": cfg["states"], "summary": summary, "photos": photos, "figures": figs,
              "provenance": {"v02_code_before": fp_before, "v02_code_after": code_fingerprint()["sha256"], "config_sha256": sha256_file(HERE / "config.yaml"),
                             "e1_stored_dino_boxes": {"whole_image": len(full), "roi": len(roi_boxes)}}}
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(summary, indent=1))
    for p in photos:
        print(p["photo_id"], "n", p["n_candidates"], "label", p["label"]["rank"], "inROI", p["label"]["inside_roi"], "| v0.2:", p["v02_baseline_first_dropped_stage"])
    for c in e1["candidates"]:
        if c["state"] != "PENDING_VERIFICATION" or c["rank"] <= 12:
            print("  E1", c["rank"], c["bbox_xyxy"], c["source_types"], c["inside_roi"], c["state"], (c["stored_qwen"] or {}).get("observed_object"))
    return out


if __name__ == "__main__":
    main()
