#!/usr/bin/env python3
"""SigLIP2 small-object test: the same 8 photos and the same Grounding DINO pass as multimodel_target_viewer; SigLIP2 names every
DINO candidate (the job Qwen did before). No Qwen call. Scored against the AI preliminary reference (small pads, red cubes).

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/siglip_small_object_test/run_test.py                                  # models + evaluation -> results/<time>_run/
  $PY experiments/siglip_small_object_test/run_test.py --evaluate-only results/<run>    # evaluation again from the saved raw data
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load_reference(cfg=CFG):
    """8 photos -> reference objects (AI preliminary review): the red cube, the pad under it, the empty pads."""
    funnel = yaml.safe_load((HERE / cfg["reference"]["cubes"]).read_text(encoding="utf-8"))
    survey = {p["sha256"]: p for p in json.loads((HERE / cfg["reference"]["pads"]).read_text(encoding="utf-8"))["photos"]}
    out = []
    for im in funnel["images"]:
        s = survey[im["sha256"]]
        objs = [{"key": "refobj_cube", "kind": "cube", "box": funnel["labels"][im["id"]]},
                {"key": "refobj_pad_with_cube", "kind": "pad_with_cube", "box": s["cube"]["box_px"]}]
        objs += [{"key": f"refobj_empty_pad_{i}", "kind": "empty_pad", "box": e["box_px"]} for i, e in enumerate(s["pallets_empty"])]
        objs += [{"key": f"refobj_added_{i}", **a} for i, a in enumerate((cfg["reference"].get("additions") or {}).get(im["id"], []))]
        out.append({"photo_id": im["id"], "photo": s["photo"], "sha256": im["sha256"], "objects": objs})
    return out


def group_of(label, ev):
    return next((g for g, labels in ev["groups"].items() if label in labels), None)


def evaluate_object(obj, cands, cand_rank, oracle_rank, ev):
    """One reference object: best-IoU DINO candidate, detection, SigLIP2 naming of that candidate, SigLIP2 naming of the reference box."""
    want = ev["object_group"][obj["kind"]]
    best, bi = max(((c, iou(obj["box"], c["bbox_xyxy"])) for c in cands), key=lambda t: t[1], default=(None, 0.0))
    r = {"key": obj["key"], "kind": obj["kind"], "box": obj["box"], "box_size_px": [obj["box"][2] - obj["box"][0], obj["box"][3] - obj["box"][1]],
         "expected_group": want, "best_iou": round(bi, 3), "detected": bi >= ev["detection_iou"], "detected_lenient": bi >= ev["detection_iou_lenient"]}
    if best is not None:
        top = cand_rank[best["id"]]["combined"]
        r.update({"candidate": best["id"], "candidate_box": [round(v, 1) for v in best["bbox_xyxy"]], "dino_score": round(best["score"], 3),
                  "dino_phrases": best["phrases"], "siglip_candidate_top3": top[:3]})
    r["identified_top1"] = r["detected"] and group_of(r["siglip_candidate_top3"][0][0], ev) == want
    r["identified_topk"] = r["detected"] and any(group_of(lb, ev) == want for lb, _ in cand_rank[best["id"]]["combined"][:ev["also_report_top_k"]])
    o = oracle_rank["combined"] if oracle_rank else None             # None: no crop of this reference box (object added after the run)
    r["siglip_oracle_top3"] = o[:3] if o else None
    r["oracle_top1_correct"] = group_of(o[0][0], ev) == want if o else None
    r["oracle_topk_correct"] = any(group_of(lb, ev) == want for lb, _ in o[:ev["also_report_top_k"]]) if o else None
    r["outcome"] = "IDENTIFIED" if r["identified_top1"] else "NAMED_OTHER" if r["detected"] else "NOT_DETECTED"
    return r


def named_targets(cands, cand_rank, objects, ev):
    """DINO candidates whose SigLIP2 top-1 is a pallet or cube label, with the reference object they match (if any)."""
    out = []
    for c in cands:
        lb, sc = cand_rank[c["id"]]["combined"][0]
        g = group_of(lb, ev)
        if g is None:
            continue
        o, oi = max(((o, iou(o["box"], c["bbox_xyxy"])) for o in objects), key=lambda t: t[1], default=(None, 0.0))
        out.append({"id": c["id"], "box": [round(v, 1) for v in c["bbox_xyxy"]], "named": lb, "score": sc, "group": g,
                    "matches_reference": o["key"] if o is not None and oi >= ev["detection_iou"] else None, "reference_iou": round(oi, 3)})
    return out


def summarise(per_photo, ev):
    rows = {}
    for kind in ("cube", "pad_with_cube", "empty_pad"):
        rs = [r for p in per_photo for r in p["objects"] if r["kind"] == kind]
        rows[kind] = {"n": len(rs), "n_with_oracle_crop": sum(r["siglip_oracle_top3"] is not None for r in rs), **{k: sum(bool(r[k]) for r in rs) for k in
                      ("detected", "detected_lenient", "identified_top1", "identified_topk", "oracle_top1_correct", "oracle_topk_correct")},
                      "top1_labels_of_detected": dict(sorted({lb: sum(1 for r in rs if r["detected"] and r["siglip_candidate_top3"][0][0] == lb)
                                                              for lb in {r["siglip_candidate_top3"][0][0] for r in rs if r["detected"]}}.items(), key=lambda t: -t[1])),
                      "oracle_top1_labels": dict(sorted({lb: sum(1 for r in rs if r["siglip_oracle_top3"] and r["siglip_oracle_top3"][0][0] == lb)
                                                         for lb in {r["siglip_oracle_top3"][0][0] for r in rs if r["siglip_oracle_top3"]}}.items(), key=lambda t: -t[1]))}
    nt = [t for p in per_photo for t in p["named_targets"]]
    rows["named_targets"] = {g: {"total": sum(t["group"] == g for t in nt), "matching_reference": sum(t["group"] == g and t["matches_reference"] is not None for t in nt)}
                             for g in ev["groups"]}
    return rows


# ================================================================================ figures
def gallery(per_photo, paths, out):
    tiles = []
    for p in per_photo:
        img = cv2.imread(paths[p["photo_id"]], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        for r in p["objects"]:
            b = r["box"]
            cx, cy, s = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2), 60
            x0, y0 = max(0, cx - s), max(0, cy - s)
            crop = img[y0:y0 + 2 * s, x0:x0 + 2 * s].copy()
            cv2.rectangle(crop, (b[0] - x0, b[1] - y0), (b[2] - x0, b[3] - y0), (0, 255, 255), 1)
            if r.get("candidate_box"):
                cb = [int(round(v)) for v in r["candidate_box"]]
                col = (0, 200, 0) if r["identified_top1"] else (0, 0, 255) if r["detected"] else (150, 150, 150)
                cv2.rectangle(crop, (cb[0] - x0, cb[1] - y0), (cb[2] - x0, cb[3] - y0), col, 1)
            crop = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_NEAREST)
            txt = np.zeros((84, 240, 3), np.uint8)
            lines = [(f"{p['photo']} {r['kind']} {r['box_size_px'][0]}x{r['box_size_px'][1]}px", (255, 255, 255)),
                     (f"{r['outcome']}  DINO IoU {r['best_iou']:.2f}", (0, 200, 0) if r["outcome"] == "IDENTIFIED" else (80, 80, 255)),
                     ("cand: " + (" / ".join(f"{lb[2:] if lb.startswith('a ') else lb} {s:.2f}" for lb, s in r["siglip_candidate_top3"][:2]) if r["detected"] else "(not detected)"), (230, 230, 230)),
                     ("ref box: " + (" / ".join(f"{lb[2:] if lb.startswith('a ') else lb} {s:.2f}" for lb, s in r["siglip_oracle_top3"][:2])
                                     if r["siglip_oracle_top3"] else "(no crop: added after the run)"), (0, 220, 255))]
            for i, (t, c) in enumerate(lines):
                cv2.putText(txt, t[:44], (4, 17 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.40, c, 1, cv2.LINE_AA)
            tiles.append(np.vstack([crop, txt]))
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    head = np.zeros((44, 240 * 6, 3), np.uint8)
    cv2.putText(head, "yellow: reference (AI preliminary)  green: DINO box named correctly  red: detected, named other  grey: best box, IoU < 0.5",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(head, "cand: SigLIP2 top 2 of the DINO box   ref box: SigLIP2 top 2 of the reference box itself (relative scores among the labels, not probabilities)",
                (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), np.vstack([head] + [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))


def named_map(p, path, out):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    for r in p["objects"]:
        b = r["box"]
        cv2.rectangle(img, (b[0] - 3, b[1] - 3), (b[2] + 3, b[3] + 3), (0, 255, 255), 1)
    for t in p["named_targets"]:
        b = [int(round(v)) for v in t["box"]]
        col = (0, 0, 255) if t["group"] == "cube" else (255, 120, 0)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), col, 2 if t["matches_reference"] is None else 1)
        cv2.putText(img, f"{t['id']} {t['named'][2:] if t['named'].startswith('a ') else t['named']}", (b[0], max(12, b[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.putText(img, f"{p['photo']}: DINO candidates SigLIP2 names a pallet (blue) or a red cube (red); thick = matches no reference; yellow = reference",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), img)


# ================================================================================ stages
def run_models(run_dir, refs, log):
    sys.path.insert(0, str((HERE / CFG["pipeline"]).resolve()))
    import mm_pipeline as MM  # read-only import; MMViewer.photo is the unchanged per-photo stage
    v = MM.MMViewer(session_dir=run_dir / "session", progress=log)
    paths = {p["photo_id"]: p["path"] for p in v.list_photos()}
    raw = run_dir / "raw"
    raw.mkdir()
    env = {}
    try:
        for ref in refs:
            t0 = time.perf_counter()
            log(f"== {ref['photo']} ({ref['photo_id']})")
            ph = v.photo(paths[ref["photo_id"]])
            labels = list(v.mc["siglip"]["labels"])
            cr = v.siglip(ph, [(c["id"], c["bbox_xyxy"]) for c in ph["dino"]], labels, ph["calls"], "eval")           # cached embeddings
            orr = v.siglip(ph, [(o["key"], o["box"]) for o in ref["objects"]], labels, ph["calls"], f"oracle_{ref['photo_id']}")
            v._save_photo(ph)
            (raw / f"{ref['photo_id']}.json").write_text(json.dumps(
                {"photo_id": ref["photo_id"], "path": paths[ref["photo_id"]], "sha256": ph["sha256"], "labels": labels, "dino": ph["dino"],
                 "siglip_candidates": cr, "siglip_reference_boxes": orr, "facility_refs": ph["refs"], "wall_s": round(time.perf_counter() - t0, 1)},
                indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            log(f"   DINO candidates {len(ph['dino'])}, {time.perf_counter() - t0:.0f} s")
        env = {"v02_worker": getattr(v.client, "env_info", None), "mm_worker": getattr(v.mm, "env_info", None)}
    finally:
        v.close()
    return env


def evaluate(run_dir, out_dir, refs, log):
    """Reads run_dir/raw, writes evaluation.json and figures to out_dir (run_dir itself only for the first evaluation)."""
    ev, per_photo, paths = CFG["evaluation"], [], {}
    for ref in refs:
        d = json.loads((run_dir / "raw" / f"{ref['photo_id']}.json").read_text(encoding="utf-8"))
        assert d["sha256"] == ref["sha256"], f"{ref['photo_id']}: photo hash differs from the reference"
        paths[ref["photo_id"]] = d["path"]
        objs = [evaluate_object(o, d["dino"], d["siglip_candidates"], d["siglip_reference_boxes"].get(o["key"]), ev) for o in ref["objects"]]
        per_photo.append({"photo": ref["photo"], "photo_id": ref["photo_id"], "n_dino": len(d["dino"]), "objects": objs,
                          "named_targets": named_targets(d["dino"], d["siglip_candidates"], ref["objects"], ev)})
    summary = summarise(per_photo, ev)
    (out_dir / "evaluation.json").write_text(json.dumps({"reference_status": CFG["reference"]["status"], "reference_objects": refs, "rules": ev,
                                                        "summary": summary, "photos": per_photo}, indent=1, ensure_ascii=False), encoding="utf-8")
    fig = out_dir / "figures"
    fig.mkdir(exist_ok=True)
    gallery(per_photo, paths, fig / "reference_objects_gallery.png")
    for p in per_photo:
        named_map(p, paths[p["photo_id"]], fig / f"named_targets_{p['photo']}.png")
    for kind, s in summary.items():
        log(f"{kind}: {json.dumps(s, ensure_ascii=False)}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate-only", metavar="RUN_DIR")
    a = ap.parse_args()
    refs = load_reference()
    run_dir = Path(a.evaluate_only).resolve() if a.evaluate_only else HERE / CFG["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_run")
    run_dir.mkdir(parents=True, exist_ok=bool(a.evaluate_only))
    out_dir = run_dir / ("reeval_" + datetime.now().strftime("%Y%m%d_%H%M%S")) if a.evaluate_only else run_dir   # never overwrite an evaluation
    out_dir.mkdir(exist_ok=False) if a.evaluate_only else None
    logf = open(out_dir / ("evaluate.log" if a.evaluate_only else "run.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if not a.evaluate_only:
        (run_dir / "config_snapshot.yaml").write_text((HERE / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "reference_snapshot.json").write_text(json.dumps(refs, indent=1), encoding="utf-8")
        t0 = time.perf_counter()
        env = run_models(run_dir, refs, log)
        pipe = (HERE / CFG["pipeline"]).resolve()
        (run_dir / "manifest.json").write_text(json.dumps({
            "run_id": run_dir.name, "created": datetime.now().astimezone().isoformat(timespec="seconds"), "wall_s": round(time.perf_counter() - t0, 1),
            "python": platform.python_version(), "git_commit": "NONE (vlm_ws is not a git repository; file hashes below)",
            "code_sha256": {str(p.relative_to(HERE.parent)): sha(p) for p in (HERE / "run_test.py", pipe / "mm_pipeline.py", pipe / "mm_worker.py",
                                                                             HERE.parent / "instruction_target_viewer/viewer_pipeline.py")},
            "config_sha256": {"config.yaml": sha(HERE / "config.yaml"), "multimodel_target_viewer/config.yaml": sha(pipe / "config.yaml")},
            "photos_sha256": {r["photo"]: r["sha256"] for r in refs}, "worker_env": env,
            "models": "see session/photos/<id>.json 'calls' (model id, commit hash, device, seconds per call)"}, indent=1, default=str), encoding="utf-8")
    evaluate(run_dir, out_dir, refs, log)
    log(f"results: {out_dir}")


if __name__ == "__main__":
    main()
