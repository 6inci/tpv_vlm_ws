#!/usr/bin/env python3
"""Follow-up of the SigLIP2 small-object test on the SAME crops as the first run (config_followup.yaml):
  1. reference-image (exemplar) search with SigLIP2 image embeddings (no text), one exemplar photo at a time
  2. label wording variants (DIAGNOSTIC)                     3. Qwen3-VL-4B on the same crops with the same closed label list
  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/siglip_small_object_test/run_followup.py --models siglip_base,qwen
  $PY experiments/siglip_small_object_test/run_followup.py --models siglip_so400m
  $PY experiments/siglip_small_object_test/run_followup.py --evaluate-only experiments/siglip_small_object_test/results/<run>
"""
import argparse
import json
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_test as T  # noqa: E402  (reference, scoring rules and figures of the first run)

FC = yaml.safe_load((HERE / "config_followup.yaml").read_text(encoding="utf-8"))
BASE = HERE / FC["base_run"]
MMDIR = (HERE / T.CFG["pipeline"]).resolve()
MMC = yaml.safe_load((MMDIR / "config.yaml").read_text(encoding="utf-8"))
VIEWS = FC["exemplar"]["views"]


def load_base():
    """The first run's photos: DINO candidates and the crop files SigLIP2 saw; reference objects from config.yaml (incl. additions)."""
    out = []
    for r in T.load_reference():
        d = json.loads((BASE / "raw" / f"{r['photo_id']}.json").read_text(encoding="utf-8"))
        assert d["sha256"] == r["sha256"]
        crops = {k: v["crops"] for k, v in {**d["siglip_candidates"], **d["siglip_reference_boxes"]}.items()}
        out.append({"photo_id": r["photo_id"], "photo": r["photo"], "path": d["path"], "objects": r["objects"], "dino": d["dino"],
                    "labels": d["labels"], "crops": crops})
    return out


def keys_of(p):
    """Boxes with crops of the first run (a reference object added afterwards has none)."""
    return [c["id"] for c in p["dino"]] + [o["key"] for o in p["objects"] if o["key"] in p["crops"]]


# ================================================================================ pure functions (tested offline)
def parse_label(text, labels):
    """Qwen reply -> (label in the list | None, error)."""
    m = re.search(r"\{.*?\}", text or "", re.S)
    if not m:
        return None, "NO_JSON"
    try:
        lb = str(json.loads(m.group(0)).get("label", "")).strip().rstrip(".").lower()
    except (json.JSONDecodeError, AttributeError):
        return None, "BAD_JSON"
    hit = next((x for x in labels if x.lower() == lb), None)
    return (hit, None) if hit else (None, f"NOT_IN_LIST: {lb[:60]}")


def rank_nms(cands, sims, nms_iou):
    kept = []
    for c in sorted(cands, key=lambda c: -sims[c["id"]]):
        if all(T.iou(c["bbox_xyxy"], k["bbox_xyxy"]) < nms_iou for k in kept):
            kept.append(c)
    return kept


def exemplar_eval(photo, sims, kind, exemplar_box, cfg):
    """Rank the photo's DINO candidates by similarity to one exemplar. kind 'pallet' (targets: pads) or 'cube'. None if no target is left."""
    kinds = ("pad_with_cube", "empty_pad") if kind == "pallet" else ("cube",)
    objs = photo["objects"]
    targets = [o for o in objs if o["kind"] in kinds and T.iou(o["box"], exemplar_box) < cfg["same_place_iou"]]
    left = [o for o in objs if o["kind"] in kinds and o not in targets]
    if not targets:
        return None
    cands = [c for c in photo["dino"] if all(T.iou(c["bbox_xyxy"], o["box"]) < cfg["match_iou"] for o in left)]
    hit = lambda c, o: T.iou(c["bbox_xyxy"], o["box"]) >= cfg["match_iou"]  # noqa: E731
    ranked = rank_nms(cands, sims, cfg["nms_iou"])
    n = len(targets)
    found = {o["key"] for o in targets for c in ranked[:n] if hit(c, o)}
    best_t = {o["key"]: max([sims[c["id"]] for c in cands if hit(c, o)], default=None) for o in targets}
    dis = [sims[c["id"]] for c in cands if all(T.iou(c["bbox_xyxy"], o["box"]) < cfg["distractor_max_iou"] for o in objs)]
    bd = max(dis) if dis else None
    r = {"kind": kind, "n_targets": n, "left_out_same_place": [o["key"] for o in left],
         "target_rank": {o["key"]: next((i + 1 for i, c in enumerate(ranked) if hit(c, o)), None) for o in targets},
         "target_kind": {o["key"]: o["kind"] for o in targets}, "recall_at_n": len(found) / n,
         "best_target_sim": {k: None if v is None else round(v, 4) for k, v in best_t.items()}, "best_distractor_sim": None if bd is None else round(bd, 4),
         "margin": None if bd is None or None in best_t.values() else round(min(best_t.values()) - bd, 4),
         "top": [{"id": c["id"], "box": [round(v, 1) for v in c["bbox_xyxy"]], "sim": round(sims[c["id"]], 4),
                  "matches": next((o["key"] for o in objs if hit(c, o)), None)} for c in ranked[:max(n, 3)]]}
    if kind == "cube":
        o, c0 = targets[0], ranked[0]
        cx, cy = (o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2
        b = c0["bbox_xyxy"]
        r["top1_hit"] = hit(c0, o)
        r["top1_contains_cube_centre"] = b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
    return r


def exemplar_all(photos, emb, cfg):
    cases = []
    for ep in photos:
        ex_pad = next((o for o in ep["objects"] if o["kind"] == "empty_pad" and (ep["photo_id"], o["key"]) in emb), None)   # exemplar needs a crop
        if ex_pad is None:
            continue
        ex_cube = next(o for o in ep["objects"] if o["kind"] == "cube")
        for kind, ex in (("pallet", ex_pad), ("cube", ex_cube)):
            X = emb[(ep["photo_id"], ex["key"])]
            for tp in photos:
                if tp is ep:
                    continue
                sims = {c["id"]: float(np.mean(np.sum(emb[(tp["photo_id"], c["id"])] * X, axis=1))) for c in tp["dino"]}
                r = exemplar_eval(tp, sims, kind, ex["box"], cfg)
                if r:
                    cases.append({"exemplar_photo": ep["photo"], "exemplar": ex["key"], "test_photo": tp["photo"], **r})
    s = {}
    for kind in ("pallet", "cube"):
        cs = [c for c in cases if c["kind"] == kind]
        ranks = [(c["target_kind"][k], rk, c["n_targets"]) for c in cs for k, rk in c["target_rank"].items()]
        s[kind] = {"cases": len(cs), "folds": len({c["exemplar_photo"] for c in cs}), "mean_recall_at_n": round(float(np.mean([c["recall_at_n"] for c in cs])), 3),
                   "cases_all_found_in_top_n": sum(c["recall_at_n"] == 1 for c in cs), "cases_margin_positive": sum((c["margin"] or -1) > 0 for c in cs),
                   "targets_ranked_within_n": {k: f"{sum(1 for kk, rk, n in ranks if kk == k and rk is not None and rk <= n)}/{sum(1 for kk, _, _ in ranks if kk == k)}"
                                               for k in sorted({kk for kk, _, _ in ranks})}}
        if kind == "cube":
            s[kind].update({"top1_hit": sum(c["top1_hit"] for c in cs), "top1_contains_cube_centre": sum(c["top1_contains_cube_centre"] for c in cs)})
    return {"summary": s, "cases": cases}


# ================================================================================ model stages
def composite(crops):
    ims = [cv2.resize(cv2.imread(crops[v]["path"]), (384, 384), interpolation=cv2.INTER_CUBIC) for v in VIEWS]
    return np.hstack([ims[0], np.full((384, 4, 3), 255, np.uint8), ims[1]])


def stage_siglip(mm, name, photos, raw, log):
    mc = FC["models"][name]
    info = mm.call("load", name="siglip", model_id=mc["model_id"], kind="siglip", dtype=mc["dtype"], device="cuda")
    log(f"{name} loaded: {info}")
    idx = [(p["photo_id"], k, v) for p in photos for k in keys_of(p) for v in VIEWS]
    pths = {p["photo_id"]: p["crops"] for p in photos}
    r = mm.call("siglip_images", paths=[pths[pid][k][v]["path"] for pid, k, v in idx], batch=16)
    labels = sorted({FC["label_variants"][vn].get(lb, lb) for vn in FC["label_variants"] for lb in photos[0]["labels"]})
    t = mm.call("siglip_texts", texts=[MMC["siglip"]["text_template"].format(label=lb) for lb in labels])
    mm.call("unload", names=["siglip"])
    np.savez_compressed(raw / f"embeddings_{name}.npz", image=np.array(r["embeddings"], np.float32), text=np.array(t["embeddings"], np.float32))
    (raw / f"embeddings_{name}.json").write_text(json.dumps({"index": idx, "labels": labels, "logit_scale": t["logit_scale"], "logit_bias": t["logit_bias"],
                                                            "model": r["model"], "commit_hash": r["commit_hash"], "seconds": r["seconds"], "device": r["device"],
                                                            "template": MMC["siglip"]["text_template"]}, indent=1), encoding="utf-8")
    log(f"{name}: {r['n']} crops in {r['seconds']} s")
    return info


def stage_qwen(client, P, photos, raw, log):
    labels = photos[0]["labels"]
    prompt = FC["qwen"]["prompt"].format(labels="\n".join(f"- {lb}" for lb in labels))
    (raw / "qwen_prompt.txt").write_text(prompt, encoding="utf-8")
    d = raw / "qwen_inputs"
    d.mkdir()
    calls, n, t0 = [], sum(len(keys_of(p)) for p in photos), time.perf_counter()
    with open(raw / "qwen_replies.jsonl", "w", encoding="utf-8") as f:
        for p in photos:
            for k in keys_of(p):
                ip = d / f"{p['photo_id']}_{k}.png"
                cv2.imwrite(str(ip), composite(p["crops"][k]))
                txt = P.qwen_call(client, prompt, ip, FC["qwen"]["max_new_tokens"], f"qwen_{p['photo_id']}_{k}", "siglip_followup_closed_set_v1", calls)
                lb, err = parse_label(txt, labels)
                f.write(json.dumps({"photo_id": p["photo_id"], "key": k, "label": lb, "error": err, "raw": txt}, ensure_ascii=False) + "\n")
                if len(calls) % 100 == 0:
                    log(f"qwen {len(calls)}/{n} ({time.perf_counter() - t0:.0f} s)")
    (raw / "qwen_model_calls.json").write_text(json.dumps(calls, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"qwen: {len(calls)} calls in {time.perf_counter() - t0:.0f} s")


# ================================================================================ evaluation
def score(photos, rank_of, ev):
    """rank_of(photo, key) -> {'combined': [(label, score), ...]} ; same scoring as the first run."""
    per = []
    for p in photos:
        cr = {c["id"]: rank_of(p, c["id"]) for c in p["dino"]}
        objs = [T.evaluate_object(o, p["dino"], cr, rank_of(p, o["key"]), ev) for o in p["objects"]]
        per.append({"photo": p["photo"], "photo_id": p["photo_id"], "n_dino": len(p["dino"]), "objects": objs,
                    "named_targets": T.named_targets(p["dino"], cr, p["objects"], ev)})
    return per, T.summarise(per, ev)


def short(s):
    keep = ("n", "detected", "identified_top1", "identified_topk", "oracle_top1_correct")
    return {k: ({kk: s[k][kk] for kk in keep} if k != "named_targets" else s[k]) for k in s}


def evaluate(run_dir, out_dir, photos, log):
    """Reads run_dir/raw, writes evaluation_followup.json and figures to out_dir (run_dir itself only for the first evaluation)."""
    sys.path.insert(0, str(MMDIR))
    import mm_pipeline as MM  # siglip_rank (pure function) of the first run
    raw, out, figs = run_dir / "raw", {}, out_dir / "figures"
    figs.mkdir(exist_ok=True)
    paths = {p["photo_id"]: p["path"] for p in photos}
    for name in ("siglip_base", "siglip_so400m"):
        if not (raw / f"embeddings_{name}.npz").exists():
            continue
        z, meta = np.load(raw / f"embeddings_{name}.npz"), json.loads((raw / f"embeddings_{name}.json").read_text(encoding="utf-8"))
        I = z["image"]
        pos = {tuple(k): i for i, k in enumerate(meta["index"])}
        emb = {(pid, k): np.stack([I[pos[(pid, k, v)]] for v in VIEWS]) for pid, k, _ in meta["index"]}
        tix = {lb: i for i, lb in enumerate(meta["labels"])}
        ex = exemplar_all(photos, emb, FC["exemplar"])
        out[f"{name}/exemplar"] = ex
        log(f"{name} exemplar: {json.dumps(ex['summary'])}")
        for vn, rep in FC["label_variants"].items():
            labels = [rep.get(lb, lb) for lb in photos[0]["labels"]]
            ev = {**T.CFG["evaluation"], "groups": {g: [rep.get(lb, lb) for lb in ls] for g, ls in T.CFG["evaluation"]["groups"].items()}}
            Tm = z["text"][[tix[lb] for lb in labels]]
            per, s = score(photos, lambda p, k: MM.siglip_rank(emb[(p["photo_id"], k)], Tm, meta["logit_scale"], meta["logit_bias"], labels)
                           if (p["photo_id"], k) in emb else None, ev)
            out[f"{name}/text_{vn}"] = {"labels_changed": rep, "summary": s, "photos": per}
            log(f"{name} text {vn}: {json.dumps(short(s))}")
            if vn == "V0_original":
                T.gallery(per, paths, figs / f"gallery_{name}_text.png")
        exemplar_figure(photos, ex, figs, name)
    if (raw / "qwen_replies.jsonl").exists():
        rep = {(r["photo_id"], r["key"]): r for r in map(json.loads, (raw / "qwen_replies.jsonl").read_text(encoding="utf-8").splitlines())}
        per, s = score(photos, lambda p, k: {"combined": [(rep[(p["photo_id"], k)]["label"] or "INVALID", 1.0)]} if (p["photo_id"], k) in rep else None,
                       T.CFG["evaluation"])
        inv = sum(r["label"] is None for r in rep.values())
        out["qwen3vl_4b/closed_set"] = {"invalid_replies": inv, "replies": len(rep), "summary": s, "photos": per}
        log(f"qwen: invalid {inv}/{len(rep)}; {json.dumps(short(s))}")
        T.gallery(per, paths, figs / "gallery_qwen3vl_4b.png")
    (out_dir / "evaluation_followup.json").write_text(json.dumps({"reference_status": T.CFG["reference"]["status"],
                                                                 "reference_objects": [{"photo": p["photo"], "objects": p["objects"]} for p in photos],
                                                                 "results": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def exemplar_figure(photos, ex, figs, name):
    """Exemplar photo_1 (first fold): per test photo the exemplar and the top-3 candidates (tight crops). green = reference object."""
    first = ex["cases"][0]["exemplar_photo"]
    crops = {(p["photo"], k): p["crops"][k]["tight"]["path"] for p in photos for k in p["crops"]}
    for kind in ("pallet", "cube"):
        rows = []
        for c in [c for c in ex["cases"] if c["exemplar_photo"] == first and c["kind"] == kind]:
            tiles = [(crops[(first, c["exemplar"])], f"exemplar {first}", (0, 220, 255))]
            tiles += [(crops[(c["test_photo"], t["id"])], f"#{i + 1} {t['sim']:.3f}", (0, 200, 0) if t["matches"] else (0, 0, 255)) for i, t in enumerate(c["top"][:3])]
            row = []
            for pth, txt, col in tiles:
                im = cv2.resize(cv2.imread(pth), (150, 150), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(im, (0, 0), (149, 149), col, 3)
                cv2.putText(im, txt, (5, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
                row.append(im)
            lab = np.zeros((150, 120, 3), np.uint8)
            cv2.putText(lab, c["test_photo"], (6, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(lab, f"recall {c['recall_at_n']:.1f}", (6, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            rows.append(np.hstack([lab] + row))
        head = np.zeros((30, rows[0].shape[1], 3), np.uint8)
        cv2.putText(head, f"{name} {kind}: exemplar = {first} reference crop; top 3 DINO candidates by image similarity (green = a reference object)",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(figs / f"exemplar_{name}_{kind}_{first}.png"), np.vstack([head] + rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="siglip_base,qwen")
    ap.add_argument("--evaluate-only", metavar="RUN_DIR")
    a = ap.parse_args()
    photos = load_base()
    models = [m for m in a.models.split(",") if m]
    run_dir = Path(a.evaluate_only).resolve() if a.evaluate_only else HERE / "results" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_followup_" + "_".join(models))
    run_dir.mkdir(parents=True, exist_ok=bool(a.evaluate_only))
    out_dir = run_dir / ("reeval_" + datetime.now().strftime("%Y%m%d_%H%M%S")) if a.evaluate_only else run_dir   # never overwrite an evaluation
    out_dir.mkdir(exist_ok=bool(not a.evaluate_only))
    logf = open(out_dir / ("evaluate.log" if a.evaluate_only else "run.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if not a.evaluate_only:
        sys.path.insert(0, str(MMDIR))
        import mm_pipeline as MM
        (run_dir / "config_followup_snapshot.yaml").write_text((HERE / "config_followup.yaml").read_text(encoding="utf-8"), encoding="utf-8")
        raw = run_dir / "raw"
        raw.mkdir()
        t0, env, info = time.perf_counter(), {}, {}
        sig = [m for m in models if m.startswith("siglip")]
        if sig:
            mm = MM.MMClient(MMC["worker"], run_dir / "worker_logs")
            try:
                env["mm_worker"] = mm.start()
                for m in sig:
                    info[m] = stage_siglip(mm, m, photos, raw, log)
            finally:
                mm.close()
        if "qwen" in models:
            client = MM.ModelWorkerClient(MM.VP.load_config()["models"], run_dir / "worker_logs")
            try:
                env["v02_worker"] = client.start()
                stage_qwen(client, MM.P, photos, raw, log)
            finally:
                client.close()
        (run_dir / "manifest.json").write_text(json.dumps({
            "run_id": run_dir.name, "created": datetime.now().astimezone().isoformat(timespec="seconds"), "wall_s": round(time.perf_counter() - t0, 1),
            "models": models, "model_info": info, "python": platform.python_version(), "base_run": str(BASE),
            "git_commit": "NONE (vlm_ws is not a git repository; file hashes below)",
            "code_sha256": {p.name: T.sha(p) for p in (HERE / "run_followup.py", HERE / "run_test.py", MMDIR / "mm_worker.py", MMDIR / "mm_pipeline.py")},
            "config_sha256": {"config_followup.yaml": T.sha(HERE / "config_followup.yaml"), "config.yaml": T.sha(HERE / "config.yaml")},
            "worker_env": env}, indent=1, default=str), encoding="utf-8")
    evaluate(run_dir, out_dir, photos, log)
    log(f"results: {out_dir}")


if __name__ == "__main__":
    main()
