#!/usr/bin/env python3
"""Qwen crop-representation comparison on E1 (conditions A / B / C, see config.yaml). Qwen only, sequential, no v0.2 change.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
  $PY experiments/qwen_crop_comparison/compare.py --dry-run     # crops + equality checks only, NO model call, writes a results folder
  $PY experiments/qwen_crop_comparison/compare.py               # 12 Qwen calls (4 control + 4 B + 4 C), GPU

Reused unchanged: v0.2 make_verify_crop (A crop regeneration), qwen_json + VERIFY_SCHEMA (the verify_box call path), ModelWorkerClient, the frozen prompt.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file, sha256_text  # noqa: E402
from semantic_task_nav.perception.grounding import VERIFY_SCHEMA, make_verify_crop, qwen_json  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_cfg():
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def make_crop(image, box, tcfg, pad_ratio, scale, marker, out_path):
    """Same geometry as v0.2 make_verify_crop; pad ratio, scale and marker are explicit. scale=None -> v0.2 rule (upscale to verify_crop_upscale_px)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    pad = max(pad_ratio * max(x2 - x1, y2 - y1), tcfg["verify_crop_min_px"] / 2)
    cx1, cy1, cx2, cy2 = int(max(0, x1 - pad)), int(max(0, y1 - pad)), int(min(w, x2 + pad)), int(min(h, y2 + pad))
    crop = image[cy1:cy2, cx1:cx2].copy()
    s = max(1.0, tcfg["verify_crop_upscale_px"] / max(crop.shape[:2])) if scale is None else scale
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    if marker:
        cv2.rectangle(crop, (round((x1 - cx1) * s), round((y1 - cy1) * s)), (round((x2 - cx1) * s), round((y2 - cy1) * s)), (0, 255, 0), 2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return {"path": str(out_path), "crop_xyxy_raw": [cx1, cy1, cx2, cy2], "scale": s, "size_wh": [crop.shape[1], crop.shape[0]], "marker": bool(marker),
            "pad_ratio": pad_ratio, "sha256": sha256_file(out_path)}


def classify(cube_id, conf_ids, answers):
    """answers: {cond: {cand_id: status}} -> per-condition label and the across-condition situation (rule declared in config.yaml)."""
    per = {}
    for cond, a in answers.items():
        cube = a[cube_id] == "yes"
        conf = [c for c in conf_ids if a[c] == "yes"]
        per[cond] = {"label": ("CUBE_AND_CONFUSABLE" if conf else "CUBE_ONLY") if cube else ("CONFUSABLE_ONLY" if conf else "NONE_ACCEPTED"),
                     "cube_status": a[cube_id], "confusables_accepted": conf}
    cube_dec = {c: per[c]["cube_status"] == "yes" for c in per}
    if not any(cube_dec.values()):
        overall = "SITUATION_3 (cube not accepted in any condition)"
    elif len(set(cube_dec.values())) > 1:
        overall = "SITUATION_4 (cube decision differs by condition; cause not established)"
    else:
        overall = "SITUATION_2 (confusable accepted)" if any(p["confusables_accepted"] for p in per.values()) else "SITUATION_1 (E1 only, not generalised)"
    return per, overall


def _t(img, text, org, color=(255, 255, 255), scale=0.5):
    text = str(text).encode("ascii", "replace").decode()
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(max(0, org[0])), int(max(th + 3, org[1]))
    cv2.rectangle(img, (x - 1, y - th - 2), (min(img.shape[1] - 1, x + tw + 1), y + bl), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def figure(rows, out):
    """rows: [{id, role, cells: {cond: {path, label}}}] -> grid, every tile resized to the same height for display only (the model saw the original sizes)."""
    H, tiles = 300, []
    for r in rows:
        line = []
        for cond in ("A", "B", "C"):
            c = r["cells"][cond]
            im = cv2.imread(c["path"])
            im = cv2.resize(im, (max(1, round(im.shape[1] * H / im.shape[0])), H), interpolation=cv2.INTER_AREA)
            im = cv2.copyMakeBorder(im, 0, 0, 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            _t(im, f"{cond}: {c['status']}  ({c['object']})", (4, 18), (0, 255, 0) if c["status"] == "yes" else (0, 200, 255) if c["status"] == "uncertain" else (180, 180, 255), 0.5)
            _t(im, c["size"], (4, H - 8), (200, 200, 200), 0.42)
            line.append(im)
        wmax = max(sum(t.shape[1] for t in line), 1)
        tiles.append((r, line, wmax))
    W = max(t[2] for t in tiles)
    canvas = []
    for r, line, _ in tiles:
        row = np.hstack(line)
        row = cv2.copyMakeBorder(row, 24, 6, 0, W - row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        _t(row, f"{r['id']}  retention rank {r['rank']}  role (AI preliminary): {r['role']}", (4, 16), (255, 255, 255), 0.55)
        canvas.append(row)
    cv2.imwrite(str(out), np.vstack(canvas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg, v02 = load_cfg(), load_config()
    mcfg, tcfg = v02["models"], v02["models"]["target_inspection"]
    img_path = Path(cfg["image"])
    assert sha256_file(img_path) == cfg["image_sha256"], "image differs from the declared one"
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    stored = (HERE / cfg["stored_run"]).resolve()
    assert sha256_file(stored / "input/raw_frame.png") == cfg["image_sha256"], "stored run used another image"
    ti = jload(stored / "analysis/target_inspection.json")
    desc = ti["description"]
    prompt = mcfg["prompts"]["verify_object"]["text"].replace("{description}", desc)
    assert mcfg["prompts"]["verify_object"]["id"] == cfg["prompt"]["id"] and sha256_text(prompt) == cfg["prompt"]["sha256"], "frozen prompt changed"
    assert mcfg["qwen"]["do_sample"] is False
    fp_before = code_fingerprint()["sha256"]
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_dryrun" if args.dry_run else "_crop_comparison"))
    out.mkdir(parents=True, exist_ok=False)
    (out / "config_snapshot.yaml").write_text((HERE / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    # ---- inputs and crops (offline)
    cands, checks = [], {}
    for c in cfg["candidates"]:
        rec = next(x for x in ti["candidates"] if x["candidate_id"] == c["id"])
        v = rec["verification"]
        box = rec["bbox_xyxy"]
        stored_crop = stored / v["crop"]["path"]
        (out / "crops").mkdir(exist_ok=True)
        shutil.copy2(stored_crop, out / "crops" / f"A_{c['id']}.png")
        a = {"path": str(out / "crops" / f"A_{c['id']}.png"), "crop_xyxy_raw": v["crop"]["crop_xyxy_raw"], "scale_rounded": v["crop"]["scale"], "marker": True, "pad_ratio": tcfg["verify_crop_pad_ratio"]}
        regen = make_verify_crop(image, box, tcfg, out / "crops" / f"_regen_A_{c['id']}.png")           # v0.2 function, unchanged
        regen_img, stored_img = cv2.imread(regen["path"]), cv2.imread(str(stored_crop))
        a["regenerated_equals_stored_pixels"] = bool(np.array_equal(regen_img, stored_img))
        a["regenerated_crop_xyxy_equal"] = regen["crop_xyxy_raw"] == v["crop"]["crop_xyxy_raw"]
        x0, y0, x1, y1 = regen["crop_xyxy_raw"]
        s_exact = max(1.0, tcfg["verify_crop_upscale_px"] / max(image[y0:y1, x0:x1].shape[:2]))          # the unrounded v0.2 scale
        b = make_crop(image, box, tcfg, tcfg["verify_crop_pad_ratio"], s_exact, False, out / "crops" / f"B_{c['id']}.png")
        cc = make_crop(image, box, tcfg, cfg["conditions"]["C"]["pad_ratio"], s_exact, True, out / "crops" / f"C_{c['id']}.png")
        own_a = make_crop(image, box, tcfg, tcfg["verify_crop_pad_ratio"], None, True, out / "crops" / f"_own_A_{c['id']}.png")
        a["own_function_reproduces_v02_crop"] = bool(np.array_equal(cv2.imread(own_a["path"]), regen_img))
        b_img = cv2.imread(b["path"])
        diff = np.any(b_img != regen_img, axis=2) if b_img.shape == regen_img.shape else None
        b["same_crop_range_and_scale_as_A"] = b["crop_xyxy_raw"] == regen["crop_xyxy_raw"] and diff is not None
        b["n_pixels_differing_from_A"] = int(diff.sum()) if diff is not None else None
        # B must differ from A ONLY on the rectangle line: rebuilding A from B by drawing the same rectangle must give A back exactly
        redraw = b_img.copy()
        cv2.rectangle(redraw, (round((box[0] - x0) * s_exact), round((box[1] - y0) * s_exact)), (round((box[2] - x0) * s_exact), round((box[3] - y0) * s_exact)), (0, 255, 0), 2)
        b["differs_from_A_only_at_marker_pixels"] = bool(np.array_equal(redraw, regen_img)) and b["n_pixels_differing_from_A"] > 0
        cc["candidate_scale_equals_A"] = cc["scale"] == s_exact
        cc["context_larger_than_A"] = cc["crop_xyxy_raw"][0] <= x0 and cc["crop_xyxy_raw"][1] <= y0 and cc["crop_xyxy_raw"][2] >= x1 and cc["crop_xyxy_raw"][3] >= y1
        for p in (out / "crops").glob("_*"):
            p.unlink()                                                # temporary regeneration files only
        cands.append({**c, "bbox_xyxy_raw_px": box, "coordinate_frame": "raw_image_px (undistorted E1 photo)", "dino_hsv_evidence": {"sources": rec["sources"], "member_boxes": rec["member_boxes"], "score": rec["score"], "shape_stats": rec.get("shape_stats")},
                      "stored_A": {"status": v["status"], "answer": v["answer"], "raw_response": v["call"]["raw_response"], "latency_s": v["call"]["latency_s"], "prompt_sha256": v["call"]["prompt_sha256"], "model": v["call"]["model"], "commit_hash": v["call"]["commit_hash"], "stored_crop": str(stored_crop)},
                      "crops": {"A": a, "B": b, "C": cc}})
    offline_ok = all(c["crops"]["A"]["regenerated_equals_stored_pixels"] and c["crops"]["A"]["own_function_reproduces_v02_crop"] and c["crops"]["B"]["same_crop_range_and_scale_as_A"]
                     and c["crops"]["B"]["differs_from_A_only_at_marker_pixels"] and c["crops"]["C"]["candidate_scale_equals_A"] and c["crops"]["C"]["context_larger_than_A"] for c in cands)
    print("offline checks ok:", offline_ok)
    if not offline_ok:
        (out / "plan.json").write_text(json.dumps({"offline_ok": False, "candidates": cands}, indent=2, default=str), encoding="utf-8")
        raise SystemExit("offline crop checks failed: no model call is made")
    if args.dry_run:
        (out / "plan.json").write_text(json.dumps({"offline_ok": True, "candidates": cands}, indent=2, default=str), encoding="utf-8")
        print("dry run written", out)
        return out

    # ---- live calls: Qwen only, sequential. A_control (stored crops), B, C.
    from semantic_task_nav.perception.model_client import ModelWorkerClient
    client = ModelWorkerClient(mcfg, out / "worker_logs")
    t0 = time.time()
    env = client.start()
    live = {}
    try:
        load = client.ensure_loaded("qwen")
        for cond in ("A_control", "B", "C"):
            for c in cands:
                path = c["crops"]["A" if cond == "A_control" else cond]["path"]
                tw = time.perf_counter()
                obj, rec = qwen_json(client, mcfg, "verify_object", prompt, path, mcfg["qwen"]["max_new_tokens"]["verify"], VERIFY_SCHEMA)
                rec["wall_s"] = round(time.perf_counter() - tw, 3)
                live[(cond, c["id"])] = {"status": obj["answer"] if obj else "invalid_response", "answer": obj, "call": rec}
                print(cond, c["id"], live[(cond, c["id"])]["status"], (obj or {}).get("observed_object"), rec["latency_s"])
    finally:
        client.close()

    # ---- assemble
    answers = {"A": {}, "B": {}, "C": {}}
    for c in cands:
        c["A_control"] = live[("A_control", c["id"])]
        c["A_control"]["raw_equals_stored"] = c["A_control"]["call"]["raw_response"] == c["stored_A"]["raw_response"]
        c["B"], c["C"] = live[("B", c["id"])], live[("C", c["id"])]
        answers["A"][c["id"]], answers["B"][c["id"]], answers["C"][c["id"]] = c["stored_A"]["status"], c["B"]["status"], c["C"]["status"]
    cube = next(c["id"] for c in cfg["candidates"] if c["role"] == "cube_estimate")
    conf = [c["id"] for c in cfg["candidates"] if c["role"] == "confusable"]
    per, overall = classify(cube, conf, answers)
    control_ok = all(c["A_control"]["raw_equals_stored"] for c in cands)
    rows = [{"id": c["id"], "rank": c["retention_rank"], "role": c["role"],
             "cells": {k: {"path": c["crops"][k]["path"], "status": (c["stored_A"]["status"] if k == "A" else c[k]["status"]),
                           "object": ((c["stored_A"]["answer"] if k == "A" else c[k]["answer"]) or {}).get("observed_object"), "size": "%dx%d px" % tuple(cv2.imread(c["crops"][k]["path"]).shape[1::-1])}
                     for k in ("A", "B", "C")}} for c in cands]
    figure(rows, out / "01_comparison.png")
    table = ["| candidate | rank | role (AI prelim.) | A stored | B no marker | C more context |", "|---|---|---|---|---|---|"]
    for c in cands:
        def cell(k):
            a = (c["stored_A"]["answer"] if k == "A" else c[k]["answer"]) or {}
            return f"{(c['stored_A']['status'] if k == 'A' else c[k]['status'])} ({a.get('observed_color')}, {a.get('observed_object')})"
        table.append(f"| {c['id']} | {c['retention_rank']} | {c['role']} | {cell('A')} | {cell('B')} | {cell('C')} |")
    (out / "comparison_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    result = {"experiment_id": cfg["experiment_id"], "created": datetime.now().isoformat(timespec="seconds"), "claims": cfg["claims"], "label_status": cfg["label_status"],
              "prompt": {"id": cfg["prompt"]["id"], "sha256": cfg["prompt"]["sha256"], "text_sent": prompt}, "image": {"path": str(img_path), "sha256": cfg["image_sha256"]},
              "model": {"load": load, "worker_env": env, "gpu_residency": "exclusive, Qwen only"}, "pipeline_control": {"all_raw_responses_equal_stored": control_ok},
              "decision_rule_applied": {"per_condition": per, "across_conditions": overall}, "table_markdown": table, "candidates": cands, "figure": "01_comparison.png",
              "offline_checks_ok": offline_ok, "total_wall_s": round(time.time() - t0, 1), "v02_code_before": fp_before, "v02_code_after": code_fingerprint()["sha256"]}
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "manifest.json").write_text(json.dumps({"files_sha256": files}, indent=2), encoding="utf-8")
    print("\n".join(table))
    print("control ok:", control_ok, "|", per, "|", overall)
    return out


if __name__ == "__main__":
    main()
