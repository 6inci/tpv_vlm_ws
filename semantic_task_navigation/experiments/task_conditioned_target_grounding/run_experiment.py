#!/usr/bin/env python3
"""Task-conditioned source-destination grounding on E1 (Exp A-F, see config.yaml). Real model calls; nothing is replayed.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
  $PY experiments/task_conditioned_target_grounding/run_experiment.py --dry-run   # ROIs, prompts, existing records; NO model call
  $PY experiments/task_conditioned_target_grounding/run_experiment.py             # Qwen + Grounding DINO, sequential (GPU)

Order: B1 existing records -> A interpretation (Qwen, 6) -> B2 detection (DINO, 6) -> C role analysis (Qwen, <= 40 crops) ->
C/D task selection (Qwen, 4) -> E route B (offline) -> F ablation -> evaluate (the ONLY reader of evaluation_reference.yaml).
Reused unchanged: v0.2 ModelWorkerClient, make_roi / zoom_roi, make_verify_crop, parse_json_object / validate, iou;
placement_target_grounding zone_crop_rect / qwen_box_to_image / norm_phrase / detections / merge / qwen_call / dino_call / drawing helpers;
semantic_navigation_integration leg_to_region / leg_record / Raster; task_route_integration build_context / layers / draw_path.
"""
import argparse
import csv
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
sys.path.insert(0, str(HERE.parent / "placement_target_grounding"))
import placement_target as P  # noqa: E402  (also puts task_route_integration and semantic_navigation_integration on sys.path)
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file, sha256_text  # noqa: E402
from semantic_task_nav.geometry.coordinate_frames import box_area, containment, iou  # noqa: E402
from semantic_task_nav.perception.grounding import make_verify_crop  # noqa: E402
from semantic_task_nav.perception.target_inspection import make_roi, zoom_roi  # noqa: E402

RI, SN, PT, FN = P.RI, P.SN, P.PT, P.FN
_S, _SN = {"type": "string"}, {"type": ["string", "null"]}
_BOX = {"type": ["array", "null"]}
INTERP_SCHEMA = {"type": "object", "required": ["source_region", "source_object", "destination_region", "destination_target_candidates"],
                 "properties": {"source_region": {"type": "object"}, "source_object": {"type": "object"}, "destination_region": {"type": "object"},
                                "destination_target_candidates": {"type": "array", "items": {"type": "object", "required": ["name_en"],
                                                                                              "properties": {"name_en": _S, "bbox_2d": _BOX}}},
                                "selected_destination": _SN}}
NEUTRAL_SCHEMA = {"type": "object", "required": ["support_candidates"],
                  "properties": {"objects": {"type": "array"}, "support_candidates": {"type": "array", "items": {"type": "object", "required": ["name_en"]}}}}
ROLE_SCHEMA = {"type": "object", "required": ["object_type", "is_support_surface", "load_state"],
               "properties": {"object_type": _S, "is_support_surface": {"type": "string", "enum": ["yes", "no", "uncertain"]},
                              "load_state": {"type": "string", "enum": ["empty", "occupied", "not_applicable", "uncertain"]}}}
SEL_SCHEMA = {"type": "object", "required": ["source_id", "destination_id"],
              "properties": {"source_id": {"type": ["number", "null"]}, "destination_id": {"type": ["number", "null"]}, "candidates": {"type": "array"}}}
STATUSES = ("SELECTED", "AMBIGUOUS", "NOT_FOUND", "UNVERIFIED")


# ================================================================================ pure functions (tested offline)
def map_boxes(obj, rect):
    """Add bbox_image_px to every bbox_2d of an interpretation / neutral reply (0-1000 of the image the model saw -> E1 px)."""
    if not obj:
        return obj
    for k in ("source_region", "source_object", "destination_region"):
        if isinstance(obj.get(k), dict):
            obj[k]["bbox_image_px"] = P.qwen_box_to_image(obj[k].get("bbox_2d"), rect)
    for k in ("destination_target_candidates", "objects", "support_candidates"):
        for c in obj.get(k) or []:
            if isinstance(c, dict):
                c["bbox_image_px"] = P.qwen_box_to_image(c.get("bbox_2d"), rect)
    return obj


def interp_phrases(parsed, max_words):
    """Qwen-generated names of every call -> phrases (ordered, unique) with their origins."""
    origins = {}
    for call, obj in parsed.items():
        if not obj:
            continue
        items = []
        if isinstance(obj.get("source_object"), dict) and obj["source_object"].get("name_en"):
            items.append(("source_object", obj["source_object"]["name_en"]))
        items += [("destination_candidate", c.get("name_en")) for c in obj.get("destination_target_candidates") or [] if isinstance(c, dict)]
        items += [("neutral_object", c.get("name_en")) for c in obj.get("objects") or [] if isinstance(c, dict)]
        items += [("neutral_support", c.get("name_en")) for c in obj.get("support_candidates") or [] if isinstance(c, dict)]
        for kind, name in items:
            p = P.norm_phrase(name or "", max_words)
            if p:
                origins.setdefault(p, []).append({"call": call, "kind": kind})
    return list(origins), origins


def bucket(box, rects):
    for name in ("DST", "SRC"):
        if P.centre_in(box, rects[name]):
            return name
    return "OTHER"


def role_queue(merged, rects, budget):
    """Candidates to role-analyse, per bucket by max detector score; the rest BEYOND_BUDGET (kept)."""
    out, beyond = [], []
    for b in ("DST", "SRC", "OTHER"):
        cs = sorted([c for c in merged if bucket(c["bbox_xyxy"], rects) == b], key=lambda c: -c["score"])
        out += [(c, b) for c in cs[:budget[b]]]
        beyond += [(c, b) for c in cs[budget[b]:]]
    return out, beyond


def numbered_set(analysed, max_n):
    keep = [a for a in analysed if a["role"] and (a["role"]["is_support_surface"] in ("yes", "uncertain") or a["role"]["load_state"] == "occupied")]
    return keep[:max_n]


def decide(sel, numbered):
    """Python decision for one instruction from the Qwen selection reply (sel) and the role analysis of the numbered set."""
    out = {"source_status": None, "source": None, "destination_status": None, "destination": None, "notes": []}
    if not numbered:
        out.update({"source_status": "NOT_FOUND", "destination_status": "NOT_FOUND", "notes": ["numbered set empty"]})
        return out
    if sel is None:
        out.update({"source_status": "AMBIGUOUS", "destination_status": "AMBIGUOUS", "notes": ["selection reply invalid"]})
        return out

    def pick(v):
        return numbered[int(v) - 1] if isinstance(v, (int, float)) and float(v).is_integer() and 1 <= int(v) <= len(numbered) else None
    s, d = pick(sel.get("source_id")), pick(sel.get("destination_id"))
    out["source_status"], out["source"] = ("SELECTED", s["id"]) if s else ("AMBIGUOUS", None)
    if sel.get("source_id") is not None and not s:
        out["notes"].append(f"source_id {sel.get('source_id')} is not a shown number")
    if d is None:
        out["destination_status"] = "AMBIGUOUS"
        if sel.get("destination_id") is not None:
            out["notes"].append(f"destination_id {sel.get('destination_id')} is not a shown number")
        return out
    r = d["role"]
    if r["is_support_surface"] == "no" or r["load_state"] == "occupied":
        out.update({"destination_status": "UNVERIFIED", "destination": d["id"],
                    "notes": out["notes"] + [f"role analysis contradicts a destination: support {r['is_support_surface']}, load {r['load_state']}"]})
    else:
        out.update({"destination_status": "SELECTED", "destination": d["id"]})
    if s and d and s["id"] == d["id"]:
        out["notes"].append("source and destination are the same candidate")
    return out


# ================================================================================ figures
def numbered_image(img, cands, path):
    im = img.copy()
    for k, c in enumerate(cands, 1):
        b = [int(round(v)) for v in c["bbox_xyxy"]]
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(str(k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x, y = b[0], max(th + 4, b[1] - 3)
        cv2.rectangle(im, (x, y - th - 4), (x + tw + 6, y + 2), (0, 0, 0), -1)
        cv2.putText(im, str(k), (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), im)


def tile_grid(items, path, per_row=8, size=180):
    tiles = []
    for it in items:
        im = cv2.imread(it["path"]) if it.get("path") else np.zeros((size, size, 3), np.uint8)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        im = cv2.copyMakeBorder(im, 44, 0, 0, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        P._t(im, it["title"][:30], (2, 14), (255, 255, 255), 0.4)
        P._t(im, it["line2"][:30], (2, 30), it.get("col", (0, 255, 255)), 0.4)
        tiles.append(im)
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    cv2.imwrite(str(path), np.vstack(rows))


# ================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    pcfg = yaml.safe_load((HERE / cfg["placement_config"]).read_text(encoding="utf-8"))
    rcfg, gcfg, v02 = RI.load_cfg(), PT.load_cfg(), load_config()
    mcfg = v02["models"]
    assert cfg["instructions"]["T1"]["text"] == rcfg["command"] and mcfg["qwen"]["do_sample"] is False
    assert cfg["detection"]["thresholds"] == {"box": mcfg["grounding_dino"]["box_threshold"], "text": mcfg["grounding_dino"]["text_threshold"]}
    vocab = pcfg["exp2"]["diagnostics"]["VOCABULARY"]["phrases"]
    img_path = Path(rcfg["image"])
    assert sha256_file(img_path) == rcfg["image_sha256"]
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    H, W = img.shape[:2]
    fp_before = code_fingerprint()["sha256"]
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_dryrun" if args.dry_run else ""))
    out.mkdir(parents=True, exist_ok=False)
    for d in ("inputs", "qwen", "dino", "role_crops", "figures"):
        (out / d).mkdir()
    (out / "config_snapshot.yaml").write_text((HERE / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    ctx = RI.build_context(rcfg, gcfg, v02, img)
    assert ctx["ft"]["accepted"]

    # ---------------------------------------------------------------- B1: existing records (read, preserved, not truth)
    ex = {"note": "read before any new inference; preserved; NOT ground truth", "sources": {k: str((HERE / v).resolve() if not v.startswith("/") else v) for k, v in cfg["existing_records"].items()}}
    s111 = Path(cfg["existing_records"]["exp111_summary"]).read_text(encoding="utf-8")
    ex["exp111_small_pallet"] = {"scene": "EXP110 development photos P01-P04 (NOT E1)", "excerpt": [l for l in s111.splitlines() if "pallet" in l.lower() or "팔레트" in l][:12],
                                 "pipeline_note": [l for l in Path(cfg["existing_records"]["exp111_design"]).read_text(encoding="utf-8").splitlines() if "팔레트 8/8" in l]}
    pr = (HERE / cfg["existing_records"]["placement_run"]).resolve()
    pc = json.loads((pr / "exp2_candidates.json").read_text(encoding="utf-8"))
    vd = pc["diagnostics"]["VOCABULARY"]["detections"]
    ex["placement_target_grounding_E1"] = {"qwen_open_question": "both conditions returned no candidate (cannot_determine)",
                                           "vocabulary_pallet_prompt": {c: sorted([(d["rank_in_condition"], round(d["score"], 3), [round(v) for v in d["bbox_xyxy"]]) for d in vd if d["condition"] == c and d["prompt"] == "pallet"])[:40]
                                                                        for c in ("VOCABULARY_FULL", "VOCABULARY_ROI_X2")},
                                           "n_boxes": {c: sum(d["condition"] == c for d in vd) for c in ("VOCABULARY_FULL", "VOCABULARY_ROI_X2")}}
    e1 = (HERE / cfg["existing_records"]["e1_v02_run"]).resolve()
    ti = json.loads((e1 / "analysis/target_inspection.json").read_text(encoding="utf-8"))
    ex["e1_v02_cube_candidate_members"] = {"candidate": ti["candidates"][0]["candidate_id"], "members": ti["candidates"][0]["member_boxes"],
                                           "note": "the larger 'small red block' DINO boxes (~48 x 42 px) cover the object under the red cube as well"}
    (out / "existing_records.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------- regions and images (offline)
    conv = next(o for o in ctx["st"]["objects"] if o["object_id"] == "conveyor_1")
    roi = make_roi(conv["bbox_xyxy"], (W, H), mcfg["target_inspection"])
    zc = pcfg["exp1"]["zone_crop"]
    dst_rect = P.zone_crop_rect([round(v, 1) for v in ctx["st"]["qwen_box_px"]], (W, H), zc)
    rects = {"SRC": roi.crop_xyxy, "DST": dst_rect}
    paths = {"full": out / "inputs/full_E1.png", "SRC": out / "inputs/src_roi_zoom.png", "DST": out / "inputs/dst_roi_x2.png"}
    shutil.copy2(img_path, paths["full"])
    cv2.imwrite(str(paths["SRC"]), zoom_roi(img, roi))
    cv2.imwrite(str(paths["DST"]), P.crop_view(img, dst_rect, zc["upscale"]))
    views = {"full": (paths["full"], [0, 0, W, H], 1.0), "SRC": (paths["SRC"], roi.crop_xyxy, roi.scale), "DST": (paths["DST"], dst_rect, zc["upscale"])}
    prompts = {"interpretation_template": cfg["interpretation"]["template"], "neutral": cfg["interpretation"]["neutral"], "role": cfg["roles"]["prompt"],
               "selection_template": cfg["selection"]["template"], "instructions": {k: v["text"] for k, v in cfg["instructions"].items()}}
    prompts["sha256"] = {k: sha256_text(v) for k, v in prompts.items() if isinstance(v, str)}
    (out / "prompts.json").write_text(json.dumps(prompts, indent=1, ensure_ascii=False), encoding="utf-8")
    regions = {"SRC": {"rect_px": roi.crop_xyxy, "scale": roi.scale, "rule": cfg["regions"]["SRC"]["rule"], "from": f"conveyor_1 {conv['bbox_xyxy']} (stored, Qwen crop yes)"},
               "DST": {"rect_px": dst_rect, "scale": zc["upscale"], "rule": cfg["regions"]["DST"]["rule"], "from": f"area_5 {ctx['st']['qwen_box_px']} (stored Qwen L2)"}}
    print("regions", regions)
    if args.dry_run:
        (out / "regions.json").write_text(json.dumps(regions, indent=1, default=str), encoding="utf-8")
        print("dry run written", out)
        return out

    from semantic_task_nav.perception.model_client import ModelWorkerClient
    client = ModelWorkerClient(mcfg, out / "worker_logs")
    calls, t0 = [], time.time()
    interp, role_rows, selections = {}, [], {}
    try:
        client.start()
        # ---------------------------------------------------------------- A: interpretation (6 Qwen calls)
        ic = cfg["interpretation"]
        for name, c in ic["calls"].items():
            p, rect, _ = views[c["image"]]
            if c["instruction"]:
                prompt, schema, pid = ic["template"].replace("{instruction}", cfg["instructions"][c["instruction"]]["text"]), INTERP_SCHEMA, ic["prompt_id"]
            else:
                prompt, schema, pid = ic["neutral"], NEUTRAL_SCHEMA, ic["neutral_prompt_id"]
            raw = P.qwen_call(client, prompt, p, ic["max_new_tokens"], f"A_{name}", pid, calls)
            (out / "qwen" / f"A_{name}_raw.txt").write_text(raw, encoding="utf-8")
            obj, err = P.parse_reply(raw, schema)
            interp[name] = {"instruction": c["instruction"], "image": c["image"], "image_rect_px": rect, "parsed": map_boxes(obj, rect), "error": err}
            print("A", name, "ERR " + err[:80] if err else (obj.get("selected_destination") if c["instruction"] else len(obj["support_candidates"])))
        (out / "task_interpretation.json").write_text(json.dumps(interp, indent=1, ensure_ascii=False), encoding="utf-8")

        # ---------------------------------------------------------------- B2: detection (6 DINO calls: 2 phrase sources x 3 images)
        dc = cfg["detection"]
        qphr, qorg = interp_phrases({k: v["parsed"] for k, v in interp.items()}, dc["phrase_max_words"])
        dets = []
        for src, phr in (("Q", qphr), ("V", vocab)):
            if not phr:
                continue
            for dname, view in dc["images"].items():
                p, rect, sc = views[view]
                raw = P.dino_call(client, p, phr, dc["thresholds"], f"B_{src}_{dname}", calls)
                (out / "dino" / f"B_{src}_{dname}_raw.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")
                for d in P.detections(raw, f"{src}_{dname}", rect, sc, dc["max_area_frac"]):
                    d.update({"phrase_source": dc["phrase_sources"][src], "image": dname, "roi_id": view, "roi_transform": {"crop_xyxy": rect, "scale": sc}})
                    dets.append(d)
        merged = P.merge(dets, dc["merge_iou"])
        byid_det = {d["id"]: d for d in dets}
        for c in merged:
            mem = [byid_det[i] for i in c["member_ids"]]
            c["phrase_sources"] = sorted({m["phrase_source"] for m in mem})
            c["images"] = sorted({m["image"] for m in mem})
            c["bucket"] = bucket(c["bbox_xyxy"], rects)
            c["best_rank_by_condition"] = {cond: min(m["rank_in_condition"] for m in mem if m["condition"] == cond) for cond in sorted({m["condition"] for m in mem})}
        print("B phrases Q", qphr, "| detections", len(dets), "merged", len(merged), {b: sum(c["bucket"] == b for c in merged) for b in ("DST", "SRC", "OTHER")})

        # ---------------------------------------------------------------- C: role analysis (task-independent, budgeted)
        queue, beyond = role_queue(merged, rects, cfg["roles"]["budget"])
        for c, b in queue:
            crop = make_verify_crop(img, c["bbox_xyxy"], mcfg["target_inspection"], out / "role_crops" / f"{c['id']}.png")
            raw = P.qwen_call(client, cfg["roles"]["prompt"], crop["path"], cfg["roles"]["max_new_tokens"], f"C_role_{c['id']}", cfg["roles"]["prompt_id"], calls)
            obj, err = P.parse_reply(raw, ROLE_SCHEMA)
            role_rows.append({"id": c["id"], "bucket": b, "bbox_xyxy": c["bbox_xyxy"], "score": c["score"], "phrases": c["phrases"], "phrase_sources": c["phrase_sources"],
                              "crop": crop, "raw": raw, "role": obj, "error": err})
        role_rows_beyond = [{"id": c["id"], "bucket": b, "bbox_xyxy": c["bbox_xyxy"], "score": c["score"], "status": "BEYOND_BUDGET"} for c, b in beyond]
        numbered = numbered_set(role_rows, cfg["selection"]["max_numbered"])
        npath = out / "inputs/selection_numbered_full.png"
        numbered_image(img, numbered, npath)
        print("C roles", len(role_rows), "numbered", len(numbered))

        # ---------------------------------------------------------------- C/D: task-conditioned selection (4 Qwen calls)
        for t, tv in cfg["instructions"].items():
            prompt = cfg["selection"]["template"].replace("{instruction}", tv["text"])
            raw = P.qwen_call(client, prompt, npath, cfg["selection"]["max_new_tokens"], f"D_select_{t}", cfg["selection"]["prompt_id"], calls) if numbered else None
            if raw is not None:
                (out / "qwen" / f"D_select_{t}_raw.txt").write_text(raw, encoding="utf-8")
            obj, err = P.parse_reply(raw, SEL_SCHEMA) if raw is not None else (None, "NOT_RUN (numbered set empty)")
            selections[t] = {"instruction": tv["text"], "parsed": obj, "error": err, "decision": decide(obj, numbered)}
            print("D", t, selections[t]["decision"])
    finally:
        client.close()
    (out / "model_calls.json").write_text(json.dumps({"calls": calls, "worker_env": client.env_info, "swaps": client.swaps, "total_wall_s": round(time.time() - t0, 1),
                                                       "n_calls": {"qwen_interpretation": sum(c["stage"].startswith("A_") for c in calls),
                                                                   "dino": sum(c["stage"].startswith("B_") for c in calls),
                                                                   "qwen_role": sum(c["stage"].startswith("C_role") for c in calls),
                                                                   "qwen_selection": sum(c["stage"].startswith("D_select") for c in calls)}}, indent=1, ensure_ascii=False), encoding="utf-8")
    num_ids = [a["id"] for a in numbered]
    cand_out = lambda b: [{**{k: c[k] for k in ("id", "rank", "bbox_xyxy", "score", "phrases", "phrase_sources", "images", "conditions", "member_ids", "best_rank_by_condition", "bucket")},   # noqa: E731
                           "retention_status": ("ROLE_ANALYSED" if c["id"] in {r["id"] for r in role_rows} else "BEYOND_BUDGET") + ("+NUMBERED" if c["id"] in num_ids else "")}
                          for c in merged if c["bucket"] == b]
    (out / "source_candidates.json").write_text(json.dumps({"region": regions["SRC"], "candidates": cand_out("SRC")}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "destination_candidates.json").write_text(json.dumps({"region": regions["DST"], "candidates": cand_out("DST")}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "all_candidates.json").write_text(json.dumps({"phrases_qwen": qphr, "phrase_origins": qorg, "phrases_vocabulary": vocab, "detections": dets, "merged": merged,
                                                         "other_candidates": cand_out("OTHER"), "rank_limit": "none"}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "object_role_analysis.json").write_text(json.dumps({"budget": cfg["roles"]["budget"], "analysed": role_rows, "beyond_budget": role_rows_beyond,
                                                               "numbered_set": [{"number": k, "id": a["id"]} for k, a in enumerate(numbered, 1)]}, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------- E: route B (offline)
    fr, s, pad, m = ctx["fr"], ctx["fr"].s, ctx["fr"].pad, ctx["ft"]["footprint_margin_cm"]
    L, Wd = map(float, cfg["route"]["robot_cm"])
    navs = {h: FN.RectNavMap(ctx["spec_bev"], fr.bev_size, ctx["pcfg"], FN.footprint_dims_cm(L, Wd, m, h), s) for h in (0, 90)}
    pA = PT.jload((HERE / cfg["semnav_run"]).resolve() / "integrated_task_plan.json")["plans"][0]
    pick_cm, old_B = pA["path_A"]["goal_cm"], pA["path_B_to_placement"]
    start_bev = fr.cm_to_bev([pick_cm])[0].tolist()
    old_rep = SN.leg_record(ctx, SN.leg_to_region(ctx, gcfg, navs, 0, start_bev, ctx["place_candidate_cm"], "placement_floor", L, Wd))

    def contact_cm(box):
        return fr.img_to_cm([[(box[0] + box[2]) / 2, box[3]]])[0]

    def route_to(box, tag):
        poly = fr.img_to_cm(PT.box_poly(box))
        bc = contact_cm(box)
        zones = [z["id"] for layer in ("traversable", "static_obstacles", "no_go") for z in ctx["spec_cm"][layer]
                 if cv2.pointPolygonTest(np.float32(z["polygon"]), (float(bc[0]), float(bc[1])), False) >= 0]
        rec = SN.leg_record(ctx, SN.leg_to_region(ctx, gcfg, navs, 0, start_bev, [poly.tolist()], tag, L, Wd))
        return {"bbox_xyxy": box, "floor_contact_cm": [round(float(v), 1) for v in bc], "floor_polygon_cm": np.round(poly, 1).tolist(),
                "zones_containing_contact": zones or ["NONE (outside every DRAFT zone)"], "route_B": rec,
                "note": "floor contact = bottom-centre of the box; the support's top surface is higher (unknown) and is not the floor point"}
    byid = {c["id"]: c for c in merged}
    routes = {"pickup_approach_cm": pick_cm, "old_B_stored": {k: old_B.get(k) for k in ("goal_cm", "length_cm", "gap_cm")},
              "old_B_recomputed": {k: old_rep.get(k) for k in ("goal_cm", "length_cm", "gap_cm")},
              "old_B_reproduced": old_rep.get("goal_cm") == old_B.get("goal_cm") and old_rep.get("length_cm") == old_B.get("length_cm"), "new": {}}
    for t in cfg["instructions"]:
        dsel = selections[t]["decision"]
        if t not in cfg["route"]["for_instructions"]:
            routes["new"][t] = {"status": "NOT_EVALUATED (other source / pick-up approach not planned)"}
        elif dsel["destination_status"] != "SELECTED":
            routes["new"][t] = {"status": f"NOT_RUN (destination {dsel['destination_status']})"}
        else:
            r = route_to(byid[dsel["destination"]]["bbox_xyxy"], f"dest_{t}")
            routes["new"][t] = {"status": "ROUTE_B_GENERATED" if r["route_B"]["path_B_feasible"] else "NO_ROUTE_B", "target": dsel["destination"], **r}
    (out / "route_B_comparison.json").write_text(json.dumps(routes, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    # ---------------------------------------------------------------- F: ablation (same numbered set)
    def floor_dist(c):
        p = contact_cm(c["bbox_xyxy"])
        return float(np.hypot(p[0] - pick_cm[0], p[1] - pick_cm[1]))
    abl = []
    A_pick = max(numbered, key=lambda a: a["score"])["id"] if numbered else None
    B_pick = min(numbered, key=lambda a: floor_dist(a))["id"] if numbered else None
    for t in cfg["instructions"]:
        dsel = selections[t]["decision"]
        C_pick = dsel["destination"] if dsel["destination_status"] == "SELECTED" else None
        rt = routes["new"][t]
        D_status = ("NOT_EVALUATED" if rt["status"].startswith("NOT_EVALUATED") else dsel["destination_status"] if C_pick is None
                    else "SELECTED" if rt["status"] == "ROUTE_B_GENERATED" else "DEFERRED_NAVIGATION")
        for cond, pick_id, status in (("A_detection_only", A_pick, "SELECTED" if A_pick else "NOT_FOUND"), ("B_detection_proximity", B_pick, "SELECTED" if B_pick else "NOT_FOUND"),
                                      ("C_semantic", dsel["destination"], dsel["destination_status"]), ("D_semantic_navigation", C_pick if D_status == "SELECTED" else None, D_status)):
            abl.append({"instruction": t, "condition": cond, "status": status, "destination": pick_id,
                        "bucket": byid[pick_id]["bucket"] if pick_id else "", "route_B_cm": (rt.get("route_B") or {}).get("length_cm") if cond.startswith("D") and pick_id else "",
                        "gap_cm": (rt.get("route_B") or {}).get("gap_cm") if cond.startswith("D") and pick_id else ""})
    tsel = {"numbered_set": [{"number": k, "id": a["id"], "bucket": a["bucket"], "object_type": a["role"]["object_type"], "support": a["role"]["is_support_surface"],
                              "load": a["role"]["load_state"]} for k, a in enumerate(numbered, 1)],
            "by_instruction": selections, "rules": "config.yaml selection", "never": "CONFIRMED (every SELECTED stays VERIFY_REQUIRED)"}
    (out / "target_selection.json").write_text(json.dumps(tsel, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(out / "ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(abl[0]))
        w.writeheader()
        w.writerows(abl)
    rows = []
    for t, iv in cfg["instructions"].items():
        a = next((v["parsed"] for k, v in interp.items() if v["instruction"] == t and v["image"] == "full"), None) or {}
        dsel = selections[t]["decision"]
        rows.append({"instruction": t, "text": iv["text"], "type": iv["type"],
                     "source_region": (a.get("source_region") or {}).get("description"), "source_object": (a.get("source_object") or {}).get("name_en"),
                     "destination_region": (a.get("destination_region") or {}).get("description"),
                     "searched_targets": "; ".join(c.get("name_en", "") for c in a.get("destination_target_candidates") or []), "qwen_selected_name": a.get("selected_destination"),
                     "source_status": dsel["source_status"], "source": dsel["source"], "destination_status": dsel["destination_status"], "destination": dsel["destination"],
                     "destination_role_analysis": next((f"{r['role']['object_type']} / {r['role']['load_state']}" for r in role_rows if r["id"] == dsel["destination"] and r["role"]), ""),
                     "reason": ((selections[t]["parsed"] or {}).get("reason") or "")[:200], "notes": "; ".join(dsel["notes"])})
    with open(out / "instruction_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    figures(img, rects, interp, merged, role_rows, numbered, npath, selections, routes, ctx, out)
    ev = evaluate(out, interp, merged, role_rows, numbered, selections, routes, abl, route_to, ctx, img)
    validation = {"image_sha256": rcfg["image_sha256"], "no_replay": "every model output in this run comes from a call logged in model_calls.json",
                  "n_model_calls": json.loads((out / "model_calls.json").read_text())["n_calls"],
                  "all_boxes_inside_image": all(0 <= d["bbox_xyxy"][0] <= W and 0 <= d["bbox_xyxy"][2] <= W and 0 <= d["bbox_xyxy"][1] <= H and 0 <= d["bbox_xyxy"][3] <= H for d in dets),
                  "every_detection_kept": len(dets) == sum(len(r["boxes_xyxy"]) for p in (out / "dino").glob("*.json") for r in json.loads(p.read_text())["results"]),
                  "beyond_budget_recorded": len(role_rows_beyond), "old_B_reproduced": routes["old_B_reproduced"],
                  "reference_read_after_decisions": (out / "target_selection.json").stat().st_mtime <= (out / "evaluation_reference_snapshot.yaml").stat().st_mtime,
                  "v02_code_before": fp_before, "v02_code_after": code_fingerprint()["sha256"]}
    validation["v02_code_unchanged"] = validation["v02_code_before"] == validation["v02_code_after"]
    (out / "validation.json").write_text(json.dumps(validation, indent=1, default=str), encoding="utf-8")
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "manifest.json").write_text(json.dumps({"files_sha256": files, "code_sha256": {p.name: sha256_file(p) for p in sorted(HERE.glob("*.py"))}}, indent=1), encoding="utf-8")
    print(json.dumps({"stages": ev["stages"], "hypotheses": ev["hypotheses"], "validation": validation}, indent=1, ensure_ascii=False, default=str))
    return out


# ================================================================================ evaluation (the only reader of the reference)
def evaluate(out, interp, merged, role_rows, numbered, selections, routes, abl, route_to, ctx, img):
    ref = yaml.safe_load((HERE / "evaluation_reference.yaml").read_text(encoding="utf-8"))
    shutil.copy2(HERE / "evaluation_reference.yaml", out / "evaluation_reference_snapshot.yaml")
    objs, thr = ref["objects"], ref["match_iou"]
    area5 = ctx["st"]["qwen_box_px"]

    def match(box):
        """reference object id matched by a box (IoU >= thr), else None"""
        best = max(objs, key=lambda k: iou(box, objs[k]["bbox_xyxy_px"]))
        return best if iou(box, objs[best]["bbox_xyxy_px"]) >= thr else None

    def qwen_locates(b, rb):
        if not b:
            return False
        rc = ((rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2)
        return iou(b, rb) >= 0.1 or (b[0] <= rc[0] <= b[2] and b[1] <= rc[1] <= b[3] and box_area(b) <= 10 * box_area(rb))
    # ---- stage 1: interpretation (A1)
    kw = ref["interpretation_keywords"]
    st1 = {}
    for name, v in interp.items():
        a = v["parsed"]
        if not a or v["instruction"] is None:
            st1[name] = {"status": "FAILED" if not a else "N/A (neutral)", "error": v["error"]}
            continue
        src_txt = f"{(a.get('source_region') or {}).get('description', '')} {(a.get('source_object') or {}).get('name_en', '')}".lower()
        dst_txt = f"{(a.get('destination_region') or {}).get('description', '')}".lower()
        db = (a.get("destination_region") or {}).get("bbox_image_px")
        dst_box_ok = bool(db) and (iou(db, area5) >= 0.1 or containment(db, area5) >= 0.5)
        st1[name] = {"source_names_conveyor": any(k in src_txt for k in kw["source"]), "destination_names_shipping": any(k in dst_txt for k in kw["destination"]),
                     "destination_box_on_area5": dst_box_ok, "source_text": src_txt, "destination_text": dst_txt}
    a1 = st1.get("A1", {})
    stage1 = "PASS" if a1.get("source_names_conveyor") and (a1.get("destination_names_shipping") or a1.get("destination_box_on_area5")) else \
             "PARTIAL" if a1.get("source_names_conveyor") or a1.get("destination_names_shipping") or a1.get("destination_box_on_area5") else "FAILED"
    # ---- discovery: Qwen mentions (located) and DINO candidates per reference object
    disc = {}
    for k, o in objs.items():
        rb = o["bbox_xyxy_px"]
        q = {}
        for name, v in interp.items():
            a = v["parsed"] or {}
            lst = [("destination_candidate", c) for c in a.get("destination_target_candidates") or []] + [("neutral_support", c) for c in a.get("support_candidates") or []] + \
                  [("neutral_object", c) for c in a.get("objects") or []] + ([("source_object", a["source_object"])] if isinstance(a.get("source_object"), dict) else [])
            q[name] = [{"kind": kind, "name": c.get("name_en"), "state": c.get("state"), "iou": round(iou(c["bbox_image_px"], rb), 3) if c.get("bbox_image_px") else None}
                       for kind, c in lst if isinstance(c, dict) and qwen_locates(c.get("bbox_image_px"), rb)]
        cands = [c for c in merged if iou(c["bbox_xyxy"], rb) >= thr]
        disc[k] = {"qwen_located_in": {n: v for n, v in q.items() if v},
                   "dino_candidates": [{"id": c["id"], "iou": round(iou(c["bbox_xyxy"], rb), 3), "phrase_sources": c["phrase_sources"], "phrases": c["phrases"],
                                        "best_rank_by_condition": c["best_rank_by_condition"]} for c in cands]}
    pd = disc["P_DST"]
    stage2 = ("PASS" if any("qwen_generated" in c["phrase_sources"] for c in pd["dino_candidates"]) else
              "PARTIAL (class-conditional vocabulary only)" if pd["dino_candidates"] else "FAILED")
    # ---- stage 3: roles
    roles = {}
    for k, o in objs.items():
        r = [x for x in role_rows if x["role"] and iou(x["bbox_xyxy"], o["bbox_xyxy_px"]) >= thr]
        exp_state = "occupied" if o["state"].startswith("occupied") else "empty"
        roles[k] = {"analysed": bool(r), "expected_load": exp_state, "answers": [{"id": x["id"], **x["role"]} for x in r],
                    "correct_load": None if not r else any(x["role"]["load_state"] == exp_state for x in r),
                    "support_yes": None if not r else any(x["role"]["is_support_surface"] in ("yes", "uncertain") for x in r)}
    key = [roles["P_SRC"], roles["P_DST"]]
    stage3 = ("UNVERIFIED (reference pallet not role-analysed)" if not all(x["analysed"] for x in key) else
              "PASS" if all(x["correct_load"] and x["support_yes"] for x in key) else "PARTIAL" if any(x["correct_load"] for x in key) else "FAILED")
    # ---- stage 4: task-conditioned selection
    byid = {c["id"]: c for c in merged}
    sel_eval = {}
    for t, s in selections.items():
        d = s["decision"]
        dm = match(byid[d["destination"]]["bbox_xyxy"]) if d["destination"] else None
        sm = match(byid[d["source"]]["bbox_xyxy"]) if d["source"] else None
        e = ref["expected"][t]
        if t == "T1":
            ok = d["destination_status"] == "SELECTED" and dm == "P_DST"
        elif t == "T4":
            ok = d["destination_status"] != "SELECTED"
        elif t == "T3":
            ok = dm != "P_DST" and (d["destination_status"] != "SELECTED" or byid[d["destination"]]["bucket"] == "SRC")
        else:
            ok = dm != "P_DST" and (d["destination_status"] != "SELECTED" or byid[d["destination"]]["bucket"] != "DST")
        sel_eval[t] = {"destination_status": d["destination_status"], "destination": d["destination"], "destination_matches": dm, "source": d["source"], "source_matches": sm,
                       "destination_bucket": byid[d["destination"]]["bucket"] if d["destination"] else None, "expected": e, "as_expected": ok,
                       "qwen_roles_for_reference": {k: next((c.get("role") for c in (s["parsed"] or {}).get("candidates", []) if isinstance(c, dict) and isinstance(c.get("id"), (int, float))
                                                              and 1 <= int(c["id"]) <= len(numbered) and match(numbered[int(c["id"]) - 1]["bbox_xyxy"]) == k), None)
                                                    for k in ("P_SRC", "P_DST")}}
    n_ok, t1_ok = sum(v["as_expected"] for v in sel_eval.values()), sel_eval["T1"]["as_expected"]
    stage4 = ("PASS" if n_ok == len(sel_eval) else "PARTIAL" if t1_ok else "PARTIAL (T1 target not selected)" if n_ok >= 2 else "FAILED")
    # ---- stage 5: navigation
    r1 = routes["new"]["T1"]
    stage5 = ("PASS" if r1["status"] == "ROUTE_B_GENERATED" and sel_eval["T1"]["destination_matches"] == "P_DST" else
              "PARTIAL (route to a non-reference target)" if r1["status"] == "ROUTE_B_GENERATED" else f"UNVERIFIED ({r1['status']})")
    oracle = route_to(objs["P_DST"]["bbox_xyxy_px"], "reference_P_DST_oracle")
    # ---- hypotheses
    a1_dst = bool(pd["qwen_located_in"].get("A1") or pd["qwen_located_in"].get("A2"))
    a3_dst = bool(pd["qwen_located_in"].get("A3"))
    ranks = {}
    for c in pd["dino_candidates"]:
        for cond, rk in c["best_rank_by_condition"].items():
            ranks[cond] = min(rk, ranks.get(cond, 10 ** 6))
    h1 = {"qwen_task_conditioned_located_P_DST": a1_dst, "qwen_neutral_located_P_DST": a3_dst, "dino_best_rank_of_P_DST_by_condition": ranks,
          "roi_improves_rank": {src: (ranks.get(f"{src}_D3", 10 ** 6) < ranks.get(f"{src}_D1", 10 ** 6)) if f"{src}_D3" in ranks else None for src in ("Q", "V")}}
    h1["status"] = "PASS" if a1_dst and not a3_dst and any(h1["roi_improves_rank"].values()) else "PARTIAL" if a1_dst or any(v for v in h1["roi_improves_rank"].values()) else "FAILED"
    hyp = {"H1_task_conditioned_discovery": h1, "H2_object_role": {"status": stage3, "roles": roles},
           "H3_task_dependent_selection": {"status": stage4, "per_instruction": {t: v["as_expected"] for t, v in sel_eval.items()},
                                           "destinations": {t: v["destination"] for t, v in sel_eval.items()}},
           "H4_navigation": {"status": stage5}}
    stages = {"1_task_interpretation": stage1, "2_context_aware_discovery": stage2, "3_object_role": stage3, "4_task_conditioned_selection": stage4, "5_navigation": stage5}
    abl_eval = [{**a, "matches": match(byid[a["destination"]]["bbox_xyxy"]) if a["destination"] else None} for a in abl]
    # evaluation figure
    im = img.copy()
    for k, o in objs.items():
        b = o["bbox_xyxy_px"]
        cv2.rectangle(im, (b[0] - 3, b[1] - 3), (b[2] + 3, b[3] + 3), (0, 0, 255), 2)
        P._t(im, f"REF {k}", (b[0], b[3] + 18), (0, 0, 255), 0.5)
    for t, v in sel_eval.items():
        if v["destination"]:
            b = [int(x) for x in byid[v["destination"]]["bbox_xyxy"]]
            cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 255, 255), 2)
            P._t(im, f"{t} dest", (b[0], b[1] - 6 - 16 * int(t[1])), (0, 255, 255), 0.5)
    lines = ["# EVALUATION (references read only here; AI-drawn, user to confirm)"] + [f"{k}: {v}" for k, v in stages.items()] + [""] + \
            [f"{t}: {v['destination_status']} {v['destination']} -> {v['destination_matches']} | expected ok: {v['as_expected']}" for t, v in sel_eval.items()] + \
            ["", f"H1: {h1['status']} (Qwen T1 located P_DST {a1_dst}, neutral {a3_dst}; DINO ranks {ranks})", "red = references; yellow = selected destinations"]
    cv2.imwrite(str(out / "figures/06_evaluation.png"), np.hstack([im, P.panel(im.shape[0], lines, 900)]))
    ev = {"reference": ref, "stage1_interpretation": st1, "discovery": disc, "roles": roles, "selection": sel_eval, "ablation": abl_eval,
          "reference_oracle_route_DIAGNOSTIC": oracle, "stages": stages, "hypotheses": hyp, "rates": "NOT COMPUTED (one photo)"}
    (out / "evaluation.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return ev


def figures(img, rects, interp, merged, role_rows, numbered, npath, selections, routes, ctx, out):
    fd = out / "figures"
    col = {"A1": (255, 0, 255), "A2": (255, 160, 40), "A3": (200, 200, 200), "T2": (0, 200, 255), "T3": (0, 255, 0), "T4": (0, 0, 255)}
    # 1 regions + interpretation boxes
    im = img.copy()
    for name, r in rects.items():
        cv2.rectangle(im, tuple(r[:2]), (r[2] - 1, r[3] - 1), (255, 255, 255), 3)
        P._t(im, f"{name} region (existing inference)", (r[0] + 6, r[3] - 10), (255, 255, 255), 0.6)
    lines = ["# Exp A - task interpretation (Qwen raw replies in qwen/)"]
    for name, v in interp.items():
        a = v["parsed"]
        if not a:
            lines.append(f"{name}: INVALID {v['error'][:80]}")
            continue
        if v["instruction"] is None:
            lines.append(f"#{name} neutral: {len(a.get('support_candidates', []))} supports: " + "; ".join(f"{c.get('name_en')} [{c.get('state')}]" for c in a["support_candidates"])[:300])
            for c in a["support_candidates"]:
                if c.get("bbox_image_px"):
                    b = [int(x) for x in c["bbox_image_px"]]
                    cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), col[name], 1)
            continue
        lines += [f"#{name} ({v['instruction']}, {v['image']}): source {(a.get('source_region') or {}).get('description')} / {(a.get('source_object') or {}).get('name_en')}",
                  f"   destination {(a.get('destination_region') or {}).get('description')}; selected {a.get('selected_destination')}",
                  "   candidates: " + "; ".join(f"{c.get('name_en')} [{c.get('state')}, {c.get('suitability')}]" for c in a["destination_target_candidates"])[:300]]
        for key in ("source_region", "destination_region"):
            b = (a.get(key) or {}).get("bbox_image_px")
            if b:
                b = [int(x) for x in b]
                cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), col[name], 2)
                P._t(im, f"{name} {key.split('_')[0]}", (b[0] + 4, b[1] + 18), col[name], 0.5)
        for c in a["destination_target_candidates"]:
            if c.get("bbox_image_px"):
                b = [int(x) for x in c["bbox_image_px"]]
                cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), col[name], 1)
    cv2.imwrite(str(fd / "01_source_destination_regions.png"), np.hstack([im, P.panel(im.shape[0], lines, 1000)]))
    # 2 candidates
    im = img.copy()
    for c in merged:
        b = [int(x) for x in c["bbox_xyxy"]]
        cc = (0, 255, 0) if len(c["phrase_sources"]) > 1 else (255, 0, 255) if c["phrase_sources"] == ["qwen_generated"] else (0, 200, 255)
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), cc, 1)
    for name, r in rects.items():
        cv2.rectangle(im, tuple(r[:2]), (r[2] - 1, r[3] - 1), (255, 255, 255), 2)
    n = {b: sum(c["bucket"] == b for c in merged) for b in ("DST", "SRC", "OTHER")}
    lines = ["# Exp B - Grounding DINO candidates (all kept, merged by IoU only)", f"{len(merged)} merged: {n}",
             "magenta = Qwen-generated phrases only; orange = vocabulary only; green = both", "white = SRC / DST regions"]
    cv2.imwrite(str(fd / "02_object_candidates.png"), np.hstack([im, P.panel(im.shape[0], lines, 700)]))
    # 3 role analysis tiles
    items = [{"path": r["crop"]["path"], "title": f"{r['id']} {r['bucket']} {(r['role'] or {}).get('object_type', 'INVALID')}",
              "line2": f"sup {(r['role'] or {}).get('is_support_surface')} / {(r['role'] or {}).get('load_state')}",
              "col": (0, 255, 0) if r["id"] in [a["id"] for a in numbered] else (150, 150, 150)} for r in role_rows]
    if items:
        tile_grid(items, fd / "03_object_role_comparison.png")
    # 4 selection
    im = cv2.imread(str(npath))
    lines = ["# Exp C/D - task-conditioned selection on the numbered set (same image for every instruction)"]
    lines += [f"{k}. {a['id']} {a['bucket']} {a['role']['object_type']} [{a['role']['is_support_surface']}/{a['role']['load_state']}]" for k, a in enumerate(numbered, 1)]
    for t, s in selections.items():
        p, d = s["parsed"] or {}, s["decision"]
        lines += ["", f"#{t}: source_id {p.get('source_id')} -> {d['source_status']} {d['source']}; destination_id {p.get('destination_id')} -> {d['destination_status']} {d['destination']}",
                  f"   {(p.get('reason') or s['error'] or '')[:160]}"] + ([f"   notes: {'; '.join(d['notes'])}"] if d["notes"] else [])
    cv2.imwrite(str(fd / "04_task_conditioned_selection.png"), np.hstack([im, P.panel(im.shape[0], lines, 1000)]))
    # 5 route
    mp = PT.Mapper("photo", ctx["fr"])
    im = img.copy()
    RI.layers(im, mp, ctx, text=False)
    PT._fill(im, mp, ctx["place_candidate_cm"], (0, 255, 0), 0.3)
    lines = ["# Exp E - route B", f"pick-up approach {routes['pickup_approach_cm']} cm", f"earlier B (bay free floor): {routes['old_B_recomputed']} (reproduced {routes['old_B_reproduced']})"]
    for t, r in routes["new"].items():
        if r.get("route_B") and r["route_B"]["path_B_feasible"]:
            rb = r["route_B"]
            PT._outline(im, mp, r["floor_polygon_cm"], (0, 255, 255), 2)
            if len(rb["waypoints_cm"]) > 1:
                RI.draw_path(im, mp, {k: rb[k] for k in ("waypoints_cm", "footprint_extent_xy_cm", "goal_cm")}, (0, 200, 255) if t == "T1" else (255, 0, 255), t, 15, coords=False)
            lines.append(f"{t}: {r['target']} goal {rb['goal_cm']} cm, {rb['length_cm']} cm, gap {rb['gap_cm']} cm; contact {r['floor_contact_cm']} in {r['zones_containing_contact']}")
        else:
            lines.append(f"{t}: {r['status']}")
    cv2.imwrite(str(fd / "05_route_B_comparison.png"), np.hstack([im, P.panel(im.shape[0], lines, 900)]))


if __name__ == "__main__":
    main()
