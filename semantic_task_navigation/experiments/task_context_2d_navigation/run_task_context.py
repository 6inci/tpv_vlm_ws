#!/usr/bin/env python3
"""Task context -> reference objects -> relation search -> roles -> source / destination (+ accessibility policy) -> 2D maps -> paths A, B.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
  $PY experiments/task_context_2d_navigation/run_task_context.py --dry-run   # images, alignment, instructions, maps; NO model call
  $PY experiments/task_context_2d_navigation/run_task_context.py             # Qwen + Grounding DINO (GPU), then planning

Reused (imported, unchanged): v0.2 worker / verify_box / make_roi / check_alignment; placement_target_grounding qwen_call, dino_call,
detections, merge, parse_reply; contextual_target_selection snap_anchor, zone_members, view_rect, zoom_scale, numbered;
semantic_navigation_integration full_plan, leg_to_region, leg_record, Raster; task_route_integration build_context, layers, draw_path.
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
sys.path.insert(0, str(HERE.parent / "contextual_target_selection"))
import placement_target as P  # noqa: E402
import select_targets as S  # noqa: E402
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file, sha256_text  # noqa: E402
from semantic_task_nav.geometry.coordinate_frames import box_area, containment, iou  # noqa: E402
from semantic_task_nav.navigation.map_builder import check_alignment  # noqa: E402
from semantic_task_nav.perception.grounding import verify_box  # noqa: E402
from semantic_task_nav.perception.target_inspection import make_roi  # noqa: E402

RI, SN, PT, FN = P.RI, P.SN, P.PT, P.FN
INTERP_SCHEMA = {"type": "object", "required": ["purpose", "source_object", "source_reference", "source_relation", "destination_reference", "destination_relation"],
                 "properties": {"source_relation": {"type": "string", "enum": ["next_to", "near", "on", "from", "none"]},
                                "destination_relation": {"type": "string", "enum": ["next_to", "near", "on", "none"]},
                                "source_object": {"type": "object"}, "source_reference": {"type": "object"}, "destination_reference": {"type": "object"},
                                "zones_seen": {"type": "array"}}}
NAV_CODES = {"NOT_ALLOWED_UNKNOWN": 0, "ALLOWED": 1, "FORBIDDEN": 2, "OBSTACLE": 3}
OCC = {"unknown": -1, "free": 0, "occupied": 1}


# ================================================================================ pure functions (tested offline)
def ref_class(name, table):
    n = (name or "").lower()
    return next((cls for cls, kws in table.items() if any(k in n for k in kws)), None)


def search_region(relation, ref_box, img_wh, tcfg, rcfg):
    if ref_box is None:
        return {"type": "NO_REFERENCE", "rect": None}
    if relation in rcfg["containment"]:
        return {"type": "containment", "rect": [int(v) for v in ref_box]}
    return {"type": "vicinity", "rect": make_roi(ref_box, img_wh, tcfg).crop_xyxy, "rule": "v0.2 make_roi (ratio 0.6, min 120 px)"}


def in_region(box, region, rcfg):
    if region["rect"] is None:
        return False
    if region["type"] == "containment":
        return containment(box, region["rect"]) >= rcfg["containment_min"]
    return P.centre_in(box, region["rect"]) or S.frac_inside(box, region["rect"]) >= 0.5


def decide_source(obj, n):
    if obj is None:
        return {"status": "AMBIGUOUS", "numbers": [], "why": "reply invalid"}
    movers = sorted({int(c["id"]) for c in obj["candidates"] if c.get("role") == "object_to_move" and S._num(c.get("id"), n)})
    if len(movers) == 1:
        return {"status": "SELECTED", "numbers": movers, "why": "exactly one candidate rated object_to_move"}
    if len(movers) > 1:
        return {"status": "AMBIGUOUS", "numbers": movers, "why": f"{len(movers)} candidates rated object_to_move (tie; no first-pick)"}
    k = S._num(obj.get("object_to_move_id"), n)
    return {"status": "SELECTED", "numbers": [k], "why": "object_to_move_id without per-candidate rating"} if k else {"status": "NOT_FOUND", "numbers": [], "why": "no candidate rated object_to_move"}


def decide_destination(obj, access, n):
    """access: number -> bool (adjacent to a drivable zone AND approach reachable). Policy in config.yaml."""
    if obj is None:
        return {"status": "AMBIGUOUS", "number": None, "kind": None, "why": "reply invalid"}
    rated = {int(c["id"]): c for c in obj["candidates"] if S._num(c.get("id"), n)}
    best = S._num(obj.get("best_destination_id"), n)
    t1 = [k for k, c in rated.items() if c["placement_role"] == "suitable" and access.get(k)]
    t2 = [k for k, c in rated.items() if c["placement_role"] == "possible" and access.get(k)]
    if best and rated.get(best, {}).get("placement_role") == "suitable" and access.get(best):
        return {"status": "SELECTED", "number": best, "kind": "DIRECT", "why": "Qwen best is suitable and accessible"}
    why0 = f"Qwen best {best} " + ("not accessible" if best and not access.get(best) else f"rated {rated.get(best, {}).get('placement_role')}" if best else "is null")
    if len(t1) == 1:
        return {"status": "SELECTED", "number": t1[0], "kind": "ALTERNATIVE", "why": why0 + "; the only accessible suitable candidate"}
    if len(t1) > 1:
        return {"status": "AMBIGUOUS", "number": None, "kind": None, "why": why0 + f"; {len(t1)} accessible suitable candidates {t1}"}
    if len(t2) == 1:
        return {"status": "SELECTED", "number": t2[0], "kind": "ALTERNATIVE_POSSIBLE", "why": why0 + "; no accessible suitable candidate, the only accessible 'possible' one"}
    if len(t2) > 1:
        return {"status": "AMBIGUOUS", "number": None, "kind": None, "why": why0 + f"; {len(t2)} accessible 'possible' candidates {t2}"}
    ok = [k for k, c in rated.items() if c["placement_role"] in ("suitable", "possible")]
    return {"status": "NOT_ACCESSIBLE" if ok else "NOT_FOUND", "number": None, "kind": None, "why": why0 + (f"; suitable/possible {ok} none accessible" if ok else "; nothing rated suitable/possible")}


def c1_source(cands, vis_level):
    """No relation, no Qwen: highest visual-evidence level, ties -> AMBIGUOUS. Detector boxes without a state: highest score."""
    if not cands:
        return {"status": "NOT_FOUND", "id": None}
    if "state" in cands[0]:
        key = lambda c: (vis_level.get(c["state"], 0), len(c.get("source_types") or []))   # noqa: E731
        top = max(key(c) for c in cands)
        tied = [c["id"] for c in cands if key(c) == top]
        return {"status": "SELECTED" if len(tied) == 1 else "AMBIGUOUS", "id": tied[0] if len(tied) == 1 else None, "tied": tied}
    c = max(cands, key=lambda c: c["score"])
    return {"status": "SELECTED", "id": c["id"], "tied": [c["id"]]}


def occupancy_and_constraints(spec_cm, raster):
    """Occupancy (free / occupied / unknown) and the navigation constraint grid from the DRAFT layers."""
    trav = raster.mask([z["polygon"] for z in spec_cm["traversable"]])
    obst = raster.mask([z["polygon"] for z in spec_cm["static_obstacles"]])
    nogo = {z["id"]: raster.mask([z["polygon"]]) for z in spec_cm["no_go"]}
    occ = np.full(trav.shape, OCC["unknown"], np.int8)
    free_phys = trav | nogo.get("Z01", np.zeros_like(trav))           # the pedestrian route is physically free (but forbidden)
    occ[free_phys] = OCC["free"]
    occ[obst] = OCC["occupied"]
    nav = np.full(trav.shape, NAV_CODES["NOT_ALLOWED_UNKNOWN"], np.int8)
    nav[trav] = NAV_CODES["ALLOWED"]
    for m in nogo.values():
        nav[m] = NAV_CODES["FORBIDDEN"]
    nav[obst] = NAV_CODES["OBSTACLE"]
    return occ, nav


# ================================================================================ main helpers
def interp_prompt(cfg, task, rules):
    rb = (cfg["interpretation"]["rules_R1"] + "\n") if rules else ""
    return cfg["interpretation"]["template"].replace("{instruction}", task["instruction"]).replace("{rules_block}", rb)


def role_prompt(cfg, key, task, rules, reference, source_object):
    rb = (cfg["interpretation"]["rules_R1"] + "\n") if rules else ""
    return cfg["roles"][key].replace("{instruction}", task["instruction"]).replace("{rules_block}", rb).replace("{reference}", reference or "").replace("{source_object}", source_object or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    rcfg = yaml.safe_load((HERE / cfg["reuse"]["route_config"]).read_text(encoding="utf-8"))
    fpcfg = yaml.safe_load((HERE / cfg["reuse"]["first_path_config"]).read_text(encoding="utf-8"))
    gcfg, v02 = PT.load_cfg(), load_config()
    mcfg, tcfg, rel = v02["models"], v02["models"]["target_inspection"], cfg["relations"]
    D = Path(cfg["image_dir"])
    fp_before = code_fingerprint()["sha256"]
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_dryrun" if args.dry_run else ""))
    out.mkdir(parents=True, exist_ok=False)
    for d in ("inputs", "figures", "qwen"):
        (out / d).mkdir()
    (out / "config_snapshot.yaml").write_text((HERE / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    e1_path = Path(rcfg["image"])
    e1 = cv2.imread(str(e1_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    H, W = e1.shape[:2]
    ctx = RI.build_context(rcfg, gcfg, v02, e1)
    fr, s, pad, m = ctx["fr"], ctx["fr"].s, ctx["fr"].pad, ctx["ft"]["footprint_margin_cm"]
    L, Wd = map(float, cfg["route"]["robot_cm"])
    funnel = yaml.safe_load((HERE.parent / "cube_candidate_funnel/config.yaml").read_text(encoding="utf-8"))
    sha_by_photo = {i["id"]: i["sha256"] for i in funnel["images"]}

    # ---------------------------------------------------------------- STEP A: images and instructions
    photos, insts = {}, []
    spec = {"reference_image": str(e1_path), "alignment_check": {"max_shift_px": 3.0, "min_response": 0.3}}
    for t in cfg["tasks"]:
        pid = t["photo"]
        if pid not in photos:
            p = D / t["file"]
            assert sha256_file(p) == sha_by_photo[pid], f"{pid} differs from the session record"
            img = cv2.imread(str(p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            al = check_alignment(spec, img, lambda x: x)
            photos[pid] = {"path": str(p), "img": img, "alignment": al}
        insts.append({"image_id": pid, "task_id": t["task_id"], "image_path": photos[pid]["path"], "instruction": t["instruction"], "type": t["type"],
                      "source_reference_description": t["source_reference"], "destination_reference_description": t["destination_reference"],
                      "intended_spatial_relation": t["relation"],
                      "scene_identity": "E1 (reference photo)" if pid == cfg["reference_photo"] else "same floor layout as E1, different arrangement of the cube / small pallets / robot",
                      "calibration_compatibility": {"camera_alignment_vs_E1": photos[pid]["alignment"]["status"], "shift_px": photos[pid]["alignment"].get("shift_px"),
                                                    "floor_transform": "E1 PROVISIONAL_UNVALIDATED transform reused" if photos[pid]["alignment"]["status"] == "ALIGNED" else "NOT applicable"},
                      "human_ground_truth": "E1 cone-row object type = pallet (user-confirmed); cube boxes AI_PRELIMINARY; destination answer not fixed"})
    (out / "image_instructions.json").write_text(json.dumps(insts, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    assert all(p["alignment"]["status"] == "ALIGNED" for p in photos.values()), "a photo is not aligned with E1: its floor coordinates would be invalid"

    # ---------------------------------------------------------------- STEP E: maps (same layout; independent of the models)
    raster = SN.CM.Raster((round((ctx["size_cm"][0] + 2 * pad) / 2), round((ctx["size_cm"][1] + 2 * pad) / 2)), 0.5, pad)
    occ, navc = occupancy_and_constraints(ctx["spec_cm"], raster)
    np.save(out / "occupancy_map.npy", occ)
    draft = {z["id"]: z for z in json.loads(Path(cfg["reuse"]["draft_zone_source"]).read_text(encoding="utf-8"))["zones"]}
    zones_sem = [{"id": z["id"], "layer": layer, "name_ko": draft.get(z["id"], {}).get("name_ko"), "role": draft.get(z["id"], {}).get("role"),
                  "polygon_cm": np.round(z["polygon"], 1).tolist(), "source": "MANUAL_DRAFT (transferred, DRAFT_UNCONFIRMED)"}
                 for layer in ("traversable", "static_obstacles", "no_go") for z in ctx["spec_cm"][layer]]
    zones_sem.append({"id": "loading_bay", "name_ko": "트럭 하역 구획(점선)", "polygon_cm": np.round(ctx["bay_cm"], 1).tolist(), "source": "GEOMETRY (stored Python dashed-line localization, E1)"})
    nav_json = {"grid": {"cell_cm": 2.0, "origin_cm": [-pad, -pad], "shape": list(navc.shape), "codes": NAV_CODES}, "grid_values": navc.tolist(),
                "rules": {"body": "only inside ALLOWED (Z02 minus obstacles); FORBIDDEN = pedestrian route Z01 + conveyor workcell Z07; outside every zone = not allowed",
                          "gripper": "may act on an object adjacent to ALLOWED while the body stays inside (policy); reach NOT verified"},
                "source": "MANUAL_DRAFT layers; NOT generated by the VLM"}
    (out / "navigation_constraints.json").write_text(json.dumps(nav_json, ensure_ascii=False), encoding="utf-8")
    counts = {"occupancy_cells": {k: int((occ == v).sum()) for k, v in OCC.items()}, "nav_cells": {k: int((navc == v).sum()) for k, v in NAV_CODES.items()}}
    print("photos", {k: (v["alignment"]["status"], v["alignment"].get("shift_px")) for k, v in photos.items()}, counts)
    if args.dry_run:
        (out / "semantic_zone_map.json").write_text(json.dumps({"zones": zones_sem}, indent=1, ensure_ascii=False), encoding="utf-8")
        print("dry run written", out)
        return out

    from semantic_task_nav.perception.model_client import ModelWorkerClient
    client, calls, t0 = ModelWorkerClient(mcfg, out / "worker_logs"), [], time.time()
    tasks = {t["task_id"]: t for t in cfg["tasks"]}
    interp, refs, cands, ctxs, sel = {}, {}, {}, {}, {}
    vocab = cfg["reuse"]["vocabulary"]
    try:
        client.start()
        # ---------------------------------------------------------------- STEP B1: interpretation R0 / R1 (Qwen)
        for tid, t in tasks.items():
            for R in ("R0", "R1"):
                pr = interp_prompt(cfg, t, R == "R1")
                raw = P.qwen_call(client, pr, photos[t["photo"]]["path"], cfg["interpretation"]["max_new_tokens"], f"interp_{tid}_{R}", cfg["interpretation"]["prompt_id"], calls)
                (out / "qwen" / f"interp_{tid}_{R}.txt").write_text(raw, encoding="utf-8")
                obj, err = P.parse_reply(raw, INTERP_SCHEMA)
                if obj:
                    for k in ("source_reference", "destination_reference"):
                        obj[k]["bbox_image_px"] = P.qwen_box_to_image(obj[k].get("bbox_2d"), [0, 0, W, H])
                    for z in obj.get("zones_seen") or []:
                        if isinstance(z, dict):
                            z["bbox_image_px"] = P.qwen_box_to_image(z.get("bbox_2d"), [0, 0, W, H])
                interp[(tid, R)] = {"parsed": obj, "error": err, "prompt_sha256": sha256_text(pr)}
                print("interp", tid, R, err or (obj["source_object"].get("name_en"), obj["source_reference"].get("name_en"), obj["source_relation"],
                                                obj["destination_reference"].get("name_en"), obj["destination_relation"]))
        # ---------------------------------------------------------------- STEP B2: reference objects and robot
        refs["U25070552"] = {"conveyor": {"box": next(o["bbox_xyxy"] for o in ctx["st"]["objects"] if o["object_id"] == "conveyor_1"), "how": "stored E1 (DINO + Qwen yes)"},
                             "truck": {"box": next(o["bbox_xyxy"] for o in ctx["st"]["objects"] if o["object_id"] == "truck_1"), "how": "stored E1 (DINO + Qwen yes)"},
                             "robot": {"box": ctx["robot"]["bbox_xyxy"], "start_img": ctx["starts"][0], "how": "stored E1 (first_vlm_navigation_path)"}}
        dino_raw = {}
        for pid, ph in photos.items():
            if pid == cfg["reference_photo"]:
                continue
            nouns = sorted({P.norm_phrase(n, 10) for (tid, R), v in interp.items() if tasks[tid]["photo"] == pid and v["parsed"]
                            for n in (v["parsed"]["source_object"].get("name_en"), v["parsed"]["source_reference"].get("name_en"), v["parsed"]["destination_reference"].get("name_en")) if n})
            prom = mcfg["grounding_dino"]["class_prompts"]["conveyor"] + mcfg["grounding_dino"]["class_prompts"]["truck"] + mcfg["grounding_dino"]["class_prompts"]["mobile_robot"]
            phrases = list(dict.fromkeys(vocab + nouns))
            dino_raw[pid] = {"refs": P.dino_call(client, ph["path"], prom, cfg["detection"]["thresholds"], f"dino_refs_{pid}", calls),
                             "full": P.dino_call(client, ph["path"], phrases, cfg["detection"]["thresholds"], f"dino_full_{pid}", calls), "phrases": phrases}
        for pid, ph in photos.items():
            if pid == cfg["reference_photo"]:
                continue
            dets = P.detections(dino_raw[pid]["refs"], "refs", [0, 0, W, H], 1.0, cfg["detection"]["max_area_frac"])
            refs[pid] = {}
            win = fpcfg["robot"]["search_window_xyxy"]
            for cls, prompts, extra in (("conveyor", mcfg["grounding_dino"]["class_prompts"]["conveyor"], None), ("truck", mcfg["grounding_dino"]["class_prompts"]["truck"], None),
                                        ("robot", mcfg["grounding_dino"]["class_prompts"]["mobile_robot"], win)):
                cs = [d for d in dets if d["status"] == "KEPT" and d["prompt"] in [p.lower() for p in prompts] and (extra is None or P.centre_in(d["bbox_xyxy"], extra))]
                recs, acc = [], None
                for k, c in enumerate(cs[:cfg["detection"]["reference_verify_max"]]):
                    desc = mcfg["verification"]["descriptions"]["mobile_robot" if cls == "robot" else cls]
                    v = verify_box(client, mcfg, ph["img"], c["bbox_xyxy"], desc, out / "inputs" / f"verify_{pid}_{cls}_{k}.png")
                    calls.append({"stage": f"verify_{pid}_{cls}_{k}", "model": v["call"]["model"], "raw_response": v["call"]["raw_response"], "latency_s": v["call"]["latency_s"],
                                  "prompt_sha256": v["call"]["prompt_sha256"], "image_path": v["crop"]["path"]})
                    recs.append({"bbox_xyxy": c["bbox_xyxy"], "score": c["score"], "status": v["status"]})
                    if v["status"] == "yes":
                        acc = c
                        break
                refs[pid][cls] = {"box": acc["bbox_xyxy"] if acc else None, "how": "fresh DINO + Qwen crop yes" if acc else "NOT VERIFIED", "checked": recs}
                if cls == "robot" and acc:
                    refs[pid]["robot"]["start_img"] = [(acc["bbox_xyxy"][0] + acc["bbox_xyxy"][2]) / 2, (acc["bbox_xyxy"][1] + acc["bbox_xyxy"][3]) / 2]
        # ROI detection for the new photos (vicinity of the verified references)
        for pid, ph in photos.items():
            if pid == cfg["reference_photo"]:
                dets_all = json.loads((HERE / cfg["reuse"]["e1_candidates"]).read_text(encoding="utf-8"))["merged"]
                cands[pid] = {"dino": dets_all, "how": "stored E1 merged DINO candidates (Qwen + vocabulary phrases, full / conveyor ROI / area_5 ROI)"}
                continue
            dets = P.detections(dino_raw[pid]["full"], "FULL", [0, 0, W, H], 1.0, cfg["detection"]["max_area_frac"])
            for cls in ("conveyor", "truck"):
                b = refs[pid][cls]["box"]
                if b is None:
                    continue
                roi = make_roi(b, (W, H), tcfg)
                zp = out / "inputs" / f"roi_{pid}_{cls}.png"
                x0, y0, x1, y1 = roi.crop_xyxy
                cv2.imwrite(str(zp), cv2.resize(ph["img"][y0:y1, x0:x1], roi.zoom_size, interpolation=cv2.INTER_CUBIC))
                raw = P.dino_call(client, zp, dino_raw[pid]["phrases"], cfg["detection"]["thresholds"], f"dino_roi_{pid}_{cls}", calls)
                dets += P.detections(raw, f"ROI_{cls}", roi.crop_xyxy, roi.scale, cfg["detection"]["max_area_frac"])
            cands[pid] = {"dino": P.merge(dets, cfg["detection"]["merge_iou"]), "how": "fresh DINO (vocabulary + interpretation nouns; full + reference ROIs), IoU merge", "n_detections": len(dets)}
        rr = json.loads(((HERE / cfg["reuse"]["retention_run"]).resolve() / "result.json").read_text(encoding="utf-8"))
        for p in rr["photos"]:
            if p["photo_id"] in photos:
                cands[p["photo_id"]]["cube"] = [{"id": f"cand_r{c['rank']}", "bbox_xyxy": c["bbox_xyxy"], "state": c["state"], "source_types": c["source_types"], "score": c["score"]}
                                                for c in p["candidates"]]
        # ---------------------------------------------------------------- STEP B3/D: relation search + roles (Qwen) per task and condition
        for tid, t in tasks.items():
            pid, ph = t["photo"], photos[t["photo"]]
            for R in ("R0", "R1"):
                it = interp[(tid, R)]["parsed"]
                rec = {"interpretation_ok": it is not None}
                if not it:
                    sel[(tid, R)] = {**rec, "source": {"status": "NOT_FOUND", "why": "interpretation invalid"}, "destination": {"status": "NOT_FOUND", "why": "interpretation invalid"}}
                    continue
                for side in ("source", "destination"):
                    rname = it[f"{side}_reference"].get("name_en")
                    rcls = ref_class(rname, rel["reference_classes"])
                    if rcls and refs[pid].get(rcls, {}).get("box"):
                        rbox, rhow = refs[pid][rcls]["box"], f"class '{rcls}' -> {refs[pid][rcls]['how']}"
                    else:
                        verified = [{"object_id": k, "bbox_xyxy": v["box"]} for k, v in refs[pid].items() if v.get("box") and k != "robot"]
                        rbox, rhow = S.snap_anchor(it[f"{side}_reference"].get("bbox_image_px"), verified, rel["snap_iou"])
                    relname = it[f"{side}_relation"]
                    region = search_region(relname, rbox, (W, H), tcfg, rel)
                    rec[f"{side}_reference"] = {"name": rname, "class": rcls, "box": rbox, "how": rhow, "relation": relname, "region": region}
                srcobj = it["source_object"].get("name_en") or ""
                # ---- source
                reg = rec["source_reference"]["region"]
                pool = cands[pid]["cube"] if "cube" in srcobj.lower() else [c for c in cands[pid]["dino"] if any(w in " ".join(c["phrases"]) for w in srcobj.lower().split()[-1:])]
                if reg["rect"] is None:
                    rec["source"] = {"status": "NO_REFERENCE", "why": f"source reference '{rec['source_reference']['name']}' not grounded"}
                else:
                    rbox = rec["source_reference"]["box"]
                    pool_all = cands[pid]["cube"] if "cube" in srcobj.lower() else [c for c in cands[pid]["dino"] if box_area(c["bbox_xyxy"]) < rel["region_scale_frac"] * box_area(reg["rect"]) and iou(c["bbox_xyxy"], rbox) < 0.5]
                    inside = [c for c in pool_all if in_region(c["bbox_xyxy"], reg, rel)]
                    inside = sorted(inside, key=lambda c: (c["bbox_xyxy"][1], c["bbox_xyxy"][0]))[:rel["max_numbered"]]
                    rec["source_candidates"] = {"pool": "red-cube candidates (retention)" if "cube" in srcobj.lower() else f"DINO candidates (phrase '{srcobj}')",
                                                "n_pool": len(pool_all), "inside": [c["id"] for c in inside], "outside": [c["id"] for c in pool_all if c not in inside]}
                    if inside:
                        vr = S.view_rect(reg["rect"], [c["bbox_xyxy"] for c in inside], (W, H))
                        sp = out / "inputs" / f"src_{tid}_{R}.png"
                        S.numbered(ph["img"], vr, S.zoom_scale(vr, tcfg), inside, sp)
                        pr = role_prompt(cfg, "source_template", t, R == "R1", rec["source_reference"]["name"], srcobj)
                        raw = P.qwen_call(client, pr, sp, cfg["roles"]["max_new_tokens"], f"source_{tid}_{R}", cfg["roles"]["source_prompt_id"], calls)
                        (out / "qwen" / f"source_{tid}_{R}.txt").write_text(raw, encoding="utf-8")
                        so, se = P.parse_reply(raw, S.SRC_SCHEMA)
                        d = decide_source(so, len(inside))
                        rec["source"] = {**d, "ids": [inside[k - 1]["id"] for k in d["numbers"]], "reply": so, "error": se, "image": str(sp), "view_rect": vr,
                                         "numbering": {k: c["id"] for k, c in enumerate(inside, 1)}}
                    else:
                        rec["source"] = {"status": "NOT_FOUND", "why": "no candidate inside the search region", "ids": []}
                rec["source_C1"] = c1_source(pool, {"VLM_ACCEPTED": 3, "PROBABLE": 2, "PENDING_VERIFICATION": 1, "REJECTED": 0})
                # ---- destination
                reg = rec["destination_reference"]["region"]
                if reg["rect"] is None:
                    rec["destination"] = {"status": "NO_REFERENCE", "why": f"destination reference '{rec['destination_reference']['name']}' not grounded"}
                else:
                    inc, regions = S.zone_members(cands[pid]["dino"], reg["rect"], rel["region_scale_frac"])
                    inc = [c for c in inc if not any(iou(c["bbox_xyxy"], cb["bbox_xyxy"]) >= 0.5 for cb in cands[pid]["cube"] + cands[pid]["dino"] if cb["id"] in rec["source"].get("ids", []))]   # the object to move is not its own destination
                    inc = sorted(inc, key=lambda c: (c["bbox_xyxy"][1], c["bbox_xyxy"][0]))[:rel["max_numbered"]]
                    vr = S.view_rect(reg["rect"], [c["bbox_xyxy"] for c in inc], (W, H))
                    dp = out / "inputs" / f"dst_{tid}_{R}.png"
                    S.numbered(ph["img"], vr, S.zoom_scale(vr, tcfg), inc, dp)
                    pr = role_prompt(cfg, "destination_template", t, R == "R1", rec["destination_reference"]["name"], srcobj)
                    raw = P.qwen_call(client, pr, dp, cfg["roles"]["max_new_tokens"], f"destination_{tid}_{R}", cfg["roles"]["destination_prompt_id"], calls)
                    (out / "qwen" / f"destination_{tid}_{R}.txt").write_text(raw, encoding="utf-8")
                    do, de = P.parse_reply(raw, S.DEST_SCHEMA)
                    rec["destination_candidates"] = {"inside": [c["id"] for c in inc], "region_scale_not_numbered": [c["id"] for c in regions],
                                                     "outside_count": len(cands[pid]["dino"]) - len(inc) - len(regions)}
                    rec["destination"] = {"reply": do, "error": de, "image": str(dp), "view_rect": vr, "numbering": {k: c["id"] for k, c in enumerate(inc, 1)}, "_inc": inc}
                c1d = max([c for c in cands[pid]["dino"]], key=lambda c: c["score"])
                rec["destination_C1"] = {"status": "SELECTED", "id": c1d["id"], "why": "highest detector score in the whole image (naive)"}
                sel[(tid, R)] = rec
                print("roles", tid, R, rec["source"].get("status"), rec["source"].get("ids"), "| dest reply", bool(rec["destination"].get("reply")))
    finally:
        client.close()
    (out / "model_calls.json").write_text(json.dumps({"calls": calls, "n": {"qwen": sum(not c["stage"].startswith("dino") for c in calls), "dino": sum(c["stage"].startswith("dino") for c in calls)},
                                                       "worker_env": client.env_info, "swaps": client.swaps, "total_wall_s": round(time.time() - t0, 1),
                                                       "peak_vram_gib": max((c.get("peak_vram_gib") or 0) for c in calls)}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    # ---------------------------------------------------------------- accessibility (Python) and destination decisions
    zr = SN.CM.Raster(fr.bev_size, s, pad)
    drivable = [z["polygon"] for z in ctx["spec_cm"]["traversable"]]
    navs = {h: FN.RectNavMap(ctx["spec_bev"], fr.bev_size, ctx["pcfg"], FN.footprint_dims_cm(L, Wd, m, h), s) for h in (0, 90)}
    acc_cache = {}

    def access(pid, c):
        key = (pid, c["id"])
        if key in acc_cache:
            return acc_cache[key]
        poly = fr.img_to_cm(PT.box_poly(c["bbox_xyxy"])).tolist()
        gap = min((g for g in (zr.gap(poly, d) for d in drivable) if g is not None), default=None)
        start = refs[pid]["robot"].get("start_img")
        leg = SN.leg_record(ctx, SN.leg_to_region(ctx, gcfg, navs, 0, fr.cm_to_bev(fr.img_to_cm([start]))[0].tolist(), [poly], f"acc_{c['id']}", L, Wd)) if start else {"path_B_feasible": False}
        a = {"gap_to_drivable_cm": None if gap is None else round(gap, 1), "adjacent": gap is not None and gap <= cfg["policy"]["adjacent_cm"],
             "approach_reachable_from_start": leg.get("path_B_feasible"), "approach_goal_cm": leg.get("goal_cm"), "body_gap_cm": leg.get("gap_cm")}
        a["accessible"] = bool(a["adjacent"] and a["approach_reachable_from_start"])
        acc_cache[key] = a
        return a
    for (tid, R), rec in sel.items():
        d = rec.get("destination", {})
        if "_inc" not in d:
            continue
        inc = d.pop("_inc")
        rated = {int(c["id"]): c for c in (d["reply"] or {}).get("candidates", []) if S._num(c.get("id"), len(inc))}
        acc = {k: access(tasks[tid]["photo"], inc[k - 1]) for k, c in rated.items() if c["placement_role"] in ("suitable", "possible")}
        dd = decide_destination(d["reply"], {k: a["accessible"] for k, a in acc.items()}, len(inc))
        d.update({**dd, "id": inc[dd["number"] - 1]["id"] if dd["number"] else None, "bbox_xyxy": inc[dd["number"] - 1]["bbox_xyxy"] if dd["number"] else None,
                  "accessibility": {str(k): {"id": inc[k - 1]["id"], "role": rated[k]["placement_role"], "appearance": rated[k].get("appearance"), **a} for k, a in acc.items()}})

    # ---------------------------------------------------------------- STEP F: paths
    routes = {}
    byid = lambda pid, i: next((c for c in cands[pid]["cube"] + cands[pid]["dino"] if c["id"] == i), None)   # noqa: E731
    for (tid, R), rec in sel.items():
        pid = tasks[tid]["photo"]
        start = refs[pid]["robot"].get("start_img")
        src, dst = rec.get("source", {}), rec.get("destination", {})
        r = {"start_img": start, "source_status": src.get("status"), "destination_status": dst.get("status"), "plans": []}
        sids = src.get("ids", []) if src.get("status") in ("SELECTED", "AMBIGUOUS") else []
        for sid in sids[:3]:
            c = byid(pid, sid)
            plan = SN.full_plan(ctx, gcfg, {"id": sid, "bbox_xyxy": c["bbox_xyxy"], "state": c.get("state", "DETECTED")}, L, Wd, m, start)
            A = plan["summary"]["paths"].get("A")
            pr = {"source_id": sid, "label": "AUTONOMOUS" if src["status"] == "SELECTED" else "DIAGNOSTIC_ONLY (source tie)",
                  "path_A": None if not A else {k: A[k] for k in ("goal_cm", "length_cm", "waypoints_cm", "heading_deg", "footprint_extent_xy_cm")} | {
                      "gap_cm": A["approach"]["body_to_target_gap_cm"], "sweep_ok": plan["summary"]["checks"]["A"]["whole_body_sweep"]["ok"],
                      "layers_clear": plan["summary"]["checks"]["A"]["swept_layer_overlap"]["clear"]}, "path_A_failure": plan["summary"]["failure_A"]}
            if A and dst.get("status") == "SELECTED":
                hA = plan["var"]["selected"]["A_heading_deg"]
                poly = fr.img_to_cm(PT.box_poly(dst["bbox_xyxy"])).tolist()
                leg = SN.leg_record(ctx, SN.leg_to_region(ctx, gcfg, plan["navs"], hA, plan["var"]["legs_A"][hA]["goal"]["xy_bev_px"], [poly], "dest", L, Wd))
                pr["path_B"] = {k: leg.get(k) for k in ("path_B_feasible", "goal_cm", "length_cm", "gap_cm", "waypoints_cm", "footprint_extent_xy_cm", "whole_body_sweep_ok", "v02_valid")} | \
                    {"layers_clear": (leg.get("layers") or {}).get("clear"), "connected": bool(leg.get("waypoints_cm") and leg["waypoints_cm"][0] == A["goal_cm"])}
            else:
                pr["path_B"] = None
                pr["path_B_why"] = "no path A" if not A else f"destination {dst.get('status')}"
            r["plans"].append(pr)
        ok = src.get("status") == "SELECTED" and dst.get("status") == "SELECTED" and r["plans"] and r["plans"][0]["path_A"] and r["plans"][0]["path_B"] and r["plans"][0]["path_B"]["path_B_feasible"]
        r["category"] = "AUTONOMOUS (plan only)" if ok else "DIAGNOSTIC_ONLY" if r["plans"] and any(p["path_A"] for p in r["plans"]) else "FAILED / UNVERIFIED"
        routes[f"{tid}_{R}"] = r
    (out / "routes.json").write_text(json.dumps(routes, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    ctx_out = {f"{tid}_{R}": {k: v for k, v in rec.items() if k not in ("source", "destination")} for (tid, R), rec in sel.items()}
    (out / "context_search_results.json").write_text(json.dumps({"references": refs, "interpretations": {f"{k[0]}_{k[1]}": v for k, v in interp.items()}, "search": ctx_out,
                                                                  "candidates_source": {pid: v["how"] for pid, v in cands.items()}}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "target_selection.json").write_text(json.dumps({f"{tid}_{R}": {"source": rec.get("source"), "destination": rec.get("destination"), "source_C1": rec.get("source_C1"),
                                                                          "destination_C1": rec.get("destination_C1")} for (tid, R), rec in sel.items()}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    zones_vlm = [{"task": f"{tid}_{R}", "description": z.get("description"), "evidence": z.get("visual_evidence"), "robot_body_access": z.get("robot_body_access"),
                  "bbox_image_px": z.get("bbox_image_px"), "source": "VLM_RECOGNIZED" + (" (with R1 rules)" if R == "R1" else " (no rules)")}
                 for (tid, R), v in interp.items() if v["parsed"] for z in (v["parsed"].get("zones_seen") or []) if isinstance(z, dict)]
    (out / "semantic_zone_map.json").write_text(json.dumps({"zones": zones_sem, "vlm_zones": zones_vlm}, indent=1, ensure_ascii=False), encoding="utf-8")
    figures(out, cfg, tasks, photos, sel, routes, refs, cands, ctx, occ, navc, raster)

    # ---------------------------------------------------------------- evaluation (reference read only here)
    ref = yaml.safe_load((HERE / "evaluation_reference.yaml").read_text(encoding="utf-8"))
    labels = funnel["labels"]
    rows = []
    for (tid, R), rec in sel.items():
        pid, rt = tasks[tid]["photo"], routes[f"{tid}_{R}"]
        src, dst = rec.get("source", {}), rec.get("destination", {})
        sboxes = [byid(pid, i)["bbox_xyxy"] for i in src.get("ids", [])]
        cube_iou = [round(iou(b, labels[pid]), 3) for b in sboxes] if "cube" in ((interp[(tid, R)]["parsed"] or {}).get("source_object", {}).get("name_en") or "") else None
        dmatch = None
        if dst.get("bbox_xyxy") and pid == cfg["reference_photo"]:
            dmatch = next((k for k, o in ref["E1_objects"].items() if iou(dst["bbox_xyxy"], o["bbox_xyxy_px"]) >= ref["match_iou"]), "other")
        first_fail = ("interpretation" if not rec.get("interpretation_ok") else "source reference" if src.get("status") == "NO_REFERENCE" else
                      "destination reference" if dst.get("status") == "NO_REFERENCE" else "source selection" if src.get("status") != "SELECTED" else
                      "destination selection" if dst.get("status") != "SELECTED" else "path A" if not (rt["plans"] and rt["plans"][0]["path_A"]) else
                      "path B" if not (rt["plans"][0]["path_B"] and rt["plans"][0]["path_B"]["path_B_feasible"]) else "none (plan level)")
        p0 = rt["plans"][0] if rt["plans"] else {}
        rows.append({"task": tid, "condition": "C2_relation_R0" if R == "R0" else "C3_relation_R1", "image": pid, "instruction": tasks[tid]["instruction"],
                     "source_reference": f"{rec.get('source_reference', {}).get('name')} [{rec.get('source_reference', {}).get('relation')}]",
                     "source_status": src.get("status"), "source": " ".join(src.get("ids", [])), "source_iou_with_cube_label": cube_iou,
                     "destination_reference": f"{rec.get('destination_reference', {}).get('name')} [{rec.get('destination_reference', {}).get('relation')}]",
                     "destination_status": dst.get("status"), "destination": dst.get("id"), "destination_kind": dst.get("kind"), "destination_matches_E1_object": dmatch,
                     "path_A_cm": (p0.get("path_A") or {}).get("length_cm"), "path_B_cm": (p0.get("path_B") or {}).get("length_cm"), "path_B_gap_cm": (p0.get("path_B") or {}).get("gap_cm"),
                     "collision_free": None if not p0.get("path_A") else bool(p0["path_A"]["sweep_ok"] and (not p0.get("path_B") or p0["path_B"]["whole_body_sweep_ok"])),
                     "forbidden_zone_clear": None if not p0.get("path_A") else bool(p0["path_A"]["layers_clear"] and (not p0.get("path_B") or p0["path_B"]["layers_clear"])),
                     "category": rt["category"], "first_failure": first_fail, "C1_source": f"{(rec.get('source_C1') or {}).get('status')} {(rec.get('source_C1') or {}).get('id') or (rec.get('source_C1') or {}).get('tied')}",
                     "C1_destination": rec.get("destination_C1", {}).get("id")})
    with open(out / "evaluation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    shutil.copy2(HERE / "evaluation_reference.yaml", out / "evaluation_reference_snapshot.yaml")
    validation = {"model_calls": json.loads((out / "model_calls.json").read_text())["n"], "photos_aligned": {k: v["alignment"]["status"] for k, v in photos.items()},
                  "no_first_pick_on_ties": all(r["source"].get("status") != "SELECTED" or len(r["source"].get("numbers", [])) == 1 for r in sel.values() if "numbers" in r.get("source", {})),
                  "no_unselected_target_in_autonomous_route": all(r["category"] != "AUTONOMOUS (plan only)" or (r["source_status"] == "SELECTED" and r["destination_status"] == "SELECTED") for r in routes.values()),
                  "evaluation_after_decisions": (out / "target_selection.json").stat().st_mtime <= (out / "evaluation.csv").stat().st_mtime,
                  "v02_code_unchanged": fp_before == code_fingerprint()["sha256"]}
    (out / "validation.json").write_text(json.dumps(validation, indent=1), encoding="utf-8")
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "manifest.json").write_text(json.dumps({"files_sha256": files, "code_sha256": {p.name: sha256_file(p) for p in sorted(HERE.glob("*.py"))}}, indent=1), encoding="utf-8")
    for r in rows:
        print({k: r[k] for k in ("task", "condition", "source_status", "source", "destination_status", "destination", "destination_kind", "destination_matches_E1_object", "path_A_cm", "path_B_cm", "category", "first_failure")})
    print(validation)
    return out


# ================================================================================ figures
def figures(out, cfg, tasks, photos, sel, routes, refs, cands, ctx, occ, navc, raster):
    fd, fr = out / "figures", ctx["fr"]
    mp = PT.Mapper("photo", fr)
    S3 = 3
    mb = PT.Mapper("bev", fr, S3)
    k = int(round(2 * S3))

    def grid_img(grid, colours):
        im = np.zeros(grid.shape + (3,), np.uint8)
        for v, col in colours.items():
            im[grid == v] = col
        return cv2.resize(im, (grid.shape[1] * k, grid.shape[0] * k), interpolation=cv2.INTER_NEAREST)

    occ_im = grid_img(occ, {OCC["free"]: (255, 255, 255), OCC["occupied"]: (0, 0, 0), OCC["unknown"]: (128, 128, 128)})
    nav_im = grid_img(navc, {NAV_CODES["ALLOWED"]: (120, 220, 120), NAV_CODES["FORBIDDEN"]: (60, 60, 230), NAV_CODES["OBSTACLE"]: (40, 40, 40), NAV_CODES["NOT_ALLOWED_UNKNOWN"]: (150, 150, 150)})
    for z in ctx["spec_cm"]["traversable"] + ctx["spec_cm"]["static_obstacles"] + ctx["spec_cm"]["no_go"]:
        PT._t(nav_im, z["id"], mb(np.mean(z["polygon"], axis=0))[0], (255, 255, 255), 0.45)
    PT._outline(nav_im, mb, ctx["bay_cm"], (230, 200, 40), 2)
    lines_occ = ["# 2D OCCUPANCY (from the DRAFT layers; same layout in every photo)", "white = free (incl. the pedestrian route: physically free)", "black = occupied (static obstacles)",
                 "grey = unknown (outside every zone; conveyor workcell equipment not measured)", "NOT generated by the VLM"]
    cv2.imwrite(str(fd / "F2_occupancy_map.png"), np.hstack([occ_im, P.panel(occ_im.shape[0], lines_occ, 620)]))
    for tid, t in tasks.items():
        pid, img = t["photo"], photos[t["photo"]]["img"]
        for R in ("R0", "R1"):
            rec, rt = sel.get((tid, R), {}), routes.get(f"{tid}_{R}", {})
            tag = f"{tid}_{R}"
            # F1 context search
            im = img.copy()
            for side, col in (("source", (0, 0, 255)), ("destination", (0, 200, 255))):
                rr = rec.get(f"{side}_reference") or {}
                if rr.get("box"):
                    b = [int(v) for v in rr["box"]]
                    cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), col, 3)
                    P._t(im, f"{side} reference: {rr.get('name')}", (b[0], b[3] + 20), col, 0.6)
                reg = (rr.get("region") or {}).get("rect")
                if reg:
                    PT._outline(im, lambda p: np.round(np.asarray(p).reshape(-1, 2)).astype(np.int32), [[reg[0], reg[1]], [reg[2], reg[1]], [reg[2], reg[3]], [reg[0], reg[3]]], col, 2, dashed=True)
                    P._t(im, f"{side} search region ({rr['region']['type']}, relation {rr.get('relation')})", (reg[0] + 4, reg[1] + 20), col, 0.55)
            sc = rec.get("source_candidates") or {}
            for c in cands[pid]["cube"] if "cube" in str(sc.get("pool", "")) else cands[pid]["dino"]:
                if c["id"] in sc.get("outside", []):
                    b = [int(v) for v in c["bbox_xyxy"]]
                    cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (150, 150, 150), 1)
                if c["id"] in sc.get("inside", []):
                    b = [int(v) for v in c["bbox_xyxy"]]
                    cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (255, 0, 255), 2)
            for i in (rec.get("destination") or {}).get("numbering", {}).values():
                c = next(x for x in cands[pid]["dino"] if x["id"] == i)
                b = [int(v) for v in c["bbox_xyxy"]]
                cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (255, 255, 0), 1)
            for i in (rec.get("source") or {}).get("ids", []):
                c = next(x for x in cands[pid]["cube"] + cands[pid]["dino"] if x["id"] == i)
                b = [int(v) for v in c["bbox_xyxy"]]
                cv2.rectangle(im, (b[0] - 4, b[1] - 4), (b[2] + 4, b[3] + 4), (0, 0, 255), 3)
            db = (rec.get("destination") or {}).get("bbox_xyxy")
            if db:
                cv2.rectangle(im, (int(db[0]) - 4, int(db[1]) - 4), (int(db[2]) + 4, int(db[3]) + 4), (0, 255, 0), 3)
            src, dst = rec.get("source") or {}, rec.get("destination") or {}
            lines = [f"# F1 CONTEXT SEARCH {tag}", "instruction (Korean, see image_instructions.json)", f"source: {src.get('status')} {src.get('ids')} - {src.get('why')}",
                     f"destination: {dst.get('status')} {dst.get('id')} {dst.get('kind')} - {dst.get('why')}", "",
                     "red / orange solid = reference objects; dashed = search regions", "magenta = source candidates inside the region; grey = outside",
                     "cyan thin = numbered destination candidates; red thick = selected / tied source; green thick = selected destination"]
            for k2, a in (dst.get("accessibility") or {}).items():
                lines.append(f"#{k2} {a['id']} {a['role']} {a.get('appearance')}: adjacent {a['adjacent']} (gap {a['gap_to_drivable_cm']} cm), approach {a['approach_reachable_from_start']}, body gap {a['body_gap_cm']}")
            cv2.imwrite(str(fd / f"F1_context_{tag}.png"), np.hstack([im, P.panel(im.shape[0], lines, 900)]))
            # F3 semantic / navigation map with approach poses; F4 paths on the 2D map
            nm = nav_im.copy()
            for k2, a in (dst.get("accessibility") or {}).items():
                if a.get("approach_goal_cm"):
                    cv2.drawMarker(nm, tuple(mb(a["approach_goal_cm"])[0]), (255, 255, 0) if a["accessible"] else (0, 0, 255), cv2.MARKER_DIAMOND, 12, 2)
            st = refs[pid]["robot"].get("start_img")
            if st:
                cv2.drawMarker(nm, tuple(mb(fr.img_to_cm([st]))[0]), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
            pm = occ_im.copy()
            PT._outline(pm, mb, ctx["bay_cm"], (230, 200, 40), 2)
            ov = img.copy()
            RI.layers(ov, mp, ctx, text=False)
            for p in rt.get("plans", []):
                for key, col, lab in (("path_A", (255, 160, 40), "A"), ("path_B", (40, 150, 255), "B")):
                    q = p.get(key)
                    if q and q.get("waypoints_cm") and len(q["waypoints_cm"]) > 1:
                        path = {"waypoints_cm": q["waypoints_cm"], "footprint_extent_xy_cm": q["footprint_extent_xy_cm"], "goal_cm": q["goal_cm"]}
                        RI.draw_path(pm, mb, path, col, lab, 10, coords=False)
                        RI.draw_path(ov, mp, path, col, lab, 15, coords=False)
                        if q.get("goal_cm"):
                            cv2.drawMarker(nm, tuple(mb(q["goal_cm"])[0]), col, cv2.MARKER_SQUARE, 14, 2)
            lab_lines = [f"# {tag}: {rt.get('category')}"] + [f"{p['label']} source {p['source_id']}: A {(p['path_A'] or {}).get('length_cm')} cm (gap {(p['path_A'] or {}).get('gap_cm')}), "
                                                            f"B {(p['path_B'] or {}).get('length_cm')} cm (gap {(p['path_B'] or {}).get('gap_cm')}) {p.get('path_B_why', '')}" for p in rt.get("plans", [])]
            cv2.imwrite(str(fd / f"F3_semantic_navigation_{tag}.png"), np.hstack([nm, P.panel(nm.shape[0], lab_lines + ["", "green = ALLOWED (body), red = FORBIDDEN (pedestrian route, conveyor workcell)",
                                                                                                                         "dark = obstacle, grey = not allowed / unknown, yellow outline = loading bay",
                                                                                                                         "diamonds = approach poses of rated destination candidates (yellow accessible, red not)",
                                                                                                                         "squares = path goals; cross = robot start"], 760)]))
            cv2.imwrite(str(fd / f"F4_paths_2d_{tag}.png"), np.hstack([pm, P.panel(pm.shape[0], lab_lines + ["", "on the occupancy map (white free / black occupied / grey unknown)",
                                                                                                             "rectangles = whole robot body every 10 cm (35x24 + 3.5 cm margin)"], 760)]))
            cv2.imwrite(str(fd / f"F5_overlay_{tag}.png"), np.hstack([ov, P.panel(ov.shape[0], lab_lines + ["", "the same path coordinates (floor cm) projected into the photo with the",
                                                                                                               "PROVISIONAL E1 transform (camera aligned)"], 900)]))


if __name__ == "__main__":
    main()
