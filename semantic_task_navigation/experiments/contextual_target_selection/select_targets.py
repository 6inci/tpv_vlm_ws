#!/usr/bin/env python3
"""Contextual source / destination target selection on E1 -> paths A and B (see config.yaml for the fixed failures F1-F3).

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
  $PY experiments/contextual_target_selection/select_targets.py --dry-run   # zones, numbered images, prompts; NO model call
  $PY experiments/contextual_target_selection/select_targets.py             # 3 Qwen calls (source zone, destination zone, destination context)

Reused: stored Qwen A1 interpretation and DINO candidates (task_conditioned_target_grounding), stored red-cube candidates (retention), v0.2
rule_based_parse / make_roi / ModelWorkerClient / parse_json_object, placement_target_grounding qwen_call / parse_reply, semantic_navigation_integration
full_plan / leg_to_region / leg_record (plan_variant, v0.2 A*, whole-body sweep), task_route_integration build_context / layers / draw_path.
"""
import argparse
import hashlib
import json
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
import placement_target as P  # noqa: E402
from semantic_task_nav.agent.task_parser import rule_based_parse  # noqa: E402
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file, sha256_text  # noqa: E402
from semantic_task_nav.geometry.coordinate_frames import box_area, iou  # noqa: E402
from semantic_task_nav.perception.target_inspection import make_roi  # noqa: E402

RI, SN, PT, FN = P.RI, P.SN, P.PT, P.FN
_SN = {"type": ["string", "null"]}
DEST_SCHEMA = {"type": "object", "required": ["candidates", "best_destination_id"],
               "properties": {"candidates": {"type": "array", "items": {"type": "object", "required": ["id", "placement_role"],
                                                                        "properties": {"id": {"type": "number"}, "placement_role": {"type": "string", "enum": ["suitable", "possible", "unsuitable"]}}}},
                              "best_destination_id": {"type": ["number", "null"]}, "reason": _SN}}
SRC_SCHEMA = {"type": "object", "required": ["candidates", "object_to_move_id"],
              "properties": {"candidates": {"type": "array", "items": {"type": "object", "required": ["id", "role"]}}, "object_to_move_id": {"type": ["number", "null"]}, "reason": _SN}}
VERIFY_SCHEMA = {"type": "object", "required": ["placement_role"], "properties": {"placement_role": {"type": "string", "enum": ["suitable", "possible", "unsuitable"]}}}


# ================================================================================ pure functions (tested offline)
def snap_anchor(qbox, objects, thr):
    """Qwen bbox of the selected facility -> stored verified object box (IoU >= thr), else the Qwen box itself."""
    if not qbox:
        return None, "no Qwen box"
    best = max(objects, key=lambda o: iou(qbox, o["bbox_xyxy"]), default=None)
    if best is not None and iou(qbox, best["bbox_xyxy"]) >= thr:
        return [round(v, 2) for v in best["bbox_xyxy"]], f"snapped to stored {best['object_id']} (IoU {iou(qbox, best['bbox_xyxy']):.2f})"
    return qbox, "Qwen box (no stored object with enough overlap)"


def frac_inside(b, z):
    ix, iy = max(0.0, min(b[2], z[2]) - max(b[0], z[0])), max(0.0, min(b[3], z[3]) - max(b[1], z[1]))
    return ix * iy / box_area(b) if box_area(b) > 0 else 0.0


def zone_members(cands, zone, region_frac):
    """Centre inside the zone or >= 50 % of the box inside. Boxes larger than region_frac of the zone are regions (recorded, not numbered)."""
    za = box_area(zone)
    inc, regions = [], []
    for c in cands:
        if P.centre_in(c["bbox_xyxy"], zone) or frac_inside(c["bbox_xyxy"], zone) >= 0.5:
            (regions if box_area(c["bbox_xyxy"]) > region_frac * za else inc).append(c)
    return inc, regions


def view_rect(zone, boxes, img_wh):
    """Zone extended so that every numbered box is drawn completely."""
    xs0 = [zone[0]] + [b[0] for b in boxes]
    ys0 = [zone[1]] + [b[1] for b in boxes]
    xs1 = [zone[2]] + [b[2] for b in boxes]
    ys1 = [zone[3]] + [b[3] for b in boxes]
    return [int(max(0, np.floor(min(xs0)))), int(max(0, np.floor(min(ys0)))), int(min(img_wh[0], np.ceil(max(xs1)))), int(min(img_wh[1], np.ceil(max(ys1))))]


def zoom_scale(rect, tcfg):
    """v0.2 make_roi zoom rule on a rectangle."""
    long_side = max(rect[2] - rect[0], rect[3] - rect[1])
    return max(1.0, min(tcfg["roi_max_zoom"], tcfg["roi_zoom_long_side_px"] / long_side))


def _num(v, n):
    return int(v) if isinstance(v, (int, float)) and float(v).is_integer() and 1 <= int(v) <= n else None


def decide_destination(zone_obj, verify_obj, n):
    if zone_obj is None:
        return {"status": "AMBIGUOUS", "number": None, "why": "zone reply invalid"}
    rated = {int(c["id"]): c for c in zone_obj["candidates"] if isinstance(c.get("id"), (int, float))}
    k = _num(zone_obj.get("best_destination_id"), n)
    if k is None:
        any_ok = any(c["placement_role"] in ("suitable", "possible") for c in rated.values())
        return {"status": "AMBIGUOUS" if any_ok else "NOT_FOUND", "number": None, "why": f"best_destination_id {zone_obj.get('best_destination_id')}"}
    if rated.get(k, {}).get("placement_role") != "suitable":
        return {"status": "UNVERIFIED", "number": k, "why": f"best {k} rated {rated.get(k, {}).get('placement_role')} in the zone call"}
    if verify_obj is None:
        return {"status": "UNVERIFIED", "number": k, "why": "context verification missing or invalid"}
    if verify_obj["placement_role"] == "unsuitable":
        return {"status": "UNVERIFIED", "number": k, "why": "context verification says unsuitable"}
    return {"status": "SELECTED", "number": k, "why": f"zone: suitable + best; context verification: {verify_obj['placement_role']}"}


def decide_source(src_obj, n):
    if src_obj is None:
        return {"status": "AMBIGUOUS", "number": None, "why": "source reply invalid"}
    k = _num(src_obj.get("object_to_move_id"), n)
    return {"status": "SELECTED", "number": k, "why": "object_to_move_id"} if k else {"status": "AMBIGUOUS", "number": None, "why": f"object_to_move_id {src_obj.get('object_to_move_id')}"}


# ================================================================================ drawing
def numbered(img, rect, scale, cands, path):
    x0, y0, x1, y1 = rect
    im = cv2.resize(img[y0:y1, x0:x1], (round((x1 - x0) * scale), round((y1 - y0) * scale)), interpolation=cv2.INTER_CUBIC)
    for k, c in enumerate(cands, 1):
        b = [int(round((c["bbox_xyxy"][0] - x0) * scale)), int(round((c["bbox_xyxy"][1] - y0) * scale)), int(round((c["bbox_xyxy"][2] - x0) * scale)), int(round((c["bbox_xyxy"][3] - y0) * scale))]
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(str(k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x, y = max(0, b[0]), max(th + 4, b[1] - 3)
        cv2.rectangle(im, (x, y - th - 4), (x + tw + 6, y + 2), (0, 0, 0), -1)
        cv2.putText(im, str(k), (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), im)
    return list(im.shape[1::-1])


def context_crop(img, box, pad, path):
    h, w = img.shape[:2]
    p = max(pad["ratio"] * max(box[2] - box[0], box[3] - box[1]), pad["min_px"])
    x0, y0, x1, y1 = int(max(0, box[0] - p)), int(max(0, box[1] - p)), int(min(w, box[2] + p)), int(min(h, box[3] + p))
    s = 1024 / max(x1 - x0, y1 - y0)
    im = cv2.resize(img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(im, (round((box[0] - x0) * s), round((box[1] - y0) * s)), (round((box[2] - x0) * s), round((box[3] - y0) * s)), (0, 255, 0), 2)
    cv2.imwrite(str(path), im)
    return {"crop_xyxy": [x0, y0, x1, y1], "scale": round(s, 4)}


# ================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    rcfg, gcfg, v02 = RI.load_cfg(), PT.load_cfg(), load_config()
    mcfg, tcfg = v02["models"], v02["models"]["target_inspection"]
    assert cfg["instruction"] == rcfg["command"]
    img_path = Path(rcfg["image"])
    assert sha256_file(img_path) == rcfg["image_sha256"]
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    H, W = img.shape[:2]
    fp_before = code_fingerprint()["sha256"]
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_dryrun" if args.dry_run else ""))
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    (out / "config_snapshot.yaml").write_text((HERE / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    ctx = RI.build_context(rcfg, gcfg, v02, img)
    run = (HERE / cfg["reuse"]["interpretation_run"]).resolve()

    # ---------------------------------------------------------------- 1. task context (stored A1 reply + deterministic parse; no new call)
    a1 = json.loads((run / "task_interpretation.json").read_text(encoding="utf-8"))["A1"]
    assert a1["instruction"] == "T1" and a1["image"] == "full" and a1["parsed"]
    A = a1["parsed"]
    parse = rule_based_parse(cfg["instruction"])
    facility = A.get("selected_destination")
    fac_box = next((c.get("bbox_image_px") for c in A["destination_target_candidates"] if c.get("name_en") == facility), None)
    verified = [o for o in ctx["st"]["objects"] if o.get("accepted")]
    d_anchor, d_how = snap_anchor(fac_box, verified, cfg["anchors"]["snap_iou"])
    if d_anchor is None:
        d_anchor, d_how = [round(v, 1) for v in ctx["st"]["qwen_box_px"]], "fallback: stored Qwen L2 area_5 box"
    src_cls = (parse.get("target") or {}).get("source_reference")
    src_objs = [o for o in verified if o["class_name"] == src_cls]
    s_anchor = [round(v, 2) for v in src_objs[0]["bbox_xyxy"]] if len(src_objs) == 1 else None
    context = {"instruction": cfg["instruction"], "stored_qwen_A1": {"run": str(run), "raw_sha256": sha256_file(run / "qwen/A_A1_raw.txt"),
                                                                    "source_region": A["source_region"], "source_object": A["source_object"],
                                                                    "destination_region": A["destination_region"], "selected_destination": facility},
               "rule_parse": parse, "destination_anchor": {"box": d_anchor, "how": d_how, "facility_from_qwen": facility},
               "source_anchor": {"box": s_anchor, "how": f"rule_parse source_reference '{src_cls}' -> {len(src_objs)} stored verified object(s)"}}

    # ---------------------------------------------------------------- 2. zones and candidates (stored; nothing dropped)
    dz, sz = make_roi(d_anchor, (W, H), tcfg).crop_xyxy, make_roi(s_anchor, (W, H), tcfg).crop_xyxy if s_anchor else None
    merged = json.loads((run / "all_candidates.json").read_text(encoding="utf-8"))["merged"]
    d_inc, d_reg = zone_members(merged, dz, cfg["candidates"]["destination_region_scale_frac"])
    d_inc = sorted(d_inc, key=lambda c: (c["bbox_xyxy"][1], c["bbox_xyxy"][0]))                 # reading order (top-left first), not by score
    rr = json.loads(((HERE / cfg["reuse"]["retention_run"]).resolve() / "result.json").read_text(encoding="utf-8"))
    e1p = next(p for p in rr["photos"] if p["photo_id"] == "U25070552")
    s_inc = sorted([{"id": f"cand_r{c['rank']}", "bbox_xyxy": c["bbox_xyxy"], "state": c["state"], "rank": c["rank"]} for c in e1p["candidates"]
                    if sz and P.centre_in(c["bbox_xyxy"], sz)], key=lambda c: (c["bbox_xyxy"][1], c["bbox_xyxy"][0]))
    assert len(d_inc) <= cfg["candidates"]["max_numbered_per_image"] and len(s_inc) <= cfg["candidates"]["max_numbered_per_image"], "split into several images (not implemented: not needed on E1)"
    dv, sv = view_rect(dz, [c["bbox_xyxy"] for c in d_inc], (W, H)), view_rect(sz, [c["bbox_xyxy"] for c in s_inc], (W, H))
    ds, ss = zoom_scale(dv, tcfg), zoom_scale(sv, tcfg)
    dimg, simg = out / "inputs/destination_zone_numbered.png", out / "inputs/source_zone_numbered.png"
    dsize, ssize = numbered(img, dv, ds, d_inc, dimg), numbered(img, sv, ss, s_inc, simg)
    zones = {"destination": {"zone_px": dz, "view_px": dv, "scale": ds, "image_size": dsize, "numbered": [{"number": k, "id": c["id"], "bbox_xyxy": c["bbox_xyxy"]} for k, c in enumerate(d_inc, 1)],
                             "region_scale_not_numbered": [{"id": c["id"], "bbox_xyxy": c["bbox_xyxy"]} for c in d_reg]},
             "source": {"zone_px": sz, "view_px": sv, "scale": ss, "image_size": ssize, "numbered": [{"number": k, **c} for k, c in enumerate(s_inc, 1)]}}
    prompts = {"destination_zone": cfg["qwen"]["destination_zone"]["template"].replace("{instruction}", cfg["instruction"])
               .replace("{destination_description}", A["destination_region"].get("description") or "").replace("{facility}", facility or ""),
               "source_zone": cfg["qwen"]["source_zone"]["template"].replace("{instruction}", cfg["instruction"])
               .replace("{source_description}", A["source_region"].get("description") or "").replace("{facility}", src_cls or "").replace("{source_object}", A["source_object"].get("name_en") or ""),
               "destination_verify": cfg["qwen"]["destination_verify"]["template"].replace("{instruction}", cfg["instruction"])}
    (out / "prompts.json").write_text(json.dumps({**prompts, "sha256": {k: sha256_text(v) for k, v in prompts.items()}}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "context_and_zones.json").write_text(json.dumps({"context": context, "zones": zones}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("anchors", d_anchor, d_how, "|", s_anchor, "| zones", dz, sz, "| numbered", len(d_inc), len(s_inc), "| regions", [c["id"] for c in d_reg])
    if args.dry_run:
        print("dry run written", out)
        return out

    # ---------------------------------------------------------------- 3. Qwen roles (separate source / destination calls) + one context verification
    from semantic_task_nav.perception.model_client import ModelWorkerClient
    client, calls, t0 = ModelWorkerClient(mcfg, out / "worker_logs"), [], time.time()
    q = cfg["qwen"]
    try:
        client.start()
        raw_s = P.qwen_call(client, prompts["source_zone"], simg, q["max_new_tokens"], "source_zone", q["source_zone"]["prompt_id"], calls)
        src_obj, src_err = P.parse_reply(raw_s, SRC_SCHEMA)
        raw_d = P.qwen_call(client, prompts["destination_zone"], dimg, q["max_new_tokens"], "destination_zone", q["destination_zone"]["prompt_id"], calls)
        dst_obj, dst_err = P.parse_reply(raw_d, DEST_SCHEMA)
        k = _num((dst_obj or {}).get("best_destination_id"), len(d_inc))
        ver_obj, ver_err, ver_crop = None, "NOT_RUN (no best destination)", None
        if k:
            vp = out / "inputs/destination_context_verify.png"
            ver_crop = context_crop(img, d_inc[k - 1]["bbox_xyxy"], q["destination_verify"]["context_pad"], vp)
            raw_v = P.qwen_call(client, prompts["destination_verify"], vp, q["max_new_tokens"], "destination_verify", q["destination_verify"]["prompt_id"], calls)
            ver_obj, ver_err = P.parse_reply(raw_v, VERIFY_SCHEMA)
    finally:
        client.close()
    (out / "model_calls.json").write_text(json.dumps({"calls": calls, "n_qwen_calls": len(calls), "n_dino_calls": 0, "worker_env": client.env_info,
                                                       "total_wall_s": round(time.time() - t0, 1)}, indent=1, ensure_ascii=False), encoding="utf-8")
    dd, sd = decide_destination(dst_obj, ver_obj, len(d_inc)), decide_source(src_obj, len(s_inc))
    dest = d_inc[dd["number"] - 1] if dd["number"] else None
    src = s_inc[sd["number"] - 1] if sd["number"] else None
    prior = json.loads((run / "object_role_analysis.json").read_text(encoding="utf-8"))
    prior_role = {r["id"]: r["role"] for r in prior["analysed"]}
    selection = {"source": {**sd, "candidate": src, "reply": src_obj, "error": src_err},
                 "destination": {**dd, "candidate": dest and {k2: dest[k2] for k2 in ("id", "bbox_xyxy", "score", "phrases")}, "reply": dst_obj, "error": dst_err,
                                 "context_verification": {"reply": ver_obj, "error": ver_err, "crop": ver_crop},
                                 "earlier_isolated_role_analysis": prior_role.get(dest["id"]) if dest else None},
                 "per_candidate_destination_roles": [{"number": c.get("id"), "candidate": d_inc[int(c["id"]) - 1]["id"] if _num(c.get("id"), len(d_inc)) else None, **c}
                                                     for c in (dst_obj or {}).get("candidates", [])],
                 "never": "CONFIRMED (every SELECTED stays VERIFY_REQUIRED)"}
    (out / "selection.json").write_text(json.dumps(selection, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("source", sd, src and src["id"], "| destination", dd, dest and dest["id"])

    # ---------------------------------------------------------------- 4. paths A and B (existing planner)
    fr, s, pad, m = ctx["fr"], ctx["fr"].s, ctx["fr"].pad, ctx["ft"]["footprint_margin_cm"]
    L, Wd = map(float, cfg["route"]["robot_cm"])
    old = PT.jload((HERE / cfg["reuse"]["semnav_run"]).resolve() / "integrated_task_plan.json")["plans"][0]
    routes = {"earlier": {"pickup_approach_cm": old["path_A"]["goal_cm"], "path_A_cm": old["path_A"]["length_cm"],
                          "path_B_free_floor": {k2: old["path_B_to_placement"].get(k2) for k2 in ("goal_cm", "length_cm", "gap_cm")}}}
    if src:
        plan = SN.full_plan(ctx, gcfg, src, L, Wd, m, ctx["starts"][0])
        A_ = plan["summary"]["paths"].get("A")
        routes["path_A"] = None if not A_ else {k2: A_[k2] for k2 in ("goal_cm", "length_cm", "waypoints_cm", "heading_deg", "footprint_extent_xy_cm")} | \
            {"gap_cm": A_["approach"]["body_to_target_gap_cm"], "checks": {k2: plan["summary"]["checks"]["A"][k2] for k2 in ("whole_body_sweep", "v02_validator_inscribed_disc", "swept_layer_overlap")}}
        routes["path_A_status"] = "GENERATED" if A_ else f"NOT_GENERATED ({plan['summary']['failure_A']})"
        hA = plan["var"]["selected"].get("A_heading_deg")
        start_bev, navs = (plan["var"]["legs_A"][hA]["goal"]["xy_bev_px"], plan["navs"]) if A_ else (None, plan["navs"])
    else:
        routes["path_A_status"] = "NOT_RUN (source AMBIGUOUS)"
        hA, navs = 0, {h: FN.RectNavMap(ctx["spec_bev"], fr.bev_size, ctx["pcfg"], FN.footprint_dims_cm(L, Wd, m, h), s) for h in (0, 90)}
        start_bev = fr.cm_to_bev(fr.img_to_cm([ctx["starts"][0]]))[0].tolist()
        routes["path_B_note"] = "no pick-up approach: destination reachability is computed from the robot start (NOT a path B)"
    if dest and start_bev is not None:
        poly = fr.img_to_cm(PT.box_poly(dest["bbox_xyxy"])).tolist()
        bc = fr.img_to_cm([[(dest["bbox_xyxy"][0] + dest["bbox_xyxy"][2]) / 2, dest["bbox_xyxy"][3]]])[0]
        zones_hit = [z["id"] for layer in ("traversable", "static_obstacles", "no_go") for z in ctx["spec_cm"][layer]
                     if cv2.pointPolygonTest(np.float32(z["polygon"]), (float(bc[0]), float(bc[1])), False) >= 0]
        rec = SN.leg_record(ctx, SN.leg_to_region(ctx, gcfg, navs, hA, start_bev, [poly], "selected_destination", L, Wd))
        routes["path_B"] = {**{k2: rec.get(k2) for k2 in ("path_B_feasible", "goal_cm", "length_cm", "gap_cm", "heading_deg", "waypoints_cm", "footprint_extent_xy_cm",
                                                            "whole_body_sweep_ok", "v02_valid")},
                            "layers_clear": (rec.get("layers") or {}).get("clear"), "target_floor_contact_cm": [round(float(v), 1) for v in bc],
                            "target_contact_zone": zones_hit or ["NONE (outside every DRAFT zone: not drivable)"],
                            "note": "floor contact = bottom-centre of the box; the top surface is higher (unknown)"}
        routes["path_B_status"] = ("GENERATED" if rec["path_B_feasible"] else "NOT_GENERATED") + ("" if src else " (reachability from the robot start only)")
    else:
        routes["path_B_status"] = f"NOT_RUN (destination {dd['status']})"
    plan_ok = sd["status"] == "SELECTED" and dd["status"] == "SELECTED" and routes["path_A_status"] == "GENERATED" and routes["path_B_status"] == "GENERATED"
    routes["task_plan"] = "PLAN_GENERATED_CONDITIONAL (targets VERIFY_REQUIRED; nothing executed)" if plan_ok else "NOT_CONFIRMED"
    routes["A_B_connected"] = bool(plan_ok and routes["path_A"]["goal_cm"] == routes["path_B"]["waypoints_cm"][0])
    (out / "routes.json").write_text(json.dumps(routes, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("routes", routes["path_A_status"], routes["path_B_status"], routes["task_plan"])

    # ---------------------------------------------------------------- figure (no reference drawn)
    mp = PT.Mapper("photo", fr)
    im = img.copy()
    RI.layers(im, mp, ctx, text=False)
    for name, z in (("source zone", sz), ("destination zone", dz)):
        if z:
            cv2.rectangle(im, tuple(z[:2]), (z[2] - 1, z[3] - 1), (255, 255, 255), 2)
            P._t(im, name, (z[0] + 6, z[3] - 8), (255, 255, 255), 0.6)
    of = old["path_B_to_placement"]
    if of.get("waypoints_cm") and len(of["waypoints_cm"]) > 1:
        cv2.polylines(im, [mp(of["waypoints_cm"])], False, (160, 160, 160), 2, cv2.LINE_AA)
    for key, col, tag in (("path_A", (255, 160, 40), "A"), ("path_B", (40, 150, 255), "B")):
        p = routes.get(key)
        if p and p.get("waypoints_cm") and len(p["waypoints_cm"]) > 1:
            RI.draw_path(im, mp, {"waypoints_cm": p["waypoints_cm"], "footprint_extent_xy_cm": p["footprint_extent_xy_cm"], "goal_cm": p["goal_cm"]}, col, tag, 15, coords=False)
    for c, col, lab in ((src, (0, 0, 255), "SOURCE"), (dest, (0, 255, 255), "DESTINATION")):
        if c:
            b = [int(v) for v in c["bbox_xyxy"]]
            cv2.rectangle(im, (b[0] - 3, b[1] - 3), (b[2] + 3, b[3] + 3), col, 3)
            P._t(im, f"{lab} {c['id']}", (b[0] - 40, b[3] + 22), col, 0.6)
    lines = ["# CONTEXTUAL TARGET SELECTION (E1) - plan only, nothing executed", f"source: {sd['status']} {src and src['id']} ({sd['why']})",
             f"destination: {dd['status']} {dest and dest['id']} ({dd['why']})", f"path A: {routes['path_A_status']} {(routes.get('path_A') or {}).get('length_cm')} cm",
             f"path B: {routes['path_B_status']} {(routes.get('path_B') or {}).get('length_cm')} cm, gap {(routes.get('path_B') or {}).get('gap_cm')} cm",
             f"earlier path B (bay free floor, grey): {routes['earlier']['path_B_free_floor']}", f"task plan: {routes['task_plan']}", "",
             "orange = path A; blue = new path B; grey = earlier path B; white = zones"]
    cv2.imwrite(str(out / "selection_and_routes.png"), np.hstack([im, P.panel(im.shape[0], lines, 900)]))

    # ---------------------------------------------------------------- evaluation (reference read only here)
    ref = yaml.safe_load((HERE / cfg["reuse"]["evaluation_reference"]).read_text(encoding="utf-8"))
    lab = yaml.safe_load((HERE / cfg["reuse"]["funnel_config"]).read_text(encoding="utf-8"))["labels"]["U25070552"]
    ev = {"read_after": "selection.json and routes.json were written before this point",
          "source_iou_with_AI_prelim_cube_label": round(iou(src["bbox_xyxy"], lab), 3) if src else None,
          "destination_iou_with_P_DST": round(iou(dest["bbox_xyxy"], ref["objects"]["P_DST"]["bbox_xyxy_px"]), 3) if dest else None,
          "P_DST_numbered": next((z["number"] for z in zones["destination"]["numbered"] if iou(z["bbox_xyxy"], ref["objects"]["P_DST"]["bbox_xyxy_px"]) >= ref["match_iou"]), None),
          "P_DST_rated": None, "reference_status": "AI_DRAWN (user to confirm); cube label AI_PRELIMINARY_REVIEW"}
    if ev["P_DST_numbered"]:
        ev["P_DST_rated"] = next((c for c in (dst_obj or {}).get("candidates", []) if _num(c.get("id"), len(d_inc)) == ev["P_DST_numbered"]), None)
    (out / "evaluation.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    validation = {"model_calls": {"qwen": len(calls), "dino": 0}, "no_candidate_filtered_by_name": True, "region_scale_boxes_recorded": [c["id"] for c in d_reg],
                  "source_and_destination_separate_calls": True, "v02_code_unchanged": fp_before == code_fingerprint()["sha256"],
                  "evaluation_written_after_decisions": (out / "selection.json").stat().st_mtime <= (out / "evaluation.json").stat().st_mtime}
    (out / "validation.json").write_text(json.dumps(validation, indent=1), encoding="utf-8")
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "manifest.json").write_text(json.dumps({"files_sha256": files, "code_sha256": {p.name: sha256_file(p) for p in sorted(HERE.glob("*.py"))}}, indent=1), encoding="utf-8")
    print(json.dumps({"evaluation": ev, "validation": validation}, indent=1, ensure_ascii=False, default=str))
    return out


if __name__ == "__main__":
    main()
