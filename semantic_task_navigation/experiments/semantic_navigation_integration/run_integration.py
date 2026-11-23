#!/usr/bin/env python3
"""Semantic context map + navigation cost map + task-aware decision on E1 (one command, one image; offline).

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
  $PY experiments/semantic_navigation_integration/run_integration.py

No model call, no GPU, no hardware. Reused (imported, unchanged): task_route_integration.build_context / summarize / swept_layer_overlap /
layers / draw_path / robot_start / panel; task_plan_prototype plan_variant / approach_goal / make_leg / RectNavMap / body_free / body_gap_cm;
v0.2 rule_based_parse, cost_field, A*, validate_path, select_goal scoring. New: context_map.py (Exp. 1), cost_map.py (Exp. 2),
decision.py (Exp. 3), the destination evaluation, the ablation table, figures and checks below.
"""
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent / "task_route_integration"))
sys.path.insert(0, str(HERE))
import context_map as CM  # noqa: E402
import cost_map as KM  # noqa: E402
import decision as DM  # noqa: E402
import route_integration as RI  # noqa: E402
from semantic_task_nav.agent.task_parser import rule_based_parse  # noqa: E402
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file  # noqa: E402
from semantic_task_nav.geometry.coordinate_frames import iou  # noqa: E402
from semantic_task_nav.navigation.planner import cost_field  # noqa: E402

PT, FN = RI.PT, RI.FN
COL = {"INSTRUCTION_GIVEN": (0, 255, 255), "VLM_INFERRED": (0, 165, 255), "GEOMETRY_DERIVED": (255, 255, 0)}
CLASS_COL = {"REACHABLE": None, "UNREACHABLE_DISCONNECTED": (140, 30, 140), "UNREACHABLE_FOOTPRINT": (160, 160, 160), "STATIC_OBSTACLE": (60, 60, 235),
             "NO_GO": (200, 60, 200), "UNKNOWN_UNVERIFIED": (75, 75, 75), "OUTSIDE_WORKSPACE": (20, 20, 20)}


def res(p):
    return (HERE / p).resolve()


def r1(v):
    return None if v is None else round(float(v), 1)


# ------------------------------------------------------------------------------------------------ inputs
def stored_candidates(cfg, rcfg, ctx):
    rr = PT.jload(res(cfg["stored"]["retention_run"]) / "result.json")
    e1p = next(p for p in rr["photos"] if p["photo_id"] == cfg["photo_id"])
    qc = PT.jload(res(cfg["stored"]["crop_comparison_run"]) / "result.json")
    out = []
    for c in e1p["candidates"]:
        views = None
        for q in qc["candidates"]:
            if iou(c["bbox_xyxy"], q["bbox_xyxy_raw_px"]) >= rcfg["condition_B"]["crop_verdict_match_iou"]:
                views = {v: {"status": q[k]["status"], "observed_object": (q[k].get("answer") or {}).get("observed_object"),
                             "observed_color": (q[k].get("answer") or {}).get("observed_color")}
                         for v, k in (("A_stored", "stored_A"), ("B_no_marker", "B"), ("C_more_context", "C"))}
        out.append({"id": f"cand_r{c['rank']}", "rank": c["rank"], "bbox_xyxy": c["bbox_xyxy"], "source_types": c["source_types"], "score": c["score"], "state": c["state"],
                    "stored_qwen": c["stored_qwen"] and {k: c["stored_qwen"].get(k) for k in ("candidate_id", "status", "observed_color", "observed_object", "reason")},
                    "crop_views": views, "floor_cm": np.round(ctx["fr"].img_to_cm(PT.box_poly(c["bbox_xyxy"])), 2).tolist()})
    return out, e1p, qc


def context_inputs(cfg, rcfg, ctx, cands, start_cm, body_fits):
    st, fr = ctx["st"], ctx["fr"]
    H, W = ctx["img"].shape[:2]
    chk = next(c for c in ctx["first"]["stages"]["3_robot_start"]["verification"]["checks"] if c["candidate_id"] == ctx["robot"]["object_id"])
    objs = []
    for o in st["objects"]:
        if o["class_name"] == "traffic_cone":
            base = [(o["bbox_xyxy"][0] + o["bbox_xyxy"][2]) / 2, o["bbox_xyxy"][3]]
            fp, note, crop = np.round(fr.img_to_cm([base])[0], 1).tolist(), "ESTIMATED floor contact = bottom centre of the box", o["basis"]
        else:
            fp, note = None, "tall object: the image box is not a floor footprint (no floor position given)"
            crop = f"stored Qwen crop check: {(o.get('verification') or {}).get('status')}"
        objs.append({"object_id": o["object_id"], "class_name": o["class_name"], "bbox_xyxy": o["bbox_xyxy"], "source": o.get("source"), "crop": crop,
                     "floor_point_cm": fp, "floor_note": note})
    draft = {z["id"]: z for z in json.loads(Path(cfg["stored"]["draft_zone_source"]).read_text(encoding="utf-8"))["zones"]}
    zones = []
    for layer in ("traversable", "static_obstacles", "no_go"):
        for z, zc in zip(ctx["zmap"][layer], ctx["spec_cm"][layer]):
            assert z["id"] == zc["id"]
            meta = draft.get(z["id"], {})
            zones.append({"id": z["id"], "layer": layer, "name_ko": meta.get("name_ko"), "role": meta.get("role"), "access_draft": meta.get("access_draft"),
                          "polygon_px": np.round(z["polygon"], 1).tolist(), "polygon_cm": np.round(zc["polygon"], 2).tolist()})
    qs = st["structured"]
    area_px = {a["id"]: [a["bbox_2d"][0] / 1000 * W, a["bbox_2d"][1] / 1000 * H, a["bbox_2d"][2] / 1000 * W, a["bbox_2d"][3] / 1000 * H] for a in qs["functional_areas"]}
    bl = st["bay"]
    ff = bl["free_floor"]["margin_0px"]
    return {"command": cfg["command"], "rule_parse": rule_based_parse(cfg["command"]), "qwen_structured": qs,
            "qwen_meta": {"run": st["qwen_run"].name, "instruction": rcfg["semantics"]["stored_destination_instruction"], "raw_response_sha256": st["raw_response_sha256"]},
            "robot": {"bbox_xyxy": [round(v, 1) for v in ctx["robot"]["bbox_xyxy"]], "start_cm": [round(float(v), 1) for v in start_cm],
                      "crop_answer": f"{chk['status']} ({chk['answer']['observed_object']})", "body_fits_at_start": body_fits},
            "objects": objs, "bay": {"polygon_px": bl["bay_polygon_xy"], "polygon_cm": np.round(ctx["bay_cm"], 2).tolist(), "status": bl["status"],
                                     "not_verified": bl["not_verified"], "stored_truck_check": next((c for c in bl["checks"] if "truck" in c["check"]), None)},
            "placement_floor": {"polygons_px": ff["polygons_xy"], "polygons_cm": [np.round(p, 2).tolist() for p in ctx["place_candidate_cm"]], "basis": ff["basis"]},
            "area_boxes_px": area_px, "area_floor_cm": {k: np.round(fr.img_to_cm(PT.box_poly(b)), 1).tolist() for k, b in area_px.items()},
            "zones": zones, "candidates": cands, "image_size": (W, H), "floor_canvas_wh": fr.bev_size, "floor_px_per_cm": fr.s, "floor_pad_cm": fr.pad}


# ------------------------------------------------------------------------------------------------ plans
def leg_to_region(ctx, gcfg, navs, h_from, start_bev, polys_cm, anchor, L, W):
    """Leg from start_bev to the nearest reachable pose of the region (task_plan_prototype approach rule), every heading; a heading change
    needs the conservative rotation test at start_bev. Selection as in plan_variant: smallest body gap, then shortest path."""
    fr, s, pad, pcfg = ctx["fr"], ctx["fr"].s, ctx["fr"].pad, ctx["pcfg"]
    polys_bev = [fr.cm_to_bev(p).tolist() for p in polys_cm]
    legs = {}
    for h, nav in navs.items():
        g = PT.approach_goal(nav, anchor, polys_bev, list(start_bev), "dropoff_approach", pcfg, gcfg["approach"]["slack_cm"] * s, s, pad)
        leg, _ = PT.make_leg(nav, nav.extent_cm, h, g, list(start_bev), pcfg, ctx["free_hi"], fr, gcfg["bev"], f"B_to_{anchor}")
        if g["status"] == "GOAL_SELECTED":
            g["body_to_target_gap_cm"] = round(min(FN.body_gap_cm(p, g["xy_cm"], FN.footprint_dims_cm(L, W, 0, h), ctx["size_cm"], pad, 8) for p in polys_cm), 1)
            g["xy_image_px"] = [round(v, 1) for v in fr.cm_to_img([g["xy_cm"]])[0]]
            top = g["candidates_top_k"][0]
            g["v02_goal_score_cm"] = round(top["path_cost_px"] / s + gcfg["approach"]["distance_weight"] * top["distance_to_target_cm"], 2)
        if (leg["failure"] or {}).get("code") == "ZERO_LENGTH_PATH":          # already at the approach pose: nothing to drive
            leg["feasible"], leg["note"] = True, "ZERO_LENGTH_PATH: the start already is the approach pose"
        if h != h_from:
            rot = navs[h_from].rotation_clear(start_bev)
            leg["rotation_at_leg_start"] = rot
            if leg["feasible"] and not rot["ok"]:
                leg["feasible"] = False
                leg["failure"] = {"stage": "heading_change", "code": "ROTATION_CLEARANCE_NOT_MET",
                                  "reason": f"circumscribed disc needs {rot['needed_px'] / s:.1f} cm, clearance {rot['clearance_px'] / s:.1f} cm"}
        legs[h] = leg
    feas = [(h, l) for h, l in legs.items() if l["feasible"]]
    sel = min(feas, key=lambda t: (t[1]["goal"]["body_to_target_gap_cm"], (t[1]["path"] or {}).get("length_cm", 0.0))) if feas else None
    return {"anchor": anchor, "legs": legs, "selected_heading_deg": sel[0] if sel else None, "selected": sel[1] if sel else None,
            "failure": None if sel else {str(h): l["failure"] for h, l in legs.items()}}


def leg_record(ctx, region):
    l = region["selected"]
    if l is None:
        return {"path_B_feasible": False, "failure": region["failure"]}
    g, p = l["goal"], l["path"] or {}
    return {"path_B_feasible": True, "heading_deg": l["heading_deg"], "goal_cm": g["xy_cm"], "goal_image_px": g.get("xy_image_px"), "length_cm": p.get("length_cm", 0.0),
            "astar_cost": p.get("astar_cost"), "gap_cm": g["body_to_target_gap_cm"], "v02_goal_score_cm": g["v02_goal_score_cm"], "note": l.get("note"),
            "waypoints_cm": p.get("waypoints_cm", [g["xy_cm"]]), "footprint_extent_xy_cm": l["footprint_extent_xy_cm"],
            "whole_body_sweep_ok": (l["sweep_check"] or {}).get("ok"), "v02_valid": (l["v02_validation"] or {}).get("valid"),
            "layers": RI.swept_layer_overlap(ctx, p["waypoints_cm"], l["footprint_extent_xy_cm"]) if p.get("waypoints_cm") and len(p["waypoints_cm"]) > 1 else None}


def full_plan(ctx, gcfg, c, L, W, m, start_img):
    """Path A (task_route_integration / plan_variant, unchanged) + path B to the placement floor of the bay (from the pick-up approach)."""
    RI.set_target(ctx, c["bbox_xyxy"])
    var, rt = PT.plan_variant(c["id"], L, W, m, start_img, ctx)
    summ = RI.summarize(var, ctx, L, W, m, {"candidate": c["id"], "state": c["state"]})
    out = {"summary": summ, "navs": rt["navs"], "var": var, "B_place": None}
    hA = var["selected"].get("A_heading_deg")
    if hA is not None:
        gA = var["legs_A"][hA]["goal"]["xy_bev_px"]
        out["B_place_region"] = leg_to_region(ctx, gcfg, rt["navs"], hA, gA, ctx["place_candidate_cm"], "placement_floor", L, W)
        out["B_place"] = leg_record(ctx, out["B_place_region"])
    out["path_A_B_feasible"] = bool("A" in summ["paths"] and out["B_place"] and out["B_place"]["path_B_feasible"])
    return out


def plan_brief(p):
    if p is None:
        return None
    s = p["summary"]
    A = s["paths"].get("A")
    return {"path_A_feasible": A is not None, "path_A_length_cm": A and A["length_cm"], "pickup_gap_cm": A and A["approach"]["body_to_target_gap_cm"],
            "pickup_approach_cm": A and A["goal_cm"], "heading_A_deg": A and A["heading_deg"], "failure_A": s["failure_A"],
            "path_B_place_feasible": bool(p["B_place"] and p["B_place"]["path_B_feasible"]), "path_B_place_length_cm": p["B_place"] and p["B_place"].get("length_cm"),
            "placement_gap_cm": p["B_place"] and p["B_place"].get("gap_cm"), "placement_approach_cm": p["B_place"] and p["B_place"].get("goal_cm"),
            "path_A_B_feasible": p["path_A_B_feasible"], "v02_codes_A_by_heading": s["v02_codes_by_heading"]["A"],
            "zero_length_A": A is None and "ZERO_LENGTH_PATH" in s["v02_codes_by_heading"]["A"].values()}


def feas_text(b):
    if not b:
        return "NOT_PLANNED"
    if b["zero_length_A"]:
        return "ZERO_LENGTH_A (the start already is the pick-up approach; plan_variant generates no path A, so no B)"
    return str(b["path_A_B_feasible"])


# ------------------------------------------------------------------------------------------------ figures
def _line(img, a, b, col, dashed):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not dashed:
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), col, 2, cv2.LINE_AA)
        return
    n = max(1, int(np.linalg.norm(b - a) // 10))
    for i in range(0, n, 2):
        p, q = a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n
        cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), col, 2, cv2.LINE_AA)


def fig_semantic(img, cmap, inp, out):
    im = (img * 0.55).astype(np.uint8)
    nodes = {n["id"]: n for n in cmap["nodes"]}
    anchor = {"task": (960, 40)}
    for n in cmap["nodes"]:
        if "bbox_xyxy" in n:
            b = n["bbox_xyxy"]
            anchor[n["id"]] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        elif "polygon_xy" in n:
            anchor[n["id"]] = tuple(np.mean(n["polygon_xy"], axis=0))
        elif "polygons_xy" in n:
            anchor[n["id"]] = tuple(np.mean(n["polygons_xy"][0], axis=0))
    for z in inp["zones"]:
        p = np.round(z["polygon_px"]).astype(np.int32)
        cv2.polylines(im, [p], True, (200, 200, 200), 1, cv2.LINE_AA)
        PT._t(im, f"{z['id']} {z['role']} [DRAFT]", np.mean(p, axis=0).astype(int) + np.array([-60, 0]), (200, 200, 200), 0.42)
    cv2.polylines(im, [np.round(inp["bay"]["polygon_px"]).astype(np.int32)], True, (230, 200, 40), 2, cv2.LINE_AA)
    for p in inp["placement_floor"]["polygons_px"]:
        cv2.polylines(im, [np.round(p).astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    for n in cmap["nodes"]:
        if n["kind"] == "vlm_functional_area":
            b = np.round(n["bbox_xyxy"]).astype(int)
            PT._outline(im, lambda p: np.round(np.asarray(p).reshape(-1, 2)).astype(np.int32), CM.box_poly(b), COL["VLM_INFERRED"], 3 if n["chosen_by_vlm"] else 1, dashed=True)
            PT._t(im, f"{n['id']}: {n['function_name']} (VLM){' CHOSEN' if n['chosen_by_vlm'] else ''}", (b[0] + 6, b[1] + 18), COL["VLM_INFERRED"], 0.45)
        elif n["kind"] in ("object", "robot"):
            b = np.round(n["bbox_xyxy"]).astype(int)
            cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (255, 255, 255), 2 if n["kind"] == "robot" or n.get("class_name") != "traffic_cone" else 1)
            if n.get("class_name") != "traffic_cone":
                PT._t(im, n["id"], (b[0], b[3] + 16), (255, 255, 255), 0.45)
        elif n["kind"] == "target_candidate":
            b = np.round(n["bbox_xyxy"]).astype(int)
            col = {"PROBABLE": (255, 0, 255), "VLM_ACCEPTED": (0, 255, 0), "REJECTED": (0, 0, 160)}.get(n["state"], (150, 150, 150))
            cv2.rectangle(im, tuple(b[:2] - 2), tuple(b[2:] + 2), col, 2 if n["state"] == "PROBABLE" else 1)
            if n["state"] == "PROBABLE":
                PT._t(im, f"r{n['rank']}", (b[2] + 4, b[1]), col, 0.45)
    for r in cmap["relations"]:
        if r["basis"] == "floor_cm_provisional" or r["subject"] not in anchor or r["object"] not in anchor:
            continue
        if nodes[r["subject"]]["kind"] == "target_candidate" and nodes[r["subject"]]["state"] == "REJECTED":
            continue
        _line(im, anchor[r["subject"]], anchor[r["object"]], COL[r["provenance"]], r["status"] != "SUPPORTED")
    PT._t(im, "TASK: red cube (INSTRUCTION), source: conveyor (INSTRUCTION), destination: (VLM)", (700, 46), COL["INSTRUCTION_GIVEN"], 0.5)
    S = cmap["summary"]
    others = [r for r in cmap["relations"] if not (nodes[r["subject"]]["kind"] == "target_candidate" and r["type"] in ("on", "near", "inside"))]
    cand_rel = {}
    for r in cmap["relations"]:
        if nodes[r["subject"]]["kind"] == "target_candidate":
            cand_rel.setdefault(f"{r['type']} {r['object']}", []).append(f"r{nodes[r['subject']]['rank']}")
    lines = ["# SEMANTIC CONTEXT MAP (E1) - no new model call; stored Qwen reply + detector boxes + Python geometry",
             f"nodes {S['n_nodes']} {S['nodes_by_kind']}", f"relations {S['n_relations']} by provenance {S['relations_by_provenance']}",
             f"by status {S['relations_by_status']}", "", "#Line colour = provenance (solid = SUPPORTED on its basis, dashed = UNVERIFIED)",
             "yellow INSTRUCTION_GIVEN | orange VLM_INFERRED | cyan GEOMETRY_DERIVED", "floor-basis relations (PROVISIONAL frame) are listed, not drawn", "",
             "#Relations (excluding per-candidate ones)"]
    lines += [f"{r['type']}: {r['subject']} -> {r['object']} [{r['provenance'][:5]}/{r['status'][:5]}, {r['basis']}]" for r in others]
    lines += ["", "#Per-candidate relations (rank list)"] + [f"{k}: {', '.join(v)}" for k, v in cand_rel.items()]
    lines += ["", "#Conflicts kept"] + [f"VLM '{c['vlm_statement'][:60]}' vs {c['versus']}" for c in cmap["conflicts"]]
    cv2.imwrite(str(out), np.hstack([im, RI.panel(im.shape[0], lines, 760)]))


def bev_canvas(ctx, S):
    fr = ctx["fr"]
    Wc, Hc = round((ctx["size_cm"][0] + 2 * fr.pad) * S), round((ctx["size_cm"][1] + 2 * fr.pad) * S)
    M = np.array([[S, 0, fr.pad * S], [0, S, fr.pad * S], [0, 0, 1]]) @ fr.H
    return cv2.warpPerspective(ctx["img"], M, (Wc, Hc)), PT.Mapper("bev", fr, S)


def cell_image(nav, classes, cost_cm, vmax, S, s, shape):
    k = int(round(nav.cell_px / s * S))
    col = np.zeros(classes.shape + (3,), np.uint8)
    for i, name in enumerate(KM.CLASSES):
        if CLASS_COL[name] is not None:
            col[classes == i] = CLASS_COL[name]
    reach = classes == KM.CLASSES.index("REACHABLE")
    if reach.any():
        heat = cv2.applyColorMap(np.clip(np.nan_to_num(cost_cm, posinf=vmax) / vmax * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        col[reach] = heat[reach]
    big = cv2.resize(col, (classes.shape[1] * k, classes.shape[0] * k), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros(shape + (3,), np.uint8)
    hh, ww = min(shape[0], big.shape[0]), min(shape[1], big.shape[1])
    canvas[:hh, :ww] = big[:hh, :ww]
    return canvas, k


def fig_costmap(ctx, km, classes, navs, cand_nav, plans, place_leg, out, prim, start_cm):
    S, s, fr = 3, ctx["fr"].s, ctx["fr"]
    photo, mb = bev_canvas(ctx, S)
    panels, vmax = [], 600.0
    specs = [(f"heading {prim} deg (primary; start heading assumed {prim}, UNKNOWN)", navs[prim], classes[str(prim)], km[prim]["safety"], km[prim]["distance"], True)]
    for h, nav in navs.items():
        if h == prim:
            continue
        specs.append((f"heading {h} deg from the start (conditional on an UNKNOWN start heading)", nav, classes[str(h)], km[h]["safety"], km[h]["distance"], False))
        v = km[h]["via_rotation_from_primary"]
        specs.append((f"heading {h} deg reached by a heading change at rotation-clear cells ({v['n_switch_cells']} switch cells)", nav, classes[f"{h}_via_rotation"], v["safety"], v["distance"], False))
    for title, nav, cl, cost, dist, primary in specs:
        im, k = cell_image(nav, cl, cost / s, vmax, S, s, photo.shape[:2])
        im = cv2.addWeighted(im, 0.75, (photo * 0.6).astype(np.uint8), 0.25, 0)
        dbig = cv2.resize(np.nan_to_num(dist / s, posinf=1e6).astype(np.float32), (cl.shape[1] * k, cl.shape[0] * k), interpolation=cv2.INTER_NEAREST)
        for lev in range(50, 1000, 50):
            cs, _ = cv2.findContours((dbig <= lev).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, (255, 255, 255), 1)
        for z in ctx["spec_cm"]["traversable"] + ctx["spec_cm"]["static_obstacles"] + ctx["spec_cm"]["no_go"]:
            PT._outline(im, mb, z["polygon"], (255, 255, 255), 1)
            PT._t(im, z["id"], mb(np.mean(z["polygon"], axis=0))[0], (255, 255, 255), 0.4)
        PT._outline(im, mb, ctx["bay_cm"], (230, 200, 40), 2)
        for p in ctx["place_candidate_cm"]:
            PT._outline(im, mb, p, (0, 255, 0), 1)
        if primary:
            rc = km[prim]["rotation_clear"] & np.isfinite(km[prim]["safety"])
            for r, c in np.argwhere(rc)[::3]:
                cv2.circle(im, (int((c + 0.5) * k), int((r + 0.5) * k)), 1, (255, 255, 255), -1)
            for cid, n in cand_nav.items():
                if n.get("approach_cm"):
                    big = cid in plans
                    cv2.drawMarker(im, tuple(mb(n["approach_cm"])[0]), (255, 0, 255) if big else (220, 220, 220), cv2.MARKER_DIAMOND, 12 if big else 6, 2 if big else 1)
                    if big:
                        PT._t(im, f"{cid} {n['path_cost_cm']:.0f}", mb(n["approach_cm"])[0] + np.array([8, -6]), (255, 0, 255), 0.42)
            if place_leg and place_leg.get("goal_cm"):
                cv2.drawMarker(im, tuple(mb(place_leg["goal_cm"])[0]), (0, 255, 0), cv2.MARKER_SQUARE, 12, 2)
        cv2.drawMarker(im, tuple(mb(start_cm)[0]), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        PT._t(im, title, (8, 20), (255, 255, 255), 0.5)
        panels.append(im)
    stack = np.vstack(panels)
    counts = classes["_counts"]
    lines = ["# NAVIGATION COST MAP (BEV, PROVISIONAL floor frame, 2 cm cells)", "colour on REACHABLE cells: v0.2 path cost (SAFETY_COST_ASTAR) from the start, 0 (blue) .. 600 cm-equivalent (red)",
             "white iso-lines: grid geodesic length every 50 cm (DISTANCE_ONLY edges)", "white dots (top panel): rotation-clear reachable cells (circumscribed disc free)",
             "grey = drivable but the body does not fit; purple = fits but disconnected; red static obstacle; magenta no-go; dark grey = outside every drivable zone (UNKNOWN is never free)",
             "magenta diamonds = approach poses of plannable candidates (label: path cost cm); light diamonds = other candidates; green square = placement approach", "",
             "#Cell counts (area cm^2 = count x 4)"]
    for h, cnt in counts.items():
        lines.append(f"{h}: " + ", ".join(f"{k} {v}" for k, v in cnt.items() if v))
    lines += ["", "body: 35x24 cm + 3.5 cm/side transform margin = 42x31 cm (x,y at heading 0); fixed heading per map; mecanum dynamics NOT modelled",
              "real-size safety NOT claimed: floor transform PROVISIONAL_UNVALIDATED, map DRAFT"]
    cv2.imwrite(str(out), np.hstack([stack, RI.panel(stack.shape[0], lines, 760)]))


def fig_plan(ctx, plan_c, cands_by_id, sel_ids, d_id, d_plan, ablation, dest_sel, out):
    mp = PT.Mapper("photo", ctx["fr"])
    im = ctx["img"].copy()
    RI.layers(im, mp, ctx, text=False)
    PT._fill(im, mp, ctx["place_candidate_cm"], (0, 255, 0), 0.35)
    first = plan_c[sel_ids[0]]["summary"] if sel_ids else None
    if first:
        nom = FN.footprint_dims_cm(35, 24, 0, 0)
        RI.robot_start(im, mp, first["start_cm"], nom, FN.footprint_dims_cm(35, 24, first["margin_per_side_cm"], 0))
    cols = [(255, 160, 40), (255, 0, 255), (120, 255, 120)]
    for cid, col in zip(sel_ids, cols):
        p = plan_c[cid]
        PT._outline(im, mp, cands_by_id[cid]["floor_cm"], (0, 0, 255), 3)
        PT._t(im, cid, mp(cands_by_id[cid]["floor_cm"][0])[0] + np.array([8, -24]), (0, 0, 255), 0.5)
        if "A" in p["summary"]["paths"]:
            RI.draw_path(im, mp, p["summary"]["paths"]["A"], col, "A", 15, coords=False)
        if p["B_place"] and p["B_place"]["path_B_feasible"] and len(p["B_place"]["waypoints_cm"]) > 1:
            RI.draw_path(im, mp, {k: p["B_place"][k] for k in ("waypoints_cm", "footprint_extent_xy_cm", "goal_cm")}, (40, 150, 255), "B", 15, coords=False)
    if d_id and d_id not in sel_ids:
        if d_plan and "A" in d_plan["summary"]["paths"]:
            cv2.polylines(im, [mp(d_plan["summary"]["paths"]["A"]["waypoints_cm"])], False, (255, 255, 255), 2, cv2.LINE_AA)
        PT._outline(im, mp, cands_by_id[d_id]["floor_cm"], (255, 255, 255), 3)
        PT._t(im, f"D (navigation only) picks {d_id} [{cands_by_id[d_id]['state']}, no Qwen answer]", mp(cands_by_id[d_id]["floor_cm"][0])[0] + np.array([-420, 40]), (255, 255, 255), 0.5)
    lines = ["# INTEGRATED TASK PLAN (E1) - PLAN ONLY, nothing executed; pick-up/placement NOT claimed",
             "start -> path A -> VERIFY_REQUIRED -> PICKUP_ASSUMED -> path B -> placement approach -> PLACEMENT_ASSUMED", "",
             "#Ablation (same image, same 38 candidates, same 6 VLM areas)"]
    for row in ablation:
        lines.append(f"{row['condition']} [{row['semantic_scale']}] {row['item']}: {row['status']} sel={row['selected'] or '-'} tied={row['tied_set'] or '-'} A={row['path_A_cm'] or '-'} B={row['path_B_cm'] or '-'}")
    lines += ["", f"#Destination (C): {dest_sel}", "light blue = path A (start -> pick-up approach); orange = path B (pick-up approach -> approach of the bay's free floor, green);",
              "thin rectangles = whole body every 15 cm; yellow = approach poses; red = selected target candidate; white = D's pick (navigation only)",
              "gaps the gripper would have to bridge are in integrated_task_plan.json (reach UNVERIFIED)"]
    cv2.imwrite(str(out), np.hstack([im, RI.panel(im.shape[0], lines, 900)]))


# ------------------------------------------------------------------------------------------------ main
def main():
    cfg, rcfg, gcfg, v02 = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8")), RI.load_cfg(), PT.load_cfg(), load_config()
    assert res(cfg["route_config"]) == (RI.HERE / "config.yaml").resolve() and cfg["command"] == rcfg["command"]
    assert cfg["decision"]["approach"] == {"slack_cm": gcfg["approach"]["slack_cm"], "distance_weight": gcfg["approach"]["distance_weight"]}
    img_path = Path(rcfg["image"])
    assert sha256_file(img_path) == rcfg["image_sha256"], "image differs from the declared one"
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    fp_before = code_fingerprint()["sha256"]
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_semnav")
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    shutil.copy2(img_path, out / "inputs/input_E1_undistorted.png")
    for name, src in (("config_snapshot.yaml", HERE / "config.yaml"), ("route_config_snapshot.yaml", RI.HERE / "config.yaml"), ("geometry_config_snapshot.yaml", PT.HERE / "config.yaml")):
        (out / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    ctx = RI.build_context(rcfg, gcfg, v02, img)
    assert ctx["ft"]["accepted"], "floor transform refused by its acceptance rule: no metric map"
    fr, s, pad, m = ctx["fr"], ctx["fr"].s, ctx["fr"].pad, ctx["ft"]["footprint_margin_cm"]
    L, W = map(float, cfg["cost_map"]["robot_cm"][:2])
    prim, dcfg, near_px, contain = cfg["cost_map"]["primary_heading_deg"], cfg["decision"], v02["planning"]["relations"]["near_px"], v02["planning"]["relations"]["contain_ratio"]
    start_img = ctx["starts"][0]
    start_cm = fr.img_to_cm([start_img])[0]
    start_bev = fr.cm_to_bev([start_cm])[0].tolist()
    navs = {h: FN.RectNavMap(ctx["spec_bev"], fr.bev_size, ctx["pcfg"], FN.footprint_dims_cm(L, W, m, h), s) for h in cfg["cost_map"]["headings_deg"]}
    body_fits = bool(navs[prim].footprint_fits_full[int(round(start_bev[1])), int(round(start_bev[0]))])

    # ================================================================ Phase 1: semantic context map
    cands, e1p, qc = stored_candidates(cfg, rcfg, ctx)
    inp = context_inputs(cfg, rcfg, ctx, cands, start_cm, body_fits)
    cmap = CM.build(inp, cfg["context"], near_px, contain)
    cmap.update({"experiment_id": cfg["experiment_id"], "command": cfg["command"], "model_calls_this_run": 0,
                 "sources": {"qwen_L2_reply": f"{inp['qwen_meta']['run']} (sha256 {inp['qwen_meta']['raw_response_sha256'][:16]}..., wording '{inp['qwen_meta']['instruction']}')",
                             "candidates": str(res(cfg["stored"]["retention_run"])), "crop_views": str(res(cfg["stored"]["crop_comparison_run"])),
                             "objects_map_bay": "task_route_integration.build_context (first_vlm_navigation_path, semantic_evidence_verification runs)"},
                 "thresholds": {"near_px": near_px, "contain_ratio": contain, "source": "v0.2 config/planning.yaml relations", "floor_adjacent_cm": cfg["context"]["floor_adjacent_cm"]},
                 "accuracy": "NOT_COMPUTED (no human ground truth)"})
    (out / "semantic_context_map.json").write_text(json.dumps(cmap, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    fig_semantic(img, cmap, inp, out / "semantic_context_map.png")
    print("phase 1:", cmap["summary"])

    # ================================================================ Phase 2: navigation cost map
    km = KM.build(navs, start_bev, ctx["pcfg"], prim)
    nav0 = navs[prim]
    xs = (np.arange(nav0.cols) * nav0.cell_px + nav0.cell_px // 2) / s - pad
    ys = (np.arange(nav0.rows) * nav0.cell_px + nav0.cell_px // 2) / s - pad
    ws = (ys[:, None] >= 0) & (ys[:, None] <= ctx["size_cm"][1]) & (xs[None, :] >= 0) & (xs[None, :] <= ctx["size_cm"][0])
    classes, counts = {}, {}
    for h, nav in navs.items():
        classes[str(h)] = KM.classify(nav, km[h]["safety"], ws)
        if "via_rotation_from_primary" in km[h]:
            classes[f"{h}_via_rotation"] = KM.classify(nav, km[h]["via_rotation_from_primary"]["safety"], ws)
    for k, cl in classes.items():
        counts[k] = {name: int((cl == i).sum()) for i, name in enumerate(KM.CLASSES)}
    classes["_counts"] = counts
    cell_cm2 = (nav0.cell_px / s) ** 2
    slack_px, wgt = dcfg["approach"]["slack_cm"] * s, dcfg["approach"]["distance_weight"]
    cand_nav = {}
    for c in cands:
        a = KM.approach(nav0, km[prim]["safety"], fr.cm_to_bev(c["floor_cm"]).tolist(), slack_px, wgt)
        if a["status"] != "GOAL_SELECTED":
            cand_nav[c["id"]] = {"status": a["status"]}
            continue
        xy = [a["xy_bev_px"][0] / s - pad, a["xy_bev_px"][1] / s - pad]
        r_, c_ = a["cell"]
        cand_nav[c["id"]] = {"status": "GOAL_SELECTED", "approach_cm": [round(v, 1) for v in xy], "cell": a["cell"], "path_cost_cm": round(a["path_cost_px"] / s, 2),
                             "geodesic_cm": r1(km[prim]["distance"][r_, c_] / s), "nearest_reachable_cm": round(a["nearest_reachable_px"] / s, 1),
                             "gap_cm": round(FN.body_gap_cm(c["floor_cm"], [round(v, 1) for v in xy], FN.footprint_dims_cm(L, W, 0, prim), ctx["size_cm"], pad, 8), 1),
                             "v02_goal_score_cm": round(a["v02_goal_score_px"] / s, 2), "rotation_clear_at_approach": bool(km[prim]["rotation_clear"][r_, c_])}
    ms_check = KM.multi_source_cost(nav0, {tuple(km[prim]["start_cell"]): 0.0}, ctx["pcfg"]["planner"], ctx["pcfg"]["safety_weight"])
    cons = {str(h): (KM.consistency(nav, km[h]["safety"], start_bev, ctx["pcfg"], cfg["cost_map"]["consistency_samples"], cfg["cost_map"]["consistency_seed"], cfg["cost_map"]["abs_tol"])
                     if km[h]["start_valid"] else {"skipped": "start cell invalid for this heading (body does not fit at the start): the whole map is unreachable"}) for h, nav in navs.items()}
    coll = {str(h): KM.collision_check(nav, nav.valid & np.isfinite(km[h]["safety"]), FN.body_free, ctx["free_hi"], nav.extent_cm, s, pad, gcfg["bev"]["check_px_per_cm"]) for h, nav in navs.items()}
    for h, nav in navs.items():
        if "via_rotation_from_primary" in km[h]:
            coll[f"{h}_via_rotation"] = KM.collision_check(nav, nav.valid & np.isfinite(km[h]["via_rotation_from_primary"]["safety"]), FN.body_free, ctx["free_hi"], nav.extent_cm, s, pad, gcfg["bev"]["check_px_per_cm"])
    rot_start = {str(h): nav.rotation_clear(start_bev) for h, nav in navs.items()}
    print("phase 2:", counts, {k: (v.get("all_equal"), v.get("max_abs_diff_px")) for k, v in cons.items()}, {k: v["n_violations"] for k, v in coll.items()})

    # ================================================================ Phase 3: decisions + plans
    vis = dcfg["visual_level"]
    plannable = [c for c in cands if vis[c["state"]] >= dcfg["plannable_min_visual_level"]]
    plans = {c["id"]: full_plan(ctx, gcfg, c, L, W, m, start_img) for c in plannable}

    def dec_inputs():
        return [{"id": c["id"], "rank": c["rank"], "visual": c["state"], "semantic": CM.source_level(cmap, c["id"])[0], "source_types": c["source_types"], "score": c["score"],
                 "nav": {**{k: cand_nav[c["id"]].get(k) for k in ("gap_cm", "v02_goal_score_cm", "path_cost_cm")},
                         "path_A_B_feasible": plans[c["id"]]["path_A_B_feasible"] if c["id"] in plans else None}} for c in cands]
    d_first = DM.target_conditions(dec_inputs(), dcfg, dcfg["semantic_source_level"]["graded"])["D_navigation_only"]["selected"]
    if d_first and d_first not in plans:
        plans[d_first] = full_plan(ctx, gcfg, next(c for c in cands if c["id"] == d_first), L, W, m, start_img)
    din = dec_inputs()
    assert all(set(d) == {"id", "rank", "visual", "semantic", "source_types", "score", "nav"} for d in din)     # no label field reaches the decision
    conds = {sc: DM.target_conditions(din, dcfg, dcfg["semantic_source_level"][sc]) for sc in ("graded", "binary")}
    C = conds["graded"]["C_visual_semantic_navigation"]
    order = [x["id"] for x in C["ranking"] if x["id"] in C["tied_set"]]
    primary_id = C["selected"] or (order[0] if C["status"] == "AMBIGUOUS" else None)

    # destination evaluation from the pick-up approach of the primary candidate (same start for every condition)
    dests, dconds = [], None
    if primary_id and plans[primary_id]["summary"]["paths"].get("A"):
        pp = plans[primary_id]
        hA = pp["var"]["selected"]["A_heading_deg"]
        gA = pp["var"]["legs_A"][hA]["goal"]["xy_bev_px"]
        img_r = CM.Raster(inp["image_size"])
        chosen = inp["qwen_structured"]["task_analysis"]["destination_area_id"]
        place_rec = pp["B_place"]
        for rank, a in enumerate(inp["qwen_structured"]["functional_areas"], 1):
            frac = img_r.frac_inside(inp["bay"]["polygon_px"], CM.box_poly(inp["area_boxes_px"][a["id"]]))
            nav_area = leg_record(ctx, leg_to_region(ctx, gcfg, pp["navs"], hA, gA, [inp["area_floor_cm"][a["id"]]], a["id"], L, W))
            grounded = frac >= contain
            dests.append({"id": a["id"], "rank": rank, "function_name": a["function_name"], "vlm_chosen": a["id"] == chosen, "grounded": grounded,
                          "bay_inside_area_box_fraction": round(frac, 3), "placement_target": "placement_floor (bay free floor)" if grounded else "area box (floor projection)",
                          "nav": place_rec if grounded else nav_area, "nav_area": nav_area,
                          "in_semantic_map": a["id"] not in cfg["context"]["excluded_vlm_areas"]})
        dconds = DM.destination_conditions(dests)
    print("phase 3:", {sc: {k: (v["status"], v.get("selected"), v.get("tied_set")) for k, v in cc.items()} for sc, cc in conds.items()},
          None if dconds is None else {k: (v["status"], v.get("selected"), v.get("tied_set")) for k, v in dconds.items()})

    # ---- integrated plan (primary = condition C, graded)
    dC = (dconds or {}).get("C_visual_semantic_navigation") or {"status": "NOT_EVALUATED", "selected": None}
    plan_ids = [C["selected"]] if C["status"] == "SELECTED" else order if C["status"] == "AMBIGUOUS" else []
    cby = {c["id"]: c for c in cands}
    integrated = []
    for cid in plan_ids:
        p = plans[cid]
        sA = p["summary"]
        states = [x for x in sA["states"] if x["state"] in ("APPROACH_PLANNED", "VERIFY_REQUIRED", "PICKUP_ASSUMED")]
        vr = next((x for x in states if x["state"] == "VERIFY_REQUIRED"), None)
        if vr:
            c = cby[cid]
            vr["why"] = [f"visual state {c['state']} (stored Qwen view A: {(c['stored_qwen'] or {}).get('status')})",
                         f"Qwen crop views A/B/C: {None if not c['crop_views'] else '/'.join(v['status'] for v in c['crop_views'].values())} (disagreement kept)",
                         "stored Qwen L2 reply says the red cube is not visible in the full image", "label is AI_PRELIMINARY_REVIEW only; nothing is CONFIRMED"]
        B = p["B_place"]
        if B and B["path_B_feasible"] and dC.get("selected"):
            states += [{"state": "TRANSPORT_PLANNED", "path": "B", "from_cm": sA["paths"]["A"]["goal_cm"], "to_cm": B["goal_cm"], "executed": False,
                        "destination": f"{dC['selected']} -> placement floor of the bay", "condition": "planned assuming pick-up"},
                       {"state": "PLACEMENT_ASSUMED", "at_cm": B["goal_cm"], "executed": False, "placement_state": "PLACEMENT_UNVERIFIED",
                        "gripper_must_bridge_cm": B["gap_cm"], "undecided": ["gripper reach / mount / height", "boundary crossing into the bay", "is the floor really empty"]}]
        assert all(x["state"] in RI.PLAN_STATES for x in states)
        integrated.append({"candidate": cid, "role": "SELECTED_CONDITIONAL" if C["status"] == "SELECTED" else f"CONDITIONAL_ON_CANDIDATE (verify in order {order.index(cid) + 1})",
                           "states": states, "path_A": sA["paths"].get("A"), "checks_A": sA["checks"].get("A"),
                           "path_B_to_placement": {k: v for k, v in (B or {}).items()}, "A_B_connected": bool(B and B.get("waypoints_cm") and sA["paths"].get("A") and B["waypoints_cm"][0] == sA["paths"]["A"]["goal_cm"]),
                           "earlier_definition_path_B_to_bay_polygon": sA["paths"].get("B") and {k: sA["paths"]["B"][k] for k in ("goal_cm", "length_cm")},
                           "rotation": {"at_start": rot_start, "at_pickup_approach": sA["paths"].get("A") and navs[prim].rotation_clear(fr.cm_to_bev([sA["paths"]["A"]["goal_cm"]])[0])}})

    # ---- ablation table
    label_box = yaml.safe_load(res(cfg["stored"]["funnel_config"]).read_text(encoding="utf-8"))["labels"][cfg["photo_id"]]
    label_id = next((c["id"] for c in cands if iou(c["bbox_xyxy"], label_box) >= 0.5), None)
    rows = []
    for sc, cc in conds.items():
        for name, d in cc.items():
            if sc == "binary" and name[0] not in "BC":          # A and D do not use the semantic scale: listed once
                continue
            ids = [d["selected"]] if d["selected"] else d["tied_set"] if d["status"] == "AMBIGUOUS" else []
            br = [plan_brief(plans.get(i)) for i in ids]
            rows.append({"condition": name, "semantic_scale": sc if name[0] in "BC" else "-", "item": "target", "status": d["status"], "selected": d["selected"],
                         "tied_set": " ".join(d["tied_set"]) if d["status"] != "SELECTED" else "", "ranking_top5": " ".join(x["id"] for x in d["ranking"][:5]),
                         "selected_or_tied_visual": " ".join(cby[i]["state"] for i in ids), "selected_or_tied_semantic": " ".join(CM.source_level(cmap, i)[0] for i in ids),
                         "nav_feasible_A_B": " ".join(feas_text(b) for b in br),
                         "path_A_cm": " ".join(str(b["path_A_length_cm"]) for b in br if b), "path_B_cm": " ".join(str(b["path_B_place_length_cm"]) for b in br if b),
                         "pickup_gap_cm": " ".join(str(b["pickup_gap_cm"]) for b in br if b), "placement_gap_cm": " ".join(str(b["placement_gap_cm"]) for b in br if b),
                         "reason": d["reasons"].get(d["selected"]) if d["selected"] else ("tied on all decisive factors" if d["status"] == "AMBIGUOUS" else d["status"]),
                         "coincides_with_AI_preliminary_label": ("n/a" if not ids else str(label_id in ids) + (" (one of the tied set)" if len(ids) > 1 else ""))})
    if dconds:
        for name, d in dconds.items():
            nv = None
            if d.get("selected"):
                dd = next(x for x in dests if x["id"] == d["selected"])
                nv = dd["nav_area"] if name.startswith("D") else dd["nav"]
            rows.append({"condition": name, "semantic_scale": "-", "item": "destination", "status": d["status"], "selected": d.get("selected"),
                         "tied_set": " ".join(d.get("tied_set") or []) if d["status"] != "SELECTED" else "", "ranking_top5": " ".join(x["id"] for x in d.get("ranking", [])[:5]),
                         "selected_or_tied_visual": "", "selected_or_tied_semantic": "" if not d.get("selected") else next(f"{x['function_name']}; vlm_chosen={x['vlm_chosen']}; grounded={x['grounded']}" for x in dests if x["id"] == d["selected"]),
                         "nav_feasible_A_B": "" if not nv else str(nv["path_B_feasible"]), "path_A_cm": "", "path_B_cm": "" if not nv else str(nv.get("length_cm")),
                         "pickup_gap_cm": "", "placement_gap_cm": "" if not nv else str(nv.get("gap_cm")),
                         "reason": d.get("reason") or (d["reasons"].get(d["selected"]) if d.get("selected") else ("tied on all decisive factors" if d["status"] == "AMBIGUOUS" else d["status"])),
                         "coincides_with_AI_preliminary_label": "n/a (no destination label)"})
    with open(out / "ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- failure attribution (independent stages)
    pA = plans.get(primary_id) if primary_id else None
    attribution = {
        "1_object_search": {"status": "OK", "detail": f"{len(cands)} target-class candidates retained (stored); {sum(c['state'] == 'PROBABLE' for c in cands)} PROBABLE, "
                            f"{sum(c['state'] == 'VLM_ACCEPTED' for c in cands)} VLM_ACCEPTED"},
        "2_object_identification": {"status": "UNVERIFIED", "detail": f"C (graded): {C['status']} {C['selected'] or C['tied_set']}; no candidate is VLM_ACCEPTED; every target is VERIFY_REQUIRED"},
        "3_semantic_relation": {"status": "UNVERIFIED", "detail": "source relation 'produced_by conveyor' is INSTRUCTION_GIVEN; only the 2D image relation on/near the conveyor box is SUPPORTED"},
        "4_destination_selection": {"status": "CONDITIONAL" if dC.get("selected") else dC["status"], "detail": f"C: {dC['status']} {dC.get('selected')}; stored Qwen reply (shorter wording) grounded to the Python bay"},
        "5_geometric_access": {"status": "UNVERIFIED", "detail": None if not pA else f"pick-up gap {plan_brief(pA)['pickup_gap_cm']} cm, placement gap {plan_brief(pA)['placement_gap_cm']} cm; gripper reach unknown; target and bay lie outside the drivable zone"},
        "6_path_planning": {"status": ("OK" if pA and pA["path_A_B_feasible"] else "FAILED") + " (inside the PROVISIONAL model)", "detail": None if not pA else plan_brief(pA)}}

    # ---- outputs
    def gl(a):
        v = np.where(np.isfinite(a), np.round(a / s, 1), np.nan)
        return [[None if np.isnan(x) else float(x) for x in row] for row in v]
    nav_json = {"experiment_id": cfg["experiment_id"], "frame": "floor_cm (PROVISIONAL_UNVALIDATED mat-seam homography); grid = BEV raster 4 px/cm, cell 8 px = 2 cm; "
                "cell (r, c) centre = ((c + 0.5) * 2 - pad, (r + 0.5) * 2 - pad) cm, pad = %g cm" % pad,
                "not_a_single_px_per_cm_on_the_photo": "image px -> floor cm via the homography; no uniform scale is applied to the photo",
                "rows": nav0.rows, "cols": nav0.cols, "cell_cm": nav0.cell_px / s, "cell_area_cm2": cell_cm2, "classes": list(KM.CLASSES), "start_cm": [round(float(v), 1) for v in start_cm],
                "start_status": "ESTIMATED (stored robot box centre; heading UNKNOWN)", "footprint_cm": {str(h): list(nav.extent_cm) for h, nav in navs.items()},
                "footprint_rule": f"{L}x{W} cm user size + {m} cm/side floor-transform margin; whole rectangle must lie on free floor (erosion); fixed heading per map",
                "cost": {"safety": "v0.2 cost_field SAFETY_COST_ASTAR (path length + risk term inside the 10 cm soft margin), cm-equivalent; equals v0.2 A* cost",
                         "distance": "v0.2 cost_field DISTANCE_ONLY edges: 8-connected grid geodesic length (cm); the simplified any-angle path is shorter or equal"},
                "dynamics": "NOT modelled (cost is geometric; mecanum slip/acceleration not represented)",
                "cell_counts": counts, "area_cm2": {k: {n: v * cell_cm2 for n, v in cnt.items()} for k, cnt in counts.items()},
                "rotation": {"test": cfg["cost_map"]["heading_change"], "at_start": rot_start,
                             "n_rotation_clear_reachable_cells_primary": int((km[prim]["rotation_clear"] & np.isfinite(km[prim]["safety"])).sum())},
                "start_valid_by_heading": {str(h): km[h]["start_valid"] for h in navs},
                "candidate_approach_primary_heading": cand_nav,
                "destination_approach_from_primary_pickup": [{"id": d["id"], "nav_to_area_box": {k: v for k, v in d["nav_area"].items() if k not in ("layers",)},
                                                              "nav_to_placement_target": {k: v for k, v in d["nav"].items() if k not in ("layers",)}} for d in dests],
                "not_confused_with": "placement_reach_map gaps are body-edge-to-point distances; these are path costs from the start",
                "grids": {}}
    for h in navs:
        nav_json["grids"][str(h)] = {"class": classes[str(h)].tolist(), "cost_safety_cm": gl(km[h]["safety"]), "geodesic_cm": gl(km[h]["distance"]),
                                     "rotation_clear": km[h]["rotation_clear"].astype(int).tolist()}
        if "via_rotation_from_primary" in km[h]:
            v = km[h]["via_rotation_from_primary"]
            nav_json["grids"][f"{h}_via_rotation"] = {"class": classes[f"{h}_via_rotation"].tolist(), "cost_safety_cm": gl(v["safety"]), "geodesic_cm": gl(v["distance"]),
                                                      "n_switch_cells": v["n_switch_cells"]}
    (out / "navigation_cost_map.json").write_text(json.dumps(nav_json, separators=(",", ":"), ensure_ascii=False, default=str), encoding="utf-8")
    np.savez_compressed(out / "navigation_cost_map_arrays.npz", **{f"safety_px_{h}": km[h]["safety"] for h in navs}, **{f"distance_px_{h}": km[h]["distance"] for h in navs},
                        **{f"class_{k}": v for k, v in classes.items() if k != "_counts"})
    fig_costmap(ctx, km, classes, navs, cand_nav, plans, (pA or {}).get("B_place"), out / "navigation_cost_map.png", prim, start_cm)

    strip = lambda p: None if p is None else {"summary": p["summary"], "B_place": p["B_place"], "path_A_B_feasible": p["path_A_B_feasible"]}   # noqa: E731
    decisions = {"experiment_id": cfg["experiment_id"], "policy": dcfg, "label_status": "AI_PRELIMINARY_REVIEW (reported only; never an input of the decision)",
                 "ai_preliminary_label_candidate": label_id, "candidates": [{**c, "semantic_source_relation": CM.source_level(cmap, c["id"]), "nav": cand_nav[c["id"]],
                                                                             "plan": plan_brief(plans.get(c["id"]))} for c in cands],
                 "target_conditions": conds, "destinations": dests, "destination_conditions": dconds, "primary_candidate_for_destination_evaluation": primary_id,
                 "photo_decision_retention_policy": e1p["photo_decision_from_stored_answers"], "generalisation": cfg["claims"]["generalisation"]}
    (out / "candidate_decisions.json").write_text(json.dumps(decisions, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    task_plan = {"experiment_id": cfg["experiment_id"], "created": datetime.now().isoformat(timespec="seconds"), "command": cfg["command"], "claims": cfg["claims"],
                 "decision_target_C_graded": {k: C[k] for k in ("status", "selected", "tied_set")}, "verification_order": order if C["status"] == "AMBIGUOUS" else None,
                 "decision_destination_C": {k: dC.get(k) for k in ("status", "selected", "tied_set")}, "plans": integrated, "failure_attribution": attribution,
                 "earlier_route_run": str(res(cfg["stored"]["route_run"])), "placement_reach_reference": str(res(cfg["stored"]["placement_reach_run"])),
                 "plans_all_evaluated": {k: strip(v) for k, v in plans.items()}, "executed": False, "no_hardware_command": True}
    (out / "integrated_task_plan.json").write_text(json.dumps(task_plan, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    fig_plan(ctx, plans, cby, plan_ids, conds["graded"]["D_navigation_only"]["selected"], plans.get(conds["graded"]["D_navigation_only"]["selected"]), rows,
             f"{dC['status']} {dC.get('selected')}", out / "integrated_task_plan.png")

    # ---- validation
    route = PT.jload(res(cfg["stored"]["route_run"]) / "task_plan.json")
    old_r1 = route["condition_B"]["conditional_plans"]["1"]           # same retained candidate box (rank 1) as cand_r1 here
    old_A, old_box = old_r1["paths"]["A"], old_r1["target"]["bbox_image_px"]
    rep = {}
    for cid, p in plans.items():
        A = p["summary"]["paths"].get("A")
        leg0 = p["var"]["legs_A"].get(prim)
        g0 = leg0["goal"] if leg0 else None
        cn = cand_nav[cid]
        rep[cid] = {"costmap_approach_equals_approach_goal": bool(g0 and g0["status"] == "GOAL_SELECTED" and [round(v, 6) for v in g0["xy_bev_px"]] == [round(v, 6) for v in nav0.cell_center(*cn["cell"])]),
                    "costmap_cost_at_goal_vs_astar_cost": None if not (leg0 and leg0["path"]) else {"costmap_px": round(float(km[prim]["safety"][tuple(cn["cell"])]), 3), "astar_cost_px": leg0["path"]["astar_cost"]},
                    "path_A": None if not A else {"sweep_ok": p["summary"]["checks"]["A"]["whole_body_sweep"]["ok"], "v02_valid": p["summary"]["checks"]["A"]["v02_validator_inscribed_disc"]["valid"],
                                                  "layers_clear": p["summary"]["checks"]["A"]["swept_layer_overlap"]["clear"]},
                    "path_B_place": None if not p["B_place"] or not p["B_place"]["path_B_feasible"] else {"sweep_ok": p["B_place"]["whole_body_sweep_ok"], "v02_valid": p["B_place"]["v02_valid"],
                                                                                                           "layers_clear": None if p["B_place"]["layers"] is None else p["B_place"]["layers"]["clear"]}}
    r1p = plans.get("cand_r1")
    validation = {
        "inputs": {"image_sha256": rcfg["image_sha256"], "candidates_equal_retention_record": [c["bbox_xyxy"] for c in cands] == [c["bbox_xyxy"] for c in e1p["candidates"]],
                   "n_candidates": len(cands), "crop_views_matched": sum(c["crop_views"] is not None for c in cands), "model_calls": 0},
        "experiment_1": {"instruction_and_vlm_relations_never_SUPPORTED": all(r["status"] != "SUPPORTED" for r in cmap["relations"] if r["provenance"] != "GEOMETRY_DERIVED"),
                         "every_relation_has_evidence": all(r["evidence"] for r in cmap["relations"]),
                         "relations_reference_existing_nodes": all(r["subject"] in {n["id"] for n in cmap["nodes"]} and r["object"] in {n["id"] for n in cmap["nodes"]} for r in cmap["relations"]),
                         "candidates_without_source_relation": sum(CM.source_level(cmap, c["id"])[0] == "none" for c in cands)},
        "experiment_2": {"astar_consistency": {k: {kk: v.get(kk) for kk in ("n_reachable_tested", "max_abs_diff_px", "all_equal", "n_unreachable_tested", "unreachable_all_no_path", "skipped")} for k, v in cons.items()},
                         "multi_source_with_start_seed_equals_v02_cost_field": bool(np.array_equal(np.isfinite(ms_check), np.isfinite(km[prim]["safety"])) and
                                                                                    np.allclose(ms_check[np.isfinite(ms_check)], km[prim]["safety"][np.isfinite(ms_check)], atol=1e-9)),
                         "reachable_cells_collision_free": {k: {"n_cells": v["n_cells_tested"], "violations": v["n_violations"]} for k, v in coll.items()},
                         "plans_vs_costmap": rep,
                         "reproduces_stored_route_path_A_cand_r1": None if not r1p or "A" not in r1p["summary"]["paths"] else {
                             "stored_goal_cm": old_A["goal_cm"], "new_goal_cm": r1p["summary"]["paths"]["A"]["goal_cm"], "stored_length_cm": old_A["length_cm"],
                             "new_length_cm": r1p["summary"]["paths"]["A"]["length_cm"], "equal": old_A["goal_cm"] == r1p["summary"]["paths"]["A"]["goal_cm"] and old_A["length_cm"] == r1p["summary"]["paths"]["A"]["length_cm"],
                             "stored_target_box": old_box},
                         "diagonal_corner_cutting": "v0.2 _neighbors forbids it; tested on a synthetic map in test_semnav_offline.py"},
        "experiment_3": {"no_CONFIRMED_state": all(c["state"] != "CONFIRMED" for c in cands),
                         "A_to_C_never_select_below_plannable": all(cc[k]["selected"] is None or vis[cby[cc[k]["selected"]]["state"]] >= dcfg["plannable_min_visual_level"]
                                                                    for cc in conds.values() for k in ("A_visual", "B_visual_semantic", "C_visual_semantic_navigation")),
                         "C_selected_or_tied_have_top_semantic_level": all(
                             all(dcfg["semantic_source_level"][sc][CM.source_level(cmap, i)[0]] == max(dcfg["semantic_source_level"][sc][CM.source_level(cmap, c["id"])[0]] for c in plannable)
                                 for i in (cc["C_visual_semantic_navigation"]["tied_set"] or [])) for sc, cc in conds.items()),
                         "decision_inputs_contain_no_label": True,
                         "A_B_connected": {p["candidate"]: p["A_B_connected"] for p in integrated}},
        "geometry_status": "inside the PROVISIONAL floor transform + DRAFT map only; real-size safety NOT claimed",
        "v02_code_before": fp_before, "v02_code_after": code_fingerprint()["sha256"]}
    validation["v02_code_unchanged"] = validation["v02_code_before"] == validation["v02_code_after"]
    (out / "validation.json").write_text(json.dumps(validation, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "manifest.json").write_text(json.dumps({"files_sha256": files, "code_sha256": {p.name: sha256_file(p) for p in sorted(HERE.glob("*.py"))}}, indent=1), encoding="utf-8")
    print("wrote", out)
    print(json.dumps({"ablation": rows, "attribution": attribution, "validation": {k: validation[k] for k in ("inputs", "experiment_1", "experiment_3", "v02_code_unchanged")},
                      "exp2": {k: validation["experiment_2"][k] for k in ("astar_consistency", "multi_source_with_start_seed_equals_v02_cost_field", "reachable_cells_collision_free", "reproduces_stored_route_path_A_cand_r1")}},
                     indent=1, ensure_ascii=False, default=str)[:9000])
    return out


if __name__ == "__main__":
    main()
