"""Experiment 2: footprint-aware navigation cost map on the PROVISIONAL floor frame.

Reused unchanged: v0.2 cost_field (Dijkstra with the A* edge costs: 8-connected, no diagonal corner cutting, SAFETY_COST_ASTAR risk term),
v0.2 plan (A*) for the consistency test, v0.2 _distance_to_polygon / select_goal scoring for approach poses, task_plan_prototype RectNavMap
(whole-rectangle erosion at a fixed heading) and body_free (independent cm-polygon raster). New: cell classes, the heading-change layer
(multi-source Dijkstra seeded only at rotation-clear cells, same edge function), the cost-map approach lookup and the checks.
"""
import heapq
import math

import numpy as np

from semantic_task_nav.navigation.goal_selector import _distance_to_polygon
from semantic_task_nav.navigation.planner import _edge_cost, _neighbors, astar, cost_field, plan

CLASSES = ("REACHABLE", "UNREACHABLE_DISCONNECTED", "UNREACHABLE_FOOTPRINT", "STATIC_OBSTACLE", "NO_GO", "UNKNOWN_UNVERIFIED", "OUTSIDE_WORKSPACE")


def classify(nav, cost, ws_cells):
    """Per-cell class (index into CLASSES), judged at the cell centre. Priority: outside workspace > static > no-go > not in a drivable zone > footprint > connectivity."""
    L = nav.layers
    out = np.full(nav.valid.shape, CLASSES.index("UNREACHABLE_DISCONNECTED"), np.int8)
    out[nav.valid & np.isfinite(cost)] = CLASSES.index("REACHABLE")
    out[nav.free & ~nav.valid] = CLASSES.index("UNREACHABLE_FOOTPRINT")
    out[~L["traversable"]] = CLASSES.index("UNKNOWN_UNVERIFIED")
    out[L["no_go"]] = CLASSES.index("NO_GO")
    out[L["static_obstacle"]] = CLASSES.index("STATIC_OBSTACLE")
    out[~ws_cells] = CLASSES.index("OUTSIDE_WORKSPACE")
    return out


def multi_source_cost(nav, seeds, planner, weight):
    """v0.2 cost_field with several seeded start cells (cell -> initial cost). Same neighbour and edge functions as v0.2 A*."""
    valid_l, risk_l = nav.valid.tolist(), np.nan_to_num(nav.risk).tolist()
    best, heap = {}, []
    for cell, c0 in seeds.items():
        if valid_l[cell[0]][cell[1]] and c0 < best.get(cell, math.inf):
            best[cell] = c0
            heap.append((c0, cell))
    heapq.heapify(heap)
    while heap:
        d, cur = heapq.heappop(heap)
        if d > best.get(cur, math.inf):
            continue
        for nxt, dc in _neighbors(cur, valid_l, nav.rows, nav.cols):
            nd = d + _edge_cost(dc, nav.cell_px, risk_l, nxt, planner, weight)
            if nd < best.get(nxt, math.inf) - 1e-12:
                best[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    dist = np.full(nav.valid.shape, np.inf)
    for cell, d in best.items():
        dist[cell] = d
    return dist


def rotation_clear_cells(nav):
    """Cells where the circumscribed disc of the footprint is on free pixels (the conservative in-place rotation test of RectNavMap.rotation_clear)."""
    return nav.valid & (nav.clearance >= nav.circumscribed_radius_px)


def build(navs, start_bev, pcfg, primary):
    """navs: heading -> RectNavMap. Cost (px of BEV) per heading from the start, and the heading-change layer for the other headings."""
    out = {}
    for h, nav in navs.items():
        s = nav.cell_of(*start_bev)
        ok = nav.in_grid(*s) and bool(nav.valid[s])
        safety = cost_field(nav, s, pcfg["planner"], pcfg["safety_weight"]) if ok else np.full(nav.valid.shape, np.inf)
        dist = cost_field(nav, s, "DISTANCE_ONLY_ASTAR", 0.0) if ok else np.full(nav.valid.shape, np.inf)
        out[h] = {"start_cell": list(s), "start_valid": ok, "safety": safety, "distance": dist, "rotation_clear": rotation_clear_cells(nav)}
    p = out[primary]
    switch = p["rotation_clear"] & np.isfinite(p["safety"])
    for h, nav in navs.items():
        if h == primary:
            continue
        seeds = {(int(r), int(c)): float(p["safety"][r, c]) for r, c in zip(*np.nonzero(switch & nav.valid))}
        seeds_d = {(int(r), int(c)): float(p["distance"][r, c]) for r, c in zip(*np.nonzero(switch & nav.valid))}
        out[h]["via_rotation_from_primary"] = {"n_switch_cells": len(seeds), "safety": multi_source_cost(nav, seeds, pcfg["planner"], pcfg["safety_weight"]),
                                               "distance": multi_source_cost(nav, seeds_d, "DISTANCE_ONLY_ASTAR", 0.0)}
    return out


def approach(nav, cost, polys_bev, slack_px, weight):
    """Approach pose from a cost map, with the exact rule of task_plan_prototype.approach_goal (v0.2 select_goal run twice):
    nearest reachable distance d_near to the polygon(s), then the lowest score = cost + weight * distance within d_near + slack."""
    dist = _distance_to_polygon(nav, polys_bev)
    reach = nav.valid & np.isfinite(cost)
    if not reach.any():
        return {"status": "CANDIDATES_UNREACHABLE"}
    d_near = float(dist[reach].min())
    band = reach & (dist <= d_near + slack_px)
    score = np.where(band, cost + weight * dist, np.inf)
    r, c = np.unravel_index(np.argsort(score, axis=None)[0], score.shape)
    return {"status": "GOAL_SELECTED", "cell": [int(r), int(c)], "xy_bev_px": list(nav.cell_center(r, c)), "path_cost_px": float(cost[r, c]),
            "distance_to_target_px": float(dist[r, c]), "nearest_reachable_px": d_near, "v02_goal_score_px": float(score[r, c])}


def consistency(nav, costs, start_bev, pcfg, n, seed, tol):
    """Re-plan with v0.2 A* to n random reachable cells and to unreachable valid cells; the A* cost must equal the cost-map value."""
    rng = np.random.default_rng(seed)
    reach = np.argwhere(nav.valid & np.isfinite(costs))
    unreach = np.argwhere(nav.valid & ~np.isfinite(costs))
    picks = reach[rng.choice(len(reach), size=min(n, len(reach)), replace=False)] if len(reach) else []
    rows, worst = [], 0.0
    s_cell = nav.cell_of(*start_bev)
    for r, c in picks:
        res = astar(nav, s_cell, (int(r), int(c)), pcfg["planner"], pcfg["safety_weight"])     # unrounded v0.2 A* cost
        a = res["cost"]
        d = abs(a - costs[r, c]) if a is not None else math.inf
        worst = max(worst, d)
        rows.append({"cell": [int(r), int(c)], "costmap_px": float(costs[r, c]), "astar_cost_px": a, "abs_diff": d, "status": res["status"]})
    un = []
    for r, c in (unreach[rng.choice(len(unreach), size=min(10, len(unreach)), replace=False)] if len(unreach) else []):
        res = plan(nav, list(start_bev), list(nav.cell_center(r, c)), pcfg, "consistency_unreachable")
        un.append({"cell": [int(r), int(c)], "status": res["status"]})
    return {"n_reachable_tested": len(rows), "max_abs_diff_px": worst, "tolerance_px": tol, "all_equal": all(x["abs_diff"] <= tol for x in rows),
            "n_unreachable_tested": len(un), "unreachable_all_no_path": all(x["status"] == "NO_PATH" for x in un),
            "compared": "v0.2 astar() cost (unrounded) vs cost-map value; unreachable cells re-planned with v0.2 plan()",
            "samples": rows, "unreachable_samples": un}


def collision_check(nav, reach, body_free, free_hi, extent_cm, s, pad, check_s):
    """Independent test of every REACHABLE cell: the body rectangle at the cell centre must lie on free pixels of the cm-polygon raster."""
    bad = []
    for r, c in np.argwhere(reach):
        x, y = nav.cell_center(r, c)
        ok, n = body_free(free_hi, (x / s - pad, y / s - pad), extent_cm, pad, check_s)
        if not ok:
            bad.append({"cell": [int(r), int(c)], "non_free_px": n})
    return {"n_cells_tested": int(reach.sum()), "n_violations": len(bad), "violations": bad[:20],
            "check": "FN.body_free on the independent 4 px/cm raster of (traversable - static - no-go) from the cm polygons"}
