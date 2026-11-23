"""Grid planning on NavMap.valid (robot-centre configuration space).

A* core (8-connected, no diagonal corner cutting, optional clearance-based safety cost) is ported
from the read-only reference
  /home/binci/vlm_ws/wanna_find_detailed_experimental_x,Y/src/vlm_reposition/planning/grid_astar.py
  (astar_grid / neighbors, STEP 15 SAFETY_COST_ASTAR)
adapted to numpy validity/risk grids in pixel units. Additions: a Dijkstra cost field (goal
scoring) and line-of-sight waypoint simplification (any-angle post-processing; Theta* not used).
"""
import heapq
import itertools
import math
from time import perf_counter

import numpy as np

NEIGHBORS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
SQRT2 = math.sqrt(2.0)


class PlannerError(ValueError):
    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def _neighbors(cell, valid_l, rows, cols):
    r, c = cell
    for dr, dc in NEIGHBORS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols) or not valid_l[nr][nc]:
            continue
        if dr and dc and (not valid_l[r + dr][c] or not valid_l[r][c + dc]):
            continue
        yield (nr, nc), (SQRT2 if dr and dc else 1.0)


def _edge_cost(dist_cells, cell_px, risk_l, nxt, planner, weight):
    d = dist_cells * cell_px
    if planner == "SAFETY_COST_ASTAR":
        d += weight * risk_l[nxt[0]][nxt[1]] * d
    return d


def check_endpoint(nav, xy, label):
    r, c = nav.cell_of(*xy)
    if not nav.in_grid(r, c):
        raise PlannerError(f"{label}_OUTSIDE_MAP", f"{label.lower()} {xy} is outside the map")
    if not nav.valid[r, c]:
        why = "not traversable" if not nav.free[r, c] else f"clearance {nav.clearance[r, c]:.1f}px < robot radius"
        raise PlannerError(f"{label}_IN_COLLISION", f"{label.lower()} {xy} is invalid ({why})")
    return r, c


def astar(nav, start_cell, goal_cell, planner="SAFETY_COST_ASTAR", weight=1.0):
    valid_l, risk_l = nav.valid.tolist(), np.nan_to_num(nav.risk).tolist()
    rows, cols, cell_px = nav.rows, nav.cols, nav.cell_px
    start_cell, goal_cell = tuple(start_cell), tuple(goal_cell)

    def h(cell):
        return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1]) * cell_px

    seq = itertools.count()
    heap = [(h(start_cell), h(start_cell), next(seq), start_cell)]
    costs, parents, closed, expanded = {start_cell: 0.0}, {}, set(), 0
    while heap:
        _, _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal_cell:
            path = [cur]
            while cur in parents:
                cur = parents[cur]
                path.append(cur)
            return {"status": "PATH_FOUND", "cells": [list(p) for p in reversed(path)], "cost": costs[goal_cell],
                    "nodes_expanded": expanded}
        for nxt, dc in _neighbors(cur, valid_l, rows, cols):
            cand = costs[cur] + _edge_cost(dc, cell_px, risk_l, nxt, planner, weight)
            if cand < costs.get(nxt, math.inf) - 1e-12:
                costs[nxt], parents[nxt] = cand, cur
                hn = h(nxt)
                heapq.heappush(heap, (cand + hn, hn, next(seq), nxt))
    return {"status": "NO_PATH", "cells": [], "cost": None, "nodes_expanded": expanded}


def cost_field(nav, start_cell, planner="SAFETY_COST_ASTAR", weight=1.0):
    """Dijkstra cost-to-come from start for every valid cell (inf = unreachable). Same edge costs as A*."""
    valid_l, risk_l = nav.valid.tolist(), np.nan_to_num(nav.risk).tolist()
    rows, cols, cell_px = nav.rows, nav.cols, nav.cell_px
    dist = np.full((rows, cols), np.inf)
    start_cell = tuple(start_cell)
    dist[start_cell] = 0.0
    best = {start_cell: 0.0}
    heap = [(0.0, start_cell)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > best.get(cur, math.inf):
            continue
        for nxt, dc in _neighbors(cur, valid_l, rows, cols):
            nd = d + _edge_cost(dc, cell_px, risk_l, nxt, planner, weight)
            if nd < best.get(nxt, math.inf) - 1e-12:
                best[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    for cell, d in best.items():
        dist[cell] = d
    return dist


def line_of_sight(valid, a, b, step=0.2):
    """True if the straight segment between cell centres a and b only passes through valid cells.
    Every sample requires all (up to 4) surrounding cell centres to be valid, which also forbids
    the diagonal corner cutting that A* forbids."""
    (r0, c0), (r1, c1) = a, b
    n = max(1, int(math.ceil(max(abs(r1 - r0), abs(c1 - c0)) / step)))
    for i in range(n + 1):
        t = i / n
        r, c = r0 + (r1 - r0) * t, c0 + (c1 - c0) * t
        for rr in {math.floor(r), math.ceil(r)}:
            for cc in {math.floor(c), math.ceil(c)}:
                if not (0 <= rr < valid.shape[0] and 0 <= cc < valid.shape[1]) or not valid[rr, cc]:
                    return False
    return True


def simplify(cells, valid):
    if len(cells) <= 2:
        return [list(c) for c in cells]
    out, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not line_of_sight(valid, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return [list(c) for c in out]


def polyline_length(pts):
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])))


def plan(nav, start_xy, goal_xy, pcfg, path_id):
    """Returns a PATH record. On failure status != PATH_FOUND, waypoints are empty (no straight-line fallback)."""
    rec = {"path_id": path_id, "coordinate_frame": "raw_image_px", "planner": pcfg["planner"],
           "simplification": pcfg["simplification"], "start_xy": list(start_xy), "goal_xy": list(goal_xy),
           "waypoints_xy": [], "raw_cells": [], "raw_waypoints_xy": [], "length_px": None, "raw_length_px": None,
           "error": None}
    try:
        s = check_endpoint(nav, start_xy, "START")
        g = check_endpoint(nav, goal_xy, "GOAL")
    except PlannerError as e:
        return {**rec, "status": e.code, "error": str(e)}
    t0 = perf_counter()
    res = astar(nav, s, g, pcfg["planner"], pcfg["safety_weight"])
    t_astar = perf_counter() - t0
    if res["status"] != "PATH_FOUND":
        return {**rec, "status": "NO_PATH", "error": "goal is not connected to start in the valid map",
                "nodes_expanded": res["nodes_expanded"], "planning_time_ms": round(t_astar * 1000, 2)}
    simp = simplify(res["cells"], nav.valid)
    raw_xy = [list(nav.cell_center(r, c)) for r, c in res["cells"]]
    simp_xy = [list(nav.cell_center(r, c)) for r, c in simp]
    return {**rec, "status": "PATH_FOUND", "raw_cells": res["cells"], "raw_waypoints_xy": raw_xy,
            "simplified_cells": simp, "waypoints_xy": simp_xy, "raw_length_px": round(polyline_length(raw_xy), 2),
            "length_px": round(polyline_length(simp_xy), 2), "astar_cost": round(res["cost"], 3),
            "nodes_expanded": res["nodes_expanded"], "planning_time_ms": round(t_astar * 1000, 2),
            "n_raw_waypoints": len(raw_xy), "n_waypoints": len(simp_xy),
            "start_cell": list(s), "goal_cell": list(g),
            "endpoint_snapping": "start/goal are snapped to their cell centres (<= cell_px/sqrt(2) px)"}
