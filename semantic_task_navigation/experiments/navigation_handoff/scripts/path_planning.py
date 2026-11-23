#!/usr/bin/env python3
"""STEP 3: orthogonal Path A (start -> grasp stop pose) and Path B (grasp stop pose -> placement stop pose) on the STEP 2 map.pgm.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/navigation_handoff/scripts/path_planning.py --photo photo_5      # new results/<time>_step3_plan_photo_5/

Inputs (read only): STEP 2 map.pgm / map_layers.npy / coordinate_frames.json, config/object_positions/<photo>.yaml (USER_CONFIRMED only).
Reused unchanged: v0.2 planner._edge_cost, planner._neighbors (fed 4 neighbours), planner.astar (4-neighbour baseline record),
v0.2 path_validator.validate_path (inscribed-disc necessary check), task_plan_prototype footprint_nav.sweep_check (axis-aligned body sweep).
New: configuration space of the margin body from the PGM cells for a heading, A* on 4 neighbours with a direction-change cost
(state = cell + direction, same v0.2 edge cost), turn-point waypoints, heading-aware approach candidates, rotating-rectangle check, figures.
Gripper offsets are unmeasured: no grasp or placement is claimed (UNVERIFIED); the front-to-target gap the gripper must bridge is reported.
"""
import argparse
import csv
import heapq
import itertools
import json
import math
import platform
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coordinate_transform as CT  # noqa: E402
import map_export as ME  # noqa: E402
import object_positions as OP  # noqa: E402
sys.path.insert(0, str(CT.EXP.parent / "task_plan_prototype"))
import footprint_nav as FN  # noqa: E402  (reused unchanged)
from semantic_task_nav.config import code_fingerprint, sha256_file  # noqa: E402
from semantic_task_nav.navigation import planner as P  # noqa: E402
from semantic_task_nav.navigation.path_validator import validate_path  # noqa: E402

NEIGHBORS4 = ((1, 0), (0, 1), (-1, 0), (0, -1))
MOVE_NAME = {(0, 1): "+x_map", (0, -1): "-x_map", (-1, 0): "+y_map", (1, 0): "-y_map"}    # grid (dr, dc): rows grow towards -y_map
MOVE_YAW = {(0, 1): 0.0, (-1, 0): 90.0, (0, -1): 180.0, (1, 0): -90.0}
FONT = cv2.FONT_HERSHEY_SIMPLEX


@contextmanager
def four_neighbours():
    """Feed the unchanged v0.2 _neighbors / astar with 4 neighbours; always restored."""
    saved = P.NEIGHBORS
    P.NEIGHBORS = NEIGHBORS4
    try:
        yield
    finally:
        P.NEIGHBORS = saved


# ================================================================================================ planning map from the PGM
class PgmMap:
    """Occupancy = the STEP 2 PGM cells. Whole-body checks use the same cells upsampled (nearest) to raster_px_per_cm."""

    def __init__(self, pgm, layers, values, grid, frame, px_per_cm, safety_margin_cm):
        assert pgm.shape == (grid.rows, grid.cols), "PGM size differs from the STEP 2 grid"
        self.grid, self.frame, self.layers = grid, frame, layers
        self.rows, self.cols = pgm.shape
        self.s = float(px_per_cm)
        self.sub = int(round(grid.res_cm * px_per_cm))
        self.cell_px = self.sub                                   # v0.2 units = px of this raster
        self.free = pgm == values["free"]
        self.free_full = np.kron(self.free.astype(np.uint8), np.ones((self.sub, self.sub), np.uint8)).astype(bool)
        self.height, self.width = self.free_full.shape
        self.clearance_full = cv2.distanceTransform(np.pad(self.free_full.astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        self.ys = np.arange(self.rows) * self.sub + self.sub // 2
        self.xs = np.arange(self.cols) * self.sub + self.sub // 2
        self.clearance = self.clearance_full[np.ix_(self.ys, self.xs)]
        self.safety_margin_px = safety_margin_cm * self.s
        self._cspace = {}

    # ---- frames: floor cm <-> raster px <-> cell
    def floor_to_px(self, pts):
        return (np.asarray(pts, np.float64).reshape(-1, 2) - [self.grid.x0, self.grid.y0]) * self.s

    def px_to_floor(self, pts):
        return np.asarray(pts, np.float64).reshape(-1, 2) / self.s + [self.grid.x0, self.grid.y0]

    def cell_of_floor(self, p):
        return int(math.floor((p[1] - self.grid.y0) / self.grid.res_cm)), int(math.floor((p[0] - self.grid.x0) / self.grid.res_cm))

    def centre_floor(self, r, c):
        x, y = self.grid.cell_center_floor(r, c)
        return np.array([float(x), float(y)])

    def in_grid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    # ---- body
    def _poly_px(self, c, L, W, yaw_map, outward_px=0.75):
        """Rectangle L (along the heading) x W around raster point c; raster y points down, so the raster angle is -yaw_map."""
        a = math.radians(-yaw_map)
        u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        hl, hw = L / 2 * self.s + outward_px, W / 2 * self.s + outward_px
        c = np.asarray(c, np.float64)
        return np.array([c + hl * u + hw * v, c + hl * u - hw * v, c - hl * u - hw * v, c - hl * u + hw * v])

    @staticmethod
    def _mask(poly, x0, y0, w, h):
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round((poly - [x0, y0]) * 16).astype(np.int32)], 1, shift=4)
        return m.astype(bool)

    def body_polygon_px(self, centre_floor, L, W, yaw_map):
        return self._poly_px(self.floor_to_px([centre_floor])[0], L, W, yaw_map)

    def body_free(self, centre_floor, L, W, yaw_map):
        """(all body pixels free, number of non-free pixels). Outside the raster counts as blocked."""
        poly = self.body_polygon_px(centre_floor, L, W, yaw_map)
        x0, y0 = np.floor(poly.min(0)).astype(int) - 1
        x1, y1 = np.ceil(poly.max(0)).astype(int) + 2
        m = self._mask(poly, x0, y0, x1 - x0, y1 - y0)
        H, W_ = self.free_full.shape
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W_), min(y1, H)
        outside = int(m.sum() - m[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0].sum())
        bad = int((m[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] & ~self.free_full[sy0:sy1, sx0:sx1]).sum()) + outside
        return bad == 0, bad

    def cspace(self, L, W, yaw_map):
        """Cells where the whole (L x W, heading yaw_map) body lies on free pixels. Axis headings: erosion with the body kernel."""
        key = (round(L, 3), round(W, 3), round(yaw_map % 360, 3))
        if key in self._cspace:
            return self._cspace[key]
        kernel = None
        if abs(((yaw_map % 90) + 45) % 90 - 45) < 1e-9:
            # kernel = the SAME rasterised body polygon that body_free() tests, centred on an integer pixel (cell centres are integer pixels)
            R = int(math.ceil(math.hypot(L, W) / 2 * self.s)) + 3
            k = self._mask(self._poly_px((R, R), L, W, yaw_map), 0, 0, 2 * R + 1, 2 * R + 1)
            ys_, xs_ = np.nonzero(k)
            if ys_.min() + ys_.max() == 2 * R and xs_.min() + xs_.max() == 2 * R:          # symmetric -> anchor at the centre is exact
                kernel = k[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1]
        if kernel is not None and kernel.all():
            fits = cv2.erode(self.free_full.astype(np.uint8), kernel.astype(np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            valid = fits[np.ix_(self.ys, self.xs)] & self.free
        else:
            valid = np.zeros((self.rows, self.cols), bool)
            for r, c in zip(*np.nonzero(self.free)):
                valid[r, c] = self.body_free(self.centre_floor(r, c), L, W, yaw_map)[0]
        rr = min(L, W) / 2 * self.s
        risk = np.where(valid, np.clip(1.0 - (self.clearance - rr) / self.safety_margin_px, 0.0, 1.0), np.nan)
        self._cspace[key] = (valid, risk)
        return self._cspace[key]

    def view(self, L, W, yaw_map):
        valid, risk = self.cspace(L, W, yaw_map)
        return SimpleNamespace(valid=valid, risk=risk, rows=self.rows, cols=self.cols, cell_px=self.cell_px)

    def v02_nav(self, L, W):
        """Object with the attributes v0.2 validate_path reads (inscribed disc of the body)."""
        return SimpleNamespace(cell_px=self.cell_px, robot_radius_px=min(L, W) / 2 * self.s, width=self.width, height=self.height,
                               clearance_full=self.clearance_full)


# ================================================================================================ search
def astar_orthogonal(view, start, goal, planner, weight, turn_px):
    """v0.2 A* loop with state (cell, last move); v0.2 _neighbors (4 neighbours) and _edge_cost, plus turn_px per direction change."""
    valid_l, risk_l = view.valid.tolist(), np.nan_to_num(view.risk).tolist()
    rows, cols, cell_px = view.rows, view.cols, view.cell_px
    start, goal = tuple(start), tuple(goal)

    def h(c):
        return (abs(c[0] - goal[0]) + abs(c[1] - goal[1])) * cell_px        # admissible: every 4-neighbour step costs >= cell_px

    seq = itertools.count()
    s0 = (start, None)
    heap = [(h(start), h(start), next(seq), s0)]
    costs, parents, closed, expanded = {s0: 0.0}, {}, set(), 0
    with four_neighbours():
        while heap:
            _, _, _, st = heapq.heappop(heap)
            if st in closed:
                continue
            closed.add(st)
            expanded += 1
            cur, last = st
            if cur == goal:
                path, s = [cur], st
                while s in parents:
                    s = parents[s]
                    path.append(s[0])
                return {"status": "PATH_FOUND", "cells": [list(p) for p in reversed(path)], "cost": costs[st], "nodes_expanded": expanded}
            for nxt, dc in P._neighbors(cur, valid_l, rows, cols):
                mv = (nxt[0] - cur[0], nxt[1] - cur[1])
                cand = costs[st] + P._edge_cost(dc, cell_px, risk_l, nxt, planner, weight) + (turn_px if last is not None and mv != last else 0.0)
                ns = (nxt, mv)
                if cand < costs.get(ns, math.inf) - 1e-12:
                    costs[ns], parents[ns] = cand, st
                    heapq.heappush(heap, (cand + h(nxt), h(nxt), next(seq), ns))
    return {"status": "NO_PATH", "cells": [], "cost": None, "nodes_expanded": expanded}


def runs_of(cells):
    """[(move (dr, dc), first index, last index)] of consecutive equal moves."""
    out = []
    for i, (a, b) in enumerate(zip(cells, cells[1:])):
        mv = (b[0] - a[0], b[1] - a[1])
        if out and out[-1][0] == mv:
            out[-1][2] = i + 1
        else:
            out.append([mv, i, i + 1])
    return [tuple(r) for r in out]


def turn_indices(cells):
    """Indices kept as waypoints: start, every direction change, goal. All raw cells stay in the path record."""
    if len(cells) < 2:
        return list(range(len(cells)))
    return [0] + [r[1] for r in runs_of(cells)[1:]] + [len(cells) - 1]


def is_orthogonal(cells):
    return all(abs(b[0] - a[0]) + abs(b[1] - a[1]) == 1 for a, b in zip(cells, cells[1:]))


def rotation_check(pm, centre_floor, L, W, yaw_from, yaw_to, step_deg):
    """Static rotating-rectangle test in both directions (no dynamics). Returns the per-direction result and the chosen one."""
    res = {}
    for name, sgn in (("ccw", 1), ("cw", -1)):
        sweep = ((yaw_to - yaw_from) * sgn) % 360
        n = max(1, int(math.ceil(sweep / step_deg)))
        first_bad = None
        for k in range(n + 1):
            a = yaw_from + sgn * sweep * k / n
            ok, bad = pm.body_free(centre_floor, L, W, a)
            if not ok:
                first_bad = {"yaw_map_deg": round(CT.wrap_deg(a), 1), "non_free_px": bad}
                break
        res[name] = {"sweep_deg": round(sweep, 1), "ok": first_bad is None, "first_blocked": first_bad}
    feas = [k for k in res if res[k]["ok"]]
    chosen = min(feas, key=lambda k: res[k]["sweep_deg"]) if feas else None
    return {"ok": chosen is not None, "direction": chosen, "from_yaw_map_deg": yaw_from, "to_yaw_map_deg": yaw_to, "step_deg": step_deg,
            "per_direction": res, "check": "margin body rasterised at every step on the PGM cells (static; dynamics not modelled)"}


def start_escape(pm, s_cell, yaw, nominal, margin_valid, max_cells):
    """Robot physically at s_cell: the nominal body must fit there; the nearest straight axis move whose nominal body sweep is free and
    that ends on a cell where the margin body fits. Returns the list of cells (start excluded) or None."""
    L, W = nominal
    ok0, _ = pm.body_free(pm.centre_floor(*s_cell), L, W, yaw)
    if not ok0:
        return None, "nominal body not free at the start"
    best = None
    for dr, dc in NEIGHBORS4:
        for k in range(1, max_cells + 1):
            r, c = s_cell[0] + k * dr, s_cell[1] + k * dc
            if not pm.in_grid(r, c):
                break
            if not pm.body_free(pm.centre_floor(r, c), L, W, yaw)[0]:
                break
            if margin_valid[r, c]:
                if best is None or k < len(best):
                    best = [[s_cell[0] + j * dr, s_cell[1] + j * dc] for j in range(1, k + 1)]
                break
    return best, None if best else f"no straight move of <= {max_cells} cells reaches a cell where the margin body fits"


# ================================================================================================ approach candidates
def gripper_corridor(pm, front_floor, target_floor, step_cm=0.5):
    """PGM layer of every sample on the centre line from the nominal body front to the target (gripper width UNKNOWN)."""
    n = max(1, int(math.ceil(np.linalg.norm(np.subtract(target_floor, front_floor)) / step_cm)))
    counts = {}
    for k in range(n + 1):
        p = np.add(front_floor, np.subtract(target_floor, front_floor) * k / n)
        r, c = pm.cell_of_floor(p)
        name = ME.CODE_NAMES[int(pm.layers[r, c])].split(" (")[0] if pm.in_grid(r, c) else "OUTSIDE_MAP"
        counts[name] = counts.get(name, 0) + 1
    return {"samples": n + 1, "layers_crossed": counts,
            "static_obstacle_under_gripper_path": "STATIC_OBSTACLE" in counts,
            "rule": "gripper may cross NOT_DRIVABLE / NO_GO (user decision 2026-11-19); the body may not. Gripper width/height UNKNOWN."}


def approach_candidates(pm, target_floor, nominal, margin, yaws, search_cm, gripper):
    L, W = nominal
    Lm, Wm = L + 2 * margin, W + 2 * margin
    tr, tc = pm.cell_of_floor(target_floor)
    out = []
    for yaw in yaws:
        valid, _ = pm.cspace(Lm, Wm, yaw)
        f = np.array([math.cos(math.radians(yaw)), -math.sin(math.radians(yaw))])      # facing direction in floor cm (y towards the camera)
        dr, dc = int(round(-f[1])), int(round(-f[0]))                                   # walk from the target opposite to the facing
        rec = {"yaw_map_deg": yaw, "facing": MOVE_NAME[(-dr, -dc)], "status": "NO_COLLISION_FREE_POSE_ON_AXIS", "searched_cm": search_cm}
        for k in range(0, int(search_cm / pm.grid.res_cm) + 1):
            r, c = tr + k * dr, tc + k * dc
            if not pm.in_grid(r, c):
                break
            ctr = pm.centre_floor(r, c)
            d = float(np.dot(np.subtract(target_floor, ctr), f))
            if d < L / 2 or not valid[r, c]:
                continue
            dv = np.subtract(target_floor, ctr)
            lat = float(f[0] * dv[1] - f[1] * dv[0])                     # signed lateral offset of the target from the body centre line
            front = ctr + f * (L / 2)
            m = pm.frame.floor_to_map([ctr])[0]
            rec.update({"status": "POSE_FOUND", "cell": [int(r), int(c)], "centre_floor_cm": ctr.round(2).tolist(),
                        "pose_map": {"x_m": round(float(m[0]), 4), "y_m": round(float(m[1]), 4), "yaw_deg": float(yaw)},
                        "centre_to_target_along_heading_cm": round(d, 2), "lateral_offset_cm": round(lat, 2),
                        "front_to_target_gap_cm": round(d - L / 2, 2),
                        "required_gripper_offset_if_dy0": {"dx_m": round(d / 100, 4), "dy_m": 0.0, "status": "DERIVED, not a measurement"},
                        "margin_body_free": pm.body_free(ctr, Lm, Wm, yaw)[0], "nominal_body_free": pm.body_free(ctr, L, W, yaw)[0],
                        "gripper_corridor": gripper_corridor(pm, front, target_floor)})
            if gripper.get("offset_x_m") is not None and gripper.get("offset_y_m") is not None:
                g = ctr + f * gripper["offset_x_m"] * 100 + np.array([-f[1], f[0]]) * (-gripper["offset_y_m"] * 100)
                rec["gripper_alignment_error_cm"] = round(float(np.linalg.norm(g - np.asarray(target_floor))), 2)
            else:
                rec["gripper_alignment_error_cm"] = None
                rec["gripper_alignment_status"] = "UNVERIFIED: gripper offset not measured (dx, dy = null)"
            break
        out.append(rec)
    return out


# ================================================================================================ legs
def plan_leg(pm, label, start_floor, start_yaw, goal_cell, goal_yaw, nominal, margin, pcfg, allow_escape):
    """One leg. Same heading: no rotation. Heading change: rotate at the start, at the goal, or at an open cell on the way
    (cell chosen by v0.2 cost_field sums, first one whose rotating-rectangle check passes). Per cell: body heading after any
    rotation at that cell (cell_yaw) and the body checked while leaving it (cell_body: nominal on the start-escape cells)."""
    L, W = nominal
    Lm, Wm = L + 2 * margin, W + 2 * margin
    turn_px, step = pcfg["turn_penalty_cm"] * pm.s, pcfg["rotation_step_deg"]
    s_cell, goal_cell = pm.cell_of_floor(start_floor), tuple(goal_cell)
    same = abs(CT.wrap_deg(goal_yaw - start_yaw)) < 1e-9
    strategies = ["no_rotation"] if same else ["rotate_at_start", "rotate_at_goal", "rotate_at_open_cell"]
    base = {"leg": label, "start_yaw_map_deg": start_yaw, "goal_yaw_map_deg": goal_yaw, "start_cell": list(s_cell), "goal_cell": list(goal_cell)}
    view_s = pm.view(Lm, Wm, start_yaw)
    escape, why = [], None
    if not view_s.valid[s_cell]:
        if allow_escape:
            escape, why = start_escape(pm, s_cell, start_yaw, nominal, view_s.valid, int(pcfg["start_escape_max_cm"] / pm.grid.res_cm))
        else:
            escape, why = None, "margin body does not fit at the start"
    if escape is None:
        return None, [{**base, "strategy": s, "status": "START_IN_COLLISION", "reason": why} for s in strategies]
    a_cell = tuple(escape[-1]) if escape else s_cell
    n_esc = len(escape)

    def search(view, a, b):
        res = astar_orthogonal(view, a, b, pcfg["planner"], pcfg["safety_weight"], turn_px)
        with four_neighbours():                                            # unchanged v0.2 A* on 4 neighbours: baseline record only
            bl = P.astar(view, a, b, pcfg["planner"], pcfg["safety_weight"])
        return res, {"n_turns": len(runs_of(bl["cells"])) - 1 if len(bl["cells"]) > 1 else 0, "n_cells": len(bl["cells"]), "status": bl["status"]}

    tried = []
    for name in strategies:
        rec = {**base, "strategy": name, "events": [], "status": None}
        if n_esc:
            rec["events"].append({"type": "start_escape", "from_index": 0, "to_index": n_esc, "body_yaw_map_deg": start_yaw,
                                  "check": "nominal body swept cell by cell; margin body fits at the end"})
        if name == "rotate_at_open_cell":
            view_g = pm.view(Lm, Wm, goal_yaw)
            if not view_g.valid[goal_cell]:
                tried.append({**rec, "status": "ENDPOINT_NOT_IN_CSPACE"})
                continue
            with four_neighbours():
                cs = P.cost_field(view_s, a_cell, pcfg["planner"], pcfg["safety_weight"])
                cg = P.cost_field(view_g, goal_cell, pcfg["planner"], pcfg["safety_weight"])
            tot = cs + cg
            okm = np.isfinite(tot) & view_s.valid & view_g.valid
            order = np.argwhere(okm)[np.argsort(tot[okm], kind="stable")][: pcfg["rotation_cell_candidates"]]
            X, rot, n_checked = None, None, 0
            for r, c in order:
                n_checked += 1
                rot = rotation_check(pm, pm.centre_floor(r, c), Lm, Wm, start_yaw, goal_yaw, step)
                if rot["ok"]:
                    X = (int(r), int(c))
                    break
            if X is None:
                tried.append({**rec, "status": "NO_ROTATION_CELL", "reason": f"{n_checked} cheapest common cells checked, none can rotate"})
                continue
            p1, b1 = search(view_s, a_cell, X)
            p2, b2 = search(view_g, X, goal_cell)
            if p1["status"] != "PATH_FOUND" or p2["status"] != "PATH_FOUND":
                tried.append({**rec, "status": "NO_PATH"})
                continue
            main = p1["cells"] + p2["cells"][1:]
            yaws = [start_yaw] * (len(p1["cells"]) - 1) + [goal_yaw] * len(p2["cells"])
            rot_at = len(p1["cells"]) - 1
            rec["rotation_cell"] = list(X)
            rec["rotation_cell_rank"] = n_checked
            baseline = {"n_turns": b1["n_turns"] + b2["n_turns"], "n_cells": b1["n_cells"] + b2["n_cells"] - 1}
        else:
            travel = goal_yaw if name == "rotate_at_start" else start_yaw
            if name == "rotate_at_start":
                rot = rotation_check(pm, pm.centre_floor(*a_cell), Lm, Wm, start_yaw, goal_yaw, step)
                if not rot["ok"]:
                    tried.append({**rec, "status": "ROTATION_BLOCKED_AT_START", "events": rec["events"] + [{"type": "rotate", "at_index": n_esc, **rot}]})
                    continue
            view = pm.view(Lm, Wm, travel)
            if not view.valid[a_cell] or not view.valid[goal_cell]:
                tried.append({**rec, "status": "ENDPOINT_NOT_IN_CSPACE",
                              "reason": f"start valid {bool(view.valid[a_cell])}, goal valid {bool(view.valid[goal_cell])} at travel yaw {travel}"})
                continue
            p, b = search(view, a_cell, goal_cell)
            if p["status"] != "PATH_FOUND":
                tried.append({**rec, "status": "NO_PATH"})
                continue
            main = p["cells"]
            if name == "rotate_at_goal":
                rot = rotation_check(pm, pm.centre_floor(*goal_cell), Lm, Wm, start_yaw, goal_yaw, step)
                if not rot["ok"]:
                    tried.append({**rec, "status": "ROTATION_BLOCKED_AT_GOAL", "events": rec["events"] + [{"type": "rotate", "at_index": n_esc + len(main) - 1, **rot}]})
                    continue
            yaws = [travel] * len(main)
            if name == "rotate_at_goal":
                yaws[-1] = goal_yaw
            rot_at = 0 if name == "rotate_at_start" else len(main) - 1 if name == "rotate_at_goal" else None
            baseline = {"n_turns": b["n_turns"], "n_cells": b["n_cells"]}
        cells = ([list(s_cell)] + escape[:-1] + main) if n_esc else main
        if rot_at is not None:
            rec["events"].append({"type": "rotate", "at_index": n_esc + rot_at, **rot})
        rec.update({"status": "PATH_FOUND", "raw_cells": [list(map(int, c)) for c in cells], "n_escape_cells": n_esc,
                    "cell_yaw": [float(start_yaw)] * n_esc + [float(y) for y in yaws], "cell_body": ["nominal"] * n_esc + ["margin"] * len(main),
                    "baseline_v02_astar_4neighbours": {**baseline, "length_cm": round((baseline["n_cells"] - 1) * pm.grid.res_cm, 1)}})
        tried.append(rec)
    ok = [t for t in tried if t["status"] == "PATH_FOUND"]
    best = min(ok, key=lambda t: (len(t["raw_cells"]), len(runs_of(t["raw_cells"])))) if ok else None
    return best, tried


def describe_leg(pm, leg, nominal, margin, pcfg):
    """Raw cells -> coordinates, waypoints (start, direction changes, rotation points, escape end, goal), segments, checks."""
    L, W = nominal
    Lm, Wm = L + 2 * margin, W + 2 * margin
    cells, yaw, body = leg["raw_cells"], leg["cell_yaw"], leg["cell_body"]
    n_esc = leg.get("n_escape_cells", 0)
    floor = np.array([pm.centre_floor(r, c) for r, c in cells])
    mp, img = pm.frame.floor_to_map(floor), pm.frame.floor_to_img(floor)
    rot_idx = {e["at_index"] for e in leg["events"] if e["type"] == "rotate"}
    wps = sorted(set(turn_indices(cells)) | rot_idx | ({n_esc} if n_esc else set()))
    segs = []
    for a, b in zip(wps, wps[1:]):
        mv = (cells[a + 1][0] - cells[a][0], cells[a + 1][1] - cells[a][1])
        rel = abs(CT.wrap_deg(MOVE_YAW[mv] - yaw[a]))
        segs.append({"from_index": a, "to_index": b, "move": MOVE_NAME[mv], "move_yaw_map_deg": MOVE_YAW[mv], "length_cm": round((b - a) * pm.grid.res_cm, 1),
                     "body_yaw_map_deg": yaw[a], "body": body[a],
                     "body_vs_move": "forward" if rel < 1e-6 else "backward" if abs(rel - 180) < 1e-6 else "lateral",
                     "constant_direction": all((cells[i + 1][0] - cells[i][0], cells[i + 1][1] - cells[i][1]) == mv for i in range(a, b))})
    checks = {"orthogonal_4_connected": is_orthogonal(cells), "segments_straight_between_waypoints": all(s["constant_direction"] for s in segs),
              "no_diagonal_segment_between_waypoints": all((floor[a][0] == floor[b][0]) or (floor[a][1] == floor[b][1]) for a, b in zip(wps, wps[1:]))}
    bad_cells = [i for i in range(n_esc, len(cells))
                 if not pm.cspace(Lm, Wm, yaw[i])[0][tuple(cells[i])] or (i > n_esc and not pm.cspace(Lm, Wm, yaw[i - 1])[0][tuple(cells[i])])]
    checks["cells_in_margin_cspace_for_arrival_and_departure_heading"] = {"ok": not bad_cells, "violations": bad_cells[:10]}
    first_bad, n_samples, fn_ok, fn_n = None, 0, True, 0                   # independent whole-body sweep per segment + reused sweep_check
    for s in segs:
        a, b = s["from_index"], s["to_index"]
        Lc, Wc = (L, W) if s["body"] == "nominal" else (Lm, Wm)
        n = max(1, int(math.ceil(np.linalg.norm(floor[b] - floor[a]) / pcfg["sweep_step_cm"])))
        for k in range(n + 1):
            p = floor[a] + (floor[b] - floor[a]) * k / n
            ok, nb = pm.body_free(p, Lc, Wc, s["body_yaw_map_deg"])
            n_samples += 1
            if not ok and first_bad is None:
                first_bad = {"floor_cm": p.round(2).tolist(), "non_free_px": nb, "body": [Lc, Wc], "yaw": s["body_yaw_map_deg"]}
        t = s["body_yaw_map_deg"]
        ext = (Lc, Wc) if round(t / 90) % 2 == 0 else (Wc, Lc)
        fn = FN.sweep_check(pm.free_full, [floor[a].tolist(), floor[b].tolist()], ext, -pm.grid.x0, pm.s, pcfg["sweep_step_cm"])
        fn_ok, fn_n = fn_ok and fn["ok"], fn_n + fn["n_samples"]
    checks["whole_body_sweep_generic"] = {"ok": first_bad is None, "n_samples": n_samples, "first_violation": first_bad,
                                          "body": f"margin body {Lm} x {Wm} cm (start-escape segment: nominal {L} x {W} cm)"}
    checks["whole_body_sweep_reused_footprint_nav"] = {"ok": fn_ok, "n_samples": fn_n}
    part_px = pm.floor_to_px(floor[[w for w in wps if w >= n_esc]]).tolist()        # escape uses the nominal body (checked above)
    if len(part_px) >= 2:
        v = validate_path(pm.v02_nav(Lm, Wm), part_px, part_px[0], part_px[-1], {"validation_tolerance_px": 2, "validation_sample_step_px": 1.0})
        checks["v02_validate_path_inscribed_disc"] = {"valid": v["valid"], "min_clearance_px": v.get("min_clearance_px"),
                                                      "required_px": v.get("required_clearance_px"), "note": "necessary condition only"}
    else:
        checks["v02_validate_path_inscribed_disc"] = {"valid": True, "note": "zero-length margin part"}
    checks["rotation_events_ok"] = all(e["ok"] for e in leg["events"] if e["type"] == "rotate")
    passed = (checks["orthogonal_4_connected"] and checks["segments_straight_between_waypoints"] and checks["no_diagonal_segment_between_waypoints"]
              and checks["cells_in_margin_cspace_for_arrival_and_departure_heading"]["ok"] and checks["whole_body_sweep_generic"]["ok"]
              and checks["whole_body_sweep_reused_footprint_nav"]["ok"] and checks["v02_validate_path_inscribed_disc"]["valid"] and checks["rotation_events_ok"])
    return {"floor": floor, "map": mp, "img": img, "waypoint_indices": wps, "segments": segs, "body_yaw": yaw, "checks": checks,
            "collision_checks_passed": bool(passed), "length_cm": round((len(cells) - 1) * pm.grid.res_cm, 1), "n_turns": len(turn_indices(cells)) - 2}


# ================================================================================================ outputs
def write_csv(path, leg, d):
    wps = set(d["waypoint_indices"])
    mv_at = {}
    for s in d["segments"]:
        for i in range(s["from_index"], s["to_index"]):
            mv_at[i] = s
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "row", "col", "x_map_m", "y_map_m", "x_floor_cm", "y_floor_cm", "u_px", "v_px",
                    "move_to_next", "move_yaw_map_deg", "body_yaw_map_deg", "is_waypoint"])
        for i, (rc, fl, m, im) in enumerate(zip(leg["raw_cells"], d["floor"], d["map"], d["img"])):
            s = mv_at.get(i)
            w.writerow([i, rc[0], rc[1], f"{m[0]:.4f}", f"{m[1]:.4f}", f"{fl[0]:.2f}", f"{fl[1]:.2f}", f"{im[0]:.1f}", f"{im[1]:.1f}",
                        s["move"] if s else "", s["move_yaw_map_deg"] if s else "", d["body_yaw"][i], int(i in wps)])


def pose_list(leg, d):
    """Pose list in the map frame: start, direction changes, rotation points (arrival pose + rotate-in-place pose), goal.
    yaw = body heading (gripper side), not the movement direction."""
    out, mp, yaw = [], d["map"], d["body_yaw"]
    rot = {e["at_index"]: e for e in leg["events"] if e["type"] == "rotate"}
    n_esc = leg.get("n_escape_cells", 0)
    last = len(mp) - 1
    for i in d["waypoint_indices"]:
        xy = {"x_m": round(float(mp[i][0]), 4), "y_m": round(float(mp[i][1]), 4)}
        kind = "start" if i == 0 else "goal" if i == last else "escape_end" if n_esc and i == n_esc else "rotation_point" if i in rot else "turn"
        arrive = leg["start_yaw_map_deg"] if i == 0 else yaw[i - 1]
        if i in rot:
            out.append({"kind": kind, "index": i, **xy, "yaw_deg": float(rot[i]["from_yaw_map_deg"])})
            out.append({"kind": "rotate_in_place", "index": i, **xy, "yaw_deg": float(rot[i]["to_yaw_map_deg"]), "direction": rot[i]["direction"]})
        else:
            out.append({"kind": kind, "index": i, **xy, "yaw_deg": float(arrive)})
    for k, p in enumerate(out):
        p["seq"] = k
        p["yaw_rad"] = round(math.radians(p["yaw_deg"]), 6)
    return out


# ================================================================================================ figures
COL_A, COL_B, COL_MOVE, COL_HEAD = (255, 140, 0), (0, 140, 255), (0, 200, 0), (0, 0, 255)


def _text(img, s, org, color=(255, 255, 255), scale=0.55, thick=1):
    org = tuple(int(round(v)) for v in org)
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _ip(p):
    return tuple(int(round(v)) for v in p)


def draw_scene(canvas, to_px, pm, scene, scale_px_per_cm):
    """Everything in floor cm, drawn through to_px (photo: inverse floor transform; map: grid scaling)."""
    lw = max(2, int(round(scale_px_per_cm * 1.2)))
    al = 12.0                                                                   # arrow length cm
    for key, col in (("A", COL_A), ("B", COL_B)):
        d = scene["legs"].get(key)
        if not d:
            continue
        pts = to_px(d["floor"])
        cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, col, lw, cv2.LINE_AA)
        for s in d["segments"]:                                                  # movement arrows (green), mid-segment
            a, b = d["floor"][s["from_index"]], d["floor"][s["to_index"]]
            if np.linalg.norm(b - a) < 3:
                continue
            mid, u = (a + b) / 2, (b - a) / np.linalg.norm(b - a)
            p0, p1 = to_px([mid - u * min(al, np.linalg.norm(b - a) / 3) / 2, mid + u * min(al, np.linalg.norm(b - a) / 3) / 2])
            cv2.arrowedLine(canvas, _ip(p0), _ip(p1), COL_MOVE, lw, cv2.LINE_AA, tipLength=0.45)
        for i in d["waypoint_indices"][1:-1]:                                    # body heading arrows (red) at turn points
            f = d["floor"][i]
            yaw = d["body_yaw"][i]
            h = np.array([math.cos(math.radians(yaw)), -math.sin(math.radians(yaw))])
            p0, p1 = to_px([f, f + h * 15])
            cv2.circle(canvas, _ip(p0), lw + 2, col, -1)
            cv2.arrowedLine(canvas, _ip(p0), _ip(p1), COL_HEAD, max(1, lw - 1), cv2.LINE_AA, tipLength=0.35)
    for pose in scene["poses"]:                                                  # body rectangles + heading arrows
        c, yaw, col = np.array(pose["floor"]), pose["yaw"], pose["color"]
        a = math.radians(yaw)
        u, v = np.array([math.cos(a), -math.sin(a)]), np.array([math.sin(a), math.cos(a)])
        L, W = pose["size"]
        rect = [c + u * L / 2 + v * W / 2, c + u * L / 2 - v * W / 2, c - u * L / 2 - v * W / 2, c - u * L / 2 + v * W / 2]
        cv2.polylines(canvas, [np.round(to_px(rect)).astype(np.int32)], True, col, lw, cv2.LINE_AA)
        p0, p1 = to_px([c, c + u * (L / 2 + 10)])
        cv2.arrowedLine(canvas, _ip(p0), _ip(p1), COL_HEAD, lw + 1, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(canvas, _ip(p0), lw + 4, pose["dot"], -1)
        cv2.circle(canvas, _ip(p0), lw + 4, (255, 255, 255), 1)
        _text(canvas, pose["label"], to_px([c + v * (W / 2 + 6)])[0] + [-40, 14], pose["dot"], 0.5, 1)
        if pose.get("target") is not None:                                     # gripper line: body front -> target (UNVERIFIED)
            q0, q1 = to_px([c + u * L / 2, pose["target"]])
            n = 12
            for k in range(0, n, 2):
                cv2.line(canvas, _ip(q0 + (q1 - q0) * k / n), _ip(q0 + (q1 - q0) * (k + 1) / n), (255, 0, 255), max(1, lw - 1), cv2.LINE_AA)
    for t in scene["targets"]:
        p = to_px([t["floor"]])[0]
        if t["kind"] == "cube":
            cv2.circle(canvas, _ip(p), lw + 4, (0, 0, 255), -1)
            cv2.circle(canvas, _ip(p), lw + 4, (255, 255, 255), 1)
        elif t["kind"] == "object":                                             # object click when a separate work target was set
            cv2.circle(canvas, _ip(p), lw + 5, (255, 255, 255), max(1, lw - 1))
        else:
            cv2.drawMarker(canvas, _ip(p), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 6 * lw, lw + 1)
        _text(canvas, t["label"], p + [8, -8], (255, 255, 255), 0.5, 1)


def legend_lines(scene):
    return [("Path A (start -> grasp stop)", COL_A), ("Path B (grasp stop -> place stop)", COL_B),
            ("green arrow: movement direction (along the path)", COL_MOVE), ("red arrow: body front / gripper side (heading)", COL_HEAD),
            ("rectangles: nominal body 35 x 24 cm at start / stops", (255, 255, 255)), ("magenta dashes: body front -> target (gripper must bridge)", (255, 0, 255)),
            ("", None)] + [(s, (0, 255, 255)) for s in scene["status_lines"]]


def render_photo(img, pm, scene, path):
    out = img.copy()
    to_px = lambda f: pm.frame.floor_to_img(np.asarray(f, np.float64).reshape(-1, 2))  # noqa: E731
    draw_scene(out, to_px, pm, scene, 5.0)
    panel = np.full((out.shape[0], 720, 3), 25, np.uint8)
    _text(panel, "PATH PLANNING ON THE PHOTO (floor-level projection)", (15, 32), (0, 255, 255), 0.62, 2)
    for i, (s, col) in enumerate(legend_lines(scene)):
        if col is not None:
            _text(panel, s, (15, 70 + 27 * i), col, 0.52)
    _text(panel, "Tall objects lean in the photo; only floor-level geometry is placed exactly.", (15, out.shape[0] - 20), (200, 200, 200), 0.48)
    cv2.imwrite(str(path), np.hstack([out, panel]))


def render_map(pm, scene, path, S=6):
    g = pm.grid
    ml, mt, mb = 70, 40, 60
    h, w = g.rows * S, g.cols * S
    canvas = np.full((mt + h + mb, ml + w + 720, 3), 25, np.uint8)
    canvas[mt:mt + h, ml:ml + w] = cv2.resize(ME.color_layers(pm.layers), (w, h), interpolation=cv2.INTER_NEAREST)
    to_px = lambda f: (np.asarray(f, np.float64).reshape(-1, 2) - [g.x0, g.y0]) / g.res_cm * S + [ml, mt]  # noqa: E731
    for v in np.arange(0.0, g.cols * g.resolution_m + g.origin_m[0] + 1e-9, 0.5):
        x = to_px(pm.frame.map_to_floor([[v, 0]]))[0, 0]
        cv2.line(canvas, (int(x), mt + h), (int(x), mt + h + 6), (255, 255, 255), 1)
        _text(canvas, f"{v:.1f}", (x - 12, mt + h + 22), scale=0.45)
    for v in np.arange(0.0, g.rows * g.resolution_m + g.origin_m[1] + 1e-9, 0.5):
        y = to_px(pm.frame.map_to_floor([[0, v]]))[0, 1]
        cv2.line(canvas, (ml - 6, int(y)), (ml, int(y)), (255, 255, 255), 1)
        _text(canvas, f"{v:.1f}", (ml - 45, y + 5), scale=0.45)
    _text(canvas, "x_map [m] ->", (ml + w // 2 - 50, mt + h + 48), scale=0.5)
    _text(canvas, "y_map [m]", (5, mt - 12), scale=0.5)
    draw_scene(canvas, to_px, pm, scene, S / g.res_cm)
    px = ml + w + 20
    _text(canvas, "PATH PLANNING ON map.pgm (ROS map frame, 0.02 m cells)", (px, mt), (0, 255, 255), 0.6, 2)
    lines = [("free / static obstacle / no-go / not drivable / unknown", (255, 255, 255))] + legend_lines(scene)
    for i, (s, col) in enumerate(lines):
        if col is not None:
            _text(canvas, s, (px, mt + 34 + 26 * i), col, 0.5)
    cv2.imwrite(str(path), canvas)


# ================================================================================================ main
def select_and_plan(pm, kind, target_floor, start_floor, start_yaw, nominal, margin, pcfg, gripper, allow_escape):
    cands = approach_candidates(pm, target_floor, nominal, margin, pcfg["approach_yaws_map_deg"], pcfg["approach_search_cm"], gripper)
    for c in cands:
        if c["status"] != "POSE_FOUND":
            continue
        best, tried = plan_leg(pm, kind, start_floor, start_yaw, tuple(c["cell"]), c["yaw_map_deg"], nominal, margin, pcfg, allow_escape)
        c["leg_attempts"] = [{k: v for k, v in t.items() if k != "raw_cells"} for t in tried]
        c["leg"] = best
        if best is None:
            c["status"] = "NO_FEASIBLE_LEG"
            continue
        c["leg_length_cm"] = round((len(best["raw_cells"]) - 1) * pm.grid.res_cm, 1)
        c["status"] = "FEASIBLE_PLAN_GRASP_UNVERIFIED"
    feas = [c for c in cands if c["status"] == "FEASIBLE_PLAN_GRASP_UNVERIFIED"]
    chosen = min(feas, key=lambda c: (c["front_to_target_gap_cm"], c["leg_length_cm"])) if feas else None
    return cands, chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", required=True)
    args = ap.parse_args()
    cfg = CT.load_config()
    pcfg, mcfg = cfg["planning"], cfg["map_export"]
    before = {"v02": code_fingerprint()["sha256"]}
    doc, pos_path = OP.load_approved(cfg, args.photo)                 # refuses anything not USER_APPROVED (position_picker.py)
    photo = next(p for p in CT.resolve_photos(cfg) if p["name"] == args.photo)
    frame, fc, map_run = OP.load_step2_frame(cfg)
    map_files = {k: map_run / "maps" / k for k in ("map.pgm", "map.yaml", "map_layers.npy", "map_preview.png")}
    before["map_files"] = {k: sha256_file(v) for k, v in map_files.items()}
    pgm, _ = ME.read_pgm(map_files["map.pgm"])
    my = yaml.safe_load(map_files["map.yaml"].read_text(encoding="utf-8"))
    g = fc["grid"]
    grid = CT.MapGrid(frame, g["extent_floor_cm"], g["resolution_m"])
    assert grid.origin_m == my["origin"] and (grid.rows, grid.cols) == pgm.shape, "STEP 2 grid and map.yaml disagree"
    pm = PgmMap(pgm, np.load(map_files["map_layers.npy"]), mcfg["values"], grid, frame, pcfg["raster_px_per_cm"], pcfg["safety_margin_cm"])
    margin = fc["floor_transform_record"]["footprint_margin_cm"]
    nominal = (cfg["robot"]["length_cm"], cfg["robot"]["width_cm"])
    gripper = {**cfg["gripper"], "offset_y_m_assumed": pcfg["gripper_dy_assumption_m"]}

    pts = {"robot_base_link": OP._pt(frame, doc["robot"]["base_link_image_px"]),
           "A_cube_object": OP._pt(frame, doc["A_cube"]["object_image_px"]), "A_grasp_target": OP._pt(frame, doc["A_cube"]["grasp_target_image_px"]),
           "B_pallet": OP._pt(frame, doc["B_placement"]["pallet_image_px"]), "B_placement_target": OP._pt(frame, doc["B_placement"]["placement_target_image_px"])}
    start_floor, start_yaw = np.array(pts["robot_base_link"]["floor_cm"]), float(doc["robot"]["yaw_map_deg"])
    cube_floor, place_floor = np.array(pts["A_grasp_target"]["floor_cm"]), np.array(pts["B_placement_target"]["floor_cm"])

    def run_all(mrg):
        gc, gsel = select_and_plan(pm, "A", cube_floor, start_floor, start_yaw, nominal, mrg, pcfg, gripper, True)
        pc, psel = ([], None)
        if gsel:
            pc, psel = select_and_plan(pm, "B", place_floor, np.array(gsel["centre_floor_cm"]), gsel["yaw_map_deg"], nominal, mrg, pcfg, gripper, False)
        return gc, gsel, pc, psel

    grasp_c, grasp, place_c, place = run_all(margin)
    sens = {}
    for m0 in pcfg["sensitivity_margins_cm"]:
        gc, gs, pc, ps = run_all(m0)
        sens[f"margin_{m0}cm"] = {"role": "SENSITIVITY (diagnostic, not a hand-off result)",
                                  "grasp": [{k: c.get(k) for k in ("yaw_map_deg", "status", "front_to_target_gap_cm", "leg_length_cm")} for c in gc],
                                  "place": [{k: c.get(k) for k in ("yaw_map_deg", "status", "front_to_target_gap_cm", "leg_length_cm")} for c in pc],
                                  "selected": {"grasp_yaw": gs and gs["yaw_map_deg"], "place_yaw": ps and ps["yaw_map_deg"]}}

    # ---- run folder: <run>/maps (the delivered static map, byte copies) + <run>/<photo>/
    stamp = datetime.now().astimezone()
    root = CT.resolve(cfg["results_dir"]) / (stamp.strftime("%Y%m%d_%H%M%S") + f"_plan_{args.photo}")
    (root / "maps").mkdir(parents=True, exist_ok=False)
    out = root / args.photo
    out.mkdir()
    for k in ("map.pgm", "map.yaml", "map_preview.png"):
        (root / "maps" / k).write_bytes(map_files[k].read_bytes())
        assert sha256_file(root / "maps" / k) == before["map_files"][k]
    img = cv2.imread(photo["path"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    legs, desc = {}, {}
    for key, sel in (("A", grasp), ("B", place)):
        if sel:
            legs[key] = sel["leg"]
            desc[key] = describe_leg(pm, sel["leg"], nominal, margin, pcfg)
            write_csv(out / f"path_{key}.csv", sel["leg"], desc[key])

    # ---- scene for the figures
    status_lines = []
    poses = [{"floor": start_floor.tolist(), "yaw": start_yaw, "size": nominal, "color": (200, 200, 200), "dot": (255, 255, 255), "label": f"START yaw {start_yaw:.0f}"}]
    if grasp:
        poses.append({"floor": grasp["centre_floor_cm"], "yaw": grasp["yaw_map_deg"], "size": nominal, "color": (0, 0, 255), "dot": (0, 0, 255),
                      "label": f"GRASP STOP yaw {grasp['yaw_map_deg']:.0f} UNVERIFIED", "target": cube_floor.tolist()})
        status_lines.append(f"Path A: {desc['A']['length_cm']} cm, {desc['A']['n_turns']} turns, collision checks {'PASS' if desc['A']['collision_checks_passed'] else 'FAIL'}")
        status_lines.append(f"grasp: gripper centre must be {grasp['centre_to_target_along_heading_cm']} cm ahead of base_link "
                            f"({grasp['front_to_target_gap_cm']} cm ahead of the body front), lateral {grasp['lateral_offset_cm']} cm - UNVERIFIED")
    else:
        status_lines.append("Path A: NOT FOUND (see path_plan.json)")
    if place:
        poses.append({"floor": place["centre_floor_cm"], "yaw": place["yaw_map_deg"], "size": nominal, "color": (255, 0, 0), "dot": (255, 0, 0),
                      "label": f"PLACE STOP yaw {place['yaw_map_deg']:.0f} UNVERIFIED", "target": place_floor.tolist()})
        status_lines.append(f"Path B: {desc['B']['length_cm']} cm, {desc['B']['n_turns']} turns, collision checks {'PASS' if desc['B']['collision_checks_passed'] else 'FAIL'}")
        status_lines.append(f"place: gripper centre must be {place['centre_to_target_along_heading_cm']} cm ahead of base_link "
                            f"({place['front_to_target_gap_cm']} cm ahead of the body front), lateral {place['lateral_offset_cm']} cm - UNVERIFIED")
    elif grasp:
        status_lines.append("Path B: NOT FOUND (see path_plan.json)")
    status_lines += [f"body margin {margin} cm/side (floor transform error) -> {nominal[0] + 2 * margin:.0f} x {nominal[1] + 2 * margin:.0f} cm",
                     "gripper offset dx, dy: NOT MEASURED -> grasp / placement UNVERIFIED",
                     "no validated homography: PROVISIONAL mat-seam transform (3.5-5.2 cm), z = 0",
                     "payload / gripper interference during Path B: UNVERIFIED", "map zones: DRAFT (not human-confirmed)"]
    targets = [{"kind": "cube", "floor": cube_floor.tolist(), "label": "A grasp target"},
               {"kind": "place", "floor": place_floor.tolist(), "label": "B placement target"}]
    for key, tkey, lab in (("A_cube_object", "A_grasp_target", "A cube (click)"), ("B_pallet", "B_placement_target", "B pallet (click)")):
        if np.linalg.norm(np.subtract(pts[key]["image_px"], pts[tkey]["image_px"])) > 0.5:
            targets.append({"kind": "object", "floor": pts[key]["floor_cm"], "label": lab})
    scene = {"legs": desc, "poses": poses, "status_lines": status_lines, "targets": targets}
    render_photo(img, pm, scene, out / "path_planning_photo.png")
    render_map(pm, scene, out / "path_planning_map.png")

    # ---- records
    def leg_rec(key):
        if key not in legs:
            return None
        lg, d = legs[key], desc[key]
        return {"strategy": lg["strategy"], "start_yaw_map_deg": lg["start_yaw_map_deg"], "goal_yaw_map_deg": lg["goal_yaw_map_deg"],
                "rotation_cell": lg.get("rotation_cell"), "length_cm": d["length_cm"], "n_raw_points": len(lg["raw_cells"]), "n_turns": d["n_turns"],
                "n_escape_cells": lg.get("n_escape_cells", 0), "events": lg["events"], "segments": d["segments"],
                "waypoint_indices": d["waypoint_indices"], "poses": pose_list(lg, d), "checks": d["checks"],
                "collision_checks_passed": d["collision_checks_passed"], "baseline_v02_astar_4neighbours": lg["baseline_v02_astar_4neighbours"],
                "raw_path_file": f"path_{key}.csv"}

    def align(sel, kind):
        if not sel:
            return None
        return {"stop_pose_map": sel["pose_map"], "target": kind, "centre_to_target_along_heading_cm": sel["centre_to_target_along_heading_cm"],
                "front_to_target_gap_cm": sel["front_to_target_gap_cm"], "lateral_offset_cm": sel["lateral_offset_cm"],
                "required_gripper_offset_if_dy0": sel["required_gripper_offset_if_dy0"], "gripper_alignment_error_cm": sel["gripper_alignment_error_cm"],
                "status": "UNVERIFIED (gripper offset dx, dy not measured)", "gripper_corridor": sel["gripper_corridor"]}

    strip = lambda cs: [{k: v for k, v in c.items() if k != "leg"} for c in cs]  # noqa: E731
    r = doc["robot"]
    unverified = ["gripper offset (dx, dy) not measured: grasp and placement alignment UNVERIFIED (dy = 0 is a temporary assumption)",
                  "a stop pose closest to the target on the map is NOT evidence that the gripper can reach / grasp / place",
                  "no validated homography (config/camera.yaml homography_status: not_available): PROVISIONAL E1 mat-seam transform, "
                  "independent seam error up to 3.5 cm, mat-node residual up to 5.2 cm (bottom row 4-5 cm)",
                  "A / B are user clicks projected at z = 0; height parallax of the cube and the pallet not corrected",
                  f"work targets: A {doc['A_cube']['grasp_target_source']}, B {doc['B_placement']['placement_target_source']} (TEMPORARY when equal to the click)",
                  f"robot base_link: {r['base_link_source']}; heading: {r['yaw_source']} (no automatic heading estimate exists)",
                  "rotation check is static (rasterised rectangle); mecanum dynamics, slip and controller tracking not modelled",
                  "gripper / payload geometry unknown: interference above obstacles and payload overhang during Path B UNVERIFIED",
                  "DRAFT zone polygons not human-confirmed; per-photo movable obstacles: none entered",
                  "Nav2 map_server load test not run (format validation only, user decision)",
                  "whether the ROS2 controller executes the lateral (mecanum) moves and the heading commands is not verified"]
    plan = {"photo": {k: photo[k] for k in ("name", "path", "sha256", "earlier_id")}, "approved_positions_file": str(pos_path),
            "approved_at": doc["approved_at"], "positions": pts, "start_yaw_map_deg": start_yaw,
            "map": {"run": str(map_run), "files_sha256": before["map_files"], "planning_occupancy": "STEP 2 map.pgm cells (no per-photo obstacle added)"},
            "body": {"nominal_cm": list(nominal), "margin_cm_per_side": margin, "planning_body_cm": [nominal[0] + 2 * margin, nominal[1] + 2 * margin]},
            "gripper": gripper, "grasp": {"candidates": strip(grasp_c), "selected_yaw_map_deg": grasp and grasp["yaw_map_deg"], "alignment": align(grasp, "A grasp target")},
            "place": {"candidates": strip(place_c), "selected_yaw_map_deg": place and place["yaw_map_deg"], "alignment": align(place, "B placement target")},
            "path_A": leg_rec("A"), "path_B": leg_rec("B"), "sensitivity": sens, "selection_rule": pcfg["selection"], "unverified": unverified,
            "claims": {"grasp_success": "NOT_CLAIMED", "placement_success": "NOT_CLAIMED", "robot_motion": "NONE (offline)"}}
    (out / "path_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (out / "object_positions.json").write_text(json.dumps({"approved_file": str(pos_path), "approved": doc, "derived": pts}, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    wy = {"frame_id": "map", "units": {"position": "m", "yaw": "deg (yaw_rad also given), ROS map frame, counter-clockwise from +x_map"},
          "note": ("Pose lists: start, rotations, direction-change points, goal. yaw = body heading (gripper side), NOT the movement direction. "
                   "Giving these poses to Nav2 as goals lets Nav2 plan its OWN path between them; it does not make the robot follow path_A.csv / path_B.csv."),
          "status": "grasp / placement UNVERIFIED (gripper offset unmeasured, provisional floor transform)",
          "path_A": leg_rec("A") and leg_rec("A")["poses"], "path_B": leg_rec("B") and leg_rec("B")["poses"],
          "grasp_stop_pose": grasp and {**grasp["pose_map"], "front_to_target_gap_cm": grasp["front_to_target_gap_cm"], "status": "UNVERIFIED"},
          "place_stop_pose": place and {**place["pose_map"], "front_to_target_gap_cm": place["front_to_target_gap_cm"], "status": "UNVERIFIED"}}
    (out / "waypoints.yaml").write_text(yaml.safe_dump(wy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    after_ok = code_fingerprint()["sha256"] == before["v02"] and all(sha256_file(v) == before["map_files"][k] for k, v in map_files.items()) \
        and sha256_file(photo["path"]) == photo["sha256"]
    validation = {"path_A": legs.get("A") and {"collision_checks_passed": desc["A"]["collision_checks_passed"], **desc["A"]["checks"]},
                  "path_B": legs.get("B") and {"collision_checks_passed": desc["B"]["collision_checks_passed"], **desc["B"]["checks"]},
                  "path_A_status": "PATH_FOUND" if grasp else "NOT_FOUND", "path_B_status": "PATH_FOUND" if place else "NOT_FOUND",
                  "grasp_candidates": [(c["yaw_map_deg"], c["status"]) for c in grasp_c], "place_candidates": [(c["yaw_map_deg"], c["status"]) for c in place_c],
                  "grasp_alignment": "UNVERIFIED", "placement_alignment": "UNVERIFIED", "map_copies_identical_to_step2": True,
                  "environment_unchanged_after_run": after_ok, "unverified": unverified}
    (out / "validation.json").write_text(json.dumps(validation, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (root / "config_snapshot.yaml").write_text(CT.CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    (root / "manifest.json").write_text(json.dumps({
        "experiment_id": cfg["experiment_id"], "step": "STEP3_path_planning", "run_id": root.name, "created": stamp.isoformat(timespec="seconds"),
        "git_commit": None, "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
        "code_sha256": {p.name: sha256_file(p) for p in (Path(__file__).resolve(), HERE / "coordinate_transform.py", HERE / "map_export.py", HERE / "object_positions.py")},
        "config_sha256": sha256_file(CT.CONFIG_PATH), "inputs": {"photo": photo["sha256"], "approved_positions": sha256_file(pos_path), **before["map_files"]},
        "v02_code_fingerprint": before["v02"], "environment_unchanged_after_run": after_ok,
        "hardware_condition": "offline; no model, GPU, ROS or robot", "outputs_sha256": {str(p.relative_to(root)): sha256_file(p) for p in files}}, indent=1), encoding="utf-8")
    summary = {"run_dir": str(root), "photo_dir": str(out), "grasp_yaw": grasp and grasp["yaw_map_deg"], "place_yaw": place and place["yaw_map_deg"],
               "grasp_candidates": [(c["yaw_map_deg"], c["status"]) for c in grasp_c], "place_candidates": [(c["yaw_map_deg"], c["status"]) for c in place_c],
               "A": leg_rec("A") and {k: leg_rec("A")[k] for k in ("length_cm", "n_turns", "collision_checks_passed", "strategy")},
               "B": leg_rec("B") and {k: leg_rec("B")[k] for k in ("length_cm", "n_turns", "collision_checks_passed", "strategy")},
               "environment_unchanged": after_ok}
    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
