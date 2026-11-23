"""Approach-pose selection. The object/region centre is never used as the robot goal.

object or semantic region (polygon, raw px)
  -> accessible candidate cells: valid robot-centre cells within [min, max] px of the polygon
  -> reachability: Dijkstra cost field from the start (same edge costs as A*)
  -> score = path cost + distance_weight * distance to polygon  -> selected goal pose
"""
import math

import cv2
import numpy as np

from ..contracts import GOAL_POSE_SCHEMA, RAW_PX, validate
from .planner import PlannerError, check_endpoint, cost_field


def _distance_to_polygon(nav, polygon):
    mask = np.ones((nav.height, nav.width), np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32)], 0)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ys = np.arange(nav.rows) * nav.cell_px + nav.cell_px // 2
    xs = np.arange(nav.cols) * nav.cell_px + nav.cell_px // 2
    return dist[np.ix_(ys, xs)]


def accessible_cell_count(nav, polygon, gcfg):
    """Collision-free robot-centre cells within [min, max] reach of a polygon (no reachability check)."""
    dist = _distance_to_polygon(nav, polygon)
    return int(((dist <= gcfg["max_dist_px"]) & (dist >= gcfg["min_dist_px"]) & nav.valid).sum())


def select_goal(nav, anchor_id, anchor_polygon, start_xy, purpose, gcfg, pcfg):
    res = {"purpose": purpose, "anchor": anchor_id, "anchor_polygon_xy": anchor_polygon, "coordinate_frame": RAW_PX,
           "params": dict(gcfg), "pose": None, "accessible_cells": [], "candidates_top_k": []}
    try:
        s = check_endpoint(nav, start_xy, "START")
    except PlannerError as e:
        return {**res, "status": e.code, "reason": str(e)}
    dist = _distance_to_polygon(nav, anchor_polygon)
    in_reach = (dist <= gcfg["max_dist_px"]) & (dist >= gcfg["min_dist_px"])
    accessible = in_reach & nav.valid
    costs = cost_field(nav, s, pcfg["planner"], pcfg["safety_weight"])
    reachable = accessible & np.isfinite(costs)
    stats = {"cells_in_reach": int(in_reach.sum()), "cells_in_reach_traversable": int((in_reach & nav.free).sum()),
             "cells_accessible_collision_free": int(accessible.sum()), "cells_reachable_from_start": int(reachable.sum())}
    res["stats"] = stats
    res["accessible_cells"] = [[int(r), int(c)] for r, c in zip(*np.nonzero(reachable))]
    if not accessible.any():
        return {**res, "status": "NO_ACCESSIBLE_CANDIDATE",
                "reason": f"no collision-free robot position within {gcfg['max_dist_px']} px of {anchor_id}"}
    if not reachable.any():
        return {**res, "status": "CANDIDATES_UNREACHABLE",
                "reason": f"{stats['cells_accessible_collision_free']} collision-free candidate cells exist, "
                          "but none is connected to the start"}
    score = np.where(reachable, costs + gcfg["distance_weight"] * dist, np.inf)
    order = np.argsort(score, axis=None)[:gcfg["top_k_report"]]
    for idx in order:
        r, c = np.unravel_index(idx, score.shape)
        res["candidates_top_k"].append({"cell": [int(r), int(c)], "xy": list(nav.cell_center(r, c)),
                                        "path_cost": round(float(costs[r, c]), 2),
                                        "distance_to_anchor_px": round(float(dist[r, c]), 2),
                                        "score": round(float(score[r, c]), 2)})
    best = res["candidates_top_k"][0]
    cx, cy = np.asarray(anchor_polygon, dtype=np.float64).mean(axis=0)
    x, y = best["xy"]
    res["pose"] = validate({"purpose": purpose, "xy": [x, y], "cell": best["cell"],
                            "heading_deg": round(math.degrees(math.atan2(cy - y, cx - x)), 1),
                            "heading_note": "image-frame direction towards the anchor centroid (x right, y down); "
                                            "mecanum base does not need to rotate along the path",
                            "coordinate_frame": RAW_PX, "anchor": anchor_id, "path_cost": best["path_cost"],
                            "metric_valid": False}, GOAL_POSE_SCHEMA)
    res["status"] = "GOAL_SELECTED"
    return res
