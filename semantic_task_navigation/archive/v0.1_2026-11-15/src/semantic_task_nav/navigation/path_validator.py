"""Independent path validation at pixel resolution.

The planner works on a cell grid; this validator samples the continuous polyline every
validation_sample_step_px and looks up the full-resolution clearance (distance to the nearest
non-free pixel). A path is accepted only if every sample has clearance >= robot_radius_px -
validation_tolerance_px (tolerance = grid discretisation).
"""
import math

import numpy as np


def validate_path(nav, waypoints_xy, start_xy, goal_xy, pcfg):
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    add("non_empty", len(waypoints_xy) >= 2, f"{len(waypoints_xy)} waypoints")
    if len(waypoints_xy) < 2:
        return {"valid": False, "checks": checks, "min_clearance_px": None, "first_violation": None}
    snap = nav.cell_px / math.sqrt(2) + 1e-6
    ds, dg = math.dist(waypoints_xy[0], start_xy), math.dist(waypoints_xy[-1], goal_xy)
    add("starts_at_start", ds <= snap, f"{ds:.2f}px (limit {snap:.2f})")
    add("ends_at_goal", dg <= snap, f"{dg:.2f}px (limit {snap:.2f})")
    need = nav.robot_radius_px - pcfg["validation_tolerance_px"]
    step = pcfg["validation_sample_step_px"]
    min_clear, first_bad, n_samples = math.inf, None, 0
    for i, (a, b) in enumerate(zip(waypoints_xy, waypoints_xy[1:])):
        n = max(1, int(math.ceil(math.dist(a, b) / step)))
        for k in range(n + 1):
            t = k / n
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            xi, yi = int(round(x)), int(round(y))
            n_samples += 1
            if not (0 <= xi < nav.width and 0 <= yi < nav.height):
                clear = -1.0
            else:
                clear = float(nav.clearance_full[yi, xi])
            min_clear = min(min_clear, clear)
            if clear < need and first_bad is None:
                first_bad = {"segment_index": i, "xy": [round(x, 1), round(y, 1)], "clearance_px": round(clear, 2),
                             "free_pixel": bool(clear > 0)}
    add("collision_free_footprint", first_bad is None,
        f"min clearance {min_clear:.2f}px, required >= {need:.2f}px over {n_samples} samples")
    return {"valid": all(c["ok"] for c in checks), "checks": checks, "min_clearance_px": round(min_clear, 2),
            "required_clearance_px": need, "first_violation": first_bad, "n_samples": n_samples}


def compare_raw_and_simplified(path_rec):
    if path_rec.get("status") != "PATH_FOUND":
        return None
    return {"raw_length_px": path_rec["raw_length_px"], "simplified_length_px": path_rec["length_px"],
            "length_reduction_pct": round(100 * (1 - path_rec["length_px"] / path_rec["raw_length_px"]), 2)
            if path_rec["raw_length_px"] else None,
            "raw_waypoints": path_rec["n_raw_waypoints"], "simplified_waypoints": path_rec["n_waypoints"]}
