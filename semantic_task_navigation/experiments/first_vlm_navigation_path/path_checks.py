"""Checks on a finished path / start / goal that do not reuse the v0.2 grid logic (rasterised discs, painted-tape pixels, object boxes).
All coordinates are raw_image_px of the E1 photo (undistorted, 1920x1080). Radii are ASSUMED/ESTIMATED image-space values, not metres."""
import math

import cv2
import numpy as np


def polyline_samples(waypoints, step):
    pts = []
    for a, b in zip(waypoints, waypoints[1:]):
        n = max(1, int(math.ceil(math.dist(a, b) / step)))
        pts += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n)]
    pts.append(tuple(waypoints[-1]))
    return pts


def swept_disc_mask(shape, waypoints, radius, step=1.0):
    m = np.zeros(shape, np.uint8)
    for x, y in polyline_samples(waypoints, step):
        cv2.circle(m, (int(round(x)), int(round(y))), int(round(radius)), 1, -1)
    return m.astype(bool)


def swept_overlap(nav, waypoints, radius, step=1.0):
    """Pixels of the union of discs (radius) along the path that fall on each obstacle layer or outside the traversable polygons."""
    m = swept_disc_mask((nav.height, nav.width), waypoints, radius, step)
    L = nav.full_layers
    return {"radius_px": radius, "swept_area_px": int(m.sum()),
            "on_static_obstacle_px": int((m & L["static_obstacle"]).sum()), "on_no_go_px": int((m & L["no_go"]).sum()),
            "on_dynamic_obstacle_px": int((m & L["dynamic_obstacle"]).sum()),
            "outside_traversable_px": int((m & ~L["traversable"]).sum())}


def yellow_crossing(waypoints, yellow, step=1.0):
    """The path centre line against the painted yellow tape pixels of THIS photo (not the DRAFT polygons)."""
    h, w = yellow.shape
    dist = cv2.distanceTransform((~yellow).astype(np.uint8), cv2.DIST_L2, 5)
    hits, dmin, first = 0, math.inf, None
    for x, y in polyline_samples(waypoints, step):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            hits += 1
            first = first or [round(x, 1), round(y, 1)]
            continue
        d = float(dist[yi, xi])
        dmin = min(dmin, d)
        if d < 1.0:
            hits += 1
            first = first or [round(x, 1), round(y, 1)]
    return {"centre_line_samples_on_tape": hits, "first_hit_xy": first, "min_distance_to_tape_px": round(dmin, 1)}


def tape_overlap_of_footprint(waypoints, yellow, radius, step=4.0):
    """Informational: how much of the assumed footprint disc lies on tape (the tape is a boundary the body should not straddle)."""
    m = swept_disc_mask(yellow.shape, waypoints, radius, step)
    return {"radius_px": radius, "swept_pixels_on_tape": int((m & yellow).sum())}


def box_overlap_px(box, layer):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return int(layer[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].sum())


def point_in_box(p, b):
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def goal_checks(nav, goal_xy, bay_poly, objects):
    """The goal is where the ROBOT stops. It must not be the truck, the bay or any detected object, and must be collision-free."""
    r, c = nav.cell_of(*goal_xy)
    poly = np.array(bay_poly, np.float32).reshape(-1, 1, 2)
    signed = float(cv2.pointPolygonTest(poly, (float(goal_xy[0]), float(goal_xy[1])), True))   # > 0 inside the bay
    boxes = {o["object_id"]: o["bbox_xyxy"] for o in objects}
    inside = [k for k, b in boxes.items() if point_in_box(goal_xy, b)]
    return {"goal_xy": [round(goal_xy[0], 1), round(goal_xy[1], 1)], "cell": [r, c],
            "in_valid_robot_cell": bool(nav.in_grid(r, c) and nav.valid[r, c]),
            "clearance_px": round(float(nav.clearance_full[int(round(goal_xy[1])), int(round(goal_xy[0]))]), 1),
            "inside_bay_polygon": signed > 0, "distance_to_bay_polygon_px": round(max(0.0, -signed), 1),
            "inside_detected_object_boxes": inside, "ok": bool(nav.valid[r, c] and signed <= 0 and not inside)}
