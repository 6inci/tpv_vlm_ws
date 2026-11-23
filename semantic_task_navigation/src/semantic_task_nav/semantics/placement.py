"""object_placement_region: where the carried object could be put down inside a semantic loading region.

Geometry only: semantic region minus the footprints of accepted detections (e.g. the truck box) and
minus map obstacles / no-go areas, eroded by half the target's size so the object fits. Whether the
surface can physically hold the object and whether the gripper can reach it are NOT measured; every
record says so. A semantic region with no remaining area is PLACEMENT_REGION_EMPTY, which is a
different failure from "no robot approach" (navigation.goal_selector).
"""
import math

import cv2
import numpy as np

from ..contracts import RAW_PX

BASIS = ("geometric: semantic region minus detected object footprints and map obstacles/no-go, eroded by half the "
         "target size; placement surface and gripper reach UNMEASURED")


def _fill(shape, polys):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p, dtype=np.float64)).astype(np.int32)], 1)
    return m.astype(bool)


def placement_region(semantic_region, image_size, objects, target, nav, cfg):
    pl = cfg["placement"]
    w, h = image_size
    sem = _fill((h, w), [semantic_region["polygon_xy"]])
    mask = sem.copy()
    excluded = []
    for o in objects:
        if not o.get("accepted") or o["class_name"] not in pl["exclude_detected_classes"]:
            continue
        b = o["bbox_xyxy"]
        box = _fill((h, w), [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]])
        overlap = int((mask & box).sum())
        if overlap:
            excluded.append({"object_id": o["object_id"], "overlap_px": overlap})
        mask &= ~box
    after_objects = int(mask.sum())
    if nav is not None:
        for key in ("static_obstacle", "no_go"):
            overlap = int((mask & nav.full_layers[key]).sum())
            if overlap:
                excluded.append({"map_layer": key, "overlap_px": overlap})
            mask &= ~nav.full_layers[key]
    if target:
        tb = target["bbox_xyxy"]
        margin = int(math.ceil(max(tb[2] - tb[0], tb[3] - tb[1]) / 2))
        margin_basis = f"half of target {target['object_id']} box size"
    else:
        margin, margin_basis = pl["default_margin_px"], "placement.default_margin_px"
    after_obstacles = int(mask.sum())
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    eroded = cv2.erode(mask.astype(np.uint8), k) if margin > 0 else mask.astype(np.uint8)
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < pl["min_area_px"]:
            continue
        approx = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append([[int(x), int(y)] for x, y in approx])
    area = int(_fill((h, w), polys).sum()) if polys else 0
    rec = {"semantic_region_id": semantic_region["region_id"], "polygons_xy": polys, "coordinate_frame": RAW_PX,
           "area_px": area, "margin_px": margin, "margin_basis": margin_basis, "excluded": excluded,
           "areas_px": {"semantic": int(sem.sum()), "after_object_footprints": after_objects,
                        "after_map_obstacles": after_obstacles, "after_margin_and_min_area": area},
           "basis": BASIS, "feasibility": "UNVERIFIED (surface and reach unmeasured)"}
    if polys:
        rec.update(status="PLACEMENT_REGION_FOUND", reason=f"{len(polys)} component(s), {area} px")
    else:
        why = ("semantic region lies entirely on excluded objects/obstacles" if after_obstacles == 0 else
               f"remaining free area ({after_obstacles} px) vanishes after a {margin} px margin or is below "
               f"min_area_px={pl['min_area_px']}")
        rec.update(status="PLACEMENT_REGION_EMPTY", reason=why)
    return rec
