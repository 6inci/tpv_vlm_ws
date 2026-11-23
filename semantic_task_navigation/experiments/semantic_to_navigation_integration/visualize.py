"""STEP 9: overlay of the integration result on the ORIGINAL image. A pure function of the image and the saved
integration_result.json. Every layer carries a provenance tag ([VLM] / [Python] / [detector+VLM crop check] / [map prior]).
A stage that failed or did not run draws nothing; it is listed under NOT DRAWN with its reason."""
import textwrap

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
COL = {"vlm": (0, 165, 255), "refined": (0, 215, 255), "adopted": (255, 190, 0), "withheld": (140, 140, 255),
       "placement": (60, 200, 60), "anchor": (255, 255, 255), "cone": (0, 110, 255), "cube": (0, 0, 255),
       "start": (200, 200, 200), "goal": (255, 0, 255), "path_a": (255, 128, 0), "path_b": (60, 60, 255),
       "map": (0, 235, 235)}


def _ascii(s):
    return str(s).encode("ascii", "replace").decode()


def _text(img, text, org, color, scale=0.55, thick=1):
    text = _ascii(text)
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x = int(max(0, min(org[0], img.shape[1] - tw - 2)))
    y = int(max(th + 3, min(org[1], img.shape[0] - bl - 2)))
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + bl), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _dashed(img, p1, p2, color, thick, dash=14):
    p1, p2 = np.array(p1, float), np.array(p2, float)
    n = max(1, int(np.linalg.norm(p2 - p1) // dash))
    for i in range(0, n, 2):
        a, b = p1 + (p2 - p1) * i / n, p1 + (p2 - p1) * min(i + 1, n) / n
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thick, cv2.LINE_AA)


def _dashed_poly(img, pts, color, thick):
    for a, b in zip(pts, pts[1:] + pts[:1]):
        _dashed(img, a, b, color, thick)


def _box_poly(b):
    return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]


def _fill(img, polys, color, alpha):
    mask = np.zeros(img.shape[:2], np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [np.round(np.array(p)).astype(np.int32)], 1)
    layer = img.copy()
    layer[mask.astype(bool)] = color
    img[:] = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)


def _polyline(img, pts, color, label):
    if len(pts) < 2:
        return
    p = [tuple(int(round(v)) for v in q) for q in pts]
    for a, b in zip(p, p[1:]):
        cv2.line(img, a, b, (0, 0, 0), 6, cv2.LINE_AA)
    for a, b in zip(p, p[1:]):
        cv2.arrowedLine(img, a, b, color, 3, cv2.LINE_AA, tipLength=0.06) if (a, b) == (p[-2], p[-1]) else \
            cv2.line(img, a, b, color, 3, cv2.LINE_AA)
    _text(img, label, (p[0][0] + 8, p[0][1] - 8), color, 0.5)


def _marker(img, xy, color, label, shape):
    c = (int(round(xy[0])), int(round(xy[1])))
    if shape == "square":
        cv2.rectangle(img, (c[0] - 9, c[1] - 9), (c[0] + 9, c[1] + 9), (0, 0, 0), 5)
        cv2.rectangle(img, (c[0] - 9, c[1] - 9), (c[0] + 9, c[1] + 9), color, 2)
    else:
        cv2.circle(img, c, 12, (0, 0, 0), 5)
        cv2.circle(img, c, 12, color, 2)
        cv2.line(img, (c[0] - 16, c[1]), (c[0] + 16, c[1]), color, 2)
        cv2.line(img, (c[0], c[1] - 16), (c[0], c[1] + 16), color, 2)
    _text(img, label, (c[0] + 16, c[1] - 12), color, 0.5)


def render(image, rec):
    img = image.copy()
    d, h = rec["draw"], img.shape[0]
    drawn, not_drawn = [], []

    for z in d["drive_zone_polygons"]:
        _dashed_poly(img, [tuple(p) for p in z], COL["map"], 2)
    if d["drive_zone_polygons"]:
        drawn.append((COL["map"], "[map prior, DRAFT_UNCONFIRMED, shown only because the map is ALIGNED] robot-drivable zone (dashed)"))

    vb = d.get("vlm_box_px")
    if vb:
        _dashed_poly(img, _box_poly(vb), COL["vlm"], 3)
        _text(img, f"[VLM] proposed destination: {d['vlm_label']} (UNVERIFIED)", (vb[0] + 4, vb[1] + 20), COL["vlm"], 0.5)
        drawn.append((COL["vlm"], "[VLM] initial proposal: destination box as output by Qwen3-VL (dashed)"))
    else:
        not_drawn.append("VLM proposal: " + str(rec["stage_status"]["vlm_proposal"]))
    if d.get("refined_box_px"):
        _dashed_poly(img, _box_poly(d["refined_box_px"]), COL["refined"], 2)
        drawn.append((COL["refined"], "[VLM grounding of its own evidence + Python screening] refined box (dashed)"))
    adopted_ok = rec["semantic_decision"]["status"].startswith("ADOPTED")
    if d.get("adopted_polygon"):
        if adopted_ok and d["adopted_passed_to_planning"]:
            _fill(img, [d["adopted_polygon"]], COL["adopted"], 0.28)
            cv2.polylines(img, [np.round(np.array(d["adopted_polygon"])).astype(np.int32)], True, COL["adopted"], 3)
            drawn.append((COL["adopted"], "[Python] ADOPTED semantic region after verification (passed to v0.2 placement)"))
        elif adopted_ok:
            _dashed_poly(img, [tuple(p) for p in d["adopted_polygon"]], COL["adopted"], 3)
            drawn.append((COL["adopted"], "[Python] adopted for semantics only; NOT passed on (map not aligned) (dashed)"))
        else:
            _dashed_poly(img, [tuple(p) for p in d["adopted_polygon"]], COL["withheld"], 3)
            drawn.append((COL["withheld"], "[Python] candidate region that was WITHHELD (failed required checks) (dashed)"))
    else:
        not_drawn.append("adopted semantic region: none (" + rec["semantic_decision"]["status"] + ")")
    if d.get("truck_box_px"):
        cv2.rectangle(img, tuple(int(v) for v in d["truck_box_px"][:2]), tuple(int(v) for v in d["truck_box_px"][2:]),
                      COL["anchor"], 2)
        drawn.append((COL["anchor"], "[detector + VLM crop check] truck used as anchor"))
    for b in d.get("cone_boxes_px", []):
        cv2.rectangle(img, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:]), COL["cone"], 2)
    if d.get("cone_boxes_px"):
        drawn.append((COL["cone"], f"[detector + VLM crop check] {len(d['cone_boxes_px'])} traffic cones (excluded from placement)"))
    if d["placement_polygons"]:
        _fill(img, d["placement_polygons"], COL["placement"], 0.45)
        for p in d["placement_polygons"]:
            cv2.polylines(img, [np.round(np.array(p)).astype(np.int32)], True, COL["placement"], 2)
        drawn.append((COL["placement"], "[Python, v0.2 placement_region] placeable area (geometric only; surface UNMEASURED)"))
    else:
        not_drawn.append("placement region: " + rec["stage_status"]["placement"])
    if d.get("target_box_px"):
        b = d["target_box_px"]
        cv2.rectangle(img, (int(b[0]) - 6, int(b[1]) - 6), (int(b[2]) + 6, int(b[3]) + 6), COL["cube"], 2)
        _text(img, "red cube", (b[0] + 10, b[1] - 8), COL["cube"], 0.5)
        drawn.append((COL["cube"], "[v0.2 HSV + DINO + VLM crop check] red cube"))
    else:
        not_drawn.append("red cube: " + rec["stage_status"]["cube_detection"])
    if d.get("start_xy") and d["paths"]:
        _marker(img, d["start_xy"], COL["start"], "start (MANUAL prior)" if d["start_manual"] else "start (detected)", "square")
        drawn.append((COL["start"], "[manual map prior] robot start" if d["start_manual"] else "[detector] robot start"))
    if d.get("pickup_goal_xy"):
        _marker(img, d["pickup_goal_xy"], COL["goal"], "pickup approach goal", "circle")
    if d.get("dropoff_goal_xy"):
        _marker(img, d["dropoff_goal_xy"], COL["goal"], "dropoff approach goal", "circle")
    if d.get("pickup_goal_xy") or d.get("dropoff_goal_xy"):
        drawn.append((COL["goal"], "[Python, v0.2 goal_selector] approach goals (reach PROVISIONAL, not measured)"))
    else:
        not_drawn.append("approach goals: " + rec["stage_status"]["approach"])
    for k, label, col in (("A", "path A start->pickup", COL["path_a"]), ("B", "path B pickup->dropoff (assumes pickup)", COL["path_b"])):
        if k in d["paths"]:
            _polyline(img, d["paths"][k], col, label)
            drawn.append((col, f"[Python, v0.2 A* + line-of-sight] {label}"))
    if not d["paths"]:
        not_drawn.append("paths: " + rec["stage_status"]["path"])

    lines = [((255, 255, 255), f"run {rec['run_id']} | image {rec['input']['image_id']} ({rec['input']['role']}) | "
                               f"instruction level {rec['input']['instruction_level']} (Korean sentence, see integration_result.json)"),
             ((0, 255, 255), f"INTEGRATION STATUS: {rec['integration_status']} | first blocking stage: {rec['first_blocking_stage']}"),
             ((200, 200, 200), "stages: " + " | ".join(f"{k}={v}" for k, v in rec["stage_status"].items())),
             ((200, 200, 200), "checks: " + " ".join(f"{c['id']}={c['status']}" for c in rec["verification"]["checks"])),
             ((0, 200, 255), ("VARIANT (diagnostic, not the primary configuration): " + ", ".join(rec["config_overrides"]))
              if rec.get("config_overrides") else "primary configuration (no overrides)"),
             ((0, 200, 255), "claims: robot_task_success NOT_CLAIMABLE | gripper reach PROVISIONAL (unmeasured) | map DRAFT_UNCONFIRMED "
                             "| image-space planning check only")]
    lines += [(c, t) for c, t in drawn]
    if not_drawn:
        lines.append(((120, 120, 255), "NOT DRAWN (not run / failed): " + "; ".join(not_drawn)))
    rows = []
    for c, t in lines:
        for i, part in enumerate(textwrap.wrap(t, 200)):
            rows.append((c, ("   " if i else "") + part, i == 0))
    panel = np.full((26 * len(rows) + 10, img.shape[1], 3), 24, np.uint8)
    for i, (c, t, first) in enumerate(rows):
        y = 26 * (i + 1) - 6
        if first and c not in ((255, 255, 255), (0, 255, 255), (200, 200, 200), (0, 200, 255), (120, 120, 255)):
            cv2.rectangle(panel, (8, y - 13), (24, y + 1), c, -1)
            cv2.putText(panel, _ascii(t), (32, y), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, _ascii(t), (8, y), FONT, 0.55, c, 1, cv2.LINE_AA)
    return np.vstack([img, panel])
