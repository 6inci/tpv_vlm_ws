"""Figures for the first VLM -> approach-path run. Drawn only from saved/computed records and the original E1 photo.
Colour = source of the information (legend panel says so too):
  orange dashed  Qwen3-VL (stored reply)      white boxes  Grounding DINO + Qwen crop check      green  Python bay (stored localization)
  blue / red / magenta  v0.2 DRAFT map prior (hand-drawn, transferred)      yellow dot / cyan line  Python: v0.2 goal_selector / A*"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "semantic_evidence_verification"))
from localize_bay import _dashed, _fill, _t, panel  # noqa: E402  (small drawing helpers of the previous step, reused)

FONT = cv2.FONT_HERSHEY_SIMPLEX
C = {"qwen": (0, 165, 255), "dino": (255, 255, 255), "robot": (255, 200, 120), "bay": (60, 200, 60), "trav": (255, 140, 40), "static": (40, 40, 255),
     "nogo": (255, 60, 255), "goal": (0, 255, 255), "path": (255, 255, 0), "valid": (255, 220, 160), "free": (255, 60, 255)}


def _ring(img, c, r, color, thick=2, dash=18):
    for a in range(0, 360, 2 * dash):
        cv2.ellipse(img, (int(c[0]), int(c[1])), (int(r), int(r)), 0, a, a + dash, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.ellipse(img, (int(c[0]), int(c[1])), (int(r), int(r)), 0, a, a + dash, color, thick, cv2.LINE_AA)


def _line(img, pts, color, thick):
    p = np.round(np.array(pts)).astype(np.int32)
    cv2.polylines(img, [p], False, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.polylines(img, [p], False, color, thick, cv2.LINE_AA)


def _poly(img, poly, color, thick=2, closed=True):
    p = np.round(np.array(poly)).astype(np.int32)
    cv2.polylines(img, [p], closed, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.polylines(img, [p], closed, color, thick, cv2.LINE_AA)


def _cross(img, c, color, size=14, thick=3):
    x, y = int(c[0]), int(c[1])
    for a, b in (((x - size, y), (x + size, y)), ((x, y - size), (x, y + size))):
        cv2.line(img, a, b, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.line(img, a, b, color, thick, cv2.LINE_AA)


def _goal(img, goal, color=C["goal"]):
    p = goal["pose"]
    x, y = p["xy"]
    cv2.circle(img, (int(x), int(y)), 13, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y)), 10, color, -1, cv2.LINE_AA)
    a = np.radians(p["heading_deg"])
    e = (int(x + 55 * np.cos(a)), int(y + 55 * np.sin(a)))
    cv2.arrowedLine(img, (int(x), int(y)), e, (0, 0, 0), 6, cv2.LINE_AA, tipLength=0.35)
    cv2.arrowedLine(img, (int(x), int(y)), e, color, 3, cv2.LINE_AA, tipLength=0.35)


def _valid_contours(img, nav, color):
    v = cv2.resize(nav.valid.astype(np.uint8), (nav.cols * nav.cell_px, nav.rows * nav.cell_px), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(v, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.polylines(img, cs, True, color, 1, cv2.LINE_AA)


def _layers(img, spec):
    if spec is None:
        return
    _fill(img, [z["polygon"] for z in spec["traversable"]], C["trav"], 0.13)
    _fill(img, [z["polygon"] for z in spec["static_obstacles"]], C["static"], 0.30)
    _fill(img, [z["polygon"] for z in spec["no_go"]], C["nogo"], 0.25)
    for z in spec["traversable"]:
        _poly(img, z["polygon"], C["trav"], 2)
    for z in spec["static_obstacles"]:
        _poly(img, z["polygon"], C["static"], 2)
    for z in spec["no_go"]:
        _poly(img, z["polygon"], C["nogo"], 2)


def _objects(img, objs, labels=True):
    for o in objs:
        b = [int(round(v)) for v in o["bbox_xyxy"]]
        col = C["robot"] if o["class_name"] == "mobile_robot" else C["dino"]
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 0, 0), 4)
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), col, 2)
        if labels and o["class_name"] != "traffic_cone":
            _t(img, "DINO " + o["object_id"], (b[0], b[1] - 4 if b[1] > 24 else b[3] + 18), col, 0.5)


def scene(ctx, variant=None, show_valid=True, show_start=True, show_free=False):
    img = ctx["image"].copy()
    _layers(img, ctx["spec"])
    v = ctx["runtime"].get(variant) if variant else None
    if v and show_valid:
        _valid_contours(img, v["nav"], C["valid"])
    _objects(img, ctx["objects"] + ([ctx["robot"]] if ctx["robot"] else []))
    if ctx.get("bay_polygon"):
        _fill(img, [ctx["bay_polygon"]], C["bay"], 0.20)
        _poly(img, ctx["bay_polygon"], C["bay"], 3)
    _dashed(img, ctx["qwen_box"], C["qwen"], 4)
    if v:
        if v["path"] and v["path"]["status"] == "PATH_FOUND":
            wp = v["path"]["waypoints_xy"]
            _line(img, wp, C["path"], 4)
            for p in wp:
                cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 0), -1)
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1)
        if v["goal"].get("pose"):
            _goal(img, v["goal"])
    if show_start and ctx.get("start_candidates"):
        for s in ctx["start_candidates"]:
            _cross(img, s["xy"], (255, 255, 255) if s["shift_px"] == 0 else (170, 170, 170), 14 if s["shift_px"] == 0 else 8, 3 if s["shift_px"] == 0 else 2)
        s0 = ctx["start_candidates"][0]["xy"]
        r_assumed = ctx["variants"][ctx["primary"]]["robot_radius_px"] if ctx["primary"] else 35
        _ring(img, s0, r_assumed, (0, 255, 255), 2)
        stress = next((k for k in ctx["variants"] if k.startswith("stress")), None)
        if stress:
            _ring(img, s0, ctx["variants"][stress]["robot_radius_px"], (0, 140, 255), 2, 26)
    return img


def legend(ctx, width):
    m = ctx["map_record"]
    pv = ctx["variants"].get(ctx["primary"], {}) if ctx["primary"] else {}
    hom = m["homography"]
    mline = "map prior: NOT USED (transfer not accepted)"
    if m["decision"]["accepted"]:
        mline = ("map transfer ACCEPTED: %d inliers, RMS %.2f px, hold-out median %.2f-%.2f px, v0.2 check_alignment %s shift %s px response %s" % (
            hom["n_inliers"], hom["inlier_residual"]["rms_px"], min(v["median_px"] for v in hom["holdout"].values()),
            max(v["median_px"] for v in hom["holdout"].values()), m["v02_check_alignment"]["status"], m["v02_check_alignment"]["shift_px"], m["v02_check_alignment"]["response"]))
    path = (pv.get("path") or {})
    lines = [
        ("[Qwen3-VL, STORED reply] proposed destination 'loading dock' (area_5): orange dashed box (no new semantic inference)", (255, 255, 255), C["qwen"]),
        ("[Grounding DINO + Qwen crop check] truck, conveyor, 6 cones (STORED) and mobile robot (this run, 1 crop check): white / light-blue boxes", (255, 255, 255), C["dino"]),
        ("[Python, STORED bay localization] painted loading bay from the dashed floor markings: green (NOT the placement area)", (255, 255, 255), C["bay"]),
        ("[v0.2 DRAFT map prior, hand-drawn, TRANSFERRED]  drivable area: blue | static obstacles: red | no-go: magenta   (MANUAL_MAP_PRIOR)", (255, 255, 255), C["trav"]),
        ("[Python, v0.2 NavMap] valid robot-centre region for the ASSUMED radius %s px: thin light-blue line; rings at the start: assumed (yellow), image-inscribed stress (orange)" % (pv.get("robot_radius_px", "-")), (255, 255, 255), C["valid"]),
        ("[Python, v0.2 goal_selector] approach goal (yellow dot, arrow = heading to the bay centre); [v0.2 A*] path (cyan), length %s px; white cross = ESTIMATED robot start" % path.get("length_px", "-"), (255, 255, 255), C["goal"]),
        (mline, (0, 255, 255), None),
        ("path checks: v0.2 validator valid=%s | independent (swept discs, tape pixels, goal) pass=%s | verdict is for the ASSUMED radius only" % (
            (pv.get("verdict") or {}).get("v02_validator"), (pv.get("verdict") or {}).get("independent")), (0, 255, 255), None),
        ("NOT VERIFIED: robot floor position/footprint | real robot radius (assumed) | gripper reach 180 px (assumed) | map polygons (DRAFT) | right boundary by pixels | placement surface", (120, 120, 255), None),
        ("This is an approach path to the bay edge only. No pickup, grasp, placement or shipping is claimed. Image space, not metric. No hardware command was made.", (120, 120, 255), None)]
    return panel(width, lines)


def overview(ctx, out):
    img = scene(ctx, ctx["primary"])
    cv2.imwrite(str(out / "01_final_overview.png"), np.vstack([img, legend(ctx, img.shape[1])]))
    return "01_final_overview.png"


def goal_zoom(ctx, out):
    img = scene(ctx, ctx["primary"], show_start=False)
    x0, y0, x1, y1, z = 1150, 0, 1900, 520, 2
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    _t(crop, "zoom 2x: Qwen box (orange dashed) | painted bay (green) | cones, truck (white) | drivable edge (blue) | approach goal + heading (yellow) | path end (cyan)", (6, 20), (0, 255, 255), 0.55)
    cv2.imwrite(str(out / "02_bay_goal_zoom.png"), crop)
    return "02_bay_goal_zoom.png"


def robot_zoom(ctx, out):
    img = scene(ctx, ctx["primary"], show_valid=False)
    x0, y0, x1, y1, z = 1480, 440, 1900, 880, 3
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    _t(crop, "zoom 3x: detected robot box (light-blue) | start estimates: white cross = outline centre, grey crosses = moved toward principal point | rings: assumed r (yellow), inscribed (orange)", (6, 20), (0, 255, 255), 0.5)
    cv2.imwrite(str(out / "03_robot_start_zoom.png"), crop)
    return "03_robot_start_zoom.png"


def variants_fig(ctx, out):
    img = ctx["image"].copy()
    _layers(img, ctx["spec"])
    cols = [(255, 255, 0), (0, 255, 0), (255, 0, 255), (0, 140, 255), (80, 80, 255)]
    rows = []
    for (k, v), col in zip(ctx["variants"].items(), cols):
        rt = ctx["runtime"][k]
        ok = rt["path"] and rt["path"]["status"] == "PATH_FOUND"
        if ok:
            _line(img, rt["path"]["waypoints_xy"], col, 3)
        if rt["goal"].get("pose"):
            _goal(img, rt["goal"], col)
        rows.append((f"{k}: goal={v['goal']['status']} path={'%s px' % rt['path']['length_px'] if ok else (rt['path'] or {}).get('status', '-')} | v0.2 validator={v['verdict']['v02_validator']} independent={v['verdict']['independent']}", col))
    _objects(img, ctx["objects"] + ([ctx["robot"]] if ctx["robot"] else []), labels=False)
    _dashed(img, ctx["qwen_box"], C["qwen"], 3)
    if ctx.get("bay_polygon"):
        _poly(img, ctx["bay_polygon"], C["bay"], 3)
    pn = panel(img.shape[1], [(t, (255, 255, 255), c) for t, c in rows] + [("variants differ only in robot radius (assumed vs image-inscribed stress) and in the estimated floor start; same map, objects and bay", (0, 255, 255), None)])
    cv2.imwrite(str(out / "05_variants.png"), np.vstack([img, pn]))
    return "05_variants.png"


def map_fig(ctx, out):
    img = ctx["image"].copy()
    ym = ctx["yellow"]
    img[ym] = (0, 255, 255)
    _layers(img, ctx["spec"])
    m = ctx["map_record"]
    rows = [("map prior transfer check: transferred DRAFT polygons on the E1 photo; yellow tape pixels of THIS photo are painted cyan-yellow", (255, 255, 255), None)]
    for c in m["decision"]["checks"]:
        rows.append((("PASS " if c["ok"] else "FAIL ") + c["check"] + ": " + str(c["detail"])[:150], (0, 255, 0) if c["ok"] else (80, 80, 255), None))
    if "edge_offsets_informational" in m:
        e = [x for x in m["edge_offsets_informational"] if x["zone"] == "Z02" and x.get("yellow_found_fraction", 0) >= 0.6]
        rows.append(("informational, DRAFT drawing + transfer error, Z02 edges that follow yellow tape: " + "; ".join(
            f"edge {x['edge']} median {x['median_abs_offset_px']} px" for x in e), (200, 200, 200), None))
    cv2.imwrite(str(out / "04_map_transfer_check.png"), np.vstack([img, panel(img.shape[1], rows)]))
    return "04_map_transfer_check.png"


def render_all(ctx, out):
    figs = []
    if ctx["spec"] is not None:
        figs.append(map_fig(ctx, out))
    if ctx["primary"] and ctx["runtime"].get(ctx["primary"]):
        figs += [overview(ctx, out), goal_zoom(ctx, out), robot_zoom(ctx, out), variants_fig(ctx, out)]
    return figs
