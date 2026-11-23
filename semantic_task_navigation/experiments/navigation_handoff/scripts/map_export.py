#!/usr/bin/env python3
"""STEP 2: static Nav2 map (map.pgm + map.yaml) from the reused DRAFT zones and E1 floor transform, plus coordinate validation.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/navigation_handoff/scripts/map_export.py          # new results/<time>_step2_map/ (never overwrites)

No model call, no GPU, no ROS. Reused unchanged: floor_transform.fit_floor_transform (task_plan_prototype), the transferred DRAFT map
(first_vlm_navigation_path), v0.2 check_alignment. Only the static zone layers go into the map: robot, cubes and pallets never do.
"""
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coordinate_transform as CT  # noqa: E402
from semantic_task_nav.config import code_fingerprint, sha256_file  # noqa: E402
from semantic_task_nav.navigation.map_builder import check_alignment  # noqa: E402

FREE, STATIC_OBSTACLE, NO_GO, NOT_DRIVABLE, UNKNOWN = 0, 1, 2, 3, 4
CODE_NAMES = {FREE: "FREE (inside drivable zone Z02, no obstacle)", STATIC_OBSTACLE: "STATIC_OBSTACLE (DRAFT staging zones)",
              NO_GO: "NO_GO (DRAFT pedestrian route / robotic workcell)", NOT_DRIVABLE: "NOT_DRIVABLE (workspace outside Z02, or a mixed boundary cell)",
              UNKNOWN: "UNKNOWN (outside the 3 x 2 mat workspace)"}
COLORS = {FREE: (175, 225, 175), STATIC_OBSTACLE: (50, 50, 200), NO_GO: (170, 60, 160), NOT_DRIVABLE: (65, 65, 65), UNKNOWN: (140, 125, 110)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------------------------------------------ map content
def zones_floor_cm(zmap, frame):
    return {layer: [{"id": z["id"], "polygon": frame.img_to_floor(z["polygon"]).tolist()} for z in zmap[layer]]
            for layer in ("traversable", "static_obstacles", "no_go")}


def rasterize(zones_cm, grid, workspace_cm, px_per_cm):
    """Per-cell layer codes. A cell is FREE only if all its sub-pixels are free, UNKNOWN only if all lie outside the workspace,
    otherwise it is occupied and carries the strongest reason present (obstacle > no-go > not drivable)."""
    sub = int(round(grid.res_cm * px_per_cm))
    h, w = grid.rows * sub, grid.cols * sub

    def fill(polys):
        m = np.zeros((h, w), np.uint8)
        for p in polys:
            q = (np.asarray(p, np.float64) - [grid.x0, grid.y0]) * px_per_cm
            cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
        return m.astype(bool)

    W, H = workspace_cm
    ws = fill([[[0, 0], [W, 0], [W, H], [0, H]]])
    trav = fill(z["polygon"] for z in zones_cm["traversable"]) & ws
    obst = fill(z["polygon"] for z in zones_cm["static_obstacles"]) & ws
    nogo = fill(z["polygon"] for z in zones_cm["no_go"]) & ws
    free = trav & ~obst & ~nogo
    cat = np.full((h, w), UNKNOWN, np.uint8)
    cat[ws] = NOT_DRIVABLE
    cat[nogo] = NO_GO
    cat[obst] = STATIC_OBSTACLE
    cat[free] = FREE
    blocks = cat.reshape(grid.rows, sub, grid.cols, sub).transpose(0, 2, 1, 3).reshape(grid.rows, grid.cols, sub * sub)
    all_free, all_unknown = (blocks == FREE).all(-1), (blocks == UNKNOWN).all(-1)
    nonfree = np.where(blocks == FREE, 255, blocks).min(-1)
    code = np.where(all_free, FREE, np.where(all_unknown, UNKNOWN, np.minimum(nonfree, NOT_DRIVABLE))).astype(np.uint8)
    hi_area_m2 = {"free": float(free.sum()) / px_per_cm ** 2 / 1e4, "workspace": float(ws.sum()) / px_per_cm ** 2 / 1e4}
    return code, hi_area_m2


def pgm_values(code, values):
    return np.where(code == FREE, values["free"], np.where(code == UNKNOWN, values["unknown"], values["occupied"])).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ files
def write_pgm(path, arr, comment):
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(f"P5\n# {comment}\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(arr, np.uint8).tobytes())


def read_pgm(path):
    """Independent binary PGM (P5, 8 bit) reader: does not reuse the writer."""
    data = Path(path).read_bytes()
    tokens, i = [], 0
    while len(tokens) < 4:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":
            while data[i:i + 1] not in (b"\n", b"\r"):
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        tokens.append(data[i:j].decode("ascii"))
        i = j
    i += 1                                              # exactly one whitespace byte after maxval
    magic, w, h, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    arr = np.frombuffer(data[i:i + w * h], np.uint8).reshape(h, w)
    return arr, {"magic": magic, "width": w, "height": h, "maxval": maxval, "payload_bytes": len(data) - i}


def write_map_yaml(path, image_name, mcfg, origin_m):
    o = ", ".join(repr(float(v)) for v in origin_m)
    Path(path).write_text(f"image: {image_name}\nmode: {mcfg['mode']}\nresolution: {mcfg['resolution_m']}\norigin: [{o}]\n"
                          f"negate: {mcfg['negate']}\noccupied_thresh: {mcfg['occupied_thresh']}\nfree_thresh: {mcfg['free_thresh']}\n",
                          encoding="utf-8")


def nav2_trinary(values, negate, occupied_thresh, free_thresh):
    """Nav2 map_server (Humble) loadMapFromFile, trinary mode: occ = 1 - v/255 (negate 0) -> 100 / 0 / -1."""
    shade = np.asarray(values, np.float64) / 255.0
    occ = shade if negate else 1.0 - shade
    return np.where(occ > occupied_thresh, 100, np.where(occ < free_thresh, 0, -1)).astype(np.int8)


# ------------------------------------------------------------------------------------------------ figures
def _text(img, s, org, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def color_layers(code):
    out = np.zeros(code.shape + (3,), np.uint8)
    for k, col in COLORS.items():
        out[code == k] = col
    return out


def render_map_preview(code, grid, frame, cfg, nodes_known_cm, path, S=5):
    ml, mt, mb, panel = 70, 30, 60, 560
    h, w = grid.rows * S, grid.cols * S
    canvas = np.full((mt + h + mb, ml + w + panel, 3), 30, np.uint8)
    canvas[mt:mt + h, ml:ml + w] = cv2.resize(color_layers(code), (w, h), interpolation=cv2.INTER_NEAREST)
    ox, oy = grid.origin_m[0], grid.origin_m[1]

    def m2c(p):
        p = np.asarray(p, np.float64).reshape(-1, 2)
        return np.c_[ml + (p[:, 0] - ox) / grid.resolution_m * S, mt + (grid.rows - (p[:, 1] - oy) / grid.resolution_m) * S]

    W, H = cfg["workspace_cm"]
    for xf in range(0, W + 1, 105):                                             # mat seams (dashed)
        a, b = m2c(frame.floor_to_map([[xf, 0], [xf, H]]))
        for t in np.arange(0, 1, 0.04):
            p, q = a + (b - a) * t, a + (b - a) * min(1, t + 0.02)
            cv2.line(canvas, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), (220, 220, 220), 1)
    for yf in range(0, H + 1, 105):
        a, b = m2c(frame.floor_to_map([[0, yf], [W, yf]]))
        for t in np.arange(0, 1, 0.03):
            p, q = a + (b - a) * t, a + (b - a) * min(1, t + 0.015)
            cv2.line(canvas, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), (220, 220, 220), 1)
    for v in np.arange(0.0, grid.cols * grid.resolution_m + ox + 1e-9, 0.5):     # ticks every 0.5 m
        x = int(round(m2c([[v, oy]])[0, 0]))
        cv2.line(canvas, (x, mt + h), (x, mt + h + 6), (255, 255, 255), 1)
        _text(canvas, f"{v:.1f}", (x - 12, mt + h + 22), scale=0.45)
    for v in np.arange(0.0, grid.rows * grid.resolution_m + oy + 1e-9, 0.5):
        y = int(round(m2c([[ox, v]])[0, 1]))
        cv2.line(canvas, (ml - 6, y), (ml, y), (255, 255, 255), 1)
        _text(canvas, f"{v:.1f}", (ml - 45, y + 5), scale=0.45)
    _text(canvas, "x_map [m] ->", (ml + w // 2 - 50, mt + h + 48), scale=0.5)
    _text(canvas, "y_map [m]", (5, mt - 10), scale=0.5)
    for k in nodes_known_cm:                                                    # mat-grid nodes (reference geometry, 105 cm)
        x, y = np.round(m2c(frame.floor_to_map([k]))[0]).astype(int)
        cv2.drawMarker(canvas, (x, y), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
    o, ex, ey = np.round(m2c([[0, 0], [0.4, 0], [0, 0.4]])).astype(int)
    cv2.arrowedLine(canvas, tuple(o), tuple(ex), (0, 0, 255), 3, tipLength=0.25)
    cv2.arrowedLine(canvas, tuple(o), tuple(ey), (0, 200, 0), 3, tipLength=0.25)
    cv2.circle(canvas, tuple(o), 6, (255, 255, 255), -1)
    _text(canvas, "+x", tuple(ex + [4, 18]), (0, 0, 255), 0.55, 2)
    _text(canvas, "+y", tuple(ey + [6, 4]), (0, 200, 0), 0.55, 2)
    _text(canvas, "map origin (0,0)", tuple(o + [8, -10]), scale=0.45)
    px = ml + w + 20
    lines = [("STATIC Nav2 map - DRAFT / PROVISIONAL", (0, 255, 255)),
             (f"{cfg['map_export']['map_id']}", (255, 255, 255)),
             (f"{grid.cols} x {grid.rows} cells, {grid.resolution_m} m/cell", (255, 255, 255)),
             (f"origin (lower-left) = [{grid.origin_m[0]:.2f}, {grid.origin_m[1]:.2f}, 0.0] m", (255, 255, 255)),
             ("frame: ROS map (right-handed, z up)", (255, 255, 255)),
             ("+y_map = away from the camera", (255, 255, 255)),
             ("", None)]
    lines += [(f"{name.split(' (')[0]}: {'free 254' if k == FREE else 'unknown 205' if k == UNKNOWN else 'occupied 0'}", COLORS[k]) for k, name in CODE_NAMES.items()]
    lines += [("", None), ("yellow x: mat-grid nodes (105 cm, reference)", (0, 255, 255)), ("dashed: mat seams", (220, 220, 220)), ("", None),
              ("NOT in this map: robot, cubes, pallets", (255, 255, 255)), ("zones: hand-drawn DRAFT, not human-confirmed", (255, 255, 255)),
              ("floor transform: PROVISIONAL_UNVALIDATED", (255, 255, 255))]
    for i, (s, col) in enumerate(lines):
        if col is not None:
            _text(canvas, s, (px, mt + 20 + 24 * i), col, 0.5)
    cv2.imwrite(str(path), canvas)


def render_map_on_photo(img, code, grid, frame, ft, path):
    """Map cells warped back onto the reference photo through the inverse floor transform (same H as the map)."""
    T = np.array([[1 / grid.res_cm, 0, -grid.x0 / grid.res_cm - 0.5], [0, 1 / grid.res_cm, -grid.y0 / grid.res_cm - 0.5], [0, 0, 1]])
    M = T @ frame.H                                                             # photo px -> map-cell px
    lay = cv2.warpPerspective(color_layers(code), M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    known = cv2.warpPerspective(np.where(code == UNKNOWN, 0, 255).astype(np.uint8), M, (img.shape[1], img.shape[0]),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    out = img.copy()
    out[known] = (0.55 * img[known] + 0.45 * lay[known]).astype(np.uint8)
    o, ex, ey = np.round(frame.map_to_img([[0, 0], [0.4, 0], [0, 0.4]])).astype(int)
    cv2.arrowedLine(out, tuple(o), tuple(ex), (0, 0, 255), 4, tipLength=0.2)
    cv2.arrowedLine(out, tuple(o), tuple(ey), (0, 200, 0), 4, tipLength=0.2)
    cv2.circle(out, tuple(o), 8, (255, 255, 255), -1)
    _text(out, "map (0,0)", tuple(o + [10, -12]), scale=0.7, thick=2)
    _text(out, "+x_map", tuple(ex + [6, 22]), (0, 0, 255), 0.7, 2)
    _text(out, "+y_map", tuple(ey + [8, 0]), (0, 200, 0), 0.7, 2)
    for n in ft["nodes"]:
        if not n["used"]:
            continue
        c, r = (int(v) for v in n["node"].split(","))
        p = np.round(n["xy_px"]).astype(int)
        q = np.round(frame.floor_to_img([[105 * c, 105 * r]])[0]).astype(int)
        cv2.circle(out, tuple(p), 7, (0, 255, 255), 2)
        cv2.drawMarker(out, tuple(q), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        _text(out, f"({c},{r}) {n['fit_residual_cm']:.1f}cm", tuple(p + [10, 22]), (0, 255, 255), 0.55, 1)
    lines = ["STEP 2 check: static map cells projected onto the reference photo (photo_5 = E1) through the SAME floor transform",
             "green: free | red: static obstacle | purple: no-go | dark: not drivable | arrows: ROS map axes (+x red, +y green, 0.4 m)",
             "yellow circle: mat-seam node found in the photo, magenta x: 105 cm grid position projected back; label = fit residual",
             "DRAFT zones (not human-confirmed), PROVISIONAL floor transform. Robot, cubes, pallets are NOT in the static map."]
    for i, s in enumerate(lines):
        _text(out, s, (12, 28 + 26 * i), scale=0.6)
    cv2.imwrite(str(path), out)


# ------------------------------------------------------------------------------------------------ main
def file_hashes(paths):
    return {str(p): sha256_file(p) for p in paths}


def main():
    cfg = CT.load_config()
    mcfg = cfg["map_export"]
    reused_files = [CT.resolve(cfg["reuse"][k]) for k in ("floor_transform_config", "stored_floor_transform", "draft_map_E1")] + \
                   [CT.EXP.parent / "task_plan_prototype" / "floor_transform.py"]
    before = {"v02_code_fingerprint": code_fingerprint()["sha256"], "reused_files": file_hashes(reused_files)}
    checks = []

    def check(cid, item, ok, detail, knowledge="CONFIRMED", status=None):
        checks.append({"id": cid, "item": item, "status": status or ("PASS" if ok else "FAIL"), "knowledge": knowledge, "detail": detail})

    # ---- photos by hash, reference photo, camera alignment of all 8
    photos = CT.resolve_photos(cfg)
    check("photos_resolved_by_hash", "8 photos identified by sha256 (file names changed)", len(photos) == len(cfg["photos"]) == 8,
          {p["name"]: {"file": Path(p["path"]).name, "earlier_id": p["earlier_id"]} for p in photos})
    ref = next(p for p in photos if p["name"] == cfg["reference_photo"])
    load = lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # noqa: E731
    ref_img = load(ref["path"])
    align_spec = {"reference_image": ref["path"], "alignment_check": cfg["alignment_check"]}
    for p in photos:
        p["alignment_vs_reference"] = check_alignment(align_spec, load(p["path"]), Path)
    check("camera_alignment_8_photos", "every photo has the camera pose of the reference photo (v0.2 check_alignment, phase correlation)",
          all(p["alignment_vs_reference"]["status"] == "ALIGNED" for p in photos),
          {p["name"]: {k: p["alignment_vs_reference"].get(k) for k in ("status", "shift_px", "response")} for p in photos})

    # ---- floor transform (reused) must reproduce the stored one
    ft, floor_cfg = CT.fit_reference_transform(cfg, ref_img)
    stored = json.loads(CT.resolve(cfg["reuse"]["stored_floor_transform"]).read_text(encoding="utf-8"))["floor_transform"]
    used = [n for n in ft["nodes"] if n["used"]]
    probe = [n["xy_px"] for n in used]
    dev = float(np.abs(CT.FT.project(ft["H_img_to_floor_cm"], probe) - CT.FT.project(stored["H_img_to_floor_cm"], probe)).max()) if ft["accepted"] else None
    check("floor_transform_reproduced", "E1 mat-seam transform recomputed with the reused function equals the stored one",
          ft["accepted"] and dev is not None and dev < 1e-6, {"accepted": ft["accepted"], "status": ft.get("status"), "max_floor_deviation_cm_at_nodes": dev})
    if not ft["accepted"]:
        raise SystemExit("floor transform not accepted: no map is written")
    frame = CT.MapFrame(ft["H_img_to_floor_cm"], cfg["map_frame"]["origin_floor_cm"])

    # ---- reference points: the 105 cm mat grid (user-stated geometry; nodes are AI-seeded, refined on image evidence)
    W, H = cfg["workspace_cm"]
    nodes_ref = []
    for n in used:
        c, r = (int(v) for v in n["node"].split(","))
        known_map = frame.floor_to_map([[105 * c, 105 * r]])[0]
        meas_map = frame.img_to_map([n["xy_px"]])[0]
        nodes_ref.append({"node": n["node"], "image_px": [round(v, 2) for v in n["xy_px"]], "known_map_m": [round(float(v), 4) for v in known_map],
                          "projected_map_m": [round(float(v), 4) for v in meas_map], "error_cm": round(float(np.linalg.norm(meas_map - known_map) * 100), 2),
                          "leave_one_out_cm": n["leave_one_out_cm"]})
    errs = np.array([r["error_cm"] for r in nodes_ref])
    check("reference_points_mat_nodes", "image -> map error at the mat-grid nodes (known 105 cm geometry; these nodes are USED in the fit)",
          True, {"n": len(nodes_ref), "rms_cm": round(float(np.sqrt((errs ** 2).mean())), 2), "max_cm": round(float(errs.max()), 2),
                 "leave_one_out_max_cm": max(r["leave_one_out_cm"] for r in nodes_ref), "nodes": nodes_ref},
          knowledge="PROVISIONAL: not independent of the fit; no human-measured reference point exists")
    check("reference_independent_seams", "inner mat seams NOT used in the fit, mapped to the floor, deviation from the known 105/210 cm lines",
          ft["seam_max_abs_dev_cm"] <= floor_cfg["accept"]["max_seam_dev_cm"],
          {"seams": ft["independent_seam_check"], "max_abs_dev_cm": ft["seam_max_abs_dev_cm"], "declared_limit_cm": floor_cfg["accept"]["max_seam_dev_cm"]},
          knowledge="PROVISIONAL")

    # ---- frame conventions: origin, axes, handedness, yaw
    corners = frame.floor_to_map([[0, H], [W, H], [W, 0], [0, 0]])
    exp = np.array([[0, 0], [W / 100, 0], [W / 100, H / 100], [0, H / 100]])
    check("map_origin_and_extent", "mat corners in the map frame: node(0,2) -> (0,0), node(3,2) -> (3.15,0), node(3,0) -> (3.15,2.10), node(0,0) -> (0,2.10)",
          np.allclose(corners, exp), {"got_m": corners.round(6).tolist(), "expected_m": exp.tolist()})
    ctr_img = frame.floor_to_img([[W / 2, H / 2]])[0]
    d_right = frame.img_to_map([ctr_img + [20, 0]])[0] - frame.img_to_map([ctr_img])[0]
    d_up = frame.img_to_map([ctr_img + [0, -20]])[0] - frame.img_to_map([ctr_img])[0]
    yaw_right, yaw_up = np.degrees(np.arctan2(d_right[1], d_right[0])), np.degrees(np.arctan2(d_up[1], d_up[0]))
    cross = float(d_right[0] * d_up[1] - d_right[1] * d_up[0])
    far, near = frame.floor_to_img([[W / 2, 20], [W / 2, H - 20]])
    check("axes_and_handedness", "at the workspace centre: image right ~ +x_map (yaw ~0), image up (away from the camera) ~ +y_map (yaw ~+90), right-handed",
          abs(yaw_right) < 15 and abs(yaw_up - 90) < 15 and cross > 0 and far[1] < near[1],
          {"yaw_map_of_image_right_deg": round(float(yaw_right), 2), "yaw_map_of_image_up_deg": round(float(yaw_up), 2),
           "z_of_right_x_up": round(cross, 6), "far_side_is_higher_in_image": bool(far[1] < near[1]),
           "note": "the small deviation from 0 / 90 deg is the tilt of the mat grid in the photo (perspective); map axes follow the mats"})
    yaw_tests = {d: CT.MapFrame.yaw_floor_to_map(d) for d in (0, 90, -90, 180, 45)}
    check("yaw_convention", "yaw_map = -yaw_floor (floor yaw counts from image right towards image down)",
          yaw_tests == {0: 0.0, 90: -90.0, -90: 90.0, 180: 180.0, 45: -45.0}, yaw_tests)

    # ---- grid and rasterisation
    pad = mcfg["pad_cm"]
    grid = CT.MapGrid(frame, [-pad, -pad, W + pad, H + pad], mcfg["resolution_m"])
    rr, cc = np.mgrid[0:grid.rows, 0:grid.cols]
    via_floor = frame.floor_to_map(np.c_[np.ravel(grid.cell_center_floor(rr, cc)[0]), np.ravel(grid.cell_center_floor(rr, cc)[1])])
    nav2 = np.c_[np.ravel(grid.cell_center_map(rr, cc)[0]), np.ravel(grid.cell_center_map(rr, cc)[1])]
    back_r, back_c = grid.map_to_cell(nav2[:, 0], nav2[:, 1])
    check("grid_convention", "every cell centre: floor route == Nav2 origin/row-flip route, and map -> cell returns the same cell",
          np.abs(via_floor - nav2).max() < 1e-9 and (back_r == rr.ravel()).all() and (back_c == cc.ravel()).all(),
          {"max_centre_difference_m": float(np.abs(via_floor - nav2).max()), **grid.to_dict()})

    zmap = json.loads(CT.resolve(cfg["reuse"]["draft_map_E1"]).read_text(encoding="utf-8"))
    zones = zones_floor_cm(zmap, frame)
    code, hi_area = rasterize(zones, grid, cfg["workspace_cm"], mcfg["raster_px_per_cm"])
    pgm = pgm_values(code, mcfg["values"])
    cell_m2 = grid.resolution_m ** 2
    counts = {CODE_NAMES[k].split(" (")[0]: int((code == k).sum()) for k in CODE_NAMES}
    free_cells_m2 = counts["FREE"] * cell_m2
    check("layers", "cell counts per layer; free area of the conservative cells vs the 0.25 cm raster of the same polygons", True,
          {"cell_counts": counts, "free_area_cells_m2": round(free_cells_m2, 4), "free_area_raster_m2": round(hi_area["free"], 4),
           "free_area_lost_by_cell_rule_pct": round(100 * (1 - free_cells_m2 / hi_area["free"]), 2),
           "workspace_area_m2": round(hi_area["workspace"], 4), "zones": {k: [z["id"] for z in v] for k, v in zones.items()}})
    near_unknown = np.zeros_like(code, bool)
    u = code == UNKNOWN
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            near_unknown |= np.roll(np.roll(u, dr, 0), dc, 1)
    check("free_sealed_from_unknown", "no FREE cell touches an UNKNOWN cell (8-neighbourhood): a planner that allows unknown space cannot leave the map",
          not (near_unknown & (code == FREE)).any(), {"free_cells_touching_unknown": int((near_unknown & (code == FREE)).sum())})
    check("static_map_has_no_movable_objects", "only DRAFT zone layers are written; robot, cubes and pallets are not", True,
          {"sources": [zmap.get("map_id"), "layers traversable/static_obstacles/no_go"], "movable_objects_written": []})

    # ---- write the map files
    stamp = datetime.now().astimezone()
    out = CT.resolve(cfg["results_dir"]) / (stamp.strftime("%Y%m%d_%H%M%S") + "_step2_map")
    (out / "maps").mkdir(parents=True, exist_ok=False)
    (out / "figures").mkdir()
    write_pgm(out / "maps/map.pgm", pgm, f"{mcfg['map_id']} {grid.resolution_m} m/pix, DRAFT zones, PROVISIONAL floor transform")
    write_map_yaml(out / "maps/map.yaml", "map.pgm", mcfg, grid.origin_m)
    np.save(out / "maps/map_layers.npy", code)

    # ---- format validation (independent re-read of the written files)
    arr, hdr = read_pgm(out / "maps/map.pgm")
    y = yaml.safe_load((out / "maps/map.yaml").read_text(encoding="utf-8"))
    req = {"image": str, "mode": str, "resolution": float, "origin": list, "negate": int, "occupied_thresh": float, "free_thresh": float}
    yaml_ok = all(k in y and isinstance(y[k], t) for k, t in req.items()) and len(y["origin"]) == 3 and (out / "maps" / y["image"]).exists()
    check("yaml_format", "map.yaml has the Nav2 map_server keys and types; image path resolves next to the yaml", yaml_ok, y)
    check("pgm_format", "map.pgm re-read by an independent parser: P5, maxval 255, size = grid, payload = w*h, values in {0, 205, 254}",
          hdr["magic"] == "P5" and hdr["maxval"] == 255 and (hdr["height"], hdr["width"]) == (grid.rows, grid.cols)
          and hdr["payload_bytes"] == grid.rows * grid.cols and set(np.unique(arr).tolist()) <= {0, 205, 254} and np.array_equal(arr, pgm),
          {**hdr, "values": {int(v): int(n) for v, n in zip(*np.unique(arr, return_counts=True))}})
    try:
        from PIL import Image
        with Image.open(out / "maps/map.pgm") as im:
            pil = np.asarray(im)
        check("pgm_second_reader", "PIL reads the same pixels", np.array_equal(pil, arr), {"mode": "L", "shape": list(pil.shape)})
    except ImportError as e:
        check("pgm_second_reader", "PIL reads the same pixels", False, str(e), status="NOT_STARTED")
    occ = nav2_trinary(arr, y["negate"], y["occupied_thresh"], y["free_thresh"])
    expect = np.where(code == FREE, 0, np.where(code == UNKNOWN, -1, 100))
    check("nav2_trinary_interpretation", "Nav2 trinary rule applied to the file gives FREE->0, UNKNOWN->-1, every other layer->100 (lethal)",
          np.array_equal(occ, expect), {"free": int((occ == 0).sum()), "occupied": int((occ == 100).sum()), "unknown": int((occ == -1).sum())})
    extent_ok = abs(hdr["width"] * y["resolution"] - grid.cols * grid.res_cm / 100) < 1e-9 and abs(hdr["height"] * y["resolution"] - grid.rows * grid.res_cm / 100) < 1e-9
    check("resolution_vs_size", "PGM size x resolution = map extent in metres", extent_ok,
          {"width_m": round(hdr["width"] * y["resolution"], 6), "height_m": round(hdr["height"] * y["resolution"], 6)})
    check("nav2_map_server_load", "ros2 run nav2_map_server map_server with this yaml", False,
          "not run: nav2 is not installed on this PC; user decision 2026-11-19 = format validation only (load test on request)", knowledge="UNKNOWN",
          status="NOT_STARTED")

    # ---- figures
    nodes_known = [[105 * c, 105 * r] for c in range(4) for r in range(3)]
    render_map_preview(code, grid, frame, cfg, nodes_known, out / "maps/map_preview.png")
    render_map_on_photo(ref_img, code, grid, frame, ft, out / "figures/map_on_photo_reference.png")

    # ---- records
    registry = [{k: p[k] for k in ("name", "current_file", "path", "sha256", "earlier_id", "earlier_file", "earlier_role", "file_name_matches_config")}
                | {"alignment_vs_reference": p["alignment_vs_reference"]} for p in photos]
    (out / "photo_registry.json").write_text(json.dumps({"reference_photo": cfg["reference_photo"], "identity": "sha256 (file names changed)",
                                                         "photos": registry}, indent=1, ensure_ascii=False), encoding="utf-8")
    frames_rec = {"frames": {"undistorted_px": "pixels of the photos as stored (already undistorted, newK = K); never undistort again",
                             "floor_cm": "PROVISIONAL E1 mat-seam floor plane: origin mat node (0,0) far-left, +x image right, +y towards the camera",
                             "map_m": "ROS2 map frame of map.yaml (REP-103 right-handed, z up): origin mat node (0,2), +y away from the camera",
                             "grid_rc": "map.pgm row/col; row 0 = largest y_map", "odom/base_link": "NOT defined here; map->odom comes from the robot's localisation"},
                  "transform": frame.to_dict(), "grid": grid.to_dict(), "layer_codes": CODE_NAMES,
                  "floor_transform_record": {k: ft[k] for k in ("method", "status", "fit_rms_cm", "fit_max_cm", "leave_one_out_max_cm",
                                                                 "seam_max_abs_dev_cm", "footprint_margin_cm", "known_limits")},
                  "status": {"floor_transform": "PROVISIONAL_UNVALIDATED", "zones": zmap["review_status"]}}
    (out / "coordinate_frames.json").write_text(json.dumps(frames_rec, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "config_snapshot.yaml").write_text(CT.CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    after = {"v02_code_fingerprint": code_fingerprint()["sha256"], "reused_files": file_hashes(reused_files)}
    photos_after = {p["name"]: sha256_file(p["path"]) == p["sha256"] for p in photos}
    check("environment_unchanged", "v0.2 code, reused files and photos have the same sha256 after the run",
          before == after and all(photos_after.values()), {"v02_code_fingerprint": after["v02_code_fingerprint"], "photos_unchanged": photos_after})
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    (out / "validation.json").write_text(json.dumps({"summary": {"PASS": sum(c["status"] == "PASS" for c in checks), "FAIL": n_fail,
                                                                 "NOT_STARTED": sum(c["status"] == "NOT_STARTED" for c in checks)},
                                                     "checks": checks}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    outputs = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {"experiment_id": cfg["experiment_id"], "step": "STEP2_map_export", "run_id": out.name, "created": stamp.isoformat(timespec="seconds"),
                "git_commit": None, "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
                "code": file_hashes([Path(__file__).resolve(), CT.HERE / "coordinate_transform.py"]), "config_sha256": sha256_file(CT.CONFIG_PATH),
                "inputs": {"photos": {p["name"]: p["sha256"] for p in photos}, "reused_files": before["reused_files"],
                           "v02_code_fingerprint": before["v02_code_fingerprint"]},
                "hardware_condition": "offline; no model, GPU, ROS or robot", "outputs_sha256": {str(p.relative_to(out)): sha256_file(p) for p in outputs}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"run_dir": str(out), "checks": {c["id"]: c["status"] for c in checks}}, indent=1))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
