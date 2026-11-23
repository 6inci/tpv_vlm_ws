#!/usr/bin/env python3
"""STEP 4-A: read-only survey of the 8 photos (no path planning, no model call, nothing existing is changed).

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/navigation_handoff/scripts/survey_photos_step4a.py          # new results/<time>_step4a_photo_survey/

Per photo: camera alignment (v0.2 check_alignment), robot-window shift against photo_5 (phase correlation), median-background change blobs,
the declared object inventory (config step4a, AI visual reading, USER_CONFIRMATION_REQUIRED) with map coordinates, distance outside the drivable
zone, overlap with free cells, straight-line obstacle crossings and the free run along the approach axis. Difficulty is a rule-based PRELIMINARY
judgement from these facts; no A* is run and no path is drawn.
"""
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coordinate_transform as CT  # noqa: E402
import map_export as ME  # noqa: E402
import object_positions as OP  # noqa: E402
from semantic_task_nav.config import code_fingerprint, sha256_file  # noqa: E402
from semantic_task_nav.navigation.map_builder import check_alignment  # noqa: E402

LAYER = {k: v.split(" (")[0] for k, v in ME.CODE_NAMES.items()}
LATERAL_NEED = CT.load_config()["step4a_rules"]["on_axis_lateral_need_cm"]
DEPTH = CT.load_config()["step4a_rules"]["lateral_check_depth_cm"]


class Ctx:
    def __init__(self, cfg):
        self.frame, fc, self.map_run = OP.load_step2_frame(cfg)
        g = fc["grid"]
        self.grid = CT.MapGrid(self.frame, g["extent_floor_cm"], g["resolution_m"])
        self.layers = np.load(self.map_run / "maps/map_layers.npy")
        self.free = self.layers == ME.FREE
        self.dt_out = cv2.distanceTransform(np.pad((~self.free).astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        zmap = json.loads(CT.resolve(cfg["reuse"]["draft_map_E1"]).read_text(encoding="utf-8"))
        self.zmap = zmap
        self.zones_cm = ME.zones_floor_cm(zmap, self.frame)

    def cell(self, floor):
        return int((floor[1] - self.grid.y0) // self.grid.res_cm), int((floor[0] - self.grid.x0) // self.grid.res_cm)

    def layer_at(self, floor):
        r, c = self.cell(floor)
        return LAYER[int(self.layers[r, c])] if 0 <= r < self.grid.rows and 0 <= c < self.grid.cols else "OUTSIDE_MAP"

    def zone_at(self, floor):
        for kind in ("static_obstacles", "no_go"):
            for z in self.zones_cm[kind]:
                if cv2.pointPolygonTest(np.float32(z["polygon"]), (float(floor[0]), float(floor[1])), False) >= 0:
                    return z["id"]
        return None


def box_facts(ctx, box, robot_floor, ncm):
    x0, y0, x1, y1 = box
    centre_img = [(x0 + x1) / 2, (y0 + y1) / 2]
    f = ctx.frame.img_to_floor([centre_img])[0]
    m = ctx.frame.floor_to_map([f])[0]
    r, c = ctx.cell(f)
    if not (0 <= r < ctx.grid.rows and 0 <= c < ctx.grid.cols):
        return None                                                        # beyond the mapped area (floor around the mats)
    outside_cm = float(ctx.dt_out[r, c]) * ctx.grid.res_cm if not ctx.free[r, c] else 0.0
    corners = ctx.frame.img_to_floor([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    msk = np.zeros_like(ctx.free, np.uint8)
    pc = np.round((corners - [ctx.grid.x0, ctx.grid.y0]) / ctx.grid.res_cm - 0.5).astype(np.int32)
    cv2.fillPoly(msk, [pc], 1)
    on_free = int((msk.astype(bool) & ctx.free).sum())
    # approach axis: direction from the target to the nearest free cell
    fr, fcl = np.nonzero(ctx.free)
    k = int(np.argmin((fr - r) ** 2 + (fcl - c) ** 2))
    dr, dc = fr[k] - r, fcl[k] - c
    vertical = abs(dr) >= abs(dc)
    line = ctx.free[:, c] if vertical else ctx.free[r, :]
    pos = fr[k] if vertical else fcl[k]
    a = b = pos
    while a - 1 >= 0 and line[a - 1]:
        a -= 1
    while b + 1 < len(line) and line[b + 1]:
        b += 1
    side = ("below the drivable zone (camera side)" if dr < 0 else "above the drivable zone (truck side)") if vertical else \
           ("left of the drivable zone" if dc > 0 else "right of the drivable zone")
    facing = (270.0 if dr < 0 else 90.0) if vertical else (180.0 if dc > 0 else 0.0)
    crossed = segment_crossings(ctx, robot_floor, f)
    # lateral room at the approach: free run across the approach axis, in the line of the nearest free cell
    tgt = c if vertical else r
    step_in = int(np.sign(dr)) if vertical else int(np.sign(dc))            # from the boundary line into the drivable zone
    rooms = []
    for j in range(int(math.ceil(DEPTH / ctx.grid.res_cm))):
        line_idx = (fr[k] if vertical else fcl[k]) + j * step_in
        across = ctx.free[line_idx, :] if vertical else ctx.free[:, line_idx]
        if not across[tgt]:
            rooms.append(-1.0)
            continue
        lo = hi = tgt
        while lo - 1 >= 0 and across[lo - 1]:
            lo -= 1
        while hi + 1 < len(across) and across[hi + 1]:
            hi += 1
        rooms.append((min(tgt - lo, hi - tgt) + 0.5) * ctx.grid.res_cm)
    lateral_room_cm = min(rooms)
    return {"box_px": list(box), "centre_px": [round(v, 1) for v in centre_img], "map_m": [round(float(m[0]), 3), round(float(m[1]), 3)],
            "layer_at_centre": ctx.layer_at(f), "outside_drivable_cm": round(outside_cm, 1), "footprint_cells_on_free": on_free,
            "approach_side": side, "facing_needed_map_deg": facing, "approach_axis": "y" if vertical else "x",
            "free_run_on_approach_axis_cm": int((b - a + 1) * ctx.grid.res_cm), "narrow_approach": (b - a + 1) * ctx.grid.res_cm < ncm,
            "straight_line_from_robot_crosses": crossed, "lateral_room_at_approach_cm": round(lateral_room_cm, 1),
            "on_axis_approach_doubtful": lateral_room_cm < LATERAL_NEED, "status": "USER_CONFIRMATION_REQUIRED"}


def segment_crossings(ctx, p_floor, q_floor):
    """Mapped static obstacles / no-go zones crossed by the straight segment, except the zones that contain its end points."""
    ends = {ctx.zone_at(p_floor), ctx.zone_at(q_floor)}
    n = int(math.dist(p_floor, q_floor)) + 1
    out = []
    for t in np.linspace(0, 1, n):
        z = ctx.zone_at(np.asarray(p_floor) + (np.asarray(q_floor) - np.asarray(p_floor)) * t)
        if z and z not in out and z not in ends:
            out.append(z)
    return out


def judge(rec):
    """Rule-based PRELIMINARY difficulty from the measured facts (A* not run)."""
    a = rec["cube"]
    start_yaw = 180.0
    notes, score = [], 0
    if a["straight_line_from_robot_crosses"]:
        notes.append(f"A: straight line blocked by {'/'.join(a['straight_line_from_robot_crosses'])} -> detour, several XY turns")
        score += 2
    else:
        notes.append("A: straight line not blocked by mapped obstacles")
    if abs(CT.wrap_deg(a["facing_needed_map_deg"] - start_yaw)) > 1:
        notes.append(f"A: body rotation {start_yaw:.0f} -> {a['facing_needed_map_deg']:.0f} deg for the gripper")
        score += 1
    if a["narrow_approach"]:
        notes.append(f"A: narrow approach ({a['free_run_on_approach_axis_cm']} cm free run)")
        score += 1
    if a["on_axis_approach_doubtful"]:
        notes.append(f"A: only {a['lateral_room_at_approach_cm']} cm lateral room (< {LATERAL_NEED}) -> facing approach likely impossible")
    if a["layer_at_centre"] == "NO_GO":
        notes.append("A: object inside a no-go zone (gripper reaches into it)")
    for i, b in enumerate(rec["pallets_empty"]):
        if b["straight_from_cube_crosses"]:
            notes.append(f"B{i + 1}: straight line from A blocked by {'/'.join(b['straight_from_cube_crosses'])}")
            score += 1
        if abs(CT.wrap_deg(b["facing_needed_map_deg"] - a["facing_needed_map_deg"])) > 1:
            notes.append(f"B{i + 1}: body rotation {a['facing_needed_map_deg']:.0f} -> {b['facing_needed_map_deg']:.0f} deg between grasp and place")
            score += 1
        if b["narrow_approach"]:
            notes.append(f"B{i + 1}: narrow approach ({b['free_run_on_approach_axis_cm']} cm)")
            score += 1
        if b["on_axis_approach_doubtful"]:
            notes.append(f"B{i + 1}: only {b['lateral_room_at_approach_cm']} cm lateral room (< {LATERAL_NEED}) -> facing approach likely impossible")
        if b["footprint_cells_on_free"]:
            notes.append(f"B{i + 1}: pallet lies partly on the drivable zone ({b['footprint_cells_on_free']} free cells) -> per-photo obstacle")
    if not rec["pallets_empty"]:
        notes.append("B: no empty pallet in this photo -> placement target must be given by the user")
    if a["footprint_cells_on_free"]:
        notes.append(f"A: object lies partly on the drivable zone ({a['footprint_cells_on_free']} free cells) -> per-photo obstacle")
    level = "HIGH" if score >= 4 else "MEDIUM" if score >= 2 else "LOW"
    return {"level_preliminary": level, "score": score, "notes": notes, "basis": "measured facts only; A* NOT run"}


def render(cfg, ctx, photos, recs, out):
    tiles = []
    for p, rec in zip(photos, recs):
        im = cv2.imread(p["path"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        for kind, col in (("traversable", (0, 200, 0)), ("static_obstacles", (0, 0, 255)), ("no_go", (200, 0, 200))):
            for z in ctx.zmap[kind]:
                cv2.polylines(im, [np.round(z["polygon"]).astype(np.int32)], True, col, 3)
        r = rec["robot"]
        cv2.drawMarker(im, tuple(int(v) for v in r["base_link_px_estimate"]), (255, 255, 0), cv2.MARKER_CROSS, 40, 5)
        cv2.arrowedLine(im, tuple(int(v) for v in r["base_link_px_estimate"]), (int(r["base_link_px_estimate"][0]) - 110, int(r["base_link_px_estimate"][1])),
                        (0, 0, 255), 6, tipLength=0.3)
        b = rec["cube"]["box_px"]
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 0, 255), 5)
        cv2.putText(im, "A?", (b[0] - 10, b[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 5)
        for i, pb in enumerate(rec["pallets_empty"]):
            bb = pb["box_px"]
            cv2.rectangle(im, tuple(bb[:2]), tuple(bb[2:]), (255, 120, 0), 5)
            cv2.putText(im, f"B{i + 1}?", (bb[0] - 10, bb[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 120, 0), 5)
        t = cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (960, 36), (0, 0, 0), -1)
        j = rec["difficulty"]
        cv2.putText(t, f"{p['current_file']} ({p['earlier_id']})  difficulty {j['level_preliminary']} (score {j['score']}, A* not run)", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y = 520
        for s in reversed(j["notes"][:5]):
            cv2.putText(t, s[:95], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 4)
            cv2.putText(t, s[:95], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y -= 18
        tiles.append(t)
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 8, 2)])
    legend = np.full((70, sheet.shape[1], 3), 25, np.uint8)
    cv2.putText(legend, "green: drivable zone (DRAFT) | red: static obstacles | purple: no-go | cyan +: robot (estimate) red arrow: gripper side | "
                "red box A?: red cube | blue box B?: empty pallet", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(legend, "All object positions: AI visual reading, USER_CONFIRMATION_REQUIRED. No path is drawn: A* has NOT been run in STEP 4-A.",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.imwrite(str(out / "comparison_8photos.png"), np.vstack([legend, sheet]))


def main():
    cfg = CT.load_config()
    s4 = cfg["step4a"]
    before = code_fingerprint()["sha256"]
    photos = CT.resolve_photos(cfg)
    ctx = Ctx(cfg)
    imgs = [cv2.imread(p["path"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) for p in photos]
    ref_photo = next(i for i, p in enumerate(photos) if p["name"] == s4["robot_reference"]["photo"])
    x0, y0, x1, y1 = s4["robot_reference"]["window_xyxy"]
    ref_win = cv2.cvtColor(imgs[ref_photo][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = np.median(np.stack(imgs), axis=0).astype(np.uint8)
    lab_bg = cv2.cvtColor(cv2.GaussianBlur(bg, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    align_spec = {"reference_image": photos[ref_photo]["path"], "alignment_check": cfg["alignment_check"]}
    recs = []
    for p, im in zip(photos, imgs):
        win = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(ref_win, win, cv2.createHanningWindow(ref_win.shape[::-1], cv2.CV_32F))
        base = np.add(s4["robot_reference"]["wheel_centre_px"], [dx, dy])
        auto = OP.robot_auto_estimate(cfg, p)
        robot_floor = ctx.frame.img_to_floor([base])[0]
        d = np.linalg.norm(cv2.cvtColor(cv2.GaussianBlur(im, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32) - lab_bg, axis=2)
        msk = cv2.morphologyEx(cv2.morphologyEx((d > s4["change_detection"]["lab_distance"]).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
                               cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(msk)
        blobs = [[int(st[k][0]), int(st[k][1]), int(st[k][0] + st[k][2]), int(st[k][1] + st[k][3])] for k in range(1, n) if st[k][4] >= s4["change_detection"]["min_area_px"]]
        inv = s4["inventory"][p["name"]]

        def supported(box):
            bx = np.array(box)
            return any(min(bx[2], b[2]) > max(bx[0], b[0]) and min(bx[3], b[3]) > max(bx[1], b[1]) for b in blobs)
        cube = box_facts(ctx, inv["cube"], robot_floor, s4["narrow_corridor_cm"])
        cube["supported_by_change_detection"] = supported(inv["cube"])
        cube_floor = ctx.frame.img_to_floor([cube["centre_px"]])[0]
        pals = []
        for pb in inv["pallet_empty"]:
            f = box_facts(ctx, pb, robot_floor, s4["narrow_corridor_cm"])
            f["supported_by_change_detection"] = supported(pb)
            f["straight_from_cube_crosses"] = segment_crossings(ctx, cube_floor, ctx.frame.img_to_floor([f["centre_px"]])[0])
            pals.append(f)
        free_blobs = []
        for b in blobs:                                                    # changed regions whose footprint touches free cells (robot excluded)
            if b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0:
                continue
            ff = box_facts(ctx, b, robot_floor, s4["narrow_corridor_cm"])
            if ff and ff["footprint_cells_on_free"]:
                free_blobs.append({"box_px": b, "map_m": ff["map_m"], "free_cells": ff["footprint_cells_on_free"]})
        rec = {"photo": p["name"], "file": p["current_file"], "sha256": p["sha256"], "earlier_id": p["earlier_id"],
               "camera_alignment_vs_photo_5": {k: check_alignment(align_spec, im, Path).get(k) for k in ("status", "shift_px", "response")},
               "robot": {"window_shift_vs_photo_5_px": [round(dx, 2), round(dy, 2)], "phase_response": round(resp, 3),
                         "base_link_px_estimate": [round(float(v), 1) for v in base],
                         "base_link_map_m_estimate": [round(float(v), 3) for v in ctx.frame.floor_to_map([robot_floor])[0]],
                         "heading": "gripper towards image left in all photos (visual); yaw not computed per photo -> USER_CONFIRMATION_REQUIRED",
                         "stored_auto_detection": auto and {k: auto[k] for k in ("bbox_image_px", "base_link_image_px", "how")},
                         "status": "ESTIMATED (photo_5 wheel reading shifted by the measured robot-window offset)"},
               "cube": cube, "pallets_empty": pals, "raw_changed_regions_touching_free_cells": free_blobs,
               "static_map_reuse": None}
        rec["difficulty"] = judge(rec)
        al_ok = rec["camera_alignment_vs_photo_5"]["status"] == "ALIGNED"
        on_line = [o for o in [("A_cube", cube)] + [(f"B{i + 1}_pallet", b) for i, b in enumerate(pals)] if o[1]["footprint_cells_on_free"]]
        rec["static_map_reuse"] = {"camera_aligned": al_ok,
                                   "inventory_objects_on_free_cells": [{"object": n_, "free_cells": o["footprint_cells_on_free"], "map_m": o["map_m"]} for n_, o in on_line],
                                   "verdict": ("NOT_REUSABLE (camera moved)" if not al_ok else
                                               "REUSABLE + per-photo obstacle for objects on the yellow line" if on_line else "REUSABLE"),
                                   "raw_change_blobs_note": cfg["step4a_rules"]["raw_change_blob_note"],
                                   "note": "static zones unchanged in all photos (same layout, aligned camera); objects beyond the yellow line are not mapped in any photo"}
        recs.append(rec)

    stamp = datetime.now().astimezone()
    out = CT.resolve(cfg["results_dir"]) / (stamp.strftime("%Y%m%d_%H%M%S") + "_step4a_photo_survey")
    out.mkdir(parents=True, exist_ok=False)
    render(cfg, ctx, photos, recs, out)
    cv2.imwrite(str(out / "median_background.png"), bg)
    (out / "survey.json").write_text(json.dumps({"created": stamp.isoformat(timespec="seconds"), "photos": recs,
                                                 "claims": {"paths": "NONE (A* not run in STEP 4-A)", "object_positions": "USER_CONFIRMATION_REQUIRED"}},
                                                indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (out / "config_snapshot.yaml").write_text(CT.CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    files = sorted(q for q in out.rglob("*") if q.is_file())
    (out / "manifest.json").write_text(json.dumps({"step": "STEP4A_photo_survey", "run_id": out.name, "python": platform.python_version(),
                                                   "code_sha256": sha256_file(Path(__file__).resolve()), "config_sha256": sha256_file(CT.CONFIG_PATH),
                                                   "photos": {p["name"]: p["sha256"] for p in photos},
                                                   "v02_code_unchanged": code_fingerprint()["sha256"] == before,
                                                   "photos_unchanged": all(sha256_file(p["path"]) == p["sha256"] for p in photos),
                                                   "outputs_sha256": {str(q.relative_to(out)): sha256_file(q) for q in files}}, indent=1), encoding="utf-8")
    for r in recs:
        print(f"{r['photo']}: align {r['camera_alignment_vs_photo_5']['status']} | robot shift {r['robot']['window_shift_vs_photo_5_px']} | "
              f"A {r['cube']['map_m']} {r['cube']['approach_side'][:22]} out {r['cube']['outside_drivable_cm']}cm run {r['cube']['free_run_on_approach_axis_cm']} "
              f"cross {r['cube']['straight_line_from_robot_crosses']} lat {r['cube']['lateral_room_at_approach_cm']} free {r['cube']['footprint_cells_on_free']} sup {r['cube']['supported_by_change_detection']} | "
              f"B {[(b['map_m'], b['free_run_on_approach_axis_cm'], b['straight_from_cube_crosses'], b['lateral_room_at_approach_cm'], b['footprint_cells_on_free'], b['supported_by_change_detection']) for b in r['pallets_empty']]} | "
              f"{r['difficulty']['level_preliminary']} {r['difficulty']['score']} | map {r['static_map_reuse']['verdict']} | on-line objects {r['static_map_reuse']['inventory_objects_on_free_cells']}")
    print("run_dir", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
