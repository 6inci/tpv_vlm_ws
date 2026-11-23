#!/usr/bin/env python3
"""STEP 3: show the robot / cube / placement candidates of one photo, and load the positions the user confirmed.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/navigation_handoff/scripts/object_positions.py --photo photo_5     # new results/<time>_step3_candidates_photo_5/

Nothing proposed here is planning input. path_planning.py accepts only config/object_positions/<photo>.yaml with status USER_CONFIRMED.
Image points are projected to the floor at z = 0 with the STEP 2 transform; height parallax is not corrected (UNVERIFIED).
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coordinate_transform as CT  # noqa: E402
from semantic_task_nav.config import sha256_file  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
REQUIRED = {"robot": ("base_link_image_px", "yaw_map_deg"), "cube": ("target_image_px",), "placement": ("target_image_px",)}


def load_step2_frame(cfg):
    run = CT.resolve(cfg["planning"]["map_run"])
    fc = json.loads((run / "coordinate_frames.json").read_text(encoding="utf-8"))
    return CT.MapFrame(fc["transform"]["H_img_to_floor_cm"], fc["transform"]["map_origin_floor_cm"]), fc, run


def _pt(frame, uv):
    f = frame.img_to_floor([uv])[0]
    m = frame.floor_to_map([f])[0]
    return {"image_px": [round(float(uv[0]), 1), round(float(uv[1]), 1)], "floor_cm": [round(float(f[0]), 1), round(float(f[1]), 1)],
            "map_m": [round(float(m[0]), 4), round(float(m[1]), 4)], "projection": "z = 0 (height parallax not corrected)"}


def gather(cfg, photo, frame):
    """Candidates from stored records only (no model call)."""
    cc = cfg["candidates"][photo["name"]]
    out = {"robot": [], "heading_options": cc["robot"]["heading_options"], "cube": [], "placement": []}
    first = json.loads(CT.resolve(cc["robot"]["stored_detection"]).read_text(encoding="utf-8"))["stages"]["3_robot_start"]
    box = first["robot"]["bbox_xyxy"]
    out["robot"].append({"id": "R_BOX", **_pt(frame, first["start_candidates"][0]["xy"]), "bbox_image_px": box,
                         "source": f"stored detection {first['robot']['source']}; point = outline (box) centre", "knowledge": "ESTIMATED"})
    for e in cc["robot"]["ai_visual_estimates"]:
        out["robot"].append({"id": e["id"], **_pt(frame, e["image_px"]), "source": "AI visual estimate: " + e["method"], "knowledge": "ESTIMATED"})
    ret = json.loads(CT.resolve(cc["cube"]["stored_candidates"]).read_text(encoding="utf-8"))
    ph = next(p for p in ret["photos"] if p["photo_id"] == photo["earlier_id"])
    for c in ph["candidates"]:
        if c["state"] in cc["cube"]["show_states"]:
            b = c["bbox_xyxy"]
            out["cube"].append({"id": f"C{c['rank']}", **_pt(frame, [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), "bbox_image_px": b,
                                "retention_state": c["state"], "source_types": c["source_types"], "point": "box centre",
                                "source": "candidate_retention_policy (stored DINO + HSV candidates; Qwen gate answers kept there)"})
    ref = yaml.safe_load(CT.resolve(cc["placement"]["ai_drawn_supports"]).read_text(encoding="utf-8"))["objects"]
    for k in cc["placement"]["show_ids"]:
        b = ref[k]["bbox_xyxy_px"]
        out["placement"].append({"id": k, **_pt(frame, [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), "bbox_image_px": b, "type": ref[k]["type"],
                                 "state": ref[k]["state"], "point": "box centre", "source": "AI_DRAWN box (task_conditioned_target_grounding), not human-confirmed"})
    bays = json.loads(CT.resolve(cc["placement"]["loading_bay"]).read_text(encoding="utf-8"))
    bay = next(b for b in bays if b["sha256_16"] == photo["sha256"][:16])
    poly = np.array(bay["polygon"], np.float64)
    fpoly = frame.img_to_floor(poly)
    cen_f = fpoly.mean(0)
    out["placement"].append({"id": "BAY", **_pt(frame, frame.floor_to_img([cen_f])[0]), "polygon_image_px": poly.round(1).tolist(),
                             "point": "centroid of the floor marking corners (on the floor)", "source": f"Python floor-marking localisation ({bay['status']})"})
    return out


def _text(img, s, org, color=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def render(img, cands, frame, path):
    a = img.copy()
    ip = lambda p: tuple(int(round(v)) for v in p)  # noqa: E731
    for r, col in zip(cands["robot"], ((255, 255, 0), (0, 255, 255))):
        if "bbox_image_px" in r:
            b = r["bbox_image_px"]
            cv2.rectangle(a, ip(b[:2]), ip(b[2:]), (255, 255, 255), 2)
        cv2.drawMarker(a, ip(r["image_px"]), col, cv2.MARKER_CROSS, 22, 3)
        _text(a, r["id"], ip(np.add(r["image_px"], [10, -12 if r["id"] == "R_BOX" else 26])), col, 0.6, 2)
    base = next(r for r in cands["robot"] if r["id"] != "R_BOX")
    for h, col in zip(cands["heading_options"], ((0, 0, 255), (200, 200, 200))):
        m0 = np.array(base["map_m"])
        tip = m0 + 0.3 * np.array([np.cos(np.radians(h["yaw_map_deg"])), np.sin(np.radians(h["yaw_map_deg"]))])
        p0, p1 = frame.map_to_img([m0, tip])
        cv2.arrowedLine(a, ip(p0), ip(p1), col, 3, tipLength=0.25)
        _text(a, h["id"], ip(p1 + [-30 if h["yaw_map_deg"] == 180 else 8, -10]), col, 0.6, 2)
    for c in cands["cube"]:
        b = c["bbox_image_px"]
        cv2.rectangle(a, ip(b[:2]), ip(b[2:]), (0, 0, 255), 2)
        cv2.circle(a, ip(c["image_px"]), 3, (0, 255, 255), -1)
        _text(a, c["id"], ip([b[0], b[1] - 6]), (0, 0, 255), 0.6, 2)
    for p in cands["placement"]:
        if "bbox_image_px" in p:
            b = p["bbox_image_px"]
            cv2.rectangle(a, ip(b[:2]), ip(b[2:]), (255, 128, 0), 2)
        else:
            cv2.polylines(a, [np.round(p["polygon_image_px"]).astype(np.int32)], True, (255, 128, 0), 2)
        cv2.drawMarker(a, ip(p["image_px"]), (255, 128, 0), cv2.MARKER_TILTED_CROSS, 18, 3)
        _text(a, p["id"], ip(np.add(p["image_px"], [10, 24])), (255, 128, 0), 0.6, 2)
    _text(a, "STEP 3 CANDIDATES (not planning input until the user confirms) - photo_5 = E1", (12, 30), (0, 255, 255), 0.8, 2)
    _text(a, "robot: R_BOX = stored detection box centre, R_WHEELS = AI reading of the 4 wheel centres | arrows: heading options H180 / H0",
          (12, 60), scale=0.6)
    _text(a, "cube: C* = stored candidates (PROBABLE) | placement: P_DST = AI-drawn pallet box, BAY = floor marking centroid", (12, 86), scale=0.6)

    def crop(cx, cy, w=320, h=180, s=2):
        x0, y0 = int(np.clip(cx - w // 2, 0, a.shape[1] - w)), int(np.clip(cy - h // 2, 0, a.shape[0] - h))
        return cv2.resize(a[y0:y0 + h, x0:x0 + w], (w * s, h * s), interpolation=cv2.INTER_CUBIC)
    rb = cands["robot"][0]["image_px"]
    cb = cands["cube"][0]["image_px"] if cands["cube"] else [442, 424]
    pb = cands["placement"][0]["image_px"]
    row = np.hstack([crop(*rb), crop(*cb), crop(*pb)])
    for i, s in enumerate(("zoom: robot", "zoom: cube", "zoom: placement")):
        _text(row, s, (10 + 640 * i, 26), (0, 255, 255), 0.7, 2)
    cv2.imwrite(str(path), np.vstack([a, row]))


def load_confirmed(cfg, photo_name):
    """The ONLY way object positions enter planning. Refuses anything that is not USER_CONFIRMED for this exact photo."""
    p = CT.resolve(cfg["object_positions_dir"]) / f"{photo_name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"{p}: no confirmed object positions for {photo_name}")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    if d.get("status") != "USER_CONFIRMED":
        raise ValueError(f"{p}: status {d.get('status')!r} is not USER_CONFIRMED")
    photo = next(x for x in cfg["photos"] if x["name"] == photo_name)
    if d.get("photo") != photo_name or d.get("photo_sha256") != photo["sha256"]:
        raise ValueError(f"{p}: photo / sha256 do not match {photo_name}")
    for sec, keys in REQUIRED.items():
        for k in keys:
            v = (d.get(sec) or {}).get(k)
            ok = isinstance(v, (int, float)) if k.endswith("_deg") else isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)
            if not ok:
                raise ValueError(f"{p}: {sec}.{k} missing or malformed ({v!r})")
    return d, p


# ================================================================================================ user-approved positions (2026-11-20)
APPROVED_SCHEMA = "navigation_handoff/approved_positions/v1"
NOTES = {"image_frame": "undistorted_px of the photo as stored (already undistorted; never undistort again)",
         "floor_projection": "z = 0 through the PROVISIONAL E1 mat-seam floor transform (no validated homography exists); height parallax not corrected",
         "work_targets": "grasp / placement work targets are where the GRIPPER CENTRE should be; TEMPORARY = the object click unless set separately. "
                         "Not a validated floor position and not evidence that grasp or placement is possible"}


def robot_auto_estimate(cfg, photo):
    """Stored automatic robot detection of this photo, or None. Heading: never estimated automatically."""
    src = cfg["robot_auto_estimates"]
    refs = json.loads(CT.resolve(src["file"]).read_text(encoding="utf-8"))["references"]
    r = (refs.get(photo["earlier_id"]) or {}).get("robot")
    if not r:
        return None
    return {"bbox_image_px": [round(float(v), 2) for v in r["box"]], "base_link_image_px": [round(float(v), 2) for v in r["start_img"]],
            "how": r["how"], "rule": src["rule"], "source_file": str(CT.resolve(src["file"])), "heading": None, "heading_note": src["heading"]}


def yaw_from_points(frame, p_img, q_img):
    """Map yaw (deg) of the floor direction from image point p to image point q."""
    a, b = frame.img_to_map([p_img, q_img])
    return round(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])), 1)


def body_corners_img(frame, base_img, yaw_map, L_cm, W_cm):
    """Nominal body rectangle at (base_link, yaw) in image px (floor level)."""
    m = frame.img_to_map([base_img])[0]
    a = math.radians(yaw_map)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    hl, hw = L_cm / 200, W_cm / 200
    return frame.map_to_img([m + hl * u + hw * v, m + hl * u - hw * v, m - hl * u - hw * v, m - hl * u + hw * v])


def build_approved(photo, st, frame, auto, approved_at):
    """st: A_obj, A_tgt, A_tgt_src, B_obj, B_tgt, B_tgt_src, robot, robot_src, yaw, yaw_src (image px / deg)."""
    def pt(uv):
        return _pt(frame, uv)
    return {"schema": APPROVED_SCHEMA, "photo": photo["name"], "photo_file": Path(photo["path"]).name, "photo_sha256": photo["sha256"],
            "earlier_id": photo["earlier_id"], "status": "USER_APPROVED", "approved_at": approved_at, "approved_by": "user (position_picker GUI)",
            "notes": NOTES,
            "A_cube": {"object_image_px": list(st["A_obj"]), "grasp_target_image_px": list(st["A_tgt"]), "grasp_target_source": st["A_tgt_src"],
                       "grasp_target_status": "UNVERIFIED", "object": pt(st["A_obj"]), "grasp_target": pt(st["A_tgt"])},
            "B_placement": {"pallet_image_px": list(st["B_obj"]), "placement_target_image_px": list(st["B_tgt"]), "placement_target_source": st["B_tgt_src"],
                            "placement_target_status": "UNVERIFIED", "pallet": pt(st["B_obj"]), "placement_target": pt(st["B_tgt"])},
            "robot": {"base_link_image_px": list(st["robot"]), "base_link_source": st["robot_src"], "base_link": pt(st["robot"]),
                      "yaw_map_deg": float(st["yaw"]), "yaw_source": st["yaw_src"], "auto_estimate": auto,
                      "base_link_note": "base_link = centre of the 4 wheels at floor level (URDF); a box centre includes height parallax"}}


def _xy_ok(v):
    return isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)


def load_approved(cfg, photo_name):
    """The ONLY planning input since 2026-11-20. Refuses anything not USER_APPROVED for this exact photo."""
    p = CT.resolve(cfg["approved_positions_dir"]) / f"{photo_name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"{p}: no approved positions for {photo_name} (approve them in position_picker.py first)")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    photo = next(x for x in cfg["photos"] if x["name"] == photo_name)
    if d.get("schema") != APPROVED_SCHEMA or d.get("status") != "USER_APPROVED":
        raise ValueError(f"{p}: schema {d.get('schema')!r} / status {d.get('status')!r} is not an approved position file")
    if d.get("photo") != photo_name or d.get("photo_sha256") != photo["sha256"]:
        raise ValueError(f"{p}: photo / sha256 do not match {photo_name}")
    need = [("A_cube", "object_image_px"), ("A_cube", "grasp_target_image_px"), ("B_placement", "pallet_image_px"),
            ("B_placement", "placement_target_image_px"), ("robot", "base_link_image_px")]
    for sec, k in need:
        if not _xy_ok((d.get(sec) or {}).get(k)):
            raise ValueError(f"{p}: {sec}.{k} missing or malformed")
    if not isinstance((d.get("robot") or {}).get("yaw_map_deg"), (int, float)):
        raise ValueError(f"{p}: robot.yaw_map_deg missing")
    return d, p


def render_preview(img, doc, frame, nominal_cm):
    a = img.copy()
    ip = lambda p: tuple(int(round(v)) for v in p)  # noqa: E731
    r = doc["robot"]
    if r.get("auto_estimate"):
        b = r["auto_estimate"]["bbox_image_px"]
        cv2.rectangle(a, ip(b[:2]), ip(b[2:]), (200, 200, 200), 1)
    poly = body_corners_img(frame, r["base_link_image_px"], r["yaw_map_deg"], *nominal_cm)
    cv2.polylines(a, [np.round(poly).astype(np.int32)], True, (255, 255, 0), 2)
    m0 = frame.img_to_map([r["base_link_image_px"]])[0]
    tip = frame.map_to_img([m0 + 0.3 * np.array([math.cos(math.radians(r["yaw_map_deg"])), math.sin(math.radians(r["yaw_map_deg"]))])])[0]
    cv2.arrowedLine(a, ip(r["base_link_image_px"]), ip(tip), (0, 0, 255), 3, tipLength=0.25)
    cv2.drawMarker(a, ip(r["base_link_image_px"]), (255, 255, 0), cv2.MARKER_CROSS, 22, 2)
    _text(a, f"ROBOT yaw {r['yaw_map_deg']:.1f}", ip(np.add(r["base_link_image_px"], [12, -14])), (255, 255, 0), 0.6, 2)
    for key, obj, tgt, col, lab in (("A_cube", "object_image_px", "grasp_target_image_px", (0, 0, 255), "A"),
                                    ("B_placement", "pallet_image_px", "placement_target_image_px", (255, 128, 0), "B")):
        o, t = doc[key][obj], doc[key][tgt]
        cv2.drawMarker(a, ip(o), col, cv2.MARKER_CROSS, 24, 2)
        if np.linalg.norm(np.subtract(o, t)) > 0.5:
            cv2.drawMarker(a, ip(t), col, cv2.MARKER_DIAMOND, 14, 2)
        _text(a, lab, ip(np.add(o, [10, -10])), col, 0.8, 2)
    _text(a, f"APPROVED {doc['approved_at']}  {doc['photo']} ({doc['photo_file']})  - coordinates PROVISIONAL, grasp/placement UNVERIFIED",
          (12, 30), (0, 255, 255), 0.65, 2)
    return a


def save_approved(cfg, doc, preview_bgr):
    """Current file data/approved_positions/<photo>.yaml (+ preview) and an immutable copy in history/."""
    d = CT.resolve(cfg["approved_positions_dir"])
    (d / "history").mkdir(parents=True, exist_ok=True)
    stamp = doc["approved_at"].replace(":", "").replace("-", "")[:15]
    text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    hist = d / "history" / f"{doc['photo']}_{stamp}.yaml"
    if hist.exists():
        raise FileExistsError(hist)
    hist.write_text(text, encoding="utf-8")
    cv2.imwrite(str(d / "history" / f"{doc['photo']}_{stamp}_preview.png"), preview_bgr)
    (d / f"{doc['photo']}.yaml").write_text(text, encoding="utf-8")
    cv2.imwrite(str(d / f"{doc['photo']}_preview.png"), preview_bgr)
    return d / f"{doc['photo']}.yaml", hist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", required=True)
    args = ap.parse_args()
    cfg = CT.load_config()
    photo = next(p for p in CT.resolve_photos(cfg) if p["name"] == args.photo)
    frame, _, map_run = load_step2_frame(cfg)
    img = cv2.imread(photo["path"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    cands = gather(cfg, photo, frame)
    out = CT.resolve(cfg["results_dir"]) / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_step3_candidates_{args.photo}")
    out.mkdir(parents=True, exist_ok=False)
    render(img, cands, frame, out / "candidates.png")
    rec = {"photo": {k: photo[k] for k in ("name", "path", "sha256", "earlier_id")}, "map_run": str(map_run), "status": "CANDIDATES_ONLY",
           "note": "shown to the user for confirmation; not planning input", "candidates": cands}
    (out / "candidates.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"created": datetime.now().astimezone().isoformat(timespec="seconds"), "photo_sha256": photo["sha256"],
                                                   "code_sha256": sha256_file(Path(__file__).resolve()), "config_sha256": sha256_file(CT.CONFIG_PATH),
                                                   "outputs": {f.name: sha256_file(f) for f in out.iterdir() if f.is_file()}}, indent=1), encoding="utf-8")
    print(json.dumps({"run_dir": str(out), "robot": [(r["id"], r["map_m"]) for r in cands["robot"]], "cube": [(c["id"], c["map_m"]) for c in cands["cube"]],
                      "placement": [(p["id"], p["map_m"]) for p in cands["placement"]]}, indent=1))


if __name__ == "__main__":
    main()
