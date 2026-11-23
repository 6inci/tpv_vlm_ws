#!/usr/bin/env python3
"""Hand-off package for the ROS2/Nav2 team member, built from an EXISTING planning run.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/navigation_handoff/scripts/make_handoff_package.py --run results/20261120_055014_wheelbound_photo_1 \
      --condition m2.0_reduced --photo photo_1 --name navigation_handoff_photo_1_20261120          # copy + summary + checks
  $PY experiments/navigation_handoff/scripts/make_handoff_package.py --name navigation_handoff_photo_1_20261120 --zip-only   # after README

Source files are copied byte for byte (sha256 verified) and never modified. No planning, no ROS, no robot.
"""
import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coordinate_transform as CT  # noqa: E402
import map_export as ME  # noqa: E402
from semantic_task_nav.config import sha256_file  # noqa: E402

REQUIRED = ["maps/map.pgm", "maps/map.yaml", "paths/path_A.csv", "paths/path_B.csv", "paths/waypoints.yaml",
            "figures/path_planning_photo.png", "figures/path_planning_map.png", "validation/validation.json"]


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(args, cfg):
    src = CT.resolve(args.run)
    cond = src / args.condition / args.photo
    step2 = CT.resolve(cfg["planning"]["map_run"])
    dest = CT.EXP / "handoff_package" / args.name
    dest.mkdir(parents=True, exist_ok=False)
    copies = {"maps/map.pgm": src / "maps/map.pgm", "maps/map.yaml": src / "maps/map.yaml", "maps/map_preview.png": src / "maps/map_preview.png",
              "paths/path_A.csv": cond / "path_A.csv", "paths/path_B.csv": cond / "path_B.csv", "paths/waypoints.yaml": cond / "waypoints.yaml",
              "figures/path_planning_photo.png": cond / "path_planning_photo.png", "figures/path_planning_map.png": cond / "path_planning_map.png",
              "validation/validation.json": cond / "validation.json", "validation/path_plan.json": cond / "path_plan.json",
              "reference/coordinate_frames.json": step2 / "coordinate_frames.json",
              "reference/approved_positions.yaml": CT.resolve(cfg["approved_positions_dir"]) / f"{args.photo}.yaml",
              "reference/comparison.json": src / "comparison.json"}
    prov = {}
    for rel, s in copies.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, dest / rel)
        h = sha256_file(dest / rel)
        assert h == sha256_file(s), rel
        prov[rel] = {"source": str(s), "sha256": h}
    return dest, src, cond, step2, prov


def check_and_summarize(dest, src, cond, step2, prov, args, cfg):
    checks = []

    def add(cid, ok, detail):
        checks.append({"id": cid, "status": "PASS" if ok else "FAIL", "detail": detail})

    add("required_files_present", all((dest / r).is_file() for r in REQUIRED), {r: (dest / r).is_file() for r in REQUIRED})
    # ---- map files
    my = yaml.safe_load((dest / "maps/map.yaml").read_text(encoding="utf-8"))
    img_rel = dest / "maps" / my["image"]
    arr, hdr = ME.read_pgm(dest / "maps/map.pgm")
    keys_ok = all(k in my for k in ("image", "mode", "resolution", "origin", "negate", "occupied_thresh", "free_thresh"))
    add("map_yaml_format_and_relative_image", keys_ok and my["image"] == "map.pgm" and img_rel.is_file() and len(my["origin"]) == 3, my)
    add("map_pgm_format", hdr["magic"] == "P5" and hdr["maxval"] == 255 and set(np.unique(arr).tolist()) <= {0, 205, 254}, hdr)
    add("map_identical_to_step2", sha256_file(dest / "maps/map.pgm") == sha256_file(step2 / "maps/map.pgm")
        and sha256_file(dest / "maps/map.yaml") == sha256_file(step2 / "maps/map.yaml"),
        {"step2_run": str(step2), "map.pgm": prov["maps/map.pgm"]["sha256"][:16]})
    # ---- path coordinates vs map (Nav2 map_server convention) and vs the frames of STEP 2
    fc = json.loads((dest / "reference/coordinate_frames.json").read_text(encoding="utf-8"))
    frame = CT.MapFrame(fc["transform"]["H_img_to_floor_cm"], fc["transform"]["map_origin_floor_cm"])
    res, ox, oy = my["resolution"], my["origin"][0], my["origin"][1]
    H = arr.shape[0]
    paths = {k: read_csv(dest / f"paths/path_{k}.csv") for k in ("A", "B")}
    for k, rows in paths.items():
        bad_cell, bad_free, bad_floor, bad_img, bad_step = [], [], [], [], []
        for i, r in enumerate(rows):
            x, y = float(r["x_map_m"]), float(r["y_map_m"])
            col = int(math.floor((x - ox) / res + 1e-9))
            row = H - 1 - int(math.floor((y - oy) / res + 1e-9))
            if (row, col) != (int(r["row"]), int(r["col"])):
                bad_cell.append(i)
            if arr[row, col] != my_free(cfg):
                bad_free.append(i)
            fl = [float(r["x_floor_cm"]), float(r["y_floor_cm"])]
            if np.abs(frame.floor_to_map([fl])[0] - [x, y]).max() > 1e-6:
                bad_floor.append(i)
            if np.abs(frame.floor_to_img([fl])[0] - [float(r["u_px"]), float(r["v_px"])]).max() > 0.06:
                bad_img.append(i)
            if i and abs(abs(x - float(rows[i - 1]["x_map_m"])) + abs(y - float(rows[i - 1]["y_map_m"])) - res) > 1e-6:
                bad_step.append(i)
        add(f"path_{k}_coordinates_match_map", not (bad_cell or bad_free or bad_floor or bad_img or bad_step),
            {"points": len(rows), "row_col_mismatch": bad_cell[:5], "not_on_free_pgm_cell": bad_free[:5], "floor_map_mismatch": bad_floor[:5],
             "image_px_mismatch": bad_img[:5], "non_4_connected_step": bad_step[:5],
             "rule": "col = floor((x - origin_x) / res), row = H - 1 - floor((y - origin_y) / res); every step is one cell along x or y"})
    # ---- start / end points
    wy = yaml.safe_load((dest / "paths/waypoints.yaml").read_text(encoding="utf-8"))
    ap = yaml.safe_load((dest / "reference/approved_positions.yaml").read_text(encoding="utf-8"))
    pA, pB = paths["A"], paths["B"]
    xy = lambda r: (round(float(r["x_map_m"]), 4), round(float(r["y_map_m"]), 4))  # noqa: E731
    robot = ap["robot"]["base_link"]["map_m"]
    snap_cm = math.dist(xy(pA[0]), robot) * 100
    add("start_end_points", xy(pA[0]) == (wy["path_A"][0]["x_m"], wy["path_A"][0]["y_m"]) and xy(pA[-1]) == (wy["grasp_stop_pose"]["x_m"], wy["grasp_stop_pose"]["y_m"])
        and xy(pB[0]) == xy(pA[-1]) and xy(pB[-1]) == (wy["place_stop_pose"]["x_m"], wy["place_stop_pose"]["y_m"]) and snap_cm <= 1.42,
        {"path_A": [xy(pA[0]), xy(pA[-1])], "path_B": [xy(pB[0]), xy(pB[-1])], "approved_robot_base_link_map_m": robot,
         "start_cell_snap_cm": round(snap_cm, 2), "grasp_stop": wy["grasp_stop_pose"], "place_stop": wy["place_stop_pose"]})
    # ---- waypoint order and direction
    issues = []
    for k in ("A", "B"):
        poses, rows = wy[f"path_{k}"], paths[k]
        idx = [p["index"] for p in poses]
        if idx != sorted(idx) or idx[0] != 0 or idx[-1] != len(rows) - 1:
            issues.append(f"{k}: indices {idx}")
        for p in poses:
            r = rows[p["index"]]
            if xy(r) != (p["x_m"], p["y_m"]):
                issues.append(f"{k}: pose {p['seq']} xy differs from csv row {p['index']}")
            if abs(math.radians(p["yaw_deg"]) - p["yaw_rad"]) > 1e-5:
                issues.append(f"{k}: pose {p['seq']} yaw_deg / yaw_rad differ")
            if p["kind"] != "rotate_in_place" and p["kind"] != "rotation_point" and abs(CT.wrap_deg(p["yaw_deg"] - float(r["body_yaw_map_deg"]))) > 1e-6 \
                    and p["index"] != 0:
                issues.append(f"{k}: pose {p['seq']} yaw differs from csv body yaw")
        for a, b in zip(poses, poses[1:]):
            if a["x_m"] != b["x_m"] and a["y_m"] != b["y_m"]:
                issues.append(f"{k}: diagonal between poses {a['seq']} and {b['seq']}")
    add("waypoints_order_and_direction", not issues, {"issues": issues, "n_poses": {k: len(wy[f"path_{k}"]) for k in ("A", "B")}})
    # ---- photo / map figures come from the same coordinates
    add("figures_from_same_run", all(prov[f"figures/{f}"]["source"].startswith(str(cond)) for f in ("path_planning_photo.png", "path_planning_map.png"))
        and all(c["status"] == "PASS" for c in checks if c["id"].startswith("path_")),
        "both figures were rendered by the planning run from the floor coordinates of path_*.csv; csv image px = inverse floor transform of the same floor points (checked above)")
    # ---- unverified items recorded
    val = json.loads((dest / "validation/validation.json").read_text(encoding="utf-8"))
    add("unverified_items_recorded", val["driving_status"].startswith("NOT_FOR_DRIVING") and val["grasp_alignment"] == "UNVERIFIED"
        and val["placement_alignment"] == "UNVERIFIED" and "UNVERIFIED" in wy["grasp_stop_pose"]["status"],
        {"driving_status": val["driving_status"], "grasp_alignment": val["grasp_alignment"], "placement_alignment": val["placement_alignment"],
         "body_overhang_objects": val["body_overhang_objects"]})
    checks.append({"id": "nav2_map_server_load", "status": "NOT_STARTED", "detail": "not run (nav2 not installed here; user decision: format check only)"})

    plan = json.loads((dest / "validation/path_plan.json").read_text(encoding="utf-8"))
    comp = json.loads((dest / "reference/comparison.json").read_text(encoding="utf-8"))
    rg = comp["robot_geometry"]
    summary = {
        "package": args.name, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": {"driving": val["driving_status"], "grasp_placement": "UNVERIFIED", "data_role": "development reference for the ROS2/Nav2 team member"},
        "map": {"yaml": "maps/map.yaml", "image": "maps/map.pgm", "resolution_m": res, "origin": my["origin"], "size_px": [arr.shape[1], arr.shape[0]],
                "size_m": [round(arr.shape[1] * res, 3), round(arr.shape[0] * res, 3)], "mode": my["mode"], "values": {"free": 254, "occupied": 0, "unknown": 205},
                "same_as_step2": str(step2 / "maps/map.pgm")},
        "frames": {"map": "ROS map frame, REP-103 (right-handed, z up), metres. Origin = floor-mat corner nearest the camera on the left; "
                          "+x = image right, +y = away from the camera",
                   "yaw": "degrees and radians, counter-clockwise from +x_map; body heading = gripper side (+X of base_link)",
                   "odom/base_link": "not provided: map->odom must come from the robot's localisation"},
        "robot_start_pose": {"path_start_map": {"x_m": wy["path_A"][0]["x_m"], "y_m": wy["path_A"][0]["y_m"], "yaw_deg": wy["path_A"][0]["yaw_deg"]},
                             "estimated_base_link_map": {"x_m": robot[0], "y_m": robot[1], "yaw_deg": ap["robot"]["yaw_map_deg"]},
                             "note": "path starts at the centre of the 2 cm cell that contains the estimated base_link"},
        "grasp_stop_pose": {**wy["grasp_stop_pose"], **{k: plan["grasp_stop"][k] for k in ("front_to_target_gap_cm", "lateral_offset_cm")}},
        "place_stop_pose": {**wy["place_stop_pose"], **{k: plan["place_stop"][k] for k in ("front_to_target_gap_cm", "lateral_offset_cm")}},
        "targets_map_m": {"A_cube_grasp_target": ap["A_cube"]["grasp_target"]["map_m"], "B_pallet_placement_target": ap["B_placement"]["placement_target"]["map_m"]},
        "paths": {k: {"file": f"paths/path_{k}.csv", "points": len(paths[k]), "length_cm": plan[f"path_{k}"]["length_cm"],
                      "rotation": [{kk: e[kk] for kk in ("at_index", "from_yaw_map_deg", "to_yaw_map_deg", "direction")}
                                   for e in plan[f"path_{k}"]["events"] if e["type"] == "rotate"],
                      "segments": [{kk: s[kk] for kk in ("move", "length_cm", "body_yaw_map_deg", "body_vs_move")} for s in plan[f"path_{k}"]["segments"]]}
                  for k in ("A", "B")},
        "footprint_used_for_planning": {"body_cm": rg["body_nominal_cm"], "wheel_envelope_cm": rg["wheel_envelope_cm"], "margin_cm_per_side": 2.0,
                                        "rule": "wheel envelope + margin on free cells (inside the yellow drivable zone); body + margin off static obstacles, "
                                                "no-go zones and unknown space; body overhang beyond the yellow line allowed (objects there are not mapped)"},
        "checks": {c["id"]: c["status"] for c in checks}, "sources": prov}
    (dest / "handoff_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (dest / "handoff_check.json").write_text(json.dumps({"checks": checks}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return checks


def my_free(cfg):
    return cfg["map_export"]["values"]["free"]


def make_zip(dest):
    z = shutil.make_archive(str(dest), "zip", root_dir=str(dest.parent), base_dir=dest.name)
    return Path(z)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run")
    ap.add_argument("--condition")
    ap.add_argument("--photo")
    ap.add_argument("--name", required=True)
    ap.add_argument("--zip-only", action="store_true")
    args = ap.parse_args()
    cfg = CT.load_config()
    if args.zip_only:
        dest = CT.EXP / "handoff_package" / args.name
        assert (dest / "README.md").is_file(), "write README.md first"
        print(make_zip(dest))
        return 0
    dest, src, cond, step2, prov = build(args, cfg)
    checks = check_and_summarize(dest, src, cond, step2, prov, args, cfg)
    print(json.dumps({"package": str(dest), "checks": {c["id"]: c["status"] for c in checks}}, indent=1))
    return 0 if all(c["status"] in ("PASS", "NOT_STARTED") for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
