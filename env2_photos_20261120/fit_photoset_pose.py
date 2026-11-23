#!/usr/bin/env python3
"""Camera pose of the shared photo set in the teammate's map frame env2_20261119 (config.yaml).

Coordinates: pixels are undistorted-image pixels of the photo set (no distortion left); map = env2_20261119 metres, floor z = 0.
Steps
  1. Floor homography: teammate register undistorted.png (newK = K) -> pose photo (SIFT + MAGSAC, floor-dominated mask).
  2. The teammate's 12 register reference pixels mapped through it -> reference pixels in the pose photo (not clicked).
  3. Pose by the teammate's pixel_to_map.fit_pose (vendor copy, unchanged) under each intrinsics hypothesis; config select_rule picks one.
  4. Gates (config): max floor error over the fit points, map->pixel->map round trip, camera height.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -I -B fit_photoset_pose.py           # fit, save camera_pose/, check
  /home/binci/miniconda3/envs/stnav_gui/bin/python -I -B fit_photoset_pose.py --check   # re-check the saved pose only (no SIFT)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
C = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
TB = Path(C["teammate_bundle"])
sys.path.insert(0, str(TB / "1_required"))
sys.path.insert(0, str(TB / "2_optional"))
import pixel_to_map as p2m  # noqa: E402  (teammate code, unchanged)
import clearance_check as cc  # noqa: E402  (teammate code, unchanged; only Env is used, for the overlay)

OUT = HERE / C["out_dir"]
POSE = OUT / "photoset_20261120_pose.json"
SIZE = tuple(C["image_size"])


def hypothesis_k(h, K, dist):
    """-> (K of the photo, dist or None)"""
    if h["alpha"] is None:
        return K, (dist if h["dist"] else None)
    return cv2.getOptimalNewCameraMatrix(K, dist, SIZE, float(h["alpha"]), SIZE)[0], None


def map_reference_px():
    hc = C["homography"]
    cv2.setRNGSeed(hc["seed"])
    reg = cv2.imread(str(TB / C["register_image"]), cv2.IMREAD_GRAYSCALE)
    new = cv2.imread(str(HERE / C["photos_dir"] / C["pose_photo"]), cv2.IMREAD_GRAYSCALE)
    x0, y0, x1, y1 = hc["mask_xyxy"]
    mask = np.zeros(reg.shape, np.uint8)
    mask[y0:y1, x0:x1] = 255
    sift = cv2.SIFT_create(nfeatures=hc["sift_features"])
    k1, d1 = sift.detectAndCompute(reg, mask)
    k2, d2 = sift.detectAndCompute(new, mask)
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < hc["ratio"] * b.distance]
    p1 = np.float32([k1[a.queryIdx].pt for a in good])
    p2 = np.float32([k2[a.trainIdx].pt for a in good])
    H, inl = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, hc["ransac_px"], maxIters=20000, confidence=0.9999)
    inl = inl.ravel().astype(bool)
    e = np.linalg.norm(cv2.perspectiveTransform(p1[inl][:, None], H)[:, 0] - p2[inl], axis=1)
    pts = json.loads((TB / C["register_points"]).read_text(encoding="utf-8"))["points"]
    ref_px = {n: [round(float(v), 2) for v in cv2.perspectiveTransform(np.float32([[p["undistorted_px"]]]), H)[0, 0]] for n, p in pts.items()}
    stats = {"matches": len(good), "inliers": int(inl.sum()), "median_residual_px": round(float(np.median(e)), 3),
             "p90_residual_px": round(float(np.percentile(e, 90)), 3), "H_register_to_photo": H.tolist()}
    return ref_px, stats


def check(pose):
    """Gates on a saved pose -> {gate: [value, ok]}"""
    K = np.array(pose["K_photo"], float)
    rv, tv = np.array(pose["rvec"], float), np.array(pose["tvec"], float)
    ref = p2m.ref_points()
    err = {n: float(np.linalg.norm(p2m.pixel_to_map([px], rv, tv, K, None)[0] - ref[n]) * 100) for n, px in pose["reference_px"].items()}
    xy = np.array([[0.7, 0.35], [1.87, 0.53], [3.1, 1.3], [2.68, 0.48]])
    rt = float(np.abs(p2m.pixel_to_map(p2m.map_to_pixel(xy, rv, tv, K, None), rv, tv, K, None) - xy).max() * 100)
    h = p2m.camera_height(rv, tv)
    g = C["gates"]
    fit_max = max(err[n] for n in pose["fit_names"])
    return {"fit_point_max_err_cm": [round(fit_max, 3), fit_max <= g["fit_point_max_err_cm"]],
            "roundtrip_max_cm": [round(rt, 6), rt <= g["roundtrip_max_cm"]],
            "camera_height_m": [round(h, 4), g["camera_height_m"][0] <= h <= g["camera_height_m"][1]]}


def overlay(pose):
    K, rv, tv = np.array(pose["K_photo"], float), np.array(pose["rvec"], float), np.array(pose["tvec"], float)
    E = cc.Env()
    img = cv2.imread(str(HERE / C["photos_dir"] / C["pose_photo"]))
    cs, _ = cv2.findContours(E.free.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        c = c[:, 0, :].astype(float)
        xy = np.c_[E.ox + (c[:, 0] + 0.5) * E.res, E.oy + (E.NY - 1 - c[:, 1] + 0.5) * E.res]
        cv2.polylines(img, [p2m.map_to_pixel(xy, rv, tv, K, None).round().astype(np.int32)], True, (255, 255, 0), 2)
    for n, (u, v) in pose["reference_px"].items():
        cv2.drawMarker(img, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(img, n, (int(u) + 6, int(v) - 6), 0, 0.7, (0, 0, 255), 2)
    for n, (u, v) in pose["zones_px"].items():
        cv2.circle(img, (int(u), int(v)), 6, (0, 140, 255), -1)
    cv2.putText(img, f"cyan = env2_20261119 boundary projected with {POSE.name} ({pose['intrinsics']}); red = reference points; orange = zones.json",
                (20, 1060), 0, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(OUT / "overlay_env2_on_photoset.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="re-check the saved pose only")
    a = ap.parse_args()
    if a.check:
        r = check(json.loads(POSE.read_text(encoding="utf-8")))
        print(json.dumps(r))
        sys.exit(0 if all(ok for _, ok in r.values()) else 1)
    OUT.mkdir(exist_ok=True)
    K, dist = p2m.intrinsics()
    ref_px, hs = map_reference_px()
    print(f"homography: {hs['inliers']}/{hs['matches']} inliers, median residual {hs['median_residual_px']} px")
    if hs["median_residual_px"] > C["homography"]["max_median_residual_px"]:
        raise SystemExit("homography residual above the config limit")
    fits = []
    for h in C["intrinsics_hypotheses"]:
        Kh, dh = hypothesis_k(h, K, dist)
        rv, tv, names, err = p2m.fit_pose(ref_px, Kh, dh, quiet=True)
        fits.append({"name": h["name"], "K_photo": Kh.tolist(), "dist": dh is not None, "rvec": rv.ravel().tolist(), "tvec": tv.ravel().tolist(),
                     "fit_names": names, "camera_height_m": round(p2m.camera_height(rv, tv), 4),
                     "floor_err_cm": {k: round(v, 3) for k, v in err.items()}, "fit_point_max_err_cm": round(max(err[n] for n in names), 3)})
        print(f"  {h['name']:22s} height {fits[-1]['camera_height_m']:.3f} m, fit points max {fits[-1]['fit_point_max_err_cm']:.2f} cm")
    best = min(fits, key=lambda f: f["fit_point_max_err_cm"])
    if best["dist"]:
        raise SystemExit("the selected hypothesis keeps lens distortion; this script only writes undistorted poses")
    R0, _ = cv2.Rodrigues(p2m.load_pose("register-raw")[0])
    c0 = -R0.T @ p2m.load_pose("register-raw")[1].ravel()
    R1, _ = cv2.Rodrigues(np.array(best["rvec"]).reshape(3, 1))
    c1 = -R1.T @ np.array(best["tvec"])
    zones = json.loads((TB / "2_optional" / "zones.json").read_text(encoding="utf-8"))["zones"]
    zpx = p2m.map_to_pixel(np.array([z["assigned_xy_m"] for z in zones]), best["rvec"], best["tvec"], np.array(best["K_photo"]), None)
    pose = {"created": datetime.now().isoformat(timespec="seconds"), "frame": "env2_20261119 map (m), floor z = 0", "pixels": "undistorted photo pixels, no dist",
            "photo": C["pose_photo"], "applies_to": f"all 15 photos in {C['photos_dir']}/ (within 1 px of each other)", "undistorted": True,
            "intrinsics": best["name"], "K_photo": best["K_photo"], "rvec": best["rvec"], "tvec": best["tvec"], "fit_names": best["fit_names"],
            "reference_px": ref_px, "floor_err_cm": best["floor_err_cm"], "camera_height_m": best["camera_height_m"],
            "vs_register_20261119_194739": {"move_cm": round(float(np.linalg.norm(c1 - c0)) * 100, 2),
                                            "turn_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R1 @ R0.T) - 1) / 2, -1, 1)))), 3)},
            "zones_px": {z["name"]: [round(float(v), 1) for v in p] for z, p in zip(zones, zpx)},
            "homography": hs, "all_hypotheses": [{k: f[k] for k in ("name", "camera_height_m", "fit_point_max_err_cm", "floor_err_cm")} for f in fits],
            "status": "TESTING: intrinsics inferred from the fit (not stated by whoever made the photos); reference px mapped by homography, not clicked",
            "use": "p2m.pixel_to_map(px, rvec, tvec, K_photo, None, z) with K_photo from this file, NOT calibration_result.json K"}
    pose["gates"] = check(pose)
    POSE.write_text(json.dumps(pose, ensure_ascii=False, indent=1), encoding="utf-8")
    overlay(pose)
    print("selected:", best["name"], "| gates:", json.dumps(pose["gates"]), "| vs register:", pose["vs_register_20261119_194739"])
    print("saved", POSE)


if __name__ == "__main__":
    main()
