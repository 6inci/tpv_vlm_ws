"""Carry the v0.2 DRAFT map prior (drawn on the raw reference photo P01) into the E1 frame.

Pipeline: P01 raw px --undistort (read-only calibration, same call as v0.2 undistort_image)--> P01 undistorted px
          --homography H (SIFT + RANSAC on the two undistorted photos)--> E1 px.
The floor is planar, so two undistorted views of it are related by a homography. The estimate is accepted only if the declared
checks hold (config.yaml map_transfer.accept). Polygons stay what they were: MANUAL_MAP_PRIOR / DRAFT_UNCONFIRMED. The reference
zone Z06 and the manual start pose of the v0.2 map are NOT carried over.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from semantic_task_nav.config import sha256_file  # noqa: E402
from semantic_task_nav.navigation.map_builder import check_alignment  # noqa: E402

LAYERS = ("traversable", "static_obstacles", "no_go")


def load_calibration(path):
    c = json.loads(Path(path).read_text())
    return np.array(c["camera_matrix"], np.float64), np.array(c["distortion_coefficients_k1_k2_p1_p2_k3"], np.float64), c


def undistort_points(pts, K, D):
    return cv2.undistortPoints(np.float32(pts).reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)


def transform_points(pts, K, D, H):
    u = undistort_points(pts, K, D).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(u, H).reshape(-1, 2)


def _stats(r):
    return {"n": int(len(r)), "median_px": round(float(np.median(r)), 3), "rms_px": round(float(np.sqrt(np.mean(r ** 2))), 3),
            "p95_px": round(float(np.percentile(r, 95)), 3), "max_px": round(float(r.max()), 3)}


def estimate_homography(ref_u, frame, mcfg):
    """ref_u: undistorted reference photo, frame: target photo (both BGR). Returns (H or None, record)."""
    cv2.setRNGSeed(int(mcfg["ransac_seed"]))
    sift = cv2.SIFT_create(nfeatures=int(mcfg["sift_features"]))
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # noqa: E731
    k1, d1 = sift.detectAndCompute(g(ref_u), None)
    k2, d2 = sift.detectAndCompute(g(frame), None)
    knn = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < mcfg["ratio_test"] * n.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    rec = {"method": "SIFT + ratio test + RANSAC homography", "n_keypoints": [len(k1), len(k2)], "n_matches": len(good),
           "ransac_threshold_px": mcfg["ransac_threshold_px"], "seed": mcfg["ransac_seed"]}
    if len(good) < 8:
        return None, {**rec, "status": "TOO_FEW_MATCHES"}
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, mcfg["ransac_threshold_px"], maxIters=5000, confidence=0.999)
    if H is None:
        return None, {**rec, "status": "NO_HOMOGRAPHY"}
    inl = mask.ravel().astype(bool)
    P1, P2 = p1[inl], p2[inl]
    res = np.linalg.norm(cv2.perspectiveTransform(P1.reshape(-1, 1, 2), H).reshape(-1, 2) - P2, axis=1)
    hold = {}
    for name, sel in (("fit_left_test_right", P2[:, 0] < frame.shape[1] / 2), ("fit_right_test_left", P2[:, 0] >= frame.shape[1] / 2),
                      ("fit_top_test_bottom", P2[:, 1] < frame.shape[0] / 2), ("fit_bottom_test_top", P2[:, 1] >= frame.shape[0] / 2)):
        Hs, _ = cv2.findHomography(P1[sel], P2[sel], cv2.RANSAC, mcfg["ransac_threshold_px"])
        if Hs is None:
            hold[name] = {"status": "NO_HOMOGRAPHY"}
            continue
        t = ~sel
        e = np.linalg.norm(cv2.perspectiveTransform(P1[t].reshape(-1, 1, 2), Hs).reshape(-1, 2) - P2[t], axis=1)
        hold[name] = _stats(e)
    rec.update({"status": "ESTIMATED", "H_ref_undistorted_to_frame": H.tolist(), "n_inliers": int(inl.sum()),
                "inlier_residual": _stats(res), "holdout": hold,
                "inlier_x_range_px": [round(float(P2[:, 0].min())), round(float(P2[:, 0].max()))],
                "inlier_y_range_px": [round(float(P2[:, 1].min())), round(float(P2[:, 1].max()))]})
    return H, rec


def accept(rec, align, mcfg):
    a = mcfg["accept"]
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    if rec.get("status") != "ESTIMATED":
        add("homography estimated", False, rec.get("status"))
        return {"accepted": False, "checks": checks}
    add("inliers", rec["n_inliers"] >= a["min_inliers"], f"{rec['n_inliers']} >= {a['min_inliers']}")
    add("inlier RMS", rec["inlier_residual"]["rms_px"] <= a["max_inlier_rms_px"],
        f"RMS {rec['inlier_residual']['rms_px']} px (limit {a['max_inlier_rms_px']}), median {rec['inlier_residual']['median_px']}, "
        f"p95 {rec['inlier_residual']['p95_px']}")
    ho = [v for v in rec["holdout"].values() if "median_px" in v]
    add("hold-out medians", len(ho) == 4 and all(v["median_px"] <= a["holdout_median_px"] for v in ho),
        [v.get("median_px") for v in rec["holdout"].values()])
    add("hold-out p95", len(ho) == 4 and all(v["p95_px"] <= a["holdout_p95_px"] for v in ho), [v.get("p95_px") for v in rec["holdout"].values()])
    add("v0.2 check_alignment on the warped reference", align.get("status") == a["v02_check_alignment"],
        {k: align.get(k) for k in ("status", "shift_px", "response", "thresholds")})
    return {"accepted": all(c["ok"] for c in checks), "checks": checks}


def build_spec(src_spec, K, D, H, image_size, reference_image, transfer_rec):
    """New map spec in the E1 frame (a NEW dict; the v0.2 file is untouched)."""
    def conv(zs):
        return [{"id": z["id"], "polygon": np.round(transform_points(z["polygon"], K, D, H), 1).tolist()} for z in zs]

    spec = {"schema_version": src_spec["schema_version"], "map_id": src_spec["map_id"] + "__transferred_to_E1",
            "coordinate_frame": src_spec["coordinate_frame"], "image_size": list(image_size),
            "review_status": "DRAFT_UNCONFIRMED",
            "map_status": "MANUAL_MAP_PRIOR (hand-drawn DRAFT zones, transformed into the E1 frame by a validated homography)",
            "review_note": src_spec["review_note"] + " Zone shapes are unchanged by the transfer; only their coordinates were mapped.",
            "traversable": conv(src_spec["traversable"]), "static_obstacles": conv(src_spec["static_obstacles"]),
            "no_go": conv(src_spec["no_go"]),
            "start_pose": None,   # the v0.2 manual start (700, 740) is NOT used: the start comes from a detection
            "reference_image": str(reference_image), "reference_image_sha256": sha256_file(reference_image),
            "reference_image_note": "P01 undistorted and warped into the E1 frame; check_alignment compares it with the E1 photo",
            "alignment_check": src_spec.get("alignment_check", {"max_shift_px": 3.0, "min_response": 0.3}),
            "provenance": {"source_map_id": src_spec["map_id"], "source_provenance": src_spec["provenance"],
                           "transfer": {k: transfer_rec[k] for k in ("method", "n_matches", "n_inliers", "inlier_residual", "holdout")},
                           "not_carried_over": ["semantic_reference_zones (Z06)", "start_pose"]}}
    return spec


def yellow_mask(img, ycfg):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= ycfg["hue"][0]) & (hsv[..., 0] <= ycfg["hue"][1]) & (hsv[..., 1] >= ycfg["min_saturation"])
         & (hsv[..., 2] >= ycfg["min_value"])).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, 4] >= ycfg["min_component_px"]:
            keep[lab == i] = 1
    return keep.astype(bool)


def edge_offsets(spec, yellow, ycfg_edge):
    """Informational: for each edge of the traversable and static-obstacle polygons, how far (along the edge normal) is the nearest
    yellow pixel? Mixes hand-drawing error of the DRAFT with transfer error, so it is reported, not used as a gate."""
    h, w = yellow.shape
    out = []
    step, srch = ycfg_edge["sample_step_px"], ycfg_edge["search_px"]
    for key in ("traversable", "static_obstacles"):
        for z in spec[key]:
            P = np.array(z["polygon"], float)
            for i in range(len(P)):
                a, b = P[i], P[(i + 1) % len(P)]
                L = float(np.linalg.norm(b - a))
                if L < 1:
                    continue
                t = (b - a) / L
                nrm = np.array([-t[1], t[0]])
                offs = []
                for s in np.arange(step / 2, L, step):
                    c = a + t * s
                    best = None
                    for d in range(-srch, srch + 1):
                        x, y = int(round(c[0] + nrm[0] * d)), int(round(c[1] + nrm[1] * d))
                        if 0 <= x < w and 0 <= y < h and yellow[y, x] and (best is None or abs(d) < abs(best)):
                            best = d
                    offs.append(best)
                found = [o for o in offs if o is not None]
                row = {"zone": z["id"], "edge": f"{i}->{(i + 1) % len(P)}", "length_px": round(L), "samples": len(offs),
                       "yellow_found_fraction": round(len(found) / len(offs), 2) if offs else 0.0}
                if found:
                    row.update({"median_abs_offset_px": round(float(np.median(np.abs(found))), 1),
                                "p95_abs_offset_px": round(float(np.percentile(np.abs(found), 95)), 1)})
                out.append(row)
    return out


def run(cfg, src_spec_path, calib_path, ref_img_path, frame, out_dir):
    """Returns (spec | None, record). spec is None when the transfer is not accepted."""
    mcfg = cfg["map_transfer"]
    K, D, calib = load_calibration(calib_path)
    src_spec = json.loads(Path(src_spec_path).read_text(encoding="utf-8"))
    ref = cv2.imread(str(ref_img_path))
    ref_u = cv2.undistort(ref, K, D, None, K)        # identical to v0.2 geometry.calibration.undistort_image
    H, rec = estimate_homography(ref_u, frame, mcfg)
    out = {"source_map": {"file": str(src_spec_path), "sha256": sha256_file(src_spec_path), "map_id": src_spec["map_id"],
                          "review_status": src_spec["review_status"]},
           "reference_image": {"file": str(ref_img_path), "sha256": sha256_file(ref_img_path),
                               "note": "raw frame of the 2026-11-14 session; zone polygons were drawn on it"},
           "calibration": {"file": str(calib_path), "sha256": sha256_file(calib_path), "rms_reprojection_px": calib.get("rms_reprojection_error_px"),
                           "read_only": True, "intrinsics_validated_for_current_install": False},
           "homography": rec}
    if H is None:
        out["decision"] = {"accepted": False, "checks": []}
        return None, out
    warped = cv2.warpPerspective(ref_u, H, (frame.shape[1], frame.shape[0]))
    wpath = Path(out_dir) / "reference_P01_warped_to_E1.png"
    cv2.imwrite(str(wpath), warped)
    spec = build_spec(src_spec, K, D, H, (frame.shape[1], frame.shape[0]), wpath, rec)
    align = check_alignment(spec, frame, lambda p: Path(p))
    out["v02_check_alignment"] = align
    out["decision"] = accept(rec, align, mcfg)
    return (spec if out["decision"]["accepted"] else None), out
