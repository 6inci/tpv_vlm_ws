"""PROVISIONAL floor-plane transform (undistorted E1 pixels -> floor cm) from the mat seams of the photo.

Why this exists: the project has no validated floor transform for the current layout (config/camera.yaml: homography_status not_available).
The calibration file is only read elsewhere; nothing here touches it. The mat size (105 cm) is the user's; pixel positions come from image evidence:
approximate grid-node seeds (AI-picked, in config.yaml) are refined by fitting the local seam lines on a thin-line (ridge) map and intersecting them.
Quality is MEASURED, not assumed: fit residuals at the nodes, leave-one-out, and an independent check on dense samples of the inner seams that
are not used in the fit. The residual image distortion (mat edges are visibly curved near the image border) shows up in these numbers.
"""
import math

import cv2
import numpy as np


def ridge_map(gray, sigma, percentile):
    """Thin-line response (bright or dark lines), normalised by a high percentile."""
    s = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    dxx, dyy = cv2.Sobel(s, cv2.CV_32F, 2, 0, ksize=3), cv2.Sobel(s, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(s, cv2.CV_32F, 1, 1, ksize=3)
    t = np.sqrt(((dxx - dyy) / 2) ** 2 + dxy ** 2)
    l1, l2 = (dxx + dyy) / 2 + t, (dxx + dyy) / 2 - t
    r = np.maximum(np.maximum(l1, 0), np.maximum(-l2, 0))
    den = np.percentile(r, percentile)
    return r / den if den > 0 else r                 # a blank image has no ridges: all-zero response, nothing is found


def _peak(R, c, n, search, thr):
    offs = np.arange(-search, search + 0.5, 0.5)
    h, w = R.shape
    vals = np.array([R[int(round(c[1] + n[1] * o)), int(round(c[0] + n[0] * o))]
                     if 0 <= int(round(c[0] + n[0] * o)) < w and 0 <= int(round(c[1] + n[1] * o)) < h else 0.0 for o in offs])
    k = int(vals.argmax())
    return (c + n * offs[k]) if vals[k] > thr else None


def fit_local_line(R, p, d, side, rp):
    """Robust straight-line fit to the seam near p along direction d. Returns None when too few ridge samples."""
    p, d = np.asarray(p, float), np.asarray(d, float)
    d = d / np.linalg.norm(d)
    for _ in range(rp["iterations"]):
        ts = np.arange(-rp["half_len_px"], rp["half_len_px"] + 1, float(rp["sample_step_px"]))
        ts = ts[ts >= 8] if side > 0 else ts[ts <= -8] if side < 0 else ts[np.abs(ts) >= 8]
        n = np.array([-d[1], d[0]])
        pts = [q for q in (_peak(R, p + d * t, n, rp["search_px"], rp["threshold"]) for t in ts) if q is not None]
        if len(pts) < rp["min_samples"]:
            return None
        vx, vy, x0, y0 = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        dn = np.array([vx, vy])
        dn = dn if np.dot(dn, d) >= 0 else -dn
        base = np.array([x0, y0])
        p, d = base + dn * np.dot(p - base, dn), dn
        res = np.abs((np.array(pts) - base) @ np.array([-dn[1], dn[0]]))
    return {"p": p, "d": d, "n_samples": len(pts), "n_inliers": int((res < 2.5).sum()), "rms_px": float(np.sqrt((res ** 2).mean()))}


def _intersect(a, b):
    t = np.linalg.solve(np.array([a["d"], -b["d"]]).T, b["p"] - a["p"])
    return a["p"] + a["d"] * t[0]


def refine_node(R, key, seed, rp):
    v = fit_local_line(R, seed["xy"], seed["v_dir"], seed["v_side"], rp)
    h = fit_local_line(R, seed["xy"], seed["h_dir"], seed["h_side"], rp)
    rec = {"node": key, "seed_px": list(seed["xy"]), "used": False}
    if v is None or h is None:
        return {**rec, "reason": "too few seam samples on the %s line" % ("vertical" if v is None else "horizontal")}
    p = _intersect(v, h)
    return {**rec, "used": True, "xy_px": [float(p[0]), float(p[1])], "shift_from_seed_px": [round(float(p[0] - seed["xy"][0]), 1), round(float(p[1] - seed["xy"][1]), 1)],
            "vertical_fit": {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items() if k in ("n_samples", "n_inliers", "rms_px")},
            "horizontal_fit": {k: (round(x, 2) if isinstance(x, float) else x) for k, x in h.items() if k in ("n_samples", "n_inliers", "rms_px")}}


def project(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), np.asarray(H, np.float64)).reshape(-1, 2)


def _fit(src, dst):
    return cv2.findHomography(np.asarray(src, np.float64), np.asarray(dst, np.float64), 0)[0]


def seam_deviation(R, H, sc, rp):
    """Dense ridge samples along an inner seam (not used in the fit) -> floor cm -> deviation from the known world line."""
    p0, p1 = np.array(sc["p0"], float), np.array(sc["p1"], float)
    length = np.linalg.norm(p1 - p0)
    u = (p1 - p0) / length
    n = np.array([-u[1], u[0]])
    pts = [q for q in (_peak(R, p0 + u * t, n, 8, rp["threshold"]) for t in np.arange(0, length, 6.0)) if q is not None]
    P = np.array(pts, np.float32)
    vx, vy, x0, y0 = cv2.fitLine(P, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    keep = np.abs((P - np.array([x0, y0])) @ np.array([-vy, vx])) < sc["inlier_px"]       # drop samples taken from objects on the seam
    W = project(H, P[keep])
    dev = W[:, 0 if sc["axis"] == "x" else 1] - sc["world_cm"]
    return {"id": sc["id"], "axis": sc["axis"], "world_cm": sc["world_cm"], "n_samples": int(len(P)), "n_used": int(keep.sum()),
            "mean_dev_cm": round(float(dev.mean()), 2), "p95_abs_dev_cm": round(float(np.percentile(np.abs(dev), 95)), 2),
            "max_abs_dev_cm": round(float(np.abs(dev).max()), 2)}


def fit_floor_transform(img_bgr, fcfg):
    """Returns the full record. H maps undistorted image px -> floor cm; `accepted` follows the thresholds declared in the config."""
    rp = fcfg["ridge"]
    R = ridge_map(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), rp["sigma"], rp["percentile"])
    nodes = [refine_node(R, k, s, rp) for k, s in fcfg["grid_seeds"].items()]
    used = [n for n in nodes if n["used"]]
    rec = {"method": "local seam-line fits intersected at the mat-grid nodes -> DLT homography (cv2.findHomography, method 0, all nodes)",
           "world_unit": "cm", "mat_cm": fcfg["mat_cm"], "nodes": nodes, "n_nodes_used": len(used), "accept_rule": fcfg["accept"]}
    if len(used) < fcfg["accept"]["min_nodes"]:
        return {**rec, "accepted": False, "reason": f"{len(used)} nodes < {fcfg['accept']['min_nodes']}", "H_img_to_floor_cm": None}
    src = [n["xy_px"] for n in used]
    dst = [[fcfg["mat_cm"] * int(n["node"].split(",")[0]), fcfg["mat_cm"] * int(n["node"].split(",")[1])] for n in used]
    H = _fit(src, dst)
    res = np.linalg.norm(project(H, src) - np.array(dst), axis=1)
    loo = []
    for k in range(len(src)):
        m = [i for i in range(len(src)) if i != k]
        loo.append(float(np.linalg.norm(project(_fit(np.array(src)[m], np.array(dst)[m]), [src[k]])[0] - np.array(dst[k]))))
    for n, r, l in zip(used, res, loo):
        n["fit_residual_cm"], n["leave_one_out_cm"] = round(float(r), 2), round(l, 2)
    seams = [seam_deviation(R, H, sc, rp) for sc in fcfg["seam_checks"]]
    seam_max = max(s["max_abs_dev_cm"] for s in seams)
    fit_rms = float(np.sqrt((res ** 2).mean()))
    ok = fit_rms <= fcfg["accept"]["max_fit_rms_cm"] and seam_max <= fcfg["accept"]["max_seam_dev_cm"]
    return {**rec, "accepted": bool(ok), "status": fcfg["status_if_accepted"] if ok else "REFUSED",
            "H_img_to_floor_cm": H.tolist(), "H_floor_cm_to_img": np.linalg.inv(H).tolist(),
            "fit_rms_cm": round(fit_rms, 2), "fit_max_cm": round(float(res.max()), 2), "leave_one_out_max_cm": round(max(loo), 2),
            "independent_seam_check": seams, "seam_max_abs_dev_cm": seam_max,
            "footprint_margin_cm": math.ceil(seam_max * 2) / 2,
            "known_limits": ["mat edges near the image border are visibly curved: residual lens distortion remains after the read-only undistortion; "
                             "the nodes on the bottom mat row (image y > 975) have the largest residuals and lie outside the planned work band",
                             "points above the floor (cube on the conveyor, tops of boxes) are displaced by parallax: only floor-level features map correctly",
                             "mat gaps are ignored as the user stated; a gap of a few mm would shift the node positions"]}


def bev_matrix(H_img_to_floor, px_per_cm, pad_cm):
    """undistorted image px -> BEV px (floor cm scaled, shifted by the pad)."""
    S = np.array([[px_per_cm, 0, pad_cm * px_per_cm], [0, px_per_cm, pad_cm * px_per_cm], [0, 0, 1]], np.float64)
    return S @ np.asarray(H_img_to_floor, np.float64)
