"""Calibration compatibility check -> map mode.

The existing calibration file is read (never modified). It is used only if the input camera model
and resolution match. Without a validated homography for the current floor layout the system stays
in IMAGE_SPACE_DEMO and metric output is disabled.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from ..config import sha256_file
from ..contracts import MAP_MODE_IMAGE


def read_exif_model(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            model = im.getexif().get(272)
    except Exception:  # unreadable EXIF is reported as unknown, not guessed
        return None
    return model.strip() if isinstance(model, str) else None


def assess_calibration(cam_cfg, frame_w, frame_h, camera_model):
    c = cam_cfg["calibration"]
    checks, reasons = [], []
    intr = None
    path = c.get("intrinsics_file")
    if path and Path(path).exists():
        data = json.loads(Path(path).read_text())
        intr = {"file": path, "sha256": sha256_file(path), "width": data["image_width"],
                "height": data["image_height"], "camera_matrix": data["camera_matrix"],
                "dist_coeffs": data["distortion_coefficients_k1_k2_p1_p2_k3"],
                "rms_px": data.get("rms_reprojection_error_px"), "calibrated_on": c.get("calibrated_on")}
        res_ok = (intr["width"], intr["height"]) == (frame_w, frame_h)
        checks.append({"check": "resolution", "expected": [intr["width"], intr["height"]],
                       "actual": [frame_w, frame_h], "ok": res_ok})
        model_ok = camera_model is not None and camera_model.lower() == str(c.get("camera_model", "")).lower()
        checks.append({"check": "camera_model", "expected": c.get("camera_model"), "actual": camera_model,
                       "ok": model_ok})
        if not res_ok:
            reasons.append("frame resolution differs from the calibration resolution")
        if not model_ok:
            reasons.append("camera model unknown or different from the calibrated camera")
    else:
        checks.append({"check": "intrinsics_file", "expected": path, "actual": None, "ok": False})
        reasons.append("no intrinsics file")
    undistort_ok = all(ch["ok"] for ch in checks)
    if not c.get("intrinsics_validated_for_current_install"):
        reasons.append("intrinsics not validated for the current camera installation")
    if c.get("homography_status") != "validated" or not c.get("homography_file"):
        reasons.append(f"homography status: {c.get('homography_status')} (no validated floor-plane transform)")
    ground_possible = (undistort_ok and c.get("intrinsics_validated_for_current_install")
                       and c.get("homography_status") == "validated" and bool(c.get("homography_file")))
    if ground_possible:
        reasons.append("ground-map planning is not implemented in v0.1; running IMAGE_SPACE_DEMO")
    return {"map_mode": MAP_MODE_IMAGE, "metric_output_enabled": False, "undistort_view_available": undistort_ok,
            "undistort_view_note": "intrinsics validity for the current install is UNVERIFIED; view only",
            "camera_model": camera_model, "frame_size": [frame_w, frame_h], "checks": checks, "reasons": reasons,
            "intrinsics": intr, "homography_file": c.get("homography_file"),
            "homography_status": c.get("homography_status")}


def undistort_image(image, intr):
    k = np.array(intr["camera_matrix"], dtype=np.float64)
    d = np.array(intr["dist_coeffs"], dtype=np.float64)
    return cv2.undistort(image, k, d, None, k)
