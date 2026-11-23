#!/usr/bin/env python3
"""Reproduce the legacy offline geometry and write regression artifacts."""

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.geometry import GeometryCompatibilityError, LegacyGeometry


DEFAULT_CALIBRATION = Path("/home/binci/camera_calibration_1920x1080/calibration_result.json")
DEFAULT_GEOMETRY = Path("/home/binci/qwen_vision_test/homography_12pt_undistorted.json")
DEFAULT_RAW_POINTS = Path("/home/binci/qwen_vision_test/homography_click_points.json")
DEFAULT_IMAGE = Path("/home/binci/qwen_vision_test/photo_04.png")
DEFAULT_REFERENCE = Path("/home/binci/qwen_vision_test/photo_04_undistorted.png")
DEFAULT_OUTPUT = WORKSPACE / "results" / "geometry_legacy_regression"


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise GeometryCompatibilityError(f"could not read image: {path}")
    return image


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--raw-points", type=Path, default=DEFAULT_RAW_POINTS)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    geometry = LegacyGeometry.load(args.calibration, args.geometry)
    raw_image, reference_image = read_image(args.image), read_image(args.reference)
    generated = geometry.undistort_image(raw_image)
    geometry.validate_image(reference_image)
    if generated.shape != reference_image.shape or generated.dtype != reference_image.dtype:
        raise GeometryCompatibilityError("generated/reference image shape or dtype differs")

    delta = np.abs(generated.astype(np.int16) - reference_image.astype(np.int16))
    changed_pixels = int(np.count_nonzero(np.any(delta != 0, axis=2)))
    image_metrics = {
        "changed_pixel_count": changed_pixels,
        "changed_pixel_ratio": changed_pixels / (generated.shape[0] * generated.shape[1]),
        "dtype": str(generated.dtype),
        "max_absolute_pixel_difference": int(delta.max()),
        "mean_absolute_pixel_difference": float(delta.mean()),
        "shape": list(generated.shape),
        "shape_match": True,
    }

    point_data = json.loads(args.raw_points.read_text())
    raw_points = np.asarray(point_data["points"], dtype=np.float64)
    if raw_points.shape != geometry.reference_undistorted_pixels.shape:
        raise GeometryCompatibilityError("raw and stored undistorted points must have matching Nx2 shapes")

    rows = []
    undistortion_differences = []
    reproduction_differences = []
    fitting_errors = []
    for index, (raw, legacy_pixel, assigned_world) in enumerate(zip(
        raw_points,
        geometry.reference_undistorted_pixels,
        geometry.reference_world_xy,
    ), start=1):
        new_pixel = np.asarray(geometry.raw_pixel_to_undistorted_pixel(*raw))
        legacy_prediction = np.asarray(geometry.undistorted_pixel_to_world(*legacy_pixel))
        new_prediction = np.asarray(geometry.undistorted_pixel_to_world(*new_pixel))
        undistortion_difference = float(np.linalg.norm(new_pixel - legacy_pixel))
        reproduction_difference = float(np.linalg.norm(new_prediction - legacy_prediction))
        fitting_error = float(np.linalg.norm(new_prediction - assigned_world))
        undistortion_differences.append(undistortion_difference)
        reproduction_differences.append(reproduction_difference)
        fitting_errors.append(fitting_error)
        rows.append({
            "point_id": f"P{index:02d}",
            "raw_u_px": raw[0],
            "raw_v_px": raw[1],
            "legacy_undistorted_u_px": legacy_pixel[0],
            "legacy_undistorted_v_px": legacy_pixel[1],
            "new_undistorted_u_px": new_pixel[0],
            "new_undistorted_v_px": new_pixel[1],
            "undistortion_reproduction_difference_px": undistortion_difference,
            "assigned_world_x_m": assigned_world[0],
            "assigned_world_y_m": assigned_world[1],
            "legacy_predicted_x_m": legacy_prediction[0],
            "legacy_predicted_y_m": legacy_prediction[1],
            "new_predicted_x_m": new_prediction[0],
            "new_predicted_y_m": new_prediction[1],
            "world_reproduction_difference_m": reproduction_difference,
            "legacy_fitting_error_m": fitting_error,
        })

    fitting_errors = np.asarray(fitting_errors)
    summary = {
        "coordinate_system": geometry.coordinate_system,
        "legacy_fitting_error_m": {
            "max": float(fitting_errors.max()),
            "mean": float(fitting_errors.mean()),
            "rmse": float(np.sqrt(np.mean(fitting_errors ** 2))),
        },
        "not_independent_accuracy_validation": True,
        "points_tested": len(rows),
        "undistortion_reproduction_difference_px": {
            "max": max(undistortion_differences),
            "mean": float(np.mean(undistortion_differences)),
        },
        "world_reproduction_difference_m": {
            "max": max(reproduction_differences),
            "mean": float(np.mean(reproduction_differences)),
        },
    }
    summary["pass"] = (
        image_metrics["changed_pixel_count"] == 0
        and summary["undistortion_reproduction_difference_px"]["max"] == 0
        and summary["world_reproduction_difference_m"]["max"] == 0
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = args.output_dir / "generated_photo_04_undistorted.png"
    if not cv2.imwrite(str(generated_path), generated):
        raise RuntimeError(f"could not write image: {generated_path}")
    write_json(args.output_dir / "image_regression_metrics.json", image_metrics)
    write_json(args.output_dir / "geometry_regression_summary.json", summary)
    with (args.output_dir / "geometry_regression_points.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Raw image: {args.image}")
    print(f"Stored new camera matrix: {args.geometry}")
    print(f"Generated image: {generated_path}")
    print(f"Image mean absolute difference: {image_metrics['mean_absolute_pixel_difference']:.12f}")
    print(f"Image max difference: {image_metrics['max_absolute_pixel_difference']}")
    print(f"Changed pixel ratio: {image_metrics['changed_pixel_ratio']:.12f}")
    print(f"Points tested: {len(rows)}")
    print(f"World reproduction max: {summary['world_reproduction_difference_m']['max']:.12g} m")
    print(f"Legacy fitting mean: {summary['legacy_fitting_error_m']['mean'] * 1000:.2f} mm")
    print(f"Legacy fitting RMSE: {summary['legacy_fitting_error_m']['rmse'] * 1000:.2f} mm")
    print(f"Legacy fitting max: {summary['legacy_fitting_error_m']['max'] * 1000:.2f} mm")
    print("PASS" if summary["pass"] else "FAIL")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
