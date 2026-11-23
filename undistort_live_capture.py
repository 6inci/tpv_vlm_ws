#!/usr/bin/env python3
"""Show and save undistorted frames; ported from camera_calibration_1920x1080/undistort_live.py."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CALIBRATION = Path(__file__).with_name("calibration_result.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0", help="camera number or /dev/v4l/by-id path")
    parser.add_argument("--mjpg", action="store_true", help="request MJPG from the camera")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
    "--save-dir",
    type=Path,
    default=Path("/home/binci/Pictures/Calibration"),
)
    parser.add_argument("--check", action="store_true", help="validate calibration file and exit")
    args = parser.parse_args()

    data = json.loads(args.calibration.read_text())
    width, height = data["image_width"], data["image_height"]
    matrix = np.array(data["camera_matrix"], dtype=np.float64)
    distortion = np.array(data["distortion_coefficients_k1_k2_p1_p2_k3"], dtype=np.float64)
    assert matrix.shape == (3, 3) and distortion.shape == (5,)
    if args.check:
        print(f"OK: {width}x{height}, RMS {data['rms_reprojection_error_px']:.3f} px")
        return

    args.save_dir.mkdir(parents=True, exist_ok=True)
    source = int(args.camera) if args.camera.isdigit() else args.camera
    camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if args.mjpg:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not camera.isOpened():
        raise SystemExit(f"camera {args.camera} could not be opened")

    ok, frame = camera.read()
    if not ok:
        raise SystemExit("camera opened but returned no frame")
    actual_size = frame.shape[1], frame.shape[0]
    if actual_size != (width, height):
        raise SystemExit(f"camera returned {actual_size[0]}x{actual_size[1]}; calibration requires {width}x{height}")
    map_x, map_y = cv2.initUndistortRectifyMap(matrix, distortion, None, matrix,
                                                (width, height), cv2.CV_16SC2)
    while ok:
        undistorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        cv2.imshow("Undistorted (s: save, q/Esc: quit)", undistorted)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            output = args.save_dir / f"undistorted_{cv2.getTickCount()}.png"
            if not cv2.imwrite(str(output), undistorted):
                raise RuntimeError(f"failed to save {output}")
            print(
                f"saved: {output.resolve()} "
                f"({undistorted.shape[1]}x{undistorted.shape[0]})"
            )
        elif key in (ord("q"), 27):
            break

        ok, frame = camera.read()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
