"""Deterministic reproduction of the legacy offline pixel-to-world geometry."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


class GeometryCompatibilityError(ValueError):
    """Raised when input is incompatible with the stored legacy geometry."""


def _matrix(data, key, shape):
    value = np.asarray(data[key], dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise GeometryCompatibilityError(f"{key} must be a finite {shape} matrix")
    return value


@dataclass(frozen=True)
class LegacyGeometry:
    calibration_size: tuple[int, int]
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    distortion: np.ndarray
    new_camera_matrix: np.ndarray
    homography: np.ndarray
    roi: tuple[int, int, int, int]
    reference_undistorted_pixels: np.ndarray
    reference_world_xy: np.ndarray

    coordinate_system = "LEGACY_WORKSPACE_COORDINATE"

    @classmethod
    def load(cls, calibration_path, geometry_path):
        calibration = json.loads(Path(calibration_path).read_text())
        geometry = json.loads(Path(geometry_path).read_text())

        calibration_size = (
            int(calibration["image_width"]),
            int(calibration["image_height"]),
        )
        image_size = (int(geometry["image_width"]), int(geometry["image_height"]))
        distortion = np.asarray(
            calibration["distortion_coefficients_k1_k2_p1_p2_k3"],
            dtype=np.float64,
        )
        pixels = np.asarray(geometry["src_pixels_undistorted"], dtype=np.float64)
        world = np.asarray(geometry["dst_world_xy"], dtype=np.float64)
        roi = tuple(int(value) for value in geometry["roi"])

        if min(*calibration_size, *image_size) <= 0:
            raise GeometryCompatibilityError("image dimensions must be positive")
        if distortion.ndim != 1 or distortion.size < 4 or not np.isfinite(distortion).all():
            raise GeometryCompatibilityError("distortion coefficients must be a finite vector")
        if pixels.ndim != 2 or pixels.shape[1:] != (2,) or pixels.shape != world.shape:
            raise GeometryCompatibilityError("reference pixel/world points must be matching Nx2 arrays")
        if not np.isfinite(pixels).all() or not np.isfinite(world).all():
            raise GeometryCompatibilityError("reference points must be finite")
        if len(roi) != 4:
            raise GeometryCompatibilityError("roi must contain four integers")
        if geometry.get("unit") != "meter":
            raise GeometryCompatibilityError("legacy world coordinates must use meters")

        return cls(
            calibration_size=calibration_size,
            image_size=image_size,
            camera_matrix=_matrix(calibration, "camera_matrix", (3, 3)),
            distortion=distortion,
            new_camera_matrix=_matrix(geometry, "new_camera_matrix", (3, 3)),
            homography=_matrix(geometry, "homography", (3, 3)),
            roi=roi,
            reference_undistorted_pixels=pixels,
            reference_world_xy=world,
        )

    @property
    def scaled_camera_matrix(self):
        matrix = self.camera_matrix.copy()
        scale_x = self.image_size[0] / self.calibration_size[0]
        scale_y = self.image_size[1] / self.calibration_size[1]
        matrix[0, (0, 2)] *= scale_x
        matrix[1, (1, 2)] *= scale_y
        return matrix

    def validate_image(self, image):
        if image is None or image.ndim not in (2, 3):
            raise GeometryCompatibilityError("input must be a decoded image array")
        actual = (image.shape[1], image.shape[0])
        if actual != self.image_size:
            raise GeometryCompatibilityError(
                f"Expected {self.image_size[0]}x{self.image_size[1]} legacy reference image, "
                f"received {actual[0]}x{actual[1]}."
            )

    def _validate_pixel(self, u, v):
        point = np.asarray([u, v], dtype=np.float64)
        if not np.isfinite(point).all():
            raise GeometryCompatibilityError("pixel coordinates must be finite")
        if not (0 <= u < self.image_size[0] and 0 <= v < self.image_size[1]):
            raise GeometryCompatibilityError(
                f"pixel ({u}, {v}) is outside {self.image_size[0]}x{self.image_size[1]}"
            )

    def undistort_image(self, image):
        self.validate_image(image)
        return cv2.undistort(
            image,
            self.scaled_camera_matrix,
            self.distortion,
            None,
            self.new_camera_matrix,
        )

    def raw_pixel_to_undistorted_pixel(self, u, v):
        self._validate_pixel(u, v)
        point = np.asarray([[[u, v]]], dtype=np.float32)
        result = cv2.undistortPoints(
            point,
            self.scaled_camera_matrix,
            self.distortion,
            P=self.new_camera_matrix,
        )[0, 0]
        return float(result[0]), float(result[1])

    def undistorted_pixel_to_world(self, u, v):
        self._validate_pixel(u, v)
        point = np.asarray([[[u, v]]], dtype=np.float32)
        result = cv2.perspectiveTransform(point, self.homography)[0, 0]
        if not np.isfinite(result).all():
            raise GeometryCompatibilityError("homography produced a non-finite world coordinate")
        return float(result[0]), float(result[1])

    def raw_pixel_to_world(self, u, v):
        return self.undistorted_pixel_to_world(
            *self.raw_pixel_to_undistorted_pixel(u, v)
        )
