"""Strict external V4L2 camera selection and bounded capture interface."""

from glob import glob
from pathlib import Path
import time

import cv2


class CameraSourceError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def audit_camera_sources(config, by_id_paths=None, video_paths=None, sysfs_root=Path("/sys/class/video4linux")):
    selector = config["camera_selector"]
    by_id_paths = sorted(by_id_paths if by_id_paths is not None else glob("/dev/v4l/by-id/*"))
    video_paths = sorted(video_paths if video_paths is not None else glob("/dev/video*"))
    devices = []
    for path in video_paths:
        name_path = Path(sysfs_root) / Path(path).name / "name"
        identity = name_path.read_text().strip() if name_path.exists() else None
        devices.append({"device_path": path, "device_identity": identity})
    configured = selector["device_path"]
    configured_exists = configured in by_id_paths or Path(configured).exists()
    identity_match = selector["required_identity_substring"] in Path(configured).name
    forbidden_detected = [
        device for device in devices
        if any(token.casefold() in (device["device_identity"] or "").casefold() for token in selector["forbidden_fallback_identity_substrings"])
    ]
    selected = configured if configured_exists and identity_match else None
    return {
        "audit_status": "RESEARCH_CAMERA_AVAILABLE" if selected else "CAMERA_NOT_FOUND",
        "selector_policy": selector["policy"],
        "configured_device_path": configured,
        "required_identity_substring": selector["required_identity_substring"],
        "configured_path_exists": configured_exists,
        "configured_identity_matches": identity_match,
        "selected_device_path": selected,
        "fallback_allowed": False,
        "fallback_used": False,
        "by_id_paths": by_id_paths,
        "video_devices": devices,
        "forbidden_fallback_devices_detected": forbidden_detected,
    }


class CameraSource:
    def __init__(self, config, audit=None):
        self.config = config
        self.audit = audit if audit is not None else audit_camera_sources(config)
        self.capture = None
        self.frame_index = 0

    def open(self):
        path = self.audit.get("selected_device_path")
        if path is None:
            raise CameraSourceError("CAMERA_NOT_FOUND", "configured external research camera is unavailable")
        self.capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise CameraSourceError("CAMERA_OPEN_FAILED", f"could not open {path}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["preferred_width"])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["preferred_height"])
        self.capture.set(cv2.CAP_PROP_FPS, self.config["preferred_fps"])
        return self.camera_info()

    def read(self):
        if self.capture is None:
            raise CameraSourceError("CAMERA_OPEN_FAILED", "camera is not open")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise CameraSourceError("FRAME_READ_FAILED", "camera frame read failed")
        metadata = {
            "frame_index": self.frame_index,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "source_id": self.config["camera_selector"]["required_identity_substring"],
            "source_path": self.audit["selected_device_path"],
            "runtime_capture_time": time.time(),
        }
        self.frame_index += 1
        return frame, metadata

    def camera_info(self):
        if self.capture is None:
            return {
                "status": "CAMERA_NOT_OPEN",
                "source_id": self.config["camera_selector"]["required_identity_substring"],
                "source_path": self.audit.get("selected_device_path"),
                "width": None,
                "height": None,
                "reported_fps": None,
            }
        return {
            "status": "CAMERA_OPEN",
            "source_id": self.config["camera_selector"]["required_identity_substring"],
            "source_path": self.audit["selected_device_path"],
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "reported_fps": float(self.capture.get(cv2.CAP_PROP_FPS)),
        }

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
