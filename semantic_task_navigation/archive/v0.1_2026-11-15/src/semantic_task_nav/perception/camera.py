"""Frame sources: single image, video file, live V4L2 camera. Frames are raw_image_px (BGR, as decoded)."""
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..geometry.calibration import read_exif_model

_frame_counter = itertools.count(1)


@dataclass
class Frame:
    image: np.ndarray
    frame_id: int
    timestamp: float
    source: dict = field(default_factory=dict)


class SourceError(RuntimeError):
    pass


def list_video_devices():
    """Capture nodes from /sys/class/video4linux (index 0 = primary node of a device)."""
    out = []
    for d in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:])):
        try:
            idx = int((d / "index").read_text().strip())
        except (OSError, ValueError):
            idx = 0
        if idx != 0:
            continue
        name = (d / "name").read_text().strip() if (d / "name").exists() else d.name
        out.append({"index": int(d.name[5:]), "path": f"/dev/{d.name}", "name": name})
    return out


class ImageSource:
    kind = "image"

    def __init__(self, path):
        self.path = str(Path(path).resolve())
        # IGNORE_ORIENTATION: keep the stored pixel grid (raw_image_px) exactly as recorded.
        img = cv2.imread(self.path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise SourceError(f"cannot read image {self.path}")
        self.image = img
        self.info = {"kind": "image", "path": self.path, "camera_model": read_exif_model(self.path),
                     "camera_model_from": "EXIF Model tag", "width": img.shape[1], "height": img.shape[0]}

    def read(self):
        return Frame(self.image.copy(), next(_frame_counter), time.time(), dict(self.info))

    def release(self):
        pass


class VideoSource:
    kind = "video"

    def __init__(self, path, loop=False):
        self.path, self.loop = str(Path(path).resolve()), loop
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise SourceError(f"cannot open video {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.info = {"kind": "video", "path": self.path, "camera_model": None, "camera_model_from": "unknown (video)",
                     "fps": self.fps}

    def read(self):
        ok, img = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self.cap.read()
        if not ok:
            return None
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return Frame(img, next(_frame_counter), time.time(), {**self.info, "video_frame_index": pos,
                                                               "width": img.shape[1], "height": img.shape[0]})

    def release(self):
        self.cap.release()


class CameraSource:
    kind = "live"

    def __init__(self, device, live_cfg):
        dev = device if device is not None else live_cfg.get("device")
        devices = list_video_devices()
        if dev is None:
            if not devices:
                raise SourceError("no /dev/video* capture device found")
            dev = devices[0]["index"]
        index = int(dev) if str(dev).isdigit() else int(str(dev).replace("/dev/video", ""))
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)  # open logic as in camera_calibration compare_raw_undistorted.py
        if live_cfg.get("fourcc"):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*live_cfg["fourcc"]))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, live_cfg["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, live_cfg["height"])
        if not self.cap.isOpened():
            raise SourceError(f"camera /dev/video{index} could not be opened")
        ok, img = self.cap.read()
        if not ok:
            self.cap.release()
            raise SourceError(f"camera /dev/video{index} returned no frame")
        name = next((d["name"] for d in devices if d["index"] == index), None)
        self.info = {"kind": "live", "device": f"/dev/video{index}", "camera_model": name,
                     "camera_model_from": "V4L2 device name", "requested_size": [live_cfg["width"], live_cfg["height"]],
                     "actual_size": [img.shape[1], img.shape[0]], "fourcc": live_cfg.get("fourcc")}

    def read(self):
        ok, img = self.cap.read()
        if not ok:
            return None
        return Frame(img, next(_frame_counter), time.time(), {**self.info, "width": img.shape[1], "height": img.shape[0]})

    def release(self):
        self.cap.release()


def frame_shift(ref_bgr, cur_bgr, scale=0.25):
    """Global shift (px, in full-resolution units) + mean absolute difference ratio, for scene-change checks."""
    if ref_bgr.shape != cur_bgr.shape:
        return {"comparable": False}
    a = cv2.resize(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale).astype(np.float32)
    b = cv2.resize(cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale).astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    return {"comparable": True, "shift_px": [dx / scale, dy / scale], "response": float(resp),
            "mean_abs_diff_ratio": float(np.mean(np.abs(a - b)) / 255.0)}
