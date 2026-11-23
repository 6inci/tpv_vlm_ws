"""Coordinate frames of the navigation hand-off.

  undistorted_px  pixels of the 8 photos as stored. They are ALREADY undistorted (newCameraMatrix = K); never undistort again.
  floor_cm        PROVISIONAL floor plane of the E1 mat-seam transform (reused unchanged: task_plan_prototype/floor_transform.py).
                  Origin = mat-grid node (0,0) (far-left mat corner), +x along the mat columns (image right),
                  +y along the mat rows TOWARDS the camera (image down). Seen from above this frame is left-handed.
  map_m           ROS2 `map` frame of map.yaml (REP-103: right-handed, z up, metres).
                  Origin = floor_cm (ox, oy) from the config, +x_map = +x_floor, +y_map = -y_floor (away from the camera):
                      x_map = (x_floor - ox) / 100,   y_map = (oy - y_floor) / 100,   yaw_map = -yaw_floor
  grid_rc         row/col of map.pgm. Row 0 = TOP image row = largest y_map (Nav2 map_server convention). map.yaml `origin`
                  = map_m position of the lower-left corner of the lower-left cell.
`odom` and `base_link` are not defined here: map -> odom has to come from the robot's localisation (outside this work).
"""
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.dont_write_bytecode = True          # importing reused modules must not write __pycache__ into earlier experiments
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP.parent / "task_plan_prototype"))
import floor_transform as FT  # noqa: E402  (reused unchanged)
from semantic_task_nav.config import sha256_file  # noqa: E402

CONFIG_PATH = EXP / "config" / "navigation_config.yaml"


def load_config(path=CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (EXP / p).resolve()


def resolve_photos(cfg):
    """Find every declared photo by sha256 (the file names changed). Each hash must match exactly one file."""
    by_hash = {}
    for f in sorted(Path(cfg["photo_dir"]).glob("*.png")):
        by_hash.setdefault(sha256_file(f), []).append(f)
    out = []
    for p in cfg["photos"]:
        files = by_hash.get(p["sha256"], [])
        if len(files) != 1:
            raise FileNotFoundError(f"{p['name']}: {len(files)} files with sha256 {p['sha256'][:16]} in {cfg['photo_dir']}")
        out.append({**p, "path": str(files[0]), "file_name_matches_config": files[0].name == p["current_file"]})
    return out


def fit_reference_transform(cfg, img_bgr):
    """E1 mat-seam floor transform, recomputed with the reused function and its declared parameters."""
    floor_cfg = yaml.safe_load(resolve(cfg["reuse"]["floor_transform_config"]).read_text(encoding="utf-8"))["floor"]
    return FT.fit_floor_transform(img_bgr, floor_cfg), floor_cfg


def wrap_deg(a):
    """Angle in (-180, 180]."""
    a = math.fmod(a, 360.0)
    a = a - 360.0 if a > 180.0 else a + 360.0 if a <= -180.0 else a
    return 0.0 if a == 0 else a          # no -0.0


class MapFrame:
    def __init__(self, H_img_to_floor_cm, origin_floor_cm):
        self.H = np.asarray(H_img_to_floor_cm, np.float64)
        self.Hinv = np.linalg.inv(self.H)
        self.ox, self.oy = float(origin_floor_cm[0]), float(origin_floor_cm[1])

    def img_to_floor(self, pts):
        return FT.project(self.H, pts)

    def floor_to_img(self, pts):
        return FT.project(self.Hinv, pts)

    def floor_to_map(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        return np.c_[(p[:, 0] - self.ox) / 100.0, (self.oy - p[:, 1]) / 100.0]

    def map_to_floor(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        return np.c_[p[:, 0] * 100.0 + self.ox, self.oy - p[:, 1] * 100.0]

    def img_to_map(self, pts):
        return self.floor_to_map(self.img_to_floor(pts))

    def map_to_img(self, pts):
        return self.floor_to_img(self.map_to_floor(pts))

    @staticmethod
    def yaw_floor_to_map(deg):
        return wrap_deg(-deg)

    @staticmethod
    def yaw_map_to_floor(deg):
        return wrap_deg(-deg)

    def to_dict(self):
        return {"H_img_to_floor_cm": self.H.tolist(), "map_origin_floor_cm": [self.ox, self.oy],
                "formulas": {"img_to_floor": "floor_cm = H_img_to_floor_cm * [u, v, 1] (perspective division)",
                             "floor_to_map": "x_map = (x_floor - ox)/100, y_map = (oy - y_floor)/100  [m]",
                             "yaw": "yaw_map = -yaw_floor (yaw_floor measured from +x_floor towards +y_floor, i.e. image right towards image down)"}}


class MapGrid:
    """Cells of map.pgm. They cover floor x in [x0, x0 + cols*res_cm) and y in [y0, y0 + rows*res_cm); row 0 = smallest y_floor = largest y_map."""

    def __init__(self, frame, extent_floor_cm, resolution_m):
        x0, y0, x1, y1 = (float(v) for v in extent_floor_cm)
        self.frame, self.resolution_m, self.res_cm = frame, float(resolution_m), float(resolution_m) * 100.0
        self.x0, self.y0 = x0, y0
        self.cols = int(math.ceil((x1 - x0) / self.res_cm - 1e-9))
        self.rows = int(math.ceil((y1 - y0) / self.res_cm - 1e-9))
        ll = frame.floor_to_map([[x0, y0 + self.rows * self.res_cm]])[0]    # lower-left corner of the lower-left cell
        self.origin_m = [round(float(ll[0]), 9), round(float(ll[1]), 9), 0.0]

    def cell_center_floor(self, r, c):
        return self.x0 + (np.asarray(c) + 0.5) * self.res_cm, self.y0 + (np.asarray(r) + 0.5) * self.res_cm

    def cell_center_map(self, r, c):
        """Nav2 map_server convention, computed WITHOUT the floor frame (used to cross-check floor_to_map)."""
        return (self.origin_m[0] + (np.asarray(c) + 0.5) * self.resolution_m,
                self.origin_m[1] + (self.rows - np.asarray(r) - 0.5) * self.resolution_m)

    def map_to_cell(self, x, y):
        c = np.floor((np.asarray(x) - self.origin_m[0]) / self.resolution_m).astype(int)
        r = self.rows - 1 - np.floor((np.asarray(y) - self.origin_m[1]) / self.resolution_m).astype(int)
        return r, c

    def to_dict(self):
        return {"rows": self.rows, "cols": self.cols, "resolution_m": self.resolution_m, "origin_m": self.origin_m,
                "extent_floor_cm": [self.x0, self.y0, self.x0 + self.cols * self.res_cm, self.y0 + self.rows * self.res_cm],
                "row0": "top image row = largest y_map (Nav2 map_server)"}
