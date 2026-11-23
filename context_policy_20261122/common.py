"""공통: 사진 묶음별 카메라, 픽셀 <-> 지도, 작업 공간 띠, 5S 구역 (config/policy.yaml).

좌표
- 사진 픽셀: 각 사진 묶음의 편 사진 픽셀 (dev: K, test: alpha 0 newK). 왜곡 계수 없음.
- 지도: env2_20261119 (m), 바닥 z = 0.
- 작업 공간 띠: 노란 선 바깥 다각형(A-B-C-D, 팀원 기준점)의 밖이면서 그 다각형에서 band_max_m 안인 바닥.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
C = yaml.safe_load((HERE / "config" / "policy.yaml").read_text(encoding="utf-8"))
TB = Path(C["teammate_bundle"])
sys.path.insert(0, str(TB / "1_required"))
sys.path.insert(0, str(TB / "2_optional"))
import pixel_to_map as p2m  # noqa: E402  (팀원 코드, 바꾸지 않음)
import clearance_check as cc  # noqa: E402  (팀원 코드, 바꾸지 않음)


class Cam:
    def __init__(self, set_name):
        K, _ = p2m.intrinsics()
        p = C["photo_sets"][set_name]["pose"]
        if p == "photo5":
            rv, tv, _ = p2m.load_pose("photo5")
            self.K, self.rv, self.tv = K, np.asarray(rv, float), np.asarray(tv, float)
        else:
            j = json.loads(Path(p).read_text(encoding="utf-8"))
            self.K, self.rv, self.tv = np.array(j["K_photo"], float), np.array(j["rvec"], float), np.array(j["tvec"], float)
        self.name = set_name

    def to_map(self, px, z=0.0):
        return p2m.pixel_to_map(np.asarray(px, float).reshape(-1, 2), self.rv, self.tv, self.K, None, z)

    def to_px(self, xy, z=0.0):
        return p2m.map_to_pixel(np.asarray(xy, float).reshape(-1, 2), self.rv, self.tv, self.K, None, z)


def outer_polygon():
    """노란 선 안쪽 가장자리 다각형 A-B-C-D (지도 m, 팀원 기준점)"""
    r = p2m.ref_points()
    return np.array([r["A"], r["B"], r["C"], r["D"]], float)


def zones():
    return json.loads((HERE / "zones" / "zones_env2.json").read_text(encoding="utf-8"))


def _poly_dist(xy, poly):
    """점 -> 다각형 경계 최단 거리 (m), 안이면 음수"""
    p = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    return -cv2.pointPolygonTest(p, (float(xy[0]), float(xy[1])), True)


def workspace(xy, band_max_m, excluded_ids):
    """물체 점 xy(지도 m)가 작업 공간 띠 안이고 금지 구역 밖인가 -> dict"""
    d_out = _poly_dist(xy, outer_polygon())               # 노란 선 밖이면 +, 안이면 -
    zs = zones()
    inside = [z["id"] for z in zs["zones"] if _poly_dist(xy, z["polygon_map_m"]) < 0]
    bad = [i for i in inside if i in excluded_ids]
    ok = (0.0 < d_out <= band_max_m) and not bad
    why = ("노란 선 안(주행 구역)" if d_out <= 0 else f"노란 선에서 {d_out * 100:.0f} cm (한도 {band_max_m * 100:g} cm)" if d_out > band_max_m
           else f"금지 구역 {bad} 안" if bad else f"노란 선 밖 {d_out * 100:.1f} cm, 금지 구역 밖")
    return {"ok": bool(ok), "beyond_line_cm": round(d_out * 100, 1), "in_zones": inside, "why": why}


def zone_address(xy, named):
    """가장 가까운 이름 있는 구역과 거리 (cm). named = {구역 id: 이름}"""
    zs = {z["id"]: z for z in zones()["zones"]}
    best = min(((max(_poly_dist(xy, zs[i]["polygon_map_m"]), 0.0), i) for i in named if i in zs), default=(None, None))
    return {"nearest_zone": best[1], "nearest_zone_name": named.get(best[1]), "gap_cm": None if best[0] is None else round(best[0] * 100, 1)}


def band_rois_px(cam, band_max_m, tile_px, img_size):
    """작업 공간 띠를 사진 위 사각형 타일들로 (DINO를 확대해 다시 돌릴 영역). 지도 1 cm 격자 -> 픽셀 -> 타일 격자"""
    poly = outer_polygon()
    xs, ys = np.meshgrid(np.arange(0.30, 3.50, 0.01), np.arange(0.00, 1.75, 0.01))
    pts = np.c_[xs.ravel(), ys.ravel()]
    d = np.array([_poly_dist(p, poly) for p in pts])
    band = pts[(d > 0) & (d <= band_max_m)]
    px = cam.to_px(band, 0.0)
    W, H = img_size
    px = px[(px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)]
    tiles = set((int(u // tile_px), int(v // tile_px)) for u, v in px)
    half = tile_px // 2           # 타일 경계에 걸친 물체를 위해 사방으로 반 타일 넓힘
    return [[max(0, i * tile_px - half), max(0, j * tile_px - half), min(W, (i + 1) * tile_px + half), min(H, (j + 1) * tile_px + half)] for i, j in sorted(tiles)]
