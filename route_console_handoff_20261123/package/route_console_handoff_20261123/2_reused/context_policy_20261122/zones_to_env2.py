#!/usr/bin/env python3
"""5S 구역 초안(semantic_zones_draft_v2.json, P01 원본 픽셀)을 env2 지도 좌표(m)로 옮긴다.

좌표 흐름: P01 원본 픽셀(왜곡 있음) -> cv2.undistortPoints(K, dist, P=K) -> P01 편 픽셀 -> 바닥 homography(SIFT + MAGSAC)
         -> 팀원 등록 사진 편 픽셀(newK = K) -> pixel_to_map(등록 자세, z = 0) -> env2 지도 m.
구역은 바닥 위 다각형으로 본다(z = 0). 확인용 그림: 시험 사진 1장과 개발 사진 1장 위에 다시 그린다.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -I -B zones_to_env2.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
C = yaml.safe_load((HERE / "config" / "policy.yaml").read_text(encoding="utf-8"))
TB = Path(C["teammate_bundle"])
sys.path.insert(0, str(TB / "1_required"))
import pixel_to_map as p2m  # noqa: E402  (팀원 코드, 바꾸지 않음)

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def pose_of(name):
    """사진 묶음 이름 -> (rvec, tvec, K)"""
    K, _ = p2m.intrinsics()
    p = C["photo_sets"][name]["pose"]
    if p == "photo5":
        rv, tv, _ = p2m.load_pose("photo5")
        return rv, tv, K
    j = json.loads(Path(p).read_text(encoding="utf-8"))
    return np.array(j["rvec"]), np.array(j["tvec"]), np.array(j["K_photo"])


def main():
    K, dist = p2m.intrinsics()
    src = json.loads(Path(C["zones_source"]["file"]).read_text(encoding="utf-8"))
    p01 = cv2.imread(C["zones_source"]["image"])
    p01u = cv2.undistort(p01, K, dist, None, K)
    reg = cv2.imread(str(TB / C["zones_source"]["register_image"]))
    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create(nfeatures=20000)
    m = np.zeros(reg.shape[:2], np.uint8)
    m[150:1040, 100:1820] = 255
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(p01u, cv2.COLOR_BGR2GRAY), m)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY), m)
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
    pa, pb = np.float32([k1[a.queryIdx].pt for a in good]), np.float32([k2[a.trainIdx].pt for a in good])
    H, inl = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, 2.0, maxIters=20000, confidence=0.9999)
    inl = inl.ravel().astype(bool)
    res = np.linalg.norm(cv2.perspectiveTransform(pa[inl][:, None], H)[:, 0] - pb[inl], axis=1)
    print(f"P01 편 사진 -> 등록 사진 homography: {inl.sum()}/{len(good)} 점, 중앙 잔차 {np.median(res):.2f} px")
    rv, tv, _ = p2m.load_pose("register-undist")

    def to_map(pts_raw):
        u = cv2.undistortPoints(np.float32(pts_raw).reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 1, 2)
        r = cv2.perspectiveTransform(u, H).reshape(-1, 2)
        return p2m.pixel_to_map(r, rv, tv, K, None, 0.0)

    zones = []
    for z in src["zones"]:
        xy = to_map(z["polygon_xy"])
        zones.append({"id": z["id"], "name_ko": z["name_ko"], "role": z["role"], "polygon_map_m": np.round(xy, 4).tolist(),
                      "semantic_origin": z.get("semantic_origin"), "access_draft": z.get("access_draft"), "source_polygon_px_P01_raw": z["polygon_xy"]})
    lms = [{"id": l["id"], "name_ko": l["name_ko"], "role": l["role"], "point_map_m": np.round(to_map([l["point_xy"]])[0], 4).tolist()} for l in src["landmarks"]]
    out = {"status": "DRAFT (원본 구역 초안이 DRAFT, 사용자 확인 필요)", "frame": "env2_20261119 map m, floor z = 0", "source": C["zones_source"],
           "source_sha256_image": src.get("image_sha256"), "homography_P01undist_to_register": {"H": H.tolist(), "inliers": int(inl.sum()),
                                                                                                 "median_residual_px": round(float(np.median(res)), 3)},
           "zones": zones, "landmarks": lms}
    (HERE / "zones" / "zones_env2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # 확인 그림: 시험 사진 1장, 개발 사진 1장
    col = {"Z01": (40, 220, 60), "Z02": (255, 200, 0), "Z03": (255, 140, 0), "Z04": (255, 140, 0), "Z05": (230, 60, 230), "Z06": (40, 120, 255),
           "Z07": (235, 40, 40), "Z08": (235, 40, 40), "Z09": (255, 105, 180), "Z10": (240, 240, 240)}
    f = ImageFont.truetype(FONT, 22)
    for name, photo in (("test", "20261120_215746_view.png"), ("dev", "5calibration.png")):
        rvp, tvp, Kp = pose_of(name)
        im = Image.open(Path(C["photo_sets"][name]["dir"]) / photo).convert("RGB")
        d = ImageDraw.Draw(im)
        for z in zones:
            px = p2m.map_to_pixel(np.array(z["polygon_map_m"]), rvp, tvp, Kp, None)
            d.polygon([tuple(p) for p in px], outline=col[z["id"]], width=4)
            c = px.mean(0)
            d.text((c[0] - 60, c[1] - 12), f"{z['id']} {z['name_ko']}", font=f, fill=col[z["id"]], stroke_width=3, stroke_fill=(0, 0, 0))
        for l in lms:
            u, v = p2m.map_to_pixel(np.array([l["point_map_m"]]), rvp, tvp, Kp, None)[0]
            d.ellipse([u - 10, v - 10, u + 10, v + 10], outline=(255, 0, 0), width=4)
            d.text((u + 12, v - 12), f"{l['id']} {l['name_ko']}", font=f, fill=(255, 80, 80), stroke_width=3, stroke_fill=(0, 0, 0))
        im.save(HERE / "zones" / f"zones_on_{name}_{Path(photo).stem}.jpg", quality=90)
    print("saved zones/zones_env2.json and overlays")


if __name__ == "__main__":
    main()
