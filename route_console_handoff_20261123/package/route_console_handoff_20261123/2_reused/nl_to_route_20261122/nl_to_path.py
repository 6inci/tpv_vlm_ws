#!/usr/bin/env python3
"""자연어 지시 -> A/B 인식·판단 -> env2 지도 좌표 -> 정차 자세 -> 경로계획 -> 사진 위 경로 그림 (설정: config/nl_to_path.yaml).

좌표
- 사진 픽셀: 공유 사진 묶음(env2_photos_20261120)의 편 사진 픽셀. K_photo(alpha 0 newK), 왜곡 계수 없음.
- 지도: env2_20261119 (m), 바닥 z = 0. yaw = +x에서 반시계, 차체 앞(그리퍼 쪽) 방향.
역할
- Qwen3-VL-4B: 지시 글자 -> 정해진 항목만. DINO: 후보 박스. SigLIP2: 예시 이미지로 큐브/빈 팔레트 판별. (hybrid_exemplar_20261121 그대로)
- Python: A/B 결정, 픽셀 -> 지도, 정차 자세, 경로, 검사(팀원 clearance_check.py), 그림.
실행 (모델은 GPU 필요)
  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY -B nl_to_path.py --photo <png> [--instruction "..."]          한 장
  $PY -B nl_to_path.py --all [--instruction "..."]                  공유 사진 15장
  $PY -B nl_to_path.py --photo <png> --ab-px x0,y0,x1,y1 x0,y0,x1,y1  모델 없이 A/B 박스를 주고 좌표·경로만 (시험용)
"""
import argparse
import hashlib
import heapq
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config" / "nl_to_path.yaml").read_text(encoding="utf-8"))
TB = Path(CFG["teammate_bundle"])
sys.path.insert(0, str(TB / "1_required"))
sys.path.insert(0, str(TB / "2_optional"))
import pixel_to_map as p2m  # noqa: E402  (팀원 코드, 바꾸지 않음)
import clearance_check as cc  # noqa: E402  (팀원 코드, 바꾸지 않음)

DIRS = [(0, 1), (-1, 0), (0, -1), (1, 0)]      # (dr, dc): 0 = +x, 1 = +y, 2 = -x, 3 = -y  (행은 위쪽이 +y)


# ================================================================================ 사진 <-> 지도
class Geo:
    def __init__(self):
        self.pose = json.loads(Path(CFG["photo_set"]["pose"]).read_text(encoding="utf-8"))
        self.K = np.array(self.pose["K_photo"], float)
        self.rv, self.tv = np.array(self.pose["rvec"], float), np.array(self.pose["tvec"], float)

    def to_map(self, px, z):
        return p2m.pixel_to_map(np.asarray(px, float).reshape(-1, 2), self.rv, self.tv, self.K, None, z)

    def to_px(self, xy, z=0.0):
        return p2m.map_to_pixel(np.asarray(xy, float).reshape(-1, 2), self.rv, self.tv, self.K, None, z)


# ================================================================================ 정차 자세와 경로 (env2 지도, 팀원 판정 계산 재사용)
class Planner:
    def __init__(self):
        self.E, self.R = cc.Env(), cc.Robot()
        E, R, pc = self.E, self.R, CFG["planner"]
        self.rows, self.cols = np.nonzero(E.free)
        self.xy = np.c_[E.ox + (self.cols + 0.5) * E.res, E.oy + (E.NY - 1 - self.rows + 0.5) * E.res]
        self.clear = {}                       # 칸 중심에 섰을 때 몸체 여유 (m). 0 = 0/180°, 1 = 90/270° (footprint가 대칭)
        for k, yaw in ((0, 0.0), (1, 90.0)):
            m = np.full(E.free.shape, -1.0)
            m[self.rows, self.cols] = np.concatenate([cc.body_clear(E, R, self.xy[i:i + 3000], yaw) for i in range(0, len(self.xy), 3000)])
            self.clear[k] = m
        self.cost = np.full(E.free.shape, 254)
        self.cost[self.rows, self.cols] = E.cost_at(self.xy[:, 0], self.xy[:, 1], R)
        hw = float(np.abs(R.fp[:, 1]).max())
        self.half_diag = float(np.hypot(*np.abs(R.fp).max(0)))
        self.grip_r = math.hypot(R.tip, hw) if R.tip is not None else 0.0
        self.rot_need_m = max(self.half_diag + pc["rotate_body_margin_m"], self.grip_r + pc["rotate_gripper_inside_m"])
        self.rot_ok = E.free & ((E.dt * E.res - 0.5 * E.res) >= self.rot_need_m) & (self.cost < 253)

    def cell(self, x, y):
        E = self.E
        return E.NY - 1 - int(math.floor((y - E.oy) / E.res)), int(math.floor((x - E.ox) / E.res))

    def centre(self, r, c):
        E = self.E
        return E.ox + (c + 0.5) * E.res, E.oy + (E.NY - 1 - r + 0.5) * E.res

    @staticmethod
    def kind(yaw):
        return 0 if round(yaw) % 180 == 0 else 1

    def stop_pose(self, target, half_depth):
        """물체 점 target(지도 m) 앞 정차 자세. 몸체 여유·중심 비용을 만족하고 물체가 그리퍼 끝 앞에 있는 칸 중,
        그리퍼 끝 가운데 -> 물체 앞면 점 거리(앞 모자람, 옆 어긋남)가 가장 작은 곳. 옆 어긋남 기준은 표시만 한다."""
        sp, R = CFG["stop_pose"], self.R
        best = None
        for yaw in sp["yaws_deg"]:
            k, th = self.kind(yaw), math.radians(yaw)
            u, n = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
            d = np.asarray(target, float) - self.xy
            fwd, lat = d @ u, d @ n
            gap = fwd - R.tip - half_depth
            clr = self.clear[k][self.rows, self.cols]
            ok = (clr >= sp["min_body_clearance_m"]) & (self.cost[self.rows, self.cols] < 253) & (gap >= sp["min_reach_gap_m"])
            if not ok.any():
                continue
            dist = np.where(ok, np.hypot(gap, lat), np.inf)
            i = int(np.argmin(dist))
            if best is None or dist[i] < best[0]:
                best = (float(dist[i]), yaw, i, float(gap[i]), float(lat[i]), float(clr[i]))
        if best is None:
            return None
        dist, yaw, i, gap, lat, clr = best
        x, y = self.xy[i]
        g = cc.gripper_tip(self.E, R, x, y, yaw)
        lat_ok = abs(lat) <= sp["max_lateral_offset_m"]
        return {"xy_m": [round(float(x), 4), round(float(y), 4)], "yaw_deg": yaw, "cell_rc": [int(self.rows[i]), int(self.cols[i])],
                "body_clearance_cm": round(clr * 100, 2), "center_cost": int(self.cost[self.rows[i], self.cols[i]]),
                "reach_gap_cm": round(gap * 100, 1), "lateral_offset_cm": round(lat * 100, 1), "tip_to_object_cm": round(dist * 100, 1),
                "lateral_within_limit": lat_ok, "within_reach": dist <= sp["max_tip_to_object_m"], "gripper_tip": g,
                "grasp_possible": "UNKNOWN (그리퍼 도달 거리·폭 실측 없음)",
                "rule": f"몸체 여유 >= {sp['min_body_clearance_m'] * 100:g} cm 인 칸 중 그리퍼 끝 -> 물체 앞면 거리 최소. "
                        f"옆 어긋남 {sp['max_lateral_offset_m'] * 100:g} cm 초과는 표시만"}

    def plan(self, start, s_yaw, goal, g_yaw):
        """A* (1 cm 칸, 4방향). 차체 방향은 출발 방향 -> (한 번 제자리 회전) -> 도착 방향. 옆걸음·꺾기·회전에 비용."""
        pc, E, R = CFG["planner"], self.E, self.R
        NY, NX, res = E.NY, E.NX, E.res
        (sr, sc), (gr, gc) = self.cell(*start), self.cell(*goal)
        turn = (round(s_yaw) - round(g_yaw)) % 360 != 0
        ks, kg = self.kind(s_yaw), self.kind(g_yaw)
        ok = {k: (self.clear[k] >= pc["min_body_clearance_m"]) & (self.cost < 253) for k in (0, 1)}
        rr, cc_ = np.mgrid[0:NY, 0:NX]
        near_start = np.hypot((rr - sr) * res, (cc_ - sc) * res) <= pc["start_relax_radius_m"]
        relax = {k: near_start & (self.clear[k] >= R.pad) & (self.cost < 253) for k in (0, 1)}
        pas = {0: ok[ks] | relax[ks], 1: (ok[kg] | relax[kg]) if turn else None}
        if not pas[0][sr, sc]:
            return {"status": "START_BLOCKED", "why": f"출발 칸 몸체 여유 {self.clear[ks][sr, sc] * 100:.1f} cm"}
        g_ph = 1 if turn else 0
        if not (ok[kg][gr, gc] or (turn is False and pas[0][gr, gc])):
            return {"status": "GOAL_BLOCKED", "why": f"도착 칸 몸체 여유 {self.clear[kg][gr, gc] * 100:.1f} cm"}
        sf, tcst, rcst = pc["strafe_cost_factor"], pc["turn_cost_m"], pc["rotate_cost_m"]
        S = 2 * 5 * NY * NX
        g = np.full(S, np.inf)
        par = np.full(S, -1, np.int64)
        enc = lambda ph, d, r, c: ((ph * 5 + d) * NY + r) * NX + c          # noqa: E731
        h = lambda ph, r, c: (abs(r - gr) + abs(c - gc)) * res + (rcst if turn and ph == 0 else 0.0)   # noqa: E731
        s0 = enc(0, 4, sr, sc)
        g[s0] = 0.0
        pq = [(h(0, sr, sc), 0.0, 0, 4, sr, sc)]
        found, n_pop = None, 0
        while pq:
            f, gs, ph, d, r, c = heapq.heappop(pq)
            s = enc(ph, d, r, c)
            if gs > g[s]:
                continue
            n_pop += 1
            if r == gr and c == gc and ph == g_ph:
                found = s
                break
            k = (kg if ph == 1 else ks)
            if turn and ph == 0 and self.rot_ok[r, c] and ok[kg][r, c] and (pc["allow_rotate_at_start"] or (r, c) != (sr, sc)):
                t, ng = enc(1, 4, r, c), gs + rcst
                if ng < g[t]:
                    g[t], par[t] = ng, s
                    heapq.heappush(pq, (ng + h(1, r, c), ng, 1, 4, r, c))
            for nd, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < NY and 0 <= nc < NX) or not pas[ph][nr, nc]:
                    continue
                ng = gs + res * (sf if (nd % 2) != k else 1.0) + (tcst if d not in (4, nd) else 0.0)
                t = enc(ph, nd, nr, nc)
                if ng < g[t]:
                    g[t], par[t] = ng, s
                    heapq.heappush(pq, (ng + h(ph, nr, nc), ng, ph, nd, nr, nc))
        if found is None:
            return {"status": "NO_PATH", "why": f"조건을 만족하는 경로 없음 (탐색 {n_pop})"}
        seq, s = [], found
        while s >= 0:
            c = s % NX
            r = (s // NX) % NY
            d = (s // (NX * NY)) % 5
            ph = s // (NX * NY * 5)
            seq.append((int(ph), int(d), int(r), int(c)))
            s = int(par[s])
        seq.reverse()
        # 꺾는 칸과 회전 칸만 경유점으로
        wps, rot_wp, prev_d = [], None, None
        for i in range(1, len(seq)):
            ph0, _, r0, c0 = seq[i - 1]
            ph1, d1, r1, c1 = seq[i]
            if ph1 != ph0:                                   # 제자리 회전
                if (r1, c1) == (gr, gc):
                    rot_wp = "도착점"
                else:
                    if not wps or wps[-1] != (r1, c1):
                        wps.append((r1, c1))
                    rot_wp = len(wps)
                prev_d = None
                continue
            if prev_d is not None and d1 != prev_d and (r0, c0) != (sr, sc):
                if not wps or wps[-1] != (r0, c0):
                    wps.append((r0, c0))
            prev_d = d1
        pts = [self.centre(sr, sc)] + [self.centre(r, c) for r, c in wps] + [self.centre(gr, gc)]
        rot_k = (rot_wp - 1) if isinstance(rot_wp, int) else None
        chk = cc.route_check(E, R, pts, s_yaw, g_yaw, rot_k)
        grip = cc.gripper_route(E, R, pts, s_yaw, g_yaw, rot_k)
        # 옆걸음 길이 (구간 방향이 그 구간의 차체 방향과 다르면 옆걸음)
        strafe, longest = 0.0, 0.0
        for j, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            yaw = g_yaw if (isinstance(rot_wp, int) and j >= rot_wp) else s_yaw      # 구간 j = pts[j] -> pts[j+1], 회전은 W{rot_wp}에서
            seg = np.subtract(b, a)
            L = float(np.hypot(*seg))
            if L > 0 and (abs(seg[0]) > abs(seg[1])) != (self.kind(yaw) == 0):
                strafe += L
                longest = max(longest, L)
        route = {"start_xy_m": [round(v, 4) for v in pts[0]], "waypoints_xy_m": [[round(v, 4) for v in p] for p in pts[1:-1]],
                 "goal_xy_m": [round(v, 4) for v in pts[-1]], "start_yaw_deg": float(s_yaw), "goal_yaw_deg": float(g_yaw),
                 "rotate_at": (f"W{rot_wp}" if isinstance(rot_wp, int) else rot_wp) if turn else None}
        bad = chk["bad_cm"] > 0 or chk["min_body_clearance_cm"] < R.pad * 100
        return {"status": "BLOCKED_BY_CHECK" if bad else "OK", "route": route, "check_teammate": chk, "gripper_line": grip,
                "strafe_total_cm": round(strafe * 100, 1), "strafe_longest_cm": round(longest * 100, 1), "n_expanded": n_pop,
                "why": "팀원 판정으로 불가 구간 있음" if bad else "팀원 판정 통과 (불가 0)"}


# ================================================================================ 출발 자세 (로봇은 15장 모두 같은 자리)
def start_pose(geo, planner, img):
    sc = CFG["start"]
    m = sc["marker"]
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, m["dictionary"])), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    shift = None
    if ids is not None and m["id"] in ids.ravel().tolist():
        cs = corners[ids.ravel().tolist().index(m["id"])][0]
        shift = float(np.abs(cs - np.array(m["corners_px"], float)).max())
    xy = geo.to_map([sc["pixel"]], sc["z_m"])[0]
    r, c = planner.cell(*xy)
    sx, sy = planner.centre(r, c)
    ok = shift is not None and shift <= m["max_shift_px"]
    return {"status": "OK" if ok else "ROBOT_MOVED_OR_MARKER_LOST", "xy_m": [round(sx, 4), round(sy, 4)], "yaw_deg": sc["yaw_deg"],
            "marker_shift_px": None if shift is None else round(shift, 1), "pixel": sc["pixel"], "z_m": sc["z_m"],
            "body_clearance_cm": round(float(planner.clear[planner.kind(sc["yaw_deg"])][r, c]) * 100, 2),
            "how": "설정의 바퀴 중심 픽셀을 z_m 높이로 투영 (추정). 사진의 ArUco 마커가 설정 꼭짓점에서 max_shift_px 안일 때만 씀"}


# ================================================================================ 인식 (기존 혼합 구조 그대로)
def make_viewer(session):
    sys.path.insert(0, CFG["perception"]["pipeline"])
    import hybrid_pipeline as HP

    class Viewer(HP.HybridViewer):
        if CFG["perception"]["skip_old_camera_checks"]:
            def zone_frac(self, ph, box, zone_ids):
                return None

            def access(self, ph, c):
                return {"accessible": None, "why": "옛 카메라 기준 검사는 이 사진 묶음에 맞지 않음 -> env2 정차 자세 단계에서 검사"}
    return Viewer(session_dir=session)


def perceive(viewer, photo, instruction):
    r = viewer.run(str(photo), instruction)
    s, d, sl = r["source"], r["destination"], (r.get("slots") or {}).get("parsed")
    A = {"status": s.get("status"), "why": s.get("why"), "box_px": (s.get("boxes") or [None])[0], "id": (s.get("ids") or [None])[0],
         "class": s.get("class"), "exemplar": s.get("exemplar")}
    kind = (sl or {}).get("destination_kind")
    B = {"status": d.get("status"), "why": d.get("why"), "box_px": d.get("bbox_xyxy"), "id": d.get("id"), "exemplar": d.get("exemplar"),
         "class": "empty_pallet" if kind == "empty_pallet" else "floor_point"}
    return {"slots": sl, "slots_raw_file": str(Path(r["files"]["run_json"]).parent / "qwen" / "slots.txt"), "A": A, "B": B,
            "stages": r["stages"], "first_failure": r["first_failure"], "timing_s": r.get("timing_s"), "wall_s": r["wall_s"],
            "run_json": r["files"]["run_json"]}


# ================================================================================ 그림 (사진 위, 한국어)
def footprint_px(geo, R, x, y, yaw):
    th = math.radians(yaw)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    body = np.array([x, y]) + R.fp @ rot.T
    tip = np.array([x, y]) + R.tip_line[[0, -1]] @ rot.T
    return geo.to_px(body), geo.to_px(tip)


def draw(img, geo, R, rec, out):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    f = ImageFont.truetype(CFG["draw"]["font"], 22)
    fb = ImageFont.truetype(CFG["draw"]["font_bold"], 24)
    P = lambda a: [tuple(map(float, p)) for p in np.asarray(a).reshape(-1, 2)]   # noqa: E731
    col = {"A": (235, 40, 40, 255), "B": (40, 110, 255, 255), "r1": (30, 220, 60, 255), "r2": (255, 150, 0, 255), "stopA": (255, 60, 220, 255),
           "stopB": (0, 220, 255, 255), "start": (255, 255, 255, 255)}
    for key, lab in (("A", "A 옮길 것"), ("B", "B 놓을 곳")):
        o = rec["perception"][key]
        if o.get("box_px"):
            x0, y0, x1, y1 = o["box_px"]
            dr.rectangle([x0, y0, x1, y1], outline=col[key], width=3)
            dr.text((x0, y0 - 30), lab, font=fb, fill=col[key], stroke_width=2, stroke_fill=(0, 0, 0, 255))
    for k, rk in (("route_1", "r1"), ("route_2", "r2")):
        pl = rec.get(k) or {}
        if pl.get("status") not in ("OK", "BLOCKED_BY_CHECK"):
            continue
        rt = pl["route"]
        pts = [rt["start_xy_m"]] + rt["waypoints_xy_m"] + [rt["goal_xy_m"]]
        px = geo.to_px(pts) + (0 if rk == "r1" else 7)        # 두 경로가 겹치면 ②를 7 px 비켜 그림 (그림에서만)
        dr.line(P(px), fill=col[rk], width=5, joint="curve")
        for j, p in enumerate(px[1:-1], 1):
            u, v = p
            dr.ellipse([u - 9, v - 9, u + 9, v + 9], fill=col[rk], outline=(0, 0, 0, 255), width=2)
            tag = f"{'①' if rk == 'r1' else '②'}W{j}" + (" 회전" if rt["rotate_at"] == f"W{j}" else "")
            pos = (u - 12 - dr.textlength(tag, font=f), v - 34) if rk == "r1" else (u + 12, v + 4)
            dr.text(pos, tag, font=f, fill=col[rk], stroke_width=3, stroke_fill=(0, 0, 0, 255))
    for key, ck in (("stop_A", "stopA"), ("stop_B", "stopB")):
        sp = rec.get(key)
        if sp:
            body, tip = footprint_px(geo, R, *sp["xy_m"], sp["yaw_deg"])
            dr.polygon(P(body), outline=col[ck], width=3)
            dr.line(P(tip), fill=col[ck], width=5)
            u, v = geo.to_px([sp["xy_m"]])[0]
            dr.text((u - 40, v - 12), "파지 정차" if key == "stop_A" else "배치 정차", font=f, fill=col[ck], stroke_width=3, stroke_fill=(0, 0, 0, 255))
    st = rec.get("start")
    if st:
        body, tip = footprint_px(geo, R, *st["xy_m"], st["yaw_deg"])
        dr.polygon(P(body), outline=col["start"], width=2)
        u, v = geo.to_px([st["xy_m"]])[0]
        dr.text((u - 18, v - 40), "출발", font=fb, fill=col["start"], stroke_width=3, stroke_fill=(0, 0, 0, 255))
    lines = rec["summary_lines"]
    h = 16 + 30 * len(lines)
    canvas = Image.new("RGB", (im.size[0], im.size[1] + h), (20, 20, 20))      # 글은 사진 아래에 붙여 사진을 가리지 않음
    canvas.paste(Image.alpha_composite(im, lay).convert("RGB"), (0, 0))
    dt = ImageDraw.Draw(canvas)
    for i, t in enumerate(lines):
        dt.text((14, im.size[1] + 8 + 30 * i), t, font=f, fill=(255, 255, 255))
    canvas.save(out, quality=92)


# ================================================================================ 한 번 실행
KO = {"SELECTED": "확정", "AMBIGUOUS": "보류(후보 여럿)", "NOT_FOUND": "찾지 못함", "NOT_ACCESSIBLE": "접근 불가", "NO_REFERENCE": "기준 시설 없음",
      "NOT_SUPPORTED": "지원 안 함", "NOT_RUN": "실행 안 됨", "GIVEN": "직접 줌"}


def run_one(photo, instruction, viewer, geo, planner, out_root, ab_px=None, reuse=None):
    t0 = time.time()
    photo = Path(photo)
    img = cv2.imread(str(photo), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    rec = {"created": datetime.now().isoformat(timespec="seconds"), "photo": str(photo), "photo_sha256": hashlib.sha256(photo.read_bytes()).hexdigest(),
           "instruction": instruction, "config": str(HERE / "config" / "nl_to_path.yaml"), "config_snapshot": CFG, "pose_file": CFG["photo_set"]["pose"],
           "frames": {"pixel": "편 사진 픽셀 (K_photo, 왜곡 없음)", "map": "env2_20261119 m, 바닥 z = 0", "yaw": "+x에서 반시계, 차체 앞 = 그리퍼"}}
    if reuse is not None:                       # 저장된 인식 결과를 그대로 쓰고 좌표·정차·경로·그림만 다시 계산
        old = json.loads((Path(reuse) / photo.stem / "nl_to_path.json").read_text(encoding="utf-8"))
        assert old["photo_sha256"] == rec["photo_sha256"] and old["instruction"] == instruction, "다른 사진이나 지시의 기록"
        rec["perception"] = {**old["perception"], "reused_from": str(Path(reuse) / photo.stem / "nl_to_path.json")}
    elif ab_px is None:
        rec["perception"] = perceive(viewer, photo, instruction)
    else:
        rec["perception"] = {"slots": None, "note": "모델 없이 A/B 박스를 직접 줌 (시험용)",
                             "A": {"status": "SELECTED", "why": "GIVEN", "box_px": ab_px[0], "class": "cube"},
                             "B": {"status": "SELECTED", "why": "GIVEN", "box_px": ab_px[1], "class": "empty_pallet"}}
    rec["start"] = start_pose(geo, planner, img)
    objs = {}
    for key in ("A", "B"):
        o = rec["perception"][key]
        if o["status"] == "SELECTED" and o.get("box_px"):
            spec = CFG["objects"][o["class"]]
            x0, y0, x1, y1 = o["box_px"]
            c_px = [(x0 + x1) / 2, (y0 + y1) / 2]
            xy = geo.to_map([c_px], spec["z_top_m"])[0]
            objs[key] = {"class": o["class"], "centre_px": [round(v, 1) for v in c_px], "z_top_m": spec["z_top_m"], "map_xy_m": [round(float(v), 4) for v in xy],
                         "inside_drive_area_cm": planner.E.side_cm(*xy), "projection": "박스 중심을 물체 윗면 높이 평면으로 투영"}
    rec["objects"] = objs
    rec["stop_A"] = planner.stop_pose(objs["A"]["map_xy_m"], CFG["objects"][objs["A"]["class"]]["half_depth_m"]) if "A" in objs else None
    rec["stop_B"] = planner.stop_pose(objs["B"]["map_xy_m"], CFG["objects"][objs["B"]["class"]]["half_depth_m"]) if "B" in objs else None
    for key in ("stop_A", "stop_B"):            # 그리퍼 끝에서 너무 먼 물체 (예: 매트 밖) -> 정차 자세로 쓰지 않음
        if rec[key] and not rec[key]["within_reach"]:
            rec[f"{key}_rejected"], rec[key] = rec[key], None
    rec["route_1"] = rec["route_2"] = None
    if rec["start"]["status"] == "OK" and rec["stop_A"]:
        rec["route_1"] = planner.plan(rec["start"]["xy_m"], rec["start"]["yaw_deg"], rec["stop_A"]["xy_m"], rec["stop_A"]["yaw_deg"])
    if rec["stop_A"] and rec["stop_B"]:
        rec["route_2"] = planner.plan(rec["stop_A"]["xy_m"], rec["stop_A"]["yaw_deg"], rec["stop_B"]["xy_m"], rec["stop_B"]["yaw_deg"])
    # 결과 판정
    p = rec["perception"]
    stage = ("A 인식" if p["A"]["status"] != "SELECTED" else "B 인식" if p["B"]["status"] != "SELECTED" else "출발 자세" if rec["start"]["status"] != "OK"
             else "파지 정차" if not rec["stop_A"] else "배치 정차" if not rec["stop_B"] else "경로 1" if rec["route_1"]["status"] != "OK"
             else "경로 2" if rec["route_2"]["status"] != "OK" else None)
    rec["result"] = {"status": "PATH_OK" if stage is None else "STOPPED", "stopped_at": stage}
    rec["wall_s"] = round(time.time() - t0, 1)
    sl = p.get("slots")
    L = [f"지시: {instruction}"]
    if sl:
        L.append(f"해석(Qwen, 글자만): 옮길 것 '{sl['object']}', 놓을 곳 {sl['destination_kind']}")
    for key, nm in (("A", "A 옮길 것"), ("B", "B 놓을 곳")):
        o, ob = p[key], objs.get(key)
        ex = o.get("exemplar") or {}
        sc = f" 예시 점수 {ex['s1']:.3f}" + (f", 2위와 차이 {ex['s1'] - ex['s2']:.3f}" if ex.get("s2") is not None else "") if ex.get("s1") is not None else ""
        L.append(f"{nm}: {KO.get(o['status'], o['status'])}{sc}" + (f" → 지도 ({ob['map_xy_m'][0]:.3f}, {ob['map_xy_m'][1]:.3f}) m, 높이 {ob['z_top_m'] * 100:g} cm" if ob else ""))
    for key, nm in (("stop_A", "파지 정차"), ("stop_B", "배치 정차")):
        sp, rj = rec.get(key), rec.get(f"{key}_rejected")
        if rj:
            L.append(f"{nm}: 물체가 그리퍼 끝에서 {rj['tip_to_object_cm']:.0f} cm (한도 {CFG['stop_pose']['max_tip_to_object_m'] * 100:g} cm) → 닿을 수 없음")
        if sp:
            L.append(f"{nm}: ({sp['xy_m'][0]:.3f}, {sp['xy_m'][1]:.3f}) m, {sp['yaw_deg']}°, 몸체 여유 {sp['body_clearance_cm']:.1f} cm, "
                     f"그리퍼 끝 → {'큐브 앞면' if key == 'stop_A' else '놓는 점'} 앞 {sp['reach_gap_cm']:.1f} cm 모자람, 옆 {sp['lateral_offset_cm']:+.1f} cm"
                     + ("" if sp["lateral_within_limit"] else " (옆 어긋남 큼)"))
    for key, nm in (("route_1", "경로 ① 출발→파지"), ("route_2", "경로 ② 파지→배치")):
        pl = rec.get(key)
        if pl and pl.get("route"):
            ck, rt = pl["check_teammate"], pl["route"]
            L.append(f"{nm}: {ck['length_m'] * 100:.0f} cm, 경유점 {len(rt['waypoints_xy_m'])}개, 회전 {rt['rotate_at'] or '없음'}, "
                     f"몸체 여유 최소 {ck['min_body_clearance_cm']:.1f} cm, 옆걸음 최장 {pl['strafe_longest_cm']:.0f} cm, 팀원 판정 불가 {ck['bad_cm']} cm")
        elif pl:
            L.append(f"{nm}: {pl['status']} - {pl['why']}")
    L.append("결과: 경로 그림까지 완료" if stage is None else f"결과: '{stage}' 단계에서 멈춤")
    rec["summary_lines"] = L
    od = out_root / photo.stem
    od.mkdir(parents=True, exist_ok=True)
    draw(img, geo, planner.R, rec, od / "path_overlay.jpg")
    rec["files"] = {"overlay": str(od / "path_overlay.jpg"), "record": str(od / "nl_to_path.json")}
    (od / "nl_to_path.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{photo.name}: {rec['result']['status']} {stage or ''} | " + " | ".join(L[2:-1]), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description="자연어 지시 -> 경로 그림")
    ap.add_argument("--photo")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--instruction", default=CFG["default_instruction"])
    ap.add_argument("--ab-px", nargs=2, help="모델 없이 A, B 박스 x0,y0,x1,y1 (시험용)")
    ap.add_argument("--reuse", help="저장된 실행 폴더의 인식 결과를 그대로 쓰고 좌표·정차·경로·그림만 다시 계산 (모델 호출 없음)")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = (Path(a.out) if a.out else HERE / "results" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()   # 모델 작업 프로세스는 다른 폴더에서 돌므로 절대 경로
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_snapshot.yaml").write_text((HERE / "config" / "nl_to_path.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    geo, planner = Geo(), Planner()
    photos = sorted(Path(CFG["photo_set"]["dir"]).glob("*.png")) if a.all else [Path(a.photo)]
    ab = [[float(v) for v in s.split(",")] for s in a.ab_px] if a.ab_px else None
    viewer = None if (ab or a.reuse) else make_viewer(out / "perception_session")
    recs = []
    try:
        for p in photos:
            recs.append(run_one(p, a.instruction, viewer, geo, planner, out, ab, a.reuse))
    finally:
        if viewer is not None:
            viewer.close()
    idx = [{"photo": Path(r["photo"]).name, "result": r["result"], "A": r["perception"]["A"]["status"], "B": r["perception"]["B"]["status"],
            "A_map": (r["objects"].get("A") or {}).get("map_xy_m"), "B_map": (r["objects"].get("B") or {}).get("map_xy_m"),
            "stop_A": r["stop_A"], "stop_B": r["stop_B"],
            "route_1": None if not r["route_1"] else {k: r["route_1"].get(k) for k in ("status", "route", "strafe_longest_cm")},
            "route_2": None if not r["route_2"] else {k: r["route_2"].get(k) for k in ("status", "route", "strafe_longest_cm")},
            "wall_s": r["wall_s"], "overlay": r["files"]["overlay"]} for r in recs]
    (out / "index.json").write_text(json.dumps(idx, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    n_ok = sum(r["result"]["status"] == "PATH_OK" for r in recs)
    print(f"\n경로 그림까지 완료 {n_ok}/{len(recs)} -> {out}")


if __name__ == "__main__":
    main()
