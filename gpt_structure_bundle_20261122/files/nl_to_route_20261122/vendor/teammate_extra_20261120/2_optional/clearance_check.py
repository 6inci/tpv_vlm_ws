#!/usr/bin/env python3
"""경로 몸체 여유 검사 (2026-11-20, 메카넘 로봇 주행 쪽에서 만듦)

주행 쪽 경로 도구(route/zone_editor.py)의 판정 계산을 따로 떼어 낸 것이다. 같은 지도·같은 값이면 같은 결과가 나온다
(주행 쪽 원본과 결과가 같은지 2026-11-20 에 대조함).
필요: Python 3.8+, numpy, opencv-python (cv2). 이 파일은 2_optional/ 폴더에 있고 1_required/map/ 의 지도를 읽는다.

판정
- 경로(출발 -> 경유점 -> 도착 꺾은선)를 1 cm 간격으로 나눈다.
- 자리마다 footprint 둘레 점(0.5 cm 간격)을 그 자리의 몸체 방향으로 놓고, 지도 경계(못 가는 칸)까지 최소 거리 = 몸체 여유.
  둘레 점이나 중심이 못 가는 칸에 있으면 -1 (닿음).
- 불가 = 여유 < footprint_padding(1 cm) 또는 로봇 중심 칸의 Nav2 비용 >= 253 (중심이 경계에서 inscribed 반경 안)
  빠듯함 = 여유 < 3 cm, 좋음 = 3 cm 이상
- 방향: 출발 방향과 도착 방향이 다르면 회전 자리(rotate_at 의 경유점, 없으면 도착점)부터 도착 방향.
  회전 자리는 2° 마다 돌린 몸체의 여유 중 최소값.
- 주의: footprint 에 그리퍼가 없다 (robot_params.json). 지도에 노란 선 밖 물체가 없다.
- 그리퍼: 물체를 잡으려면 그리퍼가 노란 선 너머로 나가야 하므로, 그리퍼는 지도 경계 판정(불가·빠듯함)에 넣지 않는다.
  대신 '그리퍼 끝 선'을 따로 보여 준다: 차체 앞 robot_params.json 의 gripper_tip_from_base_link_estimate_m(추정) 거리에서
  차체 폭 전체 너비의 선분 (그리퍼는 차체 폭 안쪽이라 보수적으로 넓게 잡음).
  - 경로 내내(1 cm 마다 그 자리 방향으로)와 회전 자리에서 2° 마다, 이 선분이 노란 선(지도 경계) 안쪽으로 몇 cm 인지(음수 = 넘음)
  - 선을 넘기 시작하는 곳 (출발에서 경로를 따라 몇 cm)
  - 출발·도착 자리의 그리퍼 끝 가운데 점
  그리퍼 끝(19.5 cm 추정)이 몸체 반대각선(19.0 cm)보다 길어서, 회전 때 몸체 여유만 보면 그리퍼 쪽은 좋게 나온다. 회전 줄을 함께 본다.
  노란 선 밖 물체(큐브·팔레트 등)와의 거리는 지도에 물체가 없어서 계산하지 못한다. 사진에서 따로 봐야 한다.
- 경로 파일 필수 항목: start_xy_m, waypoints_xy_m(없으면 []), goal_xy_m, start_yaw_deg, goal_yaw_deg, rotate_at.
  방향은 0·90·180·270 중 하나. rotate_at = "W1".."W<경유점 수>" (회전하는 경유점) · "도착점" · null(회전 없음).
  틀리면 멈춘다 (주행 쪽 도구와 같은 규칙).

명령
  python3 clearance_check.py routes/팀원시작_큐브파지.json            경로 파일 검사 (형식은 routes/ 의 예시와 같음)
  python3 clearance_check.py --pose 2.678 0.4822 270                  한 자리 검사 (x y 방향)
  --json   결과를 json 으로 출력
"""
import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MAP_YAML = HERE.parent / "1_required" / "map" / "env2_20261119.yaml"
PARAMS = HERE / "robot_params.json"


def read_map_yaml(p):
    """map_server yaml (단순 'key: value' 형식)"""
    d = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.strip().startswith("#"):
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    d["origin"] = [float(t) for t in d["origin"].strip("[]").split(",")]
    for k in ("resolution", "occupied_thresh", "free_thresh"):
        d[k] = float(d[k])
    d["negate"] = int(d.get("negate", 0))
    return d


class Env:
    def __init__(self, yaml_path=MAP_YAML):
        my = read_map_yaml(yaml_path)
        self.res = my["resolution"]
        self.ox, self.oy = my["origin"][0], my["origin"][1]
        g = cv2.imread(str(Path(yaml_path).parent / my["image"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise SystemExit(f"지도 그림을 읽지 못했습니다: {my['image']}")
        occ = g / 255.0 if my["negate"] else (255.0 - g) / 255.0
        self.free = occ < my["free_thresh"]
        self.NY, self.NX = self.free.shape
        self.dt = cv2.distanceTransform(self.free.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.dt_out = cv2.distanceTransform((~self.free).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    def side_at(self, xs, ys):
        """점마다 주행 영역 안이면 +(경계까지 cm), 밖이면 -(경계 넘어 cm). 지도 밖은 -999. 칸 중심 기준, ±0.5 cm"""
        r, c, ok = self._rc(xs, ys)
        out = np.full(np.shape(xs), -999.0)
        rr, cc = r[ok], c[ok]
        fr = self.free[rr, cc]
        out[ok] = np.where(fr, self.dt[rr, cc] * self.res * 100 - 0.5, -self.dt_out[rr, cc] * self.res * 100 + 0.5)
        return out

    def side_cm(self, x, y):
        v = float(self.side_at(np.array([x]), np.array([y]))[0])
        return None if v <= -999 else round(v, 1)

    def _rc(self, xs, ys):
        c = np.floor((np.asarray(xs, float) - self.ox) / self.res).astype(int)
        r = self.NY - 1 - np.floor((np.asarray(ys, float) - self.oy) / self.res).astype(int)
        return r, c, (r >= 0) & (r < self.NY) & (c >= 0) & (c < self.NX)

    def free_at(self, xs, ys):
        r, c, ok = self._rc(xs, ys)
        out = np.zeros(np.shape(xs), bool)
        out[ok] = self.free[r[ok], c[ok]]
        return out

    def dist_at(self, xs, ys):
        c = np.clip((np.asarray(xs, float) - self.ox) / self.res - 0.5, 0, self.NX - 1.001)
        r = np.clip(self.NY - 0.5 - (np.asarray(ys, float) - self.oy) / self.res, 0, self.NY - 1.001)
        c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
        fc, fr = c - c0, r - r0
        d = self.dt
        return (d[r0, c0] * (1 - fc) * (1 - fr) + d[r0, c0 + 1] * fc * (1 - fr)
                + d[r0 + 1, c0] * (1 - fc) * fr + d[r0 + 1, c0 + 1] * fc * fr) * self.res

    def cost_at(self, xs, ys, R):
        r, c, ok = self._rc(xs, ys)
        d = np.zeros(np.shape(xs))
        d[ok] = self.dt[r[ok], c[ok]] * self.res
        cost = np.floor(252 * np.exp(-R.k * (d - R.insc)))
        cost = np.where(d > R.r_inf, 0, cost)
        cost = np.where(d <= R.insc, 253, cost)
        return np.where(d <= 0, 254, cost).astype(int)


def _origin_to_segment(a, b):
    ab = b - a
    t = float(np.clip(-(a @ ab) / (ab @ ab), 0, 1))
    return float(np.linalg.norm(a + t * ab))


def _perimeter(poly, step):
    pts = []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        n = int(np.linalg.norm(b - a) / step) + 2
        pts.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
    return np.vstack(pts)


class Robot:
    def __init__(self, params=PARAMS):
        p = json.loads(Path(params).read_text(encoding="utf-8"))
        self.fp = np.array(p["footprint_m"], float)
        self.pad = float(p["footprint_padding_m"])
        self.r_inf, self.k = float(p["inflation_radius_m"]), float(p["cost_scaling_factor"])
        self.tight = float(p["tool_rules"]["tight_cm"]) / 100.0
        self.turn_step = float(p["tool_rules"]["turn_step_deg"])
        self.tip = p.get("gripper_tip_from_base_link_estimate_m")
        hw = float(np.abs(self.fp[:, 1]).max())
        self.tip_line = (np.c_[np.full(25, self.tip), np.linspace(-hw, hw, 25)] if self.tip is not None else None)
        padded = self.fp + np.sign(self.fp) * self.pad
        self.insc = min(_origin_to_segment(a, b) for a, b in zip(padded, np.roll(padded, -1, 0)))
        self.perimeter = _perimeter(self.fp, 0.005)


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def densify_idx(pts, step=0.01):
    pts = [np.asarray(p, float) for p in pts]
    out, vidx = [pts[0]], [0]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t for t in np.arange(1, n + 1) / n]
        vidx.append(len(out) - 1)
    return np.array(out), vidx


def body_clear(E, R, P, yaw_deg):
    th = math.radians(yaw_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    Q = P[:, None, :] + (R.perimeter @ rot.T)[None, :, :]
    inside = E.free_at(Q[..., 0], Q[..., 1]).all(1) & E.free_at(P[:, 0], P[:, 1])
    return np.where(inside, E.dist_at(Q[..., 0], Q[..., 1]).min(1) - 0.5 * E.res, -1.0)


def turn_angles(a, b, step):
    d = wrap_deg(b - a)
    n = max(1, int(math.ceil(abs(d) / step)))
    return a + d * np.arange(n + 1) / n


def route_check(E, R, pts, yaw_s, yaw_g=None, rot_k=None):
    yaw_g = yaw_s if yaw_g is None else yaw_g
    P, vidx = densify_idx(pts, 0.01)
    turn = abs(wrap_deg(yaw_g - yaw_s)) > 1e-6
    if turn and rot_k is not None and not 0 <= rot_k < len(pts) - 2:
        rot_k = None
    ri = (vidx[rot_k + 1] if rot_k is not None else vidx[-1]) if turn else None
    after = np.arange(len(P)) >= (ri if turn else len(P))
    clear = body_clear(E, R, P, yaw_s)
    rot_clear = None
    if turn:
        clear[after] = body_clear(E, R, P[after], yaw_g)
        rot_clear = float(min(body_clear(E, R, P[ri:ri + 1], a)[0] for a in turn_angles(yaw_s, yaw_g, R.turn_step)))
        clear[ri] = min(clear[ri], rot_clear)
    cost = E.cost_at(P[:, 0], P[:, 1], R)
    status = np.where((clear < R.pad) | (cost >= 253), 2, np.where(clear < R.tight, 1, 0))
    i = int(np.argmin(clear))
    return {"length_m": float(np.linalg.norm(np.diff(np.asarray(pts, float), axis=0), axis=1).sum()),
            "n_points": int(len(P)), "min_body_clearance_cm": round(float(clear[i]) * 100, 2),
            "min_at_xy_m": [round(float(v), 4) for v in P[i]],
            "rotate_xy_m": [round(float(v), 4) for v in P[ri]] if turn else None,
            "rotate_min_clearance_cm": round(rot_clear * 100, 2) if turn else None,
            "max_center_cost": int(cost.max()), "bad_cm": int((status == 2).sum()), "tight_cm": int((status == 1).sum())}


def tip_line_side(E, R, P, yaw_deg):
    """자리 P (N x 2) 마다 그리퍼 끝 선분이 노란 선 안쪽으로 몇 cm 인지 (최소, 음수 = 넘음)"""
    th = math.radians(yaw_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    Q = P[:, None, :] + (R.tip_line @ rot.T)[None, :, :]
    return E.side_at(Q[..., 0], Q[..., 1]).min(1)


def gripper_route(E, R, pts, ys, yg, rot_k):
    """경로 내내와 회전 중 그리퍼 끝 선 (route_check 와 같은 구간 나눔)"""
    if R.tip_line is None:
        return None
    P, vidx = densify_idx(pts, 0.01)
    turn = abs(wrap_deg(yg - ys)) > 1e-6
    ri = (vidx[rot_k + 1] if rot_k is not None else vidx[-1]) if turn else len(P)
    side = np.empty(len(P))
    side[:ri] = tip_line_side(E, R, P[:ri], ys) if ri > 0 else side[:ri]
    if turn:
        side[ri:] = tip_line_side(E, R, P[ri:], yg)
    i = int(np.argmin(side))
    out = {"path_min_inside_cm": round(float(side[i]), 1), "path_min_at_xy_m": [round(float(v), 4) for v in P[i]],
           "crosses_line_from_start_cm": None, "rotation_min_inside_cm": None}
    neg = np.where(side < 0)[0]
    if len(neg):
        out["crosses_line_from_start_cm"] = round(float(np.linalg.norm(np.diff(P[:neg[0] + 1], axis=0), axis=1).sum()) * 100, 1)
    if turn:
        out["rotation_min_inside_cm"] = round(float(min(tip_line_side(E, R, P[ri:ri + 1], a)[0] for a in turn_angles(ys, yg, R.turn_step))), 1)
    return out


def gripper_tip(E, R, x, y, yaw_deg):
    """그리퍼 끝 추정 위치와 노란 선(지도 경계) 안팎 거리 cm (+ 안 / - 넘음). 추정값이 없으면 None"""
    if R.tip is None:
        return None
    th = math.radians(yaw_deg)
    tx, ty = x + R.tip * math.cos(th), y + R.tip * math.sin(th)
    return {"tip_xy_m": [round(tx, 4), round(ty, 4)], "tip_inside_cm": E.side_cm(tx, ty)}


REQUIRED = ("start_xy_m", "waypoints_xy_m", "goal_xy_m", "start_yaw_deg", "goal_yaw_deg", "rotate_at")


def load_route(f):
    """경로 파일을 읽고 형식을 검사 -> (꺾은선, 출발 방향, 도착 방향, 회전 경유점 번호(0부터) 또는 None)"""
    js = json.loads(Path(f).read_text(encoding="utf-8"))
    miss = [k for k in REQUIRED if k not in js]
    if miss:
        raise SystemExit(f"{Path(f).name}: 필수 항목 없음 {miss} (waypoints_xy_m 은 경유점이 없으면 [], rotate_at 은 회전이 없으면 null)")
    if not isinstance(js["waypoints_xy_m"], list):
        raise SystemExit(f"{Path(f).name}: waypoints_xy_m 은 [[x, y], ...] 목록이어야 합니다")
    ys, yg = float(js["start_yaw_deg"]), float(js["goal_yaw_deg"])
    for lab, a in (("start_yaw_deg", ys), ("goal_yaw_deg", yg)):
        if abs(a / 90.0 - round(a / 90.0)) > 1e-6:
            raise SystemExit(f"{Path(f).name}: {lab} = {a:g}. 방향은 0, 90, 180, 270 중 하나여야 합니다")
    at, nw = js["rotate_at"], len(js["waypoints_xy_m"])
    m = re.fullmatch(r"W(\d+)", str(at or ""))
    if at not in (None, "도착점") and not (m and 1 <= int(m.group(1)) <= nw):
        raise SystemExit(f"{Path(f).name}: rotate_at = {at!r}. 'W1'~'W{nw}', '도착점', null 중 하나여야 합니다")
    pts = [js["start_xy_m"]] + js["waypoints_xy_m"] + [js["goal_xy_m"]]
    return pts, ys, yg, (int(m.group(1)) - 1 if m else None)


def check_route_file(E, R, f):
    pts, ys, yg, rot_k = load_route(f)
    js = json.loads(Path(f).read_text(encoding="utf-8"))
    out = route_check(E, R, pts, ys, yg, rot_k)
    out.update({"file": Path(f).name, "start_yaw_deg": ys, "goal_yaw_deg": yg, "rotate_at": js.get("rotate_at"),
                "gripper_at_start": gripper_tip(E, R, *pts[0], ys), "gripper_at_goal": gripper_tip(E, R, *pts[-1], yg),
                "gripper_line": gripper_route(E, R, pts, ys, yg, rot_k)})
    return out


def cm_text(v):
    return "닿음(몸체가 지도 경계 밖)" if v is not None and v < 0 else f"{v} cm"


def tip_text(g):
    if not g:
        return ""
    s = g["tip_inside_cm"]
    where = "지도 밖" if s is None else (f"노란 선 안쪽 {s} cm" if s >= 0 else f"노란 선 넘어 {-s} cm")
    return f"그리퍼 끝(추정) {g['tip_xy_m']} = {where}"


def main():
    ap = argparse.ArgumentParser(description="경로 몸체 여유 검사 (지도 env2_20261119)")
    ap.add_argument("routes", nargs="*", help="경로 json 파일")
    ap.add_argument("--pose", nargs=3, type=float, metavar=("X", "Y", "YAW"), help="한 자리 검사")
    ap.add_argument("--map", default=str(MAP_YAML))
    ap.add_argument("--params", default=str(PARAMS))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    E, R = Env(a.map), Robot(a.params)
    res = []
    if a.pose:
        x, y, yaw = a.pose
        c = float(body_clear(E, R, np.array([[x, y]]), yaw)[0])
        cost = int(E.cost_at(np.array([x]), np.array([y]), R)[0])
        res.append({"pose": [x, y, yaw], "body_clearance_cm": round(c * 100, 2), "center_cost": cost,
                    "status": "불가" if (c < R.pad or cost >= 253) else ("빠듯함" if c < R.tight else "좋음"),
                    "gripper": gripper_tip(E, R, x, y, yaw)})
    for f in a.routes:
        res.append(check_route_file(E, R, f))
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    for r in res:
        if "pose" in r:
            print(f"자리 {r['pose']}: 몸체 여유 {cm_text(r['body_clearance_cm'])}, 중심 비용 {r['center_cost']} -> {r['status']}")
            if r["gripper"]:
                print("  " + tip_text(r["gripper"]))
        else:
            print(f"{r['file']}: 길이 {r['length_m'] * 100:.1f} cm, 방향 {r['start_yaw_deg']:g}° -> {r['goal_yaw_deg']:g}°"
                  + (f" ({r['rotate_at'] or '도착점'} 에서 회전, 회전 여유 {cm_text(r['rotate_min_clearance_cm'])})" if r["rotate_min_clearance_cm"] is not None else "")
                  + f", 몸체 여유 최소 {cm_text(r['min_body_clearance_cm'])} @ {r['min_at_xy_m']}, 중심 비용 최대 {r['max_center_cost']},"
                  f" 불가 {r['bad_cm']} cm, 빠듯함 {r['tight_cm']} cm")
            for lab, g in (("출발", r["gripper_at_start"]), ("도착", r["gripper_at_goal"])):
                if g:
                    print(f"  {lab}: " + tip_text(g))
            gl = r["gripper_line"]
            if gl:
                print(f"  그리퍼 끝 선(추정, 차체 폭 전체): 경로 중 노란 선 안쪽 최소 {gl['path_min_inside_cm']} cm @ {gl['path_min_at_xy_m']} (음수 = 넘음)"
                      + (f", 출발에서 {gl['crosses_line_from_start_cm']} cm 지점부터 선을 넘음" if gl["crosses_line_from_start_cm"] is not None else ", 선을 넘지 않음")
                      + (f", 회전 중 최소 {gl['rotation_min_inside_cm']} cm" if gl["rotation_min_inside_cm"] is not None else ""))


if __name__ == "__main__":
    main()
