#!/usr/bin/env python3
"""경로 계획 콘솔의 계산 부분: 자연어 지시 + 사진 1장 -> 출발 -> (파지 정차) -> 도착 정차 경로와 경유점.

다섯 단계 (그림 vlm_five_stages.pdf 와 같은 순서)
  1 지시 이해   Qwen3-VL-4B, 글자만 -> 정해진 항목 (config interpretation). 원문 응답 보존, 형식 검사
  2 장면 인식   Grounding DINO 후보 + SigLIP2 예시 비교. 사진 모드는 저장된 인식 자료(context_policy data/test)를 다시 씀,
                실시간 모드는 그 자리에서 계산 (GPU)
  3 대상 결정   Python 규칙: 작업 공간, 5S 금지 구역, 받침, 점유, 1·2위 차이, '장소 옆' (context_policy policy.py 그대로 + 장소 조건)
  4 좌표 변환   카메라 자세로 박스 중심 -> env2 지도 (물체 윗면 높이)
  5 경로 생성   정차 자세 + 4방향 A* (nl_to_path.Planner 그대로), 팀원 경로 파일 형식으로 저장, 팀원 검사 코드로 다시 확인
보류: 대상이나 목적지를 확정하지 못하면 경로를 만들지 않는다.

좌표: 사진 픽셀 = 공유 사진 묶음의 편 사진 픽셀(K_photo). 지도 = env2_20261119 m, yaw = +x에서 반시계, 차체 앞 = 그리퍼.
기존 폴더는 읽기만 한다 (import 만). 쓰는 파일은 이 폴더의 runs/, sessions/ 아래.
"""
import copy
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config" / "console.yaml").read_text(encoding="utf-8"))
RU = CFG["reuse"]
sys.path.insert(0, RU["nl_to_route"])
sys.path.insert(0, RU["context_policy"])
import nl_to_path as N  # noqa: E402  (읽기만: Geo, Planner, start_pose, footprint_px)
import common as CM  # noqa: E402  (읽기만: Cam, zones, workspace, outer_polygon)
import policy as POL  # noqa: E402  (읽기만: load_set, score_photo, evaluate_candidates, decide)

cc = N.cc                                   # 팀원 clearance_check (바꾸지 않음)
HP = POL.HP                                 # hybrid_pipeline (exemplar_scores, bank_for)
PLACES = CFG["places"]
PLACE_IDS = list(PLACES)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONTB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
STAGES = [
    {"key": "interpret", "title": "지시 이해", "question": "무엇을 어디로 옮길까?", "who": "Qwen3-VL · 텍스트 해석", "kind": "model"},
    {"key": "perceive", "title": "장면 인식", "question": "무엇이 어디에 있을까?", "who": "DINO · SigLIP2 예시 비교", "kind": "model"},
    {"key": "decide", "title": "대상 결정", "question": "어떤 물체와 목적지를 고를까?", "who": "상태 · 공간 관계 · 작업 제약", "kind": "rule"},
    {"key": "project", "title": "좌표 변환", "question": "지도에서 어디일까?", "who": "카메라 기하 · 픽셀 → 미터", "kind": "rule"},
    {"key": "plan", "title": "경로 생성", "question": "어디에 서고 어떻게 이동할까?", "who": "실행 가능성 검사 · 경로계획", "kind": "rule"},
]


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def why_ko(w):
    """hybrid_pipeline.decide 의 영어 이유 -> 화면용 한국어 (원문은 기록에 그대로 남음)"""
    w = str(w or "")
    if w.startswith("no candidate"):
        return "조건을 모두 통과한 후보가 없음"
    m = re.match(r"top-1 ([\d.]+), margin ([\d.]+) >= ([\d.]+)", w)
    if m:
        return f"1위 점수 {m.group(1)}, 2위와 차이 {m.group(2)} (기준 {m.group(3)} 이상)"
    m = re.match(r"top-1 ([\d.]+), no other object", w)
    if m:
        return f"1위 점수 {m.group(1)}, 경쟁 후보 없음"
    m = re.match(r"top-1 ([\d.]+) vs a different object ([\d.]+): margin < ([\d.]+)", w)
    if m:
        return f"1위 {m.group(1)}와 다른 물체 {m.group(2)}의 차이가 기준 {m.group(3)}보다 작음"
    m = re.match(r"top-1 score ([\d.]+) < ([\d.]+)", w)
    if m:
        return f"1위 점수 {m.group(1)}가 기준 {m.group(2)}보다 낮음"
    return w


# ================================================================================ 1 지시 이해
SLOT_ENUMS = {
    "task": ["transport", "go_to"],
    "object": ["red cube", "any object", "none", "other"],
    "object_near": PLACE_IDS + ["none", "other"],
    "start": PLACE_IDS + ["none", "other"],
    "destination_kind": ["empty_pallet", "place", "other"],
    "destination_place": PLACE_IDS + ["none", "other"],
}


def parse_slots(text):
    """Qwen 응답 -> (항목 dict, 오류). JSON 객체 하나, 모든 항목이 정해진 값 중 하나여야 한다."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None, "응답에 JSON이 없음"
    try:
        js = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"JSON 형식 오류: {e}"
    if not isinstance(js, dict):
        return None, "JSON 객체가 아님"
    out = {}
    for k, allowed in SLOT_ENUMS.items():
        v = js.get(k)
        if v not in allowed:
            return None, f"{k} = {v!r} (허용: {', '.join(allowed)})"
        out[k] = v
    out["words"] = str(js.get("words", ""))
    return out, None


def task_plan(sl):
    """항목 -> 실행 계획 또는 보류 이유. 해석이 서로 맞지 않거나 지원하지 않는 요청이면 경로를 만들지 않는다.
    보정 규칙(기록함): 옮길 물체가 없는 이동인데 '물체 옆 장소'만 있고 출발 장소가 비었으면, 그 장소를 출발 장소로 본다.
      예) "하역장에서 충전소로 이동해" -> Qwen이 하역장을 object_near에 넣은 경우 (2026-11-22 사용자 실행에서 확인)"""
    p = {"task": sl["task"], "object": sl["object"], "near_zone": None, "near_place": None, "start_place": None,
         "dest_kind": sl["destination_kind"], "dest_place": None, "corrections": []}
    if sl["task"] == "go_to" and sl["object"] == "none" and sl["object_near"] not in ("none", "other") and sl["start"] == "none":
        sl = {**sl, "start": sl["object_near"], "object_near": "none"}
        p["corrections"].append(f"옮길 물체가 없어 '{PLACES[sl['start']]['name']}'을(를) 출발 장소로 봄 (해석에서는 물체 옆 장소로 나옴)")
    if sl["task"] == "transport":
        if sl["object"] == "none":
            return None, "옮길 물체가 없는데 '옮기기'로 해석됨 (해석이 서로 맞지 않음)"
        if sl["object"] not in CFG["graspable_objects"]:
            return None, "이 로봇이 집을 수 있는 물체는 빨간 큐브뿐"
    elif sl["object"] not in ("none",):
        return None, "물체를 말했는데 '로봇만 이동'으로 해석됨 (해석이 서로 맞지 않음)"
    if sl["object_near"] not in ("none",):
        pl = PLACES.get(sl["object_near"])
        if not pl or not pl["zone"]:
            return None, f"'{sl['object_near']}' 옆을 판단할 구역이 지도에 없음"
        p["near_zone"], p["near_place"] = pl["zone"], sl["object_near"]
    if sl["start"] not in ("none",):
        pl = PLACES.get(sl["start"])
        if not pl or not pl["stop"]:
            return None, f"출발 장소 '{sl['start']}'에 정해진 정차 자리가 없음"
        p["start_place"] = sl["start"]
    if sl["destination_kind"] == "other":
        return None, "도착지가 빈 팔레트나 지도에 있는 장소가 아님"
    if sl["destination_kind"] == "place":
        if sl["destination_place"] in ("none", "other"):
            return None, "도착지를 말하지 않았거나 지도에 없는 장소임"
        pl = PLACES.get(sl["destination_place"])
        if not pl or not pl["stop"]:
            return None, f"도착 장소 '{sl['destination_place']}'에 정해진 정차 자리가 없음"
        p["dest_place"] = sl["destination_place"]
    return p, None


# ================================================================================ 엔진
class Engine:
    def __init__(self, progress=print):
        self.progress = progress
        self.geo, self.planner = N.Geo(), N.Planner()
        self.cam = CM.Cam("test")
        self.thr = json.loads(Path(RU["thresholds"]).read_text(encoding="utf-8"))["thresholds"]
        meta, self.bank, self.ce, self.se = POL.load_set("test")          # 저장된 인식 자료 (사진 모드)
        self.cached = {p["sha256"]: p for p in meta["photos"]}
        self.refs = {r["photo_id"]: r["objects"] for r in HP.RT.load_reference()}
        tz = json.loads(Path(RU["teammate_zones"]).read_text(encoding="utf-8"))
        self.tstops = {z["name"]: z for z in tz["zones"]}
        self.zones = {z["id"]: z for z in CM.zones()["zones"]}
        self.host, self.host_state, self.host_lock = None, {"status": "대기", "detail": ""}, threading.Lock()
        self.session = None

    # ---------------------------------------------------------------- 모델 (기존 혼합 구조의 작업 프로세스를 그대로 씀)
    def ensure_host(self):
        with self.host_lock:
            if self.host is None:
                self.host_state = {"status": "불러오는 중", "detail": "모델 작업 프로세스 시작"}
                spec = importlib.util.spec_from_file_location("cp_extract", Path(RU["context_policy"]) / "extract.py")
                EXM = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(EXM)
                self.EXM = EXM
                self.session = HERE / "sessions" / datetime.now().strftime("%Y%m%d_%H%M%S")
                self.session.parent.mkdir(exist_ok=True)

                class Host(EXM.Extractor):
                    def zone_frac(self, ph, box, zone_ids):          # 옛 카메라 기준 검사는 이 사진 묶음에 맞지 않음 (nl_to_path 와 같음)
                        return None

                    def access(self, ph, c):
                        return {"accessible": None, "why": "옛 카메라 기준 검사 생략"}
                self.host = Host(self.session)
                self.host.progress = lambda m: self.progress(m)
                self.host.bank = self.bank                         # 예시 묶음은 저장된 것 (같은 기준 박스, 같은 모델)
            return self.host

    def warm_up(self):
        try:
            h = self.ensure_host()
            self.host_state = {"status": "불러오는 중", "detail": "Qwen3-VL-4B 적재"}
            h.model().ensure_loaded("qwen")
            self.host_state = {"status": "준비됨", "detail": "Qwen3-VL-4B (GPU)"}
        except Exception as e:                                   # noqa: BLE001  (화면에 그대로 보여 줌)
            self.host_state = {"status": "오류", "detail": f"{type(e).__name__}: {e}"}

    def interpret(self, instruction, log):
        h = self.ensure_host()
        ic = CFG["interpretation"]
        prompt = ic["template"].replace("{instruction}", instruction)
        t = time.time()
        raw = h.vlm_text(prompt, ic["max_new_tokens"], "console_slots", ic["prompt_id"], log)
        self.host_state = {"status": "준비됨", "detail": "Qwen3-VL-4B (GPU)"}
        sl, err = parse_slots(raw)
        return {"prompt_id": ic["prompt_id"], "prompt": prompt, "raw": raw, "slots": sl, "error": err, "seconds": round(time.time() - t, 1)}

    # ---------------------------------------------------------------- 2 장면 인식
    def perceive(self, photo, live, log):
        sha = sha256_file(photo)
        if not live and sha in self.cached:
            ph = self.cached[sha]
            src = "저장된 인식 자료 (context_policy data/test, 같은 사진·같은 모델로 미리 계산)"
            ce, se = self.ce, self.se
        else:
            ph, ce, se = self._extract(photo, log)
            src = "이번에 계산 (DINO + 띠 확대 DINO + SigLIP2 so400m)"
        cands, s, binfo = POL.score_photo(ph, self.bank, ce, se, self.refs)
        rows = POL.evaluate_candidates(self.cam, cands, s, self.thr)
        return {"photo_id": ph["photo_id"], "source": src, "n_dino": len(ph["candidates"]), "n_gated": len(cands), "bank": binfo, "rows": rows}

    def _extract(self, photo, log):
        h = self.ensure_host()
        EXM, EX = self.EXM, self.EXM.EX
        self.host_state = {"status": "계산 중", "detail": "DINO, SigLIP2"}
        ph = h.photo(str(photo))
        rois, dets = h.band_dets(ph, self.cam, ph["calls"])
        h._add(ph, dets)
        cands = h.gated(ph, log)
        pid, views = ph["photo_id"], h.hc["exemplar"]["views"]
        ce = {f"{pid}|{x['id']}": h.xemb[(pid, x["id"])] for x in cands}
        crops = {x["id"]: h._crops(ph, f"{x['id']}_sup", EXM.sup_box(x["bbox_xyxy"], EX["support_scale"])) for x in cands}
        E = h._embed([crops[x["id"]][v]["path"] for x in cands for v in views], f"support_{pid}", log)
        se = {f"{pid}|{x['id']}": E[i * len(views):(i + 1) * len(views)] for i, x in enumerate(cands)}
        gated = {x["id"] for x in cands}
        phd = {"photo_id": pid, "path": str(photo), "candidates": [{**{k: x.get(k) for k in ("id", "bbox_xyxy", "score", "phrases", "conditions")},
                                                                    "gated": x["id"] in gated} for x in ph["dino"]]}
        self.host_state = {"status": "준비됨", "detail": "Qwen3-VL-4B (GPU)"}
        return phd, ce, se

    # ---------------------------------------------------------------- 3 대상 결정
    def decide(self, plan, perc):
        rows = copy.deepcopy(perc["rows"])
        out = {"A": None, "B": None}
        if plan["task"] == "transport":
            if plan["near_zone"]:
                poly = self.zones[plan["near_zone"]]["polygon_map_m"]
                for r in rows.values():
                    d = CM._poly_dist(r["cube_map_xy_m"], poly) * 100
                    ok = d <= CFG["near_max_cm"]
                    r["A_checks"].append({"check": "장소 옆", "ok": bool(ok),
                                          "detail": f"{PLACES[plan['near_place']]['name']} 구역까지 {max(d, 0):.0f} cm (한도 {CFG['near_max_cm']} cm)"})
                    if r["A_pass"] and not ok:
                        r["A_pass"], r["A_failed_at"] = False, "장소 옆"
            out["A"] = POL.decide(rows, "A", "cube")
        if plan["dest_kind"] == "empty_pallet":
            out["B"] = POL.decide(rows, "B", "empty_pallet")
        else:
            pl = PLACES[plan["dest_place"]]
            out["B"] = {"status": "SELECTED", "why": f"지도의 장소 '{pl['name']}' (팀원 정차 자리)", "id": None, "box": None, "place": plan["dest_place"]}
        out["rows"] = rows
        return out

    # ---------------------------------------------------------------- 4·5 좌표와 경로
    def _snap(self, xy, yaw):
        """정차 칸이 몸체 여유 기준을 넘는지 보고, 아니면 goal_snap_max_cm 안의 가장 가까운 통과 칸으로 옮김"""
        P = self.planner
        k = P.kind(yaw)
        okm = (P.clear[k] >= CFG_N["planner"]["min_body_clearance_m"]) & (P.cost < 253)
        r, c = P.cell(*xy)
        if 0 <= r < okm.shape[0] and 0 <= c < okm.shape[1] and okm[r, c]:
            return r, c, 0.0
        rr, cc_ = np.nonzero(okm)
        if not len(rr):
            return None
        d = np.hypot(rr - r, cc_ - c) * self.planner.E.res
        i = int(np.argmin(d))
        if d[i] * 100 > CFG["goal_snap_max_cm"]:
            return None
        return int(rr[i]), int(cc_[i]), float(d[i] * 100)

    def place_pose(self, place_id, facing=True):
        """장소의 팀원 정차 자리. 방향 = 되는 방향 중 그 5S 구역을 바라보는 것 (구역이 없으면 첫 방향)."""
        pl = PLACES[place_id]
        z = self.tstops[pl["stop"]]
        x, y = z["assigned_xy_m"]
        heads = z["ok_headings"]
        if facing and pl["zone"]:
            poly = np.asarray(self.zones[pl["zone"]]["polygon_map_m"], float)
            # 구역 경계에서 가장 가까운 점 (변마다 투영)
            best = None
            for a, b in zip(poly, np.roll(poly, -1, 0)):
                ab = b - a
                t = float(np.clip(np.dot([x, y] - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
                q = a + t * ab
                d = float(np.hypot(*(q - [x, y])))
                if best is None or d < best[0]:
                    best = (d, q)
            ang = math.degrees(math.atan2(best[1][1] - y, best[1][0] - x)) % 360
            yaw = min(heads, key=lambda h: abs((h - ang + 180) % 360 - 180))
        else:
            ang, yaw = None, min(heads, key=lambda h: abs((h - CFG["start_heading_preference_deg"] + 180) % 360 - 180))
        sn = self._snap((x, y), yaw)
        if sn is None:
            return None
        r, c, moved = sn
        sx, sy = self.planner.centre(r, c)
        P = self.planner
        face_err = None if ang is None else round(abs((yaw - ang + 180) % 360 - 180), 1)
        return {"xy_m": [round(sx, 4), round(sy, 4)], "yaw_deg": float(yaw), "place": place_id, "place_name": pl["name"],
                "teammate_stop": pl["stop"], "teammate_xy_m": [x, y], "snapped_cm": round(moved, 1), "zone_direction_deg": None if ang is None else round(ang, 1),
                "facing_error_deg": face_err, "ok_headings": heads,
                "body_clearance_cm": round(float(P.clear[P.kind(yaw)][r, c]) * 100, 2), "center_cost": int(P.cost[r, c]),
                "gripper_tip": cc.gripper_tip(P.E, P.R, sx, sy, yaw), "rule": CFG["place_drop_rule"]}

    def geometry(self, plan, dec, perc, img):
        P = self.planner
        g = {"start": None, "objects": {}, "stops": {}, "legs": []}
        # 출발
        if plan["start_place"]:
            sp = self.place_pose(plan["start_place"], facing=False)
            g["start"] = None if sp is None else {**sp, "status": "OK", "how": f"지시의 출발 장소 '{sp['place_name']}' (팀원 정차 자리, 가정)"}
        else:
            g["start"] = N.start_pose(self.geo, P, img)
        # 물체 좌표 (박스 중심을 물체 윗면 높이로)
        for key, cls in (("A", "cube"), ("B", "empty_pallet")):
            d = dec.get(key)
            if d and d["status"] == "SELECTED" and d.get("box"):
                spec = N.CFG["objects"][cls]
                c = [(d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2]
                xy = self.geo.to_map([c], spec["z_top_m"])[0]
                g["objects"][key] = {"class": cls, "centre_px": [round(v, 1) for v in c], "z_top_m": spec["z_top_m"],
                                     "map_xy_m": [round(float(v), 4) for v in xy], "beyond_line_cm": round(-P.E.side_cm(*xy), 1)}
        # 정차 자세
        if "A" in g["objects"]:
            s = P.stop_pose(g["objects"]["A"]["map_xy_m"], N.CFG["objects"]["cube"]["half_depth_m"])
            g["stops"]["A"] = s if (s and s["within_reach"]) else None
            g["stops"]["A_rejected"] = s if (s and not s["within_reach"]) else None
        if dec.get("B") and dec["B"]["status"] == "SELECTED":
            if dec["B"].get("place"):
                g["stops"]["B"] = self.place_pose(dec["B"]["place"])
            elif "B" in g["objects"]:
                s = P.stop_pose(g["objects"]["B"]["map_xy_m"], N.CFG["objects"]["empty_pallet"]["half_depth_m"])
                g["stops"]["B"] = s if (s and s["within_reach"]) else None
                g["stops"]["B_rejected"] = s if (s and not s["within_reach"]) else None
        # 경로 (출발 -> 파지 정차 -> 도착 정차, 이동만이면 출발 -> 도착 정차)
        names = {"start": (PLACES[plan["start_place"]]["name"] if plan["start_place"] else "현재위치"),
                 "A": "큐브파지", "B": (PLACES[dec["B"]["place"]]["name"] if dec.get("B") and dec["B"].get("place") else "팔레트배치")}
        seq = ["start"] + (["A"] if plan["task"] == "transport" else []) + ["B"]
        poses = {"start": g["start"] if (g["start"] and g["start"]["status"] == "OK") else None, "A": g["stops"].get("A"), "B": g["stops"].get("B")}
        for a, b in zip(seq[:-1], seq[1:]):
            if not poses[a] or not poses[b]:
                g["legs"].append({"from": names[a], "to": names[b], "status": "NOT_RUN", "why": "앞 단계의 자리가 없음"})
                break
            pl = P.plan(poses[a]["xy_m"], poses[a]["yaw_deg"], poses[b]["xy_m"], poses[b]["yaw_deg"])
            g["legs"].append({"from": names[a], "to": names[b], **pl})
            if pl["status"] != "OK":
                break
        g["names"] = names
        return g

    # ---------------------------------------------------------------- 화면용 좌표 (사진 픽셀, 지도 m)
    def overlay(self, dec, perc, g):
        geo, R = self.geo, self.planner.R
        px = lambda a, z=0.0: np.round(geo.to_px(np.asarray(a, float).reshape(-1, 2), z), 1).tolist()   # noqa: E731
        ov = {"zones": [{"id": z["id"], "name": z.get("name_ko") or z["id"], "polygon_px": px(z["polygon_map_m"]), "polygon_m": z["polygon_map_m"],
                         "excluded": z["id"] in POL.PC["excluded_zones"]} for z in self.zones.values() if z["id"] != "Z02"],
              "drive_px": px(CM.outer_polygon()), "drive_m": CM.outer_polygon().round(4).tolist(), "candidates": {}, "targets": {}, "poses": {}, "legs": []}
        for key, cls in (("A", "cube"), ("B", "empty_pallet")):
            d = dec.get(key)
            if not d:
                continue
            top = []
            for t in (d.get("top_before_policy") or [])[:3]:
                r = dec["rows"][t["id"]]
                top.append({"id": t["id"], "box": r["bbox_xyxy"], "score": round(t["score"], 3), "failed_at": t["failed_at"]})
            ov["candidates"][key] = top
            if d["status"] == "SELECTED" and d.get("box"):
                ov["targets"][key] = {"box": d["box"], "map_xy_m": g["objects"].get(key, {}).get("map_xy_m")}
        for key, pose in (("start", g["start"]), ("A", g["stops"].get("A")), ("B", g["stops"].get("B"))):
            if pose and pose.get("xy_m") and (key != "start" or pose.get("status") == "OK"):
                body, tip = N.footprint_px(geo, R, *pose["xy_m"], pose["yaw_deg"])
                th = math.radians(pose["yaw_deg"])
                rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
                ov["poses"][key] = {"xy_m": pose["xy_m"], "yaw_deg": pose["yaw_deg"], "body_px": np.round(body, 1).tolist(), "tip_px": np.round(tip, 1).tolist(),
                                    "body_m": (np.array(pose["xy_m"]) + R.fp @ rot.T).round(4).tolist(),
                                    "tip_m": (np.array(pose["xy_m"]) + R.tip_line[[0, -1]] @ rot.T).round(4).tolist()}
        for i, L in enumerate(g["legs"]):
            if L.get("route"):
                rt = L["route"]
                pts = [rt["start_xy_m"]] + rt["waypoints_xy_m"] + [rt["goal_xy_m"]]
                ov["legs"].append({"index": i + 1, "from": L["from"], "to": L["to"], "points_m": pts, "points_px": px(pts),
                                   "rotate_at": rt["rotate_at"], "status": L["status"]})
        return ov

    # ---------------------------------------------------------------- 저장 (팀원 경로 파일 형식 + 경유점 표 + 그림)
    def export(self, rec, img, out):
        out.mkdir(parents=True, exist_ok=True)
        files, checks = [], []
        rob = json.loads((Path(N.CFG["teammate_bundle"]) / "2_optional" / "robot_params.json").read_text(encoding="utf-8"))
        rows = []
        for i, L in enumerate(rec["geometry"]["legs"]):
            if not L.get("route"):
                continue
            rt, ck = L["route"], L["check_teammate"]
            pts = [rt["start_xy_m"]] + rt["waypoints_xy_m"] + [rt["goal_xy_m"]]
            js = {"start_xy_m": rt["start_xy_m"], "waypoints_xy_m": rt["waypoints_xy_m"], "goal_xy_m": rt["goal_xy_m"],
                  "start_yaw_deg": rt["start_yaw_deg"], "goal_yaw_deg": rt["goal_yaw_deg"], "rotate_at": rt["rotate_at"],
                  "env": "env2_20261119", "frame": "map", "units": "m", "from": L["from"], "to": L["to"],
                  "length_m": ck["length_m"], "min_body_clearance_cm": ck["min_body_clearance_cm"], "max_nav2_cost": ck["max_center_cost"],
                  "bad_cm": ck["bad_cm"], "tight_cm": ck["tight_cm"], "strafe_longest_cm": L["strafe_longest_cm"],
                  "instruction": rec["instruction"], "photo": rec["photo"], "photo_sha256": rec["photo_sha256"],
                  "pixels_uv": np.round(self.geo.to_px(pts), 1).tolist(),
                  "assumptions": {"footprint_m": rob["footprint_m"], "footprint_padding_m": rob["footprint_padding_m"],
                                  "gripper_tip_from_base_link_m": rob.get("gripper_tip_from_base_link_estimate_m"),
                                  "camera_pose": N.CFG["photo_set"]["pose"], "note": "그리퍼 도달·파지 가능 여부는 실측 전 (UNVERIFIED)"},
                  "status": "PLANNED (모델 내부 검사 통과, 실제 주행 전)", "created": rec["created"], "tool": "route_console_20261122"}
            f = out / f"{i + 1}_{L['from']}_{L['to']}.json"
            f.write_text(json.dumps(js, indent=1, ensure_ascii=False), encoding="utf-8")
            files.append(f)
            r = cc.check_route_file(self.planner.E, self.planner.R, f)            # 팀원 검사 코드로 파일을 다시 읽어 확인
            checks.append({"file": f.name, "bad_cm": r["bad_cm"], "tight_cm": r["tight_cm"], "min_body_clearance_cm": r["min_body_clearance_cm"]})
            for j, p in enumerate(pts):
                lab = "출발" if j == 0 else ("도착" if j == len(pts) - 1 else f"W{j}")
                rot_i = int(rt["rotate_at"][1:]) if rt["rotate_at"] and rt["rotate_at"].startswith("W") else None
                yaw = rt["goal_yaw_deg"] if (lab == "도착" or (rot_i is not None and j >= rot_i)) else rt["start_yaw_deg"]
                u, v = self.geo.to_px([p])[0]
                rows.append([i + 1, f"{L['from']}→{L['to']}", lab, p[0], p[1], yaw, "예" if lab == f"W{rot_i}" or (lab == "도착" and rt["rotate_at"] == "도착점") else "",
                             round(float(u), 1), round(float(v), 1)])
        with open(out / "waypoints.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["경로", "구간", "점", "x_m", "y_m", "차체 방향(°)", "여기서 회전", "사진 u(px)", "사진 v(px)"])
            w.writerows(rows)
        files.append(out / "waypoints.csv")
        self.draw(img, rec, out / "overlay.jpg")
        files.append(out / "overlay.jpg")
        rec["export"] = {"dir": str(out), "files": [f.name for f in files], "teammate_check": checks}
        (out / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        with zipfile.ZipFile(out / "route_package.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.iterdir()):
                if f.suffix != ".zip":
                    z.write(f, f.name)
        return rec["export"]

    def draw(self, img, rec, path):
        """저장용 그림 (화면과 같은 색). 사진 위: 구역 선, A·B, 정차 자세, 경로와 경유점."""
        ov = rec["overlay"]
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        f, fb = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONTB, 26)
        P = lambda a: [tuple(map(float, p)) for p in a]   # noqa: E731
        col = {"A": (229, 72, 77), "B": (47, 111, 224), 1: (18, 184, 134), 2: (142, 78, 198), "start": (255, 255, 255)}
        d.line(P(ov["drive_px"] + ov["drive_px"][:1]), fill=(242, 194, 48, 140), width=2)
        for key, lab in (("A", "옮길 것"), ("B", "놓을 곳")):
            t = ov["targets"].get(key)
            if t and t.get("box"):
                x0, y0, x1, y1 = t["box"]
                d.rounded_rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], radius=6, outline=col[key] + (255,), width=5)
                d.text((x0 - 6, y0 - 40), f"{key} {lab}", font=fb, fill=col[key] + (255,), stroke_width=3, stroke_fill=(20, 24, 30, 255))
        for L in ov["legs"]:
            c = col[1 if L["index"] == 1 else 2]
            pts = P(L["points_px"])
            d.line(pts, fill=(20, 24, 30, 200), width=11, joint="curve")
            d.line(pts, fill=c + (255,), width=6, joint="curve")
            for j, (u, v) in enumerate(pts[1:-1], 1):
                d.ellipse([u - 10, v - 10, u + 10, v + 10], fill=c + (255,), outline=(20, 24, 30, 255), width=3)
                tag = f"{L['index']}-W{j}" + (" 회전" if L["rotate_at"] == f"W{j}" else "")
                d.text((u + 14, v - 14), tag, font=f, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(20, 24, 30, 255))
        for key, lab in (("start", "출발"), ("A", "파지 정차"), ("B", "도착 정차")):
            p = ov["poses"].get(key)
            if p:
                c = col["start"] if key == "start" else col[1 if key == "A" else 2]
                d.polygon(P(p["body_px"]), outline=c + (255,), width=4)
                d.line(P(p["tip_px"]), fill=(242, 194, 48, 255), width=6)
                u = sum(q[0] for q in p["body_px"]) / 4
                v = min(q[1] for q in p["body_px"])
                d.text((u - 40, v - 36), lab, font=f, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(20, 24, 30, 255))
        lines = rec["summary"]
        h = 20 + 32 * len(lines)
        canvas = Image.new("RGB", (im.size[0], im.size[1] + h), (24, 30, 38))
        canvas.paste(Image.alpha_composite(im, lay).convert("RGB"), (0, 0))
        dt = ImageDraw.Draw(canvas)
        for i, t in enumerate(lines):
            dt.text((18, im.size[1] + 10 + 32 * i), t, font=f, fill=(236, 240, 244))
        canvas.save(path, quality=92)

    # ---------------------------------------------------------------- 한 번 실행
    def run(self, photo, instruction, live, emit, out_root):
        """emit(단계 key, 상태, 요약, 자세한 값). 상태: running / done / hold / failed / skipped"""
        t0 = time.time()
        photo = Path(photo)
        img = cv2.imread(str(photo), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        rec = {"created": datetime.now().isoformat(timespec="seconds"), "instruction": instruction, "photo": str(photo),
               "photo_sha256": sha256_file(photo), "mode": "실시간" if live else "사진", "config": str(HERE / "config" / "console.yaml"),
               "frames": {"pixel": "편 사진 픽셀 (K_photo, 왜곡 없음)", "map": "env2_20261119 m, 바닥 z = 0", "yaw": "+x에서 반시계, 차체 앞 = 그리퍼"},
               "stages": {}, "result": {"status": "RUNNING"}}
        log = []
        stop = lambda key, why: (emit(key, "hold", why, {}), rec["result"].update(status="HOLD", stage=key, why=why))  # noqa: E731

        # 1
        emit("interpret", "running", "지시를 항목으로 바꾸는 중", {})
        it = self.interpret(instruction, log)
        rec["stages"]["interpret"] = it
        if not it["slots"]:
            stop("interpret", f"해석 응답을 쓸 수 없음: {it['error']}")
            return self._finish(rec, img, out_root, t0, log, emit, None, None, None)
        plan, why = task_plan(it["slots"])
        rec["plan"] = plan
        sl = it["slots"]
        nm = lambda p: PLACES[p]["name"] if p in PLACES else p   # noqa: E731
        obj = {"red cube": "빨간 큐브", "any object": "물체 (빨간 큐브)", "none": "없음 (로봇만 이동)", "other": "목록 밖 물체"}[sl["object"]]
        dest = ("빈 팔레트" if sl["destination_kind"] == "empty_pallet" else "목록 밖" if sl["destination_kind"] == "other"
                else "말하지 않음" if sl["destination_place"] == "none" else "목록 밖 장소" if sl["destination_place"] == "other" else nm(sl["destination_place"]))
        start_txt = ("지금 위치" if not plan or not plan["start_place"] else nm(plan["start_place"])) if plan else ("지금 위치" if sl["start"] == "none" else nm(sl["start"]))
        near_txt = "말하지 않음" if not plan or not plan["near_place"] else f"{nm(plan['near_place'])} 옆"
        summ = {"할 일": "옮기기" if sl["task"] == "transport" else "이동만", "옮길 것": obj, "물체 위치": near_txt, "출발": start_txt, "도착": dest}
        if plan and plan["corrections"]:
            summ["해석 보정"] = " / ".join(plan["corrections"])
        line = [("출발", start_txt)] + ([("옮길 것", obj)] if sl["task"] == "transport" else []) + [("도착", dest)]
        emit("interpret", "done" if plan else "hold", ", ".join(f"{k} {v}" for k, v in line),
             {"slots": sl, "summary": summ, "raw": it["raw"], "seconds": it["seconds"], "prompt_id": it["prompt_id"],
              "corrections": plan["corrections"] if plan else []})
        if not plan:
            stop("interpret", why)
            return self._finish(rec, img, out_root, t0, log, emit, None, None, None)

        # 2
        need_scene = plan["task"] == "transport" or plan["dest_kind"] == "empty_pallet"
        perc = None
        if need_scene:
            emit("perceive", "running", "후보를 찾고 예시와 비교하는 중" if live else "저장된 인식 자료 불러오는 중", {})
            perc = self.perceive(photo, live, log)
            rec["stages"]["perceive"] = {k: v for k, v in perc.items() if k != "rows"}
            emit("perceive", "done", f"후보 {perc['n_dino']}개 중 크기 조건 {perc['n_gated']}개 비교", {"source": perc["source"], "n_dino": perc["n_dino"], "n_gated": perc["n_gated"]})
        else:
            emit("perceive", "skipped", "물체를 찾을 필요 없음 (지도의 장소로 이동)", {})

        # 3
        emit("decide", "running", "규칙으로 고르는 중", {})
        dec = self.decide(plan, perc) if perc else {"A": None, "B": {"status": "SELECTED", "why": f"지도의 장소 '{nm(plan['dest_place'])}' (팀원 정차 자리)",
                                                                     "id": None, "box": None, "place": plan["dest_place"]}, "rows": {}}
        rec["stages"]["decide"] = {k: v for k, v in dec.items() if k != "rows"}
        KO = {"SELECTED": "확정", "AMBIGUOUS": "보류 (후보 둘 이상)", "NOT_FOUND": "찾지 못함"}

        def brief(d):
            if not d:
                return None
            top = (d.get("top_before_policy") or [{}])[0]
            return {"status": d["status"], "status_ko": KO.get(d["status"], d["status"]), "why": why_ko(d["why"]), "box": d.get("box"), "place": d.get("place"),
                    "s1": d.get("s1"), "s2": d.get("s2"), "top": top, "n_survivors": d.get("n_survivors")}
        dd = {"A": brief(dec["A"]), "B": brief(dec["B"])}
        bad = [k for k in ("A", "B") if dec[k] and dec[k]["status"] != "SELECTED"]
        txt = ", ".join(f"{k} {dd[k]['status_ko']}" for k in ("A", "B") if dd[k])
        if bad:
            k = bad[0]
            # 보류 이유를 구체적으로: 정책 전 상위 후보가 어느 검사에서 왜 떨어졌는지 (예: 노란 선에서 16 cm, 한도 15 cm)
            fails = []
            for t in (dec[k].get("top_before_policy") or [])[:2]:
                r = dec["rows"].get(t["id"]) or {}
                ck = next((c for c in r.get(f"{k}_checks", []) if not c["ok"]), None)
                if ck:
                    fails.append(f"{t['id']}(점수 {t['score']:.3f}) {ck['check']}: {ck['detail']}")
            dd[k]["fails"] = fails
            emit("decide", "hold", txt, dd)
            stop("decide", f"{'옮길 물체' if k == 'A' else '놓을 곳'}를 확정하지 못함: {why_ko(dec[k]['why'])}"
                           + (". 상위 후보: " + "; ".join(fails) if fails else ""))
            return self._finish(rec, img, out_root, t0, log, emit, plan, dec, perc)
        emit("decide", "done", txt, dd)

        # 4
        emit("project", "running", "지도 좌표로 바꾸는 중", {})
        g = self.geometry(plan, dec, perc, img)
        rec["geometry"] = g
        pj = {k: v["map_xy_m"] for k, v in g["objects"].items()}
        st = g["start"]
        emit("project", "done", (", ".join(f"{k} ({v[0]:.2f}, {v[1]:.2f}) m" for k, v in pj.items()) or "물체 좌표 없음 (장소로 이동)"),
             {"objects": g["objects"], "start": st})

        # 5
        emit("plan", "running", "정차 자리와 경로를 찾는 중", {})
        if not st or st.get("status") != "OK":
            stop("plan", "출발 자세를 알 수 없음 (로봇이 움직였거나 마커가 안 보임)")
            return self._finish(rec, img, out_root, t0, log, emit, plan, dec, perc, g)
        for key, lab in (("A", "파지 정차"), ("B", "도착 정차")):
            rj = g["stops"].get(f"{key}_rejected")
            if rj:
                stop("plan", f"{lab}: 물체가 그리퍼 끝에서 {rj['tip_to_object_cm']:.0f} cm (한도 {N.CFG['stop_pose']['max_tip_to_object_m'] * 100:g} cm), 닿을 수 없음")
                return self._finish(rec, img, out_root, t0, log, emit, plan, dec, perc, g)
        bad = next((L for L in g["legs"] if L["status"] != "OK"), None)
        if bad or not g["legs"]:
            stop("plan", f"경로 {bad['from']}→{bad['to']}: {bad.get('why', bad['status'])}" if bad else "경로 없음")
            return self._finish(rec, img, out_root, t0, log, emit, plan, dec, perc, g)
        tot = sum(L["check_teammate"]["length_m"] for L in g["legs"])
        emit("plan", "done", f"경로 {len(g['legs'])}개, 총 {tot * 100:.0f} cm, 경유점 {sum(len(L['route']['waypoints_xy_m']) for L in g['legs'])}개",
             {"legs": [{"from": L["from"], "to": L["to"], "length_cm": round(L["check_teammate"]["length_m"] * 100), "waypoints": L["route"]["waypoints_xy_m"],
                        "rotate_at": L["route"]["rotate_at"], "min_clear_cm": L["check_teammate"]["min_body_clearance_cm"], "strafe_longest_cm": L["strafe_longest_cm"],
                        "start_yaw": L["route"]["start_yaw_deg"], "goal_yaw": L["route"]["goal_yaw_deg"]} for L in g["legs"]],
              "stops": {k: g["stops"].get(k) for k in ("A", "B")}})
        rec["result"] = {"status": "PATH_OK"}
        return self._finish(rec, img, out_root, t0, log, emit, plan, dec, perc, g)

    def _finish(self, rec, img, out_root, t0, log, emit, plan, dec, perc, g=None):
        if g is None:
            g = {"start": None, "objects": {}, "stops": {}, "legs": [], "names": {}}
            rec["geometry"] = g
        rec["overlay"] = self.overlay(dec or {"A": None, "B": None, "rows": {}}, perc, g)
        rec["model_calls"] = log
        rec["wall_s"] = round(time.time() - t0, 1)
        L = [f"지시: {rec['instruction']}"]
        if rec["result"]["status"] == "PATH_OK":
            for x in g["legs"]:
                ck = x["check_teammate"]
                L.append(f"경로 {x['from']} → {x['to']}: {ck['length_m'] * 100:.0f} cm, 경유점 {len(x['route']['waypoints_xy_m'])}개, "
                         f"회전 {x['route']['rotate_at'] or '없음'}, 몸체 여유 최소 {ck['min_body_clearance_cm']:.1f} cm")
            sb = g["stops"].get("B") or {}
            if sb.get("facing_error_deg") and sb["facing_error_deg"] > 10:
                L.append(f"주의: {sb['place_name']} 정차 자리에서 되는 방향({', '.join(f'{h}°' for h in sb['ok_headings'])})으로는 그리퍼가 구역을 "
                         f"{sb['facing_error_deg']:.0f}° 비껴 봄")
            L.append("결과: 경로 완료 (모델 내부 검사 통과, 그리퍼 도달은 실측 전)")
        else:
            L.append(f"결과: 보류 - {rec['result'].get('why')}")
        rec["summary"] = L
        out = Path(out_root) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(rec['photo']).stem}"
        rec["export"] = self.export(rec, img, out)
        return rec


CFG_N = N.CFG
