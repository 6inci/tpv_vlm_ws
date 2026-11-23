#!/usr/bin/env python3
"""경로 계획 콘솔 서버 (표준 라이브러리 http.server + 브라우저 화면). 새 패키지 설치 없음.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -B server.py            # http://127.0.0.1:8765
GPU를 쓰므로 Bash 샌드박스 밖(일반 터미널)에서 실행한다. 같은 GPU를 쓰는 다른 모델 프로그램이 돌고 있으면 실행하지 않는다.
"""
import json
import mimetypes
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine as EN  # noqa: E402

CFG = EN.CFG
WEB = HERE / "web"
RUNS = HERE / CFG["output_dir"]
LIVE_DIR = RUNS / "live_frames"
PHOTOS = Path(CFG["sources"]["photos_dir"])


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================================ 상태
class State:
    def __init__(self):
        self.engine = None
        self.boot = {"status": "준비 중", "detail": "지도와 경로계획 준비"}
        self.jobs, self.lock, self.busy = {}, threading.Lock(), False
        self.thumbs = {}
        self.map_png = None
        self.live = {}


S = State()


def boot():
    try:
        t = time.time()
        S.engine = EN.Engine(progress=lambda m: log(f"모델: {m}"))
        S.boot = {"status": "준비됨", "detail": f"지도·경로계획 {time.time() - t:.1f} s"}
        log(S.boot["detail"])
        threading.Thread(target=S.engine.warm_up, daemon=True).start()
    except Exception as e:                       # noqa: BLE001  (화면에 그대로 보여 줌)
        S.boot = {"status": "오류", "detail": f"{type(e).__name__}: {e}"}
        traceback.print_exc()


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=3).stdout.strip().split(",")
        return {"used_mib": int(out[0]), "total_mib": int(out[1])}
    except Exception:                            # noqa: BLE001
        return None


def photo_list():
    skip = set(CFG["sources"].get("exclude_photos") or [])
    return sorted(p.name for p in PHOTOS.glob("*.png") if p.name not in skip)


def thumb(name, w):
    key = (name, w)
    if key not in S.thumbs:
        p = PHOTOS / name if (PHOTOS / name).exists() else LIVE_DIR / name
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        h = int(img.shape[0] * w / img.shape[1])
        S.thumbs[key] = cv2.imencode(".jpg", cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
    return S.thumbs[key]


def map_image():
    """env2 지도 (1 cm 칸)를 3배로 키운 그림. 주행 가능 = 밝게, 막힘 = 짙게, 모름 = 중간"""
    if S.map_png is None:
        E = S.engine.planner.E
        sc = 3
        g = cv2.imread(str(EN.N.TB / "1_required" / "map" / "env2_20261119.pgm"), cv2.IMREAD_GRAYSCALE)   # 팀원 지도 (trinary: 0 막힘, 205 모름, 254 빈 곳)
        rgb = np.zeros(E.free.shape + (3,), np.uint8)
        rgb[:] = (214, 220, 226)
        rgb[g < 100] = (203, 210, 218)              # 주행 영역 밖 (이 지도에는 '모름' 칸이 없음)
        rgb[E.free] = (247, 249, 251)
        S.map_png = cv2.imencode(".png", cv2.resize(rgb[:, :, ::-1], None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST))[1].tobytes()
    return S.map_png


# ================================================================================ 실시간 카메라
class Live:
    def __init__(self, dev):
        self.dev, self.frame, self.err, self.stop = dev, None, None, False
        self.cap = cv2.VideoCapture(int(dev), cv2.CAP_V4L2)
        w, h = CFG["sources"]["live"]["capture_size"]
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if not self.cap.isOpened():
            self.err = f"/dev/video{dev}을 열 수 없음"
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while not self.stop and self.cap.isOpened():
            ok, f = self.cap.read()
            if ok:
                self.frame = f
            else:
                self.err = "프레임을 읽지 못함"
                time.sleep(0.2)

    def close(self):
        self.stop = True
        time.sleep(0.1)
        self.cap.release()


def live_get(dev):
    for d, L in list(S.live.items()):
        if d != dev:
            L.close()
            S.live.pop(d)
    if dev not in S.live:
        S.live[dev] = Live(dev)
    return S.live[dev]


def live_capture(dev):
    """지금 프레임 -> 왜곡 보정(공유 사진과 같은 alpha 0 newK) -> 저장 -> 카메라 자세가 공유 사진과 같은지 확인"""
    L = live_get(dev)
    t = time.time()
    while L.frame is None and time.time() - t < 3:
        time.sleep(0.05)
    if L.frame is None:
        return {"ok": False, "why": L.err or "카메라 프레임 없음"}
    raw = L.frame.copy()
    K, dist = EN.N.p2m.intrinsics()
    Kp = S.engine.geo.K
    W, H = CFG["sources"]["live"]["capture_size"]
    if raw.shape[1] != W or raw.shape[0] != H:
        return {"ok": False, "why": f"카메라 해상도 {raw.shape[1]}×{raw.shape[0]} (필요: {W}×{H}). 보정된 외부 카메라가 아님"}
    und = cv2.undistort(raw, K, dist, None, Kp)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    name = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(LIVE_DIR / name), und)
    from semantic_task_nav.navigation.map_builder import check_alignment
    a = check_alignment({"reference_image": CFG["sources"]["live"]["reference_photo"], "alignment_check": CFG["sources"]["live"]["alignment"]},
                        und, lambda x: x)
    ok = a.get("status") == "ALIGNED"
    return {"ok": ok, "name": name, "alignment": a,
            "why": None if ok else f"카메라 자세가 공유 사진과 다름 (이동 {a.get('shift_px')} px, 응답 {a.get('response')}). 이 장면으로는 지도 좌표를 믿을 수 없음"}


# ================================================================================ 작업
def start_job(source, name, instruction):
    with S.lock:
        if S.busy:
            return None, "다른 계획이 진행 중"
        S.busy = True
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "state": "running", "source": source, "photo": name, "instruction": instruction, "created": time.time(),
           "stages": {s["key"]: {"status": "waiting", "summary": "", "detail": {}} for s in EN.STAGES}, "hold": None, "result": None}
    S.jobs[jid] = job

    def emit(key, status, summary, detail):
        if status == "hold" and job["stages"][key]["status"] in ("done", "hold"):
            job["hold"] = {"stage": key, "why": summary}
            return
        job["stages"][key] = {"status": status, "summary": summary, "detail": detail}
        if status == "hold":
            job["hold"] = {"stage": key, "why": summary}

    def work():
        try:
            photo = (LIVE_DIR / name) if source == "live" else (PHOTOS / name)
            rec = S.engine.run(photo, instruction, source == "live", emit, RUNS)
            job["result"] = {"status": rec["result"]["status"], "why": rec["result"].get("why"), "stage": rec["result"].get("stage"),
                             "overlay": rec["overlay"], "export": rec["export"], "wall_s": rec["wall_s"], "summary": rec["summary"],
                             "legs": [{"from": L["from"], "to": L["to"], "status": L["status"],
                                       "route": L.get("route"), "check": L.get("check_teammate"), "strafe_longest_cm": L.get("strafe_longest_cm")}
                                      for L in rec["geometry"]["legs"]],
                             "start": rec["geometry"].get("start"), "stops": {k: rec["geometry"]["stops"].get(k) for k in ("A", "B")},
                             "objects": rec["geometry"].get("objects"), "slots": (rec["stages"].get("interpret") or {}).get("slots"),
                             "raw": (rec["stages"].get("interpret") or {}).get("raw")}
            for k, st in job["stages"].items():
                if st["status"] in ("waiting", "running"):
                    st["status"] = "skipped" if job["hold"] else st["status"]
            job["state"] = "done"
        except Exception as e:                   # noqa: BLE001
            traceback.print_exc()
            job["state"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
            for st in job["stages"].values():
                if st["status"] == "running":
                    st["status"] = "failed"
        finally:
            S.busy = False
    threading.Thread(target=work, daemon=True).start()
    return jid, None


# ================================================================================ HTTP
class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        p = u.path
        try:
            if p in ("/", "/index.html"):
                return self.send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if p.startswith("/web/"):
                f = (WEB / p[5:]).resolve()
                if WEB in f.parents and f.exists():
                    return self.send(200, f.read_bytes(), (mimetypes.guess_type(str(f))[0] or "application/octet-stream") + "; charset=utf-8")
                return self.send(404, {"error": "없음"})
            if p == "/api/config":
                return self.send(200, {"examples": CFG["interpretation"]["examples"], "photos": photo_list(), "stages": EN.STAGES,
                                       "live_devices": CFG["sources"]["live"]["devices"], "photos_dir": str(PHOTOS)})
            if p == "/api/status":
                hs = S.engine.host_state if S.engine else {"status": "대기", "detail": ""}
                return self.send(200, {"boot": S.boot, "model": hs, "gpu": gpu(), "busy": S.busy})
            if p == "/api/photo":
                return self.send(200, thumb(q["name"], int(q.get("w", 480))), "image/jpeg")
            if p == "/api/map.png":
                return self.send(200, map_image(), "image/png")
            if p == "/api/map_meta":
                E = S.engine.planner.E
                return self.send(200, {"nx": int(E.NX), "ny": int(E.NY), "res": float(E.res), "ox": float(E.ox), "oy": float(E.oy), "scale": 3})
            if p == "/api/job":
                j = S.jobs.get(q.get("id"))
                return self.send(200 if j else 404, j or {"error": "작업 없음"})
            if p == "/api/file":
                j = S.jobs.get(q.get("job"))
                if not j or not j.get("result"):
                    return self.send(404, {"error": "파일 없음"})
                d = Path(j["result"]["export"]["dir"])
                f = (d / q["name"]).resolve()
                if f.parent != d.resolve() or not f.exists():
                    return self.send(404, {"error": "파일 없음"})
                ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
                return self.send(200, f.read_bytes(), ctype, {"Content-Disposition": f"attachment; filename*=UTF-8''{__import__('urllib.parse').parse.quote(f.name)}"})
            if p == "/api/live/stream":
                return self.stream(int(q.get("dev", 0)))
            return self.send(404, {"error": "없음"})
        except Exception as e:                   # noqa: BLE001
            traceback.print_exc()
            return self.send(500, {"error": f"{type(e).__name__}: {e}"})

    def stream(self, dev):
        L = live_get(dev)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while not L.stop:
                if L.frame is not None:
                    f = cv2.resize(L.frame, (960, int(L.frame.shape[0] * 960 / L.frame.shape[1])))
                    b = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if u.path == "/api/run":
                ins = str(body.get("instruction", "")).strip()
                if not ins:
                    return self.send(400, {"error": "작업지시를 입력하세요"})
                if S.engine is None:
                    return self.send(503, {"error": "아직 준비 중"})
                src = body.get("source", "photo")
                name = body.get("photo")
                if src == "photo" and name not in photo_list():
                    return self.send(400, {"error": "사진을 고르세요"})
                if src == "live" and not (LIVE_DIR / str(name)).exists():
                    return self.send(400, {"error": "먼저 장면을 찍으세요"})
                jid, err = start_job(src, name, ins)
                return self.send(200 if jid else 409, {"job": jid} if jid else {"error": err})
            if u.path == "/api/live/capture":
                return self.send(200, live_capture(int(body.get("dev", 0))))
            if u.path == "/api/live/stop":
                for L in S.live.values():
                    L.close()
                S.live.clear()
                return self.send(200, {"ok": True})
            return self.send(404, {"error": "없음"})
        except Exception as e:                   # noqa: BLE001
            traceback.print_exc()
            return self.send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    import signal

    def _term(*_):
        raise KeyboardInterrupt                  # 종료 신호에도 모델 작업 프로세스와 카메라를 닫음
    signal.signal(signal.SIGTERM, _term)
    RUNS.mkdir(exist_ok=True)
    threading.Thread(target=boot, daemon=True).start()
    host, port = CFG["server"]["host"], CFG["server"]["port"]
    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    log(f"경로 계획 콘솔: http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for L in S.live.values():
            L.close()
        if S.engine and S.engine.host:
            S.engine.host.close()


if __name__ == "__main__":
    main()
