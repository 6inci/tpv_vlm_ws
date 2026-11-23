#!/usr/bin/env python3
"""논문 검토용 실험 자료 묶음 (2026-11-23). 원본은 읽기만 하고 복사본만 만든다.
  python3 -B make_bundle.py
결과: package/paper_evidence_20261123/ 와 paper_evidence_20261123.zip (이 폴더). 이미 있으면 멈춘다.
새로 만드는 것은 그림 3종(기록 JSON에서 다시 그린 DINO 전체 후보, 15장 배치 모음)과 요약 CSV 2개뿐이다. 모델은 다시 돌리지 않는다.
"""
import csv
import hashlib
import json
import shutil
import unicodedata
import zipfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NAME = "paper_evidence_20261123"
ROOT = HERE / "package" / NAME
ZIP = HERE / f"{NAME}.zip"
WS = Path("/home/binci/vlm_ws")
DL = Path("/home/binci/Downloads")

PHOTOS = WS / "env2_photos_20261120/photos"
OLD = WS / "nl_to_route_20261122/results/run_20261122_all15"            # 논문 3.1 수치의 인식 실행 (GPU 1회)
FINAL = WS / "nl_to_route_20261122/results/run_20261122_all15_final"    # 같은 인식으로 정차·경로 재계산
KROC = WS / "kroc_32_materials_20261122"
CP_TEST = WS / "context_policy_20261122/results/test_20261122_183736"
CP_DEV = WS / "context_policy_20261122/results/dev_20261122_183649"
RC = WS / "route_console_20261122"
NAVH = WS / "semantic_task_navigation/experiments/navigation_handoff/handoff_package/navigation_handoff_photo_1_20261120"
VENDOR = WS / "nl_to_route_20261122/vendor/teammate_extra_20261120"
RUNS = ["20261122_192459_20261120_215815_view", "20261122_192502_20261120_215900_view", "20261122_192504_20261120_220042_view",
        "20261122_192506_20261120_215815_view", "20261122_192509_20261120_215815_view", "20261122_192510_20261120_215815_view",
        "20261122_192512_20261120_215815_view", "20261122_195711_20261120_220344_view", "20261122_200350_20261120_220632_view",
        "20261122_193058_live_simulated_220632"]
IDS = sorted(p.stem for p in PHOTOS.glob("20261120_*_view.png"))


def nfc_child(parent, name):
    """macOS zip에서 풀린 한글 이름(NFD)도 찾음"""
    return next(p for p in parent.iterdir() if unicodedata.normalize("NFC", p.name) == name)


TEAM = nfc_child(nfc_child(DL, "인계자료"), "인계자료")
ROWS = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(src, rel, note=""):
    src = Path(src)
    if src.is_dir():
        for f in sorted(src.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                put(f, f"{rel}/{f.relative_to(src)}", note)
        return
    dst = ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    assert sha(src) == sha(dst), src
    ROWS.append((rel, str(src), sha(dst), note or "복사"))


def put_jpg(src, rel, q=90):
    """큰 PNG 그림을 JPEG로 옮김 (내용 같음, 화질 q)"""
    dst = ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = cv2.imread(str(src))
    assert im is not None, src
    cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, q])
    ROWS.append((rel, str(src), sha(dst), f"PNG→JPEG q{q}"))


def made(rel, sources, note):
    ROWS.append((rel, "; ".join(str(s) for s in sources), sha(ROOT / rel), note))


def draw_dino(pid):
    """기록된 DINO 출력(병합 후 전체)을 원본 사진 위에 다시 그림. 모델 실행 없음."""
    rec = json.loads((OLD / "perception_session/photos" / f"{pid}.json").read_text(encoding="utf-8"))
    im = cv2.imread(str(PHOTOS / f"{pid}.png"))
    for d in rec["dino"]:
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox_xyxy"])
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.putText(im, d["id"], (x1, max(y1 - 2, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    t = f"{pid} | Grounding DINO-tiny raw output: {len(rec['dino'])} boxes (full image + facility ROI, IoU 0.5 merge) | redrawn from recorded JSON"
    cv2.rectangle(im, (0, 0), (1920, 26), (0, 0, 0), -1)
    cv2.putText(im, t, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    rel = f"1_객체인식/B_DINO_전체후보/{pid}_dino_all.jpg"
    (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOT / rel), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    made(rel, [OLD / "perception_session/photos" / f"{pid}.json", PHOTOS / f"{pid}.png"], "기록 JSON에서 다시 그림")
    put(OLD / "perception_session/photos" / f"{pid}.json", f"1_객체인식/B_DINO_전체후보/{pid}_dino.json")


def montage():
    tiles = []
    for pid in IDS:
        im = cv2.resize(cv2.imread(str(PHOTOS / f"{pid}.png")), (640, 360))
        cv2.rectangle(im, (0, 0), (260, 28), (0, 0, 0), -1)
        cv2.putText(im, pid.replace("_view", ""), (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(im)
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, 15, 3)]
    rel = "4_환경변화/15장_배치_한눈에.jpg"
    (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOT / rel), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    made(rel, [PHOTOS], "원본 15장 축소 모음")


def command_table():
    rel = "2_자연어명령/명령처리_요약.csv"
    with open(ROOT / rel, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["실행", "사진", "자연어 명령", "Qwen 원문 응답", "계획: 물체 옆 장소(구역)", "계획: 도착 장소",
                    "Qwen destination_place 사용 여부", "A 결과", "A 이유", "A 규칙 전 상위 후보 (점수, 탈락 규칙)",
                    "B 결과", "B 이유", "B 규칙 전 상위 후보 (점수, 탈락 규칙)", "최종", "최종 이유"])
        for r in RUNS:
            d = json.loads((RC / "runs" / r / "record.json").read_text(encoding="utf-8"))
            it, plan, dec = d["stages"]["interpret"], d.get("plan") or {}, d["stages"].get("decide") or {}
            dp = (it.get("slots") or {}).get("destination_place", "")
            used = "-" if dp in ("", "none") else ("사용" if plan.get("dest_place") == dp else "버려짐 (도착 종류가 빈 팔레트면 장소를 안 씀)")
            top = lambda x: "; ".join(f"{c['id']} {c['score']:.3f} {c['failed_at'] or '통과'}" for c in (x or {}).get("top_before_policy", []))
            A, B = dec.get("A") or {}, dec.get("B") or {}
            w.writerow([r, Path(d["photo"]).name, d["instruction"], it.get("raw"),
                        f"{plan.get('near_place')} ({plan.get('near_zone')})" if plan.get("near_zone") else "-", plan.get("dest_place") or "-",
                        used, A.get("status", "-"), A.get("why", "-"), top(A), B.get("status", "-"), B.get("why", "-"), top(B),
                        d["result"]["status"], (d["result"].get("why") or "").split(";")[0]])
    made(rel, [RC / "runs" / r / "record.json" for r in RUNS], "record.json에서 뽑은 표")


def path_table():
    rel = "3_주행목표_경로/b_15장_경로_논문시스템/경로_요약.csv"
    ix = json.loads((FINAL / "index.json").read_text(encoding="utf-8"))
    with open(ROOT / rel, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["사진", "결과", "멈춘 단계", "A 인식", "B 인식", "A 지도 (m)", "B 지도 (m)",
                    "파지 정차 (x, y m, yaw°)", "파지 정차 몸체 여유 (cm)", "파지 그리퍼 끝-물체 (cm)",
                    "배치 정차 (x, y m, yaw°)", "배치 정차 몸체 여유 (cm)", "배치 그리퍼 끝-물체 (cm)"])
        for r in ix:
            sA, sB = r.get("stop_A") or {}, r.get("stop_B") or {}
            st = lambda s: f"{s['xy_m'][0]:.3f}, {s['xy_m'][1]:.3f}, {s['yaw_deg']}" if s else "-"
            w.writerow([r["photo"], r["result"]["status"], r["result"].get("stopped_at") or "-", r["A"], r["B"],
                        r["A_map"] or "-", r["B_map"] or "-", st(sA), sA.get("body_clearance_cm", "-"), sA.get("reach_gap_cm", "-"),
                        st(sB), sB.get("body_clearance_cm", "-"), sB.get("reach_gap_cm", "-")])
    made(rel, [FINAL / "index.json"], "index.json에서 뽑은 표")


def code_excerpt():
    src = RC / "engine.py"
    lines = src.read_text(encoding="utf-8").splitlines()
    rel = "2_자연어명령/관계판단_코드발췌.txt"
    body = ["# route_console_20261122/engine.py 발췌 (공간 관계를 누가 판단하는지 보여 주는 부분만, 원문 그대로)", "",
            "## task_plan: Qwen 항목 -> 실행 계획 (116-151행)"] + lines[115:151] + ["", "## Engine.decide: '장소 옆' 판단 (246-266행)"] + lines[245:266]
    (ROOT / rel).write_text("\n".join(body) + "\n", encoding="utf-8")
    made(rel, [src], "원본 코드 일부 발췌")


def main():
    if ROOT.exists() or ZIP.exists():
        raise SystemExit(f"이미 있음: {ROOT} 또는 {ZIP}")
    ROOT.mkdir(parents=True)

    # 0 원본 사진 15장 (모델 입력)
    for pid in IDS:
        put(PHOTOS / f"{pid}.png", f"0_원본사진/{pid}.png")

    # 1 객체 인식 (논문 3.1 수치 = 기존 시스템 11-22 15:47 실행)
    for pid in IDS:
        draw_dino(pid)
    for i, pid in enumerate(IDS, 1):
        rd = OLD / "perception_session/runs" / f"run_{i:03d}_{pid}"
        put_jpg(rd / "overlay_candidates.png", f"1_객체인식/C_상위후보_점수/{pid}_candidates.jpg")
        put_jpg(rd / "overlay.png", f"1_객체인식/D_최종선택/{pid}_selected.jpg")
        put(KROC / "raw" / pid, f"1_객체인식/기록/{pid}")
    for f in ("results_per_photo.csv", "results_per_photo_detail.csv"):
        put(KROC / f, f"1_객체인식/{f}")
    put(FINAL / "정답_확인표.md", "1_객체인식/정답_확인표_AI판독.md")
    for f in ("fig_recognition_215900.png", "fig_recognition_215746.png"):
        put(KROC / "figures" / f, f"1_객체인식/논문그림후보_{f}")
    put(KROC / "prompts", "1_객체인식/프롬프트")
    for f in sorted(CP_TEST.glob("*_policy.jpg")) + [CP_TEST / "decisions.json", CP_TEST / "config_snapshot.yaml"]:
        put(f, f"1_객체인식/E_후속시스템_논문미반영/{f.name}")

    # 2 자연어 명령 처리 (콘솔 실행 10건, 실제 Qwen 해석)
    for r in RUNS:
        for f in ("record.json", "overlay.jpg"):
            put(RC / "runs" / r / f, f"2_자연어명령/실행/{r}/{f}")
    command_table()
    code_excerpt()
    put(RC / "config/console.yaml", "2_자연어명령/console.yaml")

    # 3 주행 목표와 경로
    put(NAVH, "3_주행목표_경로/a_사진1_경로_팀원실주행분")
    for pid in IDS:
        put(FINAL / pid / "path_overlay.jpg", f"3_주행목표_경로/b_15장_경로_논문시스템/{pid}_path_overlay.jpg")
        put(FINAL / pid / "nl_to_path.json", f"3_주행목표_경로/b_15장_경로_논문시스템/{pid}_nl_to_path.json")
    path_table()
    put(WS / "nl_to_route_20261122/POLICIES.md", "3_주행목표_경로/b_15장_경로_논문시스템/POLICIES.md")
    for f in sorted((CP_TEST / "paths").glob("*/path_overlay.jpg")):
        put(f, f"3_주행목표_경로/c_15장_경로_후속시스템/{f.parent.name}_path_overlay.jpg")
    put(VENDOR / "1_required/map", "3_주행목표_경로/d_env2_지도_팀원")
    put(VENDOR / "2_optional/zones.json", "3_주행목표_경로/d_env2_지도_팀원/zones.json")
    put(VENDOR / "2_optional/robot_params.json", "3_주행목표_경로/d_env2_지도_팀원/robot_params.json")

    # 4 환경 변화
    montage()
    put(KROC / "figures/A_B_확대.jpg", "4_환경변화/A_B_확대_논문시스템.jpg")
    for f in sorted(CP_DEV.glob("*_policy.jpg")) + [CP_DEV / "decisions.json"]:
        put(f, f"4_환경변화/개발사진8장_기준값설정용/{f.name}")

    # 5 실제 로봇 주행 (팀원 인계자료, 바꾸지 않음)
    for d in ("tables", "figures", "ch3_v2"):
        put(TEAM / d, f"5_로봇주행/팀원_인계자료/{d}")
    for f in ("README.md", "verification_report.md"):
        put(TEAM / f, f"5_로봇주행/팀원_인계자료/{f}")
    for f in ("route_runs_recomputed.csv", "bag_route_metrics.csv", "bag_nav2_goals.csv", "run_index.csv", "safety_stop_events.csv"):
        put(TEAM / "data" / f, f"5_로봇주행/팀원_인계자료/data/{f}")
    put(nfc_child(DL, "팀원경로_주행회신_20261120.html"), "5_로봇주행/팀원경로_주행회신_20261120.html")

    # 6 실험 환경
    put(WS / "new_experiment_environment/final_experiment_picture.png", "6_실험환경/고정카메라_전체화면_20261114.png")
    put(WS / "env2_photos_20261120/camera_pose", "6_실험환경/카메라자세")

    put(HERE / "README.md", "00_README.md")
    with open(ROOT / "MANIFEST.txt", "w", encoding="utf-8") as f:
        f.write("묶음 안 경로\t원본 경로\tsha256(묶음 파일)\t처리\n")
        for r in ROWS:
            f.write("\t".join(r) + "\n")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(ROOT.parent))
    print(len(ROWS), "files ->", ZIP, round(ZIP.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
