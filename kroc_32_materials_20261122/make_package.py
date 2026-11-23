#!/usr/bin/env python3
"""KRoC 3.2 전달 자료 만들기 (2026-11-22). 기존 실행 기록을 읽기만 하고, 이 폴더 안에 복사·정리한다. 모델 호출 없음.
근거 실행: /home/binci/vlm_ws/nl_to_route_20261122/results/run_20261122_all15 (인식, GPU 1회)
           /home/binci/vlm_ws/nl_to_route_20261122/results/run_20261122_all15_final (같은 인식 결과로 좌표·경로 재계산)
  /home/binci/miniconda3/envs/stnav_gui/bin/python -I -B make_package.py
"""
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
RUN = Path("/home/binci/vlm_ws/nl_to_route_20261122/results/run_20261122_all15")
FINAL = Path("/home/binci/vlm_ws/nl_to_route_20261122/results/run_20261122_all15_final")
PHOTOS = Path("/home/binci/vlm_ws/env2_photos_20261120/photos")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

# AI(Claude) 판독, 실행 뒤 사진을 보고 판단 (nl_to_route_20261122/results/run_20261122_all15_final/정답_확인표.md). 사람 확인 전.
VERDICT = {
    "215746": {"A": ("실패 (틀리게 확정)", "작업대 선반의 빨간 부품을 큐브로 확정 (예시 점수 0.898 ≥ 기준 0.88). 진짜 큐브 (1092, 278)는 0.836으로 기준 미만. "
                                    "사진 전체를 찾고 작업 공간 제한이 없음. 경로 단계에서 '그리퍼 끝에서 71 cm, 닿을 수 없음'으로 멈춤"),
               "B": ("성공", "")},
    "215805": {"A": ("실패 (보류)", "1위가 작업대 선반의 빨간 부품, 진짜 큐브 (1088, 302)와 점수 차 0.004 < 0.02라 보류"), "B": ("성공", "")},
    "215847": {"A": ("성공", ""), "B": ("성공 (맞는 보류)", "사진에 빈 작은 팔레트가 없음 (AI 판독). 1위 후보는 꼬깔 (점수 0.818)")},
    "220134": {"A": ("성공", ""), "B": ("실패 (놓침)", "진짜 팔레트 (472, 761)가 컨베이어 줄무늬 옆이라 DINO 후보 박스에 없음. 1위 후보는 꼬깔, 보류")},
    "220222": {"A": ("성공", ""), "B": ("실패 (놓침)", "진짜 팔레트 (1495, 890)가 충전소 기둥 옆이라 DINO 후보 박스에 없음. 1위 후보는 꼬깔, 보류")},
    "220344": {"A": ("성공", ""), "B": ("실패 (놓침)", "진짜 팔레트 (535, 470)가 컨베이어 줄무늬 옆이라 DINO 후보 박스에 없음. 1위 후보는 꼬깔, 보류")},
}
# 체크리스트 표(results_per_photo.csv)에 쓰는 한 줄 원인. 자세한 내용은 results_per_photo_detail.csv
SHORT_CAUSE = {
    ("215746", "A"): "작업대 선반의 빨간 부품을 큐브로 확정 (진짜 큐브 점수 0.836 < 기준 0.88)",
    ("215805", "A"): "선반의 빨간 부품과 진짜 큐브의 점수 차 0.004 < 0.02라 보류",
    ("215847", "B"): "사진에 빈 작은 팔레트가 없음 (맞는 보류)",
    ("220134", "B"): "팔레트가 컨베이어 줄무늬 옆이라 후보 박스에 없음 (DINO 놓침)",
    ("220222", "B"): "팔레트가 충전소 기둥 옆이라 후보 박스에 없음 (DINO 놓침)",
    ("220344", "B"): "팔레트가 컨베이어 줄무늬 옆이라 후보 박스에 없음 (DINO 놓침)",
}
TARGET = {"A": "빨간 큐브 (옮길 물체)", "B": "빈 작은 회색 팔레트 (놓을 곳)"}
STATUS_KO = {"SELECTED": "확정", "AMBIGUOUS": "보류", "NOT_FOUND": "없음"}


def rnd(b):
    return [int(round(v)) for v in b] if b else None


def main():
    for d in ("raw", "figures/path_overlay", "prompts"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(FINAL.glob("*/nl_to_path.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        photo = f.parent.name
        short = photo[9:15]
        perc = rec["perception"]
        # raw: 인식 실행 기록 원본(복사)과 최종 좌표·경로 기록
        prun = Path(perc["run_json"])
        rd = OUT / "raw" / photo
        rd.mkdir(exist_ok=True)
        shutil.copy(prun, rd / "perception_run.json")
        shutil.copy(prun.parent / "model_calls.json", rd / "model_calls.json")
        shutil.copy(prun.parent / "qwen" / "slots.txt", rd / "qwen_slots_response.txt")
        shutil.copy(f, rd / "nl_to_path_final.json")
        shutil.copy(f.parent / "path_overlay.jpg", OUT / "figures" / "path_overlay" / f"{photo}_path_overlay.jpg")
        for side in ("A", "B"):
            x = perc[side]
            ex = x.get("exemplar") or {}
            box = rnd(x.get("box_px")) if x["status"] == "SELECTED" else None
            top = rnd(ex.get("top_box"))
            obj = (rec.get("objects") or {}).get(side) or {}
            v = VERDICT.get(short, {}).get(side, ("성공", ""))
            rows.append({
                "사진 파일명": f"{photo}.png",
                "찾을 대상": TARGET[side],
                "파이프라인 결과": STATUS_KO.get(x["status"], x["status"]),
                "성공/실패 (AI 판독)": v[0],
                "출력 좌표 (픽셀 x1,y1,x2,y2)": "" if box is None else ",".join(map(str, box)),
                "출력 중심 (픽셀)": "" if box is None else f"{(box[0] + box[2]) / 2:.0f},{(box[1] + box[3]) / 2:.0f}",
                "1위 후보 (보류일 때, 픽셀)": "" if box is not None or top is None else ",".join(map(str, top)),
                "예시 점수 1위": ex.get("s1"),
                "예시 점수 2위 (다른 물체)": ex.get("s2"),
                "map 좌표 x,y (m, env2)": "" if not obj.get("map_xy_m") else ",".join(f"{c:.3f}" for c in obj["map_xy_m"]),
                "실패 원인": v[1],
                "경로 그림까지": {"PATH_OK": "완료"}.get(rec["result"]["status"], f"멈춤 ({rec['result'].get('stopped_at')})"),
            })
    with open(OUT / "results_per_photo_detail.csv", "w", encoding="utf-8-sig", newline="") as fh:   # utf-8-sig: 엑셀에서 한글이 깨지지 않게
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # 체크리스트 칸 구성 그대로 (사진 파일명, 찾을 대상, 성공/실패, 출력 좌표 (픽셀), 실패 원인)
    cols = ["사진 파일명", "찾을 대상", "성공/실패", "출력 좌표 (픽셀)", "실패 원인"]
    with open(OUT / "results_per_photo.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            out = r["출력 좌표 (픽셀 x1,y1,x2,y2)"] or "없음 (보류)"
            key = (r["사진 파일명"][9:15], "A" if r["찾을 대상"].startswith("빨간") else "B")
            w.writerow([r["사진 파일명"], r["찾을 대상"].split(" (")[0], r["성공/실패 (AI 판독)"], out, SHORT_CAUSE.get(key, "-")])
    # 프롬프트 원문 (15장 모두 같은 지시, 같은 프롬프트)
    first = sorted(RUN.glob("perception_session/runs/*"))[0]
    shutil.copy(first / "qwen" / "slots_prompt.txt", OUT / "prompts" / "qwen_slots_prompt.txt")
    ph = json.loads(next((RUN / "perception_session" / "photos").glob("*.json")).read_text(encoding="utf-8"))
    dino = {c["stage"]: c.get("prompts") for c in ph["calls"] if c["stage"].startswith("dino")}
    (OUT / "prompts" / "dino_phrases.json").write_text(json.dumps(dino, indent=1, ensure_ascii=False), encoding="utf-8")
    shutil.copy(RUN / "perception_session" / "exemplar_bank.json", OUT / "prompts" / "siglip2_exemplar_bank.json")
    shutil.copy(FINAL / "A_B_확대.jpg", OUT / "figures" / "A_B_확대.jpg")
    # 논문용 인식 그림 (원본 해상도, 굵은 박스, 큰 글자)
    font = ImageFont.truetype(FONT, 46)
    for short, extra in (("215900", None), ("215746", (1092, 278))):
        rec = json.loads(next(FINAL.glob(f"*{short}*/nl_to_path.json")).read_text(encoding="utf-8"))
        im = Image.open(PHOTOS / f"20261120_{short}_view.png").convert("RGB")
        dr = ImageDraw.Draw(im)
        for side, col, label in (("A", (230, 30, 30), "A: cube"), ("B", (30, 90, 230), "B: empty pallet")):
            b = rec["perception"][side].get("box_px")
            if rec["perception"][side]["status"] != "SELECTED" or not b:
                continue
            pad = 10
            r = [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]
            dr.rectangle(r, outline=col, width=8)
            tx, ty = r[0], max(r[1] - 62, 0)
            tw = dr.textlength(label, font=font)
            dr.rectangle([tx, ty, tx + tw + 16, ty + 58], fill=col)
            dr.text((tx + 8, ty + 2), label, font=font, fill=(255, 255, 255))
        if extra:
            x, y = extra
            dr.ellipse([x - 30, y - 30, x + 30, y + 30], outline=(20, 170, 60), width=8)
            dr.text((x + 38, y - 30), "true cube", font=font, fill=(20, 170, 60), stroke_width=3, stroke_fill=(255, 255, 255))
        im.save(OUT / "figures" / f"fig_recognition_{short}.png")
    print(f"rows {len(rows)} -> {OUT / 'results_per_photo.csv'}")


if __name__ == "__main__":
    main()
