#!/usr/bin/env python3
"""KRoC 논문 그림 3개 (2026-11-23). 저장된 실행 기록만 읽어 그린다. 모델·경로 계산은 다시 하지 않는다.
  python3 -I -B make_figures.py
출력: out/fig1_system, out/fig2_relation, out/fig3_goal_path (.png 600 dpi, .pdf)

좌표
- 사진 그림: 기록된 박스 그대로 (편 사진 픽셀, 1920x1080)
- 지도 그림: env2_20261119 지도 (m, 1 cm 칸, 원점 (0, 0)), 팀원 clearance_check.py 의 Env/Robot 으로 읽음
"""
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

WS = Path("/home/binci/vlm_ws")
OUT = Path(__file__).resolve().parent / "out"
TB = WS / "nl_to_route_20261122/vendor/teammate_extra_20261120"
sys.path[:0] = [str(TB / "1_required"), str(TB / "2_optional")]
import clearance_check as cc  # noqa: E402  (팀원 코드, 바꾸지 않음)

# 근거 기록
HY = WS / "hybrid_exemplar_20261121/session_20261121_073316"          # 개발 사진 8장, I2 "하역장 옆에 있는 물체를 ..."
RUN_SEL, RUN_HOLD = "run_008_U24898553", "run_002_U23678143"          # (a) 선택 예, (b) 보류 예
FINAL = WS / "nl_to_route_20261122/results/run_20261122_all15_final/20261120_220632_view/nl_to_path.json"  # 15장 실행 중 한 장

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
for f in (FONT, FONT_B):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": fm.FontProperties(fname=FONT).get_name(), "font.size": 7, "pdf.fonttype": 3,   # TTC 글꼴은 Type 42 로 넣을 수 없음
                     "axes.linewidth": 0.6, "savefig.dpi": 600})

C_MODEL, C_RULE, C_IN = "#dbe8f6", "#ececec", "#ffffff"
C_SEL, C_REJ, C_FAC, C_CAND = "#1a9850", "#d73027", "#f08c00", "#ffffff"
C_A, C_B = "#d62728", "#1f77b4"


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", OUT / name)


def photo_rgb(path, sha_expected=None):
    if sha_expected:
        import hashlib
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha_expected, path
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def rect(ax, b, color, lw, ls="-", z=3):
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=color, lw=lw, ls=ls, zorder=z))


def tag(ax, x, y, s, color, size=6, ha="left", va="bottom", bold=False):
    ax.text(x, y, s, color="white", fontsize=size, ha=ha, va=va, zorder=6,
            fontproperties=fm.FontProperties(fname=FONT_B if bold else FONT, size=size),
            bbox=dict(boxstyle="round,pad=0.18", fc=color, ec="none", alpha=0.92))


# ================================================================== Fig. 1 시스템 구성
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 100), ax.set_ylim(0, 50), ax.axis("off")

    def box(x0, y0, x1, y1, title, body="", fc=C_RULE, ls="-"):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.25,rounding_size=0.8",
                                    fc=fc, ec="#333333", lw=0.7, ls=ls, zorder=2))
        cy = (y0 + y1) / 2
        tall = (y1 - y0) > 12                                  # 높은 상자는 제목을 위에 붙임
        if body:
            ax.text((x0 + x1) / 2, y1 - 2.8 if tall else cy + 1.6, title, ha="center", va="center", zorder=3,
                    fontproperties=fm.FontProperties(fname=FONT_B, size=7))
            ax.text((x0 + x1) / 2, (y0 + y1 - 5.6) / 2 if tall else cy - 1.9, body, ha="center", va="center",
                    fontsize=5.8, zorder=3, linespacing=1.25)
        else:
            ax.text((x0 + x1) / 2, cy, title, ha="center", va="center", zorder=3, fontproperties=fm.FontProperties(fname=FONT_B, size=7))

    def arrow(p, q, text="", ls="-", tx=0, ty=1.2):
        ax.annotate("", xy=q, xytext=p, zorder=1,
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333", ls=ls, shrinkA=0, shrinkB=0, mutation_scale=7))
        if text:
            ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, ha="center", va="bottom", fontsize=5.8, color="#333333")

    # 입력
    box(1, 38, 15, 46, "자연어 작업지시", fc=C_IN)
    box(1, 13, 15, 22, "외부 고정 카메라", "1920×1080 영상", fc=C_IN)
    # 지시 해석
    box(19, 36, 41, 48, "Qwen3-VL-4B-Instruct", "지시 → 정해진 항목(JSON)\n영상은 입력하지 않음", fc=C_MODEL)
    box(45, 37, 67, 47, "작업 항목", "옮길 물체 · 기준 시설과 관계\n도착지 종류", fc=C_IN)
    # 장면 인식
    box(19, 12.5, 35, 22.5, "Grounding DINO", "물체·시설 후보 영역", fc=C_MODEL)
    box(39, 20, 59, 29, "SigLIP2 base", "시설 후보 확인 (텍스트 비교)", fc=C_MODEL)
    box(39, 5, 59, 14, "SigLIP2 so400m", "큐브·빈 팔레트 예시 이미지 비교", fc=C_MODEL)
    # 대상 결정
    box(63, 9, 80, 29, "대상 결정 (규칙)", "점수 기준 · 1·2위 점수 차\n점유 여부\n기준 시설 주변 영역\n(공간 관계)")
    box(63, 0.5, 80, 5.5, "보류 → 중단", fc=C_IN)
    # 주행 목표
    box(84, 39, 99, 48, "좌표 변환", "영상 픽셀 → 지도 (m)")
    box(84, 26, 99, 35.5, "정차 목표", "차체 크기 · 주행 영역\n그리퍼 방향")
    box(84, 14, 99, 22.5, "경로 계획", "4방향 A*, 여유 검사")
    box(84, 0.5, 99, 9.5, "실제 로봇 주행", "ROS 2 / Nav2\n경로 파일 전달", fc=C_IN, ls="--")

    arrow((15, 42), (19, 42))
    arrow((41, 42), (45, 42))
    arrow((15, 17.5), (19, 17.5))
    arrow((35, 19.5), (39, 24.5))
    arrow((35, 15.5), (39, 9.5))
    arrow((59, 24.5), (63, 22))
    arrow((59, 9.5), (63, 13))
    arrow((66, 37), (68, 29.2))
    arrow((71.5, 9), (71.5, 5.5), "불확실", tx=4.2, ty=-1.0)
    arrow((80, 22), (84, 41), "선택", tx=-1.6, ty=1.0)
    arrow((91.5, 39), (91.5, 35.5))
    arrow((91.5, 26), (91.5, 22.5))
    arrow((91.5, 14), (91.5, 9.5), ls="--")

    # 범례
    for i, (fc, ls, s) in enumerate([(C_MODEL, "-", "학습된 모델"), (C_RULE, "-", "Python 규칙·기하 계산"), (C_IN, "--", "별도 실험 (자동 연결 아님)")]):
        y = 7.2 - i * 3.0
        ax.add_patch(FancyBboxPatch((1.5, y - 0.9), 3.2, 1.8, boxstyle="round,pad=0.1", fc=fc, ec="#333333", lw=0.6, ls=ls))
        ax.text(5.6, y, s, va="center", fontsize=5.8)
    save(fig, "fig1_system")


# ================================================================== Fig. 2 공간 관계에 따른 선택·보류
def i2_counts():
    ev = json.loads((HY / "task_evaluation.json").read_text(encoding="utf-8"))
    rows = [r for r in ev["rows"] if r["instruction_key"] == "I2"]
    sel = [r for r in rows if r["A_status"] == "SELECTED"]
    hold = [r for r in rows if r["A_status"] == "NOT_FOUND"]
    why = {r["run_id"]: json.loads((HY / "runs" / r["run_id"] / "run.json").read_text(encoding="utf-8"))["source"]["why"] for r in hold}
    assert all(w.startswith("NOT_AT_REFERENCE") for w in why.values()), why
    out = {"n": len(rows), "sel": len(sel), "sel_right": sum(r["A_right"] for r in sel), "hold": len(hold),
           "hold_correct": sum(r["outcome"] == "CORRECT_ABSTAIN" for r in hold), "B_right": sum(r["B_right"] for r in sel),
           "gt": ev["ground_truth_status"]}
    print("I2 counts", out)
    return out


def fig2():
    n = i2_counts()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3))
    for ax, run, sub in ((axs[0], RUN_SEL, "a"), (axs[1], RUN_HOLD, "b")):
        d = json.loads((HY / "runs" / run / "run.json").read_text(encoding="utf-8"))
        ax.imshow(photo_rgb(d["photo"]["path"], d["photo"]["sha256"]))
        ax.set_xlim(0, 1920), ax.set_ylim(1080, 0), ax.set_xticks([]), ax.set_yticks([])
        ref, src = d["references"]["source"], d["source"]
        rect(ax, ref["box"], C_FAC, 1.0)
        rect(ax, ref["region"]["rect"], C_FAC, 1.0, ls=(0, (4, 2)))
        tag(ax, ref["region"]["rect"][0] + 6, ref["region"]["rect"][3] + 8, "'하역장 옆' 영역", C_FAC, va="top")
        tag(ax, ref["box"][0], ref["box"][1] + 6, "기준 시설 (트럭)", C_FAC, size=5.5, va="top")
        top = src["exemplar"]["top_id"]
        for c in src["candidates"][:5]:                    # 큐브 예시 점수 상위 5개
            if c["id"] == top:
                continue
            rect(ax, c["bbox_xyxy"], C_CAND, 0.6)
            ax.text(c["bbox_xyxy"][0], c["bbox_xyxy"][3] + 4, f"{c['siglip_top3'][0][1]:.3f}", color="white", fontsize=4.5, va="top", zorder=5)
        tb, s1 = src["exemplar"]["top_box"], src["exemplar"]["s1"]
        cx, cy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
        col = C_SEL if src["status"] == "SELECTED" else C_REJ
        ax.add_patch(plt.Circle((cx, cy), 38, fill=False, ec=col, lw=1.6, zorder=5))
        if src["status"] == "SELECTED":                       # 영역 이름표(영역 아래 왼쪽) 밑으로 내려 겹치지 않게 함
            tag(ax, cx + 40, ref["region"]["rect"][3] + 100, f"큐브 1위 {s1:.3f}: 영역 안 → 선택", col, va="top", ha="right", bold=True)
            ax.set_title(f"(a) 조건을 만족하는 물체가 있음 → 선택 ({n['sel_right']}/{n['sel']}장)", fontsize=7.5, pad=3)
        else:
            tag(ax, cx - 50, cy - 40, f"큐브 1위 {s1:.3f}: 영역 밖 → 보류", col, va="bottom", ha="right", bold=True)
            ax.set_title(f"(b) 조건을 만족하는 물체가 없음 → 보류 ({n['hold_correct']}/{n['hold']}장)", fontsize=7.5, pad=3)
    fig.subplots_adjust(wspace=0.03)
    save(fig, "fig2_relation")
    return n


# ================================================================== Fig. 3 객체 → 정차 목표 → 경로
def body_poly(R, x, y, yaw):
    th = math.radians(yaw)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return R.fp @ rot.T + [x, y], R.tip_line @ rot.T + [x, y]


def map_bg(ax, E, zoom=None):
    img = np.where(E.free, 0.93, 0.55)                    # 주행 영역 = 밝게, 그 밖 = 어둡게
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=[E.ox, E.ox + E.NX * E.res, E.oy, E.oy + E.NY * E.res], origin="upper", zorder=0)
    if zoom:
        ax.set_xlim(*zoom[0]), ax.set_ylim(*zoom[1])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5, length=2, pad=1)


def draw_robot(ax, R, x, y, yaw, color, ls="-", lw=0.9):
    body, tip = body_poly(R, x, y, yaw)
    ax.add_patch(Polygon(body, closed=True, fill=False, ec=color, lw=lw, ls=ls, zorder=4))
    ax.plot(tip[[0, -1], 0], tip[[0, -1], 1], color=color, lw=lw * 1.6, zorder=4, solid_capstyle="butt")
    th = math.radians(yaw)
    ax.annotate("", xy=(x + 0.11 * math.cos(th), y + 0.11 * math.sin(th)), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color, mutation_scale=5), zorder=4)


def fig3():
    d = json.loads(FINAL.read_text(encoding="utf-8"))
    assert d["result"]["status"] == "PATH_OK"
    E, R = cc.Env(), cc.Robot()
    A, B, P = d["objects"]["A"], d["objects"]["B"], d["perception"]
    sA, sB, st = d["stop_A"], d["stop_B"], d["start"]
    fig = plt.figure(figsize=(7.2, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.78, 0.62, 1.54], wspace=0.2)
    a0, a1, a2 = (fig.add_subplot(g) for g in gs)

    # (a) 사진 위 선택 결과
    a0.imshow(photo_rgb(d["photo"], d["photo_sha256"]))
    a0.set_xlim(0, 1920), a0.set_ylim(1080, 0), a0.set_xticks([]), a0.set_yticks([])
    for key, col, name in (("A", C_A, "옮길 물체: 큐브"), ("B", C_B, "놓을 곳: 빈 팔레트")):
        b = P[key]["box_px"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        rect(a0, b, col, 1.2)
        a0.add_patch(plt.Circle((cx, cy), 42, fill=False, ec=col, lw=1.2, zorder=5))
        tag(a0, cx - 55, cy, f"{name} ({P[key]['exemplar']['s1']:.3f})", col, ha="right", va="center", bold=True)
    a0.set_title("(a) 영상에서 대상 선택", fontsize=7.5, pad=3)

    # (b) 물체 위치와 정차 목표 (지도 확대)
    map_bg(a1, E, zoom=((2.372, 3.24), (0.08, 1.48)))     # 가로:세로 = 0.62, (a)·(c)와 높이를 맞춤
    for obj, stop, col, nm, ty in ((A, sA, C_A, "큐브", 0.90), (B, sB, C_B, "팔레트", 0.70)):  # 글상자는 두 정차 사각형 사이 빈 곳
        ox, oy = obj["map_xy_m"]
        a1.plot(ox, oy, marker="x", ms=5, mew=1.3, color=col, zorder=5)
        draw_robot(a1, R, *stop["xy_m"], stop["yaw_deg"], col)
        a1.plot(*stop["xy_m"], marker="o", ms=2.2, color=col, zorder=5)
        txt = f"{nm} 정차: 여유 {stop['body_clearance_cm']:.1f} cm\n그리퍼–물체 {stop['reach_gap_cm']:.1f} cm"
        a1.text(2.39, ty, txt, fontsize=4.6, va="center", ha="left", color=col, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=0.5, alpha=0.9))
    a1.set_title("(b) 물체(×)와 정차 목표", fontsize=7.5, pad=3)
    a1.set_xlabel("x (m)", fontsize=5.5, labelpad=1)

    # (c) 지도 위 경로
    map_bg(a2, E)
    draw_robot(a2, R, *st["xy_m"], st["yaw_deg"], "#444444", ls="--", lw=0.7)
    a2.text(st["xy_m"][0] + 0.18, st["xy_m"][1], "시작", fontsize=5.5, va="center", color="#333333")
    for key, col, lab in (("route_1", C_A, "경로 1: 시작 → 큐브 정차"), ("route_2", C_B, "경로 2: 큐브 → 팔레트 정차")):
        r = d[key]["route"]
        pts = np.array([r["start_xy_m"], *r["waypoints_xy_m"], r["goal_xy_m"]])
        off = 0.0 if key == "route_1" else 0.025              # 두 경로가 겹치는 구간을 구분해 보이도록 2.5 cm 띄워 그림
        a2.plot(pts[:, 0] + off, pts[:, 1] + off, color=col, lw=1.1, zorder=3,
                label=f"{lab} ({d[key]['check_teammate']['length_m']:.2f} m)")
    for obj, stop, col in ((A, sA, C_A), (B, sB, C_B)):
        a2.plot(*obj["map_xy_m"], marker="x", ms=4, mew=1.1, color=col, zorder=5)
        draw_robot(a2, R, *stop["xy_m"], stop["yaw_deg"], col, lw=0.7)
    a2.legend(loc="upper left", fontsize=4.8, frameon=True, framealpha=0.9, borderpad=0.3, handlelength=1.5)
    a2.set_title("(c) 주행 영역(밝은 부분) 위 경로", fontsize=7.5, pad=3)
    a2.set_xlabel("x (m)", fontsize=5.5, labelpad=1)
    save(fig, "fig3_goal_path")
    return d


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
