#!/usr/bin/env python3
"""KRoC 논문 그림 3개, 영어판 v2 (2026-11-23). PPT로 손수 만든 도식처럼 다시 그린다.
  python3 -I -B make_figures_v2.py
출력: out_v2/fig1_system, fig2_relation, fig3_goal_path (.png 600 dpi, .pdf, .svg)
근거 기록과 읽는 방법은 make_figures.py(v1)를 그대로 가져다 쓴다. 숫자는 모두 기록에서 읽는다.
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I 에서도 같은 폴더의 v1 을 찾게 함
import make_figures as V1         # 같은 폴더의 v1: 기록 경로, 사진 읽기(sha 검사), 차체 다각형, I2 집계, 팀원 Env/Robot

import matplotlib.pyplot as plt  # noqa: E402  (V1 이 Agg 백엔드를 이미 정함)
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent / "out_v2"
LS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
LSB = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
LSI = "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"
MONO = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
ICON = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"      # ✓ ✕ (Liberation 에 없음)
for f in (LS, LSB, LSI, MONO, ICON):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 6.5, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "savefig.dpi": 600, "axes.linewidth": 0.6})
FB = fm.FontProperties(fname=LSB)
FI = fm.FontProperties(fname=LSI)
FMONO = fm.FontProperties(fname=MONO)
FICON = fm.FontProperties(fname=ICON)

# 색 (dataviz 기본 팔레트: 계열 2색은 검사 통과, 상태색은 아이콘+글자와 함께만 씀)
INK, INK2, MUTED, HAIR = "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9"
PICK, PLACE = "#eb6834", "#2a78d6"            # 계열 1·2 (주황=집을 물체/경로 1, 파랑=놓을 곳/경로 2)
GOOD, BAD = "#0ca30c", "#d03b3b"              # 상태: 선택 / 보류
REGION = "#eda100"                            # 기준 시설 주변 영역
MODEL_FC, MODEL_EC = "#e7f0fb", "#2a78d6"
RULE_FC, RULE_EC = "#f2f1ed", "#8a8984"
IO_FC, IO_EC = "#ffffff", "#52514e"
GROUP_FC = "#f8f7f4"
DRIVE, NODRIVE = "#fbfaf6", "#d9d8d1"


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print("saved", OUT / name)


def axes_in(fig, l, b, w, h, **kw):
    W, H = fig.get_size_inches()
    return fig.add_axes([l / W, b / H, w / W, h / H], **kw)


def overlay(fig):
    """그림 전체를 덮는 좌표 = 인치. 화살표·머리띠 등 그림 사이 장식용"""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1], zorder=10)
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.axis("off")
    ax.patch.set_alpha(0)
    return ax


def rbox(ax, x0, y0, x1, y1, fc, ec, lw=0.8, ls="-", r=1.2, shadow=True, z=2):
    if shadow:
        ax.add_patch(FancyBboxPatch((x0 + 0.35, y0 - 0.45), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc="#000000", ec="none", alpha=0.10, zorder=z - 0.1))
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def status_icon(ax, xy, ok, size=5.5, coords="data", offset=(0, 0)):
    ax.annotate("✓" if ok else "✕", xy=xy, xycoords=coords, xytext=offset, textcoords="offset points",
                ha="center", va="center", color="white", fontproperties=fm.FontProperties(fname=ICON, size=size), zorder=12,
                bbox=dict(boxstyle="circle,pad=0.18", fc=GOOD if ok else BAD, ec="none"))


def chip(ax, xy, text, color, side="right", size=5.6, icon=None, coords="data", gap=4.5, bold_first=False):
    """흰 바탕·색 테두리·검은 글자 이름표. icon 이 있으면 점 위치에 상태 아이콘, 글자는 옆으로"""
    dx = (gap + (5 if icon is not None else 0)) * (1 if side == "right" else -1)
    if icon is not None:
        status_icon(ax, xy, icon, coords=coords)
    ax.annotate(text, xy=xy, xycoords=coords, xytext=(dx, 0), textcoords="offset points",
                ha="left" if side == "right" else "right", va="center", fontsize=size, color=INK, zorder=11,
                fontproperties=FB if bold_first else None,
                bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.35", fc="white", ec=color, lw=0.9))


# ================================================================== Fig. 1 System overview
def fig1():
    fig = plt.figure(figsize=(7.0, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 140), ax.set_ylim(0, 60), ax.axis("off")    # 1 칸 = 0.05 in (가로·세로 같음)

    def group(x0, y0, x1, y1, num, title):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=GROUP_FC, ec=HAIR, lw=0.7, zorder=0))
        ax.add_patch(Circle((x0 + 2.4, y1 - 2.4), 1.45, fc=INK2, ec="none", zorder=1))
        ax.text(x0 + 2.4, y1 - 2.45, str(num), ha="center", va="center", color="white", fontproperties=FB, fontsize=5.8, zorder=2)
        ax.text(x0 + 4.6, y1 - 2.45, title, ha="left", va="center", color=INK2, fontproperties=FB, fontsize=6.4, zorder=1)

    def block(x0, y0, x1, y1, title, sub="", kind="rule", ls="-", sub_align="center"):
        fc, ec = {"model": (MODEL_FC, MODEL_EC), "rule": (RULE_FC, RULE_EC), "io": (IO_FC, IO_EC)}[kind]
        rbox(ax, x0, y0, x1, y1, fc, ec, lw=0.9 if kind == "model" else 0.8, ls=ls)
        n = sub.count("\n") + 1 if sub else 0
        block_h = 2.4 + (0.9 + n * 2.15 if sub else 0)
        cy = (y0 + y1) / 2
        ty = cy + block_h / 2 - 1.2
        ax.text((x0 + x1) / 2, ty, title, ha="center", va="center", fontproperties=FB, fontsize=6.9, color=INK, zorder=3)
        if sub:
            sx = (x0 + x1) / 2 if sub_align == "center" else x0 + 2.0
            ax.text(sx, ty - 2.1, sub, ha=sub_align, va="top", fontsize=5.8, color=INK2, linespacing=1.3, zorder=3)

    def arrow(pts, color=INK2, lw=0.9, ls="-"):
        pts = np.asarray(pts, float)
        if len(pts) > 2:
            ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, lw=lw, ls=ls, solid_joinstyle="round", zorder=1.5)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, ls=ls,
                                     shrinkA=0, shrinkB=0, zorder=1.6))

    # 단계 묶음
    group(26, 40.5, 91, 59.5, 1, "Instruction parsing")
    group(26, 0.5, 102, 38.5, 2, "Perception & target selection")
    group(104, 16.5, 139.5, 59.5, 3, "Goal & path generation")

    # 입력: 지시문 말풍선
    rbox(ax, 1, 42.5, 23.5, 56, IO_FC, IO_EC)
    ax.add_patch(Polygon([[4.5, 42.6], [7.5, 42.6], [3.5, 39.4]], closed=True, fc=IO_FC, ec=IO_EC, lw=0.8, zorder=2.05))
    ax.plot([4.6, 7.4], [42.5, 42.5], color=IO_FC, lw=1.4, zorder=2.1)
    ax.text(12.25, 53.3, "Natural-language\ninstruction", ha="center", va="top", fontproperties=FB, fontsize=6.4, color=INK, linespacing=1.15, zorder=3)
    ax.text(12.25, 47.4, "“Move the red cube to\nthe empty pallet.”", ha="center", va="center", fontproperties=FI, fontsize=5.8,
            color=INK2, linespacing=1.25, zorder=3)

    # 입력: 카메라 사진
    d3 = json.loads(V1.FINAL.read_text(encoding="utf-8"))
    ax.text(12.25, 30.3, "Fixed external camera", ha="center", va="center", fontproperties=FB, fontsize=6.4, color=INK)
    th = cv2.resize(V1.photo_rgb(d3["photo"], d3["photo_sha256"]), (480, 270), interpolation=cv2.INTER_AREA)
    tax = fig.add_axes([1.6 / 140, 16.0 / 60, 21.3 / 140, 12.0 / 60 * (21.3 / 21.3)])
    tax.imshow(th), tax.set_xticks([]), tax.set_yticks([])
    for s in tax.spines.values():
        s.set_edgecolor(IO_EC), s.set_linewidth(0.8)
    ax.text(12.25, 14.0, "1920 × 1080, undistorted", ha="center", va="center", fontsize=5.6, color=INK2)

    # ① 지시 해석
    block(29, 44, 53, 55.5, "Qwen3-VL-4B-Instruct", "text only (no image)\n→ fixed task slots", kind="model")
    rbox(ax, 57, 43, 89, 56, "#ffffff", IO_EC)
    ax.text(59, 53.6, "Task slots (JSON)", ha="left", va="center", fontproperties=FB, fontsize=6.0, color=INK, zorder=3)
    sl = d3["perception"]["slots"]
    js = "\n".join(f'"{k}": "{sl[k]}"' for k in ("object", "source_landmark", "destination_kind"))
    ax.text(59, 50.8, js, ha="left", va="top", fontproperties=FMONO, fontsize=5.0, color=INK2, linespacing=1.35, zorder=3)

    # ② 장면 인식과 대상 결정
    block(27.8, 15, 48.6, 28, "Grounding DINO-tiny", "open-vocabulary\ncandidate boxes", kind="model")
    block(51, 22.5, 74.5, 34.5, "SigLIP2 base", "facility check\n(text prompts)", kind="model")
    block(51, 5, 74.5, 17, "SigLIP2 so400m", "exemplar matching\n(cube, empty pallet)", kind="model")
    rbox(ax, 78.5, 9, 100, 35.5, RULE_FC, RULE_EC)
    ax.text(89.25, 30.0, "Target selection", ha="center", va="center", fontproperties=FB, fontsize=6.9, color=INK, zorder=3)
    ax.text(89.25, 27.4, "(Python rules)", ha="center", va="center", fontsize=5.8, color=INK2, zorder=3)
    ax.text(80.3, 24.4, "• min. similarity score\n• top-1 / top-2 margin\n• pallet occupancy\n• near-landmark region\n   (spatial relation)",
            ha="left", va="top", fontsize=5.7, color=INK2, linespacing=1.38, zorder=3)
    rbox(ax, 78.5, 1.6, 100, 6.4, "#ffffff", BAD, lw=0.8, shadow=False)
    status_icon(ax, (81.6, 4.0), False, size=5.0)
    ax.text(83.6, 4.0, "Hold if uncertain", ha="left", va="center", fontsize=5.8, color=INK, zorder=3)

    # ③ 목표와 경로
    block(107, 47, 137, 56, "Pixel → map", "object-top plane, metres")
    block(107, 33.5, 137, 43.5, "Stop-pose generation", "robot footprint · drivable area\ngripper facing the object")
    block(107, 20, 137, 30, "Path planning", "4-connected A*\n+ clearance check")
    block(107, 2.5, 137, 13.5, "Mecanum robot", "ROS 2 / Nav2, route-file hand-off\n(separate experiment)", kind="io", ls=(0, (3, 2)))

    # 화살표
    arrow([(23.5, 49.5), (29, 49.5)])
    arrow([(53, 49.75), (57, 49.75)])
    arrow([(23.5, 21.5), (28.5, 21.5)])
    arrow([(48.6, 24), (49.8, 24), (49.8, 28.5), (51, 28.5)])
    arrow([(48.6, 19), (49.8, 19), (49.8, 11), (51, 11)])
    arrow([(74.5, 28.5), (78.5, 28.5)])
    arrow([(74.5, 11), (78.5, 11)])
    arrow([(84, 43), (84, 35.5)])
    arrow([(89.25, 9), (89.25, 6.4)], color=BAD)
    arrow([(100, 22), (103, 22), (103, 51.5), (107, 51.5)], color=GOOD, lw=1.0)
    status_icon(ax, (103, 37), True, size=5.0)
    ax.text(102.2, 33.9, "selected", ha="right", va="center", fontsize=5.6, color=INK, rotation=90)
    arrow([(122, 47), (122, 43.5)])
    arrow([(122, 33.5), (122, 30)])
    arrow([(122, 20), (122, 13.5)], ls=(0, (3, 2)))

    # 범례
    for i, (fc, ec, ls, s) in enumerate([(MODEL_FC, MODEL_EC, "-", "Pretrained model"), (RULE_FC, RULE_EC, "-", "Rule / geometry (Python)"),
                                         (IO_FC, IO_EC, (0, (3, 2)), "Separate experiment")]):
        y = 9.0 - i * 3.3
        ax.add_patch(FancyBboxPatch((1.5, y - 1.0), 3.6, 2.0, boxstyle="round,pad=0,rounding_size=0.5", fc=fc, ec=ec, lw=0.7, ls=ls))
        ax.text(6.2, y, s, ha="left", va="center", fontsize=5.7, color=INK2)
    save(fig, "fig1_system")


# ================================================================== Fig. 2 Selection by spatial relation
def fig2():
    n = V1.i2_counts()
    W, H = 7.0, 2.62
    fig = plt.figure(figsize=(W, H))
    pw, ph, yb = 3.44, 3.44 * 9 / 16, 0.34
    ov = overlay(fig)
    # 지시문 머리띠
    ov.add_patch(FancyBboxPatch((0.02, H - 0.27), W - 0.04, 0.22, boxstyle="round,pad=0,rounding_size=0.05", fc=GROUP_FC, ec=HAIR, lw=0.6))
    ov.text(0.12, H - 0.16, "Instruction:", ha="left", va="center", fontproperties=FB, fontsize=6.4, color=INK)
    ov.text(0.80, H - 0.16, "“Put the object next to the loading dock beside the conveyor belt.”", ha="left", va="center",
            fontproperties=FI, fontsize=6.4, color=INK2)
    ov.text(W - 0.12, H - 0.16, "evaluated here: pick-side relation “next to the loading dock”", ha="right", va="center", fontsize=5.8, color=INK2)

    for k, (run, left) in enumerate(((V1.RUN_SEL, 0.0), (V1.RUN_HOLD, W - pw))):
        d = json.loads((V1.HY / "runs" / run / "run.json").read_text(encoding="utf-8"))
        img = V1.photo_rgb(d["photo"]["path"], d["photo"]["sha256"])
        ref, src = d["references"]["source"], d["source"]
        x0, y0, x1, y1 = ref["region"]["rect"]
        dim = (img.astype(np.float32) * 0.45 + 18).clip(0, 255).astype(np.uint8)   # 영역 밖은 어둡게
        dim[y0:y1, x0:x1] = img[y0:y1, x0:x1]
        ax = axes_in(fig, left, yb, pw, ph)
        ax.imshow(dim), ax.set_xlim(0, 1920), ax.set_ylim(1080, 0), ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(INK2), s.set_linewidth(0.6)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=REGION, lw=1.4, ls=(0, (4, 2)), zorder=4))
        b = ref["box"]
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=REGION, lw=1.2, zorder=4))
        chip(ax, (x0, y1 + 34), "“Next to loading dock” region", REGION, side="right", gap=0)
        chip(ax, (b[0] - 6, (b[1] + b[3]) / 2), "Landmark: truck", REGION, side="left", gap=2)
        top = src["exemplar"]["top_id"]
        for c in src["candidates"][:5]:                            # 큐브 예시 점수 상위 5개 중 1위 밖
            if c["id"] != top:
                bb = c["bbox_xyxy"]
                ax.add_patch(Rectangle((bb[0] - 8, bb[1] - 8), bb[2] - bb[0] + 16, bb[3] - bb[1] + 16, fill=False, ec="white",
                                       lw=0.7, ls=(0, (2, 1.5)), zorder=4))
        tb, s1, ok = src["exemplar"]["top_box"], src["exemplar"]["s1"], src["status"] == "SELECTED"
        col = GOOD if ok else BAD
        cx, cy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
        ax.add_patch(Circle((cx, cy), 36, fill=False, ec=col, lw=1.6, zorder=5))
        # 확대 창 (원본 화소)
        ib = (1180, 420, 330, 330) if ok else (1020, 520, 330, 330)
        axins = ax.inset_axes(ib, transform=ax.transData)
        axins.imshow(img), axins.set_xlim(cx - 55, cx + 55), axins.set_ylim(cy + 55, cy - 55)
        axins.set_xticks([]), axins.set_yticks([])
        axins.add_patch(Rectangle((tb[0], tb[1]), tb[2] - tb[0], tb[3] - tb[1], fill=False, ec=col, lw=1.2))
        for s in axins.spines.values():
            s.set_edgecolor(col), s.set_linewidth(1.3)
        ax.indicate_inset_zoom(axins, edgecolor=col, alpha=0.9, lw=0.8)
        lab = f"Top-1 cube (sim. {s1:.3f})\n{'inside region → selected' if ok else 'outside region → held'}"
        ax.annotate(lab, xy=(ib[0] + ib[2] / 2, ib[1] + ib[3]), xycoords="data", xytext=(0, -4), textcoords="offset points",
                    ha="center", va="top", fontsize=5.6, color=INK, zorder=11, linespacing=1.25,
                    bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.35", fc="white", ec=col, lw=0.9))
        # 아래 제목
        sub = (f"(a) Object next to the landmark → selected  ({n['sel_right']} of {n['sel']} photos)" if ok else
               f"(b) No object next to the landmark → held  ({n['hold_correct']} of {n['hold']} photos)")
        status_icon(ov, (left + 0.09, yb - 0.15), ok, size=5.5)
        ov.text(left + 0.2, yb - 0.15, sub, ha="left", va="center", fontproperties=FB, fontsize=6.6, color=INK)
    save(fig, "fig2_relation")


# ================================================================== Fig. 3 From target to stop pose and path
def map_panel(ax, E, xlim, ylim):
    ax.imshow(E.free.astype(int), cmap=ListedColormap([NODRIVE, DRIVE]), vmin=0, vmax=1, interpolation="nearest",
              extent=[E.ox, E.ox + E.NX * E.res, E.oy, E.oy + E.NY * E.res], origin="upper", zorder=0)
    xs = E.ox + (np.arange(E.NX) + 0.5) * E.res
    ys = E.oy + (E.NY - 1 - np.arange(E.NY) + 0.5) * E.res
    ax.contour(xs, ys, E.free.astype(float), levels=[0.5], colors=[MUTED], linewidths=0.5, zorder=1)
    ax.set_xlim(*xlim), ax.set_ylim(*ylim), ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(INK2), s.set_linewidth(0.6)


def scale_bar(ax, x, y, length_m, label):
    ax.plot([x, x + length_m], [y, y], color=INK, lw=1.4, solid_capstyle="butt", zorder=6)
    for xx in (x, x + length_m):
        ax.plot([xx, xx], [y - 0.012, y + 0.012], color=INK, lw=0.8, zorder=6)
    ax.annotate(label, xy=(x + length_m / 2, y), xytext=(0, 2.5), textcoords="offset points", ha="center", va="bottom", fontsize=5.4, color=INK)


def robot(ax, R, x, y, yaw, color, ls="-", alpha=0.20, lw=0.9):
    body, tip = V1.body_poly(R, x, y, yaw)
    ax.add_patch(Polygon(body, closed=True, fc=color, ec="none", alpha=alpha, zorder=3))
    ax.add_patch(Polygon(body, closed=True, fill=False, ec=color, lw=lw, ls=ls, zorder=3.1))
    ax.plot(tip[[0, -1], 0], tip[[0, -1], 1], color=color, lw=2.0, solid_capstyle="butt", zorder=3.2)


def fig3():
    d = json.loads(V1.FINAL.read_text(encoding="utf-8"))
    assert d["result"]["status"] == "PATH_OK"
    E, R = V1.cc.Env(), V1.cc.Robot()
    A, B, P, sA, sB, st = d["objects"]["A"], d["objects"]["B"], d["perception"], d["stop_A"], d["stop_B"], d["start"]
    W, H = 7.0, 2.72
    fig = plt.figure(figsize=(W, H))
    yb, hp = 0.36, 2.05
    crop = (1150, 1920, 1000, 180)                                  # (a) 사진에서 대상이 있는 오른쪽 부분
    wa = hp * (crop[1] - crop[0]) / (crop[2] - crop[3])
    zb = ((2.372, 3.24), (0.08, 1.48))
    wb = hp * (zb[0][1] - zb[0][0]) / (zb[1][1] - zb[1][0])
    wc = W - wa - wb - 2 * 0.32
    la, lb, lc = 0.0, wa + 0.32, wa + wb + 0.64
    ov = overlay(fig)

    # (a) 영상에서 대상 선택
    a0 = axes_in(fig, la, yb, wa, hp)
    a0.imshow(V1.photo_rgb(d["photo"], d["photo_sha256"])), a0.set_xlim(crop[0], crop[1]), a0.set_ylim(crop[2], crop[3])
    a0.set_xticks([]), a0.set_yticks([])
    for s in a0.spines.values():
        s.set_edgecolor(INK2), s.set_linewidth(0.6)
    for key, col, name in (("A", PICK, "Pick: red cube"), ("B", PLACE, "Place: empty pallet")):
        b = P[key]["box_px"]
        a0.add_patch(FancyBboxPatch((b[0] - 9, b[1] - 9), b[2] - b[0] + 18, b[3] - b[1] + 18, boxstyle="round,pad=0,rounding_size=6",
                                    fill=False, ec=col, lw=1.5, zorder=5))
        chip(a0, (b[0] - 12, (b[1] + b[3]) / 2), f"{name}\nsim. {P[key]['exemplar']['s1']:.3f}", col, side="left", gap=1, size=5.5)
    chip(a0, (st["pixel"][0] - 70, st["pixel"][1] - 10), "Robot (start)", MUTED, side="left", gap=1, size=5.5)

    # (b) 물체 위치와 정차 자세
    a1 = axes_in(fig, lb, yb, wb, hp)
    map_panel(a1, E, *zb)
    for obj, stop, col, nm, lab_xy, side in ((A, sA, PICK, "Pick stop", (2.80, 1.17), "left"), (B, sB, PLACE, "Place stop", (2.83, 0.45), "right")):
        robot(a1, R, *stop["xy_m"], stop["yaw_deg"], col)
        ox, oy = obj["map_xy_m"]
        a1.plot([stop["xy_m"][0], ox], [stop["xy_m"][1], oy], color=INK2, lw=0.7, ls=(0, (2, 1.5)), zorder=4)
        a1.plot(*stop["xy_m"], marker="o", ms=2.6, color=col, mec="white", mew=0.5, zorder=5)
        a1.plot(ox, oy, marker="X", ms=6, color=col, mec="white", mew=0.6, zorder=6)
        dist = math.hypot(ox - stop["xy_m"][0], oy - stop["xy_m"][1]) * 100   # 기록된 두 점 사이 거리
        chip(a1, lab_xy, f"{nm}\nclearance {stop['body_clearance_cm']:.1f} cm\nstop→object {dist:.0f} cm", col, side=side, gap=0, size=5.0)
    scale_bar(a1, 2.42, 0.14, 0.20, "20 cm")

    # (c) 지도 위 경로
    a2 = axes_in(fig, lc, yb, wc, hp)
    map_panel(a2, E, (E.ox, E.ox + E.NX * E.res), (E.oy, E.oy + E.NY * E.res))
    robot(a2, R, *st["xy_m"], st["yaw_deg"], MUTED, ls=(0, (2, 1.5)), alpha=0.15, lw=0.7)
    for key, col, off in (("route_1", PICK, 0.0), ("route_2", PLACE, 0.025)):   # 겹치는 구간을 보이게 경로 2를 2.5 cm 띄워 그림
        r = d[key]["route"]
        pts = np.array([r["start_xy_m"], *r["waypoints_xy_m"], r["goal_xy_m"]]) + off
        a2.plot(pts[:-1, 0], pts[:-1, 1], color=col, lw=1.5, solid_joinstyle="round", solid_capstyle="round", zorder=4)
        a2.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7, color=col, lw=1.5, shrinkA=0, shrinkB=0, zorder=4))
        a2.plot(pts[1:-1, 0], pts[1:-1, 1], ls="none", marker="o", ms=2.6, mfc="white", mec=col, mew=0.8, zorder=5)
    for obj, stop, col in ((A, sA, PICK), (B, sB, PLACE)):
        robot(a2, R, *stop["xy_m"], stop["yaw_deg"], col, alpha=0.18, lw=0.7)
        a2.plot(*obj["map_xy_m"], marker="X", ms=5, color=col, mec="white", mew=0.5, zorder=6)
    chip(a2, (2.86, 1.42), "Pick", PICK, side="left", gap=0, size=5.4)
    chip(a2, (2.50, 0.20), "Place", PLACE, side="left", gap=0, size=5.4)
    chip(a2, (2.50, 0.60), "Start", MUTED, side="left", gap=0, size=5.4)
    scale_bar(a2, 0.10, 0.10, 0.50, "50 cm")

    # 제목, 단계 화살표, 범례
    for l, w, t in ((la, wa, "(a) Target selection in the camera image"), (lb, wb, "(b) Object vs. stop pose"), (lc, wc, "(c) Planned route on the map")):
        ov.text(l, yb + hp + 0.1, t, ha="left", va="bottom", fontproperties=FB, fontsize=6.8, color=INK)
    for xm, lab in ((wa + 0.16, "pixel\n→ map"), (wa + wb + 0.48, "A*\nplanning")):
        ov.add_patch(FancyArrowPatch((xm - 0.12, yb + hp / 2), (xm + 0.13, yb + hp / 2), arrowstyle="simple,head_width=7,head_length=5,tail_width=3",
                                     mutation_scale=1, fc=MUTED, ec="none"))
        ov.text(xm, yb + hp / 2 - 0.12, lab, ha="center", va="top", fontsize=5.2, color=INK2, linespacing=1.15)
    handles = [Line2D([], [], color=PICK, lw=1.5, label=f"Route 1: start → pick stop ({d['route_1']['check_teammate']['length_m']:.2f} m)"),
               Line2D([], [], color=PLACE, lw=1.5, label=f"Route 2: pick → place stop ({d['route_2']['check_teammate']['length_m']:.2f} m)"),
               Line2D([], [], ls="none", marker="X", ms=5, color=INK2, mec="white", label="Object position"),
               Patch(fc="#d6d6d6", ec=INK2, lw=0.7, label="Robot footprint (bar = gripper)"),
               Patch(fc=DRIVE, ec=MUTED, lw=0.6, label="Drivable area")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=5.6, frameon=False, bbox_to_anchor=(0.5, -0.005),
               handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
    save(fig, "fig3_goal_path")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
