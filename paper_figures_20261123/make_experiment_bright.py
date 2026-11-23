#!/usr/bin/env python3
"""밝은 실험 자료 3개. 저장된 기록만 읽으며 모델·경로 계획은 실행하지 않는다.
실행: python3 -I -B make_experiment_bright.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as V1

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent / "out_bright"
INK, MUTED, LINE = "#20364D", "#52677B", "#DCE7EE"
BLUE, MINT, CORAL = "#299FD6", "#27AD91", "#F08C6D"
PALE_BLUE, PALE_MINT, PALE_CORAL = "#E7F5FF", "#E5F8F0", "#FFF0E8"
plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "svg.fonttype": "path"})


def heading(fig, number, title, subtitle):
    fig.text(.045, .95, f"{number}  /  EXPERIMENT RESULTS", fontsize=10, color=BLUE, weight="bold")
    fig.text(.045, .884, title, fontsize=23, weight="bold")
    fig.text(.045, .829, subtitle, fontsize=10.5, color=MUTED)


def save(fig, name):
    fig.canvas.draw()
    frame = fig.bbox
    # 모든 텍스트가 출력 경계 안에 있는지 확인한다.
    for text in fig.findobj(plt.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(fig.canvas.get_renderer())
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= frame.x1 + 1 and box.y1 <= frame.y1 + 1, text.get_text()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)
    print("saved", name)


def comparison_table(summary):
    methods = [("qwen_name", "Qwen3-VL-4B", "후보별 이름 선택"),
               ("siglip_base_text", "SigLIP2 base", "이름 라벨"),
               ("siglip_so400m_text", "SigLIP2 so400m", "이름 라벨"),
               ("siglip_base_exemplar", "SigLIP2 base", "예시 이미지"),
               ("siglip_so400m_exemplar", "SigLIP2 so400m", "예시 이미지 · 현재 방식")]
    headers = ["모델 / 후보 선택 방식", "I1 선택 성공\nA·B 모두 정답", "I1 큐브\n정답", "I1 빈 팔레트\n정답", "I2 대상 정답\n수행 가능 사진", "잘못 확정\nI1·I2 합계"]
    rows = []
    for key, model, method in methods:
        r = summary[key]
        n = len(r["A_status_I1"])
        assert n == len(r["B_status_I1"]) == 8
        assert all(0 <= r[k] <= n for k in ("I1_success", "I1_A_right", "I1_B_right"))
        assert 0 <= r["I2_A_right_valid"] <= 2
        rows.append([f"{model}\n{method}", f"{r['I1_success']}/{n}", f"{r['I1_A_right']}/{n}",
                     f"{r['I1_B_right']}/{n}", f"{r['I2_A_right_valid']}/2", str(r["wrong_commits"])])
    fig = plt.figure(figsize=(12, 6))
    heading(fig, "01", "이름 라벨과 예시 이미지의 후보 선택 결과", "개발 사진 8장 · 다섯 방식 모두 같은 사진과 후보 crop 사용")
    ax = fig.add_axes([.045, .22, .91, .54])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", colLoc="center",
                     colWidths=[.33, .16, .12, .14, .15, .10], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(3)
        cell.PAD = .07
        cell.set_facecolor(PALE_BLUE if r == 0 else PALE_MINT if r >= 4 else "#F5F8FB")
        cell.get_text().set_color(INK)
        if r == 0 or (r >= 4 and c == 1):
            cell.get_text().set_weight("bold")
        if c == 0:
            cell.get_text().set_ha("left")
    fig.text(.05, .153, "I1: 빨간 큐브 → 빈 팔레트   |   I2: ‘하역장 옆’ 대상 선택 (수행 가능 사진 2장)", fontsize=10, color=MUTED)
    fig.text(.05, .105, "선택 성공은 주행·이송 완료를 뜻하지 않습니다. 개발 사진에서 기준값을 정했으며, 정답은 AI 판독입니다.", fontsize=10, color=MUTED)
    save(fig, "01_method_comparison_table")
    with (OUT / "01_method_comparison_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows([headers, *rows])


def relation_chart(counts):
    fig = plt.figure(figsize=(12, 6))
    heading(fig, "02", "공간 관계: 대상 선택·보류와 목적지 판정", "지시: 하역장 옆에 있는 물체를 컨베이어 벨트 옆에 놓아줘.")
    metrics = [("대상 선택 정답", counts["sel_right"], counts["sel"], MINT, PALE_MINT, "‘하역장 옆’ 조건을 만족한 사진"),
               ("올바른 선택 보류", counts["hold_correct"], counts["hold"], BLUE, PALE_BLUE, "큐브는 찾았으나 ‘하역장 옆’이 아닌 사진"),
               ("목적지 정답", counts["B_right"], counts["sel"], CORAL, PALE_CORAL, "대상을 선택한 사진의 ‘컨베이어 옆’ 목적지")]
    ax = fig.add_axes([.28, .27, .50, .46])
    ax.set_xlim(0, 100)
    ax.set_ylim(-.6, 2.6)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="x", length=0, pad=10)
    ax.grid(axis="x", color=LINE, linewidth=.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for y, (label, right, total, color, pale, detail) in zip([2, 1, 0], metrics):
        assert total > 0 and 0 <= right <= total
        ax.barh(y, 100, height=.34, color=pale, zorder=2)
        ax.barh(y, right / total * 100, height=.34, color=color, zorder=3)
        if right == 0:
            ax.plot(0, y, "o", color=color, ms=7, clip_on=False, zorder=4)
        ax.text(-5, y, label, ha="right", va="center", fontsize=13, weight="bold")
        ax.text(106, y + .025, f"{right}/{total}", fontsize=24, color=color, weight="bold", va="center")
        ax.text(106, y - .24, f"{right / total:.0%}", fontsize=10, color=MUTED, va="center")
        ax.text(0, y - .32, detail, fontsize=9.5, color=MUTED, va="top")
    fig.text(.05, .153, "대상 선택·보류는 출발지 관계의 결과입니다. 목적지 관계의 정답은 0/2이며, 작업 전체 성공과 구분해야 합니다.", fontsize=10, color=MUTED)
    fig.text(.05, .105, f"개발 사진 {counts['n']}장 · 막대마다 분모가 다름 · 정답: AI 판독, 사람 미확인", fontsize=10, color=MUTED)
    save(fig, "02_spatial_relation_chart")


def map_panel(ax, env, xlim, ylim):
    ax.imshow(env.free.astype(int), cmap=ListedColormap(["#F1EEF8", PALE_MINT]), vmin=0, vmax=1,
              extent=[env.ox, env.ox + env.NX * env.res, env.oy, env.oy + env.NY * env.res], origin="upper")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8, length=0)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9, labelpad=1)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)


def goal_path(records):
    d = json.loads(V1.FINAL.read_text(encoding="utf-8"))
    passed = [r for r in records if r["result"]["status"] == "PATH_OK"]
    aligned = [r for r in passed if all(r[k]["lateral_within_limit"] for k in ("stop_A", "stop_B"))]
    checks = [r[k]["check_teammate"] for r in passed for k in ("route_1", "route_2")]
    minimum = min(c["min_body_clearance_cm"] for c in checks)
    assert len(records) == 15 and len(passed) == 9 and len(aligned) == 1
    assert aligned[0]["photo_sha256"] == d["photo_sha256"]
    assert minimum >= 3 and all(c["bad_cm"] == c["tight_cm"] == 0 for c in checks)
    excess = [abs(r[k]["lateral_offset_cm"]) for r in passed for k in ("stop_A", "stop_B") if not r[k]["lateral_within_limit"]]
    env, robot = V1.cc.Env(), V1.cc.Robot()
    fig = plt.figure(figsize=(12, 7))
    heading(fig, "03", "대상 위치에서 정차 목표와 경로까지", "대표 예: 사진 220632 · 빨간 큐브를 비어 있는 팔레트로 옮겨줘.")
    a0 = fig.add_axes([.045, .31, .31, .43])
    a1 = fig.add_axes([.405, .31, .19, .43])
    a2 = fig.add_axes([.65, .31, .305, .43])
    a0.imshow(V1.photo_rgb(d["photo"], d["photo_sha256"]))
    a0.set_xlim(1150, 1920)
    a0.set_ylim(1000, 180)
    a0.set_xticks([])
    a0.set_yticks([])
    a0.set_title("(a) 영상에서 대상 선택", loc="left", pad=14, fontsize=12, weight="bold")
    map_panel(a1, env, (2.37, 3.25), (.08, 1.48))
    a1.set_title("(b) 물체와 정차 목표", pad=14, fontsize=12, weight="bold")
    map_panel(a2, env, (env.ox, env.ox + env.NX * env.res), (env.oy, env.oy + env.NY * env.res))
    a2.set_title("(c) 지도 위 두 구간 경로", loc="left", pad=14, fontsize=12, weight="bold")
    for key, stop_key, color, label in (("A", "stop_A", CORAL, "큐브"), ("B", "stop_B", BLUE, "빈 팔레트")):
        b = d["perception"][key]["box_px"]
        center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        V1.rect(a0, b, color, 2)
        a0.annotate(label, center, xytext=(-35, 0), textcoords="offset points", ha="right", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=.4", fc="white", ec=color), arrowprops=dict(arrowstyle="-", color=color))
        stop, obj = d[stop_key], d["objects"][key]["map_xy_m"]
        for ax in (a1, a2):
            V1.draw_robot(ax, robot, *stop["xy_m"], stop["yaw_deg"], color, lw=1.2)
            ax.plot(*obj, marker="x", ms=8, mew=2, color=color, zorder=6)
            ax.plot(*stop["xy_m"], marker="o", ms=4, color=color, zorder=6)
        a1.plot([obj[0], stop["xy_m"][0]], [obj[1], stop["xy_m"][1]], color=color, ls=":", lw=1)
    for key, color, style, width in (("route_1", CORAL, "-", 4), ("route_2", BLUE, "--", 2)):
        r = d[key]["route"]
        pts = np.array([r["start_xy_m"], *r["waypoints_xy_m"], r["goal_xy_m"]])
        assert pts.shape[1] == 2 and np.isfinite(pts).all()
        # 겹치는 실제 경로는 선 종류로 구분한다. 지도 좌표를 이동하지 않는다.
        a2.plot(pts[:, 0], pts[:, 1], color=color, ls=style, lw=width, zorder=3)
    a2.plot(*d["start"]["xy_m"], marker="o", ms=6, color=INK, zorder=7)
    a2.annotate("시작", d["start"]["xy_m"], xytext=(-4, -15), textcoords="offset points", ha="right", fontsize=9)
    handles = [Line2D([], [], color=CORAL, lw=3, label=f"경로 1: 시작 → 큐브 ({d['route_1']['check_teammate']['length_m']:.2f} m)"),
               Line2D([], [], color=BLUE, lw=2, ls="--", label=f"경로 2: 큐브 → 팔레트 ({d['route_2']['check_teammate']['length_m']:.2f} m)"),
               Line2D([], [], color=MUTED, marker="x", ls="none", label="물체 위치"),
               Patch(fc="white", ec=MUTED, label="차체 · 굵은 선은 그리퍼 끝")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .225), ncol=4, frameon=False, fontsize=9)
    fig.text(.05, .175, f"전체 {len(records)}장 중 경로·여유 검사 통과 {len(passed)}장  |  경로 전체 최소 차체 여유 {minimum:.1f} cm  |  두 정차 위치 모두 옆 오차 기준 이내: {len(aligned)}/{len(passed)}", fontsize=10, weight="bold")
    fig.text(.05, .125, f"이 그림은 가장 깔끔한 사례입니다. 나머지 {len(passed) - len(aligned)}장은 한 정차 위치에서 옆 어긋남 {min(excess):.1f}–{max(excess):.1f} cm (기준 2 cm).", fontsize=10, color=MUTED)
    fig.text(.05, .081, "지도: env2_20261119, 단위 m · 경로 좌표 이동 없음 · 그리퍼 끝은 추정값이며 실제 파지·이송 성공은 미검증", fontsize=9, color=MUTED)
    save(fig, "03_goal_path_visualization")
    return {"photos": len(records), "path_ok": len(passed), "aligned_both": len(aligned),
            "min_body_clearance_cm": minimum, "excess_lateral_cm": [min(excess), max(excess)]}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    ablation = V1.WS / "hybrid_exemplar_20261121/ablation_20261122_140934/ablation_summary.json"
    summary = json.loads(ablation.read_text(encoding="utf-8"))
    counts = V1.i2_counts()
    assert (counts["n"], counts["sel_right"], counts["hold_correct"], counts["B_right"]) == (8, 2, 6, 0)
    comparison_table(summary)
    relation_chart(counts)
    paths = sorted(V1.FINAL.parent.parent.glob("*/nl_to_path.json"))
    route_metrics = goal_path([json.loads(p.read_text(encoding="utf-8")) for p in paths])
    (OUT / "source_metrics.json").write_text(json.dumps({"ablation_source": str(ablation),
        "relation_source": str(V1.HY / "task_evaluation.json"), "route_sources": [str(p) for p in paths],
        "relation_counts": counts, "route_metrics": route_metrics}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
