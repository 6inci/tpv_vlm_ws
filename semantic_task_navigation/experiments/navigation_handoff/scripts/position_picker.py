#!/usr/bin/env python3
"""Position picker (navigation hand-off). Per photo the user:
  - clicks A (red cube to carry) and B (placement target on the pallet); optional separate gripper work targets,
  - checks the stored AUTOMATIC robot estimate (corrects it only if wrong) and sets the heading (no automatic heading exists),
  - approves -> data/approved_positions/<photo>.yaml (+ immutable history copy). Path planning runs only from that file.

  /home/binci/miniconda3/envs/stnav_gui/bin/python experiments/navigation_handoff/scripts/position_picker.py

Mouse: left click = place the marker of the selected mode, wheel = zoom at the cursor, right / middle drag = pan.
No model call. Coordinates use the PROVISIONAL floor transform (no validated homography); grasp / placement stay UNVERIFIED.

GUI toolkit: tkinter (standard library, already in stnav_gui). The PySide6 6.12.0 of stnav_gui aborts with "free(): invalid pointer"
after ~550 Qt signal emissions (reproduced 2026-11-20 without any GUI and in a clean environment), so Qt is no longer used here.
State handling, approval files, coordinate transform and planning are unchanged (object_positions.py, path_planning.py).
"""
import json
import math
import subprocess
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coordinate_transform as CT  # noqa: E402
import object_positions as OP  # noqa: E402

MODES = [("A_obj", "A: 빨간 큐브 위치"), ("A_tgt", "A: 그리퍼 작업점 (선택)"), ("B_obj", "B: 팔레트 위치"),
         ("B_tgt", "B: 배치 작업점 (선택)"), ("robot", "로봇 위치 수정"), ("yaw", "로봇 방향: 전면 쪽 클릭")]
HEADINGS = [(0.0, "→ 0° (이미지 오른쪽)"), (90.0, "↑ 90° (트럭 쪽)"), (180.0, "← 180° (이미지 왼쪽)"), (-90.0, "↓ -90° (카메라 쪽)")]
COL = {"A": "#e61e1e", "B": "#1e78ff", "robot": "#00e6e6", "head": "#ff2828", "body": "#ffff00", "auto": "#c8c8c8"}


class ZoomCanvas(tk.Canvas):
    """Image viewer: only the visible part of the photo is resampled (PIL) at the current zoom. Overlay drawn in canvas px."""

    def __init__(self, master, on_click=None, on_hover=None, overlay=None):
        super().__init__(master, bg="#202020", highlightthickness=0, cursor="crosshair")
        self.img, self.s, self.ox, self.oy = None, 1.0, 0.0, 0.0
        self._photo, self._pan, self._pending, self._fitted = None, None, False, False
        self.on_click, self.on_hover, self.overlay = on_click, on_hover, overlay
        self.bind("<Configure>", self._on_configure)              # not "_configure": that name is tkinter's own config() helper
        self.bind("<Button-1>", self._click)
        self.bind("<Motion>", self._motion)
        for b in (2, 3):
            self.bind(f"<ButtonPress-{b}>", self._pan_start)
            self.bind(f"<B{b}-Motion>", self._pan_move)
            self.bind(f"<ButtonRelease-{b}>", self._pan_end)
        self.bind("<Button-4>", lambda e: self._zoom(e, 1.25))          # X11 wheel
        self.bind("<Button-5>", lambda e: self._zoom(e, 0.8))
        self.bind("<MouseWheel>", lambda e: self._zoom(e, 1.25 if e.delta > 0 else 0.8))

    def set_image(self, path):
        with Image.open(path) as im:
            self.img = im.convert("RGB")
        self._fitted = False
        self.fit()

    def fit(self):
        if self.img is None:
            return
        w, h = max(self.winfo_width(), 2), max(self.winfo_height(), 2)
        W, H = self.img.size
        self.s = min(w / W, h / H)
        self.ox, self.oy = (W - w / self.s) / 2, (H - h / self.s) / 2
        self._fitted = w > 10
        self.request_render()

    def _on_configure(self, _e):
        if self.img is not None and not self._fitted:
            self.fit()
        else:
            self.request_render()

    def to_img(self, cx, cy):
        return self.ox + cx / self.s, self.oy + cy / self.s

    def to_canvas(self, x, y):
        return (x - self.ox) * self.s, (y - self.oy) * self.s

    def request_render(self):
        if not self._pending:
            self._pending = True
            self.after_idle(self.render)

    def render(self):
        self._pending = False
        self.delete("all")
        if self.img is None:
            return
        w, h = self.winfo_width(), self.winfo_height()
        W, H = self.img.size
        x0, y0 = max(0, int(math.floor(self.ox))), max(0, int(math.floor(self.oy)))
        x1, y1 = min(W, int(math.ceil(self.ox + w / self.s))), min(H, int(math.ceil(self.oy + h / self.s)))
        if x1 > x0 and y1 > y0:
            dw, dh = max(1, round((x1 - x0) * self.s)), max(1, round((y1 - y0) * self.s))
            crop = self.img.crop((x0, y0, x1, y1)).resize((dw, dh), Image.NEAREST if self.s >= 2 else Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(crop)
            cx, cy = self.to_canvas(x0, y0)
            self.create_image(round(cx), round(cy), image=self._photo, anchor="nw")
        if self.overlay:
            self.overlay(self)

    def _inside(self, x, y):
        return self.img is not None and 0 <= x < self.img.size[0] and 0 <= y < self.img.size[1]

    def _click(self, e):
        x, y = self.to_img(e.x, e.y)
        if self.on_click and self._inside(x, y):
            self.on_click(x, y)

    def _motion(self, e):
        x, y = self.to_img(e.x, e.y)
        if self.on_hover and self._inside(x, y):
            self.on_hover(x, y)

    def _zoom(self, e, f):
        if self.img is None:
            return
        ns = self.s * f
        fit_s = min(self.winfo_width() / self.img.size[0], self.winfo_height() / self.img.size[1])
        if not (fit_s * 0.5 <= ns <= 40):
            return
        ix, iy = self.to_img(e.x, e.y)
        self.s, self.ox, self.oy = ns, ix - e.x / ns, iy - e.y / ns
        self.request_render()

    def _pan_start(self, e):
        self._pan = (e.x, e.y)
        self.config(cursor="fleur")

    def _pan_move(self, e):
        if self._pan is None:
            return
        self.ox -= (e.x - self._pan[0]) / self.s
        self.oy -= (e.y - self._pan[1]) / self.s
        self._pan = (e.x, e.y)
        self.request_render()

    def _pan_end(self, _e):
        self._pan = None
        self.config(cursor="crosshair")


def draw_text(c, x, y, s, color, size=11):
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        c.create_text(x + dx, y + dy, text=s, fill="black", anchor="sw", font=("TkDefaultFont", size, "bold"))
    c.create_text(x, y, text=s, fill=color, anchor="sw", font=("TkDefaultFont", size, "bold"))


def draw_marker(c, cx, cy, color, label, shape="cross"):
    """Screen-size marker (does not scale with zoom) with a label."""
    for w, col in ((5, "black"), (2, color)):
        if shape == "cross":
            for a, b in (((-14, 0), (-3, 0)), ((3, 0), (14, 0)), ((0, -14), (0, -3)), ((0, 3), (0, 14))):
                c.create_line(cx + a[0], cy + a[1], cx + b[0], cy + b[1], fill=col, width=w)
            c.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, outline=col, width=max(1, w - 3))
        else:
            c.create_polygon(cx, cy - 8, cx + 8, cy, cx, cy + 8, cx - 8, cy, outline=col, fill="", width=w)
    if label:
        draw_text(c, cx + 10, cy - 8, label, color, 12)


class Picker:
    def __init__(self, root):
        self.root = root
        self.cfg = CT.load_config()
        self.photos = CT.resolve_photos(self.cfg)
        self.frame, _, self.map_run = OP.load_step2_frame(self.cfg)
        self.nominal = (self.cfg["robot"]["length_cm"], self.cfg["robot"]["width_cm"])
        self.state, self.photo, self.proc = {}, None, None
        root.title("Navigation Handoff - 위치 지정 (A 큐브 / B 배치 목표 / 로봇)")
        root.geometry("1700x980")

        self.status = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.status, anchor="w", relief="sunken").pack(side="bottom", fill="x")
        left = ttk.Frame(root, width=300, padding=4)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="1. 사진 선택\n/home/binci/Pictures/Calibration").pack(anchor="w")
        self.list = tk.Listbox(left, width=42, exportselection=False, activestyle="none")
        self.list.pack(fill="both", expand=True)
        self.list.bind("<<ListboxSelect>>", self._list_select)

        right = ttk.Frame(root, width=430, padding=4)
        right.pack(side="right", fill="y")
        gm = ttk.LabelFrame(right, text="2. 클릭 모드 (왼쪽 클릭으로 지정, 다시 클릭하면 수정)", padding=4)
        gm.pack(fill="x")
        self.mode = tk.StringVar(value="A_obj")
        self.mode_buttons = {}
        for k, lab in MODES:
            rb = ttk.Radiobutton(gm, text=lab, value=k, variable=self.mode)
            rb.pack(anchor="w")
            self.mode_buttons[k] = rb
        gh = ttk.LabelFrame(right, text="3. 로봇 전면(그리퍼) 방향", padding=4)
        gh.pack(fill="x", pady=4)
        for i, (yaw, lab) in enumerate(HEADINGS):
            ttk.Button(gh, text=lab, command=lambda y=yaw: self.set_yaw(y, "USER_SET_BUTTON")).grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        gt = ttk.LabelFrame(right, text="도구", padding=4)
        gt.pack(fill="x")
        for i, (lab, fn) in enumerate((("화면 맞춤", lambda: self.view.fit()), ("로봇: 자동 추정값으로", self.reset_robot),
                                       ("A 작업점 = 큐브 클릭점", lambda: self.reset_target("A")), ("B 작업점 = 팔레트 클릭점", lambda: self.reset_target("B")))):
            ttk.Button(gt, text=lab, command=fn).grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        self.info = tk.Text(right, width=58, height=26, wrap="word", state="disabled", font=("TkDefaultFont", 10))
        self.info.pack(fill="both", expand=True, pady=4)
        self.btn_approve = ttk.Button(right, text="4. 위치 승인 및 저장", command=self.approve)
        self.btn_plan = ttk.Button(right, text="5. 경로계획 실행 (승인된 좌표만)", command=self.run_plan)
        for b in (self.btn_approve, self.btn_plan):
            b.pack(fill="x", ipady=6, pady=2)

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(side="left", fill="both", expand=True)
        self.view = ZoomCanvas(self.tabs, on_click=self.on_click, on_hover=self.on_hover, overlay=self.draw_overlay)
        self.res_photo, self.res_map = ZoomCanvas(self.tabs), ZoomCanvas(self.tabs)
        self.tabs.add(self.view, text="위치 지정")
        self.tabs.add(self.res_photo, text="결과: 사진 위 경로")
        self.tabs.add(self.res_map, text="결과: 지도 위 경로")
        self.refresh_list()
        self.select_index(0)

    # ---------------------------------------------------------------- state (unchanged logic)
    def refresh_list(self):
        sel = self.list.curselection()
        self.list.delete(0, "end")
        d = CT.resolve(self.cfg["approved_positions_dir"])
        for p in self.photos:
            ok = (d / f"{p['name']}.yaml").exists()
            auto = OP.robot_auto_estimate(self.cfg, p) is not None
            self.list.insert("end", f"{p['current_file']} ({p['name']}, {p['earlier_id']}{', ' + p['earlier_role'] if p['earlier_role'] else ''})"
                                    f"  {'[승인됨]' if ok else '[미승인]'}  로봇 자동추정 {'있음' if auto else '없음'}")
        if sel:
            self.list.selection_set(sel[0])

    def select_index(self, i):
        self.list.selection_clear(0, "end")
        self.list.selection_set(i)
        self.list.see(i)
        self.select_photo(i)

    def _list_select(self, _e):
        sel = self.list.curselection()
        if sel and (self.photo is None or self.photos[sel[0]]["name"] != self.photo["name"]):
            self.select_photo(sel[0])

    def fresh_state(self, p):
        auto = OP.robot_auto_estimate(self.cfg, p)
        st = {"A_obj": None, "A_tgt": None, "A_tgt_src": "SAME_AS_OBJECT_CLICK", "B_obj": None, "B_tgt": None, "B_tgt_src": "SAME_AS_OBJECT_CLICK",
              "robot": auto["base_link_image_px"] if auto else None, "robot_src": "AUTO_STORED_DETECTION" if auto else None,
              "yaw": None, "yaw_src": None, "auto": auto, "approved": None, "dirty": False}
        try:
            doc, _ = OP.load_approved(self.cfg, p["name"])                     # reuse the approved positions of this photo
        except FileNotFoundError:
            return st
        st.update({"A_obj": doc["A_cube"]["object_image_px"], "A_tgt": doc["A_cube"]["grasp_target_image_px"],
                   "A_tgt_src": doc["A_cube"]["grasp_target_source"], "B_obj": doc["B_placement"]["pallet_image_px"],
                   "B_tgt": doc["B_placement"]["placement_target_image_px"], "B_tgt_src": doc["B_placement"]["placement_target_source"],
                   "robot": doc["robot"]["base_link_image_px"], "robot_src": doc["robot"]["base_link_source"],
                   "yaw": doc["robot"]["yaw_map_deg"], "yaw_src": doc["robot"]["yaw_source"], "approved": doc["approved_at"]})
        return st

    def select_photo(self, i):
        self.photo = self.photos[i]
        if self.photo["name"] not in self.state:
            self.state[self.photo["name"]] = self.fresh_state(self.photo)
        self.view.set_image(self.photo["path"])
        self.tabs.select(0)
        self.redraw()

    @property
    def st(self):
        return self.state[self.photo["name"]]

    def changed(self):
        self.st["dirty"] = True
        self.redraw()

    def on_click(self, x, y):
        k = self.mode.get()
        xy = [round(x, 1), round(y, 1)]
        st = self.st
        if k in ("A_obj", "B_obj"):
            s = k[0]
            st[k] = xy
            if st[f"{s}_tgt_src"] == "SAME_AS_OBJECT_CLICK":
                st[f"{s}_tgt"] = xy
        elif k in ("A_tgt", "B_tgt"):
            st[k], st[f"{k[0]}_tgt_src"] = xy, "USER_SET"
        elif k == "robot":
            st["robot"], st["robot_src"] = xy, "USER_CORRECTED"
        elif k == "yaw":
            if st["robot"] is None:
                messagebox.showwarning("방향", "먼저 로봇 위치를 정하세요.", parent=self.root)
                return
            st["yaw"], st["yaw_src"] = OP.yaw_from_points(self.frame, st["robot"], xy), "USER_SET_CLICK"
        self.changed()

    def on_hover(self, x, y):
        m = self.frame.img_to_map([[x, y]])[0]
        self.status.set(f"image ({x:.1f}, {y:.1f}) px  ->  map ({m[0]:.3f}, {m[1]:.3f}) m  [PROVISIONAL floor transform, z = 0]")

    def set_yaw(self, yaw, src):
        self.st["yaw"], self.st["yaw_src"] = float(yaw), src
        self.changed()

    def reset_robot(self):
        a = self.st["auto"]
        if not a:
            messagebox.showinfo("로봇", "이 사진에는 저장된 로봇 자동 추정 결과가 없습니다. '로봇 위치 수정' 모드로 지정하세요.", parent=self.root)
            return
        self.st["robot"], self.st["robot_src"] = a["base_link_image_px"], "AUTO_STORED_DETECTION"
        self.changed()

    def reset_target(self, s):
        self.st[f"{s}_tgt"], self.st[f"{s}_tgt_src"] = self.st[f"{s}_obj"], "SAME_AS_OBJECT_CLICK"
        self.changed()

    # ---------------------------------------------------------------- drawing / info
    def redraw(self):
        self.view.request_render()
        self.update_info()

    def draw_overlay(self, c):
        st = self.st
        tc = c.to_canvas
        if st["auto"]:
            b = st["auto"]["bbox_image_px"]
            (x0, y0), (x1, y1) = tc(b[0], b[1]), tc(b[2], b[3])
            c.create_rectangle(x0, y0, x1, y1, outline=COL["auto"], dash=(6, 4), width=2)
        if st["robot"] is not None:
            if st["yaw"] is not None:
                poly = OP.body_corners_img(self.frame, st["robot"], st["yaw"], *self.nominal)
                c.create_polygon(*[v for p in poly for v in tc(*p)], outline=COL["body"], fill="", width=2)
                m0 = self.frame.img_to_map([st["robot"]])[0]
                a = math.radians(st["yaw"])
                tip = self.frame.map_to_img([m0 + 0.3 * np.array([math.cos(a), math.sin(a)])])[0]
                (bx, by), (tx, ty) = tc(*st["robot"]), tc(*tip)
                c.create_line(bx, by, tx, ty, fill=COL["head"], width=3, arrow="last", arrowshape=(14, 16, 6))
            draw_marker(c, *tc(*st["robot"]), COL["robot"], "ROBOT" + (" (auto)" if st["robot_src"] == "AUTO_STORED_DETECTION" else ""))
        for s, obj, tgt in (("A", "A_obj", "A_tgt"), ("B", "B_obj", "B_tgt")):
            if st[obj] is not None:
                draw_marker(c, *tc(*st[obj]), COL[s], s)
            if st[tgt] is not None and st[f"{s}_tgt_src"] == "USER_SET":
                draw_marker(c, *tc(*st[tgt]), COL[s], f"{s} target", "diamond")

    def update_info(self):
        st, p = self.st, self.photo
        L = [f"사진: {p['current_file']}  ({p['name']}, 기존 ID {p['earlier_id']})", f"sha256: {p['sha256'][:16]}...", ""]
        L.append(f"상태: {'승인됨 ' + st['approved'] if st['approved'] and not st['dirty'] else '미승인' + (' (승인 후 수정됨)' if st['approved'] else '')}")
        L.append("")

        def line(name, xy):
            if xy is None:
                return f"{name}: 미지정"
            q = OP._pt(self.frame, xy)
            return f"{name}: img {q['image_px']}  map {q['map_m']} m"
        L.append(line("A 큐브", st["A_obj"]))
        L.append(line("  A 그리퍼 작업점", st["A_tgt"]) + f"  [{st['A_tgt_src']}]")
        L.append(line("B 팔레트", st["B_obj"]))
        L.append(line("  B 배치 작업점", st["B_tgt"]) + f"  [{st['B_tgt_src']}]")
        L.append(line("로봇 base_link", st["robot"]) + f"  [{st['robot_src']}]")
        L.append(f"로봇 방향 yaw_map: {'미지정' if st['yaw'] is None else st['yaw']}  [{st['yaw_src']}]")
        L.append("")
        a = st["auto"]
        L.append("로봇 자동 추정: " + (f"{a['how']}, box {a['bbox_image_px']}, 중심점 사용" if a else "저장된 결과 없음 -> 수동 지정"))
        L.append("방향 자동 추정: 프로젝트에 없음 -> 버튼 또는 클릭으로 지정")
        L += ["", "주의:", "- 좌표는 잠정 바닥 변환(매트 이음선, 오차 3.5~5.2 cm)이며 정식 Homography가 아닙니다.",
              "- 바닥(z=0) 투영이며 큐브/팔레트 높이 시차는 보정하지 않습니다.",
              "- 작업점은 그리퍼 중심이 올 목표입니다. 지정하지 않으면 클릭점과 같습니다(임시).",
              "- 그리퍼 오프셋 미측정 -> 파지/배치 가능성은 UNVERIFIED입니다.",
              "- 로봇 위치는 바퀴 중심(바닥 높이)을 가리키도록 하세요."]
        self.info.config(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(L))
        self.info.config(state="disabled")
        ready = all(st[k] is not None for k in ("A_obj", "B_obj", "robot", "yaw"))
        self.btn_approve.state(["!disabled"] if ready else ["disabled"])
        self.btn_plan.state(["!disabled"] if (st["approved"] and not st["dirty"] and self.proc is None) else ["disabled"])

    # ---------------------------------------------------------------- approve / plan
    def approve(self):
        st, p = self.st, self.photo
        if not all(st[k] is not None for k in ("A_obj", "B_obj", "robot", "yaw")):
            return
        msg = (f"{p['current_file']} 의 위치를 승인하고 저장할까요?\n\n"
               f"A 큐브 {st['A_obj']} (작업점 {st['A_tgt']}, {st['A_tgt_src']})\n"
               f"B 팔레트 {st['B_obj']} (작업점 {st['B_tgt']}, {st['B_tgt_src']})\n"
               f"로봇 {st['robot']} [{st['robot_src']}], yaw {st['yaw']}° [{st['yaw_src']}]\n\n"
               "좌표 변환은 잠정값이며 파지/배치 가능성은 검증되지 않은 상태로 저장됩니다.")
        if not messagebox.askyesno("위치 승인", msg, parent=self.root):
            return
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        doc = OP.build_approved(p, st, self.frame, st["auto"], now)
        img = cv2.imread(p["path"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cur, hist = OP.save_approved(self.cfg, doc, OP.render_preview(img, doc, self.frame, self.nominal))
        OP.load_approved(self.cfg, p["name"])                                  # the saved file must pass the planner's own check
        st["approved"], st["dirty"] = now, False
        self.refresh_list()
        self.update_info()
        self.status.set(f"저장: {cur}  (이력: {hist.name})")

    def run_plan(self):
        if not (self.st["approved"] and not self.st["dirty"]) or self.proc is not None:
            return
        self.proc = subprocess.Popen([sys.executable, str(HERE / "path_planning.py"), "--photo", self.photo["name"]],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.update_info()
        self.status.set("경로계획 실행 중...")
        self.root.after(300, self._poll_plan)

    def _poll_plan(self):
        if self.proc.poll() is None:
            self.root.after(300, self._poll_plan)
            return
        out, code = self.proc.communicate()[0], self.proc.returncode
        self.proc = None
        self.update_info()
        if code != 0:
            messagebox.showerror("경로계획 실패", out[-3000:], parent=self.root)
            return
        res = json.loads(out[out.index("{"):])
        d = Path(res["photo_dir"])
        self.res_photo.set_image(d / "path_planning_photo.png")
        self.res_map.set_image(d / "path_planning_map.png")
        self.tabs.select(1)
        self.status.set(f"경로계획 완료: {res['run_dir']}")
        messagebox.showinfo("경로계획 완료", json.dumps({k: res[k] for k in ("run_dir", "A", "B", "grasp_yaw", "place_yaw")},
                                                       ensure_ascii=False, indent=1, default=str), parent=self.root)


def main():
    root = tk.Tk()
    Picker(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
