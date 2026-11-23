#!/usr/bin/env python3
"""Instruction target viewer: choose a photo, type an instruction, run -> source A and destination B drawn on the photo.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/instruction_target_viewer/viewer_app.py                       # GUI
  $PY experiments/instruction_target_viewer/viewer_app.py --headless --photo U25070552 \
      --instruction "컨베이어 옆의 빨간 큐브를 하역장으로 옮겨줘." --instruction "..."   # same session, no window

GUI widgets reused from v0.2 (imported, unchanged): ImageView, FunctionWorker, GpuMonitor, STATUS_COLORS.
"""
import argparse
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import viewer_pipeline as V  # noqa: E402  (also puts the v0.2 src on sys.path)


def headless(args):
    v = V.Viewer()
    photos = {p["photo_id"]: p for p in v.list_photos()}
    path = photos[args.photo]["path"] if args.photo in photos else args.photo
    try:
        for ins in args.instruction:
            rec = v.run(path, ins, args.rules)
            print(f"\n=== {rec['run_id']}  ({rec['wall_s']} s, Qwen {rec['model_calls']['n_qwen']}, DINO {rec['model_calls']['n_dino']})")
            for k, val in V.summary(rec).items():
                print(f"  {k}: {val}")
            for s in rec["stages"]:
                print(f"    [{s['status']:>11}] {s['name']:<22} {s['summary'][:170]}")
    finally:
        v.close()
    print("\nsession:", v.session)


def gui(args):
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
                                   QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
                                   QVBoxLayout, QWidget)
    from semantic_task_nav.gui.main_window import STATUS_COLORS, ImageView
    from semantic_task_nav.gui.workers import FunctionWorker, GpuMonitor
    colors = {**STATUS_COLORS, "REUSED": "#2e7d32", "NOT_CHECKED": "#ef6c00"}

    class Window(QMainWindow):
        progress_sig = Signal(str)

        def __init__(self):
            super().__init__()
            self.setWindowTitle("작업지시 대상 시각화 — instruction_target_viewer")
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self.v = V.Viewer(progress=self.progress_sig.emit)       # emitted from the worker thread, delivered on the GUI thread
            self.worker, self.images, self.shown = None, {}, []
            self._build()
            self.progress_sig.connect(self.lbl_status.setText)
            self.gpu = GpuMonitor(2.0)
            self.gpu.reading.connect(self.lbl_gpu.setText)
            self.gpu.start()
            for p in self.v.list_photos():
                it = QListWidgetItem(f"{p['photo_id']}   {p['file']}")
                it.setData(Qt.ItemDataRole.UserRole, p["path"])
                self.lst.addItem(it)
            self.lbl_session.setText(f"저장 폴더: {self.v.session}")

        # -------------------------------------------------------------- layout
        def _build(self):
            left = QWidget()
            lv = QVBoxLayout(left)
            lv.addWidget(QLabel("1. 사진 선택 (/home/binci/Pictures/Calibration)"))
            self.lst = QListWidget()
            self.lst.currentItemChanged.connect(self._photo_changed)
            lv.addWidget(self.lst, 2)
            lv.addWidget(QLabel("2. 작업지시 입력"))
            self.txt = QPlainTextEdit()
            self.txt.setPlaceholderText("예: 컨베이어 옆의 빨간 큐브를 하역장으로 옮겨줘.")
            self.txt.setMaximumHeight(110)
            lv.addWidget(self.txt)
            self.cmb_rules = QComboBox()
            self.cmb_rules.addItems(["R1 · 작업장 규칙 적용", "R0 · 규칙 없이"])
            lv.addWidget(self.cmb_rules)
            hb = QHBoxLayout()
            self.btn_run = QPushButton("3. 실행")
            self.btn_run.clicked.connect(self._run)
            self.btn_stop = QPushButton("중지")
            self.btn_stop.setEnabled(False)
            self.btn_stop.clicked.connect(self._stop)
            hb.addWidget(self.btn_run, 2)
            hb.addWidget(self.btn_stop, 1)
            lv.addLayout(hb)
            self.lbl_status = QLabel("대기 중")
            self.lbl_status.setWordWrap(True)
            lv.addWidget(self.lbl_status)
            self.lbl_gpu = QLabel("GPU")
            lv.addWidget(self.lbl_gpu)
            self.lbl_session = QLabel()
            self.lbl_session.setWordWrap(True)
            self.lbl_session.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            lv.addWidget(self.lbl_session)

            center = QWidget()
            cv = QVBoxLayout(center)
            top = QHBoxLayout()
            self.chk_cand = QCheckBox("판단한 후보 모두 표시")
            self.chk_cand.toggled.connect(self._render)
            self.lbl_view = QLabel("실행 기록에서 두 줄을 선택하면(Ctrl+클릭) 나란히 비교합니다.")
            top.addWidget(self.chk_cand)
            top.addWidget(self.lbl_view, 1)
            cv.addLayout(top)
            self.split_views = QSplitter(Qt.Orientation.Vertical)
            self.view_main, self.view_cmp = ImageView("사진을 선택하세요"), ImageView("")
            self.split_views.addWidget(self.view_main)
            self.split_views.addWidget(self.view_cmp)
            self.view_cmp.hide()
            cv.addWidget(self.split_views, 1)

            right = QSplitter(Qt.Orientation.Vertical)
            self.tabs = QTabWidget()
            self.txt_result = QTextBrowser()
            self.tabs.addTab(self.txt_result, "판단 결과")
            self.tbl_stage = self._table(["단계", "상태", "내용"])
            self.tabs.addTab(self.tbl_stage, "단계별 상태")
            cand = QWidget()
            cl = QVBoxLayout(cand)
            self.tbl_cand = self._table(["쪽", "번호", "ID", "Qwen이 본 것", "역할", "적재 상태", "접근", "주행 구역까지 cm", "Qwen 이유"])
            cl.addWidget(self.tbl_cand, 2)
            self.cmb_input = QComboBox()
            self.cmb_input.currentIndexChanged.connect(self._show_input)
            cl.addWidget(QLabel("VLM에 실제로 입력된 번호 이미지"))
            cl.addWidget(self.cmb_input)
            self.view_input = ImageView("")
            cl.addWidget(self.view_input, 2)
            self.tabs.addTab(cand, "후보 판단")
            right.addWidget(self.tabs)
            hist = QWidget()
            hl = QVBoxLayout(hist)
            hl.addWidget(QLabel("실행 기록 (이 세션)"))
            self.tbl_hist = self._table(["#", "사진", "작업지시", "규칙", "A 출발 대상", "B 목적 대상", "최초 실패 단계", "초"])
            self.tbl_hist.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.tbl_hist.itemSelectionChanged.connect(self._hist_selected)
            hl.addWidget(self.tbl_hist)
            right.addWidget(hist)
            right.setSizes([620, 300])

            main = QSplitter(Qt.Orientation.Horizontal)
            main.addWidget(left)
            main.addWidget(center)
            main.addWidget(right)
            main.setSizes([330, 900, 560])
            self.setCentralWidget(main)

        @staticmethod
        def _table(cols):
            t = QTableWidget(0, len(cols))
            t.setHorizontalHeaderLabels(cols)
            t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            t.horizontalHeader().setStretchLastSection(True)
            t.verticalHeader().setVisible(False)
            return t

        # -------------------------------------------------------------- actions
        def _img(self, path):
            if path not in self.images:
                self.images[path] = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            return self.images[path]

        def _photo_changed(self, cur, _prev):
            if cur is None:
                return
            self.tbl_hist.clearSelection()
            self.shown = []
            self.view_cmp.hide()
            self.view_main.set_image(self._img(cur.data(Qt.ItemDataRole.UserRole)))
            self.txt_result.setHtml(f"<p>사진: <b>{cur.text()}</b></p><p>작업지시를 입력하고 실행하세요. 같은 사진에서 여러 지시를 연속으로 실행할 수 있습니다.</p>")

        def _run(self):
            cur, text = self.lst.currentItem(), self.txt.toPlainText().strip()
            if cur is None or not text:
                QMessageBox.information(self, "입력 필요", "사진을 고르고 작업지시를 입력하세요.")
                return
            path, rules = cur.data(Qt.ItemDataRole.UserRole), self.cmb_rules.currentText()[:2]
            self.btn_run.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.lbl_status.setText("실행 시작")
            self.worker = FunctionWorker(lambda: self.v.run(path, text, rules))
            self.worker.done.connect(self._done)
            self.worker.failed.connect(self._failed)
            self.worker.start()

        def _stop(self):
            self.v.stop_models()
            self.lbl_status.setText("중지 요청: 진행 중인 모델 호출을 끊습니다 (기록은 실패 단계와 함께 저장)")

        def _finish(self):
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)

        def _done(self, rec):
            self._finish()
            r = self.tbl_hist.rowCount()
            self.tbl_hist.insertRow(r)
            s, d = rec["source"], rec["destination"]
            vals = [r + 1, rec["photo"]["photo_id"], rec["instruction"], rec["rules"], f"{s.get('status')} {' '.join(s.get('ids') or [])}",
                    f"{d.get('status')} {d.get('id') or ''} {d.get('goal_kind') or ''}", rec["first_failure"] or "없음", rec["wall_s"]]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(str(val))
                it.setData(Qt.ItemDataRole.UserRole, len(self.v.runs) - 1)
                self.tbl_hist.setItem(r, c, it)
            self.tbl_hist.resizeColumnsToContents()
            self.tbl_hist.clearSelection()
            self.tbl_hist.selectRow(r)
            self.lbl_status.setText(f"완료: {rec['run_id']} ({rec['wall_s']} s) · 저장 {Path(rec['files']['run_json']).parent}")

        def _failed(self, msg):
            self._finish()
            self.lbl_status.setText("실행 오류 (아래 창 참고)")
            QMessageBox.critical(self, "실행 오류", msg[-3000:])

        # -------------------------------------------------------------- display
        def _hist_selected(self):
            rows = sorted({i.row() for i in self.tbl_hist.selectedItems()})
            self.shown = [self.v.runs[self.tbl_hist.item(r, 0).data(Qt.ItemDataRole.UserRole)] for r in rows[-2:]]
            self._render()

        def _render(self):
            if not self.shown:
                return
            a = self.shown[0]
            self.view_main.set_image(V.draw(self._img(a["photo"]["path"]), a, self.chk_cand.isChecked()))
            if len(self.shown) == 2:
                b = self.shown[1]
                self.view_cmp.show()
                self.view_cmp.set_image(V.draw(self._img(b["photo"]["path"]), b, self.chk_cand.isChecked()))
                self.lbl_view.setText(f"위: {a['run_id']}   아래: {b['run_id']}")
            else:
                self.view_cmp.hide()
                self.lbl_view.setText(f"{a['run_id']} · 빨강 = A 출발 대상, 파랑 = B 목적 대상, 점선 = 탐색 영역 / 동률 / 접근 불가 후보")
            self._result_text()
            self._stages(a)
            self._cands(a)

        def _result_text(self):
            sums = [V.summary(r) for r in self.shown]
            head = "".join(f"<th>{r['run_id']}</th>" for r in self.shown)
            rows = "".join(f"<tr><td><b>{k}</b></td>" + "".join(f"<td>{s.get(k, '')}</td>" for s in sums) + "</tr>" for k in sums[0])
            note = ("<p style='color:#777'>B가 GOAL_REGION이면 물체가 아니라 구역입니다. PLACEMENT_UNVERIFIED = 배치면 미확인, "
                    "VLM_SUITABLE_EMPTY = VLM이 '적합·비어 있음'으로 봄(물리 검증 아님). 접근성은 임시 바닥 변환과 DRAFT 지도 기준입니다.</p>")
            files = "".join(f"<p style='color:#777'>저장: {Path(r['files']['run_json']).parent}</p>" for r in self.shown)
            self.txt_result.setHtml(f"<table border='1' cellspacing='0' cellpadding='4'><tr><th></th>{head}</tr>{rows}</table>{note}{files}")

        def _stages(self, rec):
            t = self.tbl_stage
            t.setRowCount(0)
            for s in rec["stages"]:
                r = t.rowCount()
                t.insertRow(r)
                for c, val in enumerate((s["name"], s["status"], s["summary"])):
                    it = QTableWidgetItem(str(val))
                    if c == 1:
                        it.setForeground(QColor(colors.get(s["status"], "#000000")))
                    t.setItem(r, c, it)
            t.resizeColumnsToContents()

        def _cands(self, rec):
            t = self.tbl_cand
            t.setRowCount(0)
            for side, x in (("A", rec["source"]), ("B", rec["destination"])):
                acc = x.get("accessibility") or {}
                for k, c in enumerate(x.get("candidates") or [], 1):
                    rt = (x.get("ratings") or {}).get(c["id"]) or {}
                    a = acc.get(c["id"]) or {}
                    vals = [side, k, c["id"], ", ".join(map(str, rt.get("appearance") or [])), rt.get("role") or rt.get("placement_role") or "",
                            rt.get("load_state") or "", {True: "가능", False: "불가", None: "미확인"}.get(a.get("accessible"), "") if a else "",
                            a.get("gap_to_drivable_cm", ""), rt.get("reason") or ""]
                    r = t.rowCount()
                    t.insertRow(r)
                    for col, val in enumerate(vals):
                        t.setItem(r, col, QTableWidgetItem(str(val)))
            t.resizeColumnsToContents()
            self.cmb_input.blockSignals(True)
            self.cmb_input.clear()
            for side, x in (("A 출발", rec["source"]), ("B 목적지", rec["destination"])):
                for im in x.get("images") or []:
                    self.cmb_input.addItem(f"{side} · 번호 {im['numbers'][0]}–{im['numbers'][1]}", im["path"])
            self.cmb_input.blockSignals(False)
            self._show_input()

        def _show_input(self):
            p = self.cmb_input.currentData()
            self.view_input.set_image(cv2.imread(p) if p else None)

        def closeEvent(self, e):
            if self.worker is not None and self.worker.isRunning():
                self.v.stop_models()
                self.worker.wait(15000)
            self.gpu.stop()
            self.v.close()
            super().closeEvent(e)

    app = QApplication.instance() or QApplication(sys.argv)
    win = Window()
    if args.photo:
        for i in range(win.lst.count()):
            if win.lst.item(i).text().startswith(args.photo):
                win.lst.setCurrentRow(i)
    win.resize(1800, 1000)
    win.show()
    return app, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--photo", help="photo id (e.g. U25070552) or path")
    ap.add_argument("--instruction", action="append", default=[])
    ap.add_argument("--rules", choices=["R0", "R1"], default=None)
    args = ap.parse_args()
    if args.headless:
        assert args.photo and args.instruction, "--headless needs --photo and at least one --instruction"
        return headless(args)
    app, _win = gui(args)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
