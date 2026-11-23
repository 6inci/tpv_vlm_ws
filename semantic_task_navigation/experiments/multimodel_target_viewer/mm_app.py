#!/usr/bin/env python3
"""Multi-model target viewer: photo + natural-language instruction -> Grounding DINO, SigLIP2, VLM, Python -> A / B on the photo.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/multimodel_target_viewer/mm_app.py                         # GUI
  $PY experiments/multimodel_target_viewer/mm_app.py --headless --photo U25070552 --instruction "..." [--instruction "..."] [--vlm qwen3vl_4b_bf16]

Same window layout as instruction_target_viewer (that program is not modified); v0.2 widgets reused: ImageView, FunctionWorker, GpuMonitor.
"""
import argparse
import html
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mm_pipeline as M  # noqa: E402


def summary(rec):
    s, d = rec["source"], rec["destination"]
    it = (rec.get("interpretation") or {}).get("parsed") or {}
    a = f"{s.get('status')}"
    if s.get("ids"):
        a += " · " + ", ".join(f"{i} (VLM: {M._nm(s, i)}, SigLIP2: {M._sg(s, i)}, 박스 {[round(v) for v in b]})" for i, b in zip(s["ids"], s["boxes"]))
    b = f"{d.get('status')}"
    if d.get("id"):
        b += f" · {d['id']} (VLM: {M._nm(d, d['id'])}, 적재 상태: {d.get('load_state')}, SigLIP2: {M._sg(d, d['id'])}, 박스 {[round(v) for v in d['bbox_xyxy']]}) · {d.get('goal_kind')} · {d.get('placement')}"
    elif d.get("tied_ids"):
        b += f" · 동률 {d['tied_ids']}"
    mu = rec.get("models_used") or {}
    used = "; ".join(f"{k}: {', '.join(f'{m} ×{n}' for m, n in v.items())}" for k, v in (mu.get("this_run") or {}).items()) or "없음"
    prep = "; ".join(f"{k}: {', '.join(f'{m} ×{n}' for m, n in v.items())}" for k, v in (mu.get("photo_preparation") or {}).items()) or "없음"
    return {"입력": rec["instruction"], "사진 / VLM / 규칙": f"{rec['photo']['photo_id']} / {rec['vlm_backend']['label']} / {rec['rules']}",
            "해석": (f"옮길 물체 {it.get('source_object', {}).get('name_en')} · 출발 기준 {rec['references']['source']['name']} [{it.get('source_relation')}] · "
                   f"목적지 의미 {it.get('destination_meaning')} · 목적지 기준 {rec['references']['destination']['name']} [{it.get('destination_relation')}]") if it else
                  f"실패: {(rec.get('interpretation') or {}).get('error')}",
            "A 운반 대상": a, "A 근거": f"{s.get('why', '')}" + (f" / VLM: {s.get('reply_reason')}" if s.get("reply_reason") else ""),
            "B 목적지": b, "B 근거": f"{d.get('why', '')}" + (f" / VLM: {d.get('reply_reason')}" if d.get("reply_reason") else ""),
            "판단 상태": f"A {s.get('status')} · B {d.get('status')} · 최초 실패 단계: {rec.get('first_failure') or '없음'}",
            "이 실행에서 호출된 모델": used, "사진 준비 때 호출된 모델": prep}


def headless(args):
    v = M.MMViewer(backend=args.vlm)
    photos = {p["photo_id"]: p for p in v.list_photos()}
    path = photos[args.photo]["path"] if args.photo in photos else args.photo
    try:
        for ins in args.instruction:
            rec = v.run(path, ins, args.rules)
            print(f"\n=== {rec['run_id']}  ({rec['wall_s']} s)")
            for line in rec["narrative"]:
                print("  ·", line)
            for k, val in summary(rec).items():
                print(f"  {k}: {val}")
            for s in rec["stages"]:
                print(f"    [{s['status']:>9}] {s['name']:<24} {s['model'][:42]:<42} {s['summary'][:150]}")
            print("  overlay:", rec["files"]["overlay"])
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
    colors = {**STATUS_COLORS, "REUSED": "#2e7d32"}

    class Window(QMainWindow):
        progress_sig = Signal(str)

        def __init__(self):
            super().__init__()
            self.setWindowTitle("다중 모델 작업 대상 추론 — multimodel_target_viewer (DINO · SigLIP2 · VLM · Python)")
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self.v = M.MMViewer(progress=self.progress_sig.emit, backend=args.vlm)
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

        def _build(self):
            left = QWidget()
            lv = QVBoxLayout(left)
            lv.addWidget(QLabel("1. 사진 선택 (/home/binci/Pictures/Calibration)"))
            self.lst = QListWidget()
            self.lst.currentItemChanged.connect(self._photo_changed)
            lv.addWidget(self.lst, 2)
            lv.addWidget(QLabel("2. 작업지시 입력"))
            self.txt = QPlainTextEdit()
            self.txt.setPlaceholderText("예: 컨베이어 주변에 있는 빨간 큐브를 트럭 근처의 비어 있는 적재 장소로 옮겨줘.")
            self.txt.setMaximumHeight(110)
            lv.addWidget(self.txt)
            lv.addWidget(QLabel("VLM (자연어 해석·작업 맥락)"))
            self.cmb_vlm = QComboBox()
            for k, b in self.v.mc["vlm_backends"].items():
                self.cmb_vlm.addItem(b["label"], k)
            self.cmb_vlm.setCurrentIndex(max(0, self.cmb_vlm.findData(self.v.backend)))
            lv.addWidget(self.cmb_vlm)
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
            self.lbl_status = QLabel("대기 중 (사진의 첫 실행은 탐지·SigLIP2·모델 로딩 때문에 몇 분 걸릴 수 있습니다)")
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
            self.lbl_view = QLabel("실행 기록에서 두 줄을 선택하면(Ctrl+클릭) 위아래로 비교합니다.")
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
            self.tabs.addTab(self.txt_result, "작업 설명·결과")
            self.tbl_stage = self._table(["단계", "상태", "사용 모델", "내용"])
            self.tabs.addTab(self.tbl_stage, "모델별 단계")
            cand = QWidget()
            cl = QVBoxLayout(cand)
            self.tbl_cand = self._table(["쪽", "번호", "ID", "SigLIP2 상위 3 (상대 점수)", "VLM이 본 것", "VLM 역할", "적재 상태", "접근", "배치 금지 구역 비율", "VLM 이유"])
            cl.addWidget(self.tbl_cand, 2)
            self.cmb_input = QComboBox()
            self.cmb_input.currentIndexChanged.connect(self._show_input)
            cl.addWidget(QLabel("모델에 실제로 입력된 이미지 (VLM 번호 이미지 / 선택 후보의 SigLIP2 crop)"))
            cl.addWidget(self.cmb_input)
            self.view_input = ImageView("")
            cl.addWidget(self.view_input, 2)
            self.tabs.addTab(cand, "후보 판단")
            right.addWidget(self.tabs)
            hist = QWidget()
            hl = QVBoxLayout(hist)
            hl.addWidget(QLabel("실행 기록 (이 세션)"))
            self.tbl_hist = self._table(["#", "사진", "작업지시", "VLM", "A 운반 대상", "B 목적지", "최초 실패 단계", "초"])
            self.tbl_hist.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.tbl_hist.itemSelectionChanged.connect(self._hist_selected)
            hl.addWidget(self.tbl_hist)
            right.addWidget(hist)
            right.setSizes([640, 280])

            main = QSplitter(Qt.Orientation.Horizontal)
            main.addWidget(left)
            main.addWidget(center)
            main.addWidget(right)
            main.setSizes([330, 880, 600])
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
            self.txt_result.setHtml(f"<p>사진: <b>{html.escape(cur.text())}</b></p><p>작업지시를 입력하고 실행하세요. 같은 사진에서 여러 지시를 연속으로 실행할 수 있습니다.</p>")

        def _run(self):
            cur, text = self.lst.currentItem(), self.txt.toPlainText().strip()
            if cur is None or not text:
                QMessageBox.information(self, "입력 필요", "사진을 고르고 작업지시를 입력하세요.")
                return
            path, rules, backend = cur.data(Qt.ItemDataRole.UserRole), self.cmb_rules.currentText()[:2], self.cmb_vlm.currentData()
            self.btn_run.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.lbl_status.setText("실행 시작")
            self.worker = FunctionWorker(lambda: self.v.run(path, text, rules, backend))
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
            vals = [r + 1, rec["photo"]["photo_id"], rec["instruction"], rec["vlm_backend"]["key"], f"{s.get('status')} {' '.join(s.get('ids') or [])}",
                    f"{d.get('status')} {d.get('id') or ''}", rec["first_failure"] or "없음", rec["wall_s"]]
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

        def _hist_selected(self):
            rows = sorted({i.row() for i in self.tbl_hist.selectedItems()})
            self.shown = [self.v.runs[self.tbl_hist.item(r, 0).data(Qt.ItemDataRole.UserRole)] for r in rows[-2:]]
            self._render()

        def _render(self):
            if not self.shown:
                return
            a = self.shown[0]
            self.view_main.set_image(M.draw(self._img(a["photo"]["path"]), a, self.chk_cand.isChecked()))
            if len(self.shown) == 2:
                b = self.shown[1]
                self.view_cmp.show()
                self.view_cmp.set_image(M.draw(self._img(b["photo"]["path"]), b, self.chk_cand.isChecked()))
                self.lbl_view.setText(f"위: {a['run_id']}   아래: {b['run_id']}")
            else:
                self.view_cmp.hide()
                self.lbl_view.setText(f"{a['run_id']} · 빨강 = A 운반 대상, 파랑 = B 목적지, 초록 화살표 = 작업 방향(주행 경로 아님), 점선 = 탐색 영역 / 미확정 후보")
            self._result_text()
            self._stages(a)
            self._cands(a)

        def _result_text(self):
            parts = []
            for r in self.shown:
                parts.append(f"<h3>{html.escape(r['run_id'])}</h3><ol>" + "".join(f"<li>{html.escape(x)}</li>" for x in r["narrative"]) + "</ol>")
            sums = [summary(r) for r in self.shown]
            head = "".join(f"<th>{html.escape(r['run_id'])}</th>" for r in self.shown)
            rows = "".join(f"<tr><td><b>{k}</b></td>" + "".join(f"<td>{html.escape(str(s.get(k, '')))}</td>" for s in sums) + "</tr>" for k in sums[0])
            note = ("<p style='color:#777'>SigLIP2 점수는 설정된 라벨 목록 안에서의 상대 점수이며 분류 확률이 아닙니다. GOAL_REGION = 물체가 아닌 구역, "
                    "PLACEMENT_UNVERIFIED = 배치면 미확인, VLM_SUITABLE_EMPTY = VLM이 '적합·비어 있음'으로 봄(물리 검증 아님). 접근성은 임시 바닥 변환과 DRAFT 지도 기준입니다.</p>")
            files = "".join(f"<p style='color:#777'>저장: {html.escape(str(Path(r['files']['run_json']).parent))}</p>" for r in self.shown)
            self.txt_result.setHtml("".join(parts) + f"<table border='1' cellspacing='0' cellpadding='4'><tr><th></th>{head}</tr>{rows}</table>{note}{files}")

        def _stages(self, rec):
            t = self.tbl_stage
            t.setRowCount(0)
            for s in rec["stages"]:
                r = t.rowCount()
                t.insertRow(r)
                for c, val in enumerate((s["name"], s["status"], s.get("model", ""), s["summary"])):
                    it = QTableWidgetItem(str(val))
                    if c == 1:
                        it.setForeground(QColor(colors.get(s["status"], "#000000")))
                    t.setItem(r, c, it)
            t.resizeColumnsToContents()

        def _cands(self, rec):
            t = self.tbl_cand
            t.setRowCount(0)
            for side, x in (("A", rec["source"]), ("B", rec["destination"])):
                acc, pol = x.get("accessibility") or {}, x.get("policy_no_placement_frac") or {}
                for k, c in enumerate(x.get("candidates") or [], 1):
                    rt = (x.get("ratings") or {}).get(c["id"]) or {}
                    a = acc.get(c["id"]) or {}
                    vals = [side, k, c["id"], " · ".join(f"{lb} {s:.2f}" for lb, s in c.get("siglip_top3") or []), ", ".join(map(str, rt.get("appearance") or [])),
                            rt.get("role") or rt.get("placement_role") or "", rt.get("load_state") or "",
                            {True: "가능", False: "불가", None: "미확인"}.get(a.get("accessible"), "") if a else "", pol.get(c["id"], ""), rt.get("reason") or ""]
                    r = t.rowCount()
                    t.insertRow(r)
                    for col, val in enumerate(vals):
                        t.setItem(r, col, QTableWidgetItem(str(val)))
            t.resizeColumnsToContents()
            self.cmb_input.blockSignals(True)
            self.cmb_input.clear()
            for side, x in (("A", rec["source"]), ("B", rec["destination"])):
                for im in x.get("images") or []:
                    self.cmb_input.addItem(f"{side} VLM 번호 이미지 {im['numbers'][0]}–{im['numbers'][1]}" + ("" if im.get("usable", True) else " (응답 거부됨)"), im["path"])
            for side, x, ids in (("A", rec["source"], rec["source"].get("ids") or []), ("B", rec["destination"], [rec["destination"]["id"]] if rec["destination"].get("id") else [])):
                for i in ids:
                    sg = self.v.emb.get((rec["photo"]["photo_id"], i))
                    for view, c in ((sg or {}).get("crops") or {}).items():
                        self.cmb_input.addItem(f"{side} {i} SigLIP2 crop ({view})", c["path"])
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
    win.resize(1820, 1000)
    win.show()
    return app, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--photo", help="photo id (e.g. U25070552) or path")
    ap.add_argument("--instruction", action="append", default=[])
    ap.add_argument("--rules", choices=["R0", "R1"], default=None)
    ap.add_argument("--vlm", default=None, help="VLM backend key of config.yaml (default: the first one)")
    args = ap.parse_args()
    if args.headless:
        assert args.photo and args.instruction, "--headless needs --photo and at least one --instruction"
        return headless(args)
    app, _win = gui(args)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
