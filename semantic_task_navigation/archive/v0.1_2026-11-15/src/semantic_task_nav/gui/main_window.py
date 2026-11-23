"""PySide6 main window: camera view, command input, agent analysis, detections, result layers,
map editor, simulated hand-off, status bar, saving / paper figures / replay."""
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import visualization
from ..agent.orchestrator import LABELS, PLANS, record_handoff_event, replay_state
from ..artifacts import RunWriter
from ..config import ANALYSIS_MODES, resolve_path, sha256_file
from ..navigation.map_builder import MAP_SCHEMA, load_map_spec
from ..contracts import SchemaError, validate
from ..perception.camera import CameraSource, Frame, ImageSource, SourceError, VideoSource, frame_shift, list_video_devices
from ..perception.model_client import ModelWorkerClient
from ..robot.handoff import HandoffError
from .workers import AnalysisWorker, CameraWorker, FunctionWorker, GpuMonitor

STATUS_COLORS = {"OK": "#2e7d32", "RUNNING": "#1565c0", "FAILED": "#c62828", "UNCERTAIN": "#ef6c00",
                 "SKIPPED": "#757575", "NOT_RUN": "#757575", "CANCELLED": "#6a1b9a", "PENDING": "#9e9e9e"}
EDIT_TYPES = [("traversable", "통행 가능 영역"), ("static_obstacles", "고정 장애물"), ("no_go", "금지 영역"),
              ("start_pose", "시작점")]


def bgr_to_qimage(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ImageView(QWidget):
    """Shows an image with a uniform scale (aspect ratio kept, no distortion); clicks -> image px."""
    clicked = Signal(float, float)

    def __init__(self, placeholder=""):
        super().__init__()
        self._qimg, self._shape, self.placeholder = None, None, placeholder
        self.setMinimumSize(320, 180)

    def set_image(self, bgr):
        """Displayed copy is downscaled to the widget size (uniform INTER_AREA); coordinate mapping
        always uses the original image size, so clicks are reported in original pixels."""
        self._shape = None if bgr is None else bgr.shape[:2]
        if bgr is not None:
            h, w = bgr.shape[:2]
            s = min(1.0, max(self.width() / w, self.height() / h) * self.devicePixelRatioF())
            if s < 0.999:
                bgr = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        self._qimg = None if bgr is None else bgr_to_qimage(bgr)
        self.update()

    def image_rect(self):
        if self._qimg is None:
            return None, 0.0
        h, w = self._shape
        s = min(self.width() / w, self.height() / h)
        dw, dh = int(round(w * s)), int(round(h * s))
        return QRect((self.width() - dw) // 2, (self.height() - dh) // 2, dw, dh), s

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(25, 25, 25))
        rect, _ = self.image_rect()
        if rect is None:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
        else:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(rect, self._qimg)
        p.end()

    def mousePressEvent(self, e):
        rect, s = self.image_rect()
        if rect is None or not rect.contains(e.position().toPoint()):
            return
        self.clicked.emit((e.position().x() - rect.x()) / s, (e.position().y() - rect.y()) / s)


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("SEMANTIC TASK NAVIGATION")
        # Qt deletes the window on close (before interpreter shutdown) -> avoids PySide6 exit-order crashes.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.client = None
        self.live_frame = None          # latest frame from the selected source
        self.analysis_frame = None      # frame the current result belongs to
        self.state = None               # render state of the current result (run or replay)
        self.run_dir = None
        self.cam_worker = self.analysis_worker = None
        self.aux_workers = []
        self.source_kind = None
        self._fps, self._last_t = 0.0, None
        self.map_source = cfg["app"]["map_file"]
        self.map_spec = load_map_spec(resolve_path(self.map_source))
        self.edit_points = []
        self.last_error = ""
        self._build_ui()
        self.gpu = GpuMonitor(cfg["app"]["gui"]["gpu_poll_interval_s"])
        self.gpu.reading.connect(self.lbl_gpu.setText)
        self.gpu.start()
        self.change_timer = QTimer(self)
        self.change_timer.timeout.connect(self._check_scene_change)
        self.change_timer.start(1000)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QSplitter(Qt.Orientation.Horizontal)
        left = QSplitter(Qt.Orientation.Vertical)
        cam_box = QGroupBox("CAMERA VIEW")
        cl = QVBoxLayout(cam_box)
        self.cam_view = ImageView("입력 소스를 선택하세요")
        self.lbl_cam = QLabel("-")
        cl.addWidget(self.cam_view, 1)
        cl.addWidget(self.lbl_cam)
        res_box = QGroupBox("RESULT VIEW")
        rl = QVBoxLayout(res_box)
        row = QHBoxLayout()
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(visualization.VIEWS)
        self.cmb_view.setCurrentText("FINAL_RESULT")
        self.cmb_view.currentTextChanged.connect(self._view_changed)
        row.addWidget(QLabel("화면"))
        row.addWidget(self.cmb_view)
        self.lbl_result = QLabel("분석 결과 없음")
        row.addWidget(self.lbl_result, 1)
        rl.addLayout(row)
        grid = QGridLayout()
        self.layer_checks = {}
        for i, name in enumerate(visualization.LAYERS):
            cb = QCheckBox(name)
            cb.toggled.connect(self._render_result)
            self.layer_checks[name] = cb
            grid.addWidget(cb, i // 8, i % 8)
        rl.addLayout(grid)
        erow = QHBoxLayout()
        self.btn_edit = QPushButton("지도 편집")
        self.btn_edit.setCheckable(True)
        self.btn_edit.toggled.connect(self._edit_toggled)
        self.cmb_edit = QComboBox()
        for key, label in EDIT_TYPES:
            self.cmb_edit.addItem(label, key)
        self.btn_close_poly = QPushButton("다각형 완료")
        self.btn_close_poly.clicked.connect(self._close_polygon)
        self.btn_undo_pt = QPushButton("점 취소")
        self.btn_undo_pt.clicked.connect(lambda: (self.edit_points.pop() if self.edit_points else None,
                                                  self._render_result()))
        self.btn_clear_type = QPushButton("선택 유형 삭제")
        self.btn_clear_type.clicked.connect(self._clear_type)
        self.btn_save_map = QPushButton("지도 저장")
        self.btn_save_map.clicked.connect(self._save_map)
        self.btn_load_map = QPushButton("지도 불러오기")
        self.btn_load_map.clicked.connect(self._load_map_dialog)
        for w in (self.btn_edit, self.cmb_edit, self.btn_close_poly, self.btn_undo_pt, self.btn_clear_type,
                  self.btn_save_map, self.btn_load_map):
            erow.addWidget(w)
        self.lbl_map = QLabel()
        erow.addWidget(self.lbl_map, 1)
        rl.addLayout(erow)
        self.res_view = ImageView("분석 결과가 여기에 표시됩니다")
        self.res_view.clicked.connect(self._result_clicked)
        rl.addWidget(self.res_view, 1)
        left.addWidget(cam_box)
        left.addWidget(res_box)
        left.setSizes([420, 620])

        right = QWidget()
        rv = QVBoxLayout(right)
        cmd_box = QGroupBox("USER COMMAND")
        cv = QVBoxLayout(cmd_box)
        r1 = QHBoxLayout()
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(ANALYSIS_MODES)
        self.cmb_mode.setCurrentText(self.cfg["app"]["default_analysis_mode"])
        r1.addWidget(QLabel("모드"))
        r1.addWidget(self.cmb_mode, 1)
        cv.addLayout(r1)
        r2 = QHBoxLayout()
        self.cmb_source = QComboBox()
        self.cmb_source.addItems(["IMAGE", "VIDEO", "LIVE CAMERA"])
        self.cmb_dev = QComboBox()
        self._refresh_devices()
        btn_open = QPushButton("열기")
        btn_open.clicked.connect(self._open_source)
        r2.addWidget(QLabel("입력"))
        r2.addWidget(self.cmb_source)
        r2.addWidget(self.cmb_dev, 1)
        r2.addWidget(btn_open)
        cv.addLayout(r2)
        self.txt_cmd = QPlainTextEdit(self.cfg["app"]["default_command"])
        self.txt_cmd.setFixedHeight(60)
        cv.addWidget(self.txt_cmd)
        r3 = QHBoxLayout()
        self.btn_start = QPushButton("분석 시작")
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_stop = QPushButton("중지")
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_stop.setEnabled(False)
        self.btn_warm = QPushButton("모델 준비")
        self.btn_warm.clicked.connect(self._warm_models)
        for b in (self.btn_start, self.btn_stop, self.btn_warm):
            r3.addWidget(b)
        cv.addLayout(r3)
        rv.addWidget(cmd_box)

        agent_box = QGroupBox("AGENT ANALYSIS (실제 단계 기록: 모델 출력 · 검출 · 이벤트)")
        al = QVBoxLayout(agent_box)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["단계", "상태", "모델/방법", "시간(s)"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setWordWrap(True)
        al.addWidget(self.tree)
        rv.addWidget(agent_box, 3)

        det_box = QGroupBox("DETECTION RESULTS")
        dl = QVBoxLayout(det_box)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "클래스", "score", "bbox (raw px)", "출처", "검증"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        dl.addWidget(self.table)
        rv.addWidget(det_box, 2)

        ho_box = QGroupBox("작업 인계 (시뮬레이션 — 실제 로봇 명령 비활성)")
        hl = QVBoxLayout(ho_box)
        self.lbl_handoff = QLabel("-")
        self.lbl_handoff.setWordWrap(True)
        hl.addWidget(self.lbl_handoff)
        hb = QHBoxLayout()
        self.handoff_buttons = {}
        for ev, label in (("pickup_ok", "픽업 성공 확인"), ("pickup_failed", "픽업 실패"),
                          ("placement_ok", "배치 완료 확인"), ("placement_failed", "배치 실패"),
                          ("arrival_ok", "도착 확인")):
            b = QPushButton(label)
            b.setEnabled(False)
            b.clicked.connect(lambda _=False, e=ev: self._handoff(e))
            self.handoff_buttons[ev] = b
            hb.addWidget(b)
        hl.addLayout(hb)
        rv.addWidget(ho_box)

        br = QHBoxLayout()
        for label, fn in (("실행 결과 저장", self._save_snapshot), ("논문용 이미지 저장", self._export_figures),
                          ("이전 실행 불러오기", self._replay_dialog)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            br.addWidget(b)
        rv.addLayout(br)

        root.addWidget(left)
        root.addWidget(right)
        root.setSizes([1100, 620])
        self.setCentralWidget(root)
        sb = self.statusBar()
        self.lbl_models = {k: QLabel(f"{n}: NOT_LOADED") for k, n in (("qwen", "Qwen3-VL"), ("dino", "Grounding DINO"))}
        self.lbl_agent = QLabel("Task Agent: READY")
        self.lbl_camstat = QLabel("Camera: -")
        self.lbl_mapmode = QLabel("Map: UNCALIBRATED (IMAGE_SPACE_DEMO)")
        self.lbl_planner = QLabel("Path Planner: READY")
        self.lbl_fps = QLabel("FPS: -")
        self.lbl_lat = QLabel("Latency: -")
        self.lbl_gpu = QLabel("GPU: -")
        self.lbl_err = QLabel("")
        self.lbl_err.setStyleSheet("color:#c62828")
        for w in (*self.lbl_models.values(), self.lbl_agent, self.lbl_camstat, self.lbl_mapmode, self.lbl_planner,
                  self.lbl_fps, self.lbl_lat, self.lbl_gpu, self.lbl_err):
            sb.addWidget(w)
        self._view_changed(self.cmb_view.currentText())
        self._update_map_label()
        self._set_tree_plan(PLANS["transport_and_place"])

    # ------------------------------------------------------------------ sources
    def _refresh_devices(self):
        self.cmb_dev.clear()
        for d in list_video_devices():
            self.cmb_dev.addItem(f"{d['path']} {d['name']}", d["index"])
        if self.cmb_dev.count() == 0:
            self.cmb_dev.addItem("(카메라 없음)", None)

    def _open_source(self):
        kind = self.cmb_source.currentText()
        if kind == "IMAGE":
            p, _ = QFileDialog.getOpenFileName(self, "이미지 선택", str(resolve_path("assets/reference")),
                                               "Images (*.jpg *.jpeg *.png *.bmp)")
            if p:
                self.open_image(p)
        elif kind == "VIDEO":
            p, _ = QFileDialog.getOpenFileName(self, "동영상 선택", str(resolve_path(".")),
                                               "Video (*.mp4 *.avi *.mkv *.mov)")
            if p:
                self.open_video(p)
        else:
            self.open_live(self.cmb_dev.currentData())

    def _stop_camera(self):
        if self.cam_worker:
            self.cam_worker.stop()
            self.cam_worker = None

    def open_image(self, path):
        self._stop_camera()
        try:
            src = ImageSource(path)
        except SourceError as e:
            return self._error(str(e))
        self.source_kind = "image"
        self._on_frame(src.read())
        self.lbl_camstat.setText("Camera: IMAGE (static)")
        self.lbl_cam.setText(f"IMAGE {path} | {src.info['width']}x{src.info['height']} | camera model (EXIF): "
                             f"{src.info['camera_model']}")

    def open_video(self, path):
        self._start_stream("video", lambda: VideoSource(path))

    def open_live(self, device=None):
        self._start_stream("live", lambda: CameraSource(device, self.cfg["camera"]["live"]))

    def _start_stream(self, kind, factory):
        self._stop_camera()
        self.source_kind = kind
        self._fps, self._last_t = 0.0, None
        self.cam_worker = CameraWorker(factory, self.cfg["app"]["gui"]["preview_max_fps"])
        self.cam_worker.frame_ready.connect(self._on_frame)
        self.cam_worker.status.connect(self._cam_status)
        self.cam_worker.start()

    def _cam_status(self, state, msg):
        self.lbl_camstat.setText(f"Camera: {state}")
        if state in ("DISCONNECTED",):
            self._error(f"카메라 연결 실패/중단: {msg} — IMAGE 모드는 계속 사용할 수 있습니다")
        else:
            self.lbl_cam.setText(f"{self.source_kind.upper()} {msg}")

    def _on_frame(self, frame):
        self.live_frame = frame
        now = time.monotonic()
        if self._last_t is not None:
            inst = 1.0 / max(1e-6, now - self._last_t)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
            self.lbl_fps.setText(f"FPS: {self._fps:.1f}")
        self._last_t = now
        self.cam_view.set_image(frame.image)
        if self.btn_edit.isChecked() and self.state is None:
            self._render_result()
        if self.cam_worker is not None:
            self.cam_worker.consumed.set()

    def _check_scene_change(self):
        if self.source_kind not in ("live", "video") or self.analysis_frame is None or self.live_frame is None:
            return
        s = frame_shift(self.analysis_frame.image, self.live_frame.image)
        g = self.cfg["app"]["gui"]
        if not s["comparable"]:
            msg, bad = "분석 프레임과 현재 프레임 크기가 다름", True
        else:
            dx, dy = s["shift_px"]
            bad = max(abs(dx), abs(dy)) > g["scene_change_shift_px"] or s["mean_abs_diff_ratio"] > g[
                "scene_change_diff_ratio"]
            msg = f"분석 프레임 대비 이동 ({dx:.1f},{dy:.1f})px, 차이 {s['mean_abs_diff_ratio']:.2f}"
        self.lbl_cam.setText(("⚠ 장면 변화 — 결과 재사용 주의 | " if bad else "") + msg)
        self.lbl_cam.setStyleSheet("color:#c62828" if bad else "")

    # ------------------------------------------------------------------ models
    def _get_client(self):
        if self.client is None:
            self.client = ModelWorkerClient(self.cfg["models"], resolve_path(self.cfg["app"]["worker_log_dir"]))
        if not self.client.alive:
            self.client.start()
        return self.client

    def _warm_models(self):
        def work():
            c = self._get_client()
            return {"env": c.env_info, "qwen": c.ensure_loaded("qwen")}
        self._set_model("qwen", "LOADING")
        self._run_aux(work, lambda r: (self._set_model("qwen", "READY"),
                                       self.lbl_err.setText(""),
                                       self.lbl_agent.setToolTip(json.dumps(r, indent=1))))

    def _set_model(self, key, status):
        name = {"qwen": "Qwen3-VL", "dino": "Grounding DINO"}[key]
        self.lbl_models[key].setText(f"{name}: {status}")
        col = {"READY": "#2e7d32", "RUNNING": "#1565c0", "LOADING": "#1565c0", "ERROR": "#c62828"}.get(
            status.split()[0], "#555")
        self.lbl_models[key].setStyleSheet(f"color:{col}")

    def _run_aux(self, fn, on_done):
        w = FunctionWorker(fn)
        w.done.connect(on_done)
        w.failed.connect(self._error)
        w.finished.connect(lambda: self.aux_workers.remove(w) if w in self.aux_workers else None)
        self.aux_workers.append(w)
        w.start()
        return w

    # ------------------------------------------------------------------ analysis
    def start_analysis(self):
        if self.analysis_worker and self.analysis_worker.isRunning():
            QMessageBox.information(self, "분석 중", "이미 분석이 실행 중입니다. 중지하거나 완료된 뒤 다시 시작하세요.")
            return
        if self.live_frame is None:
            return self._error("입력 프레임이 없습니다")
        cmd = self.txt_cmd.toPlainText().strip()
        if not cmd:
            return self._error("명령을 입력하세요")
        f = self.live_frame
        self.analysis_frame = Frame(f.image.copy(), f.frame_id, f.timestamp, dict(f.source))
        mode = self.cmb_mode.currentText()
        try:
            spec = validate(copy.deepcopy(self.map_spec), MAP_SCHEMA)
        except SchemaError as e:
            return self._error(f"지도 설정이 유효하지 않습니다: {e}")
        self._set_tree_plan(PLANS["transport_and_place"])
        self.table.setRowCount(0)
        self.lbl_err.setText("")
        self.lbl_agent.setText("Task Agent: RUNNING")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.analysis_worker = AnalysisWorker(self.cfg, mode, self.analysis_frame, cmd, self._get_client, spec,
                                              self.map_source)
        self.analysis_worker.event.connect(self._on_event)
        self.analysis_worker.done.connect(self._on_done)
        self.analysis_worker.failed.connect(self._on_failed)
        self.analysis_worker.finished.connect(self._worker_finished)
        self.analysis_worker.start()

    def stop_analysis(self):
        w = self.analysis_worker
        if not (w and w.isRunning()):
            return
        w.cancel.set()
        if self.client and self.client.busy:
            self.client.kill()   # aborts the running inference; models reload on next use
            for k in self.lbl_models:
                self._set_model(k, "NOT_LOADED (stopped)")
        self.lbl_agent.setText("Task Agent: STOPPING")

    def _worker_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _set_tree_plan(self, names):
        self.tree.clear()
        self.items = {}
        for i, n in enumerate(names, 1):
            it = QTreeWidgetItem([f"[{i}] {LABELS[n]} ({n})", "PENDING", "", ""])
            self._color(it, "PENDING")
            self.tree.addTopLevelItem(it)
            self.items[n] = it

    def _color(self, it, status):
        it.setForeground(1, QBrush(QColor(STATUS_COLORS.get(status, "#000"))))

    def _update_step(self, rec):
        it = self.items.get(rec["name"])
        if it is None:
            return
        it.setText(1, rec["status"])
        it.setText(2, rec.get("model") or "")
        it.setText(3, "" if rec.get("duration_s") is None else f"{rec['duration_s']:.2f}")
        self._color(it, rec["status"])
        keep = [it.child(i) for i in range(it.childCount()) if it.child(i).data(0, Qt.ItemDataRole.UserRole) == "info"]
        it.takeChildren()
        for label, key in (("입력", "input_summary"), ("출력", "output_summary"), ("오류/사유", "error"), ("다음", "next")):
            if rec.get(key):
                c = QTreeWidgetItem([f"{label}: {rec[key]}"])
                c.setToolTip(0, str(rec[key]))
                it.addChild(c)
        for c in keep:
            it.addChild(c)
        if rec["status"] in ("RUNNING", "FAILED", "UNCERTAIN"):
            it.setExpanded(True)
        if rec["name"] == "plan_path":
            self.lbl_planner.setText(f"Path Planner: {'RUNNING' if rec['status'] == 'RUNNING' else ('ERROR' if rec['status'] == 'FAILED' else 'READY')}")

    def _on_event(self, ev):
        t = ev["type"]
        if t == "plan":
            self._set_tree_plan([s["name"] for s in ev["steps"]])
            for s in ev["steps"]:
                self._update_step(s)
        elif t in ("step_started", "step_finished"):
            self._update_step(ev["record"])
        elif t == "model_status":
            self._set_model(ev["model"], ev["status"])
        elif t == "info" and ev.get("step") in self.items:
            c = QTreeWidgetItem([f"· {ev['message']}"])
            c.setData(0, Qt.ItemDataRole.UserRole, "info")
            self.items[ev["step"]].addChild(c)
        elif t == "run_started":
            cal = ev["calibration"]
            self.lbl_mapmode.setText(f"Map: UNCALIBRATED ({cal['map_mode']})")
            self.lbl_mapmode.setToolTip("\n".join(cal["reasons"]))

    def _on_done(self, run):
        self.lbl_agent.setText("Task Agent: READY")
        self._show_state(run.state, run.writer.dir, live_note=True)

    def _on_failed(self, msg):
        self.lbl_agent.setText("Task Agent: ERROR")
        self._error(msg.splitlines()[0])
        QMessageBox.critical(self, "분석 오류", msg[:4000])

    def _show_state(self, state, run_dir, live_note=False, handoff_override=None):
        self.state, self.run_dir = state, Path(run_dir)
        r = state["result"]
        if handoff_override:
            r["handoff"] = handoff_override
        f = r["frame"]
        self.lbl_result.setText(f"{r['overall_status']} | 분석 프레임 #{f['frame_id']} @ {f['timestamp_iso']} "
                                f"({'저장된 분석 결과 — 현재 라이브 아님' if live_note else 'REPLAY'}) | {self.run_dir.name}")
        self.lbl_lat.setText(f"Latency: {r['timing']['total_s']:.1f}s")
        self._fill_table(r)
        self._update_handoff(r.get("handoff"))
        if r.get("status_reasons"):
            self.lbl_err.setText(r["status_reasons"][0][:160])
        self._render_result()

    def _fill_table(self, r):
        rows = list((r.get("detections") or {}).get("objects", []))
        t = (r.get("target_inspection") or {}).get("target")
        if t:
            rows.append(t)
        for c in (r.get("target_inspection") or {}).get("candidates", []):
            if not t or c.get("candidate_id") != t.get("candidate_id"):
                rows.append({"object_id": c["candidate_id"], "class_name": "target candidate", "score": c.get("score"),
                             "bbox_xyxy": c["bbox_xyxy"], "source": "+".join(sorted(set(c["sources"]))),
                             "verification": c["verification"], "accepted": False})
        self.table.setRowCount(len(rows))
        for i, o in enumerate(rows):
            ver = (o.get("verification") or {}).get("status", "")
            vals = [o["object_id"], o["class_name"], "" if o.get("score") is None else f"{o['score']:.3f}",
                    ", ".join(f"{v:.0f}" for v in o["bbox_xyxy"]), o.get("source", ""),
                    ("ACCEPTED " if o.get("accepted") else "") + ver]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                if j == 5:
                    item.setToolTip(json.dumps((o.get("verification") or {}).get("answer"), ensure_ascii=False))
                self.table.setItem(i, j, item)

    # ------------------------------------------------------------------ result view
    def _view_changed(self, view):
        defaults = set(visualization.VIEW_LAYERS[view])
        for name, cb in self.layer_checks.items():
            cb.blockSignals(True)
            cb.setChecked(name in defaults)
            cb.blockSignals(False)
        self._render_result()

    def _render_result(self):
        if self.btn_edit.isChecked():
            base = self.state["frame"] if self.state else (self.live_frame.image if self.live_frame else None)
            if base is not None:
                self.res_view.set_image(render_map_editor(base, self.map_spec, self.edit_points,
                                                          self.cmb_edit.currentData()))
            return
        if self.state is None:
            return
        layers = [n for n, cb in self.layer_checks.items() if cb.isChecked()]
        self.current_render = visualization.render_view(self.state, self.cmb_view.currentText(), layers)
        self.res_view.set_image(self.current_render)

    # ------------------------------------------------------------------ map editor
    def _update_map_label(self):
        self.lbl_map.setText(f"지도: {Path(str(self.map_source)).name} [{self.map_spec.get('review_status')}]")

    def _edit_toggled(self, on):
        self.edit_points = []
        self._render_result()

    def _result_clicked(self, x, y):
        if not self.btn_edit.isChecked():
            return
        kind = self.cmb_edit.currentData()
        if kind == "start_pose":
            self.map_spec["start_pose"] = {"xy": [round(x, 1), round(y, 1)], "source": "manual_gui_edit"}
            self._mark_edited()
        else:
            self.edit_points.append([round(x, 1), round(y, 1)])
        self._render_result()

    def _close_polygon(self):
        kind = self.cmb_edit.currentData()
        if kind == "start_pose" or len(self.edit_points) < 3:
            return self._error("다각형은 점 3개 이상이 필요합니다")
        lst = self.map_spec.setdefault(kind, [])
        lst.append({"id": f"{kind}_{len(lst) + 1}_gui", "polygon": self.edit_points, "source": "manual_gui_edit"})
        self.edit_points = []
        self._mark_edited()
        self._render_result()

    def _clear_type(self):
        kind = self.cmb_edit.currentData()
        if kind == "start_pose":
            self.map_spec["start_pose"] = None
        else:
            self.map_spec[kind] = []
        self._mark_edited()
        self._render_result()

    def _mark_edited(self):
        base = self.state["frame"] if self.state else (self.live_frame.image if self.live_frame else None)
        if base is not None:
            self.map_spec["image_size"] = [base.shape[1], base.shape[0]]
        self.map_spec["review_status"] = "EDITED_IN_GUI_UNCONFIRMED"
        self.map_spec["map_id"] = self.map_spec.get("map_id", "map").split("+")[0] + "+gui_edit"
        self.map_source = "(unsaved GUI edit)"
        self._editor_base = base
        self._update_map_label()

    def _save_map(self):
        try:
            validate(self.map_spec, MAP_SCHEMA)
        except SchemaError as e:
            return self._error(f"지도 저장 불가: {e}")
        p, _ = QFileDialog.getSaveFileName(self, "지도 저장", str(resolve_path("config/maps/edited_map.json")), "JSON (*.json)")
        if not p:
            return
        p = Path(p)
        if p.exists():
            return self._error("기존 지도 파일은 덮어쓰지 않습니다. 새 이름을 사용하세요")
        base = getattr(self, "_editor_base", None)
        if base is not None:
            ref = p.with_suffix(".reference.png")
            cv2.imwrite(str(ref), base)
            self.map_spec["reference_image"] = str(ref)
            self.map_spec["reference_image_sha256"] = sha256_file(ref)
            self.map_spec["reference_image_note"] = "frame on which the polygons were drawn in the GUI"
        self.map_spec["provenance"] = {"knowledge": "manual_prior", "edited_in_gui": datetime.now().isoformat(
            timespec="seconds"), "based_on": self.map_spec.get("provenance")}
        p.write_text(json.dumps(self.map_spec, indent=2, ensure_ascii=False), encoding="utf-8")
        self.map_source = str(p)
        self._update_map_label()

    def _load_map_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "지도 불러오기", str(resolve_path("config/maps")), "JSON (*.json)")
        if p:
            try:
                self.map_spec, self.map_source = load_map_spec(p), p
            except (SchemaError, ValueError, OSError) as e:
                return self._error(f"지도 불러오기 실패: {e}")
            self._update_map_label()
            self._render_result()

    # ------------------------------------------------------------------ hand-off
    def _update_handoff(self, h):
        for b in self.handoff_buttons.values():
            b.setEnabled(False)
        if not h:
            self.lbl_handoff.setText("인계 요청 없음 (실행 가능한 계획 없음)")
            return
        hist = " → ".join(x["state"] for x in h["history"])
        self.lbl_handoff.setText(f"상태: {h['state']} | 경로 B: {h.get('path_b_status')} | 하드웨어: 비활성 | 이력: {hist}")
        allowed = {"WAITING_FOR_PICKUP_CONFIRMATION": ("pickup_ok", "pickup_failed"),
                   "WAITING_FOR_PLACEMENT_CONFIRMATION": ("placement_ok", "placement_failed"),
                   "WAITING_FOR_ARRIVAL_CONFIRMATION": ("arrival_ok",)}.get(h["state"], ())
        for ev in allowed:
            self.handoff_buttons[ev].setEnabled(True)

    def _handoff(self, event):
        if self.run_dir is None:
            return
        ok = QMessageBox.question(self, "시뮬레이션 확인", f"'{event}' 를 사용자 시뮬레이션 확인으로 기록합니다.\n"
                                  "실제 로봇 동작이나 파지 결과가 아닙니다. 계속할까요?")
        if ok != QMessageBox.StandardButton.Yes:
            return
        try:
            exe = record_handoff_event(self.run_dir, event, "user_simulation_confirmation", note="GUI button")
        except (HandoffError, RuntimeError) as e:
            return self._error(str(e))
        self.state["result"]["handoff"] = exe.to_dict()
        self._update_handoff(exe.to_dict())
        self._render_result()

    # ------------------------------------------------------------------ saving / replay
    def _save_snapshot(self):
        if self.run_dir is None:
            return self._error("저장할 실행 결과가 없습니다 (분석 결과는 완료 시 자동 저장됩니다)")
        w = RunWriter(self.run_dir, existing=True)
        shot = w.write_image(w.next_numbered("gui_exports", "gui_screenshot", ".png"),
                             qimage_to_bgr(self.grab().toImage()))
        view = w.write_image(w.next_numbered("gui_exports", f"view_{self.cmb_view.currentText()}", ".png"),
                             self.current_render) if getattr(self, "current_render", None) is not None else None
        QMessageBox.information(self, "저장", f"실행 결과는 자동 저장되어 있습니다:\n{self.run_dir}\n\n추가 저장:\n{shot}\n{view}")

    def _export_figures(self):
        if self.state is None:
            return self._error("내보낼 결과가 없습니다")
        w = RunWriter(self.run_dir, existing=True)
        folder = f"figures_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run_aux(lambda: visualization.export_paper_figures(self.state, folder, w),
                      lambda out: QMessageBox.information(self, "논문용 이미지", f"{len(out)}개 저장: {w.path(folder)}"))

    def _replay_dialog(self):
        p = QFileDialog.getExistingDirectory(self, "실행 폴더 선택", str(resolve_path(self.cfg["app"]["runs_dir"])))
        if p:
            self.load_replay(p)

    def load_replay(self, run_dir):
        def done(res):
            state, run = res
            self._stop_camera()
            self.source_kind = "replay"
            self.analysis_frame = None
            self.cam_view.set_image(state["frame"])
            self.lbl_camstat.setText("Camera: REPLAY (saved frame)")
            self.lbl_cam.setText(f"REPLAY {run['dir']}")
            self._set_tree_plan([s["name"] for s in state["result"]["steps"]])
            for s in state["result"]["steps"]:
                self._update_step(s)
            self._show_state(state, run["dir"], handoff_override=run["handoff_latest"])
        self._run_aux(lambda: replay_state(run_dir), done)

    def _error(self, msg):
        self.last_error = msg
        self.lbl_err.setText(msg[:200])
        self.lbl_err.setToolTip(msg)

    def closeEvent(self, e):
        self._stop_camera()
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.cancel.set()
            if self.client and self.client.busy:
                self.client.kill()
            self.analysis_worker.wait(15000)
        for w in list(self.aux_workers):
            w.wait(15000)
        if self.client:
            self.client.close()
        self.gpu.stop()
        super().closeEvent(e)


def qimage_to_bgr(qimg):
    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    arr = np.frombuffer(qimg.constBits(), np.uint8, count=qimg.bytesPerLine() * h).reshape(h, qimg.bytesPerLine())
    return cv2.cvtColor(arr[:, :3 * w].reshape(h, w, 3).copy(), cv2.COLOR_RGB2BGR)


def render_map_editor(base, spec, pending, pending_kind):
    img = base.copy()
    over = img.copy()
    colors = {"traversable": (60, 180, 75), "static_obstacles": (40, 40, 40), "no_go": (60, 20, 200)}
    for kind, col in colors.items():
        for z in spec.get(kind, []):
            cv2.fillPoly(over, [np.round(np.array(z["polygon"])).astype(np.int32)], col)
    cv2.addWeighted(over, 0.35, img, 0.65, 0, dst=img)
    for kind, col in colors.items():
        for z in spec.get(kind, []):
            cv2.polylines(img, [np.round(np.array(z["polygon"])).astype(np.int32)], True, col, 2)
            cv2.putText(img, f"{z['id']} [MANUAL]", tuple(int(v) for v in z["polygon"][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if spec.get("start_pose"):
        cv2.circle(img, tuple(int(v) for v in spec["start_pose"]["xy"]), 12, (0, 255, 0), -1)
        cv2.putText(img, "START [MANUAL]", tuple(int(v) + 14 for v in spec["start_pose"]["xy"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if pending:
        pts = np.round(np.array(pending)).astype(np.int32)
        cv2.polylines(img, [pts], False, (0, 255, 255), 2)
        for p in pts:
            cv2.circle(img, tuple(int(v) for v in p), 5, (0, 255, 255), -1)
    cv2.putText(img, f"MAP EDITOR (manual annotation, not model output) | adding: {pending_kind} | "
                     f"review: {spec.get('review_status')}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


def launch(cfg, args):
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(cfg)
    if args.analysis_mode:
        win.cmb_mode.setCurrentText(args.analysis_mode)
    if args.command:
        win.txt_cmd.setPlainText(args.command)
    if args.map:
        win.map_spec, win.map_source = load_map_spec(args.map), args.map
        win._update_map_label()
    if args.mode == "replay" and args.run_dir:
        win.load_replay(args.run_dir)
    elif args.mode == "live":
        win.cmb_source.setCurrentText("LIVE CAMERA")
        win.open_live(args.camera)
    elif args.mode == "video" and args.video:
        win.cmb_source.setCurrentText("VIDEO")
        win.open_video(args.video)
    else:
        win.open_image(args.image or resolve_path(cfg["app"]["default_image"]))
    win.resize(1720, 1000)
    win.show()
    return app.exec()
