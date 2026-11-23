"""Test A (GUI starts + image mode), B (1920x1080 shown without distortion), I (GUI stays responsive
during a slow model call, stop works). Runs offscreen."""
import json
import threading
import time

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import REF_IMAGE  # noqa: E402
from semantic_task_nav import visualization  # noqa: E402
from semantic_task_nav.gui.main_window import ImageView, MainWindow, bgr_to_qimage, qimage_to_bgr  # noqa: E402
from semantic_task_nav.perception.model_client import ModelWorkerError  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, cfg):
    w = MainWindow(cfg)
    w.resize(1600, 950)
    w.show()
    yield w
    w.close()   # WA_DeleteOnClose: Qt deletes the window
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QApplication.processEvents()


def wait_for(cond, timeout=60):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        QTest.qWait(20)
    return cond()


def test_image_shown_with_uniform_scale(app):
    img = cv2.imread(str(REF_IMAGE))
    assert img.shape == (1080, 1920, 3)
    assert np.array_equal(qimage_to_bgr(bgr_to_qimage(img)), img)         # lossless conversion
    v = ImageView()
    v.resize(800, 800)
    v.set_image(img)
    rect, s = v.image_rect()
    assert rect.width() == 800 and rect.height() == 450                    # same scale on both axes
    assert abs(rect.width() / 1920 - rect.height() / 1080) < 1e-3
    got = []
    v.clicked.connect(lambda x, y: got.append((x, y)))
    QTest.mouseClick(v, __import__("PySide6.QtCore").QtCore.Qt.MouseButton.LeftButton,
                     pos=__import__("PySide6.QtCore").QtCore.QPoint(rect.x() + 400, rect.y() + 225))
    assert got and abs(got[0][0] - 960) < 3 and abs(got[0][1] - 540) < 3


def test_gui_image_demo_run(win, cfg):
    win.open_image(REF_IMAGE)
    win.cmb_mode.setCurrentText("IMAGE_DEMO")
    win.start_analysis()
    assert wait_for(lambda: win.state is not None)
    assert win.state["result"]["overall_status"] == "PLAN_COMPLETE"
    statuses = {win.tree.topLevelItem(i).text(0): win.tree.topLevelItem(i).text(1)
                for i in range(win.tree.topLevelItemCount())}
    assert any("parse_task" in k and v == "OK" for k, v in statuses.items())
    assert any("analyze_scene" in k and v == "NOT_RUN" for k, v in statuses.items())   # no fake model output
    assert win.table.rowCount() >= 3
    for view in visualization.VIEWS:
        win.cmb_view.setCurrentText(view)
        assert win.current_render is not None
    assert win.handoff_buttons["pickup_ok"].isEnabled() and not win.handoff_buttons["placement_ok"].isEnabled()


class FakeSlowClient:
    """Stands in for the model worker: every Qwen call takes `delay` seconds and returns an
    'unsupported' parse, so the run ends at task parsing. kill() aborts like the real worker."""
    def __init__(self, delay):
        self.delay, self.loaded, self.busy, self.killed = delay, {}, False, threading.Event()
        self.env_info, self.log_path, self.swaps = {"fake": True}, None, []

    def ensure_loaded(self, name):
        self.loaded[name] = {"model_id": "fake"}
        return self.loaded[name]

    def call(self, cmd, timeout=None, **args):
        self.busy = True
        try:
            t0 = time.time()
            while time.time() - t0 < self.delay:
                if self.killed.is_set():
                    raise ModelWorkerError("WORKER_TERMINATED", "killed")
                time.sleep(0.01)
            return {"text": json.dumps({"status": "unsupported", "task_type": None, "target": None,
                                        "destination": None, "reason": "fake"}), "model": "fake", "commit_hash": None,
                    "latency_s": self.delay, "n_output_tokens": 1, "hit_max_new_tokens": False}
        finally:
            self.busy = False

    def kill(self):
        self.killed.set()

    def close(self):
        pass


def test_gui_responsive_during_inference(win):
    win.open_image(REF_IMAGE)
    fake = FakeSlowClient(1.5)
    win.client = fake
    win._get_client = lambda: fake
    win.cmb_mode.setCurrentText("LOCAL_ONLY")
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start(20)
    win.start_analysis()
    assert wait_for(lambda: win.state is not None, 20)
    timer.stop()
    gaps = np.diff(ticks)
    assert len(ticks) > 40 and gaps.max() < 0.3, f"max event-loop gap {gaps.max():.3f}s"
    assert win.state["result"]["overall_status"] == "TASK_REJECTED"


def test_stop_button_cancels_running_inference(win):
    win.open_image(REF_IMAGE)
    fake = FakeSlowClient(30)
    win.client = fake
    win._get_client = lambda: fake
    win.cmb_mode.setCurrentText("LOCAL_ONLY")
    win.start_analysis()
    assert wait_for(lambda: fake.busy, 10)
    t0 = time.time()
    win.stop_analysis()
    assert wait_for(lambda: not fake.busy, 2) and time.time() - t0 < 2      # running inference aborted
    assert wait_for(lambda: win.state is not None, 30)                       # partial results still saved
    assert win.state["result"]["overall_status"] == "CANCELLED"
    assert win.state["result"]["steps"][0]["status"] == "CANCELLED"
