"""Background workers. Nothing slow runs on the GUI thread; results arrive through Qt signals
(queued across threads)."""
import subprocess
import threading
import time
import traceback

from PySide6.QtCore import QThread, Signal

from ..agent.orchestrator import AnalysisRun
from ..perception.camera import SourceError


class CameraWorker(QThread):
    """Reads frames from a video/live source; the preview keeps running during analysis."""
    frame_ready = Signal(object)
    status = Signal(str, str)

    def __init__(self, factory, max_fps):
        super().__init__()
        self.factory, self.period = factory, 1.0 / max(1.0, max_fps)
        self._stop = threading.Event()
        # Back-pressure: a new frame is emitted only after the GUI consumed the previous one;
        # frames read meanwhile are dropped (keeps the preview real-time and memory bounded).
        self.consumed = threading.Event()
        self.consumed.set()
        self.dropped = 0

    def run(self):
        try:
            src = self.factory()
        except SourceError as e:
            self.status.emit("DISCONNECTED", str(e))
            return
        self.status.emit("CONNECTED", str(src.info))
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                f = src.read()
                if f is None:
                    self.status.emit("ENDED" if src.kind == "video" else "DISCONNECTED", "no more frames")
                    break
                if self.consumed.is_set():
                    self.consumed.clear()
                    self.frame_ready.emit(f)
                else:
                    self.dropped += 1
                dt = self.period - (time.monotonic() - t0)
                if dt > 0:
                    self._stop.wait(dt)
        finally:
            src.release()

    def stop(self):
        self._stop.set()
        self.wait(3000)


class AnalysisWorker(QThread):
    event = Signal(object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg, mode, frame, command, client_getter, map_spec, map_source):
        super().__init__()
        self.cfg, self.mode, self.frame, self.command = cfg, mode, frame, command
        self.client_getter, self.map_spec, self.map_source = client_getter, map_spec, map_source
        self.cancel = threading.Event()
        self.run_obj = None

    def run(self):
        try:
            client = None if self.mode == "IMAGE_DEMO" else self.client_getter()
            self.run_obj = AnalysisRun(self.cfg, self.mode, self.frame, self.command, client,
                                       on_event=self.event.emit, cancel_event=self.cancel,
                                       map_spec=self.map_spec, map_source=self.map_source)
            self.run_obj.run()
            self.done.emit(self.run_obj)
        except Exception as e:  # shown in the GUI with the traceback; the run dir keeps what was saved
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class FunctionWorker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as e:  # reported to the GUI
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class GpuMonitor(QThread):
    """Polls nvidia-smi (measured values only)."""
    reading = Signal(str)

    def __init__(self, interval_s):
        super().__init__()
        self.interval = interval_s
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                used, total, util = [v.strip() for v in out.stdout.strip().split(",")[:3]]
                self.reading.emit(f"GPU {used}/{total} MiB, {util}%")
            except (OSError, ValueError, subprocess.TimeoutExpired) as e:
                self.reading.emit(f"GPU: n/a ({type(e).__name__})")
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.wait(3000)
