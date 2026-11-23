"""Client for the model worker process (separate process, qwen_vl env)."""
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

WORKER_SCRIPT = Path(__file__).with_name("model_worker.py")


class ModelWorkerError(RuntimeError):
    def __init__(self, etype, message):
        super().__init__(f"{etype}: {message}")
        self.etype, self.message = etype, message


class ModelWorkerClient:
    """One persistent worker. call() is used from a single (pipeline) thread; kill() may be called
    from any thread to abort a running request (the models are then reloaded on the next use)."""

    def __init__(self, models_cfg, log_dir):
        self.cfg = models_cfg
        self.log_dir = Path(log_dir)
        self.proc = None
        self.log_path = None
        self._lines = queue.Queue()
        self._next_id = 0
        self.busy = False
        self.loaded = {}
        self.swaps = []
        self.env_info = None

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        if self.alive:
            return self.env_info
        python = self.cfg["worker"]["python"]
        if not Path(python).exists():
            raise ModelWorkerError("WORKER_PYTHON_MISSING", python)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"worker_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1",
                    "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        self._log_file = open(self.log_path, "a", encoding="utf-8")
        # -I: isolated mode, so sibling modules in this folder can never shadow library imports.
        self.proc = subprocess.Popen([python, "-I", str(WORKER_SCRIPT)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._log_file, text=True, bufsize=1, env=env,
                                     cwd=str(WORKER_SCRIPT.parent))
        self._lines = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc, self._lines), daemon=True).start()
        self.loaded = {}
        self.env_info = self.call("ping", timeout=self.cfg["worker"]["startup_timeout_s"])
        return self.env_info

    @staticmethod
    def _reader(proc, q):
        for line in proc.stdout:
            q.put(line)
        q.put(None)  # EOF

    def call(self, cmd, timeout=None, **args):
        if not self.alive:
            raise ModelWorkerError("WORKER_NOT_RUNNING", "model worker is not running")
        self._next_id += 1
        req_id = self._next_id
        self.busy = True
        try:
            self.proc.stdin.write(json.dumps({"id": req_id, "cmd": cmd, "args": args}) + "\n")
            self.proc.stdin.flush()
            deadline = time.monotonic() + (timeout or self.cfg["worker"]["request_timeout_s"])
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.kill()
                    raise ModelWorkerError("TIMEOUT", f"{cmd} exceeded {timeout}s; worker terminated")
                try:
                    line = self._lines.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    continue
                if line is None:
                    raise ModelWorkerError("WORKER_TERMINATED", f"worker exited during {cmd} (see {self.log_path})")
                resp = json.loads(line)
                if resp.get("id") == req_id:
                    break
        except (BrokenPipeError, OSError) as e:
            raise ModelWorkerError("WORKER_TERMINATED", str(e)) from None
        finally:
            self.busy = False
        if not resp["ok"]:
            raise ModelWorkerError(resp["error"]["type"], resp["error"]["message"])
        return resp["result"]

    def ensure_loaded(self, name):
        """name: 'qwen' | 'dino'. Returns load info (model id, commit hash, load time).
        With gpu_residency=exclusive the other model is unloaded first: measured on the RTX 5070 Ti
        (12 GB), Qwen3-VL-4B bf16 (8.3 GiB) + Grounding DINO inference does not fit (CUDA OOM)."""
        if name not in self.loaded:
            if self.cfg["worker"].get("gpu_residency", "exclusive") == "exclusive":
                others = [n for n in self.loaded if n != name]
                if others:
                    self.call("unload", names=others)
                    self.swaps.append({"unloaded": others, "for": name, "t": time.time()})
                    for n in others:
                        self.loaded.pop(n)
            spec = self.cfg["qwen"] if name == "qwen" else self.cfg["grounding_dino"]
            args = {"name": name, "model_id": spec["model_id"], "revision": spec.get("revision")}
            if name == "qwen":
                args["dtype"] = spec["dtype"]
            self.loaded[name] = self.call("load", **args)
        return self.loaded[name]

    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        self.loaded = {}

    def close(self):
        if self.alive:
            try:
                self.call("shutdown", timeout=10)
            except ModelWorkerError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.loaded = {}
