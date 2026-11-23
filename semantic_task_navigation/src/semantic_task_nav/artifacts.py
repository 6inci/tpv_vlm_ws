"""Run directories: one new folder per analysis, files are never overwritten."""
import json
import secrets
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import sha256_file

SUBDIRS = ("input", "analysis", "analysis/crops", "planning", "figures", "handoff")


def new_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def dumps(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, default=json_default)


class RunWriter:
    def __init__(self, runs_dir, run_id=None, existing=False):
        if existing:
            self.dir = Path(runs_dir)
            self.run_id = self.dir.name
            if not (self.dir / "result.json").exists():
                raise FileNotFoundError(f"{self.dir} is not a run directory")
            return
        self.run_id = run_id or new_run_id()
        self.dir = Path(runs_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        for s in SUBDIRS:
            (self.dir / s).mkdir(parents=True, exist_ok=True)

    def path(self, rel):
        return self.dir / rel

    def _new(self, rel):
        p = self.path(rel)
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, rel, obj):
        p = self._new(rel)
        p.write_text(dumps(obj), encoding="utf-8")
        return p

    def write_text(self, rel, text):
        p = self._new(rel)
        p.write_text(text, encoding="utf-8")
        return p

    def write_image(self, rel, img):
        p = self._new(rel)
        if not cv2.imwrite(str(p), img):
            raise OSError(f"could not write {p}")
        return p

    def append_jsonl(self, rel, obj):
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=json_default) + "\n")

    def next_numbered(self, rel_dir, stem, ext):
        d = self.path(rel_dir)
        d.mkdir(parents=True, exist_ok=True)
        n = 1 + max([int(p.stem.rsplit("_", 1)[1]) for p in d.glob(f"{stem}_*{ext}")
                     if p.stem.rsplit("_", 1)[1].isdigit()] or [0])
        return f"{rel_dir}/{stem}_{n:03d}{ext}"

    def file_hashes(self, exclude=("manifest.json",)):
        return {str(p.relative_to(self.dir)): sha256_file(p) for p in sorted(self.dir.rglob("*"))
                if p.is_file() and str(p.relative_to(self.dir)) not in exclude}


def load_run(run_dir):
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    frame = cv2.imread(str(run_dir / "input" / "raw_frame.png"), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"{run_dir}/input/raw_frame.png missing")
    handoff_states = sorted((run_dir / "handoff").glob("state_*.json"))
    latest = json.loads(handoff_states[-1].read_text(encoding="utf-8")) if handoff_states else None
    return {"dir": run_dir, "result": result, "manifest": manifest, "frame": frame, "handoff_latest": latest}
