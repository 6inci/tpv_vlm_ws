"""Project paths, YAML configuration and provenance helpers."""
import hashlib
import os
import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILES = ("app", "models", "camera", "planning")

ANALYSIS_MODES = ("LOCAL_ONLY", "LOCAL_VLM_WITH_API_AGENT", "IMAGE_DEMO")


def load_config(config_dir=CONFIG_DIR):
    cfg = {}
    for name in CONFIG_FILES:
        with open(Path(config_dir) / f"{name}.yaml", encoding="utf-8") as f:
            cfg[name] = yaml.safe_load(f)
    return cfg


def resolve_path(path):
    """Absolute paths are kept; relative paths resolve against the project root."""
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def code_fingerprint():
    """sha256 over run.py and all src/*.py (sorted). Used because the project has no git repo."""
    files = sorted((PROJECT_ROOT / "src").rglob("*.py")) + [PROJECT_ROOT / "run.py"]
    h = hashlib.sha256()
    for f in files:
        if f.exists():
            h.update(str(f.relative_to(PROJECT_ROOT)).encode())
            h.update(f.read_bytes())
    return {"sha256": h.hexdigest(), "n_files": len(files), "git_commit": _git_commit()}


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def load_dotenv(path=PROJECT_ROOT / ".env"):
    """Minimal KEY=VALUE loader. Existing environment variables win. Values are never logged."""
    if not Path(path).exists():
        return []
    loaded = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
