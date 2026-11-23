#!/usr/bin/env python3
"""실행 전에 고정하는 코드·설정·모델 목록 (config/holdout.yaml freeze) -> FREEZE.json.

  $PY -B freeze.py --write    목록을 만든다. 정답이 확정됐거나 실행 폴더가 있으면 거부 (그 뒤에는 고칠 수 없음)
  $PY -B freeze.py --check    지금 파일들이 FREEZE.json과 같은지. 다르면 바뀐 것을 보여 주고 종료 코드 1
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import hold as H

SKIP_PARTS = {"__pycache__", "results", "runs"}


def collect():
    fz = H.CFG["freeze"]
    files = [Path(p) for p in fz["files"]]
    for d in fz["dirs"]:
        base = Path(d["dir"])
        it = base.rglob(d["glob"]) if d.get("recursive") else base.glob(d["glob"])
        files += [p for p in it if p.is_file() and not SKIP_PARTS & set(p.relative_to(base).parts)]
    out = {}
    for p in sorted(set(files)):
        if not p.exists():
            raise SystemExit(f"고정 목록의 파일이 없습니다: {p}")
        out[str(p)] = H.sha256_file(p)
    models = {}
    for m in fz["models"]:
        ref = Path(fz["hf_cache"]) / f"models--{m.replace('/', '--')}" / "refs" / "main"
        models[m] = ref.read_text().strip() if ref.exists() else None
    return {"files": out, "models": models}


def diff(old, new):
    lines = []
    for k in ("files", "models"):
        a, b = old.get(k, {}), new.get(k, {})
        lines += [f"{k}: 바뀜 {p}" for p in sorted(set(a) & set(b)) if a[p] != b[p]]
        lines += [f"{k}: 없어짐 {p}" for p in sorted(set(a) - set(b))]
        lines += [f"{k}: 새로 생김 {p}" for p in sorted(set(b) - set(a))]
    return lines


def locked():
    gt = H.load_gt()
    return bool(gt and gt.get("confirmed_by")) or bool(H.run_dirs())


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    now = collect()
    if a.write:
        if locked():
            raise SystemExit("정답이 확정됐거나 실행 폴더가 있어 고정 목록을 다시 만들 수 없습니다.")
        if H.FREEZE_FILE.exists():
            print("\n".join(diff(json.loads(H.FREEZE_FILE.read_text(encoding="utf-8")), now)) or "(이전 목록과 같음)")
        H.FREEZE_FILE.write_text(json.dumps({"created": datetime.now().isoformat(timespec="seconds"), "experiment_id": H.CFG["experiment_id"], **now},
                                            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"저장: {H.FREEZE_FILE} (파일 {len(now['files'])}개, 모델 {len(now['models'])}개)")
        return
    if not H.FREEZE_FILE.exists():
        raise SystemExit("FREEZE.json이 없습니다 (--write 먼저)")
    d = diff(json.loads(H.FREEZE_FILE.read_text(encoding="utf-8")), now)
    if d:
        print("고정한 뒤 바뀐 것:\n  " + "\n  ".join(d))
        sys.exit(1)
    print(f"고정 목록과 같음 (파일 {len(now['files'])}개, 모델 {len(now['models'])}개)")


if __name__ == "__main__":
    main()
