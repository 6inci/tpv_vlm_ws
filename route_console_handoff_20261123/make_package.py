#!/usr/bin/env python3
"""경로 계획 콘솔 시스템 묶음 만들기 (2026-11-23). 원본은 읽기만 하고 복사본만 만든다.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -B make_package.py
결과: package/route_console_handoff_20261123/ , route_console_handoff_20261123.zip (이 폴더), 같은 zip을 ~/Downloads 에도 복사.
이미 있으면 덮어쓰지 않고 멈춘다.
"""
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = Path("/home/binci/vlm_ws")
NAME = "route_console_handoff_20261123"
ROOT = HERE / "package" / NAME
ZIP = HERE / f"{NAME}.zip"
DL = Path.home() / "Downloads" / f"{NAME}.zip"
RC = WS / "route_console_20261122"

# (묶음 안 경로, 원본) — 폴더는 통째로
CONSOLE = ["engine.py", "server.py", "run_console.sh", "test_offline.py", "README.md", "config/console.yaml", "web/index.html", "web/style.css", "web/app.js"]
REUSED = [
    "context_policy_20261122/README.md", "context_policy_20261122/common.py", "context_policy_20261122/extract.py", "context_policy_20261122/policy.py",
    "context_policy_20261122/config/policy.yaml", "context_policy_20261122/zones/zones_env2.json", "context_policy_20261122/zones_to_env2.py",
    "context_policy_20261122/data/dev/calibration.json", "context_policy_20261122/data/dev/bank.json", "context_policy_20261122/data/gt_test.json",
    "hybrid_exemplar_20261121/README.md", "hybrid_exemplar_20261121/hybrid_pipeline.py", "hybrid_exemplar_20261121/config.yaml",
    "semantic_task_navigation/experiments/multimodel_target_viewer/mm_pipeline.py",
    "semantic_task_navigation/experiments/multimodel_target_viewer/mm_worker.py",
    "semantic_task_navigation/experiments/multimodel_target_viewer/config.yaml",
    "semantic_task_navigation/experiments/task_context_2d_navigation/config.yaml",
    "nl_to_route_20261122/README.md", "nl_to_route_20261122/POLICIES.md", "nl_to_route_20261122/nl_to_path.py", "nl_to_route_20261122/config/nl_to_path.yaml",
    "nl_to_route_20261122/vendor/teammate_extra_20261120",                       # 팀원 자료 (통째로, 바꾸지 않음)
    "env2_photos_20261120/README.md", "env2_photos_20261120/config.yaml", "env2_photos_20261120/fit_photoset_pose.py", "env2_photos_20261120/SHA256SUMS",
    "env2_photos_20261120/camera_pose",
]
RUNS = ["20261122_192459_20261120_215815_view", "20261122_192502_20261120_215900_view", "20261122_192504_20261120_220042_view",
        "20261122_192506_20261120_215815_view", "20261122_192509_20261120_215815_view", "20261122_192510_20261120_215815_view",
        "20261122_192512_20261120_215815_view", "20261122_195711_20261120_220344_view", "20261122_200350_20261120_220632_view",
        "20261122_193058_live_simulated_220632"]
CP_TEST = WS / "context_policy_20261122/results/test_20261122_183736"
CP_DEV = WS / "context_policy_20261122/results/dev_20261122_183649"
EVIDENCE = [("design_check_20261122_README.md", WS / "design_check_20261122/README.md"),
            ("pipeline_comparison_20261121.md", WS / "pipeline_comparison_20261121/PIPELINE_COMPARISON.md")]
FIGS = [("220103_photo.png", Path("/home/binci/Pictures/Calibration/map_images/20261120_220103_view.png")),
        ("220103_path_overlay_this_system.jpg", CP_TEST / "paths/20261120_220103_view/path_overlay.jpg"),
        ("215826_path_overlay_this_system.jpg", CP_TEST / "paths/20261120_215826_view/path_overlay.jpg"),
        ("215847_hold_example_policy.jpg", CP_TEST / "20261120_215847_view_policy.jpg"),
        ("vlm_five_stages.pdf", Path("/home/binci/Pictures/vlm_five_stages.pdf"))]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def put(src, dst, rows):
    if src.is_dir():
        for f in sorted(src.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                put(f, dst / f.relative_to(src), rows)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    h = sha(src)
    assert h == sha(dst), src
    rows.append((str(dst.relative_to(ROOT)), str(src), h))


def run_rows():
    out = []
    for r in RUNS:
        rec = json.loads((RC / "runs" / r / "record.json").read_text(encoding="utf-8"))
        legs = [f"{L['from']}→{L['to']}" for L in (rec.get("geometry") or {}).get("legs", [])]
        why = (rec["result"].get("why") or "").split(";")[0]          # 상위 후보 1위까지만
        out.append(f"- `{r}`: \"{rec['instruction']}\" → {rec['result']['status']}" + (f" ({', '.join(legs)})" if legs else "") + (f" — {why}" if why else ""))
    return out


def photo_summary():
    d = json.loads((CP_TEST / "decisions.json").read_text(encoding="utf-8"))
    c = {}
    for p in d["photos"]:
        for k in ("A", "B"):
            g = p[k].get("grade")
            c[(k, g)] = c.get((k, g), 0) + 1
    return c


def main():
    for p in (ROOT, ZIP, DL):
        if p.exists():
            raise SystemExit(f"{p} 이미 있음. 덮어쓰지 않음")
    rows = []
    for f in CONSOLE:
        put(RC / f, ROOT / "1_console" / f, rows)
    for f in REUSED:
        put(WS / f, ROOT / "2_reused" / f, rows)
    put(CP_TEST, ROOT / "3_results/photos15_this_system", rows)
    put(CP_DEV, ROOT / "3_results/dev8_threshold_photos", rows)
    for r in RUNS:
        for f in sorted((RC / "runs" / r).iterdir()):
            if f.name != "route_package.zip":                       # 같은 파일을 묶은 zip이라 뺌
                put(f, ROOT / "3_results/console_runs" / r / f.name, rows)
    for n, s in EVIDENCE:
        put(s, ROOT / "4_design_evidence" / n, rows)
    for n, s in FIGS:
        put(s, ROOT / "5_paper_figures" / n, rows)

    c = photo_summary()
    readme = (Path(__file__).parent / "README_template.md").read_text(encoding="utf-8")
    readme = readme.replace("{created}", datetime.now().isoformat(timespec="minutes")).replace("{runs}", "\n".join(run_rows()))
    readme = readme.replace("{grades}", ", ".join(f"{k} {g} {n}" for (k, g), n in sorted(c.items())))
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    (ROOT / "MANIFEST.txt").write_text("묶음 안 경로 | 원본 경로 | sha256 (원본과 같음을 확인함)\n" + "\n".join(" | ".join(r) for r in rows) + "\n", encoding="utf-8")

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(ROOT.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(ROOT.parent))
    with zipfile.ZipFile(ZIP) as zf:
        bad, n = zf.testzip(), len(zf.namelist())
    shutil.copy2(ZIP, DL)
    assert sha(ZIP) == sha(DL)
    print(f"파일 {len(rows) + 2}개, zip 항목 {n}개, 손상 {bad}, {ZIP.stat().st_size / 1e6:.1f} MB")
    print("zip:", ZIP)
    print("다운로드 폴더:", DL)


if __name__ == "__main__":
    main()
