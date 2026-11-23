#!/usr/bin/env python3
"""Semantic Task Navigation entry point.

GUI (stnav_gui env):
  python run.py                                   # GUI with the default reference image
  python run.py --mode image --image <file>       # GUI with an image
  python run.py --mode video --video <file>       # GUI with a video file
  python run.py --mode live [--camera 0]          # GUI with a live camera
  python run.py --mode replay --run-dir runs/<id> # GUI replay of a saved run
Headless (no display needed):
  python run.py --mode image --image <file> --headless [--analysis-mode IMAGE_DEMO] [--command "..."]
  python run.py --mode replay --run-dir runs/<id> --headless   # re-render figures and compare
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from semantic_task_nav.config import ANALYSIS_MODES, load_config, load_dotenv, resolve_path  # noqa: E402


def headless_image(cfg, args):
    from semantic_task_nav.agent.orchestrator import AnalysisRun
    from semantic_task_nav.perception.camera import ImageSource
    from semantic_task_nav.perception.model_client import ModelWorkerClient

    mode = args.analysis_mode or cfg["app"]["default_analysis_mode"]
    frame = ImageSource(args.image or resolve_path(cfg["app"]["default_image"])).read()
    client = None
    if mode != "IMAGE_DEMO":
        client = ModelWorkerClient(cfg["models"], resolve_path(cfg["app"]["worker_log_dir"]))
        client.start()

    def show(ev):
        if ev["type"] == "step_finished":
            r = ev["record"]
            print(f"[{r['status']:>9}] {r['name']:<24} {r['duration_s'] or 0:7.2f}s  "
                  f"{(r['output_summary'] or r['error'] or '')[:150]}", flush=True)
        elif ev["type"] == "info":
            print(f"            {ev['step']}: {ev['message']}", flush=True)

    try:
        run = AnalysisRun(cfg, mode, frame, args.command or cfg["app"]["default_command"], client, on_event=show,
                          map_spec=None, map_source=args.map or cfg["app"]["map_file"])
        result = run.run()
    finally:
        if client:
            client.close()
    print(json.dumps({"run_dir": str(run.writer.dir), "overall_status": result["overall_status"],
                      "status_reasons": result["status_reasons"]}, ensure_ascii=False, indent=1))
    return 0 if result["overall_status"] in ("PLAN_COMPLETE", "PARTIAL_PLAN") else 2


def headless_replay(args):
    import cv2
    import numpy as np
    from semantic_task_nav import visualization
    from semantic_task_nav.agent.orchestrator import replay_state

    state, run = replay_state(args.run_dir)
    report = {}
    for name, view in visualization.FIGURES:
        fig = Path(args.run_dir) / "figures" / f"{name}.png"
        saved = cv2.imread(str(fig), cv2.IMREAD_COLOR) if fig.exists() else None
        if saved is None:   # e.g. figures added after the run's code version (v0.1 runs have 10 figures)
            report[name] = "NOT_IN_RUN"
            continue
        again = visualization.paper_figure(state, view)
        report[name] = "IDENTICAL" if saved.shape == again.shape and np.array_equal(saved, again) else "DIFFERENT"
    print(json.dumps({"run_dir": str(run["dir"]), "overall_status": run["result"]["overall_status"],
                      "figures": report}, indent=1))
    return 0 if all(v in ("IDENTICAL", "NOT_IN_RUN") for v in report.values()) else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["gui", "image", "video", "live", "replay"], default="gui")
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--camera", help="device index or /dev/videoN (default: config/camera.yaml)")
    ap.add_argument("--run-dir")
    ap.add_argument("--command", help="natural-language instruction")
    ap.add_argument("--analysis-mode", choices=ANALYSIS_MODES)
    ap.add_argument("--map", help="map JSON (default: config/app.yaml map_file)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=JSON",
                    help="config override for ablations, recorded in the run manifest, e.g. "
                         "--set 'planning.destination.candidate_sources=[\"geometric_bands\"]'")
    args = ap.parse_args()
    load_dotenv()
    cfg = load_config()
    for item in args.set:
        key, _, raw = item.partition("=")
        *parents, leaf = key.split(".")
        node = cfg
        for k in parents:
            node = node[k]          # unknown keys fail loudly
        if leaf not in node:
            ap.error(f"--set: unknown config key {key}")
        try:
            node[leaf] = json.loads(raw)
        except json.JSONDecodeError:
            node[leaf] = raw
    cfg["app"]["config_overrides"] = list(args.set)
    if args.headless:
        if args.mode == "replay":
            if not args.run_dir:
                ap.error("--run-dir is required")
            return headless_replay(args)
        if args.mode in ("image", "gui"):
            return headless_image(cfg, args)
        ap.error("--headless supports --mode image or --mode replay")
    from semantic_task_nav.gui.main_window import launch
    return launch(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
