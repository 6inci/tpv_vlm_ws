"""Test A (image mode runs), J (artifacts + metadata), K (failures), L (hand-off on a run), M (replay).
Uses IMAGE_DEMO (manual test data, no model) and a fake model client where a model is needed."""
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import EXAMPLE_COMMAND, REF_IMAGE
from semantic_task_nav import visualization
from semantic_task_nav.agent.orchestrator import AnalysisRun, record_handoff_event, replay_state
from semantic_task_nav.artifacts import RunWriter
from semantic_task_nav.config import sha256_file
from semantic_task_nav.contracts import (DETECTION_SCHEMA, GOAL_POSE_SCHEMA, HANDOFF_SCHEMA, PATH_SCHEMA,
                                         REGION_SCHEMA, RELATION_SCHEMA, TASK_SCHEMA, validate)
from semantic_task_nav.perception.camera import ImageSource


def run_demo(cfg, command=EXAMPLE_COMMAND, cancel=None):
    frame = ImageSource(REF_IMAGE).read()
    r = AnalysisRun(cfg, "IMAGE_DEMO", frame, command, cancel_event=cancel)
    return r, r.run()


def test_image_demo_end_to_end(cfg):
    run, res = run_demo(cfg)
    d = run.writer.dir
    assert res["overall_status"] == "PLAN_COMPLETE", res["status_reasons"]
    assert res["analysis_mode"] == "IMAGE_DEMO" and res["map_mode"] == "IMAGE_SPACE_DEMO"
    for rel in ("input/raw_frame.png", "input/command.txt", "analysis/task_parse.json", "analysis/detections.json",
                "analysis/scene_graph.json" if False else "analysis/semantic_regions.json", "analysis/agent_events.jsonl",
                "planning/map.json", "planning/pickup_path.json", "planning/delivery_path.json", "planning/validation.json",
                "handoff/state_001.json", "result.json", "manifest.json"):
        assert (d / rel).exists(), rel
    assert len(list((d / "figures").glob("*.png"))) == len(visualization.FIGURES)
    # contracts of every structure
    validate(res["task"], TASK_SCHEMA)
    for o in res["detections"]["objects"] + [res["target_inspection"]["target"]]:
        validate(o, DETECTION_SCHEMA)
        assert o["knowledge"] == "manual_test_data"            # never reported as model output
    for r in res["scene_graph"]["relations"]:
        validate(r, RELATION_SCHEMA)
    validate(res["region_grounding"]["selected_loading_region"], REGION_SCHEMA)
    for k in ("pickup", "dropoff"):
        validate(res["goals"][k]["pose"], GOAL_POSE_SCHEMA)
    for p in res["paths"].values():
        validate(p, PATH_SCHEMA)
        assert p["status"] == "PATH_FOUND"
    assert all(v["valid"] for v in res["validation"].values())
    for f in (d / "handoff").glob("*_request.json"):
        validate(json.loads(f.read_text()), HANDOFF_SCHEMA)
    assert res["evaluation_record"]["end_to_end_planning"]["counts_as_automatic_e2e"] is False
    assert res["handoff"]["state"] == "WAITING_FOR_PICKUP_CONFIRMATION"
    # pickup pose is not the object centre and lies away from the object box
    cube = res["target_inspection"]["target"]["bbox_xyxy"]
    px, py = res["goals"]["pickup"]["pose"]["xy"]
    assert not (cube[0] <= px <= cube[2] and cube[1] <= py <= cube[3])


def test_navigate_command_plans_to_loading_area(cfg):
    run, res = run_demo(cfg, "트럭이 있는 하역장으로 이동해.")
    assert res["task"]["task_type"] == "navigate_to_region" and res["overall_status"] == "PLAN_COMPLETE"
    assert "inspect_target" not in [s["name"] for s in res["steps"]] and res["goals"].get("pickup") is None
    assert res["paths"]["A"]["path_id"] == "A_start_to_goal" and res["validation"]["A"]["valid"]
    assert res["handoff"]["state"] == "WAITING_FOR_ARRIVAL_CONFIRMATION"
    assert res["internal_error"] is None


def test_manifest_metadata_and_hashes(cfg):
    run, res = run_demo(cfg)
    m = json.loads((run.writer.dir / "manifest.json").read_text())
    for key in ("run_id", "created", "input", "command", "models", "prompts", "inference_params", "camera_calibration",
                "map", "coordinate_frames", "planning", "timing", "code", "manual_inputs", "auto_inferred",
                "overall_status", "config_snapshot"):
        assert key in m, key
    assert m["input"]["source_sha256"] == sha256_file(REF_IMAGE)
    assert m["map"]["alignment"]["status"] == "ALIGNED"
    for rel, h in m["files_sha256"].items():
        assert sha256_file(run.writer.dir / rel) == h, rel
    assert "OPENAI_API_KEY" not in json.dumps(m) or m["config_snapshot"]["app"]["api_agent"]["api_key_env"] == "OPENAI_API_KEY"


def test_replay_reproduces_figures(cfg):
    run, _ = run_demo(cfg)
    state, loaded = replay_state(run.writer.dir)
    for name, view in visualization.FIGURES:
        saved = cv2.imread(str(run.writer.dir / "figures" / f"{name}.png"))
        assert np.array_equal(saved, visualization.paper_figure(state, view)), name
    assert [(s["name"], s["status"]) for s in loaded["result"]["steps"]] == [
        (s["name"], s["status"]) for s in run.result["steps"]]
    assert all(s["status"] != "RUNNING" for s in loaded["result"]["steps"])


def test_handoff_events_append_and_never_overwrite(cfg):
    run, _ = run_demo(cfg)
    d = run.writer.dir
    first = (d / "handoff/state_001.json").read_bytes()
    exe = record_handoff_event(d, "pickup_ok", "user_simulation_confirmation")
    assert exe.state == "WAITING_FOR_PLACEMENT_CONFIRMATION"
    assert (d / "handoff/state_002.json").exists() and (d / "handoff/state_001.json").read_bytes() == first
    assert replay_state(d)[1]["handoff_latest"]["state"] == "WAITING_FOR_PLACEMENT_CONFIRMATION"
    with pytest.raises(FileExistsError):
        RunWriter(d, existing=True).write_json("result.json", {})


def test_missing_truck_gives_no_destination_and_no_path_b(cfg, tmp_path):
    demo = json.loads(Path(cfg["app"]["demo_inputs_file"]).read_text() if Path(cfg["app"]["demo_inputs_file"]).is_absolute()
                      else (REF_IMAGE.parents[2] / cfg["app"]["demo_inputs_file"]).read_text())
    demo["objects"] = [o for o in demo["objects"] if o["class_name"] != "truck"]
    p = tmp_path / "demo.json"
    p.write_text(json.dumps(demo))
    cfg["app"]["demo_inputs_file"] = str(p)
    run, res = run_demo(cfg)
    steps = {s["name"]: s for s in res["steps"]}
    assert steps["infer_destination"]["status"] == "FAILED" and "NO_TRUCK" in steps["infer_destination"]["error"]
    assert steps["select_dropoff_pose"]["status"] == "SKIPPED"
    assert "B" not in res["paths"] and res["region_grounding"].get("selected_loading_region") is None
    assert res["overall_status"] == "PARTIAL_PLAN" and res["handoff"]["state"] == "NOT_EXECUTABLE"


def test_demo_inputs_refused_for_other_image(cfg, tmp_path):
    img = cv2.imread(str(REF_IMAGE))
    other = tmp_path / "other.png"
    cv2.imwrite(str(other), np.roll(img, 40, axis=1))
    r = AnalysisRun(cfg, "IMAGE_DEMO", ImageSource(other).read(), EXAMPLE_COMMAND)
    res = r.run()
    steps = {s["name"]: s for s in res["steps"]}
    assert "DEMO_INPUT_IMAGE_MISMATCH" in steps["detect_objects"]["error"]
    assert "MAP_CAMERA_MISALIGNED" in steps["build_navigation_map"]["error"]
    assert res["paths"] == {} and res["overall_status"] == "NO_EXECUTABLE_PLAN"


def test_unsupported_command_stops_after_parsing(cfg):
    run, res = run_demo(cfg, "빨간 큐브를 회전시켜")
    assert res["overall_status"] == "TASK_REJECTED" and [s["name"] for s in res["steps"]] == ["parse_task", "save_results"]
    assert res["task"] is None and res["task_parse"]["error_code"] == "UNSUPPORTED_TASK"


def test_cancel_before_start(cfg):
    ev = threading.Event()
    ev.set()
    run, res = run_demo(cfg, cancel=ev)
    assert res["overall_status"] == "CANCELLED" and (run.writer.dir / "result.json").exists()


def test_run_directories_are_unique(cfg):
    a, _ = run_demo(cfg)
    b, _ = run_demo(cfg)
    assert a.writer.dir != b.writer.dir
