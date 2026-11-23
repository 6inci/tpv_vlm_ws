"""Task agent orchestrator.

A Python state machine runs a fixed, whitelisted sequence of tools for each supported task type.
Language/vision models are called only inside specific tools (task parsing, scene analysis,
candidate verification, region choice). They never choose tools, never run code and never
produce coordinates that reach the planner without deterministic checks.
"""
import json
import platform
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .. import visualization
from ..artifacts import RunWriter, load_run
from ..config import code_fingerprint, resolve_path, sha256_bytes, sha256_file, sha256_text
from ..contracts import MANUAL, MANUAL_TEST, OBSERVED, RAW_PX, RESULT_SCHEMA, validate
from ..geometry.calibration import assess_calibration, undistort_image
from ..navigation.goal_selector import accessible_cell_count, select_goal
from ..navigation.map_builder import (MapError, NavMap, check_alignment, dynamic_obstacles_from_detections,
                                      load_map_spec)
from ..navigation.path_validator import compare_raw_and_simplified, validate_path
from ..navigation.planner import plan
from ..perception import grounding, target_inspection
from ..perception.model_client import ModelWorkerError
from ..perception.scene_graph import add_region_relation, build_scene_graph
from ..robot.handoff import TaskExecution, make_request
from ..semantics.region_grounding import ground_loading_region, manual_loading_region
from . import task_parser

TOOLS = ("parse_task", "analyze_scene", "detect_objects", "inspect_target", "build_scene_graph",
         "build_navigation_map", "infer_destination", "select_pickup_pose", "select_dropoff_pose", "plan_path",
         "validate_path", "create_handoff_request", "save_results")
PLANS = {  # the map is built before destination inference so region candidates can be checked for access
    "transport_and_place": list(TOOLS),
    "navigate_to_region": ["parse_task", "analyze_scene", "detect_objects", "build_scene_graph", "build_navigation_map",
                           "infer_destination", "select_dropoff_pose", "plan_path", "validate_path",
                           "create_handoff_request", "save_results"],
}
LABELS = {"parse_task": "명령 해석", "analyze_scene": "장면 이해", "detect_objects": "객체 탐지·검증",
          "inspect_target": "대상(빨간 큐브) 정밀 확인", "build_scene_graph": "공간 관계",
          "infer_destination": "하역 영역 추론", "build_navigation_map": "지도 생성",
          "select_pickup_pose": "픽업 접근 위치", "select_dropoff_pose": "하역 접근 위치", "plan_path": "경로계획",
          "validate_path": "경로 검증", "create_handoff_request": "작업 인계", "save_results": "결과 저장"}
APPEND_ONLY = ("analysis/agent_events.jsonl", "handoff/events.jsonl")
REQUIRES = {"inspect_target": ["detections"], "build_scene_graph": ["detections"], "infer_destination": ["detections"],
            "select_pickup_pose": ["target", "nav", "start"], "select_dropoff_pose": ["region", "nav", "dropoff_origin"],
            "plan_path": ["nav", "any_goal"], "validate_path": ["paths"], "create_handoff_request": ["paths"]}


class Cancelled(Exception):
    pass


class StepFailure(Exception):
    def __init__(self, code, message, status="FAILED"):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.status = code, message, status


class TrackedClient:
    """Proxy around ModelWorkerClient: model-status events + a log line per call."""
    NAMES = {"qwen": "Qwen3-VL", "dino": "Grounding DINO"}

    def __init__(self, client, emit, log):
        self.client, self.emit, self.log = client, emit, log

    @property
    def loaded(self):
        return self.client.loaded

    def ensure_loaded(self, name):
        before = set(self.client.loaded)
        if name not in before:
            self.emit(name, "LOADING")
        info = self.client.ensure_loaded(name)
        for gone in before - set(self.client.loaded):
            self.emit(gone, "UNLOADED (GPU exclusive)")
        self.emit(name, "READY")
        return info

    def call(self, cmd, timeout=None, **args):
        model = {"qwen_generate": "qwen", "dino_detect": "dino"}.get(cmd)
        if model:
            self.emit(model, "RUNNING")
        t0 = time.perf_counter()
        rec = {"t": time.time(), "cmd": cmd, "model": model, "image_path": args.get("image_path"),
               "prompt_sha256": sha256_text(args["prompt"]) if "prompt" in args else None,
               "prompts": args.get("prompts"), "max_new_tokens": args.get("max_new_tokens")}
        try:
            r = self.client.call(cmd, timeout=timeout, **args)
        except ModelWorkerError as e:
            if model:
                self.emit(model, "ERROR")
            self.log({**rec, "ok": False, "error": f"{e.etype}: {e.message}",
                      "wall_s": round(time.perf_counter() - t0, 3)})
            raise
        if model:
            self.emit(model, "READY")
        self.log({**rec, "ok": True, "wall_s": round(time.perf_counter() - t0, 3), "latency_s": r.get("latency_s"),
                  "n_output_tokens": r.get("n_output_tokens"), "hit_max_new_tokens": r.get("hit_max_new_tokens"),
                  "peak_vram_gib": r.get("peak_allocated_gib")})
        return r


class AnalysisRun:
    def __init__(self, cfg, analysis_mode, frame, command, client=None, on_event=None, cancel_event=None,
                 map_spec=None, map_source=None):
        if analysis_mode not in ("LOCAL_ONLY", "LOCAL_VLM_WITH_API_AGENT", "IMAGE_DEMO"):
            raise ValueError(f"unknown analysis mode {analysis_mode}")
        if analysis_mode != "IMAGE_DEMO" and client is None:
            raise ValueError(f"{analysis_mode} needs a model worker client")
        self.cfg, self.mode, self.frame, self.command = cfg, analysis_mode, frame, command.strip()
        self.on_event = on_event or (lambda e: None)
        self.cancel = cancel_event or threading.Event()
        self.writer = RunWriter(resolve_path(cfg["app"]["runs_dir"]))
        self.run_id = self.writer.run_id
        self.raw_client = client
        self.client = TrackedClient(client, self._model_status, self._log_model_call) if client else None
        self.map_source = map_source or cfg["app"]["map_file"]
        self.map_spec = map_spec or load_map_spec(resolve_path(self.map_source))
        self.ctx = {"goals": {}, "paths": {}, "validation": {}}
        self.steps, self.plan = {}, []
        self.result = self.state = None
        self.t0 = time.time()

    # ------------------------------------------------------------ events
    def _emit(self, etype, step=None, **data):
        ev = {"t": round(time.time(), 3), "run_id": self.run_id, "type": etype, "step": step, **data}
        self.writer.append_jsonl("analysis/agent_events.jsonl", ev)
        self.on_event(ev)

    def _model_status(self, model, status):
        self._emit("model_status", None, model=model, status=status)

    def _log_model_call(self, rec):
        self.writer.append_jsonl("analysis/model_calls.jsonl", rec)

    def _info(self, step):
        return lambda level, msg: self._emit("info", step, level=level, message=msg)

    def _rel(self, obj):
        base = str(self.writer.dir) + "/"
        if isinstance(obj, dict):
            return {k: self._rel(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._rel(v) for v in obj]
        if isinstance(obj, str) and obj.startswith(base):
            return obj[len(base):]
        return obj

    # ------------------------------------------------------------ step machinery
    def _set_plan(self, names):
        self.plan = list(names)
        for n in names:
            self.steps.setdefault(n, {"name": n, "label": LABELS[n], "status": "PENDING", "model": None,
                                      "input_summary": None, "output_summary": None, "duration_s": None,
                                      "error": None, "next": None})
        for i, n in enumerate(self.plan):
            self.steps[n]["next"] = self.plan[i + 1] if i + 1 < len(self.plan) else None
        self._emit("plan", None, steps=[dict(self.steps[n]) for n in self.plan])

    def _missing(self, name):
        have = {"detections": "detections" in self.ctx, "target": bool(self.ctx.get("target")),
                "nav": "nav" in self.ctx, "start": "start" in self.ctx, "region": "region" in self.ctx,
                "paths": any(p.get("status") == "PATH_FOUND" for p in self.ctx["paths"].values()),
                "any_goal": any((g or {}).get("pose") for g in self.ctx["goals"].values())}
        transport = self.ctx["task"]["task_type"] == "transport_and_place"
        have["dropoff_origin"] = bool((self.ctx["goals"].get("pickup") or {}).get("pose")) if transport else have["start"]
        return [r for r in REQUIRES.get(name, []) if not have[r]]

    def _run_step(self, name):
        if self.cancel.is_set():
            raise Cancelled()
        rec = self.steps[name]
        rec.update(status="RUNNING", started_at=datetime.now().isoformat(timespec="seconds"))
        self._emit("step_started", name, record=dict(rec))
        t0 = time.perf_counter()
        try:
            out = getattr(self, "_" + name)() or {}
            rec.update(status=out.get("status", "OK"), model=out.get("model"), input_summary=out.get("input"),
                       output_summary=out.get("output"))
        except StepFailure as e:
            rec.update(status=e.status, error=f"{e.code}: {e.message}")
        except ModelWorkerError as e:
            if self.cancel.is_set():
                rec.update(status="CANCELLED", error="stopped by user (worker terminated)")
                raise Cancelled() from None
            rec.update(status="FAILED", error=f"MODEL_{e.etype}: {e.message}")
        except Cancelled:
            rec.update(status="CANCELLED", error="stopped by user")
            raise
        finally:
            rec["duration_s"] = round(time.perf_counter() - t0, 3)
            self._emit("step_finished", name, record=dict(rec))

    # ------------------------------------------------------------ main
    def run(self):
        error = None
        try:
            self._save_input()
            self._set_plan(PLANS["transport_and_place"])
            self._run_step("parse_task")
            if "task" not in self.ctx:
                self._set_plan(["parse_task", "save_results"])
                for n in TOOLS:
                    if n not in ("parse_task", "save_results"):
                        self.steps.pop(n, None)
            else:
                plan_names = PLANS[self.ctx["task"]["task_type"]]
                for n in TOOLS:
                    if n not in plan_names:
                        self.steps.pop(n, None)
                self._set_plan(plan_names)
                for name in plan_names[1:-1]:
                    missing = self._missing(name)
                    if missing:
                        self.steps[name].update(status="SKIPPED", error=f"blocked: missing {', '.join(missing)}")
                        self._emit("step_finished", name, record=dict(self.steps[name]))
                        continue
                    self._run_step(name)
        except Cancelled:
            self.ctx["cancelled"] = True
            for r in self.steps.values():
                if r["status"] in ("PENDING", "RUNNING"):
                    r.update(status="CANCELLED", error=r["error"] or "stopped by user")
        except Exception as e:  # recorded in the run, then re-raised (not hidden)
            error = e
            self.ctx["internal_error"] = traceback.format_exc()
            for r in self.steps.values():
                if r["status"] == "RUNNING":
                    r.update(status="FAILED", error=f"INTERNAL_ERROR: {type(e).__name__}: {e}")
        self.steps.setdefault("save_results", {"name": "save_results", "label": LABELS["save_results"],
                                               "status": "PENDING", "model": None, "input_summary": None,
                                               "output_summary": None, "duration_s": None, "error": None,
                                               "next": None})
        if "save_results" not in self.plan:
            self.plan.append("save_results")
        self.cancel_was_set = self.cancel.is_set()
        self.cancel = threading.Event()  # saving is never cancelled
        self._run_step("save_results")
        self._emit("run_finished", None, overall_status=self.result["overall_status"] if self.result else "SAVE_FAILED",
                   run_dir=str(self.writer.dir))
        if error is not None:
            raise error
        return self.result

    # ------------------------------------------------------------ tools
    def _save_input(self):
        img = self.frame.image
        h, w = img.shape[:2]
        self.writer.write_image("input/raw_frame.png", img)
        self.writer.write_text("input/command.txt", self.command + "\n")
        src = dict(self.frame.source)
        info = {"frame_id": self.frame.frame_id, "timestamp": self.frame.timestamp,
                "timestamp_iso": datetime.fromtimestamp(self.frame.timestamp).isoformat(timespec="milliseconds"),
                "source": src, "width": w, "height": h, "coordinate_frame": RAW_PX,
                "raw_frame_png": "input/raw_frame.png", "decoded_pixels_sha256": sha256_bytes(img.tobytes())}
        if src.get("path") and Path(src["path"]).is_file():
            info["source_sha256"] = sha256_file(src["path"])
        self.ctx["frame_info"] = info
        self.ctx["calibration"] = assess_calibration(self.cfg["camera"], w, h, src.get("camera_model"))
        self.writer.write_json("input/frame.json", info)
        self._emit("run_started", None, frame=info, mode=self.mode, command=self.command,
                   calibration=self.ctx["calibration"])

    def _qwen_text(self, prompt):
        self.client.ensure_loaded("qwen")
        r = self.client.call("qwen_generate", prompt=prompt,
                             max_new_tokens=self.cfg["models"]["qwen"]["max_new_tokens"]["task_parse"])
        self.ctx["task_parse_call"] = {k: r[k] for k in ("model", "commit_hash", "latency_s", "n_output_tokens",
                                                          "hit_max_new_tokens")}
        return r["text"]

    def _parse_task(self):
        mcfg, api = self.cfg["models"], self.cfg["app"]["api_agent"]
        pcfg = mcfg["prompts"]["task_parse"]
        record = {"command": self.command, "mode": self.mode}
        try:
            if self.mode == "IMAGE_DEMO":
                model = "rule_based_demo (keyword rules, not an LLM)"
                task = task_parser.parse_rule_based(self.command)
            else:
                task = None
                if self.mode == "LOCAL_VLM_WITH_API_AGENT":
                    model = f"{api['provider']}:{api['model']}"
                    try:
                        task = task_parser.parse_with_llm(
                            self.command, lambda p: task_parser.openai_compatible_generate(api, p), pcfg,
                            "openai_compatible_api", api["model"])
                    except task_parser.ApiUnavailable as e:
                        record["api_fallback"] = str(e)
                        self._emit("info", "parse_task", level="warning",
                                   message=f"API unavailable ({e}); using the local Qwen3-VL backend")
                if task is None:
                    model = mcfg["qwen"]["model_id"]
                    task = task_parser.parse_with_llm(self.command, self._qwen_text, pcfg, "qwen3_vl_local", model)
        except task_parser.TaskParseError as e:
            record.update(status="FAILED", error_code=e.code, error=e.message, raw_response=e.raw,
                          parsed_output=e.parsed, call=self.ctx.get("task_parse_call"))
            self.writer.write_json("analysis/task_parse.json", record)
            self.ctx["task_parse_error"] = record
            raise StepFailure(e.code, e.message) from None
        record.update(status="OK", task=task, call=self.ctx.get("task_parse_call"))
        self.writer.write_json("analysis/task_parse.json", record)
        self.ctx["task"] = task
        return {"model": model, "input": self.command, "output": task_summary(task)}

    def _analyze_scene(self):
        if self.mode == "IMAGE_DEMO":
            self.ctx["scene_analysis"] = {"status": "NOT_RUN", "reason": "IMAGE_DEMO: no model executed"}
            return {"status": "NOT_RUN", "model": "none", "output": "IMAGE_DEMO: 모델 미실행"}
        res = grounding.analyze_scene(self.client, self.cfg["models"], self.writer.path("input/raw_frame.png"),
                                      task_summary(self.ctx["task"]))
        self.ctx["scene_analysis"] = self._rel(res)
        self.writer.write_json("analysis/scene_analysis.json", self.ctx["scene_analysis"])
        if res["status"] != "OK":
            raise StepFailure("SCENE_ANALYSIS_INVALID", res["call"]["error"])
        o = res["output"]
        return {"model": res["call"]["model"], "input": "full frame",
                "output": f"scene_type={o['scene_type']}; equipment={', '.join(o['observed_equipment'][:6])}; "
                          f"uncertain={len(o['uncertain'])}"}

    def _detect_objects(self):
        task = self.ctx["task"]
        fid = self.frame.frame_id
        if self.mode == "IMAGE_DEMO":
            demo = self._demo_inputs()
            objs = grounding.manual_objects(demo, fid)
            if task["target"] is not None:      # navigate_to_region has no object to carry
                key = target_key(task)
                self.ctx["demo_target"] = next((o for o in objs if o["class_name"] == key), None)
                objs = [o for o in objs if o["class_name"] != key]
            det = {"objects": objs, "mode": "manual_test_data (IMAGE_DEMO)", "unverified_candidates": {}}
            model = "manual_test_data"
        else:
            classes = ["conveyor", "truck", "mobile_robot"] if task["task_type"] == "transport_and_place" else [
                "truck", "mobile_robot"]
            extra = [target_key(task)] if task["task_type"] == "transport_and_place" else []
            det = grounding.detect_reference_objects(self.client, self.cfg["models"], self.frame.image,
                                                     self.writer.path("input/raw_frame.png"), fid, classes, extra,
                                                     self.writer.path("analysis/crops"), self._info("detect_objects"))
            det = self._rel(det)
            model = f"{det['detector']['model']} + {self.cfg['models']['qwen']['model_id']} (verify)"
        self.ctx["detections"] = det
        self._determine_start()
        self.writer.write_json("analysis/detections.json", det)
        acc = [o for o in det["objects"] if o.get("accepted")]
        counts = {}
        for o in acc:
            counts[o["class_name"]] = counts.get(o["class_name"], 0) + 1
        start = self.ctx.get("start")
        return {"model": model, "input": "full frame",
                "output": f"accepted={counts or 'none'}; candidates={len(det['objects'])}; start="
                          f"{start['method'] if start else self.ctx.get('start_error', 'none')}"}

    def _demo_inputs(self):
        path = resolve_path(self.cfg["app"]["demo_inputs_file"])
        demo = json.loads(Path(path).read_text(encoding="utf-8"))
        src_sha = self.ctx["frame_info"].get("source_sha256")
        if src_sha != demo.get("applies_to_image_sha256"):
            raise StepFailure("DEMO_INPUT_IMAGE_MISMATCH",
                              "IMAGE_DEMO manual inputs belong to a different image; nothing is assumed for this frame")
        self.ctx["demo_inputs_file"] = {"path": str(path), "sha256": sha256_file(path)}
        return demo

    def _determine_start(self):
        robots = [o for o in self.ctx["detections"]["objects"] if o["class_name"] == "mobile_robot" and o.get("accepted")]
        policy = self.cfg["planning"]["start_policy"]
        if robots and policy == "detected_else_manual":
            if len(robots) > 1:
                self.ctx["start_error"] = f"START_AMBIGUOUS: {len(robots)} mobile robots accepted"
                return
            b = robots[0]["bbox_xyxy"]
            self.ctx["start"] = {"xy": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], "robot_ref": robots[0]["object_id"],
                                 "method": "detected_robot_bbox_centre (image space)",
                                 "knowledge": robots[0]["knowledge"], "coordinate_frame": RAW_PX}
        elif self.map_spec.get("start_pose"):
            self.ctx["start"] = {"xy": list(self.map_spec["start_pose"]["xy"]), "robot_ref": "mobile_robot_unobserved",
                                 "method": "manual_start_pose_from_map", "knowledge": MANUAL,
                                 "coordinate_frame": RAW_PX, "note": self.map_spec["start_pose"].get("note")}
        else:
            self.ctx["start_error"] = "NO_START: robot not detected and the map has no manual start pose"

    def _inspect_target(self):
        task = self.ctx["task"]
        acc = [o for o in self.ctx["detections"]["objects"] if o.get("accepted")]
        conveyors = [o for o in acc if o["class_name"] == "conveyor"]
        demo = self.mode == "IMAGE_DEMO"
        full = self.ctx["detections"].get("unverified_candidates", {}).get(target_key(task), [])
        res = target_inspection.inspect_target(None if demo else self.client, self.cfg["models"], self.cfg["planning"],
                                               self.frame.image, self.frame.frame_id, task["target"], conveyors, full,
                                               self.writer.path("analysis/roi"), model_mode=not demo,
                                               emit=self._info("inspect_target"))
        if demo:
            res["target"] = self.ctx.get("demo_target")
            res["status"] = "MANUAL_TARGET_IMAGE_DEMO" if res["target"] else "TARGET_NOT_FOUND"
        res = self._rel(res)
        self.ctx["target_inspection"] = res
        self.writer.write_json("analysis/target_inspection.json", res)
        if res["target"]:
            self.ctx["target"] = res["target"]
        model = "OpenCV HSV/shape (+manual target)" if demo else f"HSV/shape + DINO candidates; " \
                                                                  f"{self.cfg['models']['qwen']['model_id']} verify"
        out = (f"ROIs={len(res['rois'])}, candidates={res['n_candidates']}, verified={res['n_verified']}, "
               f"accepted={res['n_accepted']} -> {res['status']}")
        if res["status"] == "TARGET_NOT_FOUND":
            raise StepFailure("TARGET_NOT_FOUND", out)
        if res["status"] == "TARGET_AMBIGUOUS":
            raise StepFailure("TARGET_AMBIGUOUS", out + f" {res.get('ambiguous_candidates')}", status="UNCERTAIN")
        return {"model": model, "input": f"{len(conveyors)} verified conveyor ROI(s)", "output": out}

    def _build_scene_graph(self):
        acc = [o for o in self.ctx["detections"]["objects"] if o.get("accepted")]
        g = build_scene_graph(acc, self.ctx.get("target"), self.ctx["task"], self.ctx.get("start"), self.cfg["planning"])
        self.ctx["scene_graph"] = g
        rels = [f"{r['subject']} {r['type']} {r['object']} [{r['status']}]" for r in g["relations"]]
        return {"model": "geometry (image px)", "input": f"{len(g['nodes'])} nodes",
                "output": "; ".join(rels) or "no relations"}

    def _infer_destination(self):
        acc = [o for o in self.ctx["detections"]["objects"] if o.get("accepted")]
        trucks = [o for o in acc if o["class_name"] == "truck"]
        if not trucks:
            res = {"status": "FAILED_NO_TRUCK", "reason": "no truck accepted in this frame"}
            self.ctx["region_grounding"] = res
            self.writer.write_json("analysis/semantic_regions.json", res)
            raise StepFailure("NO_TRUCK", "destination inference needs a truck; none was accepted in this frame")
        if len(trucks) > 1:
            res = {"status": "TRUCK_AMBIGUOUS", "trucks": [t["object_id"] for t in trucks]}
            self.ctx["region_grounding"] = res
            self.writer.write_json("analysis/semantic_regions.json", res)
            raise StepFailure("TRUCK_AMBIGUOUS", f"{len(trucks)} trucks accepted", status="UNCERTAIN")
        truck = trucks[0]
        if self.mode == "IMAGE_DEMO":
            res = manual_loading_region(self._demo_inputs(), truck["object_id"])
            model = "manual_test_data"
        else:
            nav = self.ctx.get("nav")
            gcfg = self.cfg["planning"]["goal_selection"]["dropoff"]
            access = None if nav is None else (lambda poly: accessible_cell_count(nav, poly, gcfg))
            res = ground_loading_region(self.client, self.cfg["models"], self.cfg["planning"], self.frame.image, truck,
                                        self.writer.path("analysis"), self.map_spec.get("semantic_reference_zones", []),
                                        access)
            model = self.cfg["models"]["qwen"]["model_id"] + " (plausibility) + map accessibility"
        res = self._rel(res)
        self.ctx["region_grounding"] = res
        self.writer.write_json("analysis/semantic_regions.json", res)
        reg = res.get("selected_loading_region")
        if not reg:
            raise StepFailure(res["status"], res.get("reason", ""),
                              status="FAILED" if res["status"] == "SEMANTIC_REGION_INACCESSIBLE" else "UNCERTAIN")
        self.ctx["region"] = reg
        if "scene_graph" in self.ctx:
            add_region_relation(self.ctx["scene_graph"], truck["object_id"], reg,
                                "MANUAL" if reg["verification_status"] == "manual_unconfirmed" else "INFERRED")
        return {"model": model, "input": f"{truck['object_id']} + {len(res['loading_area_candidates'])} candidates",
                "output": f"{res['status']}: {reg['region_id']} ({reg['verification_status']})"}

    def _build_navigation_map(self):
        img = self.frame.image
        h, w = img.shape[:2]
        align = check_alignment(self.map_spec, img, resolve_path)
        rec = {"spec": self.map_spec, "source_file": str(self.map_source),
               "spec_sha256": sha256_text(json.dumps(self.map_spec, sort_keys=True)), "alignment": align,
               "build_params": {k: self.cfg["planning"][k] for k in ("cell_px", "robot_radius_px", "safety_margin_px")}}
        self.ctx["map_record"] = rec
        if align["status"] == "MISALIGNED":
            self.writer.write_json("planning/map.json", rec)
            raise StepFailure("MAP_CAMERA_MISALIGNED", f"frame does not match the map reference image: {align}")
        dyn = dynamic_obstacles_from_detections(self.ctx.get("detections", {}).get("objects", []),
                                                self.cfg["planning"]["dynamic_obstacle_classes"], self.map_spec, (w, h))
        try:
            nav = NavMap(self.map_spec, (w, h), self.cfg["planning"], dyn)
        except MapError as e:
            self.writer.write_json("planning/map.json", rec)
            raise StepFailure(e.code, str(e)) from None
        self.ctx["nav"] = nav
        rec["summary"] = nav.summary()
        self.writer.write_json("planning/map.json", rec)
        s = rec["summary"]["cell_counts"]
        return {"model": "map_builder (manual map prior + detections)", "input": rec["spec"]["map_id"],
                "output": f"alignment={align['status']}; valid cells={s['valid_robot_center']}/"
                          f"{nav.rows * nav.cols}; dynamic obstacles={len(dyn)}; review={self.map_spec['review_status']}"}

    def _select_pickup_pose(self):
        t = self.ctx["target"]
        b = t["bbox_xyxy"]
        poly = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        g = select_goal(self.ctx["nav"], t["object_id"], poly, self.ctx["start"]["xy"], "pickup_approach",
                        self.cfg["planning"]["goal_selection"]["pickup"], self.cfg["planning"])
        self.ctx["goals"]["pickup"] = g
        if g["status"] != "GOAL_SELECTED":
            raise StepFailure(g["status"], g["reason"])
        return {"model": "goal_selector (Dijkstra cost + distance)", "input": t["object_id"],
                "output": f"pose={[round(v) for v in g['pose']['xy']]} px, cost={g['pose']['path_cost']}, "
                          f"reachable cells={g['stats']['cells_reachable_from_start']}"}

    def _select_dropoff_pose(self):
        reg = self.ctx["region"]
        transport = self.ctx["task"]["task_type"] == "transport_and_place"
        origin = self.ctx["goals"]["pickup"]["pose"]["xy"] if transport else self.ctx["start"]["xy"]
        g = select_goal(self.ctx["nav"], reg["region_id"], reg["polygon_xy"], origin, "dropoff_approach",
                        self.cfg["planning"]["goal_selection"]["dropoff"], self.cfg["planning"])
        self.ctx["goals"]["dropoff"] = g
        if g["status"] != "GOAL_SELECTED":
            raise StepFailure(g["status"], g["reason"])
        return {"model": "goal_selector (Dijkstra cost + distance)", "input": reg["region_id"],
                "output": f"pose={[round(v) for v in g['pose']['xy']]} px, cost={g['pose']['path_cost']}"}

    def _plan_path(self):
        nav, pcfg = self.ctx["nav"], self.cfg["planning"]
        start = self.ctx.get("start", {}).get("xy")
        pick = (self.ctx["goals"].get("pickup") or {}).get("pose")
        drop = (self.ctx["goals"].get("dropoff") or {}).get("pose")
        paths = self.ctx["paths"]
        if self.ctx["task"]["task_type"] == "transport_and_place":
            if start and pick:
                paths["A"] = plan(nav, start, pick["xy"], pcfg, "A_start_to_pickup")
            if pick and drop:
                paths["B"] = plan(nav, pick["xy"], drop["xy"], pcfg, "B_pickup_to_dropoff")
                if paths["B"]["status"] == "PATH_FOUND":
                    paths["B"]["execution_note"] = "planned assuming pickup; becomes active only after confirmed pickup"
        elif start and drop:
            paths["A"] = plan(nav, start, drop["xy"], pcfg, "A_start_to_goal")
        for k, rel in (("A", "planning/pickup_path.json"), ("B", "planning/delivery_path.json")):
            if k in paths:
                self.writer.write_json(rel, paths[k])
        bad = {k: p["status"] for k, p in paths.items() if p["status"] != "PATH_FOUND"}
        summary = "; ".join(f"{k}: {p['status']} len={p['length_px']}px wp={p.get('n_waypoints')} "
                            f"(raw {p.get('n_raw_waypoints')})" for k, p in paths.items())
        if bad or not paths:
            raise StepFailure("PATH_NOT_FOUND", summary or "no path requested")
        return {"model": f"{pcfg['planner']} + line-of-sight simplification", "input": f"{len(paths)} path(s)",
                "output": summary}

    def _validate_path(self):
        nav, pcfg = self.ctx["nav"], self.cfg["planning"]
        out = {}
        for k, p in self.ctx["paths"].items():
            if p["status"] != "PATH_FOUND":
                continue
            v = validate_path(nav, p["waypoints_xy"], p["start_xy"], p["goal_xy"], pcfg)
            v["raw_path_check"] = validate_path(nav, p["raw_waypoints_xy"], p["start_xy"], p["goal_xy"], pcfg)["valid"]
            v["raw_vs_simplified"] = compare_raw_and_simplified(p)
            out[k] = v
            if not v["valid"]:
                p["status"] = "REJECTED_BY_VALIDATOR"
        self.ctx["validation"] = out
        self.writer.write_json("planning/validation.json", out)
        bad = [k for k, v in out.items() if not v["valid"]]
        summary = "; ".join(f"{k}: {'valid' if v['valid'] else 'INVALID'} min clearance {v['min_clearance_px']}px"
                            for k, v in out.items())
        if bad:
            raise StepFailure("PATH_INVALID", summary)
        return {"model": "pixel-level clearance validator", "input": f"{len(out)} path(s)", "output": summary}

    def _create_handoff_request(self):
        task, mm = self.ctx["task"], self.ctx["calibration"]["map_mode"]
        paths, val = self.ctx["paths"], self.ctx["validation"]
        ok = {k: paths.get(k, {}).get("status") == "PATH_FOUND" and val.get(k, {}).get("valid") for k in ("A", "B")}
        reqs = {}
        if task["task_type"] == "transport_and_place":
            t = self.ctx["target"]
            tinfo = {"object_id": t["object_id"], "class_name": task["target"]["class"], "color": task["target"]["color"]}
            if ok["A"]:
                reqs["pickup"] = make_request("PICKUP_HANDOFF", tinfo, self.ctx["goals"]["pickup"]["pose"], mm,
                                              "WAITING_FOR_LOCAL_CONTROLLER")
            if ok["B"]:
                reqs["placement"] = make_request("PLACEMENT_HANDOFF", tinfo, self.ctx["goals"]["dropoff"]["pose"], mm,
                                                 "BLOCKED_UNTIL_PICKUP_CONFIRMED")
            state = "WAITING_FOR_PICKUP_CONFIRMATION" if ok["A"] and ok["B"] else "NOT_EXECUTABLE"
            exe = TaskExecution(task["task_type"], reqs, state, "PLANNED_ASSUMING_PICKUP" if ok["B"] else None,
                                hardware_enabled=self.cfg["app"]["handoff"]["hardware_enabled"])
        else:
            if ok["A"]:
                reqs["navigation"] = make_request("NAVIGATION_GOAL", None, self.ctx["goals"]["dropoff"]["pose"], mm,
                                                  "WAITING_FOR_LOCAL_CONTROLLER")
            exe = TaskExecution(task["task_type"], reqs,
                                "WAITING_FOR_ARRIVAL_CONFIRMATION" if ok["A"] else "NOT_EXECUTABLE",
                                hardware_enabled=self.cfg["app"]["handoff"]["hardware_enabled"])
        self.ctx["execution"] = exe
        for k, r in reqs.items():
            self.writer.write_json(f"handoff/{k}_request.json", r)
        self.writer.write_json("handoff/state_001.json", exe.to_dict())
        return {"model": "handoff (simulation; hardware disabled)", "input": ", ".join(reqs) or "none",
                "output": f"state={exe.state}; path B={exe.path_b_status}",
                "status": "OK" if exe.state != "NOT_EXECUTABLE" else "UNCERTAIN"}

    def _save_results(self):
        # result.json is written inside this step; record its outcome first (an exception below still
        # propagates and marks the in-memory step as failed).
        self.steps["save_results"].update(status="OK", model="artifacts", input_summary=f"run {self.run_id}",
                                          output_summary=f"{self.writer.dir} (result.json, manifest.json, figures)")
        result = self._assemble_result()
        state = {"frame": self.frame.image, "result": result, "nav": self.ctx.get("nav"),
                 "roi_images": load_roi_images(self.writer.dir, result), "undistorted": None}
        calib = self.ctx["calibration"]
        if calib["undistort_view_available"]:
            state["undistorted"] = undistort_image(self.frame.image, calib["intrinsics"])
            self.writer.write_image("analysis/undistorted_view.png", state["undistorted"])
        result["figures"] = self._rel(visualization.export_paper_figures(state, "figures", self.writer))
        validate(result, RESULT_SCHEMA)
        self.writer.write_json("result.json", result)
        self.writer.write_json("manifest.json", self._manifest(result))
        self.result, self.state = result, state
        return {"model": "artifacts", "input": f"run {self.run_id}",
                "output": f"{self.writer.dir} ({len(result['figures'])} figures)"}

    # ------------------------------------------------------------ result / manifest
    def _overall(self):
        if getattr(self, "cancel_was_set", False):
            return "CANCELLED"
        if "internal_error" in self.ctx:
            return "INTERNAL_ERROR"
        if "task" not in self.ctx:
            return "TASK_REJECTED"
        exe = self.ctx.get("execution")
        if exe and exe.state != "NOT_EXECUTABLE":
            return "PLAN_COMPLETE"
        if any(p.get("status") == "PATH_FOUND" for p in self.ctx["paths"].values()):
            return "PARTIAL_PLAN"
        return "NO_EXECUTABLE_PLAN"

    def _manual_inputs(self):
        out = [{"item": "navigation map", "source": str(self.map_source), "review_status": self.map_spec["review_status"],
                "knowledge": MANUAL}]
        if (self.ctx.get("start") or {}).get("knowledge") == MANUAL:
            out.append({"item": "robot start pose", "source": "map.start_pose", "knowledge": MANUAL})
        if self.mode == "IMAGE_DEMO":
            out.append({"item": "objects / target / loading region", "source": self.ctx.get("demo_inputs_file"),
                        "knowledge": MANUAL_TEST})
            out.append({"item": "task parse", "source": "rule_based_demo keyword rules", "knowledge": "rule_based"})
        out.append({"item": "robot footprint, reach distances", "source": "config/planning.yaml (image-space assumptions)",
                    "knowledge": MANUAL})
        return out

    def _assemble_result(self):
        ctx = self.ctx
        task = ctx.get("task")
        steps = [self.steps[n] for n in self.plan if n in self.steps]
        overall = self._overall()
        ti, rg = ctx.get("target_inspection"), ctx.get("region_grounding")
        exe = ctx.get("execution")
        demo = self.mode == "IMAGE_DEMO"
        manual = self._manual_inputs()
        val = ctx.get("validation", {})
        evaluation = {
            "note": "automatic statuses only; accuracy needs ground truth (scripts/evaluate_run.py). "
                    "Planning success is not robot execution success.",
            "task_parsing": {"backend": (task or {}).get("parser", {}).get("backend")
                             or (ctx.get("task_parse_error") or {}).get("mode"),
                             "status": "OK" if task else (ctx.get("task_parse_error") or {}).get("error_code"),
                             "input_provenance": "rule_based" if demo else "model", "ground_truth": None},
            "object_grounding": {"accepted": [o["object_id"] for o in (ctx.get("detections") or {}).get("objects", [])
                                              if o.get("accepted")],
                                 "target_status": (ti or {}).get("status"),
                                 "input_provenance": MANUAL_TEST if demo else "model", "ground_truth": None},
            "semantic_grounding": {"status": (rg or {}).get("status"),
                                   "selected": ((rg or {}).get("selected_loading_region") or {}).get("region_id"),
                                   "prior_zone_consistency": (rg or {}).get("prior_zone_consistency"),
                                   "input_provenance": MANUAL_TEST if demo else "model", "ground_truth": None},
            "goal_selection": {k: (ctx["goals"].get(k) or {}).get("status") for k in ("pickup", "dropoff")},
            "path_planning": {k: {"status": p["status"], "valid": (val.get(k) or {}).get("valid"),
                                  "length_px": p["length_px"]} for k, p in ctx["paths"].items()},
            "end_to_end_planning": {"status": overall, "uses_manual_inputs": [m["item"] for m in manual],
                                    "automatic_perception": not demo,
                                    "counts_as_automatic_e2e": (not demo) and overall == "PLAN_COMPLETE"},
        }
        reasons = [f"{s['name']}: {s['error']}" for s in steps if s.get("error")]
        nav = ctx.get("nav")
        result = {
            "schema_version": "stnav-result-0.1", "run_id": self.run_id, "analysis_mode": self.mode,
            "map_mode": ctx["calibration"]["map_mode"], "metric_output_enabled": False, "command": self.command,
            "frame": ctx["frame_info"], "calibration": ctx["calibration"], "steps": steps,
            "overall_status": overall, "status_reasons": reasons,
            "status_note": "PLAN_COMPLETE = both approach poses and paths exist and passed validation in image space; "
                           "it does not mean the robot moved or grasped anything.",
            "task": task, "task_parse": ctx.get("task_parse_error"),
            "scene_analysis": ctx.get("scene_analysis"), "detections": ctx.get("detections"),
            "target_inspection": ti, "start": ctx.get("start"), "start_error": ctx.get("start_error"),
            "scene_graph": ctx.get("scene_graph"), "region_grounding": rg,
            "map": ctx.get("map_record") and {k: v for k, v in ctx["map_record"].items()},
            "goals": ctx["goals"], "paths": ctx["paths"], "validation": val,
            "handoff": exe.to_dict() if exe else None, "evaluation_record": evaluation,
            "provenance": {"manual_inputs": manual,
                           "auto_inferred": [] if demo else ["task parse (LLM)", "scene description (VLM)",
                                                             "object candidates (detector) + verification (VLM)",
                                                             "target selection", "loading-area choice (VLM)"],
                           "deterministic": ["coordinate transforms", "relations", "map", "goal selection",
                                             "A* + simplification", "validation"]},
            "timing": {"total_s": round(time.time() - self.t0, 3),
                       "per_step_s": {s["name"]: s["duration_s"] for s in steps}},
            "internal_error": ctx.get("internal_error"),
        }
        if nav is not None:
            result["map"]["summary"] = nav.summary()
        return result

    def _manifest(self, result):
        import PIL
        mcfg = self.cfg["models"]
        calls = []
        p = self.writer.path("analysis/model_calls.jsonl")
        if p.exists():
            calls = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        models = {"used": self.mode != "IMAGE_DEMO", "worker_env": getattr(self.raw_client, "env_info", None),
                  "worker_log": str(getattr(self.raw_client, "log_path", None)),
                  "qwen": {"model_id": mcfg["qwen"]["model_id"], "requested_revision": mcfg["qwen"]["revision"]},
                  "grounding_dino": {"model_id": mcfg["grounding_dino"]["model_id"],
                                     "requested_revision": mcfg["grounding_dino"]["revision"]},
                  "commit_hashes_seen": sorted({h for h in _find_values(result, "commit_hash") if h}),
                  "loaded_at_end": dict(getattr(self.raw_client, "loaded", {}) or {}),
                  "gpu_swaps_total_in_worker": len(getattr(self.raw_client, "swaps", []) or []),
                  "n_calls": len(calls), "peak_vram_gib_max": max([c.get("peak_vram_gib") or 0 for c in calls] or [0])}
        return {
            "run_id": self.run_id, "created": datetime.fromtimestamp(self.t0).isoformat(timespec="seconds"),
            "finished": datetime.now().isoformat(timespec="seconds"), "overall_status": result["overall_status"],
            "analysis_mode": self.mode, "map_mode": result["map_mode"], "command": self.command,
            "input": {k: self.ctx["frame_info"].get(k) for k in ("source", "source_sha256", "decoded_pixels_sha256",
                                                                  "raw_frame_png", "frame_id", "timestamp_iso",
                                                                  "width", "height")},
            "code": code_fingerprint(),
            "software": {"python": sys.executable, "python_version": platform.python_version(),
                         "numpy": np.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__},
            "models": models,
            "prompts": {k: {"id": v["id"], "template_sha256": sha256_text(v["text"])} for k, v in mcfg["prompts"].items()},
            "inference_params": {"qwen": {"dtype": mcfg["qwen"]["dtype"], "do_sample": mcfg["qwen"]["do_sample"],
                                          "max_new_tokens": mcfg["qwen"]["max_new_tokens"]},
                                 "grounding_dino": {k: mcfg["grounding_dino"][k] for k in
                                                    ("box_threshold", "text_threshold", "nms_iou", "max_area_frac",
                                                     "class_prompts", "verify_top_k")},
                                 "target_inspection": mcfg["target_inspection"], "gpu_residency":
                                     mcfg["worker"].get("gpu_residency")},
            "camera_calibration": {k: result["calibration"].get(k) for k in
                                   ("map_mode", "metric_output_enabled", "undistort_view_available", "checks",
                                    "reasons", "homography_status")} | {
                "intrinsics_file": (result["calibration"].get("intrinsics") or {}).get("file"),
                "intrinsics_sha256": (result["calibration"].get("intrinsics") or {}).get("sha256")},
            "map": {"file": str(self.map_source), "map_id": self.map_spec["map_id"],
                    "review_status": self.map_spec["review_status"],
                    "spec_sha256": (self.ctx.get("map_record") or {}).get("spec_sha256"),
                    "alignment": (self.ctx.get("map_record") or {}).get("alignment")},
            "coordinate_frames": {"detections": RAW_PX, "roi": "roi_local_px (raw = offset + roi/scale)",
                                  "planning": f"grid_row_col with cell_px={self.cfg['planning']['cell_px']} over {RAW_PX}",
                                  "ground_map_xy": "disabled (no validated homography)"},
            "planning": {k: self.cfg["planning"][k] for k in ("planner", "simplification", "cell_px", "robot_radius_px",
                                                               "safety_margin_px", "safety_weight",
                                                               "validation_tolerance_px", "goal_selection")},
            "manual_inputs": result["provenance"]["manual_inputs"],
            "auto_inferred": result["provenance"]["auto_inferred"],
            "timing": result["timing"],
            "config_snapshot": self.cfg,
            "files_sha256": self.writer.file_hashes(exclude=APPEND_ONLY + ("manifest.json",)),
            "append_only_logs": {"files": list(APPEND_ONLY),
                                 "note": "appended after the manifest (run_finished, simulated hand-off events); not hashed"},
        }


def _find_values(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _find_values(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _find_values(v, key)


def target_key(task):
    return f"{task['target']['color']}_{task['target']['class']}"


def task_summary(task):
    t, d = task["target"], task["destination"]
    tgt = "none" if t is None else (f"{t['color']} {t['class']}" + (f" ({t['source_relation']} {t['source_reference']})"
                                                                     if t["source_reference"] else ""))
    return f"{task['task_type']}: target={tgt}; destination={d['region_type']} near {d['reference_object']}"


def load_roi_images(run_dir, result):
    out = {}
    for r in ((result.get("target_inspection") or {}).get("rois") or []):
        p = Path(run_dir) / r["zoom_path"]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            out[r["roi_index"]] = img
    return out


def replay_state(run_dir):
    """Rebuild the exact render state of a saved run (map rebuilt from the stored spec + parameters)."""
    run = load_run(run_dir)
    result = run["result"]
    nav = None
    m = result.get("map") or {}
    if m.get("summary"):
        h, w = run["frame"].shape[:2]
        nav = NavMap(m["spec"], (w, h), m["build_params"], m["summary"]["dynamic_obstacles"])
        if nav.summary()["layer_sha256"] != m["summary"]["layer_sha256"]:
            raise RuntimeError("replayed map layers differ from the saved run (code or spec changed)")
    und_path = Path(run_dir) / "analysis" / "undistorted_view.png"
    und = cv2.imread(str(und_path), cv2.IMREAD_COLOR) if und_path.exists() else None
    state = {"frame": run["frame"], "result": result, "nav": nav, "roi_images": load_roi_images(run_dir, result),
             "undistorted": und}
    return state, run


def record_handoff_event(run_dir, event, source, note=None):
    """Advance the simulated execution state of a saved run; appends a new numbered snapshot."""
    writer = RunWriter(run_dir, existing=True)
    files = sorted(Path(run_dir, "handoff").glob("state_*.json"))
    if not files:
        raise RuntimeError("run has no handoff state (no executable plan)")
    exe = TaskExecution.from_dict(json.loads(files[-1].read_text(encoding="utf-8")))
    exe.report(event, source, note)
    writer.write_json(writer.next_numbered("handoff", "state", ".json"), exe.to_dict())
    writer.append_jsonl("handoff/events.jsonl", {"t": time.time(), "event": event, "source": source, "note": note,
                                                 "state": exe.state})
    return exe
