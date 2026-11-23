#!/usr/bin/env python3
"""Qwen3-VL semantic-map diagnostic: ONE image, ONE new prompt (sm0), ONE inference. Exploratory; no accuracy is computed.

Reuses the existing model worker (semantic_task_nav.perception.model_client / model_worker, unchanged) and only its
`qwen_generate` command: no detector, no other model, no verifier, no planner is called. The prompt is frozen before the run
and compared by hash. The model receives one image path and one prompt text.

The reply is treated as uncertain input: the raw text is kept; a strict schema check lists every violation; objects/areas that
are usable for drawing are extracted WITHOUT correcting anything (no box is moved, no label changed, no entry added). If the
reply is truncated or not valid JSON, the longest complete prefix may be recovered offline for inspection and is then labelled
SALVAGED_PARTIAL (it is not model output beyond the cut).

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  PYTHONDONTWRITEBYTECODE=1 $PY experiments/qwen_semantic_map_diagnostic/semantic_map_probe.py prompts [--show]
  PYTHONDONTWRITEBYTECODE=1 $PY experiments/qwen_semantic_map_diagnostic/semantic_map_probe.py freeze
  PYTHONDONTWRITEBYTECODE=1 $PY experiments/qwen_semantic_map_diagnostic/semantic_map_probe.py run
"""
import argparse
import json
import platform
import re
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from semantic_task_nav.config import code_fingerprint, load_config, sha256_file, sha256_text  # noqa: E402
from semantic_task_nav.contracts import SchemaError, parse_json_object, validate  # noqa: E402

FROZEN_FILE = HERE / "prompt_frozen.json"
MOBILITY = ["mobile_robot", "other_vehicle", "static_equipment_or_goods", "fixed_structure", "uncertain"]
LEVEL = ["low", "medium", "high"]
LIMITS = {"objects": 40, "relations": 25, "semantic_areas": 10}

_BBOX = {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4}
_STRS = {"type": "array", "items": {"type": "string"}}
SCHEMA = {
    "type": "object",
    "required": ["scene_summary", "objects", "relations", "semantic_areas", "destination", "unverifiable_or_unclear"],
    "properties": {
        "scene_summary": {"type": "string"},
        "objects": {"type": "array", "items": {
            "type": "object", "required": ["id", "name", "bbox_2d", "mobility", "mobility_evidence", "estimated_function",
                                            "observed_evidence", "uncertainty", "uncertainty_note"],
            "properties": {"id": {"type": "string", "minLength": 1}, "name": {"type": "string", "minLength": 1}, "bbox_2d": _BBOX,
                           "mobility": {"type": "string", "enum": MOBILITY}, "mobility_evidence": {"type": "string"},
                           "estimated_function": {"type": "string"}, "observed_evidence": _STRS,
                           "uncertainty": {"type": "string", "enum": LEVEL}, "uncertainty_note": {"type": "string"}}}},
        "relations": {"type": "array", "items": {
            "type": "object", "required": ["id", "subject", "relation", "object", "basis", "evidence"],
            "properties": {"id": {"type": "string"}, "subject": {"type": "string"}, "relation": {"type": "string"},
                           "object": {"type": "string"}, "basis": {"type": "string", "enum": ["observed", "inferred"]},
                           "evidence": {"type": "string"}}}},
        "semantic_areas": {"type": "array", "items": {
            "type": "object", "required": ["id", "function_name", "bbox_2d", "related_objects", "function_evidence", "basis",
                                            "uncertainty", "uncertainty_note"],
            "properties": {"id": {"type": "string", "minLength": 1}, "function_name": {"type": "string", "minLength": 1},
                           "bbox_2d": _BBOX, "related_objects": _STRS, "function_evidence": _STRS,
                           "basis": {"type": "string", "enum": ["observed", "inferred"]},
                           "uncertainty": {"type": "string", "enum": LEVEL}, "uncertainty_note": {"type": "string"}}}},
        "destination": {"type": "object", "required": ["interpreted_purpose", "decision", "proposed_area_id", "selection_evidence",
                                                       "related_objects", "related_areas", "unknowns"],
                        "properties": {"interpreted_purpose": {"type": "string"},
                                       "decision": {"type": "string", "enum": ["proposed", "withheld"]},
                                       "proposed_area_id": {"type": ["string", "null"]}, "selection_evidence": {"type": "string"},
                                       "related_objects": _STRS, "related_areas": _STRS, "unknowns": _STRS}},
        "unverifiable_or_unclear": _STRS,
    },
}


# ------------------------------------------------------------------ prompt
def load_all():
    cfg = yaml.safe_load(open(HERE / "config.yaml", encoding="utf-8"))
    prompts = yaml.safe_load(open(HERE / "prompts.yaml", encoding="utf-8"))
    return cfg, prompts


def render_prompt(prompts, instruction):
    return "\n\n".join([prompts["head"].replace("{instruction}", instruction).strip("\n"), prompts["tail"].strip("\n")]) + "\n"


def check_frozen(cfg, prompts):
    if not FROZEN_FILE.exists():
        return False, "prompt_frozen.json is missing; run `semantic_map_probe.py freeze` first"
    fr = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))
    cur = sha256_text(render_prompt(prompts, cfg["instruction"]))
    if fr["prompt_sha256"] != cur or fr["instruction"] != cfg["instruction"]:
        return False, "rendered prompt differs from the frozen prompt"
    return True, f"frozen prompt matches ({fr['prompt_version']}, frozen_at {fr['frozen_at']})"


# ------------------------------------------------------------------ geometry
def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def containment(inner, outer):
    ix, iy = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0])), max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / a if a > 0 else 0.0


def norm_to_px(b, w, h):
    return [b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h]


# ------------------------------------------------------------------ parsing a possibly broken reply
def _closers(prefix):
    """Closing brackets needed to complete `prefix`; None if the prefix ends inside a string or is unbalanced."""
    stack, in_str, esc = [], False, False
    for ch in prefix:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
    return None if in_str else "".join(reversed(stack))


def salvage_prefix(text):
    """Longest complete prefix of a truncated JSON reply, closed offline. Returns (obj, info) or (None, info)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
    cuts = [m.end() for m in re.finditer(r"[}\]]", t)]
    for end in reversed(cuts[-3000:]):
        prefix = t[:end]
        cl = _closers(prefix)
        if cl is None:
            continue
        try:
            obj = json.loads(prefix + cl)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj, {"kept_chars": end, "total_chars": len(t), "closers_added": cl,
                         "note": "text after the cut is NOT part of the recovered structure"}
    return None, {"kept_chars": 0, "total_chars": len(t)}


def parse_reply(text):
    """(obj | None, parse_status, info, schema_errors)"""
    try:
        obj = parse_json_object(text)
        status, info = "STRICT_JSON", {}
    except (json.JSONDecodeError, SchemaError) as e:
        obj, info = salvage_prefix(text)
        status = "SALVAGED_PARTIAL" if obj is not None else "UNPARSEABLE"
        info = {**info, "json_error": str(e)[:300]}
    errors = []
    if obj is not None:
        try:
            validate(obj, SCHEMA)
        except SchemaError as e:
            errors = e.errors
    return obj, status, info, errors


def usable(obj, size=(1920, 1080)):
    """Entries that can be drawn: id + name + a 4-number box in range. Nothing is corrected; the rest is listed as dropped."""
    out, dropped = {"objects": [], "semantic_areas": [], "relations": []}, []

    def ok_box(b):
        return (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in b)
                and 0 <= min(b) and max(b) <= 1000 and b[0] < b[2] and b[1] < b[3])
    for key, namekey in (("objects", "name"), ("semantic_areas", "function_name")):
        for i, e in enumerate((obj or {}).get(key) or []):
            if isinstance(e, dict) and isinstance(e.get("id"), str) and isinstance(e.get(namekey), str) and ok_box(e.get("bbox_2d")):
                out[key].append(e)
            else:
                dropped.append({"section": key, "index": i, "entry": e})
    out["relations"] = [r for r in ((obj or {}).get("relations") or []) if isinstance(r, dict)]
    return out, dropped


def consistency(obj, size=(1920, 1080)):
    """Descriptive self-consistency of the model's own output (its boxes against each other). No scene knowledge is used."""
    u, dropped = usable(obj, size)
    objs, areas = u["objects"], u["semantic_areas"]
    c = {"counts": {k: len((obj or {}).get(k) or []) for k in ("objects", "relations", "semantic_areas")},
         "over_limit": {k: len((obj or {}).get(k) or []) for k in LIMITS if len((obj or {}).get(k) or []) > LIMITS[k]},
         "entries_not_drawable": len(dropped)}
    for key, items in (("objects", objs), ("semantic_areas", areas), ("relations", u["relations"])):
        ids = [e.get("id") for e in items]
        c[f"duplicate_{key}_ids"] = sorted({i for i in ids if ids.count(i) > 1})
    obj_ids, area_ids = {o["id"] for o in objs}, {a["id"] for a in areas}
    c["relations_referencing_missing_ids"] = [r.get("id") for r in u["relations"]
                                              if r.get("subject") not in obj_ids | area_ids or r.get("object") not in obj_ids | area_ids]
    c["area_related_objects_missing"] = {a["id"]: [i for i in a.get("related_objects", []) if i not in obj_ids] for a in areas
                                         if any(i not in obj_ids for i in a.get("related_objects", []))}
    c["near_identical_object_boxes"] = [[a["id"], b["id"], round(iou(a["bbox_2d"], b["bbox_2d"]), 3)] for i, a in enumerate(objs)
                                        for b in objs[i + 1:] if iou(a["bbox_2d"], b["bbox_2d"]) >= 0.9]
    c["area_box_copies_an_object_box"] = [[a["id"], o["id"], round(iou(a["bbox_2d"], o["bbox_2d"]), 3)] for a in areas for o in objs
                                          if iou(a["bbox_2d"], o["bbox_2d"]) >= 0.8]
    d = (obj or {}).get("destination")
    if isinstance(d, dict):
        c["destination"] = {"decision": d.get("decision"), "proposed_area_id": d.get("proposed_area_id"),
                            "proposed_area_exists": d.get("proposed_area_id") in area_ids,
                            "decision_consistent": (d.get("decision") == "proposed") == bool(d.get("proposed_area_id")),
                            "related_objects_missing": [i for i in d.get("related_objects", []) if i not in obj_ids],
                            "related_areas_missing": [i for i in d.get("related_areas", []) if i not in area_ids]}
    else:
        c["destination"] = None
    robots = [o for o in objs if o.get("mobility") == "mobile_robot"]
    c["mobile_robot_entries"] = [o["id"] for o in robots]
    c["boxes_of_mobile_robot_entries_containing_other_objects"] = {
        r["id"]: [o["id"] for o in objs if o["id"] != r["id"] and containment(o["bbox_2d"], r["bbox_2d"]) >= 0.5] for r in robots}
    c["mobility_counts"] = {m: sum(1 for o in objs if o.get("mobility") == m) for m in MOBILITY}
    return c, dropped


# ------------------------------------------------------------------ CLI
def v02_fingerprint():
    cfg_hash = {str(p.relative_to(PROJECT)): sha256_file(p) for p in sorted((PROJECT / "config").rglob("*")) if p.is_file()}
    return {"src_and_run_py": code_fingerprint()["sha256"], "config_files": cfg_hash}


def cmd_prompts(args):
    cfg, prompts = load_all()
    p = render_prompt(prompts, cfg["instruction"])
    ok, msg = check_frozen(cfg, prompts)
    print(f"prompt_version {prompts['prompt_version']} | {len(p)} chars | sha256 {sha256_text(p)[:16]} | frozen: {'YES' if ok else 'NO'} ({msg})")
    if args.show:
        print("\n" + p)


def cmd_freeze(args):
    cfg, prompts = load_all()
    if FROZEN_FILE.exists():
        raise SystemExit("prompt_frozen.json already exists; a changed prompt needs a new version and a new frozen file.")
    p = render_prompt(prompts, cfg["instruction"])
    FROZEN_FILE.write_text(json.dumps({
        "experiment_id": cfg["experiment_id"], "prompt_version": prompts["prompt_version"],
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"), "instruction": cfg["instruction"],
        "prompt_sha256": sha256_text(p), "n_chars": len(p), "prompts_yaml_sha256": sha256_file(HERE / "prompts.yaml"),
        "note": "written before the single inference run; the run refuses if the rendered prompt differs"}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"frozen {sha256_text(p)[:16]}")


def selection_ok(cfg):
    sel = json.loads((HERE / "image_selection.json").read_text(encoding="utf-8"))
    img = HERE / cfg["image"]["file"]
    h = sha256_file(img)
    return (sel["recorded_BEFORE_any_model_inference"] and sel["selected"]["sha256"] == cfg["image"]["sha256"] == h
            and not sel["original_file_modified"]), sel


def render_run(run_dir):
    """(Re)draw the three images and review sheets from a saved run. Reads files only; calls no model."""
    import cv2

    import visualize
    run_dir = Path(run_dir)
    cfg, _ = load_all()
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    obj = json.loads((run_dir / "structured.json").read_text(encoding="utf-8"))
    image = cv2.imread(str(HERE / cfg["image"]["file"]))
    meta = {"run_id": man["run_id"], "parse_status": man.get("parse_status"), "image_name": Path(cfg["image"]["file"]).name}
    return visualize.render_all(image, obj, run_dir, meta, usable)


def previous_runs(results_dir, prompt_sha, image_sha):
    out = []
    for m in sorted(Path(results_dir).glob("*/manifest.json")):
        mm = json.loads(m.read_text(encoding="utf-8"))
        if mm.get("prompt_sha256") == prompt_sha and mm.get("image_sha256") == image_sha:
            out.append(m.parent.name)
    return out


def cmd_run(args):
    from semantic_task_nav.perception.model_client import ModelWorkerClient
    cfg, prompts = load_all()
    ok, msg = check_frozen(cfg, prompts)
    if not ok:
        raise SystemExit(f"run refused: {msg}")
    sel_ok, sel = selection_ok(cfg)
    if not sel_ok:
        raise SystemExit("run refused: image_selection.json does not match the copied image (or was not recorded before inference)")
    prompt = render_prompt(prompts, cfg["instruction"])
    prev = previous_runs(HERE / cfg["results_dir"], sha256_text(prompt), cfg["image"]["sha256"])
    if prev and cfg["one_run_per_image_and_prompt"] and not args.allow_repeat:
        raise SystemExit(f"run refused: this image + prompt was already run ({prev}); a repeat needs --allow-repeat")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
    run_dir = HERE / cfg["results_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    img_path = HERE / cfg["image"]["file"]
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    before = v02_fingerprint()
    manifest = {"experiment_id": cfg["experiment_id"], "run_id": run_id, "status": "RUNNING", "repeat_of": prev or None,
                "started": datetime.now().astimezone().isoformat(timespec="seconds"), "instruction": cfg["instruction"],
                "prompt_version": prompts["prompt_version"], "prompt_sha256": sha256_text(prompt), "frozen_check": msg,
                "image_file": cfg["image"]["file"], "image_source_path": cfg["image"]["source_path"], "image_sha256": cfg["image"]["sha256"],
                "image_size": cfg["image"]["size"], "reviewer_inventory_sha256": sha256_file(HERE / cfg["reviewer_inventory"]),
                "reviewer_inventory_given_to_model": False, "image_selection_sha256": sha256_file(HERE / "image_selection.json"),
                "models_used": cfg["model_use"], "not_used": ["Grounding DINO", "SAM2", "YOLO-World", "V1/V2 verifiers", "A*", "ROS2/Nav2", "hardware"],
                "model_input_audit": {"worker_call": "qwen_generate", "keys": ["image_path", "max_new_tokens", "prompt"], "n_images": 1},
                "generation": cfg["generation"], "files_sha256": {n: sha256_file(HERE / n) for n in
                                                                    ("config.yaml", "prompts.yaml", "semantic_map_probe.py", "visualize.py")},
                "host": {"python": sys.version.split()[0], "platform": platform.platform()}, "v02_before": before}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    client = ModelWorkerClient(load_config()["models"], run_dir / "worker_logs")
    try:
        manifest["worker_env"] = client.start()
        manifest["qwen_load"] = client.ensure_loaded("qwen")
        print("model loaded; running ONE inference ...", flush=True)
        t0 = time.perf_counter()
        r = client.call("qwen_generate", timeout=cfg["generation"]["request_timeout_s"], prompt=prompt, image_path=str(img_path),
                        max_new_tokens=cfg["generation"]["max_new_tokens"])
        wall = time.perf_counter() - t0
        manifest["loaded_models_at_end"] = sorted(client.loaded)
        manifest["gpu_swaps"] = list(getattr(client, "swaps", []))
    except BaseException as e:
        manifest["status"], manifest["error"] = "FAILED", f"{type(e).__name__}: {e}"
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    finally:
        client.close()
    (run_dir / "raw_response.txt").write_text(r["text"], encoding="utf-8")
    obj, pstatus, pinfo, errors = parse_reply(r["text"])
    cons, dropped = consistency(obj)
    result = {"run_id": run_id, "parse_status": pstatus, "parse_info": pinfo, "schema_errors": errors, "n_schema_errors": len(errors),
              "entries_not_drawable": dropped, "consistency": cons,
              "call": {k: r.get(k) for k in ("latency_s", "n_input_tokens", "n_output_tokens", "hit_max_new_tokens", "peak_allocated_gib",
                                             "image_size", "max_new_tokens", "model", "commit_hash")}, "wall_s": round(wall, 2)}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if obj is not None:
        (run_dir / "structured.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest.update(status="DONE", finished=datetime.now().astimezone().isoformat(timespec="seconds"),
                    parse_status=pstatus, v02_after=v02_fingerprint())
    manifest["v02_unchanged"] = manifest["v02_before"] == manifest["v02_after"]
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"parse={pstatus} schema_errors={len(errors)} tokens={r['n_output_tokens']} hit_max={r['hit_max_new_tokens']} {r['latency_s']}s")
    if obj is not None:
        render_run(run_dir)
    print(f"counts={cons['counts']} mobile_robot_entries={cons['mobile_robot_entries']}")
    print(f"done -> {run_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--show", action="store_true")
    p.set_defaults(fn=cmd_prompts)
    sub.add_parser("freeze").set_defaults(fn=cmd_freeze)
    rr = sub.add_parser("render", help="redraw the images of a saved run (no model call)")
    rr.add_argument("run_dir")
    rr.set_defaults(fn=lambda a: print({k: str(v) for k, v in render_run(a.run_dir).items() if k != "review"}))
    r = sub.add_parser("run")
    r.add_argument("--allow-repeat", action="store_true")
    r.set_defaults(fn=cmd_run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
