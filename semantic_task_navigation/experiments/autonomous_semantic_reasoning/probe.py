#!/usr/bin/env python3
"""Autonomous Semantic Reasoning Probe (independent of the v0.2 pipeline).

Question: given only a photo and a natural-language instruction, can Qwen3-VL propose the functional floor
area the instruction asks for, without any pre-made candidate regions, map, zone polygon or answer key?

What this script does (and does not do)
  * reuses the existing model worker (`semantic_task_nav.perception.model_client`) unchanged;
  * sends exactly ONE image + ONE prompt text per call; nothing else is passed to the model;
  * keeps prompt text, raw reply, validated structure, overlay image, timing and manifest in a NEW folder
    under results/ for every run (existing runs and results are never touched);
  * does not plan paths, does not touch the map/GUI/planner, does not send hardware commands;
  * computes no accuracy: there is no human-confirmed answer region.

Usage (from the project root, in the stnav_gui env):
  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/autonomous_semantic_reasoning/probe.py prompts            # show hashes, no GPU
  $PY experiments/autonomous_semantic_reasoning/probe.py run --split dev    # dev image(s), GPU
  $PY experiments/autonomous_semantic_reasoning/probe.py freeze             # pre-register prompt hashes
  $PY experiments/autonomous_semantic_reasoning/probe.py run --split eval   # refuses unless frozen
"""
import argparse
import json
import platform
import secrets
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from semantic_task_nav.config import code_fingerprint, load_config, sha256_file, sha256_text  # noqa: E402
from semantic_task_nav.contracts import SchemaError, parse_json_object, validate  # noqa: E402

FROZEN_FILE = HERE / "prompts_frozen.json"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ------------------------------------------------------------------ response schema
_BBOX = {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4}
_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}
RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["observed_objects", "functional_areas", "task_analysis", "unknown_from_image",
                 "physical_conditions_to_verify"],
    "properties": {
        "observed_objects": {"type": "array", "items": {
            "type": "object", "required": ["label", "count", "location_in_image", "visible_evidence"],
            "properties": {"label": {"type": "string", "minLength": 1}, "count": {"type": "integer", "minimum": 1},
                           "location_in_image": _STR, "visible_evidence": _STR}}},
        "functional_areas": {"type": "array", "items": {
            "type": "object",
            "required": ["id", "function_name", "bbox_2d", "observed_evidence", "inference", "confidence"],
            "properties": {"id": {"type": "string", "minLength": 1}, "function_name": {"type": "string", "minLength": 1},
                           "bbox_2d": _BBOX, "observed_evidence": _STR_LIST, "inference": _STR,
                           "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}}},
        "task_analysis": {
            "type": "object", "required": ["place_needed_for_task", "decision", "destination_area_id", "reason"],
            "properties": {"place_needed_for_task": _STR, "decision": {"type": "string", "enum": ["proposed", "withheld"]},
                           "destination_area_id": {"type": ["string", "null"]}, "reason": _STR}},
        "unknown_from_image": _STR_LIST,
        "physical_conditions_to_verify": _STR_LIST,
    },
}
LIMITS = {"observed_objects": 12, "functional_areas": 6}


def check_response(obj):
    """Consistency notes on a schema-valid reply. They are recorded, never used to 'fix' the reply."""
    issues = []
    boxes = [(f"functional_areas[{i}]", a["bbox_2d"]) for i, a in enumerate(obj["functional_areas"])]
    for name, b in boxes:
        if not (b[0] < b[2] and b[1] < b[3]):
            issues.append(f"{name}.bbox_2d is empty or inverted: {b}")
    ids = [a["id"] for a in obj["functional_areas"]]
    if len(set(ids)) != len(ids):
        issues.append(f"duplicate area ids: {ids}")
    ta = obj["task_analysis"]
    if ta["decision"] == "proposed" and ta["destination_area_id"] not in ids:
        issues.append(f"decision=proposed but destination_area_id {ta['destination_area_id']!r} is not in {ids}")
    if ta["decision"] == "withheld" and ta["destination_area_id"] is not None:
        issues.append("decision=withheld but destination_area_id is not null")
    for k, n in LIMITS.items():
        if len(obj[k]) > n:
            issues.append(f"{k} has {len(obj[k])} items, more than the requested {n}")
    return issues


# ------------------------------------------------------------------ prompts
def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment():
    return load_yaml(HERE / "config.yaml"), load_yaml(HERE / "prompts.yaml")


def render_prompt(prompts, condition, instruction):
    head = prompts["head"].replace("{instruction}", instruction).strip("\n")
    prior = (prompts["prior"].get(condition) or "").strip("\n")
    tail = prompts["tail"].strip("\n")
    return "\n\n".join(p for p in (head, prior, tail) if p) + "\n"


def prompt_table(exp, prompts):
    return {c: {"sha256": sha256_text(render_prompt(prompts, c, exp["instruction"])),
                "n_chars": len(render_prompt(prompts, c, exp["instruction"]))} for c in exp["conditions"]}


def frozen_record(exp, prompts):
    return {"experiment_id": exp["experiment_id"], "prompt_version": prompts["prompt_version"],
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "prompts_yaml_sha256": sha256_file(HERE / "prompts.yaml"), "instruction": exp["instruction"],
            "conditions": prompt_table(exp, prompts),
            "note": "Written before the first eval run. `run --split eval` refuses if any hash differs."}


def check_frozen(exp, prompts):
    """Return (ok, message)."""
    if not FROZEN_FILE.exists():
        return False, "prompts_frozen.json is missing; run `probe.py freeze` first"
    fr = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))
    cur = prompt_table(exp, prompts)
    if fr["instruction"] != exp["instruction"]:
        return False, "instruction differs from the frozen one"
    for c in exp["conditions"]:
        if fr["conditions"].get(c, {}).get("sha256") != cur[c]["sha256"]:
            return False, f"condition {c}: rendered prompt differs from the frozen prompt"
    return True, f"frozen prompts match ({fr['prompt_version']}, frozen_at {fr['frozen_at']})"


# ------------------------------------------------------------------ geometry / drawing helpers
def norm_to_px(b, w, h):
    return [b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h]


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a, b):
    return box_area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b):
    i = inter_area(a, b)
    u = box_area(a) + box_area(b) - i
    return i / u if u > 0 else 0.0


def _ascii(s):
    return str(s).encode("ascii", "replace").decode()


def _label(img, text, org, color, scale=0.6, thick=1):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x = int(max(0, min(org[0], img.shape[1] - tw - 2)))
    y = int(max(th + 3, min(org[1], img.shape[0] - bl - 2)))
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + bl), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _box(img, b, color, thick):
    p1, p2 = (int(round(b[0])), int(round(b[1]))), (int(round(b[2])), int(round(b[3])))
    cv2.rectangle(img, p1, p2, (0, 0, 0), thick + 2)
    cv2.rectangle(img, p1, p2, color, thick)


AREA_COLORS = [(255, 255, 0), (255, 0, 255), (255, 255, 255), (0, 128, 255), (255, 128, 0), (128, 255, 128)]


def legend_panel(width, lines, line_h=24):
    panel = np.full((line_h * len(lines) + 10, width, 3), 24, np.uint8)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(panel, _ascii(txt), (8, line_h * (i + 1) - 4), FONT, 0.55, col, 1, cv2.LINE_AA)
    return panel


def draw_overlays(image, result):
    """Return the areas overlay (BGR). Boxes are the model's own proposals, drawn unmodified."""
    h, w = image.shape[:2]
    head = f"{result['image_id']} / condition {result['condition']} / {result['status']}"
    s = result.get("structured")
    if s is None:
        panel = legend_panel(w, [(head, (255, 255, 255)), (f"no valid structure: {result.get('error')}"[:230], (80, 80, 255))])
        return np.vstack([image, panel])
    dest_id = s["task_analysis"]["destination_area_id"] if s["task_analysis"]["decision"] == "proposed" else None
    areas_img = image.copy()
    lines = [(f"{head} / decision={s['task_analysis']['decision']} destination={dest_id}", (255, 255, 255))]
    for i, a in enumerate(s["functional_areas"]):
        col = AREA_COLORS[i % len(AREA_COLORS)]
        b = norm_to_px(a["bbox_2d"], w, h)
        is_dest = a["id"] == dest_id
        _box(areas_img, b, col, 6 if is_dest else 3)
        _label(areas_img, ("[DEST] " if is_dest else "") + _ascii(f"{a['id']} {a['function_name']}")[:60],
               (b[0] + 4, b[1] + 22), col)
        lines.append((f"{'*' if is_dest else ' '}{a['id']} [{a['confidence']}] {a['function_name']}", col))
    for row in textwrap.wrap("reason: " + s["task_analysis"]["reason"], 200)[:3]:
        lines.append((row, (200, 200, 200)))
    inv = ", ".join(f"{o['label']} x{o['count']}" for o in s["observed_objects"])
    for row in textwrap.wrap("objects: " + inv, 200)[:2]:
        lines.append((row, (160, 160, 160)))
    return np.vstack([areas_img, legend_panel(w, lines)])


# ------------------------------------------------------------------ one model call
def run_condition(client, exp, prompts, img_rec, condition, out_dir):
    """One image + one prompt -> files in out_dir. `client` needs ensure_loaded(name) and call(cmd, **args)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    prompt = render_prompt(prompts, condition, exp["instruction"])
    image_path = HERE / img_rec["file"]
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    client.ensure_loaded("qwen")
    call_args = {"prompt": prompt, "image_path": str(image_path), "max_new_tokens": exp["generation"]["max_new_tokens"]}
    r = client.call("qwen_generate", **call_args)
    (out_dir / "raw_response.txt").write_text(r["text"], encoding="utf-8")
    result = {"image_id": img_rec["id"], "image_role": img_rec["role"], "condition": condition,
              "prompt_version": prompts["prompt_version"], "prompt_sha256": sha256_text(prompt),
              "image_sha256": img_rec["sha256"], "image_file": img_rec["file"],
              # exactly what reached the model: one image path + one prompt text + a token limit
              "model_input_audit": {"worker_call_keys": sorted(call_args), "n_images": 1,
                                    "prompt_sha256": sha256_text(prompt)},
              "call": {k: r.get(k) for k in ("latency_s", "n_input_tokens", "n_output_tokens", "hit_max_new_tokens",
                                             "peak_allocated_gib", "image_size", "max_new_tokens", "model",
                                             "commit_hash")},
              "status": "OK", "error": None, "consistency_issues": [], "structured": None}
    try:
        obj = validate(parse_json_object(r["text"]), RESPONSE_SCHEMA)
        result["structured"] = obj
        result["consistency_issues"] = check_response(obj)
    except (json.JSONDecodeError, SchemaError) as e:
        result["status"], result["error"] = "INVALID_RESPONSE", f"{type(e).__name__}: {e}"
    if result["structured"] is not None:
        result["decision"] = result["structured"]["task_analysis"]["decision"]
        (out_dir / "structured.json").write_text(json.dumps(result["structured"], indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    image = cv2.imread(str(image_path))
    cv2.imwrite(str(out_dir / "overlay_areas.png"), draw_overlays(image, result))
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


# ------------------------------------------------------------------ CLI
def select_images(exp, split, ids):
    imgs = [i for i in exp["images"] if i["role"] == split and (not ids or i["id"] in ids)]
    if not imgs:
        raise SystemExit(f"no images for split={split} ids={ids}")
    for i in imgs:
        p = HERE / i["file"]
        h = sha256_file(p)
        if h != i["sha256"]:
            raise SystemExit(f"{i['id']}: sha256 differs from config.yaml ({h[:12]} != {i['sha256'][:12]})")
        if h.startswith(exp["excluded_image_sha256_prefix"]):
            raise SystemExit(f"{i['id']}: this is the v0.2 development image; not allowed here")
    return imgs


def cmd_prompts(args):
    exp, prompts = load_experiment()
    print(f"prompt_version {prompts['prompt_version']}  prompts.yaml sha256 {sha256_file(HERE / 'prompts.yaml')[:16]}")
    for c, v in prompt_table(exp, prompts).items():
        print(f"  {c:3s} {v['n_chars']:5d} chars  sha256 {v['sha256'][:16]}")
    ok, msg = check_frozen(exp, prompts)
    print(f"frozen: {'YES' if ok else 'NO'} ({msg})")
    if args.show:
        for c in exp["conditions"]:
            print(f"\n===== {c} =====\n{render_prompt(prompts, c, exp['instruction'])}")


def cmd_freeze(args):
    exp, prompts = load_experiment()
    if FROZEN_FILE.exists():
        raise SystemExit("prompts_frozen.json already exists; changing frozen prompts means a new prompt_version "
                         "and a new frozen file (move the old one aside by hand, never overwrite).")
    FROZEN_FILE.write_text(json.dumps(frozen_record(exp, prompts), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {FROZEN_FILE}")


def cmd_run(args):
    from semantic_task_nav.perception.model_client import ModelWorkerClient
    exp, prompts = load_experiment()
    conds = args.conditions or exp["conditions"]
    frozen_ok, frozen_msg = check_frozen(exp, prompts)
    if args.split == "eval" and not frozen_ok:
        raise SystemExit(f"eval run refused: {frozen_msg}")
    imgs = select_images(exp, args.split, args.images)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.split}_" + secrets.token_hex(2)
    run_dir = HERE / exp["results_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    models_cfg = load_config()["models"]
    client = ModelWorkerClient(models_cfg, run_dir / "worker_logs")
    manifest = {"experiment_id": exp["experiment_id"], "run_id": run_id, "split": args.split, "status": "RUNNING",
                "started": datetime.now().astimezone().isoformat(timespec="seconds"),
                "prompt_version": prompts["prompt_version"], "frozen_check": {"ok": frozen_ok, "message": frozen_msg},
                "instruction": exp["instruction"], "generation": exp["generation"],
                "prompts": {c: prompt_table(exp, prompts)[c] for c in conds},
                "files_sha256": {n: sha256_file(HERE / n) for n in ("config.yaml", "prompts.yaml", "probe.py")},
                "existing_code_fingerprint": code_fingerprint(),
                "host": {"python": sys.version.split()[0], "platform": platform.platform()},
                "images": [{k: i[k] for k in ("id", "role", "file", "source_path", "sha256")} for i in imgs],
                "conditions": conds, "results": []}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"run {run_id}  prompts {prompts['prompt_version']}  frozen: {frozen_msg}")
    try:
        manifest["worker_env"] = client.start()
        for img in imgs:
            for c in conds:
                t0 = time.perf_counter()
                res = run_condition(client, exp, prompts, img, c, run_dir / img["id"] / c)
                manifest["results"].append({"image_id": img["id"], "condition": c, "status": res["status"],
                                            "decision": res.get("decision"), "latency_s": res["call"]["latency_s"],
                                            "n_output_tokens": res["call"]["n_output_tokens"],
                                            "hit_max_new_tokens": res["call"]["hit_max_new_tokens"]})
                print(f"  {img['id']} {c:3s} {res['status']:16s} decision={res.get('decision')} "
                      f"{res['call']['n_output_tokens']} tok {res['call']['latency_s']}s "
                      f"(wall {time.perf_counter() - t0:.0f}s) issues={len(res['consistency_issues'])}", flush=True)
        manifest["model"] = client.loaded.get("qwen")
        manifest["status"] = "DONE"
    except BaseException as e:   # recorded, then re-raised: a failed run must not look finished
        manifest["status"], manifest["error"] = "FAILED", f"{type(e).__name__}: {e}"
        raise
    finally:
        manifest["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        client.close()
    print(f"done -> {run_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts", help="show prompt hashes and frozen state (no GPU)")
    p.add_argument("--show", action="store_true", help="print the full rendered prompts")
    p.set_defaults(fn=cmd_prompts)
    sub.add_parser("freeze", help="record prompt hashes before the first eval run").set_defaults(fn=cmd_freeze)
    r = sub.add_parser("run", help="run the model (GPU)")
    r.add_argument("--split", choices=["dev", "eval"], required=True)
    r.add_argument("--images", nargs="*", help="image ids, default: all of the split")
    r.add_argument("--conditions", nargs="*", help="default: all conditions")
    r.set_defaults(fn=cmd_run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
