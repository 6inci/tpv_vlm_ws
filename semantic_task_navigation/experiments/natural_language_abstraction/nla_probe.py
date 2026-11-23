#!/usr/bin/env python3
"""Natural Language Abstraction probe: same photo, same model, same prompt body -- only the instruction changes.

Reuses experiments/autonomous_semantic_reasoning/probe.py UNCHANGED (imported by file path): prompt rendering,
response-schema validation, consistency checks, overlay drawing, and through it the existing model worker.
What this file adds: (1) the instruction level (L1-L3) as the run variable, which the parent probe cannot
select (its single instruction comes from config.yaml), (2) three extra task_analysis fields in the schema,
(3) a new frozen-prompt file and a new results/ folder.

Nothing but one image path and one prompt text reaches the model. No path planning, no ROS2, no hardware.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY experiments/natural_language_abstraction/nla_probe.py prompts [--show]
  $PY experiments/natural_language_abstraction/nla_probe.py run --split smoke          # D1, format check only
  $PY experiments/natural_language_abstraction/nla_probe.py freeze
  $PY experiments/natural_language_abstraction/nla_probe.py run --split exploratory    # E1,E2; refuses unless frozen
"""
import argparse
import copy
import importlib.util
import json
import platform
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "autonomous_semantic_reasoning"
FROZEN_FILE = HERE / "prompts_frozen.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


asr = _load("asr_probe", PARENT / "probe.py")     # read-only reuse of the parent experiment

# ------------------------------------------------------------------ schema = parent schema + 3 task_analysis fields
_STR = {"type": "string"}
SCHEMA = copy.deepcopy(asr.RESPONSE_SCHEMA)
_ta = SCHEMA["properties"]["task_analysis"]
_ta["required"] = ["inferred_task_purpose", "info_given_by_instruction", "info_added_by_you"] + _ta["required"]
_ta["properties"].update({
    "inferred_task_purpose": {"type": "string", "minLength": 1},
    "info_given_by_instruction": {"type": "array", "items": _STR},
    "info_added_by_you": {"type": "array", "items": {
        "type": "object", "required": ["statement", "basis"],
        "properties": {"statement": {"type": "string", "minLength": 1},
                       "basis": {"type": "string", "enum": ["image", "general_knowledge", "assumption"]}}}},
})


# Smoke run 20261118_185651_smoke_8a5b: Qwen listed {"label": "red cube", "count": 0} ("looked for, not visible").
# The parent schema demanded count >= 1 and rejected the whole reply. A count of 0 is an honest statement, so this
# schema accepts it (validator only; the prompt is unchanged). nla_compare.py lists such entries separately.
SCHEMA["properties"]["observed_objects"]["items"]["properties"]["count"]["minimum"] = 0


def load_experiment():
    return asr.load_yaml(HERE / "config.yaml"), asr.load_yaml(HERE / "prompts.yaml")


def render(prompts, exp, level):
    return asr.render_prompt(prompts, exp["condition"], exp["instruction_levels"][level])


def prompt_table(exp, prompts):
    return {lv: {"instruction": exp["instruction_levels"][lv],
                 "sha256": asr.sha256_text(render(prompts, exp, lv)), "n_chars": len(render(prompts, exp, lv))}
            for lv in exp["instruction_levels"]}


def check_frozen(exp, prompts):
    if not FROZEN_FILE.exists():
        return False, "prompts_frozen.json is missing; run `nla_probe.py freeze` first"
    fr = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))
    cur = prompt_table(exp, prompts)
    for lv, v in cur.items():
        f = fr["levels"].get(lv, {})
        if f.get("sha256") != v["sha256"] or f.get("instruction") != v["instruction"]:
            return False, f"level {lv}: rendered prompt or instruction differs from the frozen one"
    return True, f"frozen prompts match ({fr['prompt_version']}, frozen_at {fr['frozen_at']})"


# ------------------------------------------------------------------ one model call
def run_level(client, exp, prompts, img_rec, level, out_dir):
    """One image + one prompt -> files in out_dir. `client` needs ensure_loaded(name) and call(cmd, **args)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    instruction = exp["instruction_levels"][level]
    prompt = render(prompts, exp, level)
    image_path = HERE / img_rec["file"]
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    client.ensure_loaded("qwen")
    call_args = {"prompt": prompt, "image_path": str(image_path), "max_new_tokens": exp["generation"]["max_new_tokens"]}
    r = client.call("qwen_generate", **call_args)
    (out_dir / "raw_response.txt").write_text(r["text"], encoding="utf-8")
    result = {"image_id": img_rec["id"], "image_role": img_rec["role"], "condition": level,   # `condition` = instruction level
              "instruction": instruction, "prior_condition": exp["condition"],
              "classification": exp["classification"], "prompt_version": prompts["prompt_version"],
              "prompt_sha256": asr.sha256_text(prompt), "image_sha256": img_rec["sha256"], "image_file": img_rec["file"],
              "model_input_audit": {"worker_call_keys": sorted(call_args), "n_images": 1,
                                    "prompt_sha256": asr.sha256_text(prompt)},
              "call": {k: r.get(k) for k in ("latency_s", "n_input_tokens", "n_output_tokens", "hit_max_new_tokens",
                                             "peak_allocated_gib", "image_size", "max_new_tokens", "model", "commit_hash")},
              "status": "OK", "error": None, "consistency_issues": [], "structured": None}
    try:
        obj = asr.validate(asr.parse_json_object(r["text"]), SCHEMA)
        result["structured"] = obj
        result["consistency_issues"] = asr.check_response(obj)
    except (json.JSONDecodeError, asr.SchemaError) as e:
        result["status"], result["error"] = "INVALID_RESPONSE", f"{type(e).__name__}: {e}"
    if result["structured"] is not None:
        result["decision"] = result["structured"]["task_analysis"]["decision"]
        (out_dir / "structured.json").write_text(json.dumps(result["structured"], indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    cv2.imwrite(str(out_dir / "overlay_areas.png"), asr.draw_overlays(cv2.imread(str(image_path)), result))
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


# ------------------------------------------------------------------ CLI
def select_images(exp, split, ids):
    imgs = [i for i in exp["images"] if i["role"] == split and (not ids or i["id"] in ids)]
    if not imgs:
        raise SystemExit(f"no images for split={split} ids={ids}")
    for i in imgs:
        h = asr.sha256_file(HERE / i["file"])
        if h != i["sha256"]:
            raise SystemExit(f"{i['id']}: sha256 differs from config.yaml ({h[:12]} != {i['sha256'][:12]})")
        if h.startswith(exp["excluded_image_sha256_prefix"]):
            raise SystemExit(f"{i['id']}: this is the v0.2 development image; not allowed here")
    return imgs


def cmd_prompts(args):
    exp, prompts = load_experiment()
    print(f"prompt_version {prompts['prompt_version']}  prompts.yaml sha256 {asr.sha256_file(HERE / 'prompts.yaml')[:16]}")
    for lv, v in prompt_table(exp, prompts).items():
        print(f"  {lv} {v['n_chars']:5d} chars  sha256 {v['sha256'][:16]}  {v['instruction']}")
    ok, msg = check_frozen(exp, prompts)
    print(f"frozen: {'YES' if ok else 'NO'} ({msg})")
    if args.show:
        for lv in exp["instruction_levels"]:
            print(f"\n===== {lv} =====\n{render(prompts, exp, lv)}")


def cmd_freeze(args):
    exp, prompts = load_experiment()
    if FROZEN_FILE.exists():
        raise SystemExit("prompts_frozen.json already exists; changing frozen prompts means a new prompt_version "
                         "and a new frozen file (move the old one aside by hand, never overwrite).")
    rec = {"experiment_id": exp["experiment_id"], "prompt_version": prompts["prompt_version"],
           "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
           "prompts_yaml_sha256": asr.sha256_file(HERE / "prompts.yaml"), "levels": prompt_table(exp, prompts),
           "note": "Written before the first exploratory run. `run --split exploratory` refuses if any hash differs."}
    FROZEN_FILE.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {FROZEN_FILE}")


def cmd_run(args):
    from semantic_task_nav.perception.model_client import ModelWorkerClient
    exp, prompts = load_experiment()
    levels = args.levels or list(exp["instruction_levels"])
    frozen_ok, frozen_msg = check_frozen(exp, prompts)
    if args.split == "exploratory" and not frozen_ok:
        raise SystemExit(f"exploratory run refused: {frozen_msg}")
    imgs = select_images(exp, args.split, args.images)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.split}_" + secrets.token_hex(2)
    run_dir = HERE / exp["results_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    client = ModelWorkerClient(asr.load_config()["models"], run_dir / "worker_logs")
    parent_files = ("probe.py", "prompts.yaml", "prompts_frozen.json", "config.yaml")
    manifest = {"experiment_id": exp["experiment_id"], "run_id": run_id, "split": args.split, "status": "RUNNING",
                "classification": exp["classification"], "classification_note": exp["classification_note"],
                "started": datetime.now().astimezone().isoformat(timespec="seconds"),
                "prompt_version": prompts["prompt_version"], "frozen_check": {"ok": frozen_ok, "message": frozen_msg},
                "prior_condition": exp["condition"], "levels": levels,
                "conditions": levels,                           # key read by the parent's compare helpers
                "prompts": {lv: prompt_table(exp, prompts)[lv] for lv in levels}, "generation": exp["generation"],
                "files_sha256": {n: asr.sha256_file(HERE / n) for n in ("config.yaml", "prompts.yaml", "nla_probe.py")},
                # proof that the parent experiment (imported, read-only) is what it was:
                "parent_files_sha256": {n: asr.sha256_file(PARENT / n) for n in parent_files},
                "existing_code_fingerprint": asr.code_fingerprint(),
                "host": {"python": sys.version.split()[0], "platform": platform.platform()},
                "images": [{k: i[k] for k in ("id", "role", "file", "sha256")} for i in imgs], "results": []}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"run {run_id}  prompts {prompts['prompt_version']}  frozen: {frozen_msg}")
    try:
        manifest["worker_env"] = client.start()
        for img in imgs:
            for lv in levels:
                t0 = time.perf_counter()
                res = run_level(client, exp, prompts, img, lv, run_dir / img["id"] / lv)
                manifest["results"].append({"image_id": img["id"], "level": lv, "status": res["status"],
                                            "decision": res.get("decision"), "latency_s": res["call"]["latency_s"],
                                            "n_output_tokens": res["call"]["n_output_tokens"],
                                            "hit_max_new_tokens": res["call"]["hit_max_new_tokens"]})
                print(f"  {img['id']} {lv} {res['status']:16s} decision={res.get('decision')} "
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
    p.add_argument("--show", action="store_true")
    p.set_defaults(fn=cmd_prompts)
    sub.add_parser("freeze", help="record prompt hashes before the first exploratory run").set_defaults(fn=cmd_freeze)
    r = sub.add_parser("run", help="run the model (GPU)")
    r.add_argument("--split", choices=["smoke", "exploratory"], required=True)
    r.add_argument("--images", nargs="*")
    r.add_argument("--levels", nargs="*", help="default: L1 L2 L3")
    r.set_defaults(fn=cmd_run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
