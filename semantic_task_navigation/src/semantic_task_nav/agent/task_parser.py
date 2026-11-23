"""Natural-language command -> validated structured task.

Backends
- qwen3_vl_local : existing Qwen3-VL-4B via the model worker (text only)
- openai_compatible_api : optional; text only (images are never sent); key from env/.env
- rule_based_demo : IMAGE_DEMO only. Deterministic keyword rules, labelled as such; it is NOT an LLM.

The model only interprets language. Python decides whether the task is executable, and the
action sequence comes from a fixed template (the model cannot add tools or steps).
"""
import json
import os
import re
import urllib.error
import urllib.request

from ..contracts import TASK_SCHEMA, SchemaError, parse_json_object, validate

ACTION_TEMPLATES = {
    "transport_and_place": ["detect_target", "locate_pickup_region", "navigate_to_pickup", "handoff_to_local_grasping",
                            "confirm_pickup", "navigate_to_loading_area", "handoff_to_placement", "verify_completion"],
    "navigate_to_region": ["locate_destination_region", "navigate_to_destination", "verify_arrival"],
}

# What v0.1 can actually execute. Anything else is rejected explicitly.
CLASS_SYNONYMS = {"cube": "cube", "block": "cube", "cube_block": "cube"}
SUPPORTED_TARGET_COLORS = {"red"}            # target inspection implements HSV red only
SOURCE_SYNONYMS = {"conveyor": "conveyor", "conveyor_belt": "conveyor", "belt_conveyor": "conveyor",
                   "production_line": "conveyor", "conveyor_line": "conveyor"}
SOURCE_RELATIONS = {"produced_by", "on", "near", "from", None}
REGION_SYNONYMS = {"loading_area": "loading_area", "loading_dock": "loading_area", "loading_zone": "loading_area",
                   "loading_bay": "loading_area", "unloading_area": "loading_area", "unloading_zone": "loading_area",
                   "loading_unloading_area": "loading_area", "dock": "loading_area", "dock_area": "loading_area"}
REFERENCE_SYNONYMS = {"truck": "truck", "lorry": "truck", "delivery_truck": "truck", "cargo_truck": "truck"}

LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["status", "task_type", "target", "destination", "reason"],
    "properties": {
        "status": {"type": "string", "enum": ["ok", "unsupported", "ambiguous", "missing_information"]},
        "task_type": {"type": ["string", "null"], "enum": ["transport_and_place", "navigate_to_region", None]},
        "target": {"type": ["object", "null"], "required": ["class", "color", "count", "source_reference", "source_relation"],
                   "properties": {"class": {"type": "string"}, "color": {"type": ["string", "null"]},
                                  "count": {"type": "integer", "minimum": 0},
                                  "source_reference": {"type": ["string", "null"]},
                                  "source_relation": {"type": ["string", "null"]}}},
        "destination": {"type": ["object", "null"], "required": ["region_type", "reference_object"],
                        "properties": {"region_type": {"type": "string"},
                                       "reference_object": {"type": ["string", "null"]}}},
        "reason": {"type": "string"},
    },
}


class TaskParseError(Exception):
    def __init__(self, code, message, raw=None, parsed=None):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.raw, self.parsed = code, message, raw, parsed


def _norm(word):
    return None if word is None else re.sub(r"[\s\-/]+", "_", word.strip().lower())


def build_task(obj, backend, raw, model=None, prompt_id=None):
    """Validate a parser output (LLM or rule-based) and turn it into an executable task."""
    try:
        validate(obj, LLM_OUTPUT_SCHEMA)
    except SchemaError as e:
        raise TaskParseError("SCHEMA_INVALID", str(e), raw, obj) from None
    status = obj["status"]
    if status != "ok":
        code = {"unsupported": "UNSUPPORTED_TASK", "ambiguous": "AMBIGUOUS_COMMAND",
                "missing_information": "MISSING_INFORMATION"}[status]
        raise TaskParseError(code, obj["reason"], raw, obj)
    task_type, target, dest = obj["task_type"], obj["target"], obj["destination"]
    if task_type is None:
        raise TaskParseError("SCHEMA_INVALID", "status ok but task_type is null", raw, obj)
    if dest is None:
        raise TaskParseError("MISSING_INFORMATION", "no destination", raw, obj)
    notes = []
    if task_type == "transport_and_place":
        if target is None:
            raise TaskParseError("MISSING_INFORMATION", "transport task without a target object", raw, obj)
        if target["count"] != 1:
            raise TaskParseError("MULTIPLE_TARGETS", f"instruction asks for {target['count']} objects; "
                                 "v0.1 executes exactly one target", raw, obj)
        cls = CLASS_SYNONYMS.get(_norm(target["class"]))
        if cls is None:
            raise TaskParseError("UNSUPPORTED_TARGET", f"target class '{target['class']}' is not supported "
                                 f"(supported: {sorted(set(CLASS_SYNONYMS.values()))})", raw, obj)
        color = _norm(target["color"])
        if color not in SUPPORTED_TARGET_COLORS:
            raise TaskParseError("UNSUPPORTED_TARGET", f"target color '{target['color']}' is not supported "
                                 f"(supported: {sorted(SUPPORTED_TARGET_COLORS)})", raw, obj)
        src = target["source_reference"]
        src_n = None if src is None else SOURCE_SYNONYMS.get(_norm(src))
        if src is not None and src_n is None:
            raise TaskParseError("UNSUPPORTED_SOURCE", f"source reference '{src}' is not supported", raw, obj)
        rel = _norm(target["source_relation"])
        if rel not in SOURCE_RELATIONS:
            notes.append(f"source_relation '{target['source_relation']}' not recognised -> null")
            rel = None
        if src_n is None:
            rel = None
        target_out = {"class": cls, "color": color, "source_reference": src_n, "source_relation": rel}
    else:
        if target is not None:
            notes.append("navigate_to_region: parser returned a target; ignored")
        target_out = None
    region = REGION_SYNONYMS.get(_norm(dest["region_type"]))
    if region is None:
        raise TaskParseError("UNSUPPORTED_DESTINATION", f"destination '{dest['region_type']}' is not supported "
                             "(supported: loading_area anchored on a truck)", raw, obj)
    ref = dest["reference_object"]
    ref_n = None if ref is None else REFERENCE_SYNONYMS.get(_norm(ref))
    if ref_n is None:
        raise TaskParseError("UNSUPPORTED_DESTINATION", "loading_area grounding needs a truck reference; "
                             f"got reference_object={ref!r}", raw, obj)
    task = {
        "task_type": task_type,
        "target": target_out,
        "destination": {"region_type": region, "reference_object": ref_n},
        "actions": list(ACTION_TEMPLATES[task_type]),
        "actions_source": "fixed_template (not model output)",
        "parser": {"backend": backend, "model": model, "prompt_id": prompt_id, "raw_response": raw,
                   "parsed_output": obj, "normalization_notes": notes},
    }
    return validate(task, TASK_SCHEMA)


def parse_with_llm(command, generate_text, prompt_cfg, backend, model):
    """generate_text(prompt) -> raw string. Raises TaskParseError; never fabricates a plan."""
    raw = generate_text(prompt_cfg["text"].replace("{command}", command))
    try:
        obj = parse_json_object(raw)
    except (json.JSONDecodeError, SchemaError) as e:
        raise TaskParseError("INVALID_JSON", f"model reply is not a JSON object: {e}", raw) from None
    return build_task(obj, backend, raw, model=model, prompt_id=prompt_cfg["id"])


# ------------------------------------------------------------------ rule-based (IMAGE_DEMO)
_COLORS = [("빨간", "red"), ("빨강", "red"), ("붉은", "red"), ("red", "red"), ("파란", "blue"), ("파랑", "blue"),
           ("blue", "blue"), ("초록", "green"), ("녹색", "green"), ("green", "green"), ("노란", "yellow"),
           ("yellow", "yellow")]
_UNSUPPORTED_VERBS = ["회전", "돌려", "rotate", "쌓아", "stack", "정렬", "sort", "열어", "open", "청소", "clean"]
_TRANSPORT_VERBS = ["놓아", "놓아줘", "옮겨", "가져다", "운반", "배치", "실어", "place", "bring", "deliver", "put", "carry", "move the"]
_NAV_VERBS = ["이동", "가줘", "가라", "가 줘", "navigate", "go to", "drive to"]


def rule_based_parse(command):
    """Deterministic keyword rules for IMAGE_DEMO. Output has the same shape as the LLM output."""
    c = command.lower()
    base = {"status": "ok", "task_type": None, "target": None, "destination": None,
            "reason": "rule_based_demo keyword rules (not an LLM)"}
    if any(v in c for v in _UNSUPPORTED_VERBS):
        return {**base, "status": "unsupported", "reason": "unsupported action keyword"}
    colors = []
    for k, v in _COLORS:
        if k in c and v not in colors:
            colors.append(v)
    has_cube = any(k in c for k in ("큐브", "cube", "블록", "block"))
    count = 2 if (len(colors) > 1 or any(k in c for k in ("두 개", "2개", "모든", " all ", "both"))) else 1
    dest = None
    if any(k in c for k in ("하역장", "적재장", "하역 구역", "loading")):
        dest = {"region_type": "loading_area", "reference_object": "truck" if ("트럭" in c or "truck" in c) else None}
    if has_cube and any(v in c for v in _TRANSPORT_VERBS):
        src = "conveyor" if ("컨베이어" in c or "conveyor" in c) else None
        rel = None
        if src:
            rel = "produced_by" if ("생산" in c or "produced" in c) else ("on" if ("위" in c or " on " in c) else "from")
        target = {"class": "cube", "color": colors[0] if colors else None, "count": count,
                  "source_reference": src, "source_relation": rel}
        if dest is None:
            return {**base, "status": "missing_information", "task_type": "transport_and_place", "target": target,
                    "reason": "no destination keyword"}
        return {**base, "task_type": "transport_and_place", "target": target, "destination": dest}
    if dest is not None and any(v in c for v in _NAV_VERBS):
        return {**base, "task_type": "navigate_to_region", "destination": dest}
    return {**base, "status": "missing_information", "reason": "no supported object/destination keywords"}


def parse_rule_based(command):
    obj = rule_based_parse(command)
    return build_task(obj, "rule_based_demo", json.dumps(obj, ensure_ascii=False), model=None, prompt_id=None)


# ------------------------------------------------------------------ optional API backend
class ApiUnavailable(Exception):
    pass


def openai_compatible_generate(api_cfg, prompt):
    """Text-only chat completion. The key comes from the environment and is never logged."""
    key = os.environ.get(api_cfg["api_key_env"])
    if not key:
        raise ApiUnavailable(f"API_KEY_MISSING: environment variable {api_cfg['api_key_env']} is not set")
    body = {"model": api_cfg["model"], "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(api_cfg["base_url"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=api_cfg.get("timeout_s", 60)) as r:
            data = json.load(r)
    except urllib.error.URLError as e:
        raise ApiUnavailable(f"API_REQUEST_FAILED: {type(e).__name__}") from None
    return data["choices"][0]["message"]["content"]
