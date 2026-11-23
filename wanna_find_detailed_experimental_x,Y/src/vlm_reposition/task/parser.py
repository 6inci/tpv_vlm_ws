"""Surface phrase extraction with deterministic task canonicalization."""

import json
from pathlib import Path
import re

from vlm_reposition.world import get_zone, resolve_zone_alias


REQUIRED_FIELDS = frozenset({"action_phrase", "target_phrase", "destination_phrase"})


class TaskValidationError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _normalize(value):
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold().replace("_", " ")).split())


def _text(value):
    if not isinstance(value, str) or not value.strip():
        return None
    stripped = value.strip()
    return None if stripped.casefold() in {"null", "none", "n/a"} else stripped


def _target_key(value):
    normalized = _normalize(value)
    words = normalized.split()
    if words and words[0] in {"a", "an", "the"}:
        normalized = " ".join(words[1:])
    return normalized


def load_task_language(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != 1:
        raise TaskValidationError("INVALID_SCHEMA", "task language schema_version must be 1")
    supported = config.get("supported_actions")
    unsupported = config.get("unsupported_action_phrases")
    targets = config.get("target_aliases")
    ambiguous = config.get("ambiguous_target_phrases")
    if not isinstance(supported, dict) or not supported:
        raise TaskValidationError("INVALID_SCHEMA", "supported_actions must be a non-empty object")
    if not isinstance(unsupported, list) or not unsupported:
        raise TaskValidationError("INVALID_SCHEMA", "unsupported_action_phrases must be non-empty")
    if not isinstance(targets, dict) or not targets:
        raise TaskValidationError("INVALID_SCHEMA", "target_aliases must be a non-empty object")
    if not isinstance(ambiguous, list) or not ambiguous:
        raise TaskValidationError("INVALID_SCHEMA", "ambiguous_target_phrases must be non-empty")

    action_aliases = {}
    for canonical, aliases in supported.items():
        if not isinstance(canonical, str) or not canonical or not isinstance(aliases, list) or not aliases:
            raise TaskValidationError("INVALID_SCHEMA", "supported action aliases are invalid")
        for alias in aliases:
            key = _normalize(alias) if isinstance(alias, str) else ""
            if not key or key in action_aliases:
                raise TaskValidationError("INVALID_SCHEMA", f"duplicate or invalid action alias: {alias!r}")
            action_aliases[key] = canonical
    for alias in unsupported:
        key = _normalize(alias) if isinstance(alias, str) else ""
        if not key or key in action_aliases:
            raise TaskValidationError("INVALID_SCHEMA", f"duplicate or invalid action alias: {alias!r}")
        action_aliases[key] = None

    target_aliases = {}
    for grounding_phrase, aliases in targets.items():
        if not isinstance(grounding_phrase, str) or not grounding_phrase or not isinstance(aliases, list) or not aliases:
            raise TaskValidationError("INVALID_SCHEMA", "target aliases are invalid")
        for alias in aliases:
            key = _target_key(alias) if isinstance(alias, str) else ""
            if not key or key in target_aliases:
                raise TaskValidationError("INVALID_SCHEMA", f"duplicate or invalid target alias: {alias!r}")
            target_aliases[key] = grounding_phrase
    if any(not isinstance(value, str) or not _normalize(value) for value in ambiguous):
        raise TaskValidationError("INVALID_SCHEMA", "ambiguous target phrases are invalid")
    return config


def make_task_prompt():
    return '''You extract surface phrases from one mobile-robot instruction.

Return exactly one JSON object with exactly these keys:
{"action_phrase":"...","target_phrase":"...","destination_phrase":"..."}

Rules:
- action_phrase is the shortest action verb phrase explicitly present in the instruction.
- target_phrase is the explicit object phrase, without grammatical case particles or English determiners.
- destination_phrase is the shortest explicit location, direction, or region, without prepositions or grammatical case particles. Extract it even when it is not a configured zone name.
- Preserve the user's wording and language whenever possible.
- Do not translate targets to match an ontology and do not generate canonical action names.
- Use the JSON null literal, not the string "null", when a field is absent. Never infer a missing field.
- Preserve demonstratives as the target phrase; do not guess their referent.
- Do not add keys or generate coordinates, bounding boxes, zone IDs, polygons, paths, poses, velocities, or robot commands.
- Return JSON only, without Markdown or explanation.'''


def prepare_task_inputs(processor, model, instruction):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": make_task_prompt()}]},
        {"role": "user", "content": [{"type": "text", "text": instruction}]},
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


def parse_task_model_output(model_output):
    if not isinstance(model_output, str) or not model_output.strip():
        raise TaskValidationError("JSON_PARSE_ERROR", "model output must be non-empty text")
    decoder = json.JSONDecoder()
    parsed = None
    for index, character in enumerate(model_output):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(model_output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    if parsed is None:
        raise TaskValidationError("JSON_PARSE_ERROR", "model output has no JSON object")
    if set(parsed) != REQUIRED_FIELDS:
        missing = REQUIRED_FIELDS - set(parsed)
        code = "INVALID_SCHEMA"
        for field, field_code in (
            ("action_phrase", "MISSING_ACTION"),
            ("target_phrase", "MISSING_TARGET"),
            ("destination_phrase", "MISSING_DESTINATION"),
        ):
            if field in missing:
                code = field_code
                break
        raise TaskValidationError(code, "task output must contain exactly the required fields")
    if any(value is not None and not isinstance(value, str) for value in parsed.values()):
        raise TaskValidationError("INVALID_SCHEMA", "task fields must be text or null")
    return parsed


def resolve_action_phrase(surface_phrase, language_config):
    surface = _text(surface_phrase)
    if surface is None:
        return {"surface_phrase": None, "canonical": None, "resolution_status": "MISSING_ACTION"}
    key = _normalize(surface)
    for canonical, aliases in language_config["supported_actions"].items():
        if key in {_normalize(alias) for alias in aliases}:
            return {"surface_phrase": surface, "canonical": canonical, "resolution_status": "SUPPORTED_ACTION"}
    if key in {_normalize(alias) for alias in language_config["unsupported_action_phrases"]}:
        return {"surface_phrase": surface, "canonical": None, "resolution_status": "UNSUPPORTED_ACTION"}
    return {"surface_phrase": surface, "canonical": None, "resolution_status": "ACTION_NOT_RESOLVED"}


def canonicalize_target_phrase(surface_phrase, language_config):
    surface = _text(surface_phrase)
    if surface is None:
        return None
    key = _target_key(surface)
    if key in {_target_key(value) for value in language_config["ambiguous_target_phrases"]}:
        return {"surface_phrase": surface, "grounding_phrase": None, "canonicalization_status": "TARGET_AMBIGUOUS"}
    for grounding_phrase, aliases in language_config["target_aliases"].items():
        if key in {_target_key(alias) for alias in aliases}:
            return {"surface_phrase": surface, "grounding_phrase": grounding_phrase, "canonicalization_status": "KNOWN_ALIAS"}
    return {"surface_phrase": surface, "grounding_phrase": surface, "canonicalization_status": "OPEN_VOCABULARY"}


def unsupported_action_in_instruction(instruction, language_config):
    instruction_key = f" {_normalize(instruction)} "
    for phrase in language_config["unsupported_action_phrases"]:
        if f" {_normalize(phrase)} " in instruction_key:
            return phrase
    return None


def _failure(instruction, status, error_code, reason, parsed=None, action=None):
    return {
        "status": status,
        "raw_instruction": instruction,
        "error_code": error_code,
        "reason": reason,
        "parsed_fields": parsed,
        "action": action,
        "target_grounding_status": "NOT_RUN",
        "zone_resolution_status": "NOT_RUN",
        "path_status": "NOT_RUN",
    }


def _unsupported_guard_failure(instruction, matched_phrase, parsed=None):
    return _failure(
        instruction,
        "UNSUPPORTED_ACTION",
        "UNSUPPORTED_ACTION",
        f"raw instruction contains unsupported action phrase: {matched_phrase}",
        parsed,
        {"surface_phrase": matched_phrase, "canonical": None, "resolution_status": "UNSUPPORTED_ACTION_RAW_GUARD"},
    )


def validate_task_fields(instruction, parsed, zones, language_config):
    guarded_phrase = unsupported_action_in_instruction(instruction, language_config)
    if guarded_phrase is not None:
        return _unsupported_guard_failure(instruction, guarded_phrase, parsed)

    action = resolve_action_phrase(parsed.get("action_phrase"), language_config)
    if action["resolution_status"] == "MISSING_ACTION":
        return _failure(instruction, "TASK_INCOMPLETE", "MISSING_ACTION", "action is missing", parsed, action)
    if action["resolution_status"] != "SUPPORTED_ACTION":
        return _failure(
            instruction,
            "UNSUPPORTED_ACTION",
            action["resolution_status"],
            f"action is not supported: {action['surface_phrase']}",
            parsed,
            action,
        )

    target = canonicalize_target_phrase(parsed.get("target_phrase"), language_config)
    if target is None:
        return _failure(instruction, "TASK_INCOMPLETE", "MISSING_TARGET", "target is missing", parsed, action)
    if target["canonicalization_status"] == "TARGET_AMBIGUOUS":
        return _failure(
            instruction,
            "TARGET_AMBIGUOUS",
            "TARGET_AMBIGUOUS",
            f"target phrase is referential: {target['surface_phrase']}",
            parsed,
            action,
        )

    destination = _text(parsed.get("destination_phrase"))
    if destination is None:
        return _failure(instruction, "TASK_INCOMPLETE", "MISSING_DESTINATION", "destination is missing", parsed, action)
    zone_id = resolve_zone_alias(destination, zones)
    if zone_id == "ZONE_NOT_RESOLVED":
        result = _failure(
            instruction,
            "DESTINATION_NOT_RESOLVED",
            "ZONE_NOT_RESOLVED",
            f"destination is not a configured zone alias: {destination}",
            parsed,
            action,
        )
        result["target"] = target
        result["zone_resolution_status"] = "ZONE_NOT_RESOLVED"
        return result
    return {
        "status": "VALID_TASK",
        "raw_instruction": instruction,
        "action": action,
        "target": target,
        "destination": {"surface_phrase": destination, "zone_id": zone_id},
    }


def _parse_failure(instruction, error):
    status = "TASK_INCOMPLETE" if error.code in {"MISSING_ACTION", "MISSING_TARGET", "MISSING_DESTINATION"} else "INVALID_MODEL_OUTPUT"
    return _failure(instruction, status, error.code, str(error))


def evaluate_task_output(instruction, model_output, zones, language_config):
    guarded_phrase = unsupported_action_in_instruction(instruction, language_config)
    if guarded_phrase is not None:
        try:
            parsed = parse_task_model_output(model_output)
        except TaskValidationError:
            parsed = None
        return parsed, _unsupported_guard_failure(instruction, guarded_phrase, parsed)
    try:
        parsed = parse_task_model_output(model_output)
    except TaskValidationError as error:
        return None, _parse_failure(instruction, error)
    return parsed, validate_task_fields(instruction, parsed, zones, language_config)


def assemble_integrated_task(task, zones, grounding):
    if task.get("status") != "VALID_TASK":
        raise TaskValidationError("INVALID_SCHEMA", "only VALID_TASK can be integrated")
    if grounding.get("target_status") != "TARGET_FOUND":
        raise TaskValidationError("INVALID_SCHEMA", "target grounding must be TARGET_FOUND")
    if _normalize(grounding.get("query", "")) != _normalize(task["target"]["grounding_phrase"]):
        raise TaskValidationError("INVALID_SCHEMA", "grounding query does not match canonical target")
    zone = get_zone(zones, task["destination"]["zone_id"])
    return {
        "status": "VALID_TASK",
        "language_status": "VALID_TASK",
        "raw_instruction": task["raw_instruction"],
        "action": task["action"],
        "target": {
            **task["target"],
            "grounding": {
                "status": grounding["target_status"],
                "detected_label": grounding["detected_label"],
                "world_coordinate": grounding["world_coordinate"],
                "active_workspace_status": grounding["active_workspace_status"],
            },
        },
        "destination": {
            "surface_phrase": task["destination"]["surface_phrase"],
            "zone_id": zone["id"],
            "center": zone["center"],
            "polygon": zone["polygon"],
            "coordinate_frame": zones["coordinate_frame"],
            "unit": zones["unit"],
        },
        "feasibility_status": "NOT_EVALUATED",
        "path_status": "NOT_RUN",
    }
