#!/usr/bin/env python3
"""Run local text-only task parsing and offline target/zone integration."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.perception import generate_qwen_text, load_local_qwen
from vlm_reposition.task import (
    assemble_integrated_task,
    evaluate_task_output,
    load_task_language,
    prepare_task_inputs,
)
from vlm_reposition.world import load_zones


DEFAULT_CONFIG = WORKSPACE / "config" / "task_parser.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "task_parser_rework"


def load_json(path):
    return json.loads(Path(path).read_text())


def load_groundings(result_dir):
    groundings = {}
    for path in sorted(Path(result_dir).glob("*_run01.json")):
        result = load_json(path)
        result["source_result"] = str(path)
        groundings[result["query"]] = result
    return groundings


def integrate_if_available(validated, zones, groundings):
    if validated["status"] != "VALID_TASK":
        return None
    grounding_phrase = validated["target"]["grounding_phrase"]
    grounding = groundings.get(grounding_phrase)
    if grounding is None:
        return {
            "status": "TARGET_GROUNDING_NOT_AVAILABLE",
            "target_phrase": grounding_phrase,
        }
    integrated = assemble_integrated_task(validated, zones, grounding)
    integrated["target"]["grounding"]["source_result"] = grounding["source_result"]
    return integrated


def result_matches_expected(validated, expected, integrated):
    if validated["status"] != expected["status"]:
        return False
    if expected["status"] != "VALID_TASK":
        return validated.get("error_code") == expected.get("error_code")
    return (
        validated["action"]["canonical"] == expected["action"]
        and validated["target"]["grounding_phrase"] == expected["target_phrase"]
        and validated["destination"]["zone_id"] == expected["zone_id"]
        and integrated is not None
        and integrated.get("status") == "VALID_TASK"
    )


def repeatability_summary(case, records):
    values = {
        "status": [record["validated"]["status"] for record in records],
        "action_surface": [
            record["validated"].get("action", {}).get("surface_phrase") for record in records
        ],
        "canonical_action": [
            record["validated"].get("action", {}).get("canonical") for record in records
        ],
        "target_surface": [
            record["validated"].get("target", {}).get("surface_phrase") for record in records
        ],
        "grounding_phrase": [
            record["validated"].get("target", {}).get("grounding_phrase") for record in records
        ],
        "destination_surface": [
            record["validated"].get("destination", {}).get("surface_phrase") for record in records
        ],
        "zone_id": [
            record["validated"].get("destination", {}).get("zone_id") for record in records
        ],
    }
    return {
        "case_id": case["id"],
        "instruction": case["instruction"],
        "runs": len(records),
        **{f"{key}_consistent": len(set(items)) == 1 for key, items in values.items()},
        "values": values,
    }


def infer(processor, model, instruction, max_new_tokens):
    inputs = prepare_task_inputs(processor, model, instruction)
    return generate_qwen_text(processor, model, inputs, max_new_tokens=max_new_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--instruction")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    config = load_json(args.config)
    if not Path(config["model_cache_path"]).is_dir():
        raise RuntimeError(f"local model cache not found: {config['model_cache_path']}")

    zones = load_zones(config["zone_config_path"])
    language_config = load_task_language(config["task_language_config_path"])
    groundings = load_groundings(config["target_grounding_result_dir"])
    processor, model, model_load_sec = load_local_qwen(config["model_id"])

    if args.instruction:
        raw_output, inference_sec = infer(
            processor, model, args.instruction, args.max_new_tokens
        )
        parsed, validated = evaluate_task_output(
            args.instruction, raw_output, zones, language_config
        )
        print(json.dumps({
            "raw_model_output": raw_output,
            "parsed": parsed,
            "validated": validated,
            "integrated": integrate_if_available(validated, zones, groundings),
            "model_load_sec": model_load_sec,
            "inference_sec": inference_sec,
        }, ensure_ascii=False, indent=2))
        return

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    dataset = load_json(config["instruction_dataset_path"])
    all_records = []
    repeatability = []
    for case in dataset["cases"]:
        run_count = 3 if case.get("repeatability") else 1
        case_records = []
        for run_number in range(1, run_count + 1):
            raw_output, inference_sec = infer(
                processor, model, case["instruction"], args.max_new_tokens
            )
            parsed, validated = evaluate_task_output(
                case["instruction"], raw_output, zones, language_config
            )
            integrated = integrate_if_available(validated, zones, groundings)
            record = {
                "case_id": case["id"],
                "category": case["category"],
                "core_metric": case.get("core_metric", True),
                "run": run_number,
                "instruction": case["instruction"],
                "raw_model_output": raw_output,
                "parsed": parsed,
                "validated": validated,
                "integrated": integrated,
                "inference_sec": inference_sec,
                "expected": case["expected"],
            }
            record["matches_expected"] = result_matches_expected(
                validated, case["expected"], integrated
            )
            case_records.append(record)
            all_records.append(record)
            print(
                f"{case['id']} run {run_number}: {validated['status']} "
                f"match={record['matches_expected']} {inference_sec:.3f}s"
            )
        if run_count > 1:
            repeatability.append(repeatability_summary(case, case_records))

    first_runs = [record for record in all_records if record["run"] == 1]
    core_first_runs = [record for record in first_runs if record["core_metric"]]
    additional = [record for record in first_runs if not record["core_metric"]]
    valid = [record for record in core_first_runs if record["category"] == "valid"]
    invalid = [record for record in core_first_runs if record["category"] != "valid"]
    repeatable = all(
        item["status_consistent"]
        and item["action_surface_consistent"]
        and item["canonical_action_consistent"]
        and item["target_surface_consistent"]
        and item["grounding_phrase_consistent"]
        and item["destination_surface_consistent"]
        and item["zone_id_consistent"]
        for item in repeatability
    )
    passed = all(record["matches_expected"] for record in first_runs) and repeatable

    import torch
    from transformers import __version__ as transformers_version

    summary = {
        "result": "PASS" if passed else "PARTIAL",
        "verification_type": "Deterministic Task Canonicalization + Unsupported Action Guard",
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_status": dataset["status"],
        "dataset_path": config["instruction_dataset_path"],
        "model": config["model_id"],
        "mode": "text-only",
        "model_cache": config["model_cache_path"],
        "local_files_only": True,
        "model_load_sec": model_load_sec,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "metrics": {
            "case_count": len(core_first_runs),
            "valid_case_count": len(valid),
            "valid_task_parse_success": sum(record["matches_expected"] for record in valid),
            "action_field_correct": sum(
                record["validated"].get("action", {}).get("canonical")
                == record["expected"].get("action")
                for record in valid
            ),
            "target_phrase_correct": sum(
                record["validated"].get("target", {}).get("grounding_phrase")
                == record["expected"].get("target_phrase")
                for record in valid
            ),
            "destination_zone_correct": sum(
                record["validated"].get("destination", {}).get("zone_id")
                == record["expected"].get("zone_id")
                for record in valid
            ),
            "invalid_case_count": len(invalid),
            "invalid_rejection_correct": sum(record["matches_expected"] for record in invalid),
            "additional_regression_count": len(additional),
            "additional_regression_correct": sum(record["matches_expected"] for record in additional),
        },
        "repeatability": repeatability,
        "model_generated_metric_coordinates": False,
        "model_generated_path": False,
        "model_selected_robot_velocity": False,
    }

    args.output_dir.mkdir(parents=True)
    for record in all_records:
        path = args.output_dir / f"{record['case_id']}_run{record['run']:02d}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "task_parser_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
