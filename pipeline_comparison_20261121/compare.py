#!/usr/bin/env python3
"""Same-benchmark comparison of the pipelines (8 photos x I1/I2, ground truth fixed before the runs).
Usage: compare.py <itv_session_dir>
Reads:  model_performance_20261120/session_20261120_224831/task_evaluation.json  (multimodel 8B / 4B, already scored)
        <itv_session_dir> (instruction_target_viewer; scored here with the same evaluate_tasks.py, written into that session)
        v02_runs/*/result.json (v0.2)
        navigation_handoff unified + batch results (human/AI-given targets -> planner only: the upper bound of the planning side)
Writes: comparison.json in this folder (and task_evaluation.json/.md inside the itv session).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MP = Path("/home/binci/vlm_ws/model_performance_20261120")
sys.path.insert(0, str(MP))
import evaluate_tasks as ET  # noqa: E402  (same scoring rules as the 32-run benchmark)

NH = Path("/home/binci/vlm_ws/semantic_task_navigation/experiments/navigation_handoff/results")


def summarize(rows):
    c = Counter(r["outcome"] for r in rows)
    valid = [r for r in rows if r["valid_task"] or (r["borderline"] and r["A_commit"])]
    return {"runs": len(rows), "success": c["SUCCESS"], "valid_runs": len([r for r in rows if r["valid_task"]]),
            "A_right": sum(r["A_right"] for r in rows), "B_right_with_A": sum(r["A_right"] and r["B_right"] for r in rows),
            "wrong_commit": c["WRONG_COMMIT"], "abstain_b_wrong": c["ABSTAIN_B_WRONG"],
            "correct_abstain": c["CORRECT_ABSTAIN"], "abstain_on_valid": c["ABSTAIN"], "a_only": c["A_ONLY"],
            "cube_in_vlm_picks_I1": sum("CUBE" in (r.get("A_vlm_pointed_at") or []) for r in rows if r["instruction_key"] == "I1"),
            "mean_wall_s": round(sum(float(r["wall_s"] or 0) for r in rows) / max(len(rows), 1), 1),
            "first_failure": dict(Counter(r["first_failure"] for r in rows)), "_valid_runs_scored": len(valid)}


def v02_rows(gt):
    rows = []
    for d in sorted((HERE / "v02_runs").glob("2*")):
        res = json.loads((d / "result.json").read_text(encoding="utf-8"))
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8")) if (d / "manifest.json").exists() else {}
        cmd = json.dumps(man, ensure_ascii=False) + json.dumps(res, ensure_ascii=False)
        ik = "I1" if "팔레트" in cmd else "I2"
        img = next((p for p in gt["photos"] if gt["photos"][p]["file"] in cmd), None)
        rejected = res.get("overall_status") == "TASK_REJECTED"
        valid = ik == "I1" or img in gt["instructions"]["I2"]["valid_a"]
        rows.append({"backend": "v02_qwen3vl_4b", "photo_id": img, "instruction_key": ik, "run_id": d.name,
                     "first_failure": (res.get("status_reasons") or ["-"])[0].split(":")[0], "wall_s": None,
                     "A_commit": False if rejected else None, "A_right": False, "B_right": False, "valid_task": valid, "borderline": False,
                     "outcome": ("ABSTAIN" if valid else "CORRECT_ABSTAIN") if rejected else "CHECK_MANUALLY",
                     "overall_status": res.get("overall_status"), "reason": (res.get("status_reasons") or [""])[0][:160]})
    return rows


def oracle_rows():
    """Planner only, targets given (USER_APPROVED or AI-read): Path A = robot -> cube grasp stop, Path B = cube -> empty pallet stop."""
    batch = json.loads((NH / "20261120_080841_step4b_batch/comparison.json").read_text(encoding="utf-8"))["rows"]
    uni = {}
    for p in ("photo_1", "photo_3", "photo_5"):
        v = json.loads((NH / f"20261120_174948_unified/{p}/validation.json").read_text(encoding="utf-8"))
        uni[p] = v
    out = []
    for r in batch:
        a, b = r["path_A_status"], r["path_B_status"]
        src = "batch 080841"
        if r["photo"] == "photo_5":          # unified rules (pallets as obstacles) supersede the older run: B NOT_FOUND (UNIFIED_REPORT.md)
            a, b, src = "VALID", "NOT_FOUND", "unified 174948"
        out.append({"photo": r["photo"], "path_A": a, "path_B": b, "both_valid": a == "VALID" and b == "VALID", "positions": r["positions"], "source": src})
    return out


def main(itv_session):
    gt = yaml.safe_load((HERE / "ground_truth_COPY.yaml").read_text(encoding="utf-8"))
    mm = json.loads((MP / "session_20261120_224831/task_evaluation.json").read_text(encoding="utf-8"))["rows"]
    ET.main(itv_session)                              # writes task_evaluation.json/.md inside the itv session (this folder)
    itv = json.loads((Path(itv_session) / "task_evaluation.json").read_text(encoding="utf-8"))["rows"]
    v02 = v02_rows(gt)
    groups = {"P1 v0.2 (11-17) Qwen4B+DINO, fixed task types": v02,
              "P2 instruction_target_viewer (11-19) Qwen4B+DINO+Qwen roles": itv,
              "P3a multimodel (11-20) Qwen8B-NF4+DINO+SigLIP2": [r for r in mm if r["backend"] == "qwen3vl_8b_nf4"],
              "P3b multimodel (11-20) Qwen4B+DINO+SigLIP2": [r for r in mm if r["backend"] == "qwen3vl_4b_bf16"]}
    summ = {}
    for k, rows in groups.items():
        if k.startswith("P1"):
            c = Counter(r["outcome"] for r in rows)
            summ[k] = {"runs": len(rows), "success": 0, "A_right": 0, "wrong_commit": 0, "correct_abstain": c["CORRECT_ABSTAIN"],
                       "abstain_on_valid": c["ABSTAIN"], "check_manually": c["CHECK_MANUALLY"],
                       "first_failure": dict(Counter(r["first_failure"] for r in rows))}
        else:
            summ[k] = summarize(rows)
    oracle = oracle_rows()
    summ["P0 human-given targets + planner (navigation_handoff, I1 only)"] = {
        "photos": len(oracle), "A_and_B_valid": sum(r["both_valid"] for r in oracle), "A_valid": sum(r["path_A"] == "VALID" for r in oracle)}
    out = {"ground_truth_status": gt["status"], "summary": summ, "oracle_rows": oracle,
           "rows": {k: v for k, v in groups.items()}}
    (HERE / "comparison.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    for r in v02:
        print(r["photo_id"], r["instruction_key"], r["overall_status"], r["reason"][:120])
    for r in oracle:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1])
