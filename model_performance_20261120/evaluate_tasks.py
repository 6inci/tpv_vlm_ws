#!/usr/bin/env python3
"""Score a benchmark session against ground_truth.yaml. Usage: evaluate_tasks.py <session_dir>
Writes <session_dir>/task_evaluation.json and task_evaluation.md only.

Commit = the pipeline's own final status SELECTED (TYPE_UNCERTAIN / AMBIGUOUS / NOT_FOUND / NOT_ACCESSIBLE are not commits).
Outcomes per run:
  SUCCESS         valid task, A committed on the right object and B committed on an acceptable target
  A_ONLY          valid task, A right, B not committed (abstained / not accessible)
  CORRECT_ABSTAIN invalid task (no graspable object at the named place) and A not committed
  ABSTAIN         valid task and A not committed
  ABSTAIN_B_WRONG valid task, A not committed, but B committed on a wrong target (not executable: handoff has no source)
  WRONG_COMMIT    A committed on a wrong object, or A right and B committed on a wrong target
                  (the dangerous case: the robot would act on it)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BAD_B_WORDS = ("robot arm", "conveyor", "production line", "machine")


def centre_in(b, r, t=0):
    if not b or len(b) < 4:
        return False
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return r[0] - t <= cx <= r[2] + t and r[1] - t <= cy <= r[3] + t


def main(session):
    session = Path(session)
    gt = yaml.safe_load((HERE / "ground_truth.yaml").read_text(encoding="utf-8"))
    tol = gt["match_tolerance_px"]
    idx = json.loads((session / "benchmark_index.json").read_text(encoding="utf-8"))
    rows = []
    for it in idx:
        rec = json.loads((session / "runs" / it["run_id"] / "run.json").read_text(encoding="utf-8"))
        calls = json.loads((session / "runs" / it["run_id"] / "model_calls.json").read_text(encoding="utf-8"))
        g, ik = gt["photos"][it["photo_id"]], it["instruction_key"]
        ins = gt["instructions"][ik]
        s, d = rec["source"], rec["destination"]
        a_commit = s.get("status") == "SELECTED"
        a_boxes = s.get("boxes") or []
        a_right = a_commit and len(a_boxes) == 1 and centre_in(a_boxes[0], g["cube_pallet"], tol)
        b_commit = d.get("status") == "SELECTED"
        bb = d.get("bbox_xyxy")
        appearance = " ".join(map(str, ((d.get("ratings") or {}).get(d.get("id")) or {}).get("appearance") or [])).lower()
        if ik == "I1":
            valid_a, border = True, False
            b_right = b_commit and any(centre_in(bb, e, tol) for e in g["empty_pallets"])
        else:
            valid_a = it["photo_id"] in ins["valid_a"]
            border = it["photo_id"] in ins["borderline_a"]
            b_right = b_commit and centre_in(bb, ins["conveyor_near_box"]) and not any(w in appearance for w in BAD_B_WORDS)
        # The handoff carries no source when A is not committed, so a robot would not act on B alone.
        if (a_commit and not a_right) or (a_right and b_commit and not b_right):
            outcome = "WRONG_COMMIT"
        elif valid_a or (border and a_commit):
            outcome = "SUCCESS" if (a_right and b_right) else ("A_ONLY" if a_right else
                                                               ("ABSTAIN_B_WRONG" if b_commit and not b_right else "ABSTAIN"))
        else:
            outcome = "CORRECT_ABSTAIN"
        # What the VLM itself pointed at, even when Python did not commit it (TYPE_UNCERTAIN etc.).
        # TRUCK_BED_BOX = the cardboard box on the truck bed (not graspable by this robot), region declared before scoring.
        def what(b):
            if centre_in(b, g["cube_pallet"], tol):
                return "CUBE"
            if centre_in(b, [1540, 0, 1700, 110]):
                return "TRUCK_BED_BOX"
            return "OTHER"
        a_intent = [what(b) for b in a_boxes]
        vlm = [c for c in calls if "Qwen" in str(c.get("model", "")) or "qwen" in str(c.get("model", "")).lower()]
        rows.append({**it, "A_commit": a_commit, "A_right": a_right, "A_box": a_boxes, "A_vlm_pointed_at": a_intent,
                     "A_type_check": (s.get("type_check") or {}).get("status"), "B_commit": b_commit, "B_right": b_right,
                     "B_box": bb, "B_appearance": appearance, "B_goal_kind": d.get("goal_kind"), "outcome": outcome,
                     "valid_task": bool(valid_a), "borderline": border, "vlm_calls": len(vlm),
                     "vlm_latency_s": round(sum(float(c.get("latency_s") or 0) for c in vlm), 1),
                     "peak_vram_gib": max([float(c.get("peak_vram_gib") or 0) for c in calls] or [0])})
    summ = defaultdict(lambda: Counter())
    for r in rows:
        k = (r["backend"], r["instruction_key"])
        summ[k][r["outcome"]] += 1
        summ[k]["runs"] += 1
        summ[k]["A_right"] += int(r["A_right"])
        summ[k]["B_right"] += int(r["B_right"])
        summ[k]["wall_s_total"] += r["wall_s"]
    out = {"ground_truth_status": gt["status"], "rows": rows,
           "summary": {f"{b}|{i}": dict(c) for (b, i), c in sorted(summ.items())}}
    (session / "task_evaluation.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["| backend | instr | photo | A status | VLM pointed at | A right | B status | B right | outcome | first failure | wall s |",
             "|" + "---|" * 11]
    for r in rows:
        lines.append(f"| {r['backend']} | {r['instruction_key']} | {r['photo_id']} | {r['A_status']} | {','.join(r['A_vlm_pointed_at']) or '-'} | "
                     f"{r['A_right']} | {r['B_status']} | {r['B_right']} | {r['outcome']} | {r['first_failure']} | {r['wall_s']} |")
    (session / "task_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
