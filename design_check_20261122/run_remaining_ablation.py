#!/usr/bin/env python3
"""Finish the step-4 ablation of hybrid_exemplar_20261121 (ablation_offline.py was stopped by its 900 s timeout after 2 of 5 variants,
ablation_20261122_135357/). This file imports ablation_offline.py read-only and repeats its main loop UNCHANGED for the remaining
variants; outputs go to /home/binci/vlm_ws/design_check_20261122/ablation_rest_<time>/. Offline: stored embeddings only, no model call.
Run with python -B.
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HYB = Path("/home/binci/vlm_ws/hybrid_exemplar_20261121")
sys.path.insert(0, str(HYB))
import ablation_offline as A  # noqa: E402  (module-level code only reads config and data)

OUT = Path(__file__).resolve().parent
VARIANTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["siglip_so400m_text", "siglip_base_exemplar", "siglip_so400m_exemplar"]


def main():
    R, AC, M = A.R, A.AC, A.M
    gt = A.yaml.safe_load((R.BENCH / "ground_truth.yaml").read_text(encoding="utf-8"))
    root = OUT / f"ablation_rest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root.mkdir()
    shutil.copy(HYB / "ablation_config.yaml", root / "ablation_config_snapshot.yaml")
    D = {"base": A.load_emb(AC["data"]["base"])[0], "namer": {}}
    labels = json.loads((R.BASE / "raw" / f"{next(iter(gt['photos']))}.json").read_text(encoding="utf-8"))["labels"]
    for model in ("base", "so400m"):
        img, T, meta = A.load_emb(AC["data"][model])
        tix = {lb: i for i, lb in enumerate(meta["labels"])}
        Tm = T[[tix[lb] for lb in labels]]
        D["namer"][f"siglip_{model}_text"] = (lambda img, Tm, meta: lambda p, k: M.siglip_rank(img[(p, k)], Tm, meta["logit_scale"], meta["logit_bias"], labels)
                                              ["combined"][0][0])(img, Tm, meta)
    sys.path.insert(0, str(R.BENCH))
    import evaluate_tasks as E
    summary = {}
    for v in VARIANTS:
        vw = A.AblationViewer(root / v, v, D)
        photos = {p["photo_id"]: p for p in vw.list_photos()}
        index = []
        for pid in gt["photos"]:
            for ik in ("I1", "I2"):
                vw.slots_now = R.ASSUMED_SLOTS[ik]
                rec = vw.run(photos[pid]["path"], gt["instructions"][ik]["text"])
                s, d = rec["source"], rec["destination"]
                index.append({"backend": v, "photo_id": pid, "instruction_key": ik, "run_id": rec["run_id"], "wall_s": rec["wall_s"],
                              "first_failure": rec["first_failure"], "A_status": s.get("status"), "B_status": d.get("status"),
                              "A_ids": s.get("ids"), "A_why": s.get("why"), "A_ex": s.get("exemplar"), "B_id": d.get("id"), "B_why": d.get("why"),
                              "B_ex": d.get("exemplar")})
                print(v, pid, ik, s.get("status"), d.get("status"), flush=True)
        (root / v / "benchmark_index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        E.main(root / v)
        rows = json.loads((root / v / "task_evaluation.json").read_text(encoding="utf-8"))["rows"]
        summary[v] = {"I1_success": sum(r["outcome"] == "SUCCESS" for r in rows if r["instruction_key"] == "I1"),
                      "I1_A_right": sum(r["A_right"] for r in rows if r["instruction_key"] == "I1"),
                      "I1_B_right": sum(r["B_right"] for r in rows if r["instruction_key"] == "I1"),
                      "I2_A_right_valid": sum(r["A_right"] for r in rows if r["instruction_key"] == "I2" and r["valid_task"]),
                      "I2_correct_abstain": sum(r["outcome"] == "CORRECT_ABSTAIN" for r in rows if r["instruction_key"] == "I2"),
                      "wrong_commits": sum(r["outcome"] == "WRONG_COMMIT" for r in rows)}
    if "siglip_so400m_exemplar" in VARIANTS:
        ref = json.loads((A.REF_REPLAY / "benchmark_index.json").read_text(encoding="utf-8"))
        new = json.loads((root / "siglip_so400m_exemplar" / "benchmark_index.json").read_text(encoding="utf-8"))
        summary["_reproduces_replay_20261121_071147"] = len(ref) == len(new) and all(
            (a["A_status"], a["B_status"], (a["A"] or {}).get("exemplar"), (a["B"] or {}).get("exemplar")) == (b["A_status"], b["B_status"], b["A_ex"], b["B_ex"])
            for a, b in zip(ref, new))
    (root / "ablation_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n=== ABLATION SUMMARY ===")
    for v, s in summary.items():
        print(v, json.dumps(s, ensure_ascii=False))
    print("root:", root)


if __name__ == "__main__":
    main()
