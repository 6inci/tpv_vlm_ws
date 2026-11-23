#!/usr/bin/env python3
"""Offline ablation of step 4 only (ablation_config.yaml): which candidate is the cube / an empty pallet.
The hybrid decision code (HybridViewer._infer via replay_offline.ReplayViewer) is unchanged; only identity_scores() is swapped.
Stored outputs only (no model call). Slots are ASSUMED correct, as in the replay; I2 B (near-landmark step) is not replayed.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -B ablation_offline.py
Output: ablation_<stamp>/<variant>/ (one session per variant, scored by the unchanged 11-20 scorer) + ablation_summary.json
"""
import copy
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_offline as R  # noqa: E402

HP, M = R.HP, R.M
AC = yaml.safe_load((HERE / "ablation_config.yaml").read_text(encoding="utf-8"))
REF_REPLAY = HERE / "replay_20261121_071147"        # the so400m exemplar variant must reproduce this run


def load_emb(stem):
    stem = (HERE / stem).resolve()
    z, meta = np.load(f"{stem}.npz"), json.loads(Path(f"{stem}.json").read_text(encoding="utf-8"))
    pos, I = {tuple(k): i for i, k in enumerate(meta["index"])}, z["image"]     # read the array once (z["image"] decompresses on every access)
    img = {(p, k): np.stack([I[pos[(p, k, v)]] for v in HP.CFG["exemplar"]["views"]]) for p, k, _ in meta["index"]}
    return img, z["text"], meta


class AblationViewer(R.ReplayViewer):
    def __init__(self, session_dir, variant, D):
        super().__init__(session_dir)
        self.variant, self.D = variant, D
        self.hc = copy.deepcopy(HP.CFG)
        if AC["variants"][variant]["kind"] == "name":
            self.hc["decision"]["cube_min_score"] = AC["name_cube_min_score"]
        (self.session / "ablation_variant.json").write_text(json.dumps({"variant": variant, **AC["variants"][variant], "decision": self.hc["decision"]}, indent=1),
                                                            encoding="utf-8")

    def identity_scores(self, ph, cands, bank):
        v, pid = self.variant, ph["photo_id"]
        if v == "siglip_so400m_exemplar":
            return super().identity_scores(ph, cands, bank)
        if v == "siglip_base_exemplar":
            E = self.D["base"]
            embs = {x["id"]: E[(pid, x["id"])] for x in cands}
            return {cls: HP.exemplar_scores(embs, [E[(e["photo_id"], e["key"])] for e in bank if e["cls"] == cls]) for cls in self.hc["exemplar"]["classes"]}
        name = self.D["namer"][v]
        return {cls: {x["id"]: 1.0 if name(pid, x["id"]) in labs else 0.0 for x in cands} for cls, labs in AC["target_labels"].items()}


def main():
    gt = yaml.safe_load((R.BENCH / "ground_truth.yaml").read_text(encoding="utf-8"))
    root = HERE / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root.mkdir()
    shutil.copy(HERE / "ablation_config.yaml", root / "ablation_config_snapshot.yaml")
    D = {"base": load_emb(AC["data"]["base"])[0], "namer": {}}
    qwen = {(r["photo_id"], r["key"]): r["label"] for r in map(json.loads, (HERE / AC["data"]["qwen_replies"]).resolve().read_text(encoding="utf-8").splitlines())}
    D["namer"]["qwen_name"] = lambda p, k: qwen.get((p, k))
    labels = json.loads((R.BASE / "raw" / f"{next(iter(gt['photos']))}.json").read_text(encoding="utf-8"))["labels"]
    for model in ("base", "so400m"):
        img, T, meta = load_emb(AC["data"][model])
        tix = {lb: i for i, lb in enumerate(meta["labels"])}
        Tm = T[[tix[lb] for lb in labels]]
        D["namer"][f"siglip_{model}_text"] = (lambda img, Tm, meta: lambda p, k: M.siglip_rank(img[(p, k)], Tm, meta["logit_scale"], meta["logit_bias"], labels)
                                              ["combined"][0][0])(img, Tm, meta)
    summary = {}
    for v in AC["variants"]:
        vw = AblationViewer(root / v, v, D)
        photos = {p["photo_id"]: p for p in vw.list_photos()}
        index = []
        for pid in gt["photos"]:
            for ik in ("I1", "I2"):
                vw.slots_now = R.ASSUMED_SLOTS[ik]
                rec = vw.run(photos[pid]["path"], gt["instructions"][ik]["text"])
                s, d = rec["source"], rec["destination"]
                index.append({"backend": v, "photo_id": pid, "instruction_key": ik, "run_id": rec["run_id"], "wall_s": rec["wall_s"],
                              "first_failure": rec["first_failure"], "A_status": s.get("status"), "B_status": d.get("status"),
                              "A_ids": s.get("ids"), "A_why": s.get("why"), "A_ex": s.get("exemplar"), "B_id": d.get("id"), "B_why": d.get("why"), "B_ex": d.get("exemplar"),
                              "n_named": {c: int(sum(x >= 0.5 for x in sc.values())) for c, sc in (rec.get("exemplar") or {}).get("scores", {}).items()}
                              if AC["variants"][v]["kind"] == "name" else None})
        (root / v / "benchmark_index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        sys.path.insert(0, str(R.BENCH))
        import evaluate_tasks as E
        E.main(root / v)
        ev = json.loads((root / v / "task_evaluation.json").read_text(encoding="utf-8"))
        rows = ev["rows"]
        summary[v] = {"I1_success": sum(r["outcome"] == "SUCCESS" for r in rows if r["instruction_key"] == "I1"),
                      "I1_A_right": sum(r["A_right"] for r in rows if r["instruction_key"] == "I1"),
                      "I1_B_right": sum(r["B_right"] for r in rows if r["instruction_key"] == "I1"),
                      "I2_A_right_valid": sum(r["A_right"] for r in rows if r["instruction_key"] == "I2" and r["valid_task"]),
                      "I2_correct_abstain": sum(r["outcome"] == "CORRECT_ABSTAIN" for r in rows if r["instruction_key"] == "I2"),
                      "wrong_commits": sum(r["outcome"] == "WRONG_COMMIT" for r in rows),
                      "A_status_I1": [r["A_status"] for r in rows if r["instruction_key"] == "I1"],
                      "B_status_I1": [r["B_status"] for r in rows if r["instruction_key"] == "I1"],
                      "n_named_I1": [x["n_named"] for x in index if x["instruction_key"] == "I1"]}
    # reproduction check: the so400m exemplar variant vs the earlier replay (decisions and scores)
    ref = json.loads((REF_REPLAY / "benchmark_index.json").read_text(encoding="utf-8"))
    new = json.loads((root / "siglip_so400m_exemplar" / "benchmark_index.json").read_text(encoding="utf-8"))
    same = all((a["A_status"], a["B_status"], (a["A"] or {}).get("exemplar"), (a["B"] or {}).get("exemplar")) ==
               (b["A_status"], b["B_status"], b["A_ex"], b["B_ex"]) for a, b in zip(ref, new)) and len(ref) == len(new)
    summary["_reproduces_replay_20261121_071147"] = same
    (root / "ablation_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n=== ABLATION SUMMARY ===")
    for v, s in summary.items():
        print(v, json.dumps(s, ensure_ascii=False))
    print("root:", root)


if __name__ == "__main__":
    main()
