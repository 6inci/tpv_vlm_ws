#!/usr/bin/env python3
"""OFFLINE REPLAY of the hybrid decision code (HybridViewer._infer, unchanged) on STORED data. No GPU, no model call.

Why: the GPU needs a reboot (2026-11-21 07:09 kernel: NVRM Xid 31/154, "uvm ... requiring os reboot to recover").
What is replayed from storage (siglip_small_object_test, 2026-11-20, read only):
  - DINO candidates + facility boxes + robot start of the same MMViewer photo stage  (results/20261120_133027_run/raw/<id>.json)
  - SigLIP2 so400m embeddings of the same crops                                     (results/20261120_141616_followup_siglip_so400m/raw/)
What is NOT real model output here:
  - Qwen slots: ASSUMED correct (fixed JSON per instruction below). The real text-only Qwen call must run after the reboot.
  - Near-landmark destination (I2 B): needs Qwen + SigLIP2 base -> not replayed (recorded as model_worker FAILED: REPLAY_NO_GPU).
  - photo_6's added empty pallet has no stored crop -> it is not in the exemplar bank (the live run would include it).
Output: replay_<stamp>/ in this folder, scored by the unchanged 11-20 scorer.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -B replay_offline.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid_pipeline as HP  # noqa: E402

M = HP.M
SST = Path(HP.CFG["reference_objects"]) / "results"
BASE, XRUN = SST / "20261120_133027_run", SST / "20261120_141616_followup_siglip_so400m"
BENCH = Path("/home/binci/vlm_ws/model_performance_20261120")
ASSUMED_SLOTS = {   # what a correct parse of each instruction is (NOT a model output)
    "I1": {"object": "red cube", "object_words": "빨간 큐브", "source_landmark": "none", "source_relation": "none",
           "destination_kind": "empty_pallet", "destination_landmark": "none", "destination_relation": "none", "destination_words": "비어 있는 팔레트"},
    "I2": {"object": "any object", "object_words": "물체", "source_landmark": "truck_loading_bay", "source_relation": "near",
           "destination_kind": "near_landmark", "destination_landmark": "conveyor", "destination_relation": "near", "destination_words": "컨베이어 벨트 옆"}}


class ReplayViewer(HP.HybridViewer):
    def __init__(self, session_dir):
        super().__init__(session_dir=session_dir)
        z = np.load(XRUN / "raw" / "embeddings_siglip_so400m.npz")
        meta = json.loads((XRUN / "raw" / "embeddings_siglip_so400m.json").read_text(encoding="utf-8"))
        assert meta["model"] == HP.CFG["exemplar_model"]["model_id"]
        pos, I = {tuple(k): i for i, k in enumerate(meta["index"])}, z["image"]     # read the array once (z["image"] decompresses on every access)
        self.stored = {(p, k): np.stack([I[pos[(p, k, v)]] for v in HP.CFG["exemplar"]["views"]]) for p, k, _ in meta["index"]}
        self.slots_now = None

    def photo(self, path):
        pid = self.pid_for(path)
        if pid in self.photos:
            return self.photos[pid]
        d = json.loads((BASE / "raw" / f"{pid}.json").read_text(encoding="utf-8"))
        assert d["sha256"] == self._sha(path)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        geo = self.geometry()
        ph = {"photo_id": pid, "path": str(path), "sha256": d["sha256"], "size": [img.shape[1], img.shape[0]], "img": img, "calls": [],
              "alignment": M.check_alignment({"reference_image": geo["e1_path"], "alignment_check": {"max_shift_px": 3.0, "min_response": 0.3}}, img, lambda x: x)
              if geo["ok"] else {"status": "NOT_AVAILABLE", "reason": geo["why"]},
              "refs": d["facility_refs"], "dino": d["dino"], "phrases_done": [], "cube": None, "cube_how": "not used",
              "detection_how": f"REPLAY: stored DINO candidates and facility boxes of {BASE.name} (same MMViewer photo stage)"}
        self.photos[pid] = ph
        return ph

    def build_bank(self, log):
        if self.bank is None:
            kinds = {v["reference_kind"]: k for k, v in HP.CFG["exemplar"]["classes"].items()}
            self.bank = [{"cls": kinds[o["kind"]], "photo_id": pid, "key": o["key"], "box": o["box"], "slot": HP.slot_box(objs, o), "embs": self.stored[(pid, o["key"])]}
                         for pid, objs in self.refs.items() for o in objs if o["kind"] in kinds and (pid, o["key"]) in self.stored]
        return self.bank

    def gated(self, ph, log):
        g = HP.CFG["candidate_gate"]
        cands = [x for x in ph["dino"] if g["min_side_px"] <= HP.side_px(x["bbox_xyxy"]) <= g["max_side_px"]]
        for x in cands:
            self.xemb[(ph["photo_id"], x["id"])] = self.stored[(ph["photo_id"], x["id"])]
        return cands

    def vlm_text(self, prompt, max_new_tokens, stage, prompt_id, log):
        return json.dumps(self.slots_now, ensure_ascii=False)       # ASSUMED, not a model output

    def siglip(self, *a, **k):
        raise M.ModelWorkerError("REPLAY_NO_GPU", "near-landmark destination needs SigLIP2 base + Qwen; not replayed")

    def vlm(self, *a, **k):
        raise M.ModelWorkerError("REPLAY_NO_GPU", "VLM not available in the replay")


def main():
    gt = yaml.safe_load((BENCH / "ground_truth.yaml").read_text(encoding="utf-8"))
    session = HERE / f"replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    v = ReplayViewer(session)
    photos = {p["photo_id"]: p for p in v.list_photos()}
    index = []
    for pid in gt["photos"]:
        for ik in ("I1", "I2"):
            v.slots_now = ASSUMED_SLOTS[ik]
            rec = v.run(photos[pid]["path"], gt["instructions"][ik]["text"])
            rec["REPLAY"] = {"slots": "ASSUMED_CORRECT_NOT_MODEL_OUTPUT", "data": [str(BASE), str(XRUN)]}
            (Path(rec["files"]["run_json"])).write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            row = {"backend": "REPLAY_assumed_slots", "photo_id": pid, "instruction_key": ik, "run_id": rec["run_id"], "wall_s": rec["wall_s"],
                   "first_failure": rec["first_failure"], "A_status": rec["source"].get("status"), "B_status": rec["destination"].get("status"),
                   "A": {k: rec["source"].get(k) for k in ("why", "exemplar")}, "B": {k: rec["destination"].get(k) for k in ("why", "exemplar", "occupied_excluded")},
                   "bank": rec.get("exemplar", {}).get("bank_used"), "n_gated": rec.get("exemplar", {}).get("n_gated")}
            index.append(row)
            print(pid, ik, row["A_status"], row["B_status"], "|", (rec["source"].get("why") or "")[:110], "|", (rec["destination"].get("why") or "")[:140], flush=True)
    (session / "benchmark_index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    sys.path.insert(0, str(BENCH))
    import evaluate_tasks as E
    E.main(session)
    print("session:", session)


if __name__ == "__main__":
    main()
