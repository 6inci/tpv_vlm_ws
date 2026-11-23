#!/usr/bin/env python3
"""판별 자료 추출 (GPU, 모델 호출은 여기서만). 사진 묶음 하나의 모든 사진에 대해 저장한다 -> data/<set>/

1. 기존 사진 단계 그대로 (hybrid_exemplar_20261121 -> MMViewer: DINO 전체 사진 + 시설 주변, SigLIP2 base 시설 확인)
2. 작업 공간 띠 확대 DINO (config extract: 띠 폭 15 cm, 타일 400 px, 2배) -> 같은 후보 목록에 합침 (IoU 0.5)
3. 크기 조건(10-90 px)을 통과한 후보마다 SigLIP2 so400m 임베딩 (딱 맞게 / 주변 포함 2장)
4. 받침 확인 crop (후보 긴 변 x 2.4 정사각형) 임베딩
5. 예시 묶음: 개발 8장 기준 박스의 큐브 / 빈 팔레트 / 큐브가 올라간 팔레트
점수 계산과 정책은 모델 없이 policy.py가 한다.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -B extract.py --set dev     (GPU, Bash 샌드박스 밖)
  /home/binci/miniconda3/envs/stnav_gui/bin/python -B extract.py --set test
"""
import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as CM  # noqa: E402

sys.path.insert(0, "/home/binci/vlm_ws/hybrid_exemplar_20261121")
import hybrid_pipeline as HP  # noqa: E402  (읽기만)

M, P = HP.M, HP.P
EX = CM.C["extract"]


class Extractor(HP.HybridViewer):
    def __init__(self, session):
        super().__init__(session_dir=session)
        self.hc = copy.deepcopy(HP.CFG)                  # 이 객체에서만 바꿈 (기존 설정 파일은 그대로)
        self.hc["exemplar"]["classes"] = {k: {"reference_kind": v} for k, v in EX["exemplar_classes"].items()}
        self.hc["candidate_gate"] = dict(EX["candidate_gate"])

    def band_dets(self, ph, cam, log):
        rois = CM.band_rois_px(cam, EX["band_max_m"], EX["tile_px"], ph["size"])
        c, det, z, out = self.model(), self.tc["detection"], EX["zoom"], []
        d = self.session / "photos" / f"{ph['photo_id']}_inputs"
        d.mkdir(parents=True, exist_ok=True)
        for k, (x0, y0, x1, y1) in enumerate(rois):
            zp = d / f"band_{k:02d}.png"
            cv2.imwrite(str(zp), cv2.resize(ph["img"][y0:y1, x0:x1], ((x1 - x0) * z, (y1 - y0) * z), interpolation=cv2.INTER_CUBIC))
            raw = P.dino_call(c, zp, EX["band_phrases"], det["thresholds"], f"dino_band_{ph['photo_id']}_{k:02d}", log)
            out += P.detections(raw, f"BAND_{k:02d}", [x0, y0, x1, y1], z, det["max_area_frac"])
        return rois, out


def sup_box(b, scale):
    cx, cy, s = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1]) * scale / 2
    return [cx - s, cy - s, cx + s, cy + s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(CM.C["photo_sets"]))
    a = ap.parse_args()
    out = HERE / "data" / a.set
    out.mkdir(parents=True, exist_ok=True)
    cam = CM.Cam(a.set)
    vw = Extractor(out / "session")
    log, t0 = [], time.time()
    views = vw.hc["exemplar"]["views"]
    try:
        bank = vw.build_bank(log)
        np.savez_compressed(out / "bank_emb.npz", **{f"{e['photo_id']}|{e['key']}": e["embs"] for e in bank})
        (out / "bank.json").write_text(json.dumps([{k: v for k, v in e.items() if k != "embs"} for e in bank], indent=1), encoding="utf-8")
        photos, cemb, semb = [], {}, {}
        for p in sorted(Path(CM.C["photo_sets"][a.set]["dir"]).glob("*.png")):
            tp = time.time()
            ph = vw.photo(str(p))
            n0 = len(ph["dino"])
            rois, dets = vw.band_dets(ph, cam, ph["calls"])
            vw._add(ph, dets)
            cands = vw.gated(ph, log)
            for x in cands:
                cemb[f"{ph['photo_id']}|{x['id']}"] = vw.xemb[(ph["photo_id"], x["id"])]
            crops = {x["id"]: vw._crops(ph, f"{x['id']}_sup", sup_box(x["bbox_xyxy"], EX["support_scale"])) for x in cands}
            E = vw._embed([crops[x["id"]][v]["path"] for x in cands for v in views], f"support_{ph['photo_id']}", log)
            for i, x in enumerate(cands):
                semb[f"{ph['photo_id']}|{x['id']}"] = E[i * len(views):(i + 1) * len(views)]
            gated = {x["id"] for x in cands}
            photos.append({"photo_id": ph["photo_id"], "path": str(p), "sha256": ph["sha256"], "size": ph["size"],
                           "refs": {k: {kk: v.get(kk) for kk in ("box", "how")} for k, v in ph["refs"].items()},
                           "band_rois_px": rois, "n_dino_before_band": n0, "n_dino_after_band": len(ph["dino"]),
                           "candidates": [{**{k: x.get(k) for k in ("id", "bbox_xyxy", "score", "phrases", "conditions")}, "gated": x["id"] in gated,
                                           "support_box": sup_box(x["bbox_xyxy"], EX["support_scale"]) if x["id"] in gated else None} for x in ph["dino"]],
                           "seconds": round(time.time() - tp, 1)})
            print(f"{ph['photo_id']}: DINO {n0} -> 띠 확대 뒤 {len(ph['dino'])} (타일 {len(rois)}), 판별 후보 {len(cands)}, {time.time() - tp:.0f} s", flush=True)
        np.savez_compressed(out / "cand_emb.npz", **cemb)
        np.savez_compressed(out / "support_emb.npz", **semb)
        meta = {"created": datetime.now().isoformat(timespec="seconds"), "set": a.set, "config": CM.C, "exemplar_model": vw.hc["exemplar_model"],
                "views": views, "photos": photos, "wall_s": round(time.time() - t0, 1)}
        (out / "features.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        (out / "model_calls.json").write_text(json.dumps(log, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"saved {out}  ({time.time() - t0:.0f} s)")
    finally:
        vw.close()


if __name__ == "__main__":
    main()
