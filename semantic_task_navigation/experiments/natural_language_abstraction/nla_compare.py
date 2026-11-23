#!/usr/bin/env python3
"""Comparison of one NLA run: results/<run>/<image>/<L1|L2|L3>/result.json -> results/<run>/comparison/.
Reuses the parent experiment's compare helpers (imported by file path, unchanged) and adds the fields that are
specific to this experiment: what the model says the instruction gave vs what it added itself.

It computes NO accuracy, IoU-vs-truth or success rate (there is no human-confirmed answer polygon). Box overlaps
between instruction levels compare the model's own boxes with each other (consistency). The keyword check below
is a mechanical heuristic that only flags claims for the reviewer to read; it is not a judgement.

  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python experiments/natural_language_abstraction/nla_compare.py experiments/natural_language_abstraction/results/<run_id>
"""
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nla_probe as nla  # noqa: E402  (also loads the parent probe as nla.asr)

asr = nla.asr
_spec = importlib.util.spec_from_file_location("asr_compare", nla.PARENT / "compare.py")
cmp = importlib.util.module_from_spec(_spec)
sys.modules["asr_compare"] = cmp
_spec.loader.exec_module(cmp)

# English term in a model statement -> the Korean word that would have to be in the instruction for the statement
# "this is given by the instruction" to be literally true.
TERM_TO_KOREAN = {"truck": "트럭", "loading": "하역", "unload": "하역", "ship": "출하", "dispatch": "출하",
                  "logistic": "물류", "next": "다음", "produc": "생산", "cube": "큐브", "red": "빨간"}


def not_in_instruction(statements, instruction):
    out = []
    for st in statements:
        low = st.lower()
        missing = sorted({k for t, k in TERM_TO_KOREAN.items() if t in low and k not in instruction})
        if missing:
            out.append({"statement": st, "terms_absent_from_instruction": missing})
    return out


def per_result(r):
    row = cmp.per_result(r)
    row["instruction"] = r["instruction"]
    s = r.get("structured")
    if s is None:
        return row
    ta = s["task_analysis"]
    added = ta["info_added_by_you"]
    row.update({"inferred_task_purpose": ta["inferred_task_purpose"],
                "info_given_by_instruction": ta["info_given_by_instruction"],
                "info_added_by_you": added,
                "info_added_by_basis": {b: sum(1 for a in added if a["basis"] == b)
                                        for b in ("image", "general_knowledge", "assumption")},
                "given_claims_not_literally_in_instruction": not_in_instruction(ta["info_given_by_instruction"], r["instruction"]),
                "objects_listed_with_count_0": [o["label"] for o in s["observed_objects"] if o["count"] == 0],
                "area_boxes_px": {a["id"]: [round(v, 1) for v in asr.norm_to_px(a["bbox_2d"], 1920, 1080)]
                                  for a in s["functional_areas"]}})
    return row


def build(run_dir):
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    res = cmp.load_results(run_dir)
    levels, imgs = manifest["levels"], sorted({k[0] for k in res})
    rows = [per_result(r) for r in res.values()]
    pairs = []

    def iou_of(a, b):
        da, db = cmp.dest_of(a), cmp.dest_of(b)
        return None if not (da and db) else round(asr.iou(da["box_px"], db["box_px"]), 3)
    for im in imgs:
        for a, b in combinations(levels, 2):
            if (im, a) in res and (im, b) in res:
                pairs.append({"kind": "same_image_different_instruction", "image": im, "a": a, "b": b,
                              "dest_iou": iou_of(res[(im, a)], res[(im, b)])})
    for lv in levels:
        for ia, ib in combinations(imgs, 2):
            if (ia, lv) in res and (ib, lv) in res:
                pairs.append({"kind": "same_instruction_different_image", "level": lv, "a": ia, "b": ib,
                              "dest_iou": iou_of(res[(ia, lv)], res[(ib, lv)])})
    return {"run_id": manifest["run_id"], "split": manifest["split"], "prompt_version": manifest["prompt_version"],
            "classification": manifest["classification"],
            "note": "Exploratory diagnostic. No accuracy is computed. dest_iou is None when either side withheld.",
            "results": sorted(rows, key=lambda x: (x["image_id"], x["condition"])), "pairwise_destination_iou": pairs}, res, levels


def to_markdown(summary):
    L = [f"# NLA 비교 요약 — run `{summary['run_id']}` (prompt {summary['prompt_version']}, {summary['classification']})", "",
         f"> {summary['note']}", "",
         "| 이미지 | 수준 | 상태 | 결정 | 목적지 기능 이름 | 신뢰도 | 목적지 박스(px) | 지시가 줬다고 주장 / 직접 일치하지 않는 주장 | 모델이 덧붙인 것 (image/지식/가정) | 토큰 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["results"]:
        d = r.get("destination")
        g = f"{len(r.get('info_given_by_instruction', []))} / {len(r.get('given_claims_not_literally_in_instruction', []))}" if r["status"] == "OK" else "-"
        b = r.get("info_added_by_basis")
        bs = f"{b['image']}/{b['general_knowledge']}/{b['assumption']}" if b else "-"
        L.append(f"| {r['image_id']} | {r['condition']} | {r['status']} | {r.get('decision', '-')} | "
                 f"{d['function_name'] if d else '-'} | {d['confidence'] if d else '-'} | {d['box_px'] if d else '-'} | {g} | {bs} | {r['n_output_tokens']} |")
    L += ["", "## 목적지 박스끼리의 겹침 (모델 자신의 박스끼리, 정확도 아님; 한쪽이라도 보류면 None)", "",
          "| 종류 | 대상 | a | b | IoU |", "|---|---|---|---|---|"]
    for p in summary["pairwise_destination_iou"]:
        L.append(f"| {p['kind']} | {p.get('image') or p.get('level')} | {p['a']} | {p['b']} | {p['dest_iou']} |")
    L += ["", "## 모델 응답 (원문 그대로)", ""]
    for r in summary["results"]:
        L.append(f"### {r['image_id']} / {r['condition']} — 지시: {r['instruction']}")
        if r["status"] != "OK":
            L += [f"- 상태 {r['status']}", ""]
            continue
        d = r.get("destination")
        L += [f"- 추론한 작업 목적: {r['inferred_task_purpose']}",
              f"- 결정: `{r['decision']}` / 필요한 장소: {r['place_needed']}", f"- 이유: {r['reason']}"]
        if d:
            L += [f"- 목적지 `{d['id']}` = {d['function_name']} ({d['confidence']}), 박스 {d['box_px']}",
                  f"  - 관찰 근거: {d['observed_evidence']}", f"  - 추론: {d['inference']}"]
        L += [f"- 지시문이 줬다고 한 것: {r['info_given_by_instruction']}",
              f"  - 그중 지시문에 글자로 없는 용어를 포함한 주장(휴리스틱): {r['given_claims_not_literally_in_instruction'] or '없음'}",
              f"- 모델이 덧붙인 것: {[(a['basis'], a['statement']) for a in r['info_added_by_you']]}",
              f"- 제안한 영역들: {r['area_functions']}", f"- 확인 불가로 적은 것: {r['unknown_from_image']}",
              f"- 물리 검증이 필요하다고 적은 것: {r['physical_conditions_to_verify']}",
              f"- 빨간 큐브 보고: {r['reports_red_cube'] or '없음'} / count 0으로 적은 객체: {r['objects_listed_with_count_0'] or '없음'}"]
        if r["consistency_issues"]:
            L.append(f"- 일관성 문제: {r['consistency_issues']}")
        L.append("")
    return "\n".join(L)


def main():
    run_dir = Path(sys.argv[1]).resolve()
    summary, res, levels = build(run_dir)
    out = run_dir / "comparison"
    out.mkdir(exist_ok=False)
    for im in sorted({k[0] for k in res}):
        p = cmp.panel(run_dir, res, im, levels)
        if p is not None:
            cv2.imwrite(str(out / f"{im}_panel.png"), p)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
