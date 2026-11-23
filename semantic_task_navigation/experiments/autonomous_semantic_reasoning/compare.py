#!/usr/bin/env python3
"""Cross-condition comparison for one probe run. Reads results/<run>/<image>/<condition>/result.json and writes
results/<run>/comparison/{summary.json, summary.md, <image>_panel.png}.

It computes NO accuracy. Overlaps between conditions or with the model's own object boxes describe consistency,
not correctness: there is no human-confirmed answer region for these photos.

  /home/binci/miniconda3/envs/stnav_gui/bin/python experiments/autonomous_semantic_reasoning/compare.py results/<run_id>
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe  # noqa: E402


def load_results(run_dir):
    out = {}
    for p in sorted(Path(run_dir).glob("*/*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[(r["image_id"], r["condition"])] = r
    return out


def dest_of(r, w=1920, h=1080):
    s = r.get("structured")
    if not s or s["task_analysis"]["decision"] != "proposed":
        return None
    a = next((a for a in s["functional_areas"] if a["id"] == s["task_analysis"]["destination_area_id"]), None)
    return None if a is None else {"area": a, "box_px": probe.norm_to_px(a["bbox_2d"], w, h)}


def mentions(text_parts, words):
    t = " ".join(text_parts).lower()
    return any(w in t for w in words)


def per_result(r, w=1920, h=1080):
    s = r.get("structured")
    row = {"image_id": r["image_id"], "condition": r["condition"], "status": r["status"],
           "n_output_tokens": r["call"]["n_output_tokens"], "latency_s": r["call"]["latency_s"],
           "hit_max_new_tokens": r["call"]["hit_max_new_tokens"], "consistency_issues": r["consistency_issues"]}
    if s is None:
        return row
    row.update({"decision": s["task_analysis"]["decision"], "n_objects": len(s["observed_objects"]),
                "n_areas": len(s["functional_areas"]), "place_needed": s["task_analysis"]["place_needed_for_task"],
                "reason": s["task_analysis"]["reason"], "n_unknown": len(s["unknown_from_image"]),
                "n_physical_conditions": len(s["physical_conditions_to_verify"]),
                "unknown_from_image": s["unknown_from_image"],
                "physical_conditions_to_verify": s["physical_conditions_to_verify"],
                "objects": [f"{o['label']} x{o['count']} ({o['location_in_image']})" for o in s["observed_objects"]],
                "reports_red_cube": [f"{o['label']} x{o['count']} @ {o['location_in_image']}: {o['visible_evidence']}"
                                     for o in s["observed_objects"] if "cube" in o["label"].lower()],
                "area_functions": [f"{a['id']}: {a['function_name']} ({a['confidence']})" for a in s["functional_areas"]]})
    d = dest_of(r, w, h)
    if d:
        a, b = d["area"], d["box_px"]
        row["destination"] = {"id": a["id"], "function_name": a["function_name"], "confidence": a["confidence"],
                              "inference": a["inference"], "observed_evidence": a["observed_evidence"],
                              "box_px": [round(v, 1) for v in b],
                              "area_frac_of_image": round(probe.box_area(b) / (w * h), 4),
                              "center_px": [round((b[0] + b[2]) / 2, 1), round((b[1] + b[3]) / 2, 1)],
                              "cites_truck": mentions([a["function_name"], a["inference"]] + a["observed_evidence"], ["truck"]),
                              "cites_floor_line_or_bay": mentions(a["observed_evidence"] + [a["inference"]],
                                                                   ["dashed", "lines", "parking", "boundary", "border", "outline"])}
        # relation to the model's OTHER proposed areas (its own boxes; descriptive only)
        rel = []
        for o in s["functional_areas"]:
            if o["id"] == a["id"]:
                continue
            ob = probe.norm_to_px(o["bbox_2d"], w, h)
            inter = probe.inter_area(b, ob)
            if inter > 0:
                rel.append({"id": o["id"], "function_name": o["function_name"], "iou": round(probe.iou(b, ob), 3),
                            "other_inside_destination": round(inter / max(probe.box_area(ob), 1e-9), 2)})
        row["overlapping_other_proposed_areas"] = rel
        row["truck_areas_proposed"] = [o["id"] for o in s["functional_areas"]
                                       if mentions([o["function_name"]] + o["observed_evidence"], ["truck"])]
    return row


def build(run_dir):
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    res = load_results(run_dir)
    rows = [per_result(r) for r in res.values()]
    imgs = sorted({k[0] for k in res})
    conds = [c for c in manifest["conditions"]]
    pair = []
    for im in imgs:                                   # same image, different condition
        for a, b in combinations(conds, 2):
            da, db = dest_of(res[(im, a)]) if (im, a) in res else None, dest_of(res[(im, b)]) if (im, b) in res else None
            pair.append({"kind": "same_image_different_condition", "image": im, "a": a, "b": b,
                         "dest_iou": None if not (da and db) else round(probe.iou(da["box_px"], db["box_px"]), 3)})
    for c in conds:                                   # same condition, different image (cube position only)
        for ia, ib in combinations(imgs, 2):
            da, db = dest_of(res[(ia, c)]) if (ia, c) in res else None, dest_of(res[(ib, c)]) if (ib, c) in res else None
            pair.append({"kind": "same_condition_different_image", "condition": c, "a": ia, "b": ib,
                         "dest_iou": None if not (da and db) else round(probe.iou(da["box_px"], db["box_px"]), 3)})
    return {"run_id": manifest["run_id"], "prompt_version": manifest["prompt_version"], "split": manifest["split"],
            "note": "No accuracy is computed. dest_iou compares the model's own boxes with each other (consistency).",
            "results": rows, "pairwise_destination_iou": pair}, res


def panel(run_dir, res, image_id, conds, width=900):
    tiles = []
    for c in conds:
        p = Path(run_dir) / image_id / c / "overlay_areas.png"
        if p.exists():
            im = cv2.imread(str(p))
            tiles.append(cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA))
    if not tiles:
        return None
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
    return np.hstack(tiles)


def to_markdown(summary):
    L = [f"# 조건 비교 요약 — run `{summary['run_id']}` (prompt {summary['prompt_version']}, split {summary['split']})", "",
         f"> {summary['note']}", "",
         "| 이미지 | 조건 | 상태 | 결정 | 목적지 기능 이름 | 신뢰도 | 목적지 박스(px) | 이미지 대비 면적 | 트럭을 근거로 인용 | 영역/객체 수 | 토큰 | 시간(s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["results"]:
        d = r.get("destination")
        L.append(f"| {r['image_id']} | {r['condition']} | {r['status']} | {r.get('decision', '-')} | "
                 f"{d['function_name'] if d else '-'} | {d['confidence'] if d else '-'} | {d['box_px'] if d else '-'} | "
                 f"{d['area_frac_of_image'] if d else '-'} | "
                 f"{d['cites_truck'] if d else '-'} | "
                 f"{r.get('n_areas', '-')}/{r.get('n_objects', '-')} | {r['n_output_tokens']} | {r['latency_s']} |")
    L += ["", "## 목적지 박스끼리의 겹침 (IoU, 일관성 지표이며 정확도가 아님)", "", "| 종류 | 대상 | a | b | IoU |", "|---|---|---|---|---|"]
    for p in summary["pairwise_destination_iou"]:
        L.append(f"| {p['kind']} | {p.get('image') or p.get('condition')} | {p['a']} | {p['b']} | {p['dest_iou']} |")
    L += ["", "## 모델 응답 요약 (원문 그대로)", ""]
    for r in summary["results"]:
        L.append(f"### {r['image_id']} / {r['condition']}")
        if r["status"] != "OK":
            L += [f"- 상태 {r['status']}", ""]
            continue
        d = r.get("destination")
        L += [f"- decision: `{r['decision']}`, 필요한 장소: {r['place_needed']}", f"- 이유: {r['reason']}"]
        if d:
            L += [f"- 목적지 `{d['id']}` = {d['function_name']} ({d['confidence']})", f"  - 관찰 근거: {d['observed_evidence']}",
                  f"  - 추론: {d['inference']}"]
        L += [f"- 제안한 영역들: {r['area_functions']}", f"- 관찰한 객체: {r['objects']}",
              f"- 빨간 큐브 보고: {r['reports_red_cube'] or '없음'}", f"- 확인 불가로 적은 것: {r['unknown_from_image']}",
              f"- 물리 검증이 필요하다고 적은 것: {r['physical_conditions_to_verify']}"]
        if d:
            L += [f"- 목적지와 겹치는 다른 제안 영역: {r.get('overlapping_other_proposed_areas')}",
                  f"- 트럭을 근거로 든 영역: {r.get('truck_areas_proposed')}"]
        if r["consistency_issues"]:
            L += [f"- 일관성 문제: {r['consistency_issues']}"]
        L.append("")
    return "\n".join(L)


def main():
    run_dir = Path(sys.argv[1]).resolve()
    summary, res = build(run_dir)
    out = run_dir / "comparison"
    out.mkdir(exist_ok=False)
    for im in sorted({k[0] for k in res}):
        p = panel(run_dir, res, im, summary_conditions(run_dir))
        if p is not None:
            cv2.imwrite(str(out / f"{im}_panel.png"), p)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print(f"wrote {out}")


def summary_conditions(run_dir):
    return json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))["conditions"]


if __name__ == "__main__":
    main()
