#!/usr/bin/env python3
"""V1 (truck-centred, semantic_to_navigation_integration) vs V2 (evidence + purpose, verifier2) on the SAME inputs.

  stored    the VLM proposals saved in the three integration runs (DEV_REF, E1, REF_P01) -- offline, no model call
  synthetic small hand-typed cases for situations the stored runs do not contain (scenarios.py)

Both stages write into a NEW timestamp folder under results/. Nothing is overwritten. No threshold is tuned: V2's relation
thresholds are v0.2's, and V1 is replayed with the values V1 stored (including its `--set` variants).

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  PYTHONDONTWRITEBYTECODE=1 $PY experiments/semantic_evidence_verification/compare.py [--stage stored|synthetic|both]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import observations as O  # noqa: E402
import scenarios as SC  # noqa: E402
import v1_replay as R  # noqa: E402
import verifier2 as V  # noqa: E402
from semantic_task_nav.config import load_config  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_cfg():
    return V.load_config(), load_config()["planning"]


# ------------------------------------------------------------------ agreement between V1 and V2
def v1_failed_ids(v1):
    return [c["id"] for c in v1["checks"] if c["status"] == "FAILED" and c["required"]]


def v1_not_run_ids(v1):
    return [c["id"] for c in v1["checks"] if c["status"] == "NOT_RUN"]


def agreement(v1_decision, v2_cand):
    """Two levels. semantic: did each accept the SEMANTICS of the proposal? hand-over: did each let it reach the planner?
    (no ground truth: 'false reject/accept' below are candidates to be judged, with the reason, not verdicts.)"""
    v1_sem = v1_decision.startswith("ADOPTED")
    v1_hand = v1_decision == "ADOPTED"
    v2_sem = v2_cand["semantic_verdict"] == V.SEMANTICALLY_SUPPORTED
    v2_hand = v2_cand["final_disposition"] == V.READY and not v2_cand["planning_preconditions"]["blocked_by"]
    v2_ready = v2_cand["final_disposition"] == V.READY
    if v1_sem == v2_sem:
        sem = "AGREE_ACCEPT" if v1_sem else "AGREE_BLOCK"
    else:
        sem = "V1_ACCEPTS__V2_BLOCKS (V1 false-accept candidate)" if v1_sem else \
              "V1_BLOCKS__V2_ACCEPTS (V1 false-reject / conflation candidate)"
    hand = "AGREE_PASS" if (v1_hand and v2_ready) else "AGREE_BLOCK" if (not v1_hand and not v2_ready) else \
           "V1_PASSES__V2_BLOCKS" if v1_hand else "V1_BLOCKS__V2_PASSES"
    return {"semantic_level": sem, "hand_over_level": hand, "v1_semantic_accept": v1_sem, "v2_semantic_accept": v2_sem,
            "v1_hand_over": v1_hand, "v2_hand_over_if_preconditions_met": v2_ready, "v2_hand_over": v2_hand}


# ------------------------------------------------------------------ overlay
def _t(img, text, org, color, scale=0.55):
    text = str(text).encode("ascii", "replace").decode()
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + bl), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def render_case(image, case, v2c, v1_rows, agr):
    img = image.copy()
    cand = case["candidates"][0]
    b = [int(round(v)) for v in cand["box_px"]]
    cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 165, 255), 4)
    _t(img, f"[VLM proposal] {cand['function_claimed']}", (b[0] + 4, b[1] + 22), (0, 165, 255))
    for ce in v2c["class_evidence"]:
        for k, i in enumerate(ce["instances"]):
            bb = [int(round(v)) for v in i["bbox_xyxy"]]
            col = (60, 200, 60) if i["relation"] == V.HOLDS else (60, 60, 255)
            cv2.rectangle(img, tuple(bb[:2]), tuple(bb[2:]), col, 3)
            if k == 0:                      # one label per class: instance count and relation summary
                _t(img, f"{ce['class']} x{len(ce['instances'])}: {ce['n_related']} related" +
                   (f" (gap {i['gap_px']}px)" if len(ce["instances"]) == 1 else ""), (bb[0], max(18, bb[1] - 6)), col, 0.5)
    for loc in case["observations"]["localizations"]:
        lb = [int(round(v)) for v in loc["box_px"]]
        cv2.rectangle(img, tuple(lb[:2]), tuple(lb[2:]), (255, 190, 0), 2)
    lines = [((255, 255, 255), f"{case['id']} | purpose from the INSTRUCTION: {v2c['purpose']['purposes'] or 'none'} "
                               f"needs {v2c['purpose']['requires_any_of'] or '-'}"),
             ((0, 165, 255), "V1 (truck-centred): " + " | ".join(f"{r['label']}: {r['decision']} failed={r['failed']} not_run={r['not_run']}" for r in v1_rows)),
             ((60, 220, 220), f"V2 semantic verdict: {v2c['semantic_verdict']}  -> final: {v2c['final_disposition']}"),
             ((200, 200, 200), f"V2 placement: {v2c['placement']['readiness']} | planner support: {v2c['planner_support']['status']} | "
                               f"preconditions blocked by: {v2c['planning_preconditions']['blocked_by'] or 'none'}"),
             ((200, 200, 200), f"agreement - semantic level: {agr['semantic_level']} | hand-over level: {agr['hand_over_level']}")]
    for cl in v2c["claims"]:
        lines.append(((160, 160, 160), f"claim '{cl['text']}': {cl['kind']} {cl['classes']} -> {cl['status']}"
                                       f"{' / ' + cl['relation'] if cl['relation'] else ''}{' (hedged by the VLM)' if cl['hedged'] else ''}"))
    lines.append(((120, 200, 120), "legend: orange = VLM proposal | green box = verified object, relation holds | red box = verified "
                                   "object, relation fails | blue = verified localization (V2) | VLM wording is never evidence"))
    panel = np.full((26 * len(lines) + 10, img.shape[1], 3), 24, np.uint8)
    for i, (c, t) in enumerate(lines):
        cv2.putText(panel, t.encode("ascii", "replace").decode()[:215], (8, 26 * (i + 1) - 6), FONT, 0.55, c, 1, cv2.LINE_AA)
    return np.vstack([img, panel])


# ------------------------------------------------------------------ stored stage
def stored_stage(cfg, v02, out):
    v1 = R.load_v1()
    v1cfg = R.v1_config()
    rows, reliable = [], True
    for cc in cfg["stored_cases"]:
        case = O.build_case(cc, HERE)
        v2 = V.verify_case(case, cfg, v02)
        v2c = v2["candidates"][0]
        v1_rows = []
        for run in case["_v1_runs"]:
            stored = O.v1_stored(run)
            ic = R.icfg_with_overrides(v1cfg, stored["overrides"])
            rp = R.replay(v1, R.inputs_from_run(run, case), ic, v02)
            rep = R.reproduces(stored, rp)
            reliable &= rep["checks_equal"] and rep["decision_equal"]
            ratio = next((c["detail"].get("area_ratio_to_truck") for c in stored["checks"] if c["id"] == "V3b"), None)
            v1_rows.append({"run_id": run["run_id"], "label": "V1" + (f" variant {','.join(stored['overrides'])}" if stored["overrides"] else " primary"),
                            "decision": stored["semantic_decision"]["status"], "integration_status": stored["integration_status"],
                            "checks": {c["id"]: c["status"] for c in stored["checks"]}, "failed": v1_failed_ids(stored),
                            "not_run": v1_not_run_ids(stored),
                            "reasons": stored["semantic_decision"].get("reasons"), "replay_reproduces_stored": rep,
                            "agreement": agreement(stored["semantic_decision"]["status"], v2c), "v3b_area_ratio_to_truck": ratio})
        img = cv2.imread(case["image_path"])
        png = f"{case['id']}_v1_vs_v2.png"
        cv2.imwrite(str(out / png), render_case(img, case, v2c, v1_rows, v1_rows[0]["agreement"]))
        rows.append({"case_id": case["id"], "instruction": case["instruction"], "image": case["image_path"],
                     "vlm_raw_response": case["vlm_raw_response_file"], "runs_used": case["runs"], "v1": v1_rows, "v2": v2,
                     "other_areas_proposed_by_vlm_not_verified": case["other_areas_proposed_by_vlm"], "overlay": png})
    return rows, reliable


# ------------------------------------------------------------------ synthetic stage
def synthetic_stage(cfg, v02, out):
    v1 = R.load_v1()
    v1cfg = R.v1_config()
    rows = []
    for sc in SC.scenarios():
        c, scfg = sc["case"], SC.scenario_cfg(cfg, sc)
        v2 = V.verify_case(c, scfg, v02)
        chosen = next(r for r in v2["candidates"] if r["candidate_id"] == c["chosen"])
        rp = R.replay(v1, SC.v1_inputs(c), v1cfg, v02)
        dec = "SEMANTIC_NOT_VERIFIED" if rp["decision"] == "WITHHELD" else rp["decision"]
        agr = agreement(rp["decision"], chosen)
        ok_v2 = chosen["final_disposition"] == sc["expect"]["v2"] and (chosen["hand_over_to_planner"] == sc["expect"]["hand_over"])
        ok_v1 = rp["decision"] == sc["expect"]["v1"]
        other_ok = all(next(r for r in v2["candidates"] if r["candidate_id"] == k)["final_disposition"] == v
                       for k, v in sc["expect"].get("v2_other", {}).items())
        rows.append({"scenario": sc["id"], "description": sc["description"], "instruction": c["instruction"],
                     "chosen_by_vlm": c["chosen"], "v1_decision": dec, "v1_failed": [x["id"] for x in rp["checks"] if x["status"] == "FAILED"],
                     "v1_not_run": [x["id"] for x in rp["checks"] if x["status"] == "NOT_RUN"],
                     "v1_reasons": rp["reasons"], "v2": {r["candidate_id"]: {"semantic_verdict": r["semantic_verdict"],
                     "final_disposition": r["final_disposition"], "hand_over": r["hand_over_to_planner"], "reasons": r["semantic_reasons"],
                     "blockers": r["all_blockers_after_semantics"], "purpose": r["purpose"]["purposes"]} for r in v2["candidates"]},
                     "selection": v2["selection"], "agreement": agr, "matches_expectation": {"v2": ok_v2, "v1": ok_v1, "other": other_ok},
                     "expect": sc["expect"]})
    return rows


# ------------------------------------------------------------------ markdown
def md_stored(rows, reliable):
    L = ["## 저장된 VLM 제안: V1 대 V2 (오프라인, 모델 호출 없음)", "",
         f"V1 재현(replay)이 저장된 V1 결과와 모든 검사에서 일치: **{reliable}** (False이면 이 비교는 신뢰할 수 없다)", "",
         "| 사례 | VLM 제안 | V1 결정 (실패한 검사 / 미실행) | V2 의미 판정 | V2 최종 | 의미 수준 일치 | 계획 전달 수준 일치 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["v2"]["candidates"][0]
        for v in r["v1"]:
            L.append(f"| {r['case_id']} | {c['vlm_function_claim']} | {v['label']}: {v['decision']} ({', '.join(v['failed']) or '-'} / {', '.join(v['not_run']) or '-'}) | "
                     f"{c['semantic_verdict']} | {c['final_disposition']} | {v['agreement']['semantic_level']} | {v['agreement']['hand_over_level']} |")
    for r in rows:
        c = r["v2"]["candidates"][0]
        L += ["", f"### {r['case_id']}", f"- 지시문의 목적: {c['purpose']['purposes']} → 필요한 능력 {c['purpose']['requires_any_of']}",
              f"- VLM 제안: {c['vlm_function_claim']} (VLM의 기능 설명은 근거로 쓰지 않음)"]
        for cl in c["claims"]:
            L.append(f"  - 근거 `{cl['text']}`: {cl['kind']} {cl['classes']} → **{cl['status']}**" + (f" / {cl['relation']}" if cl["relation"] else "")
                     + (" / VLM 스스로 추정 표현" if cl["hedged"] else ""))
        for ce in c["class_evidence"]:
            for i in ce["instances"]:
                L.append(f"    - {ce['class']} `{i['object_id']}`: 영역 안 비율 {i['fraction_of_object_inside_region']}, 간격 {i['gap_px']} px "
                         f"(인접 허용 {i['adjacency_distance_px']} px) → {i['relation']}")
        L += [f"- 의미 판정: **{c['semantic_verdict']}** ({c['semantic_verdict_ko']}): {'; '.join(c['semantic_reasons'])}",
              f"- 배치: {c['placement']['readiness']} — {'; '.join(c['placement']['reasons']) or '-'}",
              f"- 계획 인터페이스: {c['planner_support']['status']} (모델링되지 않은 물리 근거 {c['planner_support']['unmodeled_physical_claims']}, "
              f"관찰되지 않은 장애물 후보 클래스 {c['planner_support']['classes_cited_but_not_observed']})",
              f"- 계획 전제조건: {c['planning_preconditions']['blocked_by'] or '막힘 없음'}",
              f"- **최종: {c['final_disposition']}** ({c['final_disposition_ko']}), 계획기로 전달 {c['hand_over_to_planner']}"]
        for v in r["v1"]:
            L.append(f"- {v['label']} (`{v['run_id']}`): {v['decision']}; 실패/미실행 {v['failed']}; {v['reasons']}")
    return L


def md_synth(rows):
    L = ["## 합성 시나리오: V1 대 V2", "", "| 시나리오 | V1 | V2 (VLM이 고른 후보) | 의미 수준 | 계획 전달 수준 | 기대와 일치 (V2/V1) |", "|---|---|---|---|---|---|"]
    for r in rows:
        v2 = r["v2"][r["chosen_by_vlm"]]
        L.append(f"| {r['scenario']} | {r['v1_decision']} ({', '.join(r['v1_failed']) or '-'}) | {v2['final_disposition']} | "
                 f"{r['agreement']['semantic_level']} | {r['agreement']['hand_over_level']} | {r['matches_expectation']['v2']}/{r['matches_expectation']['v1']} |")
    L.append("")
    for r in rows:
        L.append(f"- **{r['scenario']}**: {r['description']}. V2 이유: " + "; ".join(r["v2"][r["chosen_by_vlm"]]["reasons"])
                 + (f" (후보별: { {k: v['final_disposition'] for k, v in r['v2'].items()} } — {r['selection']})" if len(r["v2"]) > 1 else ""))
    return L


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items() if k not in ("layers", "_v1_runs")}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["stored", "synthetic", "both"], default="both")
    args = ap.parse_args()
    cfg, v02 = load_cfg()
    out = HERE / cfg["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.stage}")
    out.mkdir(parents=True, exist_ok=False)
    res = {"experiment_id": cfg["experiment_id"], "created": datetime.now().astimezone().isoformat(timespec="seconds"),
           "note": "No ground truth exists for these proposals; 'false accept/reject' are candidates to be judged with the stated reason."}
    md = [f"# V1 대 V2 검증 비교 ({datetime.now():%Y-%m-%d %H:%M})", ""]
    if args.stage in ("stored", "both"):
        rows, reliable = stored_stage(cfg, v02, out)
        res["stored"], res["v1_replay_reproduces_all_stored_runs"] = rows, reliable
        md += md_stored(rows, reliable) + [""]
    if args.stage in ("synthetic", "both"):
        rows = synthetic_stage(cfg, v02, out)
        res["synthetic"] = rows
        md += md_synth(rows)
    (out / "comparison.json").write_text(json.dumps(jsonable(res), indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
