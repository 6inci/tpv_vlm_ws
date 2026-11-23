#!/usr/bin/env python3
"""Build the data file of the human-evaluation page from the STORED task_context_2d_navigation run (read only; no model, no planning).

  /home/binci/miniconda3/envs/stnav_gui/bin/python experiments/human_evaluation/build_data.py

Writes (only inside this folder): data.js (window.HE_DATA), human_evaluation_template.json, assets/<photo>.png (copies of the three photos).
"""
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RUN = EXP / "task_context_2d_navigation/results/20261119_174000"
REL_RUN = "../task_context_2d_navigation/results/20261119_174000"
PHOTO_LABEL = {"U25070552": "E1", "U26007129": "D1", "U24898553": "P4"}
ITEMS = [("source_accuracy", "출발 대상 정확성"), ("destination_suitability", "목적 대상의 작업 적합성"), ("relation_interpretation", "자연어 공간 관계 해석"),
         ("search_region", "주변 탐색 영역의 적절성"), ("workplace_rules", "5S 관련 운영 규칙 적용"), ("path_visual", "경로의 시각적 타당성")]
RULES = [("pedestrian_forbidden", "초록 바닥·발자국 보행자 통로: 로봇 본체 진입 금지"), ("drive_zone_only", "로봇 본체는 노란 외곽선 안 주행 구역에서만 이동"),
         ("workcell_forbidden", "흰 테두리 컨베이어·로봇 팔 작업셀: 본체 진입 금지"), ("storage_approach_only", "노란 테두리 적치·보관 구역: 본체 진입 없이 접근만"),
         ("truck_bay", "트럭과 흰 점선 구획 = 트럭 하역장"), ("cones_boundary", "안전 콘·바닥 경계선 = 작업공간 구분 표시"),
         ("gripper_adjacent", "본체는 주행 구역에 정차, 인접 구역 물체를 집거나 놓을 수 있음"), ("no_placement_on_walkway", "보행자 통로에는 물건을 놓지 않음")]


def rel(p):
    """absolute path inside the run -> path relative to index.html"""
    p = Path(p)
    try:
        return f"{REL_RUN}/{p.relative_to(RUN).as_posix()}"
    except ValueError:
        return None


def main():
    ii = json.loads((RUN / "image_instructions.json").read_text(encoding="utf-8"))
    cs = json.loads((RUN / "context_search_results.json").read_text(encoding="utf-8"))
    ts = json.loads((RUN / "target_selection.json").read_text(encoding="utf-8"))
    rt = json.loads((RUN / "routes.json").read_text(encoding="utf-8"))
    ev = {f"{r['task']}_{'R0' if r['condition'].startswith('C2') else 'R1'}": r for r in csv.DictReader(open(RUN / "evaluation.csv", encoding="utf-8"))}
    retention = json.loads((EXP / "candidate_retention_policy/results/20261119_073841_retention/result.json").read_text(encoding="utf-8"))
    cube = {p["photo_id"]: {f"cand_r{c['rank']}": c for c in p["candidates"]} for p in retention["photos"]}
    e1_dino = {c["id"]: c for c in json.loads((EXP / "task_conditioned_target_grounding/results/20261119_145217/all_candidates.json").read_text(encoding="utf-8"))["merged"]}
    (HERE / "assets").mkdir(exist_ok=True)
    photos = {}
    for t in ii:
        pid = t["image_id"]
        if pid in photos:
            continue
        dst = HERE / "assets" / f"{pid}.png"
        if not dst.exists():
            shutil.copy2(t["image_path"], dst)
        h, w = cv2.imread(str(dst)).shape[:2]
        photos[pid] = {"label": PHOTO_LABEL[pid], "file": f"assets/{pid}.png", "w": w, "h": h, "source_path": t["image_path"],
                       "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "alignment": t["calibration_compatibility"]}
    tasks = []
    for t in ii:
        tasks.append({"task_id": t["task_id"], "photo": t["image_id"], "instruction": t["instruction"], "type": t["type"],
                      "reference_note": ("사용자 확인 사실: 트럭 앞 콘 사이의 작은 회색 물체는 팔레트 모형이다. (모든 출하 명령의 유일한 목적지라는 뜻은 아님)"
                                         if t["image_id"] == "U25070552" else None)})
    conds = []
    for t in ii:
        for R in ("R0", "R1"):
            k = f"{t['task_id']}_{R}"
            it = (cs["interpretations"].get(k) or {}).get("parsed") or {}
            s, d, se = ts[k]["source"], ts[k]["destination"], cs["search"][k]

            def box_of(i):
                c = cube[t["image_id"]].get(i) or (e1_dino.get(i) if t["image_id"] == "U25070552" else None)
                return None if c is None else [round(v, 1) for v in c["bbox_xyxy"]]
            src_num = s.get("numbering") or {}
            dst_num = d.get("numbering") or {}
            conds.append({
                "cond_id": k, "task_id": t["task_id"], "R": R, "photo": t["image_id"],
                "condition_text": "R0: 자연어 + 이미지 (운영 규칙 미제공)" if R == "R0" else "R1: 자연어 + 이미지 + 작업장 운영 규칙(DRAFT) 제공",
                "interpretation": {"purpose": it.get("purpose"), "source_object": (it.get("source_object") or {}).get("name_en"),
                                   "source_reference": (it.get("source_reference") or {}).get("name_en"), "source_relation": it.get("source_relation"),
                                   "destination_meaning": it.get("destination_meaning"), "destination_reference": (it.get("destination_reference") or {}).get("name_en"),
                                   "destination_relation": it.get("destination_relation"),
                                   "zones_seen": [{"description": z.get("description"), "access": z.get("robot_body_access"), "evidence": z.get("visual_evidence")}
                                                  for z in it.get("zones_seen") or [] if isinstance(z, dict)],
                                   "raw_file": f"{REL_RUN}/qwen/interp_{k}.txt"},
                "references": {side: {"name": se[f"{side}_reference"]["name"], "box": se[f"{side}_reference"]["box"], "how": se[f"{side}_reference"]["how"],
                                      "relation": se[f"{side}_reference"]["relation"], "region_type": se[f"{side}_reference"]["region"]["type"],
                                      "region": se[f"{side}_reference"]["region"]["rect"]} for side in ("source", "destination")},
                "source": {"status": s.get("status"), "why": s.get("why"), "selected": [{"id": i, "box": box_of(i), "state": (cube[t["image_id"]].get(i) or {}).get("state")} for i in s.get("ids", [])],
                           "n_inside": len((se.get("source_candidates") or {}).get("inside", [])), "n_outside": len((se.get("source_candidates") or {}).get("outside", [])),
                           "ratings": [{"number": c.get("id"), "id": src_num.get(str(c.get("id"))), "role": c.get("role"), "appearance": c.get("appearance"), "reason": c.get("reason")}
                                       for c in (s.get("reply") or {}).get("candidates", [])],
                           "model_input": rel(s["image"]) if s.get("image") else None, "raw_file": f"{REL_RUN}/qwen/source_{k}.txt"},
                "destination": {"status": d.get("status"), "kind": d.get("kind"), "why": d.get("why"), "id": d.get("id"), "box": d.get("bbox_xyxy"),
                                "ratings": [{"number": c.get("id"), "id": dst_num.get(str(c.get("id"))), "role": c.get("placement_role"), "appearance": c.get("appearance"),
                                             "load": c.get("load_state"), "reason": c.get("reason")} for c in (d.get("reply") or {}).get("candidates", [])],
                                "best_number": (d.get("reply") or {}).get("best_destination_id"),
                                "accessibility": d.get("accessibility") or {}, "model_input": rel(d["image"]) if d.get("image") else None,
                                "raw_file": f"{REL_RUN}/qwen/destination_{k}.txt"},
                "baseline_C1": {"source": ts[k]["source_C1"], "destination": ts[k]["destination_C1"]},
                "routes": {"category_by_code": rt[k]["category"], "plans": [{"label": p["label"], "source_id": p["source_id"],
                                                                             "path_A_cm": (p["path_A"] or {}).get("length_cm"), "path_A_gap_cm": (p["path_A"] or {}).get("gap_cm"),
                                                                             "path_A_sweep_ok": (p["path_A"] or {}).get("sweep_ok"), "path_A_layers_clear": (p["path_A"] or {}).get("layers_clear"),
                                                                             "path_B_cm": (p["path_B"] or {}).get("length_cm"), "path_B_gap_cm": (p["path_B"] or {}).get("gap_cm"),
                                                                             "path_B_sweep_ok": (p["path_B"] or {}).get("whole_body_sweep_ok"), "path_B_layers_clear": (p["path_B"] or {}).get("layers_clear"),
                                                                             "path_B_why": p.get("path_B_why")} for p in rt[k]["plans"]]},
                "automatic_checks": {kk: ev[k][kk] for kk in ("collision_free", "forbidden_zone_clear", "category", "first_failure")},
                "figures": {"F1_context": f"{REL_RUN}/figures/F1_context_{k}.png", "F3_semantic_navigation": f"{REL_RUN}/figures/F3_semantic_navigation_{k}.png",
                            "F4_paths_2d": f"{REL_RUN}/figures/F4_paths_2d_{k}.png", "F5_overlay": f"{REL_RUN}/figures/F5_overlay_{k}.png"}})
    for c in conds:                                   # every figure must exist
        for f in list(c["figures"].values()) + [c["source"]["model_input"], c["destination"]["model_input"]]:
            assert f is None or (HERE / f).resolve().exists(), f
    data = {"meta": {"built": datetime.now().isoformat(timespec="seconds"), "ai_run": str(RUN), "note": "AI outputs of the stored run; nothing here is a human judgement",
                     "occupancy_figure": f"{REL_RUN}/figures/F2_occupancy_map.png", "ai_cube_labels": "AI_PRELIMINARY_REVIEW labels are deliberately NOT included"},
            "photos": photos, "tasks": tasks, "conditions": conds, "items": [{"key": k, "label": l} for k, l in ITEMS], "rules": [{"key": k, "label": l} for k, l in RULES],
            "ratings": ["SUCCESS", "FAILURE", "UNCERTAIN", "NOT_EVALUATED"]}
    (HERE / "data.js").write_text("// generated by build_data.py from the stored AI run (read only)\nwindow.HE_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    template = {"schema": "human_evaluation_v1", "status": "INCOMPLETE", "evaluator": None, "ai_run": str(RUN),
                "ground_truth": {t["task_id"]: {"saved_at": None, "source_boxes": [], "source_hold": False, "destination_boxes": [], "destination_hold": False,
                                                "source_relation": None, "destination_relation": None, "relation_note": "", "ambiguity": None, "rules": [], "rules_other": "",
                                                "certainty": {"source": None, "destination": None, "relation": None, "rules": None}, "comment": ""} for t in tasks},
                "evaluations": {c["cond_id"]: {"items": {k: {"rating": None, "comment": ""} for k, _ in ITEMS}, "overall_comment": "", "updated_at": None} for c in conds}}
    (HERE / "human_evaluation_template.json").write_text(json.dumps(template, indent=1, ensure_ascii=False), encoding="utf-8")
    print("tasks", len(tasks), "conditions", len(conds), "photos", list(photos))


if __name__ == "__main__":
    main()
