#!/usr/bin/env python3
"""두 버전의 A·B 결정을 같은 정답·같은 규칙으로 채점하고 판정 규칙을 적용 (config/holdout.yaml scoring, decision_rule). 모델 없음.

  $PY -B score.py --run results/run_<시각>                 독립 시험 채점 -> <run>/score.json, score.md
  $PY -B score.py --old DIR --new decisions.json --gt FILE  임의 출력 채점 (test_offline.py가 15장 결과 재현에 씀)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import hold as H

SC = H.CFG["scoring"]
GRADES = ["맞음", "맞는 보류", "틀리게 확정", "놓침(보류)"]


def normalize_gt(gt):
    """{stem: {'cube': [[u, v], ...], 'empty_pallet': [...], 'scenario': ...}}. 15장 gt_test.json 형식(점 하나 또는 null)도 받음"""
    out = {}
    for stem, e in gt["photos"].items():
        row = {"scenario": e.get("scenario")}
        for k in ("cube", "empty_pallet"):
            v = e.get(k)
            row[k] = [] if v is None else ([v] if v and not isinstance(v[0], list) else [list(q) for q in v])
        out[stem] = row
    return out


def centre(b):
    return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]


def grade(selected, box, pts):
    if not pts:
        return "틀리게 확정" if selected else "맞는 보류"
    if not selected:
        return "놓침(보류)"
    c = centre(box)
    return "맞음" if any(np.hypot(c[0] - p[0], c[1] - p[1]) <= SC["match_px"] for p in pts) else "틀리게 확정"


def old_decisions(old_dir):
    out = {}
    for f in sorted(Path(old_dir).glob("*/nl_to_path.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        p = j["perception"]
        out[Path(j["photo"]).stem] = {k: (p[k]["status"] == "SELECTED", p[k].get("box_px")) for k in ("A", "B")} | {
            "path_ok": j["result"]["status"] == "PATH_OK", "result": j["result"]}
    return out


def new_decisions(decisions_json):
    j = json.loads(Path(decisions_json).read_text(encoding="utf-8"))
    out = {}
    for r in j["photos"]:
        out[Path(r["photo"]).stem] = {k: (r[k]["status"] == "SELECTED", r[k]["box"]) for k in ("A", "B")} | {
            "path_ok": (r.get("path_result") or {}).get("status") == "PATH_OK", "result": r.get("path_result")}
    return out


def score(gt, systems):
    """systems = {'old': decisions, 'new': decisions} -> 사진별 등급과 묶음별 집계"""
    rows, missing = [], []
    for stem in sorted(gt):
        g = gt[stem]
        row = {"photo": stem, "scenario": g["scenario"], "gt": {k: g[k] for k in ("cube", "empty_pallet")}}
        for name, dec in systems.items():
            d = dec.get(stem)
            if d is None:
                missing.append(f"{name}: {stem} 결과 없음")
                continue
            row[name] = {"A": grade(*d["A"], g["cube"]), "B": grade(*d["B"], g["empty_pallet"]), "path_ok": d["path_ok"]}
        rows.append(row)
    if missing:
        raise SystemExit("채점할 결과가 빠졌습니다:\n  " + "\n  ".join(missing))

    def tally(sel):
        t = {}
        for name in systems:
            c = {k: {g: sum(r[name][k] == g for r in sel) for g in GRADES} for k in ("A", "B")}
            c["paths"] = sum(r[name]["path_ok"] for r in sel)
            c["wrong_commits"] = c["A"]["틀리게 확정"] + c["B"]["틀리게 확정"]
            c["correct"] = sum(c[k]["맞음"] + c[k]["맞는 보류"] for k in ("A", "B"))
            t[name] = c
        return {"n_photos": len(sel), **t}

    prim = set(H.CFG["scenarios"]["primary"])
    groups = {"all": rows, "primary": [r for r in rows if r["scenario"] in prim], "edge": [r for r in rows if r["scenario"] not in prim]}
    return rows, {k: tally(v) for k, v in groups.items()}


def decide(tot):
    """판정 규칙 (사전 등록): primary 묶음에서 새 버전 잘못 확정 <= 예전, 맞음 >= 예전 -> 새 버전을 본문으로"""
    p = tot[H.CFG["decision_rule"]["set"]]
    if p["n_photos"] == 0:
        return {"verdict": "판정 불가 (primary 사진 없음)"}
    o, n = p["old"], p["new"]
    ok = n["wrong_commits"] <= o["wrong_commits"] and n["correct"] >= o["correct"]
    return {"verdict": "새 버전을 본문 기준으로" if ok else "예전 실행을 본문 기준으로 유지",
            "rule": H.CFG["decision_rule"]["adopt_new_if"],
            "numbers": {"wrong_commits": {"old": o["wrong_commits"], "new": n["wrong_commits"]}, "correct": {"old": o["correct"], "new": n["correct"]}},
            "n_photos": p["n_photos"]}


def markdown(rows, tot, verdict):
    L = [f"# 독립 시험 채점 ({H.CFG['experiment_id']})", "", f"- 판정: **{verdict['verdict']}**"]
    if "numbers" in verdict:
        nb = verdict["numbers"]
        L += [f"  - primary {verdict['n_photos']}장: 잘못 확정 예전 {nb['wrong_commits']['old']} / 새 {nb['wrong_commits']['new']}, "
              f"맞음(맞는 보류 포함) 예전 {nb['correct']['old']} / 새 {nb['correct']['new']}",
              f"  - 규칙: {verdict['rule']}"]
    for gname, t in tot.items():
        if not t["n_photos"]:
            continue
        L += ["", f"## {gname} ({t['n_photos']}장)", "", "| | 예전 | 새 버전 |", "|---|---|---|"]
        for k in ("A", "B"):
            for g in GRADES:
                L.append(f"| {k} {g} | {t['old'][k][g]} | {t['new'][k][g]} |")
        L.append(f"| 경로 그림까지 | {t['old']['paths']} | {t['new']['paths']} |")
    L += ["", "## 사진별", "", "| 사진 | 시나리오 | A 예전 | A 새 | B 예전 | B 새 | 경로 예전 | 경로 새 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['photo']} | {r['scenario']} | {r['old']['A']} | {r['new']['A']} | {r['old']['B']} | {r['new']['B']} | "
                 f"{'O' if r['old']['path_ok'] else '-'} | {'O' if r['new']['path_ok'] else '-'} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--old")
    ap.add_argument("--new")
    ap.add_argument("--gt")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.run:
        run = Path(a.run).resolve()
        if not (run / "RUN_DONE").exists():
            raise SystemExit(f"끝나지 않은 실행입니다: {run}")
        man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if H.sha256_file(H.GT_FILE) != man["gt_sha256"]:
            raise SystemExit("실행 뒤 정답 파일이 바뀌었습니다. 채점하지 않습니다.")
        old_dir, new_file, gt_file, out = run / "old", run / "new" / "decisions.json", H.GT_FILE, run
    else:
        if not (a.old and a.new and a.gt and a.out):
            ap.error("--run 또는 --old --new --gt --out")
        old_dir, new_file, gt_file, out = Path(a.old), Path(a.new), Path(a.gt), Path(a.out)
    gt = normalize_gt(json.loads(Path(gt_file).read_text(encoding="utf-8")))
    rows, tot = score(gt, {"old": old_decisions(old_dir), "new": new_decisions(new_file)})
    verdict = decide(tot)
    out.mkdir(parents=True, exist_ok=True)
    (out / "score.json").write_text(json.dumps({"verdict": verdict, "totals": tot, "photos": rows, "gt_file": str(gt_file),
                                                "match_px": SC["match_px"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    md = markdown(rows, tot, verdict)
    (out / "score.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    sys.exit(main())
