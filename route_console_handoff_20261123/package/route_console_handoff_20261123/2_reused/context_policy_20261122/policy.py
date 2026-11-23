#!/usr/bin/env python3
"""맥락 정책으로 A(큐브)·B(빈 팔레트) 고르기 (모델 호출 없음, extract.py가 저장한 자료만 씀). 설정: config/policy.yaml의 policy.

후보마다 차례로 검사하고, 떨어진 이유를 모두 기록한다 (1위 후보가 무엇이었는지 항상 남김).
  A (옮길 큐브):  크기 10-90 px -> 작업 공간(지도: 노란 선 밖 15 cm 안, 5S 금지 구역 밖) -> 큐브 예시 >= 0.88 -> 받침(큐브가 올라간 팔레트) >= 기준
                 -> 남은 후보끼리 1·2위 차이 >= 0.02
  B (빈 팔레트):  크기 -> 작업 공간 -> 점유 아님(큐브 같은 박스 중심이 안에 없음) -> 빈 팔레트 예시 >= 기준 -> 남은 후보끼리 차이 >= 0.02
  관계(기록): 후보마다 가장 가까운 5S 구역과 거리(cm). "X 옆" = 구역 X까지 near_max_cm 이내 (EXP120 J-geo 방식, Python 계산)
기준값(받침, 빈 팔레트)은 개발 8장에서만 정한다 (--calibrate). 시험 15장은 그 값을 그대로 쓴다.
  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  $PY -B policy.py --calibrate                 # 개발 8장 -> data/dev/calibration.json
  $PY -B policy.py --set dev|test [--paths]   # 결정, 채점, 그림 -> results/<set>_<시각>/  (--paths: 시험 사진에서 경로 그림까지)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as CM  # noqa: E402

sys.path.insert(0, "/home/binci/vlm_ws/hybrid_exemplar_20261121")
import hybrid_pipeline as HP  # noqa: E402  (읽기만: exemplar_scores, decide, bank_for, centre_in)

PC = CM.C["policy"]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONTB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def side(b):
    return max(b[2] - b[0], b[3] - b[1])


def centre(b):
    return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]


def load_set(name):
    d = HERE / "data" / name
    meta = json.loads((d / "features.json").read_text(encoding="utf-8"))
    bank = json.loads((d / "bank.json").read_text(encoding="utf-8"))
    be, ce, se = (np.load(d / f) for f in ("bank_emb.npz", "cand_emb.npz", "support_emb.npz"))
    for e in bank:
        e["embs"] = be[f"{e['photo_id']}|{e['key']}"]
    return meta, bank, {k: ce[k] for k in ce.files}, {k: se[k] for k in se.files}


def gt_for(name):
    """{photo_id: {'cube': [u, v] | None, 'empty_pallet': [u, v] | None}}"""
    if name == "dev":
        out = {}
        for r in HP.RT.load_reference():
            cube = next((centre(o["box"]) for o in r["objects"] if o["kind"] == "cube"), None)
            pads = [centre(o["box"]) for o in r["objects"] if o["kind"] == "empty_pad"]
            out[r["photo_id"]] = {"cube": cube, "empty_pallet": pads[0] if pads else None, "empty_pallet_all": pads}
        return out
    g = json.loads((HERE / "data" / "gt_test.json").read_text(encoding="utf-8"))["photos"]
    return {k: {**v, "empty_pallet_all": [v["empty_pallet"]] if v["empty_pallet"] else []} for k, v in g.items()}


def score_photo(ph, bank, ce, se, refs):
    """후보마다 예시 점수 (cube, empty_pallet, loaded_pallet)와 받침 점수"""
    pid = ph["photo_id"]
    used, dropped = HP.bank_for(bank, refs.get(pid, []), pid, HP.CFG["exemplar"]["same_place_iou"])
    cands = [x for x in ph["candidates"] if x["gated"]]
    emb = {x["id"]: ce[f"{pid}|{x['id']}"] for x in cands}
    sup = {x["id"]: se[f"{pid}|{x['id']}"] for x in cands}
    cls = sorted({e["cls"] for e in bank})
    s = {c: HP.exemplar_scores(emb, [e["embs"] for e in used if e["cls"] == c]) for c in cls}
    s["support"] = HP.exemplar_scores(sup, [e["embs"] for e in used if e["cls"] == "loaded_pallet"])
    return cands, s, {"bank_used": {c: sum(e["cls"] == c for e in used) for c in cls}, "bank_dropped_same_place": len(dropped)}


def evaluate_candidates(cam, cands, s, thr):
    """후보마다 단계별 통과 여부와 지도 좌표"""
    rows = {}
    cube_like = [x for x in cands if s["cube"][x["id"]] >= PC["cube_min_score"]]
    for x in cands:
        i, b = x["id"], x["bbox_xyxy"]
        r = {"id": i, "bbox_xyxy": b, "side_px": round(side(b), 1), "conditions": x.get("conditions"),
             "score": {k: round(v[i], 4) for k, v in s.items()}}
        for cls in ("cube", "empty_pallet"):
            xy = cam.to_map([centre(b)], PC["z_top_m"][cls])[0]
            w = CM.workspace(xy, PC["band_max_m"], PC["excluded_zones"])
            r[f"{cls}_map_xy_m"] = [round(float(v), 4) for v in xy]
            r[f"{cls}_workspace"] = w
        r["zone"] = CM.zone_address(r["cube_map_xy_m"], PC["named_zones"])
        # A
        a = [("작업 공간", r["cube_workspace"]["ok"], r["cube_workspace"]["why"]),
             ("큐브 예시", r["score"]["cube"] >= PC["cube_min_score"], f"{r['score']['cube']:.3f} (기준 {PC['cube_min_score']})"),
             ("받침", thr["support"] is not None and r["score"]["support"] >= thr["support"], f"{r['score']['support']:.3f} (기준 {thr['support']})")]
        # B
        occ = [y["id"] for y in cube_like if y["id"] != i and HP.centre_in(y["bbox_xyxy"], b)]
        bb = [("작업 공간", r["empty_pallet_workspace"]["ok"], r["empty_pallet_workspace"]["why"]),
              ("점유 아님", not occ, "큐브 같은 박스 " + (", ".join(occ) if occ else "없음")),
              ("빈 팔레트 예시", thr["pallet"] is not None and r["score"]["empty_pallet"] >= thr["pallet"], f"{r['score']['empty_pallet']:.3f} (기준 {thr['pallet']})")]
        for key, steps in (("A", a), ("B", bb)):
            fail = next((n for n, ok, _ in steps if not ok), None)
            r[f"{key}_checks"] = [{"check": n, "ok": bool(ok), "detail": d} for n, ok, d in steps]
            r[f"{key}_pass"] = fail is None
            r[f"{key}_failed_at"] = fail
        rows[i] = r
    return rows


def decide(rows, key, cls):
    surv = [{"id": i, "bbox_xyxy": r["bbox_xyxy"]} for i, r in rows.items() if r[f"{key}_pass"]]
    sc = {i: r["score"][cls] for i, r in rows.items()}
    d = HP.decide(surv, sc, None, PC["min_margin"], PC["nms_iou"], PC["distinct_iou"])
    top_all = sorted(rows.values(), key=lambda r: -r["score"][cls])[:5]     # 정책 전 순위 (무엇이 1위였는지 남김)
    return {"status": d["status"], "why": d["why"], "id": d["top"]["id"] if d["status"] == "SELECTED" else None,
            "box": d["top"]["bbox_xyxy"] if d["status"] == "SELECTED" else None, "s1": d["s1"], "s2": d["s2"],
            "n_survivors": len(surv), "top_before_policy": [{"id": r["id"], "score": r["score"][cls], "failed_at": r[f"{key}_failed_at"]} for r in top_all],
            "survivors_ranked": [{"id": x["id"], "score": round(sc[x["id"]], 4)} for x in d["ranked"]]}


def grade(dec, gt_pt, gt_all):
    """맞음 / 틀리게 확정 / 맞는 보류 / 놓침"""
    if dec["status"] == "SELECTED":
        c = centre(dec["box"])
        ok = any(np.hypot(c[0] - g[0], c[1] - g[1]) <= PC["match_px"] for g in gt_all)
        return "맞음" if ok else "틀리게 확정"
    return "맞는 보류" if gt_pt is None else "놓침(보류)"


def run_set(name, thr):
    meta, bank, ce, se = load_set(name)
    cam, gt = CM.Cam(name), gt_for(name)
    refs = {r["photo_id"]: r["objects"] for r in HP.RT.load_reference()}
    out = []
    for ph in meta["photos"]:
        cands, s, binfo = score_photo(ph, bank, ce, se, refs)
        rows = evaluate_candidates(cam, cands, s, thr)
        g = gt.get(ph["photo_id"]) or gt.get(Path(ph["path"]).stem) or {}
        A, B = decide(rows, "A", "cube"), decide(rows, "B", "empty_pallet")
        A["grade"], B["grade"] = grade(A, g.get("cube"), [g["cube"]] if g.get("cube") else []), grade(B, g.get("empty_pallet"), g.get("empty_pallet_all") or [])
        # 정답 물체가 후보에 있었는가, 있었다면 어디서 떨어졌는가
        for dec, key, cls in ((A, "A", "cube"), (B, "B", "empty_pallet")):
            gp = g.get(cls)
            hit = None if gp is None else min(rows.values(), key=lambda r: np.hypot(centre(r["bbox_xyxy"])[0] - gp[0], centre(r["bbox_xyxy"])[1] - gp[1]), default=None)
            near = hit is not None and np.hypot(centre(hit["bbox_xyxy"])[0] - gp[0], centre(hit["bbox_xyxy"])[1] - gp[1]) <= PC["match_px"]
            dec["truth_candidate"] = None if gp is None else ({"id": hit["id"], "score": hit["score"][cls], "failed_at": hit[f"{key}_failed_at"],
                                                               "checks": hit[f"{key}_checks"]} if near else "후보 박스 없음")
        out.append({"photo_id": ph["photo_id"], "photo": ph["path"], "bank": binfo, "n_candidates": len(cands), "A": A, "B": B, "rows": rows})
    return meta, out


def calibrate():
    """개발 8장: 진짜 큐브의 받침 점수, 진짜 빈 팔레트의 빈 팔레트 점수 -> 최소값 - margin"""
    meta, bank, ce, se = load_set("dev")
    cam, gt = CM.Cam("dev"), gt_for("dev")
    refs = {r["photo_id"]: r["objects"] for r in HP.RT.load_reference()}
    pos = {"support": [], "pallet": []}
    neg = {"support": [], "pallet": []}
    recall = {"cube": 0, "empty_pallet": 0, "n_cube": 0, "n_empty_pallet": 0}
    for ph in meta["photos"]:
        cands, s, _ = score_photo(ph, bank, ce, se, refs)
        rows = evaluate_candidates(cam, cands, s, {"support": None, "pallet": None})
        g = gt[ph["photo_id"]]
        for cls, key, sk in (("cube", "support", "support"), ("empty_pallet", "pallet", "empty_pallet")):
            gts = [g["cube"]] if cls == "cube" else g["empty_pallet_all"]
            recall[f"n_{cls}"] += len(gts)
            for gp in gts:
                m = [r for r in rows.values() if np.hypot(centre(r["bbox_xyxy"])[0] - gp[0], centre(r["bbox_xyxy"])[1] - gp[1]) <= PC["match_px"]]
                if m:
                    recall[cls] += 1
                    pos[key].append(max(r["score"][sk] for r in m))
            for r in rows.values():
                is_t = any(np.hypot(centre(r["bbox_xyxy"])[0] - gp[0], centre(r["bbox_xyxy"])[1] - gp[1]) <= PC["match_px"] for gp in gts)
                pre_ok = r[f"{cls}_workspace"]["ok"] and (r["score"]["cube"] >= PC["cube_min_score"] if cls == "cube" else True)
                if not is_t and pre_ok:
                    neg[key].append(r["score"][sk])
    thr = {k: round(min(v) - PC["calibrate_margin"], 4) if v else None for k, v in pos.items()}
    rep = {"created": datetime.now().isoformat(timespec="seconds"), "rule": {"support": PC["support_rule"], "pallet": PC["pallet_rule"]},
           "thresholds": thr, "positives": {k: sorted(round(x, 4) for x in v) for k, v in pos.items()},
           "negatives_after_earlier_checks": {k: {"n": len(v), "max": round(max(v), 4) if v else None, "n_at_or_above_threshold": sum(x >= thr[k] for x in v) if thr[k] else None}
                                              for k, v in neg.items()},
           "candidate_recall_dev": recall}
    (HERE / "data" / "dev" / "calibration.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


# ================================================================================ 그림 (후보 순위와 떨어진 이유)
def draw_photo(cam, rec, out):
    im = Image.open(rec["photo"]).convert("RGB")
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    f, fb = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONTB, 22)
    for z in CM.zones()["zones"]:
        px = cam.to_px(np.array(z["polygon_map_m"]))
        excl = z["id"] in PC["excluded_zones"]
        d.polygon([tuple(p) for p in px], outline=(255, 80, 80, 200) if excl else (255, 220, 0, 200), width=2)
        c = px.mean(0)
        d.text((c[0] - 40, c[1] - 10), f"{z['id']}{' 금지' if excl else ''}", font=f, fill=(255, 255, 255, 230), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    col = {"A": (235, 40, 40, 255), "B": (40, 120, 255, 255)}
    lines = []
    for key, cls, nm in (("A", "cube", "A 큐브"), ("B", "empty_pallet", "B 빈 팔레트")):
        dec = rec[key]
        for k, t in enumerate(dec["top_before_policy"][:3]):        # 정책 전 상위 3개: 통과 / 떨어진 단계
            r = rec["rows"][t["id"]]
            x0, y0, x1, y1 = r["bbox_xyxy"]
            chosen = dec["id"] == t["id"]
            d.rectangle([x0, y0, x1, y1], outline=col[key] if chosen else (200, 200, 200, 255), width=4 if chosen else 2)
            tag = f"{key}{k + 1} {t['score']:.3f} " + ("확정" if chosen else ("통과" if t["failed_at"] is None else f"×{t['failed_at']}"))
            d.text((x0, y1 + 2 + (0 if key == "A" else 20)), tag, font=f, fill=col[key] if chosen else (255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        tc = dec.get("truth_candidate")
        lines.append(f"{nm}: {dec['status']} ({dec['grade']}) - {dec['why']}" + (f" | 정답 물체: {tc if isinstance(tc, str) else ('통과' if tc['failed_at'] is None else '×' + tc['failed_at'])}" if tc else ""))
        lines.append("   정책 전 상위: " + ", ".join(f"{t['id']} {t['score']:.3f}" + ("" if t["failed_at"] is None else f"(×{t['failed_at']})") for t in dec["top_before_policy"][:5]))
    h = 14 + 28 * len(lines)
    canvas = Image.new("RGB", (im.size[0], im.size[1] + h), (20, 20, 20))
    canvas.paste(Image.alpha_composite(im.convert("RGBA"), lay).convert("RGB"), (0, 0))
    dt = ImageDraw.Draw(canvas)
    for i, t in enumerate(lines):
        dt.text((12, im.size[1] + 6 + 28 * i), t, font=f, fill=(255, 255, 255))
    canvas.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--set", choices=["dev", "test"])
    ap.add_argument("--paths", action="store_true", help="시험 사진: A·B가 모두 확정이면 nl_to_path로 경로 그림까지")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
        return
    cal = json.loads((HERE / "data" / "dev" / "calibration.json").read_text(encoding="utf-8"))
    thr = cal["thresholds"]
    meta, recs = run_set(a.set, thr)
    od = HERE / "results" / f"{a.set}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    od.mkdir(parents=True)
    (od / "config_snapshot.yaml").write_text((HERE / "config" / "policy.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    cam = CM.Cam(a.set)
    summ = {"A": {}, "B": {}}
    for r in recs:
        draw_photo(cam, r, od / f"{Path(r['photo']).stem}_policy.jpg")
        for k in ("A", "B"):
            summ[k][r[k]["grade"]] = summ[k].get(r[k]["grade"], 0) + 1
        print(f"{Path(r['photo']).stem}: A {r['A']['status']}({r['A']['grade']}) B {r['B']['status']}({r['B']['grade']}) | "
              f"A 정책 전 1위 {r['A']['top_before_policy'][0]['id']} ✗{r['A']['top_before_policy'][0]['failed_at']} | "
              f"B 정책 전 1위 {r['B']['top_before_policy'][0]['id']} ✗{r['B']['top_before_policy'][0]['failed_at']}", flush=True)
    if a.paths and a.set == "test":
        sys.path.insert(0, "/home/binci/vlm_ws/nl_to_route_20261122")
        import nl_to_path as N                               # 읽기만 (좌표·정차·경로·그림 함수 재사용)
        geo, planner = N.Geo(), N.Planner()
        ins = N.CFG["default_instruction"]
        for r in recs:
            if r["A"]["status"] == "SELECTED" and r["B"]["status"] == "SELECTED":
                pr = N.run_one(Path(r["photo"]), ins, None, geo, planner, od / "paths", ab_px=[r["A"]["box"], r["B"]["box"]])
                pj = Path(pr["files"]["record"])
                j = json.loads(pj.read_text(encoding="utf-8"))
                j["perception"]["note"] = "A·B = context_policy_20261122 policy.py 결정 (모델 출력은 extract.py, 정책은 Python)"
                j["policy_decision"] = {k: {kk: r[k][kk] for kk in ("status", "why", "id", "box", "s1", "s2", "grade")} for k in ("A", "B")}
                pj.write_text(json.dumps(j, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
                r["path_result"] = pr["result"]
    (od / "decisions.json").write_text(json.dumps({"set": a.set, "thresholds": thr, "summary": summ, "photos": recs}, indent=1, ensure_ascii=False, default=str),
                                       encoding="utf-8")
    print("요약:", json.dumps(summ, ensure_ascii=False), "->", od)
    if a.paths:
        print("경로 그림까지:", sum((r.get("path_result") or {}).get("status") == "PATH_OK" for r in recs), "/", len(recs))


if __name__ == "__main__":
    main()
