#!/usr/bin/env python3
"""Back end of the instruction target viewer: photo + natural-language instruction -> source A, destination B, reasons and a stage trace.

Every instruction is inferred anew (Qwen3-VL interpretation and role calls). Only the per-photo detections, the reference objects and the
E1 geometry are reused within a session.

Reused unchanged (imported, nothing copied or modified):
  task_context_2d_navigation  prompts, rules R1, relation rules, ref_class, search_region, in_region, decide_source, decide_destination
  placement_target_grounding  qwen_call, dino_call, detections, parse_reply, qwen_box_to_image, norm_phrase
  contextual_target_selection snap_anchor, zone_members, view_rect, zoom_scale, numbered, SRC_SCHEMA, DEST_SCHEMA
  task_route_integration      build_context (E1 floor transform, transferred DRAFT map, stored E1 objects and robot start)
  semantic_navigation_integration leg_to_region, leg_record (approach reachability = the task_context accessibility policy)
  v0.2 src                     model worker, verify_box, make_roi, check_alignment
Not done here: new maps, new planners, robot commands.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP / "task_context_2d_navigation"))
import run_task_context as TC  # noqa: E402
from semantic_task_nav.config import code_fingerprint, load_config, sha256_file  # noqa: E402
from semantic_task_nav.geometry.coordinate_frames import box_area, iou  # noqa: E402
from semantic_task_nav.navigation.map_builder import check_alignment  # noqa: E402
from semantic_task_nav.perception.grounding import verify_box  # noqa: E402
from semantic_task_nav.perception.model_client import ModelWorkerClient, ModelWorkerError  # noqa: E402
from semantic_task_nav.perception.target_inspection import make_roi  # noqa: E402

P, S, PT, FN, SN, RI = TC.P, TC.S, TC.PT, TC.FN, TC.SN, TC.RI
PROBLEM = ("FAILED", "UNCERTAIN")
FRAMES = {"bbox_xyxy": "pixels of the photo as stored (undistorted image, 1920x1080)",
          "floor_cm": "floor cm of the PROVISIONAL_UNVALIDATED E1 mat-seam transform (valid only for photos aligned with E1); box corners projected on the floor plane",
          "qwen_bbox_2d": "Qwen 0-1000 coordinates of the image Qwen saw (converted to photo px before use)"}


# ================================================================================ pure functions (tested offline)
def photo_id(path):
    """undistorted_25070552942982.png -> U25070552 (naming of every earlier experiment); other file names -> the stem."""
    stem = Path(path).stem
    d = stem.split("_")[-1]
    return f"U{d[:8]}" if stem.startswith("undistorted_") and d.isdigit() else stem


def chunks(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]


def merge_replies(replies, sizes, best_key):
    """Per-image Qwen replies (local numbers 1..n_k) -> one reply with global numbers. One image: the reply itself.
    Any invalid reply -> None (the decision then says 'reply invalid', as in task_context). Several images: the per-image best is kept
    only if exactly one image named one."""
    if not replies or any(r is None for r in replies):
        return None
    if len(replies) == 1:
        return replies[0]
    out, bests, off = [], [], 0
    for r, n in zip(replies, sizes):
        out += [{**c, "id": S._num(c.get("id"), n) + off} for c in r["candidates"] if S._num(c.get("id"), n)]
        b = S._num(r.get(best_key), n)
        if b:
            bests.append(b + off)
        off += n
    return {"candidates": out, best_key: bests[0] if len(bests) == 1 else None, "per_image_best": bests,
            "reason": " | ".join(str(r.get("reason")) for r in replies)}


def _names_area(text, words):
    t = str(text).lower().replace("-", " ").split()
    return any(w in t for w in words)


def goal_kind(cand, appearance, img_wh, gcfg):
    """OBJECT or GOAL_REGION (display label; deterministic rule of config.yaml, not validated)."""
    words, ph = gcfg["region_words"], cand.get("phrases") or []
    frac = box_area(cand["bbox_xyxy"]) / float(img_wh[0] * img_wh[1])
    if ph and all(_names_area(p, words) for p in ph):
        return "GOAL_REGION", f"every detector phrase names an area {ph}"
    if appearance and _names_area(appearance[0], words):
        return "GOAL_REGION", f"Qwen appearance names an area ('{appearance[0]}')"
    if frac >= gcfg["min_image_frac"]:
        return "GOAL_REGION", f"box covers {frac:.1%} of the image (>= {gcfg['min_image_frac']:.0%})"
    return "OBJECT", f"detector phrases {ph[:4]}; box {frac:.2%} of the image"


def placement_status(kind, rating):
    ok = kind == "OBJECT" and (rating or {}).get("placement_role") == "suitable" and (rating or {}).get("load_state") == "empty"
    return "VLM_SUITABLE_EMPTY" if ok else "PLACEMENT_UNVERIFIED"


def tied_destinations(rated, acc):
    """Candidates behind an AMBIGUOUS destination decision (same tiers as task_context decide_destination)."""
    t1 = [k for k, c in rated.items() if c["placement_role"] == "suitable" and (acc.get(k) or {}).get("accessible") is not False]
    t2 = [k for k, c in rated.items() if c["placement_role"] == "possible" and (acc.get(k) or {}).get("accessible") is not False]
    return t1 if len(t1) > 1 else t2 if not t1 and len(t2) > 1 else []


def first_failure(stages):
    return next((s["name"] for s in stages if s["status"] in PROBLEM), None)


def stage(name, status, summary, **detail):
    return {"name": name, "status": status, "summary": summary, **detail}


# ================================================================================ session
class Viewer:
    """One program session: geometry once, per-photo detections once, every instruction inferred anew. Results in one new session folder."""

    def __init__(self, session_dir=None, progress=print):
        self.vc = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
        self.tc_path = (HERE / self.vc["reuse"]["task_context_config"]).resolve()
        self.tc = yaml.safe_load(self.tc_path.read_text(encoding="utf-8"))
        self.rcfg = yaml.safe_load((TC.HERE / self.tc["reuse"]["route_config"]).read_text(encoding="utf-8"))
        self.fpcfg = yaml.safe_load((TC.HERE / self.tc["reuse"]["first_path_config"]).read_text(encoding="utf-8"))
        self.gcfg, self.v02 = PT.load_cfg(), load_config()
        self.mcfg, self.tcfg, self.rel = self.v02["models"], self.v02["models"]["target_inspection"], self.tc["relations"]
        self.progress = progress
        self.session = Path(session_dir) if session_dir else HERE / self.vc["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_session")
        self.session.mkdir(parents=True, exist_ok=False)
        (self.session / "runs").mkdir()
        (self.session / "photos").mkdir()
        self.client, self.photos, self.runs, self.geo, self._acc, self.n = None, {}, [], None, {}, 0
        self.v02_sha = code_fingerprint()["sha256"]
        self._write_session()

    # ---------------------------------------------------------------- shared resources
    def list_photos(self):
        return [{"photo_id": photo_id(p), "file": p.name, "path": str(p)} for p in sorted(Path(self.vc["photo_dir"]).glob("*.png"))]

    def geometry(self):
        """E1 floor transform + transferred DRAFT map + footprint maps (Python only, built once)."""
        if self.geo is None:
            self.progress("바닥 변환·DRAFT 지도 준비 (Python, 모델 없음)")
            e1 = cv2.imread(self.rcfg["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            ctx = RI.build_context(self.rcfg, self.gcfg, self.v02, e1)
            if "fr" not in ctx:
                self.geo = {"ok": False, "ctx": ctx, "why": "E1 floor transform not accepted"}
                return self.geo
            fr, (L, Wd) = ctx["fr"], map(float, self.tc["route"]["robot_cm"])
            m = ctx["ft"]["footprint_margin_cm"]
            navs = {h: FN.RectNavMap(ctx["spec_bev"], fr.bev_size, ctx["pcfg"], FN.footprint_dims_cm(L, Wd, m, h), fr.s) for h in (0, 90)}
            self.geo = {"ok": True, "ctx": ctx, "navs": navs, "zr": SN.CM.Raster(fr.bev_size, fr.s, fr.pad),
                        "drivable": [z["polygon"] for z in ctx["spec_cm"]["traversable"]], "L": L, "W": Wd}
        return self.geo

    def model(self):
        if self.client is None or not self.client.alive:
            self.progress("모델 작업 프로세스 시작 (qwen_vl 환경, 기존 v0.2 worker)")
            self.client = ModelWorkerClient(self.mcfg, self.session / "worker_logs")
            self.client.start()
        return self.client

    def stop_models(self):
        """Abort a running model call (GUI 'stop'); the next call restarts the worker."""
        if self.client is not None:
            self.client.kill()

    def close(self):
        if self.client is not None:
            self.client.close()
        self._write_session()

    # ---------------------------------------------------------------- per photo (detections reused for every instruction on it)
    def photo(self, path):
        pid = photo_id(path)
        if pid in self.photos:
            return self.photos[pid]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        H, W = img.shape[:2]
        ph = {"photo_id": pid, "path": str(path), "sha256": sha256_file(path), "size": [W, H], "img": img, "calls": [],
              "alignment": check_alignment({"reference_image": self.rcfg["image"], "alignment_check": {"max_shift_px": 3.0, "min_response": 0.3}}, img, lambda x: x)}
        geo = self.geometry()
        vocab = self.tc["reuse"]["vocabulary"]
        if ph["sha256"] == self.rcfg["image_sha256"]:
            st, ctx = geo["ctx"]["st"], geo["ctx"]
            box = {o["object_id"]: o["bbox_xyxy"] for o in st["objects"]}
            ph["refs"] = {"conveyor": {"box": box["conveyor_1"], "how": "stored E1 (DINO + Qwen yes)"}, "truck": {"box": box["truck_1"], "how": "stored E1 (DINO + Qwen yes)"},
                          "robot": {"box": ctx["robot"]["bbox_xyxy"], "start_img": ctx["starts"][0], "how": "stored E1 (first_vlm_navigation_path)"}}
            stored = json.loads((TC.HERE / self.tc["reuse"]["e1_candidates"]).read_text(encoding="utf-8"))["merged"]
            ph["dino"] = [{"id": c["id"], "bbox_xyxy": c["bbox_xyxy"], "score": c["score"], "phrases": c["phrases"], "conditions": c["conditions"],
                           "source": "stored E1 merged DINO candidate"} for c in stored]
            ph["phrases_done"] = sorted({p for c in stored for p in c["phrases"]} | set(vocab))
            ph["detection_how"] = f"stored E1 references and {len(stored)} merged DINO candidates ({self.tc['reuse']['e1_candidates']}); no new E1 DINO for these phrases"
        else:
            self._fresh_photo(ph, vocab)
        rr = json.loads(((TC.HERE / self.tc["reuse"]["retention_run"]).resolve() / "result.json").read_text(encoding="utf-8"))
        cube = next((p["candidates"] for p in rr["photos"] if p["photo_id"] == pid), None)
        ph["cube"] = None if cube is None or ph["sha256"] != self._session_sha(pid) else \
            [{"id": f"cand_r{c['rank']}", "bbox_xyxy": c["bbox_xyxy"], "state": c["state"], "source_types": c["source_types"], "score": c["score"]} for c in cube]
        ph["cube_how"] = "stored red-cube candidates (candidate_retention_policy, HSV + DINO)" if ph["cube"] is not None else "no stored red-cube candidates for this photo: DINO candidates are used"
        self.photos[pid] = ph
        self._save_photo(ph)
        return ph

    def _session_sha(self, pid):
        funnel = yaml.safe_load((EXP / "cube_candidate_funnel/config.yaml").read_text(encoding="utf-8"))
        return next((i["sha256"] for i in funnel["images"] if i["id"] == pid), None)

    def _fresh_photo(self, ph, vocab):
        """Same procedure as task_context for a non-E1 photo: DINO references + full image, Qwen crop check of references and robot, ROI DINO."""
        c, pid, (W, H) = self.model(), ph["photo_id"], ph["size"]
        det, gd = self.tc["detection"], self.mcfg["grounding_dino"]["class_prompts"]
        inp = self.session / "photos" / f"{pid}_inputs"
        inp.mkdir(exist_ok=True)
        self.progress(f"{pid}: 기준 시설·로봇·물체 탐지 (Grounding DINO, 사진당 한 번)")
        raw_refs = P.dino_call(c, ph["path"], gd["conveyor"] + gd["truck"] + gd["mobile_robot"], det["thresholds"], f"dino_refs_{pid}", ph["calls"])
        raw_full = P.dino_call(c, ph["path"], vocab, det["thresholds"], f"dino_full_{pid}", ph["calls"])
        dets = P.detections(raw_refs, "refs", [0, 0, W, H], 1.0, det["max_area_frac"])
        ph["refs"], win = {}, self.fpcfg["robot"]["search_window_xyxy"]
        self.progress(f"{pid}: 기준 시설·로봇 확인 (Qwen3-VL 크롭 확인)")
        for cls, key, extra in (("conveyor", "conveyor", None), ("truck", "truck", None), ("robot", "mobile_robot", win)):
            prompts = [p.lower() for p in gd[key]]
            cs = [d for d in dets if d["status"] == "KEPT" and d["prompt"] in prompts and (extra is None or P.centre_in(d["bbox_xyxy"], extra))]
            recs, acc = [], None
            for k, d in enumerate(cs[:det["reference_verify_max"]]):
                v = verify_box(c, self.mcfg, ph["img"], d["bbox_xyxy"], self.mcfg["verification"]["descriptions"][key], inp / f"verify_{cls}_{k}.png")
                ph["calls"].append({"stage": f"verify_{pid}_{cls}_{k}", "model": v["call"]["model"], "raw_response": v["call"]["raw_response"],
                                    "latency_s": v["call"]["latency_s"], "prompt_sha256": v["call"]["prompt_sha256"], "image_path": v["crop"]["path"]})
                recs.append({"bbox_xyxy": d["bbox_xyxy"], "score": d["score"], "status": v["status"]})
                if v["status"] == "yes":
                    acc = d
                    break
            ph["refs"][cls] = {"box": acc["bbox_xyxy"] if acc else None, "how": "fresh DINO + Qwen crop yes" if acc else "NOT VERIFIED", "checked": recs}
            if cls == "robot" and acc:
                b = acc["bbox_xyxy"]
                ph["refs"]["robot"]["start_img"] = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]   # task_context rule for non-E1 photos (box centre)
        ph["dino"], ph["phrases_done"] = [], []
        self._add(ph, P.detections(raw_full, "FULL", [0, 0, W, H], 1.0, det["max_area_frac"]) + self._roi_dets(ph, vocab, ph["calls"], "v"))
        ph["phrases_done"] = list(vocab)
        ph["detection_how"] = "fresh DINO this session (vocabulary; full image + verified conveyor / truck ROIs), IoU merge 0.5"

    def _roi_dets(self, ph, phrases, log, tag):
        c, (W, H), det, out = self.model(), ph["size"], self.tc["detection"], []
        for cls in ("conveyor", "truck"):
            b = ph["refs"].get(cls, {}).get("box")
            if b is None:
                continue
            roi = make_roi(b, (W, H), self.tcfg)
            zp = self.session / "photos" / f"{ph['photo_id']}_inputs" / f"roi_{cls}.png"
            zp.parent.mkdir(exist_ok=True)
            if not zp.exists():
                x0, y0, x1, y1 = roi.crop_xyxy
                cv2.imwrite(str(zp), cv2.resize(ph["img"][y0:y1, x0:x1], roi.zoom_size, interpolation=cv2.INTER_CUBIC))
            raw = P.dino_call(c, zp, phrases, det["thresholds"], f"dino_roi_{ph['photo_id']}_{cls}_{tag}", log)
            out += P.detections(raw, f"ROI_{cls}_{tag}", roi.crop_xyxy, roi.scale, det["max_area_frac"])
        return out

    def _add(self, ph, dets):
        """Incremental form of placement_target.merge (greedy IoU clustering, highest score first): ids stay stable within the session."""
        reg, thr = ph["dino"], self.tc["detection"]["merge_iou"]
        nxt = max([int(c["id"][1:]) for c in reg if c["id"][1:].isdigit()], default=0)
        added = 0
        for d in sorted([d for d in dets if d["status"] == "KEPT"], key=lambda d: -d["score"]):
            hit = next((c for c in reg if iou(d["bbox_xyxy"], c["bbox_xyxy"]) >= thr), None)
            if hit:
                hit["phrases"] = sorted(set(hit["phrases"]) | {d["prompt"]})
                hit["conditions"] = sorted(set(hit["conditions"]) | {d["condition"]})
            else:
                nxt += 1
                added += 1
                reg.append({"id": f"m{nxt}", "bbox_xyxy": d["bbox_xyxy"], "score": d["score"], "phrases": [d["prompt"]], "conditions": [d["condition"]],
                            "source": "fresh DINO (this session)"})
        return added

    def _save_photo(self, ph):
        keep = {k: v for k, v in ph.items() if k != "img"}
        (self.session / "photos" / f"{ph['photo_id']}.json").write_text(json.dumps(keep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    # ---------------------------------------------------------------- accessibility (Python, task_context policy)
    def access(self, ph, c):
        key = (ph["photo_id"], tuple(c["bbox_xyxy"]))
        if key in self._acc:
            return self._acc[key]
        geo, start = self.geometry(), ph["refs"].get("robot", {}).get("start_img")
        if not geo["ok"] or ph["alignment"]["status"] != "ALIGNED" or start is None:
            why = geo.get("why") if not geo["ok"] else "photo not aligned with E1 (floor transform invalid)" if ph["alignment"]["status"] != "ALIGNED" else "robot start not found"
            a = {"accessible": None, "why": f"not checked: {why}"}
        else:
            ctx, fr = geo["ctx"], geo["ctx"]["fr"]
            poly = fr.img_to_cm(PT.box_poly(c["bbox_xyxy"])).tolist()
            gap = min((g for g in (geo["zr"].gap(poly, d) for d in geo["drivable"]) if g is not None), default=None)
            leg = SN.leg_record(ctx, SN.leg_to_region(ctx, self.gcfg, geo["navs"], 0, fr.cm_to_bev(fr.img_to_cm([start]))[0].tolist(), [poly], f"acc_{c['id']}", geo["L"], geo["W"]))
            a = {"gap_to_drivable_cm": None if gap is None else round(gap, 1), "adjacent": gap is not None and gap <= self.tc["policy"]["adjacent_cm"],
                 "approach_reachable_from_start": leg.get("path_B_feasible"), "approach_goal_cm": leg.get("goal_cm"), "approach_goal_image_px": leg.get("goal_image_px"),
                 "body_gap_cm": leg.get("gap_cm"), "floor_polygon_cm": np.round(poly, 1).tolist()}
            a["accessible"] = bool(a["adjacent"] and a["approach_reachable_from_start"])
            a["why"] = (f"{a['gap_to_drivable_cm']} cm from the drivable zone (limit {self.tc['policy']['adjacent_cm']} cm); "
                        f"approach pose {'reachable' if a['approach_reachable_from_start'] else 'NOT reachable'} from the robot start")
        self._acc[key] = a
        return a

    def floor_cm(self, ph, box):
        geo = self.geometry()
        if not geo["ok"] or ph["alignment"]["status"] != "ALIGNED":
            return None
        return np.round(geo["ctx"]["fr"].img_to_cm(PT.box_poly(box)), 1).tolist()

    # ---------------------------------------------------------------- one instruction
    def run(self, path, instruction, rules=None):
        rules = rules or self.vc["default_rules"]
        t0, calls, stages = time.time(), [], []
        ph = self.photo(path)
        self.n += 1
        rid = f"run_{self.n:03d}_{ph['photo_id']}"
        rdir = self.session / "runs" / rid
        for d in ("inputs", "qwen"):
            (rdir / d).mkdir(parents=True)
        W, H = ph["size"]
        rec = {"run_id": rid, "created": datetime.now().isoformat(timespec="seconds"), "instruction": instruction, "rules": rules,
               "photo": {k: ph[k] for k in ("photo_id", "path", "sha256", "size")}, "alignment_with_E1": ph["alignment"].get("status"),
               "detection_reused": ph["detection_how"], "cube_candidates": ph["cube_how"], "frames": FRAMES,
               "policy": {"config": str(self.tc_path), "prompt_ids": [self.tc["interpretation"]["prompt_id"], self.tc["roles"]["source_prompt_id"], self.tc["roles"]["destination_prompt_id"]],
                          "rules_text": self.tc["interpretation"]["rules_R1"] if rules == "R1" else None, "viewer_config": self.vc},
               "model_called": True, "source": {"status": "NOT_RUN"}, "destination": {"status": "NOT_RUN"}}
        try:
            self._infer(ph, instruction, rules, rdir, rec, calls, stages)
        except ModelWorkerError as e:     # stop button or worker failure: the run is kept with the stage that failed
            stages.append(stage("model_worker", "FAILED", f"{e.etype}: {e.message}"))
        rec["stages"], rec["first_failure"] = stages, first_failure(stages)
        rec["wall_s"] = round(time.time() - t0, 1)
        rec["model_calls"] = {"n_qwen": sum(c["stage"].startswith(("interp", "source", "destination")) for c in calls), "n_dino": sum(c["stage"].startswith("dino") for c in calls),
                              "models": sorted({f"{c['model']} @ {c.get('commit_hash')}" for c in calls}), "file": "model_calls.json"}
        (rdir / "model_calls.json").write_text(json.dumps(calls, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        rec["handoff"] = self._handoff(ph, rec)
        cv2.imwrite(str(rdir / "overlay.png"), draw(ph["img"], rec))
        cv2.imwrite(str(rdir / "overlay_candidates.png"), draw(ph["img"], rec, show_candidates=True))
        rec["files"] = {"overlay": str(rdir / "overlay.png"), "overlay_candidates": str(rdir / "overlay_candidates.png"), "run_json": str(rdir / "run.json")}
        (rdir / "run.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        self.runs.append(rec)
        self._save_photo(ph)
        self._write_session()
        return rec

    def _infer(self, ph, instruction, rules, rdir, rec, calls, stages):
        c, (W, H), tc, rel = self.model(), ph["size"], self.tc, self.rel
        task, r1 = {"instruction": instruction}, rules == "R1"
        # ---- 1 interpretation (Qwen, full photo)
        self.progress("1/5 작업지시 해석 (Qwen3-VL)")
        pr = TC.interp_prompt(tc, task, r1)
        (rdir / "qwen" / "interpretation_prompt.txt").write_text(pr, encoding="utf-8")   # exact prompt text kept next to the reply
        raw = P.qwen_call(c, pr, ph["path"], tc["interpretation"]["max_new_tokens"], "interpretation", tc["interpretation"]["prompt_id"], calls)
        (rdir / "qwen" / "interpretation.txt").write_text(raw, encoding="utf-8")
        it, err = P.parse_reply(raw, TC.INTERP_SCHEMA)
        rec["interpretation"] = {"parsed": it, "error": err, "raw_file": str(rdir / "qwen" / "interpretation.txt")}
        if not it:
            stages.append(stage("interpretation", "FAILED", err))
            return
        for k in ("source_reference", "destination_reference"):
            it[k]["bbox_image_px"] = P.qwen_box_to_image(it[k].get("bbox_2d"), [0, 0, W, H])
        srcobj = it["source_object"].get("name_en") or ""
        stages.append(stage("interpretation", "OK", f"move '{srcobj}' | source: {it['source_relation']} '{it['source_reference'].get('name_en')}' | "
                                                    f"destination: {it['destination_relation']} '{it['destination_reference'].get('name_en')}' | purpose: {it.get('purpose')}"))
        # ---- 2 detections for nouns not yet searched on this photo (DINO); otherwise the photo's detections are reused
        nouns = list(dict.fromkeys(P.norm_phrase(n, 10) for n in (srcobj, it["source_reference"].get("name_en"), it["destination_reference"].get("name_en")) if n))
        new = [n for n in nouns if n not in ph["phrases_done"]]
        if new:
            self.progress(f"2/5 새 물체 이름 탐지 (Grounding DINO): {', '.join(new)}")
            raw_full = P.dino_call(c, ph["path"], new, tc["detection"]["thresholds"], f"dino_full_{rec['run_id']}", calls)
            added = self._add(ph, P.detections(raw_full, f"FULL_{rec['run_id']}", [0, 0, W, H], 1.0, tc["detection"]["max_area_frac"])
                              + self._roi_dets(ph, new, calls, rec["run_id"]))
            ph["phrases_done"] = sorted(set(ph["phrases_done"]) | set(new))
            stages.append(stage("detection", "OK", f"new phrases {new}: {added} new candidates; {len(ph['dino'])} detector candidates on this photo"))
        else:
            stages.append(stage("detection", "REUSED", f"all nouns {nouns} already searched on this photo; {len(ph['dino'])} detector candidates ({ph['detection_how']})"))
        # ---- 3 reference objects -> search regions (task_context rules)
        refs = {}
        for side in ("source", "destination"):
            rname = it[f"{side}_reference"].get("name_en")
            rcls = TC.ref_class(rname, rel["reference_classes"])
            if rcls and ph["refs"].get(rcls, {}).get("box"):
                rbox, rhow = ph["refs"][rcls]["box"], f"class '{rcls}' -> {ph['refs'][rcls]['how']}"
            else:
                verified = [{"object_id": k, "bbox_xyxy": v["box"]} for k, v in ph["refs"].items() if v.get("box") and k != "robot"]
                rbox, rhow = S.snap_anchor(it[f"{side}_reference"].get("bbox_image_px"), verified, rel["snap_iou"])
            relname = it[f"{side}_relation"]
            if not rname and rbox is None and self.vc["whole_image_when_no_reference"]:
                region = {"type": "WHOLE_IMAGE", "rect": [0, 0, W, H], "rule": "the instruction names no reference (viewer rule, not task_context)"}
            else:
                region = TC.search_region(relname, rbox, (W, H), self.tcfg, rel)
            refs[side] = {"name": rname, "class": rcls, "box": rbox, "how": rhow, "relation": relname, "region": region}
            stages.append(stage(f"{side}_reference", "FAILED" if region["rect"] is None else "OK",
                                f"'{rname}' not grounded ({rhow})" if region["rect"] is None else f"'{rname}' [{relname}] -> {region['type']} {region['rect']} ({rhow})"))
        rec["references"] = refs
        # ---- 4 source: candidates in the region -> Qwen roles -> decision (ties -> AMBIGUOUS)
        src = {"status": "NO_REFERENCE", "ids": [], "why": f"source reference '{refs['source']['name']}' not grounded"}
        reg = refs["source"]["region"]
        if reg["rect"] is not None:
            use_cube = "cube" in srcobj.lower() and ph["cube"] is not None
            if use_cube:
                pool, how = ph["cube"], "red-cube candidates (" + ph["cube_how"] + ")"
            else:
                rb = refs["source"]["box"]
                pool = [d for d in ph["dino"] if box_area(d["bbox_xyxy"]) < rel["region_scale_frac"] * box_area(reg["rect"]) and (rb is None or iou(d["bbox_xyxy"], rb) < 0.5)]
                how = "detector candidates in the region (task_context rule)"
                if reg["type"] == "WHOLE_IMAGE":
                    head = srcobj.lower().split()[-1:]
                    pool = [d for d in pool if any(w in " ".join(d["phrases"]) for w in head)]
                    how = f"detector candidates whose phrase contains '{' '.join(head)}' (whole-image search)"
            inside = sorted([d for d in pool if TC.in_region(d["bbox_xyxy"], reg, rel)], key=lambda d: (d["bbox_xyxy"][1], d["bbox_xyxy"][0]))
            stages.append(stage("source_candidates", "OK" if inside else "FAILED", f"{len(inside)} of {len(pool)} {how} inside the region"))
            if inside:
                self.progress(f"3/5 옮길 물체 판단 (Qwen3-VL, 후보 {len(inside)}개)")
                reply, imgs, err = self._roles(ph, reg, inside, "source_template", S.SRC_SCHEMA, "object_to_move_id", instruction, r1, refs["source"]["name"], srcobj, rdir, "source", calls)
                if err:
                    stages.append(stage("source_roles", "FAILED", err))
                    src = {"status": err.split(" ")[0], "ids": [], "why": err, "images": imgs}
                else:
                    d = TC.decide_source(reply, len(inside))
                    ratings = {inside[int(r["id"]) - 1]["id"]: r for r in (reply or {}).get("candidates", []) if S._num(r.get("id"), len(inside))}
                    src = {**d, "ids": [inside[k - 1]["id"] for k in d["numbers"]], "boxes": [inside[k - 1]["bbox_xyxy"] for k in d["numbers"]],
                           "ratings": ratings, "numbering": {k: x["id"] for k, x in enumerate(inside, 1)}, "images": imgs, "reply_reason": (reply or {}).get("reason"),
                           "candidates": [{"id": x["id"], "bbox_xyxy": x["bbox_xyxy"]} for x in inside]}
                    stages.append(stage("source_roles", "OK" if reply else "FAILED", "Qwen reply valid" if reply else "Qwen reply invalid (see qwen/)"))
                    stages.append(stage("source_decision", "OK" if d["status"] == "SELECTED" else "UNCERTAIN" if d["status"] == "AMBIGUOUS" else "FAILED",
                                        f"{d['status']} {src['ids']}: {d['why']}"))
            else:
                src = {"status": "NOT_FOUND", "ids": [], "why": "no candidate inside the search region"}
        rec["source"] = src
        # ---- 5 destination: candidates in the region (minus the object to move) -> Qwen roles -> accessibility (Python) -> decision
        dst = {"status": "NO_REFERENCE", "why": f"destination reference '{refs['destination']['name']}' not grounded"}
        reg = refs["destination"]["region"]
        if reg["rect"] is not None:
            inc, regions = S.zone_members(ph["dino"], reg["rect"], rel["region_scale_frac"])
            sboxes = [x["bbox_xyxy"] for x in (ph["cube"] or []) + ph["dino"] if x["id"] in src.get("ids", [])]
            inc = sorted([x for x in inc if not any(iou(x["bbox_xyxy"], b) >= 0.5 for b in sboxes)], key=lambda x: (x["bbox_xyxy"][1], x["bbox_xyxy"][0]))
            stages.append(stage("destination_candidates", "OK" if inc else "FAILED",
                                f"{len(inc)} detector candidates inside the region ({len(regions)} region-scale boxes not numbered; the object to move excluded)"))
            if inc:
                self.progress(f"4/5 놓을 곳 판단 (Qwen3-VL, 후보 {len(inc)}개)")
                reply, imgs, err = self._roles(ph, reg, inc, "destination_template", S.DEST_SCHEMA, "best_destination_id", instruction, r1, refs["destination"]["name"], srcobj, rdir, "destination", calls)
                if err:
                    stages.append(stage("destination_roles", "FAILED", err))
                    dst = {"status": err.split(" ")[0], "why": err, "images": imgs}
                else:
                    stages.append(stage("destination_roles", "OK" if reply else "FAILED", "Qwen reply valid" if reply else "Qwen reply invalid (see qwen/)"))
                    rated = {int(r["id"]): r for r in (reply or {}).get("candidates", []) if S._num(r.get("id"), len(inc))}
                    todo = [k for k, r in rated.items() if r["placement_role"] in ("suitable", "possible")]
                    self.progress(f"5/5 접근 가능성 계산 (Python 기존 접근 규칙, 후보 {len(todo)}개)")
                    acc = {k: self.access(ph, inc[k - 1]) for k in todo}
                    checked = all(a["accessible"] is not None for a in acc.values())
                    stages.append(stage("accessibility", "OK" if checked else "NOT_CHECKED",
                                        f"{sum(bool(a['accessible']) for a in acc.values())} of {len(acc)} suitable/possible candidates accessible" if checked
                                        else next(a["why"] for a in acc.values() if a["accessible"] is None)))
                    dd = TC.decide_destination(reply, {k: (a["accessible"] if checked else True) for k, a in acc.items()}, len(inc))
                    if not checked and dd["kind"]:
                        dd["kind"] += "_ACCESSIBILITY_NOT_CHECKED"
                    dst = {**dd, "id": inc[dd["number"] - 1]["id"] if dd["number"] else None, "bbox_xyxy": inc[dd["number"] - 1]["bbox_xyxy"] if dd["number"] else None,
                           "ratings": {inc[k - 1]["id"]: r for k, r in rated.items()}, "numbering": {k: x["id"] for k, x in enumerate(inc, 1)}, "images": imgs,
                           "reply_reason": (reply or {}).get("reason"), "candidates": [{"id": x["id"], "bbox_xyxy": x["bbox_xyxy"], "phrases": x["phrases"]} for x in inc],
                           "accessibility": {inc[k - 1]["id"]: a for k, a in acc.items()},
                           "tied_ids": [inc[k - 1]["id"] for k in tied_destinations(rated, acc)] if dd["status"] == "AMBIGUOUS" else []}
                    if dd["number"]:
                        x, r = inc[dd["number"] - 1], rated.get(dd["number"])
                        dst["goal_kind"], dst["goal_kind_why"] = goal_kind(x, (r or {}).get("appearance"), (W, H), self.vc["goal_region"])
                        dst["placement"] = placement_status(dst["goal_kind"], r)
                    stages.append(stage("destination_decision", "OK" if dd["status"] == "SELECTED" else "UNCERTAIN" if dd["status"] == "AMBIGUOUS" else "FAILED",
                                        f"{dd['status']} {dst['id'] or ''} {dd['kind'] or ''}: {dd['why']}"))
            else:
                dst = {"status": "NOT_FOUND", "why": "no candidate inside the destination region"}
        rec["destination"] = dst

    def _roles(self, ph, reg, cands, key, schema, best_key, instruction, r1, reference, srcobj, rdir, tag, calls):
        """Numbered image(s) of the region -> Qwen role call(s). More than max_numbered candidates -> several images (no truncation)."""
        groups = chunks(cands, self.rel["max_numbered"])
        if len(groups) > self.vc["max_role_calls_per_side"]:
            return None, [], f"TOO_MANY_CANDIDATES ({len(cands)} candidates need {len(groups)} Qwen calls > {self.vc['max_role_calls_per_side']})"
        pr = TC.role_prompt(self.tc, key, {"instruction": instruction}, r1, reference, srcobj)
        (rdir / "qwen" / f"{tag}_prompt.txt").write_text(pr, encoding="utf-8")            # same prompt for every numbered image of this side
        replies, imgs = [], []
        for g_i, g in enumerate(groups, 1):
            vr = S.view_rect(reg["rect"], [x["bbox_xyxy"] for x in g], ph["size"])
            p = rdir / "inputs" / f"{tag}_{g_i}.png"
            S.numbered(ph["img"], vr, S.zoom_scale(vr, self.tcfg), g, p)
            raw = P.qwen_call(self.model(), pr, p, self.tc["roles"]["max_new_tokens"], f"{tag}_{g_i}", self.tc["roles"][f"{tag}_prompt_id"], calls)
            (rdir / "qwen" / f"{tag}_{g_i}.txt").write_text(raw, encoding="utf-8")
            obj, err = P.parse_reply(raw, schema)
            replies.append(obj)
            imgs.append({"path": str(p), "view_rect": vr, "numbers": [sum(map(len, groups[:g_i - 1])) + 1, sum(map(len, groups[:g_i]))], "error": err})
        return merge_replies(replies, [len(g) for g in groups], best_key), imgs, None

    def _handoff(self, ph, rec):
        """A / B in a form the existing map and planner code can take later (ids, photo px, floor cm, approach pose). Nothing is sent anywhere."""
        s, d = rec["source"], rec["destination"]
        out = {"photo_id": ph["photo_id"], "frames": FRAMES, "source": None, "destination": None}
        if s.get("status") == "SELECTED":
            out["source"] = {"id": s["ids"][0], "bbox_xyxy": s["boxes"][0], "floor_polygon_cm": self.floor_cm(ph, s["boxes"][0])}
        if d.get("status") == "SELECTED":
            a = d["accessibility"].get(d["id"], {})
            out["destination"] = {"id": d["id"], "bbox_xyxy": d["bbox_xyxy"], "goal_kind": d["goal_kind"], "placement": d["placement"],
                                  "floor_polygon_cm": self.floor_cm(ph, d["bbox_xyxy"]), "approach_goal_cm": a.get("approach_goal_cm"), "approach_goal_image_px": a.get("approach_goal_image_px")}
        return out

    def _write_session(self):
        idx = {"session": str(self.session), "v02_code_sha256_at_start": self.v02_sha, "v02_code_unchanged_now": code_fingerprint()["sha256"] == self.v02_sha,
               "task_context_config": str(self.tc_path),
               "runs": [{"run_id": r["run_id"], "photo_id": r["photo"]["photo_id"], "instruction": r["instruction"], "rules": r["rules"],
                         "source": f"{r['source'].get('status')} {' '.join(r['source'].get('ids') or [])}".strip(),
                         "destination": f"{r['destination'].get('status')} {r['destination'].get('id') or ''}".strip(),
                         "first_failure": r["first_failure"], "wall_s": r["wall_s"], "dir": str(Path(r["files"]["run_json"]).parent)} for r in self.runs]}
        (self.session / "session.json").write_text(json.dumps(idx, indent=1, ensure_ascii=False), encoding="utf-8")


# ================================================================================ drawing (English labels on the photo; Korean in the GUI panel)
COL = {"A": (40, 40, 235), "B": (235, 110, 20), "regA": (80, 170, 255), "regB": (255, 210, 90), "ref": (235, 235, 235), "cand": (170, 170, 170), "off": (150, 150, 150)}


def _dashed(im, b, col, th=2, dash=12):
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    for (a0, a1, fixed, horiz) in ((x0, x1, y0, True), (x0, x1, y1, True), (y0, y1, x0, False), (y0, y1, x1, False)):
        for s in range(a0, a1, 2 * dash):
            e = min(s + dash, a1)
            p, q = ((s, fixed), (e, fixed)) if horiz else ((fixed, s), (fixed, e))
            cv2.line(im, p, q, col, th, cv2.LINE_AA)


def _label(im, text, x, y, col, sc=0.62, th=2):
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
    x = int(min(max(0, x), im.shape[1] - tw - 6))
    y = int(min(max(tht + 6, y), im.shape[0] - 4))
    cv2.rectangle(im, (x, y - tht - 6), (x + tw + 6, y + 4), (0, 0, 0), -1)
    cv2.putText(im, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def _name(rating, fallback):
    ap = (rating or {}).get("appearance") or []
    return str(ap[0]) if ap else fallback


def draw(img, rec, show_candidates=False):
    im = img.copy()
    refs, s, d = rec.get("references") or {}, rec.get("source") or {}, rec.get("destination") or {}
    it = (rec.get("interpretation") or {}).get("parsed") or {}
    srcname = (it.get("source_object") or {}).get("name_en") or "object"
    for side, col in (("source", COL["regA"]), ("destination", COL["regB"])):
        r = refs.get(side)
        if not r:
            continue
        if r["region"].get("rect") and r["region"]["type"] != "WHOLE_IMAGE":
            _dashed(im, r["region"]["rect"], col, 2, 14)
            _label(im, f"{'A' if side == 'source' else 'B'} search: {r['relation']} {r['name']} ({r['region']['type']})", r["region"]["rect"][0] + 4, r["region"]["rect"][1] + 22, col, 0.5, 1)
        if r.get("box"):
            b = [int(round(v)) for v in r["box"]]
            cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), COL["ref"], 1)
            _label(im, f"ref: {r['name']}", b[0], b[3] + 18, COL["ref"], 0.45, 1)
    if show_candidates:
        for side, key in (("source", "A"), ("destination", "B")):
            x = s if side == "source" else d
            for k, c in enumerate(x.get("candidates") or [], 1):
                b = [int(round(v)) for v in c["bbox_xyxy"]]
                rt = (x.get("ratings") or {}).get(c["id"]) or {}
                tag = rt.get("role") or rt.get("placement_role") or "-"
                cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), COL["cand"], 1)
                _label(im, f"{key}{k} {tag[:4]}", b[0], b[1] - 2, COL["cand"], 0.38, 1)
    # destination candidates that Qwen accepted but that are not accessible (drawn when they are the reason B is missing)
    for cid, a in (d.get("accessibility") or {}).items():
        if a.get("accessible") is False and (d.get("status") == "NOT_ACCESSIBLE" or show_candidates):
            c = next(x for x in d["candidates"] if x["id"] == cid)
            _dashed(im, c["bbox_xyxy"], COL["off"], 2, 6)
            _label(im, f"B? not accessible ({a.get('gap_to_drivable_cm')} cm) {cid}", c["bbox_xyxy"][0], c["bbox_xyxy"][3] + 18, COL["off"], 0.45, 1)
    ca = cb = None
    if s.get("status") == "SELECTED":
        b = [int(round(v)) for v in s["boxes"][0]]
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), COL["A"], 3)
        _label(im, f"A SOURCE: {_name(s['ratings'].get(s['ids'][0]), srcname)} [{s['ids'][0]}]", b[0], b[1] - 6, COL["A"])
        ca = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
    elif s.get("status") == "AMBIGUOUS":
        for i, (cid, bx) in enumerate(zip(s["ids"], s["boxes"]), 1):
            _dashed(im, bx, COL["A"], 2, 6)
            _label(im, f"A? tie {i}/{len(s['ids'])} [{cid}]", bx[0], bx[1] - 6, COL["A"], 0.5, 1)
    if d.get("status") == "SELECTED":
        b = [int(round(v)) for v in d["bbox_xyxy"]]
        if d.get("goal_kind") == "GOAL_REGION":
            ov = im.copy()
            cv2.rectangle(ov, tuple(b[:2]), tuple(b[2:]), COL["B"], -1)
            im = cv2.addWeighted(ov, 0.25, im, 0.75, 0)
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), COL["B"], 3)
        _label(im, f"B DESTINATION: {_name(d['ratings'].get(d['id']), 'target')} [{d['id']}]", b[0], b[1] - 30, COL["B"])
        _label(im, f"{d.get('goal_kind')} | {d.get('placement')} | {d.get('kind')}", b[0], b[1] - 6, COL["B"], 0.48, 1)
        cb = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        g = (d.get("accessibility") or {}).get(d["id"], {}).get("approach_goal_image_px")
        if g:
            gp = (int(round(g[0])), int(round(g[1])))
            cv2.drawMarker(im, gp, COL["B"], cv2.MARKER_TILTED_CROSS, 18, 2)
            _label(im, "robot stop (approach)", gp[0] + 10, gp[1] + 20, COL["B"], 0.42, 1)
    elif d.get("status") == "AMBIGUOUS":
        for i, cid in enumerate(d.get("tied_ids") or [], 1):
            c = next(x for x in d["candidates"] if x["id"] == cid)
            _dashed(im, c["bbox_xyxy"], COL["B"], 2, 6)
            _label(im, f"B? {i}/{len(d['tied_ids'])} {cid}", c["bbox_xyxy"][0], c["bbox_xyxy"][1] - 6, COL["B"], 0.45, 1)
    if ca and cb:
        cv2.arrowedLine(im, ca, cb, (255, 255, 255), 4, cv2.LINE_AA, tipLength=0.03)
        cv2.arrowedLine(im, ca, cb, (60, 220, 60), 2, cv2.LINE_AA, tipLength=0.03)
    banner = f"{rec['run_id']} | rules {rec['rules']} | A: {s.get('status')} | B: {d.get('status')} | first failure: {rec.get('first_failure') or 'none'}"
    _label(im, banner, 8, 30, (255, 255, 255), 0.62, 2)
    return im


# ================================================================================ text summary (Korean; GUI panel and command line)
STATUS_KO = {"SELECTED": "선택됨", "AMBIGUOUS": "판단 보류(동률)", "NOT_FOUND": "찾지 못함", "NOT_ACCESSIBLE": "접근 불가", "NO_REFERENCE": "기준 시설 못 찾음",
             "NOT_RUN": "실행 안 됨"}


def summary(rec):
    it = (rec.get("interpretation") or {}).get("parsed") or {}
    s, d, refs = rec["source"], rec["destination"], rec.get("references") or {}
    st = lambda x: STATUS_KO.get(x.get("status"), x.get("status"))   # noqa: E731
    out = {"입력": rec["instruction"], "사진": f"{rec['photo']['photo_id']} ({Path(rec['photo']['path']).name})", "규칙": rec["rules"]}
    if it:
        out["해석"] = (f"목적: {it.get('purpose')} / 옮길 물체: {(it.get('source_object') or {}).get('name_en')} / "
                     f"출발 기준: {refs.get('source', {}).get('name')} [{it.get('source_relation')}] / "
                     f"목적지 의미: {it.get('destination_meaning')} / 목적지 기준: {refs.get('destination', {}).get('name')} [{it.get('destination_relation')}]")
    else:
        out["해석"] = f"실패: {(rec.get('interpretation') or {}).get('error')}"
    a = f"{st(s)}"
    if s.get("ids"):
        a += " " + ", ".join(f"{i} ({_name(s['ratings'].get(i), '?')}, {[round(v) for v in b]})" for i, b in zip(s["ids"], s["boxes"]))
    out["A 출발 대상"] = a
    out["A 근거"] = f"{s.get('why')}" + (f" / Qwen: {s.get('reply_reason')}" if s.get("reply_reason") else "")
    b = st(d)
    if d.get("id"):
        r = d["ratings"].get(d["id"], {})
        acc = d["accessibility"].get(d["id"], {})
        b += (f" {d['id']} ({_name(r, '?')}, {[round(v) for v in d['bbox_xyxy']]}) / {d.get('goal_kind')} / {d.get('placement')} / 적재 상태: {r.get('load_state')} / "
              f"역할: {r.get('placement_role')} / {d.get('kind')} / 주행 구역까지 {acc.get('gap_to_drivable_cm')} cm")
    out["B 목적 대상"] = b
    out["B 근거"] = f"{d.get('why')}" + (f" / Qwen: {d.get('reply_reason')}" if d.get("reply_reason") else "")
    out["판단 상태"] = f"A {s.get('status')} · B {d.get('status')} · 최초 실패 단계: {rec.get('first_failure') or '없음'}"
    return out
