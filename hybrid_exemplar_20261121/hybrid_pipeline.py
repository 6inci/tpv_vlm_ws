#!/usr/bin/env python3
"""Hybrid pipeline (config.yaml), a subclass of multimodel_target_viewer.MMViewer (imported read-only, nothing modified).

  photo stage (unchanged MMViewer.photo): Grounding DINO candidates + SigLIP2-base facility check (truck, conveyor, robot)
  1 Qwen3-VL-4B, TEXT ONLY      instruction -> closed-list slots (object, landmark, relation, destination kind). No image, no box.
  2 SigLIP2 so400m              size-gated DINO candidates vs site exemplar crops (leave-one-photo-out bank)
  3 Python                      facility regions (verified facility box, never a VLM box), graspable list
  4 Python                      A = cube found in the WHOLE photo (score >= threshold, margin to any different object), then relation check
  5 Python                      B (empty pallet) = exemplar top-1 among candidates without a cube-like box inside, margin, no-placement zone,
                                accessibility. B (near a landmark) = the unchanged MMViewer destination step (VLM roles on a numbered image)
Coordinates: every box is undistorted_px of the 1920x1080 photo (same as MMViewer).
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
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, CFG["pipeline_base"])
import mm_pipeline as M  # noqa: E402
sys.path.insert(0, CFG["reference_objects"])
import run_test as RT  # noqa: E402  (load_reference only: AI-read boxes -> exemplar crops and the same-place drop)

TC, P, S, VP = M.TC, M.P, M.S, M.VP
iou = M.iou
_slots = CFG["interpretation"]
SLOT_SCHEMA = {"type": "object",
               "required": ["object", "source_landmark", "source_relation", "destination_kind", "destination_landmark", "destination_relation"],
               "properties": {"object": {"type": "string", "enum": ["red cube", "any object", "other"]},
                              "source_landmark": {"type": "string", "enum": ["truck_loading_bay", "conveyor", "mobile_robot", "none", "other"]},
                              "destination_landmark": {"type": "string", "enum": ["truck_loading_bay", "conveyor", "mobile_robot", "none", "other"]},
                              "source_relation": {"type": "string", "enum": ["near", "on", "none"]},
                              "destination_relation": {"type": "string", "enum": ["near", "on", "none"]},
                              "destination_kind": {"type": "string", "enum": ["empty_pallet", "near_landmark", "other"]}}}
OBJECT_EN = {"red cube": "red cube", "any object": "object"}      # English noun for the unchanged destination role prompt / SigLIP2 labels


# ================================================================================ pure functions (test_offline.py)
def side_px(b):
    return max(b[2] - b[0], b[3] - b[1])


def centre_in(a, b):
    """Centre of box a inside box b."""
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def distinct(a, b, thr):
    """Different objects: low overlap and neither box holds the other's centre (a cube and the pallet under it are NOT distinct)."""
    return iou(a, b) < thr and not centre_in(a, b) and not centre_in(b, a)


def exemplar_scores(cand_embs, bank_embs):
    """cand_embs {id: (V, D)}, bank_embs [(V, D)] unit vectors -> {id: mean over exemplars and views of the view-matched cosine}."""
    B = np.stack(bank_embs)
    return {k: float(np.mean(np.sum(B * np.asarray(e)[None], axis=2))) for k, e in cand_embs.items()}


def decide(cands, score, min_score, min_margin, nms_iou, distinct_iou):
    """Top-1 by score. s2 = best score of a DIFFERENT object. SELECTED needs s1 >= min_score (if given) and s1 - s2 >= min_margin."""
    ranked = []
    for c in sorted(cands, key=lambda c: -score[c["id"]]):
        if all(iou(c["bbox_xyxy"], k["bbox_xyxy"]) < nms_iou for k in ranked):
            ranked.append(c)
    if not ranked:
        return {"status": "NOT_FOUND", "top": None, "s1": None, "s2": None, "why": "no candidate", "ranked": []}
    top = ranked[0]
    s1 = score[top["id"]]
    others = [score[c["id"]] for c in cands if distinct(c["bbox_xyxy"], top["bbox_xyxy"], distinct_iou)]
    s2 = max(others) if others else None
    out = {"top": top, "s1": round(s1, 4), "s2": None if s2 is None else round(s2, 4), "ranked": ranked[:5]}
    if min_score is not None and s1 < min_score:
        return {**out, "status": "NOT_FOUND", "why": f"top-1 score {s1:.3f} < {min_score}"}
    if s2 is not None and s1 - s2 < min_margin:
        return {**out, "status": "AMBIGUOUS", "why": f"top-1 {s1:.3f} vs a different object {s2:.3f}: margin < {min_margin}"}
    return {**out, "status": "SELECTED", "why": f"top-1 {s1:.3f}" + (f", margin {s1 - s2:.3f} >= {min_margin}" if s2 is not None else ", no other object")}


def slot_box(objs, o):
    """Pallet slot of a reference object: a cube's slot is the pad_with_cube box of its photo."""
    return next(x["box"] for x in objs if x["kind"] == "pad_with_cube") if o["kind"] == "cube" else o["box"]


def bank_for(bank, test_objs, test_pid, thr):
    """Leave-one-photo-out; also drop exemplars whose slot overlaps a pallet slot of the test photo (same place seen again).
    test_objs are used ONLY to drop exemplars (never as candidates or answers)."""
    slots = [o["box"] for o in test_objs if o["kind"] in ("pad_with_cube", "empty_pad")]
    keep, dropped = [], []
    for e in bank:
        if e["photo_id"] == test_pid:
            continue
        (dropped if any(iou(e["slot"], s) >= thr for s in slots) else keep).append(e)
    return keep, dropped


def source_classes(obj, graspable):
    want = ["cube"] if obj == "red cube" else list(graspable) if obj == "any object" else []
    return [c for c in want if c in graspable]


# ================================================================================ viewer
class HybridViewer(M.MMViewer):
    def __init__(self, session_dir, progress=print):
        super().__init__(session_dir=session_dir, progress=progress, backend=CFG["vlm_backend"])
        assert self.mc["vlm_backends"][self.backend]["where"] == "v02_worker", "text-only slot call is written for the v0.2 worker (4B)"
        self.hc, self.xm, self.xemb, self.bank = CFG, None, {}, None
        self.refs = {r["photo_id"]: r["objects"] for r in RT.load_reference()}
        (self.session / "hybrid_config_snapshot.yaml").write_text((HERE / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    # ---------------------------------------------------------------- so400m in its own mm_worker process (unchanged worker file)
    def _x(self, device):
        if self.xm is None or not self.xm.alive:
            self.progress("so400m 작업 프로세스 시작 (qwen_vl_q4 환경)")
            self.xm = M.MMClient(self.mc["worker"], self.session / "worker_logs")
            self.xm.start()
        xc = self.hc["exemplar_model"]
        if "siglip" not in self.xm.loaded:
            self.xm.loaded["siglip"] = self.xm.call("load", name="siglip", model_id=xc["model_id"], kind="siglip", dtype=xc["dtype"], device="cpu")
        if device == "cuda":     # 4B bf16 (8.3 GiB) does not fit next to so400m + DINO on 12 GB
            self._unload(self.client, "qwen")
            if self.mm is not None and self.mm.alive and self.mm.loaded.get("siglip", {}).get("device") == "cuda":
                self._siglip("cpu")
        if self.xm.loaded["siglip"].get("device") != device:
            self.xm.call("siglip_device", device=device)
            self.xm.loaded["siglip"]["device"] = device
        return self.xm

    def _embed(self, paths, tag, log):
        m = self._x("cuda")
        tw = time.perf_counter()
        r = m.call("siglip_images", paths=paths, batch=self.hc["exemplar_model"]["batch"])
        log.append({"stage": f"siglip_so400m_{tag}", "model": r["model"], "commit_hash": r["commit_hash"], "n_images": r["n"], "seconds": r["seconds"],
                    "wall_s": round(time.perf_counter() - tw, 2), "device": r["device"], "peak_vram_gib": r.get("peak_allocated_gib")})
        self._x("cpu")
        return np.array(r["embeddings"], np.float32)

    def stop_models(self):
        super().stop_models()
        if self.xm is not None:
            self.xm.kill()

    def close(self):
        if self.xm is not None:
            self.xm.close()
        super().close()

    # ---------------------------------------------------------------- exemplar bank (reference crops of all photos; selected per run)
    def build_bank(self, log):
        if self.bank is not None:
            return self.bank
        views, vc = self.hc["exemplar"]["views"], self.mc["siglip"]["crops"]
        kinds = {v["reference_kind"]: k for k, v in self.hc["exemplar"]["classes"].items()}
        paths = {p["photo_id"]: p["path"] for p in self.list_photos()}
        d = self.session / "exemplar_bank"
        d.mkdir(exist_ok=True)
        items, files = [], []
        for pid, objs in self.refs.items():
            img = cv2.imread(paths[pid], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            H, W = img.shape[:2]
            for o in objs:
                if o["kind"] not in kinds:
                    continue
                fs = []
                for v in views:
                    r = M.square_crop(o["box"], (W, H), vc[v]["pad_ratio"], vc[v]["min_px"])
                    p = d / f"{pid}_{o['key']}_{v}.png"
                    cv2.imwrite(str(p), img[r[1]:r[3], r[0]:r[2]])
                    fs.append(str(p))
                items.append({"cls": kinds[o["kind"]], "photo_id": pid, "key": o["key"], "box": o["box"], "slot": slot_box(objs, o), "crops": fs})
                files += fs
        self.progress(f"so400m 예시 이미지 {len(items)}개 임베딩")
        E, nv = self._embed(files, "exemplar_bank", log), len(views)
        for i, it in enumerate(items):
            it["embs"] = E[i * nv:(i + 1) * nv]
        (self.session / "exemplar_bank.json").write_text(json.dumps([{k: v for k, v in it.items() if k != "embs"} for it in items], indent=1), encoding="utf-8")
        np.savez_compressed(self.session / "exemplar_bank_embeddings.npz", **{f"{it['photo_id']}_{it['key']}": it["embs"] for it in items})
        self.bank = items
        return items

    def gated(self, ph, log):
        """Size-gated DINO candidates of the photo with so400m embeddings (computed once per photo)."""
        g, views = self.hc["candidate_gate"], self.hc["exemplar"]["views"]
        cands = [x for x in ph["dino"] if g["min_side_px"] <= side_px(x["bbox_xyxy"]) <= g["max_side_px"]]
        todo = [x for x in cands if (ph["photo_id"], x["id"]) not in self.xemb]
        if todo:
            self.progress(f"so400m 후보 비교 ({len(todo)}개 후보 × {len(views)} 이미지, DINO 후보 {len(ph['dino'])}개 중 크기 조건 통과)")
            crops = {x["id"]: self._crops(ph, x["id"], x["bbox_xyxy"]) for x in todo}
            E = self._embed([crops[x["id"]][v]["path"] for x in todo for v in views], f"candidates_{ph['photo_id']}", log)
            for i, x in enumerate(todo):
                self.xemb[(ph["photo_id"], x["id"])] = E[i * len(views):(i + 1) * len(views)]
        return cands

    def identity_scores(self, ph, cands, bank):
        """Step 4: {class: {candidate id: score}} = so400m exemplar similarity. Replaced only by the offline ablation (ablation_offline.py)."""
        embs = {x["id"]: self.xemb[(ph["photo_id"], x["id"])] for x in cands}
        return {cls: exemplar_scores(embs, [e["embs"] for e in bank if e["cls"] == cls]) for cls in self.hc["exemplar"]["classes"]}

    def vlm_text(self, prompt, max_new_tokens, stage, prompt_id, log):
        """Text-only Qwen call on the v0.2 worker (qwen_generate with image_path=None)."""
        if self.mm is not None and self.mm.alive and self.mm.loaded.get("siglip", {}).get("device") == "cuda":
            self._siglip("cpu")
        c = self.model()
        c.ensure_loaded("qwen")
        tw = time.perf_counter()
        r = c.call("qwen_generate", prompt=prompt, image_path=None, max_new_tokens=max_new_tokens)
        log.append({"stage": stage, "model": r["model"], "commit_hash": r["commit_hash"], "prompt_id": prompt_id, "prompt_sha256": M.sha256_text(prompt),
                    "image_path": None, "n_input_tokens": r["n_input_tokens"], "n_output_tokens": r["n_output_tokens"], "hit_max_new_tokens": r["hit_max_new_tokens"],
                    "latency_s": r["latency_s"], "wall_s": round(time.perf_counter() - tw, 2), "peak_vram_gib": r.get("peak_allocated_gib"), "raw_response": r["text"]})
        return r["text"]

    # ---------------------------------------------------------------- one instruction (same record layout as MMViewer.run)
    def run(self, path, instruction, rules=None):
        rules = rules or self.hc["rules"]
        t0, calls, stages = time.time(), [], []
        ph = self.photo(path)
        t_photo = round(time.time() - t0, 1)
        self.n += 1
        rid = f"run_{self.n:03d}_{ph['photo_id']}"
        rdir = self.session / "runs" / rid
        for d in ("inputs", "qwen"):
            (rdir / d).mkdir(parents=True)
        rec = {"run_id": rid, "created": datetime.now().isoformat(timespec="seconds"), "instruction": instruction, "rules": rules, "pipeline": "hybrid_exemplar_20261121",
               "vlm_backend": {"key": self.backend, **self.mc["vlm_backends"][self.backend]}, "exemplar_model": self.hc["exemplar_model"],
               "photo": {k: ph[k] for k in ("photo_id", "path", "sha256", "size")}, "alignment_with_E1": ph["alignment"].get("status"),
               "photo_detection": {"how": ph["detection_how"], "file": str(self.session / "photos" / f"{ph['photo_id']}.json")},
               "frames": VP.FRAMES, "config": {"hybrid": str(HERE / "config.yaml"), "viewer": str(Path(CFG["pipeline_base"]) / "config.yaml")},
               "model_called": True, "source": {"status": "NOT_RUN"}, "destination": {"status": "NOT_RUN"}, "timing_s": {"photo_stage_this_run": t_photo}}
        try:
            self._infer(ph, instruction, rules, rdir, rec, calls, stages)
        except M.ModelWorkerError as e:
            stages.append({"name": "model_worker", "status": "FAILED", "summary": f"{e.etype}: {e.message}", "model": "-"})
        rec["stages"], rec["first_failure"] = stages, M.first_failure(stages)
        rec["wall_s"] = round(time.time() - t0, 1)
        (rdir / "model_calls.json").write_text(json.dumps(calls, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        rec["models_used"] = M.models_used(calls, ph)
        rec["handoff"] = self._handoff(ph, rec)
        rec["narrative"] = narrative(rec)
        cv2.imwrite(str(rdir / "overlay.png"), M.draw(ph["img"], rec))
        cv2.imwrite(str(rdir / "overlay_candidates.png"), M.draw(ph["img"], rec, show_candidates=True))
        rec["files"] = {"overlay": str(rdir / "overlay.png"), "overlay_candidates": str(rdir / "overlay_candidates.png"), "run_json": str(rdir / "run.json")}
        (rdir / "run.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        self.runs.append(rec)
        self._save_photo(ph)
        self._write_session()
        return rec

    def _infer(self, ph, instruction, rules, rdir, rec, calls, stages):
        (W, H), rel, dc, hc, ck = ph["size"], self.rel, self.hc["decision"], self.hc, self.mc["checks"]
        vname, xname = self.mc["vlm_backends"][self.backend]["label"], hc["exemplar_model"]["model_id"]
        st = lambda name, status, summary, model: stages.append({"name": name, "status": status, "summary": summary, "model": model})   # noqa: E731
        tm = rec["timing_s"]
        # ---- 2 exemplar scores (so400m). Runs first: it does not depend on the slots, and Qwen then stays loaded for the rest of the run
        t = time.perf_counter()
        self.progress("1/4 예시 이미지 비교 (SigLIP2 so400m)")
        bank, dropped = bank_for(self.build_bank(calls), self.refs.get(ph["photo_id"], []), ph["photo_id"], hc["exemplar"]["same_place_iou"])
        cands = self.gated(ph, calls)
        score = self.identity_scores(ph, cands, bank)
        rec["exemplar"] = {"bank_used": {cls: [f"{e['photo_id']}:{e['key']}" for e in bank if e["cls"] == cls] for cls in hc["exemplar"]["classes"]},
                           "bank_dropped_same_place": [f"{e['photo_id']}:{e['key']}" for e in dropped],
                           "n_dino": len(ph["dino"]), "n_gated": len(cands),
                           "scores": {cls: {x["id"]: round(s, 4) for x in cands for s in [score[cls][x["id"]]]} for cls in score}}
        tm["exemplar"] = round(time.perf_counter() - t, 2)
        st("exemplar_scores", "OK", f"{len(cands)} of {len(ph['dino'])} DINO candidates in the size gate; exemplars "
                                    + ", ".join(f"{c} {len(v)}" for c, v in rec["exemplar"]["bank_used"].items()) + f" (dropped same place: {len(dropped)})", xname)
        cube_like = [x for x in cands if score["cube"][x["id"]] >= dc["cube_min_score"]]
        # ---- 1 slots: Qwen, text only
        t = time.perf_counter()
        self.progress(f"2/4 작업지시 → 항목 ({vname}, 글자만)")
        pr = _slots["template"].replace("{instruction}", instruction)
        (rdir / "qwen" / "slots_prompt.txt").write_text(pr, encoding="utf-8")
        raw = self.vlm_text(pr, _slots["max_new_tokens"], "slots", _slots["prompt_id"], calls)
        (rdir / "qwen" / "slots.txt").write_text(raw, encoding="utf-8")
        sl, err = P.parse_reply(raw, SLOT_SCHEMA)
        rec["slots"] = {"parsed": sl, "error": err}
        tm["slots"] = round(time.perf_counter() - t, 2)
        if not sl:
            st("slots", "FAILED", f"MODEL_RESPONSE_ERROR: {err}", vname)
            return
        st("slots", "OK", f"object '{sl['object']}' ({sl.get('object_words')}) | from: {sl['source_relation']} {sl['source_landmark']} | "
                          f"to: {sl['destination_kind']} ({sl['destination_relation']} {sl['destination_landmark']})", vname)
        # ---- 3 landmarks -> regions (verified facility boxes only)
        refs = {}
        for side in ("source", "destination"):
            lm, relname = sl[f"{side}_landmark"], hc["relation_to_tc"][sl[f"{side}_relation"]]
            fac = hc["landmark_to_facility"].get(lm)
            if lm == "none":
                box, region, how = None, {"type": "WHOLE_IMAGE", "rect": [0, 0, W, H], "rule": "no landmark named"}, "-"
            elif fac and ph["refs"].get(fac, {}).get("box"):
                box, how = ph["refs"][fac]["box"], ph["refs"][fac]["how"]
                region = TC.search_region(relname, box, (W, H), self.tcfg, rel)
            else:
                box, region = None, {"type": "NO_REFERENCE", "rect": None}
                how = f"landmark '{lm}' is not a known facility" if not fac else f"facility '{fac}' not verified in this photo ({ph['refs'].get(fac, {}).get('how')})"
            refs[side] = {"name": lm, "class": fac, "box": box, "how": how, "relation": relname, "region": region}
            st(f"{side}_reference", "FAILED" if region["rect"] is None else "OK", f"{lm} [{relname}] -> {region['type']} ({how})", "Python")
        rec["references"] = refs

        def cand_rows(d, cls):
            return [{"id": c["id"], "bbox_xyxy": c["bbox_xyxy"], "phrases": c.get("phrases"), "siglip_top3": [[f"exemplar {cls}", round(score[cls][c["id"]], 3)]]} for c in d["ranked"]]
        # ---- 4 source: identify in the whole photo, then the relation
        t = time.perf_counter()
        classes = source_classes(sl["object"], hc["graspable_classes"])
        src = {"status": "NOT_FOUND", "ids": [], "boxes": []}
        if not classes:
            src["why"] = f"NOT_GRASPABLE: '{sl['object']}' ({sl.get('object_words')}) is not in the graspable list {hc['graspable_classes']}"
            st("source_decision", "FAILED", src["why"], "Python")
        elif refs["source"]["region"]["rect"] is None:
            src = {"status": "NO_REFERENCE", "ids": [], "boxes": [], "why": refs["source"]["how"]}
            st("source_decision", "FAILED", src["why"], "Python")
        else:
            res = {c: decide(cands, score[c], dc.get(f"{c}_min_score"), dc["min_margin"], dc["nms_iou"], dc["distinct_iou"]) for c in classes}
            sel = [c for c, d in res.items() if d["status"] == "SELECTED"]
            d = res[sel[0]] if len(sel) == 1 else res[classes[0]]
            src = {"status": d["status"], "ids": [], "boxes": [], "why": d["why"], "class": sel[0] if len(sel) == 1 else classes[0],
                   "exemplar": {"s1": d["s1"], "s2": d["s2"], "top_id": d["top"]["id"] if d["top"] else None, "top_box": d["top"]["bbox_xyxy"] if d["top"] else None},
                   "candidates": cand_rows(d, classes[0])}
            if len(sel) > 1:
                src.update(status="AMBIGUOUS", why=f"more than one graspable class found: {sel}")
            elif d["status"] == "SELECTED":
                top = d["top"]
                if refs["source"]["region"]["type"] != "WHOLE_IMAGE" and not TC.in_region(top["bbox_xyxy"], refs["source"]["region"], rel):
                    src.update(status="NOT_FOUND", why=f"NOT_AT_REFERENCE: the {src['class']} ({d['why']}) is not {refs['source']['relation']} '{refs['source']['name']}'")
                else:
                    src.update(ids=[top["id"]], boxes=[top["bbox_xyxy"]])
            st("source_decision", {"SELECTED": "OK", "AMBIGUOUS": "UNCERTAIN"}.get(src["status"], "FAILED"), f"{src['status']} {src['ids']}: {src['why']}", xname + " + Python")
        rec["source"] = src
        tm["source"] = round(time.perf_counter() - t, 2)
        # ---- 5 destination
        t = time.perf_counter()
        reg, kind = refs["destination"]["region"], sl["destination_kind"]
        if reg["rect"] is None:
            dst = {"status": "NO_REFERENCE", "why": refs["destination"]["how"]}
            st("destination_decision", "FAILED", dst["why"], "Python")
        elif kind == "empty_pallet":
            pool = [x for x in cands if reg["type"] == "WHOLE_IMAGE" or TC.in_region(x["bbox_xyxy"], reg, rel)]
            occ = [x["bbox_xyxy"] for x in cube_like] + list(src.get("boxes") or [])
            free = [x for x in pool if not any(centre_in(b, x["bbox_xyxy"]) for b in occ)]
            d = decide(free, score["empty_pallet"], None, dc["min_margin"], dc["nms_iou"], dc["distinct_iou"])
            dst = {"status": d["status"], "why": d["why"], "id": None, "bbox_xyxy": None, "occupied_excluded": len(pool) - len(free),
                   "exemplar": {"s1": d["s1"], "s2": d["s2"], "top_id": d["top"]["id"] if d["top"] else None, "top_box": d["top"]["bbox_xyxy"] if d["top"] else None},
                   "candidates": cand_rows(d, "empty_pallet")}
            if d["status"] == "SELECTED":
                top = d["top"]
                f = self.zone_frac(ph, top["bbox_xyxy"], ck["no_placement_zones"])
                a = self.access(ph, top)
                dst.update(policy_no_placement_frac={top["id"]: f}, accessibility={top["id"]: a}, goal_kind="OBJECT", placement="EXEMPLAR_EMPTY_PALLET",
                           load_state="empty (no cube-like box inside)", kind="EXEMPLAR_TOP1")
                if f is not None and f >= ck["no_placement_min_frac"]:
                    dst.update(status="NOT_FOUND", why=f"POLICY_NO_PLACEMENT: {f:.0%} of the pallet in {ck['no_placement_zones']}", policy_excluded=[top["id"]])
                elif a["accessible"] is False:
                    dst.update(status="NOT_ACCESSIBLE", why=a["why"])
                else:
                    dst.update(id=top["id"], bbox_xyxy=top["bbox_xyxy"], why=d["why"] + (f"; {a['why']}" if a["accessible"] else f"; accessibility {a['why']}"))
                    if a["accessible"] is None:
                        dst["kind"] += "_ACCESSIBILITY_NOT_CHECKED"
            st("destination_decision", {"SELECTED": "OK", "AMBIGUOUS": "UNCERTAIN"}.get(dst["status"], "FAILED"),
               f"{dst['status']} {dst.get('id') or ''}: {dst['why']} ({dst['occupied_excluded']} candidates with a cube-like box inside excluded)", xname + " + Python")
        elif kind == "near_landmark":     # unchanged MMViewer destination step (P3b)
            inc, regions = S.zone_members(ph["dino"], reg["rect"], rel["region_scale_frac"])
            sboxes = src.get("boxes") or []
            inc = sorted([x for x in inc if not any(iou(x["bbox_xyxy"], b) >= 0.5 for b in sboxes)], key=lambda x: (x["bbox_xyxy"][1], x["bbox_xyxy"][0]))
            if inc:
                srcobj = OBJECT_EN.get(sl["object"], sl.get("object_words") or "")
                labels, _ = self.labels_for(srcobj)
                sg = self.siglip(ph, [(x["id"], x["bbox_xyxy"]) for x in inc], labels, calls, "destination")
                dst = self._side(ph, reg, inc, sg, "destination", instruction, rules == "R1", refs["destination"]["name"], srcobj, rdir, calls, stages, vname)
            else:
                dst = {"status": "NOT_FOUND", "why": "no candidate inside the destination region"}
                st("destination_decision", "FAILED", dst["why"], "Python")
        else:
            dst = {"status": "NOT_SUPPORTED", "why": f"destination kind '{kind}' ({sl.get('destination_words')}) is not supported"}
            st("destination_decision", "FAILED", dst["why"], "Python")
        rec["destination"] = dst
        tm["destination"] = round(time.perf_counter() - t, 2)
        # ---- Python: chosen ids exist among the detected candidates with identical boxes
        reg_boxes = {x["id"]: x["bbox_xyxy"] for x in ph["dino"]}
        pairs = list(zip(src.get("ids") or [], src.get("boxes") or [])) + ([(dst["id"], dst["bbox_xyxy"])] if dst.get("id") else [])
        bad = [i for i, b in pairs if i not in reg_boxes or [round(v, 2) for v in reg_boxes[i]] != [round(v, 2) for v in b]]
        st("selection_check", "FAILED" if bad else "OK", f"ID/box mismatch {bad}" if bad else f"{len(pairs)} chosen id(s) exist among the DINO candidates", "Python")


KO = {"SELECTED": "확정", "AMBIGUOUS": "보류(후보 여럿)", "NOT_FOUND": "찾지 못함/보류", "NOT_ACCESSIBLE": "접근 불가", "NO_REFERENCE": "기준 시설 없음",
      "NOT_SUPPORTED": "지원 안 함", "NOT_RUN": "실행 안 됨", "MODEL_RESPONSE_INVALID": "모델 응답 오류", "TOO_MANY_CANDIDATES": "후보 너무 많음"}


def narrative(rec):
    sl = (rec.get("slots") or {}).get("parsed")
    if not sl:
        return ["작업지시를 항목으로 바꾸지 못했습니다(Qwen 응답 형식 오류). A와 B를 정하지 않았습니다."]
    s, d = rec["source"], rec["destination"]
    return [f"해석(Qwen 4B, 글자만): 옮길 것 '{sl['object']}', 출발 기준 {sl['source_landmark']}({sl['source_relation']}), "
            f"놓을 곳 {sl['destination_kind']}({sl['destination_relation']} {sl['destination_landmark']}).",
            f"A: {KO.get(s.get('status'), s.get('status'))} {' '.join(s.get('ids') or [])} - {s.get('why', '')}",
            f"B: {KO.get(d.get('status'), d.get('status'))} {d.get('id') or ''} - {d.get('why', '')}"]
