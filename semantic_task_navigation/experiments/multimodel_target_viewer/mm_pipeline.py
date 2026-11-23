#!/usr/bin/env python3
"""Multi-model target viewer back end: Grounding DINO -> SigLIP2 -> VLM -> Python checks -> A / B, Korean plan text, overlay.

  Grounding DINO  v0.2 worker (qwen_vl env)            candidates and boxes (fresh pass per photo; new nouns per instruction)
  SigLIP2         mm_worker.py (qwen_vl_q4 env)         object-type comparison of every candidate (object crop + context crop) and of the
                                                       reference facilities proposed by DINO (relative similarity among config labels)
  VLM             selectable: Qwen3-VL-8B NF4 (mm_worker) or Qwen3-VL-4B bf16 (v0.2 worker): interpretation (task_context prompt) and
                  role judgement with the SigLIP2 hints
  Python          reply mapping / degenerate-reply checks, relation regions, duplicates, no-placement zone, accessibility, decisions

Imported read-only (nothing modified): instruction_target_viewer.viewer_pipeline (Viewer: geometry, access, floor_cm, detection merge,
ROI detection, handoff, photo list), task_context_2d_navigation (interpretation prompt, rules R1, relation regions, decide_source,
decide_destination), placement_target_grounding (qwen_call, dino_call, detections, parse_reply), contextual_target_selection (numbered
images, zone members), v0.2 src (model worker client, make_roi, check_alignment).
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "instruction_target_viewer"))
import viewer_pipeline as VP  # noqa: E402  (also puts the v0.2 src and task_context on sys.path)
from semantic_task_nav.config import sha256_file, sha256_text  # noqa: E402
from semantic_task_nav.geometry.coordinate_frames import box_area, iou  # noqa: E402
from semantic_task_nav.navigation.map_builder import check_alignment  # noqa: E402
from semantic_task_nav.perception.model_client import ModelWorkerClient, ModelWorkerError  # noqa: E402

TC, P, S, PT = VP.TC, VP.P, VP.S, VP.PT
WORKER = HERE / "mm_worker.py"
PROBLEM = ("FAILED", "UNCERTAIN")


# ================================================================================ pure functions (tested offline)
def ref_class(name, rc):
    """VLM reference name -> facility. 'robot' only by exact name (so that 'robot arm' is not the mobile robot)."""
    n = (name or "").lower().strip()
    if n in [k.lower() for k in rc["robot_exact"]]:
        return "robot"
    return TC.ref_class(name, {k: v for k, v in rc.items() if k != "robot_exact"})


def as_label(noun):
    n = (noun or "").strip().lower()
    return n if n.split()[:1] in (["a"], ["an"], ["the"]) else (("an " if n[:1] in "aeiou" else "a ") + n)


def square_crop(box, img_wh, pad_ratio, min_px):
    """Square window around the box (side = longer side * (1 + 2 pad), at least min_px), shifted inside the image."""
    w, h = img_wh
    side = min(max(max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * pad_ratio), min_px), w, h)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0, y0 = min(max(0.0, cx - side / 2), w - side), min(max(0.0, cy - side / 2), h - side)
    return [int(round(x0)), int(round(y0)), int(round(x0 + side)), int(round(y0 + side))]


def siglip_rank(img_embs, txt_embs, scale, bias, labels):
    """Image (V x D) and text (L x D) unit embeddings -> per view: sigmoid (SigLIP native, independent per label) and the softmax over
    THESE labels (relative ranking only, not a class probability); combined = mean of the views' relative scores."""
    logits = np.asarray(img_embs) @ np.asarray(txt_embs).T * scale + bias
    rel = np.exp(logits - logits.max(axis=1, keepdims=True))
    rel /= rel.sum(axis=1, keepdims=True)
    comb = rel.mean(axis=0)
    order = np.argsort(-comb)
    return {"combined": [(labels[i], round(float(comb[i]), 3)) for i in order],
            "views": [{"top3": [(labels[i], round(float(r[i]), 3)) for i in np.argsort(-r)[:3]], "sigmoid_top": round(float(1 / (1 + np.exp(-lg.max()))), 4)}
                      for r, lg in zip(rel, logits)]}


def check_reply(reply, n, dmin, incomplete_fatal):
    """Python checks of one VLM role reply against its numbered image (local numbers 1..n). -> (usable, problems)."""
    if reply is None:
        return False, ["INVALID_RESPONSE: not valid JSON for the schema"]
    ids = [c.get("id") for c in reply["candidates"]]
    nums = [S._num(i, n) for i in ids]
    probs = []
    bad = [i for i, x in zip(ids, nums) if x is None]
    if bad:
        probs.append(f"MAPPING_ERROR: ids outside 1..{n}: {bad}")
    dup = sorted({x for x in nums if x and nums.count(x) > 1})
    if dup:
        probs.append(f"MAPPING_ERROR: numbers rated more than once {dup}")
    miss = sorted(set(range(1, n + 1)) - set(nums))
    if miss:
        probs.append(f"MAPPING_INCOMPLETE: numbers not rated {miss}")
    if len(reply["candidates"]) >= dmin:
        sig = {(str((c.get("appearance") or [""])[0]).strip().lower(), c.get("role") or c.get("placement_role"), c.get("load_state")) for c in reply["candidates"]}
        if len(sig) == 1:
            probs.append(f"DEGENERATE_RESPONSE: all {len(reply['candidates'])} candidates rated identically {sorted(sig)[0]}")
    fatal = [p for p in probs if not p.startswith("MAPPING_INCOMPLETE") or incomplete_fatal]
    return not fatal, probs


def mentioned_facilities(text, words):
    """Facilities named in the instruction text ('로봇 팔' / 'robot arm' are not the mobile robot)."""
    t = re.sub(r"로봇\s*팔|robot\s*arm", " ", text.lower())
    return [f for f, ws in words.items() if any(w.lower() in t for w in ws)]


def all_duplicates(boxes, thr):
    return len(boxes) > 1 and all(iou(a, b) >= thr for i, a in enumerate(boxes) for b in boxes[i + 1:])


def first_failure(stages):
    return next((s["name"] for s in stages if s["status"] in PROBLEM), None)


# ================================================================================ worker client for mm_worker.py
class MMClient(ModelWorkerClient):
    """The v0.2 client (call / kill / close reused unchanged) started on mm_worker.py with the qwen_vl_q4 python."""

    def __init__(self, wcfg, log_dir):
        super().__init__({"worker": wcfg}, log_dir)

    def start(self):
        if self.alive:
            return self.env_info
        python = self.cfg["worker"]["python"]
        if not Path(python).exists():
            raise ModelWorkerError("WORKER_PYTHON_MISSING", python)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"mm_worker_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
                    "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        self._log_file = open(self.log_path, "a", encoding="utf-8")
        self.proc = subprocess.Popen([python, "-I", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log_file,
                                     text=True, bufsize=1, env=env, cwd=str(WORKER.parent))
        self._lines = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc, self._lines), daemon=True).start()
        self.loaded = {}
        self.env_info = self.call("ping", timeout=self.cfg["worker"]["startup_timeout_s"])
        return self.env_info


# ================================================================================ session
class MMViewer(VP.Viewer):
    def __init__(self, session_dir=None, progress=print, backend=None):
        self.mc = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
        session_dir = session_dir or HERE / self.mc["results_dir"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_session")
        super().__init__(session_dir=session_dir, progress=progress)
        self.backend = backend or next(iter(self.mc["vlm_backends"]))
        self.mm, self.emb, self.temb, self.tscale = None, {}, {}, None

    # ---------------------------------------------------------------- photos are identified by content (sha256), not by file name
    def _known(self):
        if not hasattr(self, "_known_sha"):
            f = yaml.safe_load((EXP / "cube_candidate_funnel/config.yaml").read_text(encoding="utf-8"))
            self._known_sha = {i["sha256"]: i["id"] for i in f["images"]}
            self._sha_cache = {}
        return self._known_sha

    def _sha(self, path):
        self._known()
        key = (str(path), Path(path).stat().st_mtime_ns)
        if key not in self._sha_cache:
            self._sha_cache[key] = sha256_file(path)
        return self._sha_cache[key]

    def pid_for(self, path):
        """Session photo id (e.g. U25070552) from the content hash of the earlier session records; unknown photos -> file stem."""
        return self._known().get(self._sha(path), Path(path).stem)

    def list_photos(self):
        return [{"photo_id": self.pid_for(p), "file": p.name, "path": str(p)} for p in sorted(Path(self.mc["photo_dir"]).glob("*.png"))]

    def e1_path(self):
        """The E1 reference photo of the floor transform, found by its recorded sha256 in the photo folder (file names may change)."""
        return next((Path(p["path"]) for p in self.list_photos() if self._sha(p["path"]) == self.rcfg["image_sha256"]), None)

    def geometry(self):
        """Same as instruction_target_viewer Viewer.geometry, but the E1 image is located by hash instead of the configured file name."""
        if self.geo is None:
            e1p = self.e1_path()
            if e1p is None:
                self.geo = {"ok": False, "why": f"E1 reference photo (sha256 {self.rcfg['image_sha256'][:12]}) not found in {self.mc['photo_dir']}"}
                return self.geo
            self.progress("바닥 변환·DRAFT 지도 준비 (Python, 모델 없음)")
            e1 = cv2.imread(str(e1p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            ctx = VP.RI.build_context(self.rcfg, self.gcfg, self.v02, e1)
            if "fr" not in ctx:
                self.geo = {"ok": False, "ctx": ctx, "why": "E1 floor transform not accepted"}
                return self.geo
            fr, (L, Wd) = ctx["fr"], map(float, self.tc["route"]["robot_cm"])
            m = ctx["ft"]["footprint_margin_cm"]
            navs = {h: VP.FN.RectNavMap(ctx["spec_bev"], fr.bev_size, ctx["pcfg"], VP.FN.footprint_dims_cm(L, Wd, m, h), fr.s) for h in (0, 90)}
            self.geo = {"ok": True, "ctx": ctx, "navs": navs, "zr": VP.SN.CM.Raster(fr.bev_size, fr.s, fr.pad),
                        "drivable": [z["polygon"] for z in ctx["spec_cm"]["traversable"]], "L": L, "W": Wd, "e1_path": str(e1p)}
        return self.geo

    # ---------------------------------------------------------------- models and GPU memory (one large model at a time)
    def mmc(self):
        if self.mm is None or not self.mm.alive:
            self.progress("새 모델 작업 프로세스 시작 (qwen_vl_q4 환경: SigLIP2, Qwen3-VL-8B)")
            self.mm = MMClient(self.mc["worker"], self.session / "worker_logs")
            self.mm.start()
        return self.mm

    def _unload(self, client, name):
        if client is not None and client.alive and name in client.loaded:
            client.call("unload", names=[name])
            client.loaded.pop(name)

    def _siglip(self, device):
        m, sc = self.mmc(), self.mc["siglip"]
        if "siglip" not in m.loaded:
            self.progress("SigLIP2 로딩")
            m.loaded["siglip"] = m.call("load", name="siglip", model_id=sc["model_id"], kind="siglip", dtype=sc["dtype"], device="cpu")
        if device == "cuda":
            self._unload(self.client, "qwen")
            self._unload(m, "vlm") if self.mc["vlm_backends"][self.backend]["where"] != "mm_worker" else None
        if m.loaded["siglip"].get("device") != device:
            m.call("siglip_device", device=device)
            m.loaded["siglip"]["device"] = device
        return m

    def stop_models(self):
        super().stop_models()
        if self.mm is not None:
            self.mm.kill()

    def close(self):
        if self.mm is not None:
            self.mm.close()
        super().close()

    def vlm(self, prompt, image_path, max_new_tokens, stage, prompt_id, log):
        b = self.mc["vlm_backends"][self.backend]
        if self.mm is not None and self.mm.alive and "siglip" in self.mm.loaded and self.mm.loaded["siglip"].get("device") == "cuda":
            self._siglip("cpu")
        if b["where"] == "v02_worker":
            self._unload(self.mm, "vlm")
            return P.qwen_call(self.model(), prompt, image_path, max_new_tokens, stage, prompt_id, log)
        m = self.mmc()
        self._unload(self.client, "qwen")
        if "vlm" not in m.loaded:
            self.progress(f"{b['label']} 로딩 (처음 한 번, 수십 초)")
            m.loaded["vlm"] = m.call("load", name="vlm", model_id=b["model_id"], kind="qwen3vl", quantization=b.get("quantization"), revision=b.get("revision"))
        tw = time.perf_counter()
        r = m.call("vlm_generate", prompt=prompt, image_path=str(image_path), max_new_tokens=max_new_tokens)
        log.append({"stage": stage, "model": r["model"], "commit_hash": r["commit_hash"], "quantization": r["quantization"], "prompt_id": prompt_id,
                    "prompt_sha256": sha256_text(prompt), "image_path": str(image_path), "image_sha256": sha256_file(image_path), "image_size": r["image_size"],
                    "n_input_tokens": r["n_input_tokens"], "n_output_tokens": r["n_output_tokens"], "hit_max_new_tokens": r["hit_max_new_tokens"],
                    "latency_s": r["latency_s"], "wall_s": round(time.perf_counter() - tw, 2), "peak_vram_gib": r.get("peak_allocated_gib"), "raw_response": r["text"]})
        return r["text"]

    # ---------------------------------------------------------------- SigLIP2
    def _crops(self, ph, key, box):
        d = self.session / "photos" / f"{ph['photo_id']}_siglip"
        d.mkdir(exist_ok=True)
        out = {}
        for view, c in self.mc["siglip"]["crops"].items():
            r = square_crop(box, ph["size"], c["pad_ratio"], c["min_px"])
            p = d / f"{key}_{view}.png"
            if not p.exists():
                cv2.imwrite(str(p), ph["img"][r[1]:r[3], r[0]:r[2]])
            out[view] = {"path": str(p), "rect": r}
        return out

    def _texts(self, labels, log):
        new = [lb for lb in labels if lb not in self.temb]
        if new:
            m = self._siglip(self.mm.loaded["siglip"]["device"] if self.mm and "siglip" in self.mm.loaded else "cpu")
            r = m.call("siglip_texts", texts=[self.mc["siglip"]["text_template"].format(label=lb) for lb in new])
            self.temb.update(dict(zip(new, r["embeddings"])))
            self.tscale = (r["logit_scale"], r["logit_bias"])
            log.append({"stage": "siglip_texts", "model": r["model"], "commit_hash": r["commit_hash"], "n_texts": len(new)})
        return np.array([self.temb[lb] for lb in labels])

    def siglip(self, ph, items, labels, log, tag):
        """items: [(key, box)] -> key -> ranking. Image embeddings are computed once per photo and box (cached)."""
        todo = [(k, b) for k, b in items if (ph["photo_id"], k) not in self.emb]
        if todo:
            self.progress(f"SigLIP2 객체 종류 비교 ({len(todo)}개 후보 × 2 이미지)")
            crops = {k: self._crops(ph, k, b) for k, b in todo}
            paths = [crops[k][v]["path"] for k, _ in todo for v in self.mc["siglip"]["crops"]]
            m = self._siglip("cuda")
            r = m.call("siglip_images", paths=paths, batch=self.mc["siglip"]["batch"])
            log.append({"stage": f"siglip_images_{tag}", "model": r["model"], "commit_hash": r["commit_hash"], "n_images": r["n"], "seconds": r["seconds"],
                        "device": r["device"], "peak_vram_gib": r.get("peak_allocated_gib")})
            nv = len(self.mc["siglip"]["crops"])
            for i, (k, _) in enumerate(todo):
                self.emb[(ph["photo_id"], k)] = {"embs": r["embeddings"][i * nv:(i + 1) * nv], "crops": crops[k]}
        T = self._texts(labels, log)
        out = {}
        for k, _ in items:
            e = self.emb[(ph["photo_id"], k)]
            rk = siglip_rank(e["embs"], T, self.tscale[0], self.tscale[1], labels)
            rk["crops"] = e["crops"]
            out[k] = rk
        return out

    def labels_for(self, srcobj):
        base = list(self.mc["siglip"]["labels"])
        lab = as_label(srcobj) if srcobj and srcobj.strip().lower().split()[-1:] and srcobj.strip().lower().split()[-1] not in self.mc["siglip"]["generic_nouns"] else None
        if lab and lab not in base:
            base.append(lab)
        return base, lab

    # ---------------------------------------------------------------- per photo: fresh DINO, SigLIP2 reference check, SigLIP2 of all candidates
    def photo(self, path):
        pid = self.pid_for(path)
        if pid in self.photos:
            return self.photos[pid]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        H, W = img.shape[:2]
        geo = self.geometry()
        ph = {"photo_id": pid, "path": str(path), "sha256": self._sha(path), "size": [W, H], "img": img, "calls": [],
              "alignment": check_alignment({"reference_image": geo["e1_path"], "alignment_check": {"max_shift_px": 3.0, "min_response": 0.3}}, img, lambda x: x)
              if geo["ok"] else {"status": "NOT_AVAILABLE", "reason": geo["why"]}}
        c, det, gd = self.model(), self.tc["detection"], self.mcfg["grounding_dino"]["class_prompts"]
        vocab = list(dict.fromkeys(self.tc["reuse"]["vocabulary"] + self.mc["detection"]["extra_phrases"]))
        self.progress(f"{pid}: Grounding DINO 객체 탐지 (사진당 한 번)")
        raw_refs = P.dino_call(c, ph["path"], gd["conveyor"] + gd["truck"] + gd["mobile_robot"], det["thresholds"], f"dino_refs_{pid}", ph["calls"])
        raw_full = P.dino_call(c, ph["path"], vocab, det["thresholds"], f"dino_full_{pid}", ph["calls"])
        dets = P.detections(raw_refs, "refs", [0, 0, W, H], 1.0, det["max_area_frac"])
        labels, sc = list(self.mc["siglip"]["labels"]), self.mc["siglip"]
        win = self.fpcfg["robot"]["search_window_xyxy"]
        ph["refs"] = {}
        for cls, key, extra in (("conveyor", "conveyor", None), ("truck", "truck", None), ("robot", "mobile_robot", win)):
            prompts = [p.lower() for p in gd[key]]
            # duplicates of one box from different prompts are merged first (they used up the check budget in the first E1 run)
            cs = P.merge([d for d in dets if d["status"] == "KEPT" and d["prompt"] in prompts and (extra is None or P.centre_in(d["bbox_xyxy"], extra))],
                         det["merge_iou"])[:self.mcfg["grounding_dino"]["verify_top_k"][key]]
            rk = self.siglip(ph, [(f"ref_{cls}_{k}", d["bbox_xyxy"]) for k, d in enumerate(cs)], labels, ph["calls"], f"refs_{pid}")
            recs, acc = [], None
            for k, d in enumerate(cs):
                topk = [lb for lb, _ in rk[f"ref_{cls}_{k}"]["combined"][:sc["reference_check_top_k"]]]
                ok = sc["reference_labels"][cls] in topk
                recs.append({"bbox_xyxy": d["bbox_xyxy"], "dino_score": d["score"], "siglip_top3": rk[f"ref_{cls}_{k}"]["combined"][:3], "verified": ok})
                if ok:
                    acc = d
                    break
            ph["refs"][cls] = {"box": acc["bbox_xyxy"] if acc else None, "checked": recs,
                               "how": f"DINO + SigLIP2 ('{sc['reference_labels'][cls]}' in top {sc['reference_check_top_k']})" if acc else
                               f"NOT VERIFIED (the facility label was never in SigLIP2 top {sc['reference_check_top_k']})"}
            if cls == "robot" and acc:
                b = acc["bbox_xyxy"]
                ph["refs"]["robot"]["start_img"] = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]      # task_context rule (box centre)
        ph["dino"], ph["phrases_done"] = [], []
        self._add(ph, P.detections(raw_full, "FULL", [0, 0, W, H], 1.0, det["max_area_frac"]) + self._roi_dets(ph, vocab, ph["calls"], "v"))
        ph["phrases_done"] = list(vocab)
        ph["detection_how"] = "fresh Grounding DINO this session (task_context vocabulary + red cube / traffic cone / mobile robot; full image + verified facility ROIs), IoU merge 0.5"
        rr = json.loads(((TC.HERE / self.tc["reuse"]["retention_run"]).resolve() / "result.json").read_text(encoding="utf-8"))
        cube = next((p["candidates"] for p in rr["photos"] if p["photo_id"] == pid), None)
        ph["cube"] = None if cube is None or ph["sha256"] != self._session_sha(pid) else \
            [{"id": f"cand_r{c['rank']}", "bbox_xyxy": c["bbox_xyxy"], "state": c["state"], "source_types": c["source_types"], "score": c["score"]} for c in cube]
        ph["cube_how"] = "stored red-cube candidates of this photo (candidate_retention_policy: HSV colour + DINO)" if ph["cube"] is not None else "none stored"
        self.siglip(ph, [(x["id"], x["bbox_xyxy"]) for x in ph["dino"] + (ph["cube"] or [])], labels, ph["calls"], f"all_{pid}")
        self.photos[pid] = ph
        self._save_photo(ph)
        return ph

    # ---------------------------------------------------------------- policy helpers (Python)
    def zone_frac(self, ph, box, zone_ids):
        geo = self.geometry()
        if not geo["ok"] or ph["alignment"]["status"] != "ALIGNED":
            return None
        poly = geo["ctx"]["fr"].img_to_cm(PT.box_poly(box)).tolist()
        mp = geo["zr"].mask([poly])
        mz = geo["zr"].mask([z["polygon"] for z in geo["ctx"]["spec_cm"]["no_go"] if z["id"] in zone_ids])
        return round(float((mp & mz).sum() / max(1, mp.sum())), 3)

    # ---------------------------------------------------------------- one instruction
    def run(self, path, instruction, rules=None, backend=None):
        rules, self.backend = rules or self.mc["default_rules"], backend or self.backend
        t0, calls, stages = time.time(), [], []
        ph = self.photo(path)
        self.n += 1
        rid = f"run_{self.n:03d}_{ph['photo_id']}"
        rdir = self.session / "runs" / rid
        for d in ("inputs", "qwen"):
            (rdir / d).mkdir(parents=True)
        rec = {"run_id": rid, "created": datetime.now().isoformat(timespec="seconds"), "instruction": instruction, "rules": rules,
               "vlm_backend": {"key": self.backend, **self.mc["vlm_backends"][self.backend]},
               "photo": {k: ph[k] for k in ("photo_id", "path", "sha256", "size")}, "alignment_with_E1": ph["alignment"].get("status"),
               "photo_detection": {"how": ph["detection_how"], "file": str(self.session / "photos" / f"{ph['photo_id']}.json"), "cube": ph["cube_how"]},
               "frames": VP.FRAMES, "config": {"viewer": str(HERE / "config.yaml"), "task_context": str(self.tc_path)},
               "model_called": True, "source": {"status": "NOT_RUN"}, "destination": {"status": "NOT_RUN"}}
        try:
            self._infer(ph, instruction, rules, rdir, rec, calls, stages)
        except ModelWorkerError as e:
            stages.append({"name": "model_worker", "status": "FAILED", "summary": f"{e.etype}: {e.message}", "model": "-"})
        rec["stages"], rec["first_failure"] = stages, first_failure(stages)
        rec["wall_s"] = round(time.time() - t0, 1)
        (rdir / "model_calls.json").write_text(json.dumps(calls, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        rec["models_used"] = models_used(calls, ph)
        rec["handoff"] = self._handoff(ph, rec)
        rec["narrative"] = narrative(rec)
        cv2.imwrite(str(rdir / "overlay.png"), draw(ph["img"], rec))
        cv2.imwrite(str(rdir / "overlay_candidates.png"), draw(ph["img"], rec, show_candidates=True))
        rec["files"] = {"overlay": str(rdir / "overlay.png"), "overlay_candidates": str(rdir / "overlay_candidates.png"), "run_json": str(rdir / "run.json")}
        (rdir / "run.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        self.runs.append(rec)
        self._save_photo(ph)
        self._write_session()
        return rec

    def _infer(self, ph, instruction, rules, rdir, rec, calls, stages):
        (W, H), tc, rel, ck = ph["size"], self.tc, self.rel, self.mc["checks"]
        r1, vname = rules == "R1", self.mc["vlm_backends"][self.backend]["label"]
        dino_name = self.mcfg["grounding_dino"]["model_id"]
        st = lambda name, status, summary, model, **kw: stages.append({"name": name, "status": status, "summary": summary, "model": model, **kw})   # noqa: E731
        # ---- 1 VLM interpretation (task_context prompt, unchanged)
        self.progress(f"1/6 작업지시 해석 ({vname})")
        pr = TC.interp_prompt(tc, {"instruction": instruction}, r1)
        (rdir / "qwen" / "interpretation_prompt.txt").write_text(pr, encoding="utf-8")
        raw = self.vlm(pr, ph["path"], tc["interpretation"]["max_new_tokens"], "interpretation", tc["interpretation"]["prompt_id"], calls)
        (rdir / "qwen" / "interpretation.txt").write_text(raw, encoding="utf-8")
        it, err = P.parse_reply(raw, TC.INTERP_SCHEMA)
        rec["interpretation"] = {"parsed": it, "error": err}
        if not it:
            st("interpretation", "FAILED", f"MODEL_RESPONSE_ERROR: {err}", vname)
            return
        for k in ("source_reference", "destination_reference"):
            it[k]["bbox_image_px"] = P.qwen_box_to_image(it[k].get("bbox_2d"), [0, 0, W, H])
        srcobj = it["source_object"].get("name_en") or ""
        st("interpretation", "OK", f"move '{srcobj}' | from: {it['source_relation']} '{it['source_reference'].get('name_en')}' | to: {it['destination_relation']} "
                                   f"'{it['destination_reference'].get('name_en')}' ({it.get('destination_meaning')})", vname)
        # ---- 2 DINO for nouns not yet searched on this photo
        nouns = list(dict.fromkeys(P.norm_phrase(n, 10) for n in (srcobj, it["source_reference"].get("name_en"), it["destination_reference"].get("name_en")) if n))
        new = [n for n in nouns if n not in ph["phrases_done"]]
        if new:
            self.progress(f"2/6 새 물체 이름 탐지 (Grounding DINO): {', '.join(new)}")
            raw_full = P.dino_call(self.model(), ph["path"], new, tc["detection"]["thresholds"], f"dino_full_{rec['run_id']}", calls)
            added = self._add(ph, P.detections(raw_full, f"FULL_{rec['run_id']}", [0, 0, W, H], 1.0, tc["detection"]["max_area_frac"])
                              + self._roi_dets(ph, new, calls, rec["run_id"]))
            ph["phrases_done"] = sorted(set(ph["phrases_done"]) | set(new))
            st("detection", "OK", f"DINO called for new phrases {new}: {added} new candidates ({len(ph['dino'])} on this photo)", dino_name)
        else:
            st("detection", "REUSED", f"phrases {nouns} already searched; {len(ph['dino'])} DINO candidates of this photo reused (DINO ran when the photo was first used)", dino_name)
        # ---- 3 references -> regions (task_context rules; no reference named -> whole image)
        refs = {}
        for side in ("source", "destination"):
            rname = it[f"{side}_reference"].get("name_en")
            rcls = ref_class(rname, self.mc["reference_classes"])
            if rcls and ph["refs"].get(rcls, {}).get("box"):
                rbox, rhow = ph["refs"][rcls]["box"], f"facility '{rcls}': {ph['refs'][rcls]['how']}"
            else:
                verified = [{"object_id": k, "bbox_xyxy": v["box"]} for k, v in ph["refs"].items() if v.get("box") and k != "robot"]
                rbox, rhow = S.snap_anchor(it[f"{side}_reference"].get("bbox_image_px"), verified, rel["snap_iou"])
            relname = it[f"{side}_relation"]
            region = {"type": "WHOLE_IMAGE", "rect": [0, 0, W, H], "rule": "no reference named"} if (not rname and rbox is None) else \
                TC.search_region(relname, rbox, (W, H), self.tcfg, rel)
            refs[side] = {"name": rname, "class": rcls, "box": rbox, "how": rhow, "relation": relname, "region": region}
            st(f"{side}_reference", "FAILED" if region["rect"] is None else "OK",
               f"'{rname}' named but not found in the photo ({rhow})" if region["rect"] is None else f"'{rname}' [{relname}] -> {region['type']} ({rhow})", "Python")
        rec["references"] = refs
        missing = [f for f in mentioned_facilities(instruction, self.mc["checks"]["facility_words"]) if f not in (refs["source"]["class"], refs["destination"]["class"])]
        rec["interpretation_check"] = {"mentioned_but_not_used": missing}
        st("interpretation_check", "UNCERTAIN" if missing else "OK",
           f"the instruction names {missing} but neither interpreted reference is that facility (source '{refs['source']['name']}', destination "
           f"'{refs['destination']['name']}')" if missing else "every facility named in the instruction is one of the interpreted references", "Python")
        labels, src_label = self.labels_for(srcobj)
        rec["siglip_labels"] = labels
        # ---- 4 source
        src = {"status": "NO_REFERENCE", "ids": [], "why": f"source reference '{refs['source']['name']}' not found in the photo"}
        reg = refs["source"]["region"]
        if reg["rect"] is not None:
            if "cube" in srcobj.lower() and ph["cube"] is not None:
                pool, how = ph["cube"], "stored red-cube candidates"
            else:
                rb = refs["source"]["box"]
                pool = [d for d in ph["dino"] if box_area(d["bbox_xyxy"]) < rel["region_scale_frac"] * box_area(reg["rect"]) and (rb is None or iou(d["bbox_xyxy"], rb) < 0.5)]
                how = "DINO candidates in the region"
                if reg["type"] == "WHOLE_IMAGE":
                    head = srcobj.lower().split()[-1:]
                    pool = [d for d in pool if any(w in " ".join(d["phrases"]) for w in head)]
                    how = f"DINO candidates whose phrase contains '{' '.join(head)}'"
            inside = sorted([d for d in pool if TC.in_region(d["bbox_xyxy"], reg, rel)], key=lambda d: (d["bbox_xyxy"][1], d["bbox_xyxy"][0]))
            st("source_candidates", "OK" if inside else "FAILED", f"{len(inside)} of {len(pool)} {how} inside the region" if inside else
               f"TRANSPORT_OBJECT_NOT_FOUND: none of {len(pool)} {how} inside the region",
               "stored HSV colour + DINO candidates (candidate_retention_policy)" if how == "stored red-cube candidates" else dino_name)
            if inside:
                sg = self.siglip(ph, [(x["id"], x["bbox_xyxy"]) for x in inside], labels, calls, "source")
                st("source_siglip", "OK", f"{len(inside)} candidates compared with {len(labels)} labels (object crop + context crop)", self.mc["siglip"]["model_id"])
                src = self._side(ph, reg, inside, sg, "source", instruction, r1, refs["source"]["name"], srcobj, rdir, calls, stages, vname)
                if src["status"] == "SELECTED":
                    k = self.mc["siglip"]["type_check_top_k"]
                    topk = sg[src["ids"][0]]["combined"][:k]
                    shown = ", ".join(f"'{lb}' {sc_:.2f}" for lb, sc_ in topk)
                    if src_label is None:
                        src["type_check"] = {"status": "SKIPPED", "why": f"'{srcobj}' is a generic noun"}
                    elif src_label in [lb for lb, _ in topk]:
                        src["type_check"] = {"status": "AGREE", "why": f"'{src_label}' within SigLIP2 top {k} ({shown})"}
                    else:
                        src["type_check"] = {"status": "DISAGREE", "why": f"'{src_label}' not within SigLIP2 top {k} ({shown})"}
                        src["status"], src["why"] = "TYPE_UNCERTAIN", f"VLM chose {src['ids'][0]} but {src['type_check']['why']}"
                    st("source_type_check", {"AGREE": "OK", "SKIPPED": "OK", "DISAGREE": "UNCERTAIN"}[src["type_check"]["status"]],
                       f"{src['type_check']['status']}: {src['type_check']['why']}", self.mc["siglip"]["model_id"] + " + Python")
            else:
                src = {"status": "NOT_FOUND", "ids": [], "why": "TRANSPORT_OBJECT_NOT_FOUND: no candidate inside the search region"}
        rec["source"] = src
        # ---- 5 destination
        dst = {"status": "NO_REFERENCE", "why": f"destination reference '{refs['destination']['name']}' not found in the photo"}
        reg = refs["destination"]["region"]
        if reg["rect"] is not None:
            inc, regions = S.zone_members(ph["dino"], reg["rect"], rel["region_scale_frac"])
            sboxes = [x["bbox_xyxy"] for x in (ph["cube"] or []) + ph["dino"] if x["id"] in src.get("ids", [])]
            inc = sorted([x for x in inc if not any(iou(x["bbox_xyxy"], b) >= 0.5 for b in sboxes)], key=lambda x: (x["bbox_xyxy"][1], x["bbox_xyxy"][0]))
            st("destination_candidates", "OK" if inc else "FAILED", f"{len(inc)} DINO candidates inside the region (object to move excluded; {len(regions)} region-scale boxes not numbered)"
               if inc else "NO_DESTINATION_CANDIDATE in the region", dino_name)
            if inc:
                sg = self.siglip(ph, [(x["id"], x["bbox_xyxy"]) for x in inc], labels, calls, "destination")
                st("destination_siglip", "OK", f"{len(inc)} candidates compared with {len(labels)} labels", self.mc["siglip"]["model_id"])
                dst = self._side(ph, reg, inc, sg, "destination", instruction, r1, refs["destination"]["name"], srcobj, rdir, calls, stages, vname)
            else:
                dst = {"status": "NOT_FOUND", "why": "no candidate inside the destination region"}
        rec["destination"] = dst
        # ---- Python: the chosen ids exist among the detected candidates and carry the same boxes
        reg_boxes = {x["id"]: x["bbox_xyxy"] for x in (ph["cube"] or []) + ph["dino"]}
        pairs = [(i, b) for i, b in zip(src.get("ids") or [], src.get("boxes") or [])] + ([(dst["id"], dst["bbox_xyxy"])] if dst.get("id") else [])
        bad = [i for i, b in pairs if i not in reg_boxes or [round(v, 2) for v in reg_boxes[i]] != [round(v, 2) for v in b]]
        st("selection_check", "FAILED" if bad else "OK", f"ID/box mismatch {bad}" if bad else
           f"{len(pairs)} chosen id(s) exist among the detected candidates with identical boxes", "Python")

    def _side(self, ph, reg, cands, sg, side, instruction, r1, reference, srcobj, rdir, calls, stages, vname):
        """Numbered image(s) + SigLIP2 hints -> VLM roles -> Python checks -> decision for one side."""
        tc, ck, n_all = self.tc, self.mc["checks"], len(cands)
        st = lambda name, status, summary, model: stages.append({"name": name, "status": status, "summary": summary, "model": model})   # noqa: E731
        groups = VP.chunks(cands, self.rel["max_numbered"])
        base = {"candidates": [{"id": x["id"], "bbox_xyxy": x["bbox_xyxy"], "phrases": x.get("phrases"), "siglip_top3": sg[x["id"]]["combined"][:3],
                                "siglip_views": sg[x["id"]]["views"]} for x in cands], "numbering": {k: x["id"] for k, x in enumerate(cands, 1)}}
        if len(groups) > self.mc["max_role_calls_per_side"]:
            st(f"{side}_vlm_roles", "FAILED", f"TOO_MANY_CANDIDATES: {n_all} candidates need {len(groups)} VLM calls (limit {self.mc['max_role_calls_per_side']})", vname)
            return {**base, "status": "TOO_MANY_CANDIDATES", "ids": [], "why": f"{n_all} candidates in the region"}
        key = "source_template" if side == "source" else "destination_template"
        replies, imgs, problems, usable = [], [], [], True
        self.progress(f"{'3' if side == 'source' else '5'}/6 {'옮길 물체' if side == 'source' else '놓을 곳'} 판단 ({vname}, 후보 {n_all}개)")
        for g_i, g in enumerate(groups, 1):
            off = sum(map(len, groups[:g_i - 1]))
            vr = S.view_rect(reg["rect"], [x["bbox_xyxy"] for x in g], ph["size"])
            p = rdir / "inputs" / f"{side}_{g_i}.png"
            S.numbered(ph["img"], vr, S.zoom_scale(vr, self.tcfg), g, p)
            hints = "\n".join(f"{k}번: " + " · ".join(f"{lb} {s:.2f}" for lb, s in sg[x["id"]]["combined"][:3]) for k, x in enumerate(g, 1))
            pr = (self.mc["roles"][key].replace("{instruction}", instruction).replace("{rules_block}", (tc["interpretation"]["rules_R1"] + "\n") if r1 else "")
                  .replace("{reference}", reference or "none").replace("{source_object}", srcobj or "").replace("{siglip_hints}", hints).replace("{n}", str(len(g))))
            (rdir / "qwen" / f"{side}_{g_i}_prompt.txt").write_text(pr, encoding="utf-8")
            raw = self.vlm(pr, p, self.mc["roles"]["max_new_tokens"], f"{side}_{g_i}", self.mc["roles"][f"{side}_prompt_id"], calls)
            (rdir / "qwen" / f"{side}_{g_i}.txt").write_text(raw, encoding="utf-8")
            obj, err = P.parse_reply(raw, S.SRC_SCHEMA if side == "source" else S.DEST_SCHEMA)
            ok, pb = check_reply(obj, len(g), ck["degenerate_min_candidates"], side == "source")
            usable &= ok
            problems += [f"image {g_i}: {x}" for x in pb] + ([f"image {g_i}: {err}"] if err else [])
            replies.append(obj)
            imgs.append({"path": str(p), "view_rect": vr, "numbers": [off + 1, off + len(g)], "usable": ok, "problems": pb})
        reply = VP.merge_replies(replies, [len(g) for g in groups], "object_to_move_id" if side == "source" else "best_destination_id") if usable else None
        rated = {int(r["id"]): r for r in (reply or {}).get("candidates", []) if S._num(r.get("id"), n_all)}
        out = {**base, "images": imgs, "problems": problems, "ratings": {cands[k - 1]["id"]: r for k, r in rated.items()}, "reply_reason": (reply or {}).get("reason")}
        st(f"{side}_vlm_roles", "OK" if usable and not problems else "UNCERTAIN" if usable else "FAILED",
           ("reply usable" if usable else "MODEL_RESPONSE_ERROR (reply rejected by the Python checks)") + (f"; {' | '.join(problems)}" if problems else ""), vname)
        if not usable:
            return {**out, "status": "MODEL_RESPONSE_INVALID", "ids": [], "why": " | ".join(problems)}
        byid = {x["id"]: x for x in cands}
        if side == "source":
            d = TC.decide_source(reply, n_all)
            ids = [cands[k - 1]["id"] for k in d["numbers"]]
            if d["status"] == "AMBIGUOUS" and all_duplicates([byid[i]["bbox_xyxy"] for i in ids], ck["duplicate_iou"]):
                keep = max(ids, key=lambda i: byid[i].get("score") or 0)
                d = {**d, "status": "SELECTED", "why": f"tied {ids} are one object (pairwise IoU >= {ck['duplicate_iou']}); kept {keep} (highest detector score)"}
                ids = [keep]
            assert all(i in byid for i in ids)           # selected ids exist among the numbered candidates
            st("source_decision", "OK" if d["status"] == "SELECTED" else "UNCERTAIN" if d["status"] == "AMBIGUOUS" else "FAILED",
               f"{d['status']} {ids}: {d['why']}" + (" (MULTIPLE_SOURCE_CANDIDATES)" if d["status"] == "AMBIGUOUS" else ""), "Python")
            return {**out, **d, "ids": ids, "boxes": [byid[i]["bbox_xyxy"] for i in ids]}
        # destination: no-placement zone (Python policy), accessibility (Python), task_context decision
        todo = [k for k, r in rated.items() if r["placement_role"] in ("suitable", "possible")]
        policy, judged = {}, json.loads(json.dumps(reply))
        for k in todo:
            f = self.zone_frac(ph, cands[k - 1]["bbox_xyxy"], ck["no_placement_zones"])
            policy[cands[k - 1]["id"]] = f
            if f is not None and f >= ck["no_placement_min_frac"]:
                next(c for c in judged["candidates"] if S._num(c.get("id"), n_all) == k)["placement_role"] = "unsuitable"
        excluded = [i for i, f in policy.items() if f is not None and f >= ck["no_placement_min_frac"]]
        st("destination_policy", "OK" if not excluded else "UNCERTAIN",
           f"no-placement zone {ck['no_placement_zones']}: " + (f"excluded {excluded} (POLICY_NO_PLACEMENT)" if excluded else f"none of {len(todo)} suitable/possible candidates inside"), "Python")
        todo = [k for k in todo if cands[k - 1]["id"] not in excluded]
        self.progress(f"6/6 접근 가능성 계산 (Python, 후보 {len(todo)}개)")
        acc = {k: self.access(ph, cands[k - 1]) for k in todo}
        checked = all(a["accessible"] is not None for a in acc.values())
        st("destination_access", "OK" if checked else "UNCERTAIN", (f"{sum(bool(a['accessible']) for a in acc.values())} of {len(acc)} candidates accessible from the robot start"
                                                                     if checked else next(a["why"] for a in acc.values() if a["accessible"] is None)), "Python")
        dd = TC.decide_destination(judged, {k: (a["accessible"] if checked else True) for k, a in acc.items()}, n_all)
        tied = [cands[k - 1]["id"] for k in VP.tied_destinations({k: r for k, r in {int(c["id"]): c for c in judged["candidates"] if S._num(c.get("id"), n_all)}.items()}, acc)] \
            if dd["status"] == "AMBIGUOUS" else []
        if dd["status"] == "AMBIGUOUS" and all_duplicates([byid[i]["bbox_xyxy"] for i in tied], ck["duplicate_iou"]):
            keep = max(tied, key=lambda i: byid[i].get("score") or 0)
            dd = {**dd, "status": "SELECTED", "number": next(k for k, x in enumerate(cands, 1) if x["id"] == keep), "kind": "DUPLICATE_TIE_RESOLVED",
                  "why": dd["why"] + f"; tied {tied} are one object, kept {keep}"}
        if not checked and dd["kind"]:
            dd["kind"] += "_ACCESSIBILITY_NOT_CHECKED"
        res = {**out, **dd, "id": cands[dd["number"] - 1]["id"] if dd["number"] else None, "bbox_xyxy": cands[dd["number"] - 1]["bbox_xyxy"] if dd["number"] else None,
               "accessibility": {cands[k - 1]["id"]: a for k, a in acc.items()}, "policy_no_placement_frac": policy, "policy_excluded": excluded, "tied_ids": tied if dd["status"] == "AMBIGUOUS" else []}
        if dd["number"]:
            r = rated.get(dd["number"]) or {}
            res["goal_kind"], res["goal_kind_why"] = VP.goal_kind(cands[dd["number"] - 1], r.get("appearance"), ph["size"], self.mc["goal_region"])
            res["placement"], res["load_state"] = VP.placement_status(res["goal_kind"], r), r.get("load_state")
        label = {"AMBIGUOUS": "MULTIPLE_DESTINATION_CANDIDATES", "NOT_FOUND": "NO_SUITABLE_DESTINATION", "NOT_ACCESSIBLE": "NOT_ACCESSIBLE"}.get(dd["status"], "")
        st("destination_decision", "OK" if dd["status"] == "SELECTED" else "UNCERTAIN" if dd["status"] == "AMBIGUOUS" else "FAILED",
           f"{dd['status']} {res['id'] or ''} {dd['kind'] or ''} {label}: {dd['why']}", "Python")
        return res


# ================================================================================ record helpers
def models_used(calls, ph):
    """Which model actually ran, in this instruction run and when the photo was prepared (from the call logs)."""
    def summ(cs):
        out = {}
        for c in cs:
            k = "Grounding DINO" if c["stage"].startswith("dino") else "SigLIP2" if c["stage"].startswith("siglip") else "VLM"
            if c["stage"].startswith("verify"):
                k = "VLM"
            m = f"{c['model']}" + (f" ({c['quantization']})" if c.get("quantization") else "")
            out.setdefault(k, {}).setdefault(m, 0)
            out[k][m] += 1
        return out
    return {"this_run": summ(calls), "photo_preparation": summ(ph["calls"])}


REL_KO = {"next_to": "옆", "near": "근처", "on": "위", "from": "에서 나온", "none": "(관계 없음)"}
ST_KO = {"SELECTED": "선정", "TYPE_UNCERTAIN": "종류 불확실(미확정)", "AMBIGUOUS": "후보 여럿(미확정)", "NOT_FOUND": "찾지 못함", "NOT_ACCESSIBLE": "접근 불가",
         "NO_REFERENCE": "기준 시설 못 찾음", "MODEL_RESPONSE_INVALID": "모델 응답 오류", "TOO_MANY_CANDIDATES": "후보 너무 많음", "NOT_RUN": "실행 안 됨"}


def _nm(side, cid):
    r = (side.get("ratings") or {}).get(cid) or {}
    ap = r.get("appearance") or []
    return str(ap[0]) if ap else "?"


def _sg(side, cid):
    c = next((x for x in side.get("candidates") or [] if x["id"] == cid), None)
    return f"{c['siglip_top3'][0][0]} {c['siglip_top3'][0][1]:.2f}" if c and c.get("siglip_top3") else "?"


def narrative(rec):
    """Korean description of what was found / chosen / planned. Only what the record shows; unchecked things are said to be unchecked."""
    it = (rec.get("interpretation") or {}).get("parsed")
    s, d, refs = rec["source"], rec["destination"], rec.get("references") or {}
    vl = rec["vlm_backend"]["label"]
    out = []
    if not it:
        return [f"작업지시를 해석하지 못했습니다 ({vl} 응답 오류). A와 B를 정하지 않았습니다."]
    so = it["source_object"].get("name_en")
    out.append(f"작업지시 해석({vl}): '{so}'을(를) '{refs['source']['name']}' {REL_KO.get(refs['source']['relation'], refs['source']['relation'])}에서 "
               f"'{it.get('destination_meaning')}'(기준: '{refs['destination']['name']}' {REL_KO.get(refs['destination']['relation'], '')})(으)로 옮기는 작업입니다.")
    n_s = len(s.get("candidates") or [])
    if s.get("status") == "SELECTED":
        i = s["ids"][0]
        out.append(f"출발 탐색 영역에서 후보 {n_s}개를 비교했고, {i}(VLM: {_nm(s, i)}, SigLIP2 최상위: {_sg(s, i)})를 운반 대상 A로 선정했습니다.")
    elif s.get("status") == "TYPE_UNCERTAIN":
        out.append(f"VLM은 {s['ids'][0]}({_nm(s, s['ids'][0])})를 운반 대상으로 골랐지만, SigLIP2 비교 결과({_sg(s, s['ids'][0])})와 맞지 않아 A를 확정하지 않았습니다.")
    elif s.get("status") == "AMBIGUOUS":
        out.append(f"운반 대상 후보 {len(s['ids'])}개({', '.join(s['ids'])})가 똑같이 해당해 A를 확정하지 않았습니다.")
    else:
        out.append(f"운반 대상 A: {ST_KO.get(s.get('status'), s.get('status'))}. {s.get('why', '')}")
    n_d = len(d.get("candidates") or [])
    if d.get("status") == "SELECTED":
        i = d["id"]
        acc = (d.get("accessibility") or {}).get(i, {})
        out.append(f"목적지 탐색 영역에서 후보 {n_d}개를 비교했고, {i}(VLM: {_nm(d, i)}, 적재 상태: {d.get('load_state')}, SigLIP2 최상위: {_sg(d, i)})를 목적지 B로 선정했습니다"
                   f" ({d.get('goal_kind')}, {d.get('placement')}). 로봇 정차 위치에서 접근 가능으로 계산됐습니다(주행 구역까지 {acc.get('gap_to_drivable_cm')} cm).")
    elif d.get("status") == "AMBIGUOUS":
        out.append(f"목적지 후보 {len(d.get('tied_ids') or [])}개({', '.join(d.get('tied_ids') or [])})가 같은 조건을 만족해 B를 확정하지 않았습니다.")
    elif d.get("status") == "NOT_ACCESSIBLE":
        out.append(f"VLM이 적합하다고 본 목적지 후보는 있었지만 모두 로봇이 접근할 수 없어 B를 확정하지 않았습니다. ({d.get('why')})")
    else:
        out.append(f"목적지 B: {ST_KO.get(d.get('status'), d.get('status'))}. {d.get('why', '')}")
    miss = (rec.get("interpretation_check") or {}).get("mentioned_but_not_used")
    if miss:
        out.append(f"확인 필요: 지시에 나온 시설 {miss}이(가) 해석된 기준 시설에 없습니다. 아래 결과는 해석이 바뀐 기준으로 찾은 것입니다.")
    if s.get("status") == "SELECTED" and d.get("status") == "SELECTED":
        out.append(("(확인 필요) " if miss else "") + f"계획: (1) A({s['ids'][0]})로 가서 물체를 집는다 → (2) B({d['id']}) 근처 정차 위치로 이동한다 → (3) B에 내려놓는다. "
                   "사진의 화살표는 작업 방향이며 충돌을 피하는 주행 경로가 아닙니다. 배치면 크기와 그리퍼 도달은 확인하지 않았습니다.")
    elif s.get("status") == "SELECTED":
        out.append("계획 미확정: A는 정했지만 B가 정해지지 않아 이동·배치 단계를 만들지 않았습니다.")
    else:
        out.append("계획 미확정: A가 확정되지 않아 작업 계획을 만들지 않았습니다.")
    return out


# ================================================================================ drawing
COL = {"A": (40, 40, 235), "B": (235, 110, 20), "regA": (80, 170, 255), "regB": (255, 210, 90), "ref": (235, 235, 235), "cand": (170, 170, 170), "off": (150, 150, 150)}


def draw(img, rec, show_candidates=False):
    im = img.copy()
    refs, s, d = rec.get("references") or {}, rec.get("source") or {}, rec.get("destination") or {}
    for side, col in (("source", COL["regA"]), ("destination", COL["regB"])):
        r = refs.get(side)
        if not r:
            continue
        if r["region"].get("rect") and r["region"]["type"] != "WHOLE_IMAGE":
            VP._dashed(im, r["region"]["rect"], col, 2, 14)
            VP._label(im, f"{'A' if side == 'source' else 'B'} search: {r['relation']} {r['name']}", r["region"]["rect"][0] + 4, r["region"]["rect"][1] + 22, col, 0.5, 1)
        if r.get("box"):
            b = [int(round(v)) for v in r["box"]]
            cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), COL["ref"], 1)
            VP._label(im, f"ref: {r['name']}", b[0], b[3] + 18, COL["ref"], 0.45, 1)
    if show_candidates:
        for x, key in ((s, "A"), (d, "B")):
            for k, c in enumerate(x.get("candidates") or [], 1):
                b = [int(round(v)) for v in c["bbox_xyxy"]]
                rt = (x.get("ratings") or {}).get(c["id"]) or {}
                cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), COL["cand"], 1)
                VP._label(im, f"{key}{k} {(rt.get('role') or rt.get('placement_role') or '-')[:4]} | {c['siglip_top3'][0][0][:18]}", b[0], b[1] - 2, COL["cand"], 0.36, 1)
    if d.get("status") in ("NOT_ACCESSIBLE",) or show_candidates:
        for cid, a in (d.get("accessibility") or {}).items():
            if a.get("accessible") is False:
                c = next(x for x in d["candidates"] if x["id"] == cid)
                VP._dashed(im, c["bbox_xyxy"], COL["off"], 2, 6)
                VP._label(im, f"B? not accessible ({a.get('gap_to_drivable_cm')} cm) {cid}", c["bbox_xyxy"][0], c["bbox_xyxy"][3] + 18, COL["off"], 0.45, 1)
    for cid in d.get("policy_excluded") or []:
        c = next(x for x in d["candidates"] if x["id"] == cid)
        VP._dashed(im, c["bbox_xyxy"], COL["off"], 2, 4)
        VP._label(im, f"B? excluded: no-placement zone {cid}", c["bbox_xyxy"][0], c["bbox_xyxy"][3] + 18, COL["off"], 0.45, 1)
    ca = cb = None
    if s.get("status") == "SELECTED":
        b = [int(round(v)) for v in s["boxes"][0]]
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), COL["A"], 3)
        VP._label(im, f"A {s['ids'][0]}: {_nm(s, s['ids'][0])} (SigLIP2: {_sg(s, s['ids'][0])})", b[0], b[1] - 6, COL["A"])
        ca = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
    elif s.get("status") in ("AMBIGUOUS", "TYPE_UNCERTAIN"):
        for i, (cid, bx) in enumerate(zip(s["ids"], s["boxes"]), 1):
            VP._dashed(im, bx, COL["A"], 2, 6)
            what = "type uncertain" if s["status"] == "TYPE_UNCERTAIN" else f"tie {i}/{len(s['ids'])}"
            VP._label(im, f"A? {what} {cid}", bx[0], bx[1] - 6, COL["A"], 0.5, 1)
    if d.get("status") == "SELECTED":
        b = [int(round(v)) for v in d["bbox_xyxy"]]
        if d.get("goal_kind") == "GOAL_REGION":
            ov = im.copy()
            cv2.rectangle(ov, tuple(b[:2]), tuple(b[2:]), COL["B"], -1)
            im = cv2.addWeighted(ov, 0.25, im, 0.75, 0)
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), COL["B"], 3)
        VP._label(im, f"B {d['id']}: {_nm(d, d['id'])} | load: {d.get('load_state')} (SigLIP2: {_sg(d, d['id'])})", b[0], b[1] - 30, COL["B"])
        VP._label(im, f"{d.get('goal_kind')} | {d.get('placement')} | {d.get('kind')}", b[0], b[1] - 6, COL["B"], 0.48, 1)
        cb = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        g = (d.get("accessibility") or {}).get(d["id"], {}).get("approach_goal_image_px")
        if g:
            gp = (int(round(g[0])), int(round(g[1])))
            cv2.drawMarker(im, gp, COL["B"], cv2.MARKER_TILTED_CROSS, 18, 2)
            VP._label(im, "robot stop at B (approach pose)", gp[0] + 10, gp[1] + 20, COL["B"], 0.42, 1)
    elif d.get("status") == "AMBIGUOUS":
        for i, cid in enumerate(d.get("tied_ids") or [], 1):
            c = next(x for x in d["candidates"] if x["id"] == cid)
            VP._dashed(im, c["bbox_xyxy"], COL["B"], 2, 6)
            VP._label(im, f"B? {i}/{len(d['tied_ids'])} {cid}", c["bbox_xyxy"][0], c["bbox_xyxy"][1] - 6, COL["B"], 0.45, 1)
    if ca:
        VP._label(im, "1 PICK", ca[0] - 30, ca[1] + 40, (255, 255, 255), 0.6, 2)
    if ca and cb:
        cv2.arrowedLine(im, ca, cb, (255, 255, 255), 5, cv2.LINE_AA, tipLength=0.03)
        cv2.arrowedLine(im, ca, cb, (60, 220, 60), 2, cv2.LINE_AA, tipLength=0.03)
        mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
        VP._label(im, "task direction A->B (not a driving path)", mid[0] - 150, mid[1] - 8, (60, 220, 60), 0.5, 1)
        VP._label(im, "2 MOVE  3 PLACE", cb[0] - 60, cb[1] + 40, (255, 255, 255), 0.6, 2)
    banner = f"{rec['run_id']} | {rec['vlm_backend']['key']} | A: {s.get('status')} | B: {d.get('status')} | first failure: {rec.get('first_failure') or 'none'}"
    VP._label(im, banner, 8, 30, (255, 255, 255), 0.6, 2)
    return im
