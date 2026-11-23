#!/usr/bin/env python3
"""Model inference worker. Runs as a separate process inside the existing `qwen_vl` env.

Protocol: one JSON request per stdin line -> one JSON response per stdout line.
  {"id": 1, "cmd": "qwen_generate", "args": {...}} -> {"id": 1, "ok": true, "result": {...}}
  errors -> {"id": 1, "ok": false, "error": {"type": "...", "message": "..."}}
Commands: ping, load, qwen_generate, dino_detect, unload, shutdown.
Only locally cached weights are used (local_files_only=True). Nothing is downloaded.
Loading code follows EXP110 tools/run_models.py (read-only reference).
Self-contained: stdlib + torch/transformers/Pillow only (no project imports).
"""
import gc
import json
import os
import platform
import sys
import time
import traceback


class Worker:
    def __init__(self):
        import torch
        self.torch = torch
        self.models = {}

    # ------------------------------------------------------------ info
    def ping(self):
        import transformers
        t = self.torch
        return {"python": sys.executable, "python_version": platform.python_version(), "pid": os.getpid(),
                "torch": t.__version__, "torch_cuda": t.version.cuda, "transformers": transformers.__version__,
                "cuda_available": t.cuda.is_available(),
                "gpu": t.cuda.get_device_name(0) if t.cuda.is_available() else None,
                "loaded": {k: v["info"] for k, v in self.models.items()}}

    def _mem(self):
        t = self.torch
        if not t.cuda.is_available():
            return {}
        return {"allocated_gib": round(t.cuda.memory_allocated() / 2**30, 3),
                "peak_allocated_gib": round(t.cuda.max_memory_allocated() / 2**30, 3)}

    # ------------------------------------------------------------ loading
    def load(self, name, model_id, revision=None, dtype="bfloat16"):
        if name in self.models:
            return self.models[name]["info"]
        t0 = time.perf_counter()
        from transformers import AutoProcessor
        kw = {"local_files_only": True}
        if revision:
            kw["revision"] = revision
        if name == "qwen":
            from transformers import Qwen3VLForConditionalGeneration
            proc = AutoProcessor.from_pretrained(model_id, **kw)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id, dtype=getattr(self.torch, dtype), **kw).to("cuda").eval()
        elif name == "dino":
            from transformers import AutoModelForZeroShotObjectDetection
            proc = AutoProcessor.from_pretrained(model_id, **kw)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, **kw).to("cuda").eval()
        else:
            raise ValueError(f"unknown model name {name}")
        info = {"model_id": model_id, "requested_revision": revision,
                "commit_hash": getattr(model.config, "_commit_hash", None),
                "dtype": str(next(model.parameters()).dtype), "load_s": round(time.perf_counter() - t0, 2),
                **self._mem()}
        self.models[name] = {"proc": proc, "model": model, "info": info}
        return info

    def unload(self, names=None):
        for n in list(names or self.models):
            self.models.pop(n, None)
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return {"loaded": list(self.models), **self._mem()}

    # ------------------------------------------------------------ inference
    def qwen_generate(self, prompt, image_path=None, max_new_tokens=256):
        from PIL import Image
        m = self.models["qwen"]
        content = []
        image_size = None
        if image_path:
            img = Image.open(image_path).convert("RGB")
            image_size = list(img.size)
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        inp = m["proc"].apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt").to("cuda")
        self.torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            out = m["model"].generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        dt = time.perf_counter() - t0
        gen = out[0][inp["input_ids"].shape[-1]:]
        text = m["proc"].decode(gen, skip_special_tokens=True)
        return {"text": text, "n_input_tokens": int(inp["input_ids"].shape[-1]), "n_output_tokens": int(gen.shape[-1]),
                "hit_max_new_tokens": int(gen.shape[-1]) >= max_new_tokens, "latency_s": round(dt, 3),
                "image_path": image_path, "image_size": image_size, "max_new_tokens": max_new_tokens,
                "model": m["info"]["model_id"], "commit_hash": m["info"]["commit_hash"], **self._mem()}

    def dino_detect(self, image_path, prompts, box_threshold, text_threshold):
        from PIL import Image
        m = self.models["dino"]
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        self.torch.cuda.reset_peak_memory_stats()
        results = []
        for p in prompts:
            text = p.lower().strip().rstrip(".") + "."
            inputs = m["proc"](images=img, text=text, return_tensors="pt").to("cuda")
            t0 = time.perf_counter()
            with self.torch.inference_mode():
                o = m["model"](**inputs)
            self.torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            r = m["proc"].post_process_grounded_object_detection(
                o, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(h, w)])[0]
            labels = r.get("text_labels", r.get("labels"))
            boxes = [[round(float(v), 2) for v in b] for b in r["boxes"].tolist()]
            results.append({"prompt": text, "boxes_xyxy": boxes, "scores": [round(float(s), 4) for s in r["scores"]],
                            "phrases": [str(l) for l in labels], "latency_s": round(dt, 4)})
        return {"image_path": image_path, "image_size": [w, h], "box_threshold": box_threshold,
                "text_threshold": text_threshold, "results": results, "model": m["info"]["model_id"],
                "commit_hash": m["info"]["commit_hash"], **self._mem()}


def main():
    proto = os.fdopen(os.dup(1), "w", buffering=1)   # protocol channel
    os.dup2(2, 1)                                     # anything printed by libraries -> stderr (log)
    sys.stdout = sys.stderr
    worker = None
    for line in sys.stdin:
        if not line.strip():
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id, cmd, args = req.get("id"), req["cmd"], req.get("args", {})
            if cmd == "shutdown":
                proto.write(json.dumps({"id": req_id, "ok": True, "result": {"bye": True}}) + "\n")
                break
            if worker is None:
                worker = Worker()
            handler = {"ping": worker.ping, "load": worker.load, "unload": worker.unload,
                       "qwen_generate": worker.qwen_generate, "dino_detect": worker.dino_detect}.get(cmd)
            if handler is None:
                raise ValueError(f"unknown command {cmd!r}")
            resp = {"id": req_id, "ok": True, "result": handler(**args)}
        except Exception as e:  # reported to the caller, never swallowed
            etype = type(e).__name__
            if etype == "OutOfMemoryError" and worker is not None:
                worker.unload()
                etype = "CUDA_OOM"
            traceback.print_exc(file=sys.stderr)
            resp = {"id": req_id, "ok": False, "error": {"type": etype, "message": str(e)[:2000]}}
        proto.write(json.dumps(resp) + "\n")
        proto.flush()


if __name__ == "__main__":
    main()
