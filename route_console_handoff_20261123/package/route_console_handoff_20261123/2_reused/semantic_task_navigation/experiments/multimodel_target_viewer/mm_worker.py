#!/usr/bin/env python3
"""Model worker of the multi-model viewer: SigLIP2 (image/text embeddings) and a Qwen3-VL VLM in 4-bit NF4.
Runs as a separate process in the existing qwen_vl_q4 env (python -I; the env is used as is, nothing installed).
Protocol (same as the v0.2 worker): one JSON request per stdin line -> one JSON response per stdout line.
  {"id": 1, "cmd": "...", "args": {...}} -> {"id": 1, "ok": true, "result": {...}} | {"id": 1, "ok": false, "error": {...}}
Commands: ping, load, unload, siglip_device, siglip_images, siglip_texts, vlm_generate, shutdown.
Only locally cached weights are loaded (local_files_only=True).
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
        self.torch, self.models = torch, {}

    def ping(self):
        import transformers
        t = self.torch
        return {"python": sys.executable, "python_version": platform.python_version(), "pid": os.getpid(), "torch": t.__version__, "torch_cuda": t.version.cuda,
                "transformers": transformers.__version__, "cuda_available": t.cuda.is_available(), "gpu": t.cuda.get_device_name(0) if t.cuda.is_available() else None,
                "loaded": {k: v["info"] for k, v in self.models.items()}}

    def _mem(self):
        t = self.torch
        return {"allocated_gib": round(t.cuda.memory_allocated() / 2**30, 3), "peak_allocated_gib": round(t.cuda.max_memory_allocated() / 2**30, 3)} if t.cuda.is_available() else {}

    def load(self, name, model_id, kind, dtype="float16", quantization=None, device="cuda", revision=None):
        if name in self.models:
            return self.models[name]["info"]
        t0 = time.perf_counter()
        from transformers import AutoProcessor
        kw = {"local_files_only": True}
        if revision:
            kw["revision"] = revision
        if kind == "siglip":
            from transformers import AutoModel
            proc = AutoProcessor.from_pretrained(model_id, **kw)
            model = AutoModel.from_pretrained(model_id, dtype=getattr(self.torch, dtype), **kw).to(device).eval()
        elif kind == "qwen3vl":
            from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
            proc = AutoProcessor.from_pretrained(model_id, **kw)
            if quantization == "nf4":
                q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self.torch.bfloat16)
                model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, quantization_config=q, device_map={"": 0}, **kw).eval()
            else:
                model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=self.torch.bfloat16, **kw).to("cuda").eval()
        else:
            raise ValueError(f"unknown model kind {kind}")
        info = {"model_id": model_id, "revision": revision, "kind": kind, "quantization": quantization, "commit_hash": getattr(model.config, "_commit_hash", None),
                "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device), "load_s": round(time.perf_counter() - t0, 2), **self._mem()}
        self.models[name] = {"proc": proc, "model": model, "info": info}
        return info

    def unload(self, names=None):
        for n in list(names or self.models):
            self.models.pop(n, None)
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return {"loaded": list(self.models), **self._mem()}

    def siglip_device(self, device):
        m = self.models["siglip"]
        m["model"].to(device)
        m["info"]["device"] = device
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return {"device": device, **self._mem()}

    @staticmethod
    def _features(out):
        return out if hasattr(out, "norm") else out.pooler_output     # transformers versions differ: tensor or model output

    def siglip_images(self, paths, batch=16):
        from PIL import Image
        m, t = self.models["siglip"], self.torch
        dev = next(m["model"].parameters()).device
        embs, t0 = [], time.perf_counter()
        for i in range(0, len(paths), batch):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            inp = m["proc"](images=ims, return_tensors="pt").to(dev)
            with t.inference_mode():
                f = self._features(m["model"].get_image_features(pixel_values=inp["pixel_values"].to(m["model"].dtype)))
            embs += (f / f.norm(dim=-1, keepdim=True)).float().cpu().tolist()
        return {"embeddings": embs, "n": len(paths), "seconds": round(time.perf_counter() - t0, 2), "device": str(dev),
                "model": m["info"]["model_id"], "commit_hash": m["info"]["commit_hash"], **self._mem()}

    def siglip_texts(self, texts):
        m, t = self.models["siglip"], self.torch
        dev = next(m["model"].parameters()).device
        inp = m["proc"](text=texts, padding="max_length", max_length=64, return_tensors="pt").to(dev)
        with t.inference_mode():
            f = self._features(m["model"].get_text_features(**inp))
        return {"embeddings": (f / f.norm(dim=-1, keepdim=True)).float().cpu().tolist(), "logit_scale": float(m["model"].logit_scale.exp()),
                "logit_bias": float(m["model"].logit_bias), "model": m["info"]["model_id"], "commit_hash": m["info"]["commit_hash"]}

    def vlm_generate(self, prompt, image_path=None, max_new_tokens=256):
        """Same message format and greedy decoding as the v0.2 worker's qwen_generate."""
        from PIL import Image
        m = self.models["vlm"]
        content, image_size = [], None
        if image_path:
            img = Image.open(image_path).convert("RGB")
            image_size = list(img.size)
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        inp = m["proc"].apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt").to("cuda")
        self.torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            out = m["model"].generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inp["input_ids"].shape[-1]:]
        return {"text": m["proc"].decode(gen, skip_special_tokens=True), "n_input_tokens": int(inp["input_ids"].shape[-1]), "n_output_tokens": int(gen.shape[-1]),
                "hit_max_new_tokens": int(gen.shape[-1]) >= max_new_tokens, "latency_s": round(time.perf_counter() - t0, 3), "image_path": image_path,
                "image_size": image_size, "max_new_tokens": max_new_tokens, "model": m["info"]["model_id"], "commit_hash": m["info"]["commit_hash"],
                "quantization": m["info"]["quantization"], **self._mem()}


def main():
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
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
            handler = {"ping": worker.ping, "load": worker.load, "unload": worker.unload, "siglip_device": worker.siglip_device,
                       "siglip_images": worker.siglip_images, "siglip_texts": worker.siglip_texts, "vlm_generate": worker.vlm_generate}.get(cmd)
            if handler is None:
                raise ValueError(f"unknown command {cmd!r}")
            resp = {"id": req_id, "ok": True, "result": handler(**args)}
        except Exception as e:  # reported to the caller, never swallowed
            etype = type(e).__name__
            if etype == "OutOfMemoryError":
                etype = "CUDA_OOM"
            traceback.print_exc(file=sys.stderr)
            resp = {"id": req_id, "ok": False, "error": {"type": etype, "message": str(e)[:2000]}}
        proto.write(json.dumps(resp) + "\n")
        proto.flush()


if __name__ == "__main__":
    main()
