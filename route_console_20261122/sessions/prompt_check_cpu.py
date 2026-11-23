#!/usr/bin/env python3
"""console_slots_v3 프롬프트를 CPU에서 확인 (GPU는 실행 중인 콘솔이 쓰고 있어서). 같은 모델, greedy. 결과는 이 폴더의 prompt_check_<시각>.json.
  /home/binci/miniconda3/envs/qwen_vl/bin/python -B sessions/prompt_check_cpu.py
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE.parent / "config" / "console.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE.parent))
import engine as EN  # noqa: E402  (parse_slots, task_plan만 씀)
INS = [
    "하역장에서 충전소 옆으로 이동시켜줘",
    "상차구역에서 충전소 옆으로 이동시켜줘",
    "상차구역에서 빨간 큐브를 챙겨 충전소 옆으로 이동시켜줘",
    "보행자 통로에서 하역장으로 가줘",
    "컨베이어에서 소화기 앞까지 이동해",
    "빨간 큐브를 비어 있는 팔레트로 옮겨줘.",
    "충전소로 이동해.",
    "하역장에서 컨베이어로 옮겨줘",
]


def main():
    mid = "Qwen/Qwen3-VL-4B-Instruct"
    proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16, local_files_only=True).eval()
    ic = CFG["interpretation"]
    out = {"prompt_id": ic["prompt_id"], "device": "cpu", "created": datetime.now().isoformat(timespec="seconds"), "runs": []}
    for ins in INS:
        msgs = [{"role": "user", "content": [{"type": "text", "text": ic["template"].replace("{instruction}", ins)}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        t = time.time()
        with torch.inference_mode():
            o = model.generate(**inp, max_new_tokens=ic["max_new_tokens"], do_sample=False)
        raw = proc.decode(o[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        sl, err = EN.parse_slots(raw)
        plan, why = EN.task_plan(sl) if sl else (None, err)
        r = {"instruction": ins, "raw": raw, "slots": sl, "error": err, "plan_start": plan and plan["start_place"], "plan_near": plan and plan["near_place"],
             "plan_dest": plan and (plan["dest_place"] or plan["dest_kind"]), "corrections": plan and plan["corrections"], "hold": None if plan else why,
             "seconds": round(time.time() - t, 1)}
        out["runs"].append(r)
        print(json.dumps({k: r[k] for k in ("instruction", "plan_start", "plan_near", "plan_dest", "corrections", "hold", "seconds")}, ensure_ascii=False), flush=True)
        print("    slots:", sl, flush=True)
    (HERE / f"prompt_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
