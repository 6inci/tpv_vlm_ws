#!/usr/bin/env python3
"""Feasibility probes for the task-flow plan (2026-11-20). Read-only on every existing asset.

Runs inside the existing `qwen_vl` (4B bf16) or `qwen_vl_q4` (8B nf4) env. Local cache only.
Writes ONLY under /home/binci/vlm_ws/task_flow_plan_20261120/sim/.

Probes (all coordinates: undistorted_px of the 1920x1080 photos; model boxes are 0-1000 normalized
to the image actually given to the model and are converted back here):
  full_v1    whole photo, native resolution, "list loose cargo items"          (baseline)
  roi_v1     6 fixed 480x270 windows per photo, x4 Lanczos zoom, generic wording
  roi_v2     same windows, wording that describes what the site pallet looks like
  landmarks  whole photo, locate truck / loading bay / conveyor / charger / walkway / mobile robot
  zones_A/B  whole photo, "where may the robot drive" without / with 5S floor-marking knowledge
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path("/home/binci/vlm_ws/task_flow_plan_20261120")
PHOTO_DIR = Path("/home/binci/Pictures/Calibration")
PHOTOS = {1: "1calibration.png", 2: "2calibration.png", 3: "3calibration.png", 4: "4calibration.png",
          5: "5calibration.png", 6: "6calibration.png", 7: "7calibration.png",
          8: "undistorted_26007129092710.png"}

# Fixed windows (x1, y1, x2, y2), 480x270 each. Same camera pose in all 8 photos (STEP 4-A: ALIGNED).
ROIS = {"TRUCK": (1380, 130, 1860, 400), "TOPMID": (1000, 140, 1480, 410), "TL": (380, 150, 860, 420),
        "CONV": (250, 290, 730, 560), "BL": (380, 740, 860, 1010), "BR": (1400, 740, 1880, 1010)}
ZOOM = 4

ITEM_SCHEMA = ('Return JSON only, no prose:\n{"items": [{"type": "cube_on_pallet" | "loose_cube" | "empty_pallet" | '
               '"other_small_item", "color": <main colour>, "bbox_2d": [x1, y1, x2, y2], '
               '"description": <short phrase>}]}\n'
               'bbox_2d are integers normalized to 0-1000 relative to this image. If there is no such item, '
               'return {"items": []}.')
P_ITEMS_V1 = ("This is a top-down camera view of a miniature warehouse floor. A small mobile robot with a front "
              "gripper must pick up one small loose cargo item.\n"
              "List every small loose cargo item lying on the floor that such a robot could carry (for example a "
              "small cube or block, possibly sitting on a small pallet) and every small empty pallet.\n"
              "Do NOT list traffic cones, fixed machines, robot arms, shelves, vehicles, drums, floor markings or "
              "large stacks of goods.\n" + ITEM_SCHEMA)
P_ITEMS_V2 = ("This is a top-down camera view of a miniature warehouse floor. A small mobile robot with a front "
              "gripper must pick up one small loose cargo item.\n"
              "Site knowledge: in this site a small pallet is a square grey slatted plate, about as wide as the base "
              "of a traffic cone; seen from above it looks like a small grey grate lying on the black floor. "
              "The cargo is a small red cube that sits on top of such a pallet.\n"
              "List every small pallet that carries a red cube (type cube_on_pallet) and every small pallet that "
              "carries nothing (type empty_pallet).\n"
              "Do NOT list traffic cones, fixed machines, robot arms, shelves, vehicles, drums, floor markings, "
              "large pallets or large stacks of goods.\n" + ITEM_SCHEMA)
# v3 (written AFTER seeing the 4B roi_v2 result on these same 8 photos = development tuning, not evaluation):
# adds a size hint that Python can compute from the known pallet size and the zoom, the pallet colour, and "once".
P_ITEMS_V3 = ("This is a top-down camera view of a miniature warehouse floor. A small mobile robot with a front "
              "gripper must pick up one small loose cargo item.\n"
              "Site knowledge: in this site a small pallet is a square GREY slatted plate lying flat on the black "
              "floor; seen from above it looks like a small grey grate. In this image a small pallet is about one "
              "tenth of the image width (bbox about 100 units wide and 150-200 units high). The cargo is a small red "
              "cube that sits on top of such a grey pallet.\n"
              "List every small grey pallet. Use type cube_on_pallet only if a red cube is clearly visible on it, "
              "otherwise type empty_pallet. Report each pallet exactly once.\n"
              "Do NOT list traffic cones, fixed machines, robot arms, shelves, racks, vehicles, drums, posts, signs, "
              "floor markings, objects that are not grey plates, large pallets or large stacks of goods.\n" + ITEM_SCHEMA)
P_LANDMARKS = ("This is a top-down camera view of a miniature logistics work floor.\n"
               "Locate each of the following if it is visible: (1) the truck, (2) the truck loading bay = the marked "
               "parking space where the truck stands, (3) the conveyor belt production line, (4) the charging "
               "station, (5) the pedestrian walkway, (6) the mobile robot = the wheeled ground vehicle that drives "
               "on the floor (not a fixed robot arm).\n"
               'Return JSON only, no prose:\n{"objects": [{"label": "truck" | "loading_bay" | "conveyor" | '
               '"charging_station" | "walkway" | "mobile_robot", "bbox_2d": [x1, y1, x2, y2], '
               '"visual_evidence": <short phrase>}]}\n'
               "bbox_2d are integers normalized to 0-1000 relative to this image. Omit a label that is not visible.")
P_ZONE_TAIL = ('Return JSON only, no prose:\n{"drive_area": {"bbox_2d": [x1, y1, x2, y2], "boundary_marking": '
               '<what marks its boundary>, "reason": <one sentence>},\n "keep_out": [{"name": <short name>, '
               '"bbox_2d": [x1, y1, x2, y2], "reason": <short phrase>}]}\n'
               "bbox_2d are integers normalized to 0-1000 relative to this image.")
P_ZONES_A = ("This is a top-down camera view of a miniature logistics work floor. A wheeled mobile robot works here.\n"
             "Which floor area is the robot allowed to drive in, and which areas must it keep out of?\n" + P_ZONE_TAIL)
P_ZONES_B = ("This is a top-down camera view of a miniature logistics work floor. A wheeled mobile robot works here.\n"
             "General knowledge (5S workplace floor marking): yellow lines mark aisles and vehicle traffic lanes; "
             "white lines mark equipment and work cells; green marks pedestrian or safe areas; striped tape marks "
             "hazard or keep-clear borders. Vehicles stay inside their marked lane and do not enter marked "
             "equipment, storage or pedestrian areas.\n"
             "Which floor area is the robot allowed to drive in, and which areas must it keep out of?\n" + P_ZONE_TAIL)
PROMPTS = {"items_v1": P_ITEMS_V1, "items_v2": P_ITEMS_V2, "items_v3": P_ITEMS_V3, "landmarks": P_LANDMARKS,
           "zones_A": P_ZONES_A, "zones_B": P_ZONES_B}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load_model(variant):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if variant == "4b":
        mid = "Qwen/Qwen3-VL-4B-Instruct"
        proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16,
                                                                local_files_only=True).to("cuda").eval()
    else:
        from transformers import BitsAndBytesConfig
        mid = "Qwen/Qwen3-VL-8B-Instruct"
        kw = {"local_files_only": True, "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"}
        proc = AutoProcessor.from_pretrained(mid, **kw)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = Qwen3VLForConditionalGeneration.from_pretrained(mid, quantization_config=q, device_map={"": 0},
                                                                **kw).eval()
    return torch, proc, model, mid


def generate(torch, proc, model, img, prompt, max_new_tokens):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inp["input_ids"].shape[-1]:]
    return {"text": proc.decode(gen, skip_special_tokens=True), "n_in": int(inp["input_ids"].shape[-1]),
            "n_out": int(gen.shape[-1]), "hit_max": int(gen.shape[-1]) >= max_new_tokens,
            "latency_s": round(time.perf_counter() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["4b", "8b_nf4"], default="4b")
    ap.add_argument("--probes", default="landmarks,full_v1,roi_v1,roi_v2,zones_A,zones_B")
    ap.add_argument("--photos", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    assert ROOT in out.resolve().parents, "output must stay inside the plan folder"
    out.mkdir(parents=True, exist_ok=False)
    crops = ROOT / "sim" / "crops"
    crops.mkdir(exist_ok=True)
    photos = [int(x) for x in a.photos.split(",")]
    torch, proc, model, mid = load_model(a.model)
    manifest = {"model": mid, "variant": a.model, "zoom": ZOOM, "rois": ROIS, "probes": a.probes.split(","),
                "prompts_sha256": {k: sha(v.encode()) for k, v in PROMPTS.items()},
                "photos": {p: sha((PHOTO_DIR / PHOTOS[p]).read_bytes()) for p in photos},
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "do_sample": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "prompts.json").write_text(json.dumps(PROMPTS, indent=1, ensure_ascii=False))
    log = open(out / "calls.jsonl", "w")

    def call(probe, photo, roi, img, frame, prompt_key, mnt):
        r = generate(torch, proc, model, img, PROMPTS[prompt_key], mnt)
        r.update({"probe": probe, "photo": photo, "roi": roi, "frame_px": frame, "image_size": list(img.size),
                  "prompt": prompt_key})
        log.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.flush()
        print(probe, photo, roi, r["latency_s"], "s", r["n_in"], "->", r["n_out"], flush=True)

    for probe in a.probes.split(","):
        for p in photos:
            full = Image.open(PHOTO_DIR / PHOTOS[p]).convert("RGB")
            if probe == "landmarks":
                call(probe, p, None, full, [0, 0, 1920, 1080], "landmarks", 512)
            elif probe == "full_v1":
                call(probe, p, None, full, [0, 0, 1920, 1080], "items_v1", 600)
            elif probe in ("zones_A", "zones_B"):
                if p in (1, 5, 8):
                    call(probe, p, None, full, [0, 0, 1920, 1080], probe, 600)
            elif probe in ("roi_v1", "roi_v2", "roi_v3"):
                for name, box in ROIS.items():
                    cp = crops / f"photo{p}_{name}_x{ZOOM}.png"
                    if not cp.exists():
                        c = full.crop(box)
                        c.resize((c.width * ZOOM, c.height * ZOOM), Image.LANCZOS).save(cp)
                    call(probe, p, name, Image.open(cp).convert("RGB"), list(box),
                         "items_" + probe[-2:], 600)
    log.close()
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    sys.exit(main())
