#!/usr/bin/env python3
"""Score a probe_qwen.py run against the reference boxes. Usage: evaluate.py <run_dir>

Reference = STEP 4-A survey boxes (AI-read, copied to inputs/survey_step4a_COPY.json) plus the photo_6 empty
pallet added in siglip_small_object_test. Checked by the AI on a contact sheet (sim/results/slot_contact_sheet.jpg);
NOT human-confirmed. All boxes: undistorted_px.
Writes <run_dir>/evaluation.json only.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/binci/vlm_ws/task_flow_plan_20261120")
LOADED = {"cube_on_pallet", "loose_cube"}


def reference():
    d = json.loads((ROOT / "inputs" / "survey_step4a_COPY.json").read_text())
    ref = {}
    for i, p in enumerate(d["photos"], 1):
        ref[i] = [{"kind": "loaded", "box": p["cube"]["box_px"]}] + \
                 [{"kind": "empty", "box": q["box_px"]} for q in p["pallets_empty"]]
    ref[6].append({"kind": "empty", "box": [420, 408, 468, 450]})
    return ref


def parse(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t), True
    except Exception:
        return None, False


def to_px(b, frame, _size):
    x1, y1, x2, y2 = frame
    return [x1 + b[0] / 1000 * (x2 - x1), y1 + b[1] / 1000 * (y2 - y1),
            x1 + b[2] / 1000 * (x2 - x1), y1 + b[3] / 1000 * (y2 - y1)]


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def match(det, lab):
    cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
    inside = lab[0] - 8 <= cx <= lab[2] + 8 and lab[1] - 8 <= cy <= lab[3] + 8
    small = (det[2] - det[0]) <= 3 * (lab[2] - lab[0]) and (det[3] - det[1]) <= 3 * (lab[3] - lab[1])
    return (inside and small) or iou(det, lab) >= 0.3


def main(run):
    run = Path(run)
    ref = reference()
    calls = [json.loads(l) for l in (run / "calls.jsonl").read_text().splitlines()]
    out = {"reference_status": "AI_READ_NOT_HUMAN_CONFIRMED", "probes": {}, "details": []}
    for probe in ("full_v1", "roi_v1", "roi_v2", "roi_v3"):
        rows = [c for c in calls if c["probe"] == probe]
        if not rows:
            continue
        s = {"calls": len(rows), "parse_fail": 0, "hit_max_tokens": 0,
             "loaded_expected": 0, "loaded_found": 0, "loaded_found_right_type": 0,
             "empty_expected": 0, "empty_found": 0, "empty_found_right_type": 0,
             "false_items": 0, "windows_without_reference": 0, "windows_without_reference_clean": 0}
        for c in rows:
            js, ok = parse(c["text"])
            s["hit_max_tokens"] += int(c["hit_max"])
            items = (js or {}).get("items", []) if isinstance(js, dict) else []
            if not ok:
                s["parse_fail"] += 1
            f = c["frame_px"]
            dets = []
            for it in items:
                b = it.get("bbox_2d")
                if isinstance(b, list) and len(b) == 4:
                    dets.append({"type": it.get("type"), "px": [round(v, 1) for v in to_px(b, f, c["image_size"])],
                                 "desc": it.get("description"), "color": it.get("color")})
            labs = [l for l in ref[c["photo"]]
                    if f[0] <= (l["box"][0] + l["box"][2]) / 2 <= f[2] and f[1] <= (l["box"][1] + l["box"][3]) / 2 <= f[3]]
            used = set()
            for l in labs:
                k = l["kind"]
                s[f"{k}_expected"] += 1
                hit = [i for i, d in enumerate(dets) if match(d["px"], l["box"])]
                if hit:
                    s[f"{k}_found"] += 1
                    used.update(hit)
                    right = any((dets[i]["type"] in LOADED) == (k == "loaded") and dets[i]["type"] != "other_small_item"
                                for i in hit)
                    s[f"{k}_found_right_type"] += int(right)
            # a detection on a reference object that is only partly inside this window is not a false item
            extra = [d for i, d in enumerate(dets) if i not in used
                     and not any(match(d["px"], l["box"]) for l in ref[c["photo"]])]
            s["false_items"] += len(extra)
            if not labs:
                s["windows_without_reference"] += 1
                s["windows_without_reference_clean"] += int(len(extra) == 0)
            out["details"].append({"probe": probe, "photo": c["photo"], "roi": c["roi"], "n_items": len(dets),
                                   "expected": [l["kind"] for l in labs], "dets": dets,
                                   "false": [d["desc"] for d in extra]})
        out["probes"][probe] = s
    (run / "evaluation.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["probes"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
