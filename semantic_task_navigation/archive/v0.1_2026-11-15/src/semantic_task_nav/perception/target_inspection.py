"""Small red-cube inspection.

1 conveyor ROI (verified conveyors only; full image if none) -> 2 uniform zoom -> 3 HSV colour +
shape candidates (+ detector on the zoomed ROI, + full-image detector candidates inside the ROI)
-> 4 merge -> 5 Qwen3-VL crop verification per candidate -> 6 boxes back in raw_image_px
-> 7 decision. A red blob alone is never accepted as the target.
"""
from pathlib import Path

import cv2
import numpy as np

from ..contracts import OBSERVED, RAW_PX, ROI_PX
from ..geometry.coordinate_frames import RoiTransform, box_area, box_center, box_gap, containment, expand_box, iou
from .grounding import dino_candidates, make_detection, verify_box


def make_roi(box, image_size, tcfg):
    """Isotropic margin = max(ratio * long side, min px): the 'vicinity' of an object should not depend
    on its aspect ratio (DEV NOTE 2026-11-15: per-axis margins cut the reference cube off a narrow conveyor ROI)."""
    w, h = image_size
    margin = max(tcfg["roi_margin_ratio"] * max(box[2] - box[0], box[3] - box[1]), tcfg["roi_min_margin_px"])
    crop = expand_box(box, 0.0, margin, w, h)
    long_side = max(crop[2] - crop[0], crop[3] - crop[1])
    scale = min(tcfg["roi_max_zoom"], tcfg["roi_zoom_long_side_px"] / long_side)
    return RoiTransform(crop, max(scale, 1.0))


def zoom_roi(image, roi):
    x0, y0, x1, y1 = roi.crop_xyxy
    return cv2.resize(image[y0:y1, x0:x1], roi.zoom_size, interpolation=cv2.INTER_CUBIC)


def hsv_candidates(zoom_bgr, roi, tcfg):
    """Red-colour blobs with cube-like shape statistics. Boxes in both roi_local_px and raw_image_px."""
    hsv = cv2.cvtColor(zoom_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in tcfg["hsv_red_ranges"]:
        mask |= cv2.inRange(hsv, (lo, tcfg["hsv_min_saturation"], tcfg["hsv_min_value"]), (hi, 255, 255))
    k = max(3, int(round(roi.scale)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area_raw = cv2.contourArea(cnt) / roi.scale ** 2
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = cv2.contourArea(cnt) / hull_area if hull_area > 0 else 0.0
        aspect = max(bw, bh) / max(1, min(bw, bh))
        stats = {"area_raw_px": round(area_raw, 1), "aspect": round(aspect, 2), "solidity": round(solidity, 3)}
        ok = (tcfg["candidate_min_area_raw_px"] <= area_raw <= tcfg["candidate_max_area_raw_px"]
              and aspect <= tcfg["candidate_max_aspect"] and solidity >= tcfg["candidate_min_solidity"])
        if not ok:
            continue
        box_roi = [x, y, x + bw, y + bh]
        out.append({"bbox_roi": {"xyxy": box_roi, "frame": ROI_PX}, "bbox_xyxy": roi.roi_to_raw_box(box_roi),
                    "score": None, "prompt": None, "source": "hsv_color_shape", "shape_stats": stats})
    return out


def _same_object(a, b, thr):
    small, large = (a, b) if box_area(a) <= box_area(b) else (b, a)
    return iou(a, b) >= thr or containment(small, large) >= 0.8


def merge_candidates(cands, thr):
    """Union candidates from several sources. Boxes are merged when IoU >= thr or one lies inside the
    other (a colour blob inside a detector box). The merged box is the colour blob when one exists
    (tight box on the coloured object), otherwise the highest-score detector box. All member boxes are
    kept as evidence. DEV NOTE 2026-11-15: IoU-only merging left the tight HSV box of the reference cube
    (IoU 0.10 with the detector box of cube+tray) unverified."""
    merged = []
    for c in cands:
        for m in merged:
            if any(_same_object(c["bbox_xyxy"], x["bbox_xyxy"], thr) for x in m["members"]):
                m["sources"].append(c["source"])
                m["members"].append(c)
                if c.get("score") is not None and (m.get("score") is None or c["score"] > m["score"]):
                    m["score"], m["prompt"] = c["score"], c.get("prompt")
                if c["source"] == "hsv_color_shape" and "hsv_color_shape" not in m["sources"][:-1]:
                    m["bbox_xyxy"], m["bbox_roi"], m["roi_index"] = c["bbox_xyxy"], c.get("bbox_roi"), c.get("roi_index")
                    m["shape_stats"] = c.get("shape_stats")
                break
        else:
            merged.append({**c, "sources": [c["source"]], "members": [c]})
    for m in merged:
        m["member_boxes"] = [{"source": x["source"], "bbox_xyxy": [round(float(v), 2) for v in x["bbox_xyxy"]],
                              "score": x.get("score")} for x in m["members"]]
    merged.sort(key=lambda m: (-len(set(m["sources"])), -(m["score"] or 0.0), -m.get("shape_stats", {}).get("solidity", 0)))
    return merged


def inspect_target(client, mcfg, pcfg, image, frame_id, task_target, conveyors, full_image_candidates, out_dir,
                   model_mode=True, emit=lambda *a, **k: None):
    tcfg = mcfg["target_inspection"]
    h, w = image.shape[:2]
    out_dir = Path(out_dir)
    notes = []
    if conveyors:
        rois = [(c["object_id"], make_roi(c["bbox_xyxy"], (w, h), tcfg)) for c in conveyors]
    else:
        rois = [("full_image", RoiTransform([0, 0, w, h], 1.0))]
        notes.append("NO_VERIFIED_CONVEYOR: searched the full image without a conveyor ROI")
    roi_records, cands = [], []
    key = f"{task_target['color']}_{task_target['class']}"
    for k, (anchor, roi) in enumerate(rois):
        zoom = zoom_roi(image, roi)
        zpath = out_dir / f"roi_{k + 1}_zoom.png"
        zpath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(zpath), zoom)
        roi_records.append({"roi_index": k + 1, "anchor": anchor, "transform": roi.to_dict(), "zoom_path": str(zpath)})
        hc = hsv_candidates(zoom, roi, tcfg)
        for c in hc:
            c["roi_index"] = k + 1
        cands += hc
        if model_mode and mcfg["grounding_dino"]["roi_detector"] and anchor != "full_image":
            dc, _ = dino_candidates(client, mcfg, zpath, roi.zoom_size, mcfg["grounding_dino"]["class_prompts"][key])
            for c in dc:
                c["bbox_roi"] = {"xyxy": [round(v, 2) for v in c["bbox_xyxy"]], "frame": ROI_PX}
                c["bbox_xyxy"] = roi.roi_to_raw_box(c["bbox_xyxy"])
                c["source"] = "grounding_dino_roi"
                c["roi_index"] = k + 1
            cands += [c for c in dc if (c["bbox_xyxy"][2] - c["bbox_xyxy"][0]) * (c["bbox_xyxy"][3] - c["bbox_xyxy"][1])
                      <= tcfg["candidate_max_area_raw_px"] * 4]
        emit("info", f"ROI {k + 1} ({anchor}): zoom x{roi.scale:.2f}, {len(hc)} colour/shape candidates")
    for c in full_image_candidates:
        if box_area(c["bbox_xyxy"]) > tcfg["candidate_max_area_raw_px"] * 4:   # same size cap as ROI detector boxes
            continue
        cx, cy = box_center(c["bbox_xyxy"])
        if any(r.crop_xyxy[0] <= cx < r.crop_xyxy[2] and r.crop_xyxy[1] <= cy < r.crop_xyxy[3] for _, r in rois):
            cands.append({**c, "source": "grounding_dino_full_image"})
    merged = merge_candidates(cands, tcfg["merge_iou"])
    desc = mcfg["verification"]["descriptions"]["target"].replace("{color}", task_target["color"]).replace(
        "{object_class}", task_target["class"])
    to_verify = merged[:tcfg["max_candidates_to_verify"]]
    for i, m in enumerate(merged):
        m["candidate_id"] = f"target_cand{i + 1}"
        if not model_mode:
            m["verification"] = {"status": "not_run", "reason": "IMAGE_DEMO: no model"}
        elif m in to_verify:
            m["verification"] = verify_box(client, mcfg, image, m["bbox_xyxy"], desc, out_dir / f"verify_target_{i + 1}.png")
            emit("info", f"verify {m['candidate_id']} ({'+'.join(sorted(set(m['sources'])))}): {m['verification']['status']}")
        else:
            m["verification"] = {"status": "not_run", "reason": f"beyond max_candidates_to_verify={len(to_verify)}"}
        m["bbox_xyxy"] = [round(float(v), 2) for v in m["bbox_xyxy"]]
        m["coordinate_frame"] = RAW_PX
        m.pop("members", None)
    accepted = [m for m in merged if m["verification"]["status"] == "yes"]
    result = {"rois": roi_records, "candidates": merged, "n_candidates": len(merged), "n_verified": len(to_verify) if model_mode else 0,
              "n_accepted": len(accepted), "description": desc, "notes": notes, "target": None}
    if not model_mode:
        result["status"] = "NOT_RUN_IMAGE_DEMO"
        return result
    if len(accepted) > 1 and task_target.get("source_reference") == "conveyor" and conveyors:
        near = [m for m in accepted if min(box_gap(m["bbox_xyxy"], c["bbox_xyxy"]) for c in conveyors)
                <= pcfg["relations"]["near_px"]]
        notes.append(f"{len(accepted)} accepted candidates; {len(near)} near a verified conveyor")
        if len(near) == 1:
            accepted = near
    if not accepted:
        result["status"] = "TARGET_NOT_FOUND"
    elif len(accepted) > 1:
        result["status"] = "TARGET_AMBIGUOUS"
        result["ambiguous_candidates"] = [m["candidate_id"] for m in accepted]
    else:
        m = accepted[0]
        result["status"] = "TARGET_SELECTED"
        # Box comes from the candidate source(s); the verifier (Qwen) is recorded in `verification`.
        result["target"] = make_detection(f"{key}_1", key, m, frame_id,
                                          "candidate:" + "+".join(sorted(set(m["sources"]))), None, OBSERVED,
                                          accepted=True, verification=m["verification"],
                                          candidate_sources=sorted(set(m["sources"])), candidate_id=m["candidate_id"])
    return result
