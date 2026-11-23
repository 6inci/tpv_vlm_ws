"""Scene graph: nodes = accepted objects / regions / poses, edges = typed relations.

Geometric relations (near, overlaps) are computed from image-space boxes and are labelled as such
(not metric). Relations that a single still image cannot establish (e.g. produced_by) are stored
as UNVERIFIED with the observable evidence attached, never as facts.
"""
from ..contracts import INFERRED, MANUAL, MANUAL_TEST, OBSERVED, RELATION_SCHEMA, validate
from ..geometry.coordinate_frames import box_gap, containment


def _rel(graph, rtype, subj, obj, method, evidence, status, verified, knowledge):
    r = {"relation_id": f"rel_{len(graph['relations']) + 1}", "type": rtype, "subject": subj, "object": obj,
         "method": method, "evidence": evidence, "status": status, "verified": verified, "knowledge": knowledge}
    graph["relations"].append(validate(r, RELATION_SCHEMA))
    return r


def build_scene_graph(objects, target, task, start, pcfg):
    """objects: accepted detections (conveyor/truck/mobile_robot); target: accepted target or None."""
    nodes = [dict(id=o["object_id"], kind="object", class_name=o["class_name"], bbox_xyxy=o["bbox_xyxy"],
                  frame=o["coordinate_frame"], knowledge=o["knowledge"]) for o in objects]
    if target:
        nodes.append(dict(id=target["object_id"], kind="object", class_name=target["class_name"],
                          bbox_xyxy=target["bbox_xyxy"], frame=target["coordinate_frame"], knowledge=target["knowledge"]))
    graph = {"nodes": nodes, "relations": [], "frame": "raw_image_px",
             "note": "geometric relations use image-space boxes (IMAGE_SPACE_DEMO); distances are pixels, not metres"}
    near_px, contain_ratio = pcfg["relations"]["near_px"], pcfg["relations"]["contain_ratio"]
    pool = list(objects) + ([target] if target else [])
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if a["class_name"] == b["class_name"]:
                continue
            gap = box_gap(a["bbox_xyxy"], b["bbox_xyxy"])
            know = MANUAL_TEST if MANUAL_TEST in (a["knowledge"], b["knowledge"]) else OBSERVED
            if gap <= near_px:
                _rel(graph, "near", a["object_id"], b["object_id"], "bbox_gap_image_px",
                     {"gap_px": round(gap, 1), "threshold_px": near_px}, "COMPUTED", False, know)
            c = containment(a["bbox_xyxy"], b["bbox_xyxy"])
            if c >= contain_ratio:
                _rel(graph, "overlaps_in_image", a["object_id"], b["object_id"], "bbox_containment_image_px",
                     {"containment": round(c, 3), "note": "2D overlap; 'on/inside' in 3D is not established"},
                     "COMPUTED", False, know)
    if target and task.get("target") and task["target"].get("source_reference"):
        src_cls = task["target"]["source_reference"]
        rel = task["target"]["source_relation"] or "from"
        srcs = [o for o in objects if o["class_name"] == src_cls]
        for s in srcs:
            gap = box_gap(target["bbox_xyxy"], s["bbox_xyxy"])
            _rel(graph, rel, target["object_id"], s["object_id"], "requested_by_instruction",
                 {"observable": {"gap_px": round(gap, 1), "near": gap <= near_px},
                  "note": "a single still frame cannot show that the object was produced by / came from this source"},
                 "UNVERIFIED", False, INFERRED)
        if not srcs:
            graph["unresolved"] = [f"instruction refers to a {src_cls}, but no {src_cls} was accepted in this frame"]
    if start:
        _rel(graph, "located_at", start["robot_ref"], "start_pose",
             start["method"], {"xy": start["xy"], "frame": start["coordinate_frame"]},
             "MANUAL" if start["knowledge"] == MANUAL else "COMPUTED", False, start["knowledge"])
    return graph


def add_region_relation(graph, truck_id, region, status):
    _rel(graph, "associated_with", truck_id, region["region_id"], region["source"],
         region["evidence"], "INFERRED" if status != "MANUAL" else "MANUAL", False,
         INFERRED if status != "MANUAL" else MANUAL)
    graph["nodes"].append({"id": region["region_id"], "kind": "region", "region_type": region["region_type"],
                           "polygon_xy": region["polygon_xy"], "frame": region["coordinate_frame"],
                           "verification_status": region["verification_status"]})
