"""Small SYNTHETIC cases for situations the three stored runs do not cover. They test the logic of V1 and V2, not model quality.
Every boxed object here is typed in by hand (there is no photograph); nothing is derived from, or tuned to, a stored result.
Geometry reuses the truck-bay layout of the development scene only to keep the numbers readable.

Each scenario carries what V1 needs to be replayed (V1's gate is interleaved with model calls; see v1_replay.py): the judge
answer is fixed to 'yes' so that V1's decision depends only on its geometric/evidence rules.
"""
import copy

import numpy as np

SIZE = [1920, 1080]
TRUCK = [1424, 73, 1623, 204]
BAY_TIGHT = [1400, 60, 1670, 240]
COARSE = [1250, 0, 1830, 280]
SHIPPING = "빨간 큐브를 출하할 수 있는 적절한 장소로 옮겨줘."
CHARGING = "빨간 큐브를 충전할 수 있는 곳으로 옮겨줘."
NO_PURPOSE = "빨간 큐브를 옮겨줘."
V02 = ("truck", "conveyor", "mobile_robot")


def cand(cid, func, box, evidence, reasoning="placed where the vehicle stops"):
    return {"id": cid, "function_claimed": func, "box_px": box, "observed_evidence": evidence, "reasoning_text": reasoning,
            "confidence": "high"}


def obj(oid, cls, box):
    return {"object_id": oid, "class_name": cls, "bbox_xyxy": box, "basis": "synthetic (hand-typed)"}


def case(cid, instruction, candidates, objects, searched=(), localizations=(), aligned=True, cube="OK", chosen=None):
    searched = {c: {"status": "searched", "scope": "full_image"} for c in (*V02, *searched)}
    layers = {"traversable": np.zeros((SIZE[1], SIZE[0]), bool), "no_go": np.zeros((SIZE[1], SIZE[0]), bool)}
    return {"id": cid, "image_size": SIZE, "instruction": instruction, "candidates": candidates,
            "chosen": chosen or candidates[0]["id"],
            "observations": {"objects": objects, "searched": searched, "localizations": list(localizations),
                             "map": {"alignment": "ALIGNED" if aligned else "MISALIGNED", "layers": layers if aligned else None},
                             "preconditions": {"cube": cube, "start": "manual_prior"}}}


def loc(cid, box=(1405, 65, 1665, 235)):
    return {"candidate_id": cid, "box_px": list(box), "source": "evidence_grounded_refinement", "query": "synthetic"}


CONES = [obj(f"traffic_cone_{i}", "traffic_cone", b) for i, b in enumerate(
    [[1440, 200, 1470, 235], [1500, 205, 1530, 238], [1560, 200, 1590, 235]], 1)]


def v1_inputs(c):
    """What V1's replay needs, derived from the scenario's own case (chosen candidate)."""
    chosen = next(x for x in c["candidates"] if x["id"] == c["chosen"])
    obs = c["observations"]
    locs = [l for l in obs["localizations"] if l["candidate_id"] == chosen["id"]]
    return {"area": {"id": chosen["id"], "function_name": chosen["function_claimed"], "inference": chosen["reasoning_text"],
                     "observed_evidence": chosen["observed_evidence"]},
            "vbox": chosen["box_px"], "size": tuple(c["image_size"]),
            "accepted": [o for o in obs["objects"] if o["class_name"] in V02],
            "refinement_chosen": locs[0]["box_px"] if locs else None,
            "cone_hits": {"traffic_cone": [o["bbox_xyxy"] for o in obs["objects"] if o["class_name"] == "traffic_cone"]},
            "judge": "yes", "map_aligned": obs["map"]["alignment"] == "ALIGNED", "layers": obs["map"]["layers"], "other_areas": []}


def scenarios():
    out = []

    def add(sid, text, c, expect, cfg_extra=None):
        out.append({"id": sid, "description": text, "case": c, "expect": expect, "registry_extra": cfg_extra or {}})

    ev_truck = ["white truck parked in the bay", "orange cones around the bay"]
    add("S1_consistent_and_localized", "evidence exists, stands in the right relation, spot localized",
        case("S1", SHIPPING, [cand("area_1", "loading dock", BAY_TIGHT, ev_truck)], [obj("truck_1", "truck", TRUCK), *CONES],
             searched=["traffic_cone"], localizations=[loc("area_1")]),
        {"v2": "SEMANTICALLY_SUPPORTED_READY_FOR_PLANNING", "hand_over": True, "v1": "ADOPTED"})
    add("S2_evidence_object_absent", "the VLM cites a truck; the detector searched and found none",
        case("S2", SHIPPING, [cand("area_1", "loading dock", BAY_TIGHT, ["white truck parked in the bay"])], []),
        {"v2": "EVIDENCE_NOT_ESTABLISHED", "hand_over": False, "v1": "WITHHELD"})
    add("S3_evidence_elsewhere", "the cited truck exists but stands far from the proposed region",
        case("S3", SHIPPING, [cand("area_1", "loading dock", BAY_TIGHT, ["white truck parked in the bay"])],
             [obj("truck_1", "truck", [200, 500, 400, 650])]),
        {"v2": "EVIDENCE_RELATION_INCONSISTENT", "hand_over": False, "v1": "WITHHELD"})
    add("S4_multiple_candidates", "two destination candidates; the VLM chose the weaker one; the verifier does not swap",
        case("S4", SHIPPING, [cand("area_1", "loading dock", BAY_TIGHT, ["white truck parked in the bay"]),
                              cand("area_2", "exit zone", [1250, 800, 1830, 1030], ["lightning bolt symbol on the floor"])],
             [obj("truck_1", "truck", TRUCK)], localizations=[loc("area_1")], chosen="area_2"),
        {"v2": "EVIDENCE_NOT_ESTABLISHED", "hand_over": False, "v1": "WITHHELD", "v2_other": {"area_1": "SEMANTICALLY_SUPPORTED_READY_FOR_PLANNING"}})
    add("S5_semantic_ok_placement_unverified", "truck inside a wide region, no localization of a spot",
        case("S5", SHIPPING, [cand("area_1", "loading dock", COARSE, ev_truck)], [obj("truck_1", "truck", TRUCK), *CONES],
             searched=["traffic_cone"]),
        {"v2": "SEMANTICALLY_SUPPORTED_PLACEMENT_UNVERIFIED", "hand_over": False, "v1": "WITHHELD"})
    add("S6_non_truck_carrier", "a delivery van (registered as a carrier by CONFIG only) in a tight, localized region",
        case("S6", SHIPPING, [cand("area_1", "dispatch bay", BAY_TIGHT, ["white delivery van parked"])],
             [obj("van_1", "delivery_van", TRUCK)], searched=["delivery_van"], localizations=[loc("area_1")]),
        {"v2": "SEMANTICALLY_SUPPORTED_READY_FOR_PLANNING", "hand_over": True, "v1": "WITHHELD"},
        {"delivery_van": {"terms": ["delivery van", "van"], "backend": "v02_detection", "role": "equipment",
                          "capabilities": ["carrier_vehicle"], "obstacle": True}})
    add("S7_purpose_mismatch", "the instruction asks for a CHARGING place; the VLM proposes the truck bay",
        case("S7", CHARGING, [cand("area_1", "charging spot", BAY_TIGHT, ev_truck)], [obj("truck_1", "truck", TRUCK), *CONES],
             searched=["traffic_cone"], localizations=[loc("area_1")]),
        {"v2": "PURPOSE_FIT_UNDETERMINED", "hand_over": False, "v1": "ADOPTED"})
    add("S8_unmodeled_obstacle_claim", "the VLM also cites crates by the bay: a physical object no detector or obstacle model knows",
        case("S8", SHIPPING, [cand("area_1", "loading dock", BAY_TIGHT, [*ev_truck, "crates stacked beside the bay"])],
             [obj("truck_1", "truck", TRUCK), *CONES], searched=["traffic_cone"], localizations=[loc("area_1")]),
        {"v2": "SEMANTICALLY_SUPPORTED_PLANNER_UNSUPPORTED", "hand_over": False, "v1": "ADOPTED"})
    add("S9_purpose_unknown", "the instruction names no purpose the table knows",
        case("S9", NO_PURPOSE, [cand("area_1", "loading dock", BAY_TIGHT, ev_truck)], [obj("truck_1", "truck", TRUCK), *CONES],
             searched=["traffic_cone"], localizations=[loc("area_1")]),
        {"v2": "PURPOSE_FIT_UNDETERMINED", "hand_over": False, "v1": "ADOPTED"})
    add("S10_other_function_verified", "pallets are verified inside the region, but they show storage, not shipping",
        case("S10", SHIPPING, [cand("area_1", "dispatch area", [1200, 400, 1800, 700], ["pallets with goods"])],
             [obj("pallet_1", "pallet", [1250, 450, 1400, 560]), obj("pallet_2", "pallet", [1450, 450, 1600, 560])],
             searched=["pallet"], localizations=[loc("area_1", (1220, 420, 1780, 680))]),
        {"v2": "PURPOSE_FIT_UNDETERMINED", "hand_over": False, "v1": "WITHHELD"})
    return out


def scenario_cfg(cfg, sc):
    c = copy.deepcopy(cfg)
    c["registry"].update(sc["registry_extra"])
    return c
