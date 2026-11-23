"""Experiment 3: candidate decision under ablation conditions (pure functions).

Lexicographic evidence tiers. DECISIVE factors (higher = better) decide; TIE-BREAK factors (lower = better) only order the list.
The rank-1 eligible candidate is SELECTED only if its decisive tuple is strictly greater than every other eligible candidate's;
otherwise the photo stays AMBIGUOUS and the tied set is returned. Navigation never precedes the visual/semantic factors, so a short
path cannot outrank stronger evidence. Nothing is ever CONFIRMED: every selected target is VERIFY_REQUIRED.
"""
STATUSES = ("SELECTED", "AMBIGUOUS", "DEFERRED_NAVIGATION", "NO_ELIGIBLE_CANDIDATE", "NOT_DETERMINABLE")


def decide(cands, decisive, tiebreak=(), gate=lambda c: True, nav_factor=None):
    """cands: dicts with 'id' and 'rank'. decisive / tiebreak: [(name, fn)], fn(c) -> number. nav_factor: name of a decisive factor that,
    when 0 for the top tier, turns the decision into DEFERRED_NAVIGATION (lower tiers are NOT promoted)."""
    dvals = lambda c: tuple(f(c) for _, f in decisive)        # noqa: E731
    ranked = sorted(cands, key=lambda c: (0 if gate(c) else 1, tuple(-v for v in dvals(c)), tuple(f(c) for _, f in tiebreak), c["rank"]))
    elig = [c for c in ranked if gate(c)]
    names = [n for n, _ in decisive]
    out = {"decisive_factors": names, "tie_break_factors": [n for n, _ in tiebreak], "n_candidates": len(cands), "n_eligible": len(elig),
           "ranking": [{"position": i + 1, "id": c["id"], "eligible": gate(c), "decisive": dict(zip(names, dvals(c))),
                        "tie_break": {n: f(c) for n, f in tiebreak}} for i, c in enumerate(ranked)]}
    if not elig:
        return {**out, "status": "NO_ELIGIBLE_CANDIDATE", "selected": None, "tied_set": [], "reasons": {c["id"]: "gate not passed" for c in cands}}
    top = dvals(elig[0])
    tied = [c["id"] for c in elig if dvals(c) == top]
    status = "SELECTED" if len(tied) == 1 else "AMBIGUOUS"
    if nav_factor is not None and dict(zip(names, top))[nav_factor] == 0:
        status = "DEFERRED_NAVIGATION"
    reasons = {}
    for c in ranked:
        if not gate(c):
            reasons[c["id"]] = "gate not passed"
            continue
        v = dvals(c)
        if v == top:
            reasons[c["id"]] = {"SELECTED": "selected (strictly dominant on the decisive factors)", "AMBIGUOUS": f"tied with {len(tied) - 1} other(s) on all decisive factors",
                                "DEFERRED_NAVIGATION": "top evidence tier, but no feasible path A+B"}[status]
        else:
            k = next(i for i in range(len(v)) if v[i] != top[i])
            reasons[c["id"]] = f"lower {names[k]} ({v[k]} < {top[k]})"
    return {**out, "status": status, "selected": tied[0] if status == "SELECTED" else None, "tied_set": tied, "reasons": reasons}


def target_conditions(cands, dcfg, sem_scale):
    """The four target conditions on ONE candidate list (controlled variable). cands carry 'visual', 'semantic' (on/near/none), 'nav'."""
    vis = lambda c: dcfg["visual_level"][c["visual"]]                     # noqa: E731
    sem = lambda c: sem_scale[c["semantic"]]                               # noqa: E731
    feas = lambda c: 1 if (c["nav"].get("path_A_B_feasible") is True) else 0   # noqa: E731
    gap = lambda c: c["nav"]["gap_cm"] if c["nav"].get("gap_cm") is not None else 1e9   # noqa: E731
    score = lambda c: c["nav"]["v02_goal_score_cm"] if c["nav"].get("v02_goal_score_cm") is not None else 1e9   # noqa: E731
    plannable = lambda c: vis(c) >= dcfg["plannable_min_visual_level"]    # noqa: E731
    vis_tb = [("n_source_types_neg", lambda c: -len(c["source_types"])), ("detector_score_neg", lambda c: -(c["score"] or 0.0))]   # no score (HSV only) = lowest
    return {
        "A_visual": decide(cands, [("visual", vis)], vis_tb, plannable),
        "B_visual_semantic": decide(cands, [("visual", vis), ("semantic", sem)], vis_tb, plannable),
        "C_visual_semantic_navigation": decide(cands, [("visual", vis), ("semantic", sem), ("nav_feasible", feas)], [("gap_cm", gap), ("v02_goal_score_cm", score)],
                                               plannable, nav_factor="nav_feasible"),
        "D_navigation_only": decide(cands, [("neg_v02_goal_score_cm", lambda c: -score(c))], (), lambda c: c["nav"].get("v02_goal_score_cm") is not None)}


def destination_conditions(dests):
    """dests: dicts with 'id', 'rank', 'vlm_chosen', 'grounded', 'nav' (to the placement target, used by C) and 'nav_area'
    (to the area box itself, used by D, which knows nothing about grounding). Both from the same pick-up approach."""
    feas = lambda d: 1 if d["nav"].get("path_B_feasible") is True else 0   # noqa: E731
    gap = lambda d: d["nav"]["gap_cm"] if d["nav"].get("gap_cm") is not None else 1e9   # noqa: E731
    score = lambda d: d["nav"]["v02_goal_score_cm"] if d["nav"].get("v02_goal_score_cm") is not None else 1e9   # noqa: E731
    ascore = lambda d: d["nav_area"]["v02_goal_score_cm"] if d["nav_area"].get("v02_goal_score_cm") is not None else 1e9   # noqa: E731
    chosen, grounded = (lambda d: int(d["vlm_chosen"])), (lambda d: int(d["grounded"]))
    return {
        "A_visual": {"status": "NOT_DETERMINABLE", "selected": None, "reason": "visual evidence alone carries no notion of a place 'where it can be shipped'"},
        "B_visual_semantic": decide(dests, [("vlm_chosen", chosen), ("grounded_to_observed_bay", grounded)]),
        "C_visual_semantic_navigation": decide(dests, [("vlm_chosen", chosen), ("grounded_to_observed_bay", grounded), ("nav_feasible", feas)], [("gap_cm", gap), ("v02_goal_score_cm", score)],
                                               nav_factor="nav_feasible"),
        "D_navigation_only": decide(dests, [("neg_v02_goal_score_cm", lambda d: -ascore(d))], (), lambda d: d["nav_area"].get("v02_goal_score_cm") is not None)}
