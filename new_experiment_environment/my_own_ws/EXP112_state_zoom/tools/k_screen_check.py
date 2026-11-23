#!/usr/bin/env python3
"""EXP112: how strict is the dev screening rule "zoom condition beats Z-full by >= k pallets"? (no data, no model)

Exact: with d discordant pallets (one condition right, the other wrong) and no true difference, P(net >= k).
Simulation: p_full vs p_zoom per pallet; pallets either independent, or the two pallets of a layout share one
outcome (full clustering, the other extreme); conditions independent given p (assumption, stated in the report).
"best of two" = Z-crop and Z-both both tried, the better one is screened (family false-pass under no difference).
Usage: python tools/k_screen_check.py --out analysis/k_screen_<ts>   |   --selftest
"""
import argparse, json, sys
from math import comb
from pathlib import Path

import numpy as np


def p_net_at_least(d, k):
    """P(W - (d - W) >= k), W ~ Bin(d, 0.5)."""
    return sum(comb(d, w) for w in range(d + 1) if 2 * w - d >= k) / 2 ** d


def simulate(p_full, p_zoom, n_layouts, k, clustered, best_of_two, n_sim=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = n_layouts if clustered else 2 * n_layouts
    w = 2 if clustered else 1  # pallets per simulated unit
    full = (rng.random((n_sim, n)) < p_full).sum(1) * w
    z1 = (rng.random((n_sim, n)) < p_zoom).sum(1) * w
    if best_of_two:
        z2 = (rng.random((n_sim, n)) < p_zoom).sum(1) * w
        z1 = np.maximum(z1, z2)
    return float(np.mean(z1 - full >= k))


def selftest():
    assert p_net_at_least(3, 3) == 1 / 8 and p_net_at_least(4, 3) == 1 / 16 and p_net_at_least(2, 3) == 0.0
    assert abs(simulate(0.5, 0.5, 8, 0, False, False) - 0.57) < 0.03  # P(zoom >= full) with ties, sanity
    print('selftest PASS')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=False)
    exact = {d: {k: round(p_net_at_least(d, k), 3) for k in (2, 3, 4)} for d in range(2, 13)}
    sims = []
    for split, n_lay in (('dev', 8), ('test', 16)):
        for p_zoom in (0.5, 0.65, 0.8, 0.95):
            for clustered in (False, True):
                for best in (False, True):
                    sims.append({'split': split, 'pallets': 2 * n_lay, 'layouts': n_lay, 'p_full': 0.5, 'p_zoom': p_zoom,
                                 'clustered': clustered, 'best_of_two': best,
                                 **{f'P(pass k={k})': round(simulate(0.5, p_zoom, n_lay, k, clustered, best), 3) for k in (2, 3, 4, 5)}})
    res = {'assumptions': 'p_full = 0.5 (EXP120 oracle J-vlm state 6/12); conditions independent given p; '
                          'clustered = the two pallets of a layout share one outcome (upper extreme of within-layout correlation)',
           'exact_null_by_discordant': exact, 'simulation': sims}
    (out / 'k_screen.json').write_text(json.dumps(res, indent=1))
    L = ['# k 선별 기준 점검 (자료 없음, 계산만) — 자동 생성', '', f"- 가정: {res['assumptions']}", '',
         '## 차이가 없을 때, 갈린 팔레트 수(d)별로 net ≥ k가 나올 확률 (정확 계산)', '', '| d | k=2 | k=3 | k=4 |', '|---|---|---|---|']
    L += [f'| {d} | {v[2]} | {v[3]} | {v[4]} |' for d, v in exact.items()]
    L += ['', '## 모의실험 (p_full = 0.5)', '', '| 분할 | 팔레트 | p_zoom | 배치 내 공유 | 둘 중 나은 쪽 | k=2 | k=3 | k=4 | k=5 |', '|---|---|---|---|---|---|---|---|---|']
    L += [f"| {s['split']} | {s['pallets']} | {s['p_zoom']} | {'예' if s['clustered'] else '아니오'} | {'예' if s['best_of_two'] else '아니오'} | "
          f"{s['P(pass k=2)']} | {s['P(pass k=3)']} | {s['P(pass k=4)']} | {s['P(pass k=5)']} |" for s in sims]
    (out / 'SUMMARY.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
