#!/usr/bin/env python3
"""Decision value under a review budget (reviewer request): risk-coverage and matched escalation budgets.

For a verifier score p, the best policy that automates a fraction k of claims automates the k most confident ones
(|p - 0.5| largest) and decides by p >= 0.5. Its error among automated decisions, as a function of coverage k, is the
risk-coverage curve. We compute it for the real (weak) verifier and for the synthetic decision-ready verifier of the R6
experiment, with paper-clustered bootstrap bands, and overlay the observed operating points of every agent arm
(coverage = share automated; risk = wrong automatic decisions / automated decisions).
Matched budgets: at escalation budgets of 25/50/75%, wrong automatic decisions as a share of ALL claims.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import r6_experiment as R6  # noqa: E402

GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)
BUDGETS = (0.25, 0.5, 0.75)


def curve(p, y, grid=GRID):
    order = np.argsort(-np.abs(p - 0.5), kind='stable'); wrong = ((p >= 0.5) != (y == 1))[order]
    return np.array([wrong[:max(1, int(round(k * len(p))))].mean() for k in grid])


def band(p, y, g, n=1000, seed=7):
    uniq = np.unique(g); pos = {k: np.flatnonzero(g == k) for k in uniq}; rng = np.random.default_rng(seed); B = []
    for _ in range(n):
        ii = np.concatenate([pos[k] for k in rng.choice(uniq, len(uniq))]); B.append(curve(p[ii], y[ii]))
    B = np.array(B); return np.quantile(B, .025, axis=0), np.quantile(B, .975, axis=0)


def main():
    rows, y, oof = R6.load()
    A = [r for r in rows if r['scenario'] == 'A']
    ids = [r['claim_id'] for r in A]; yy = np.array([y[c] for c in ids]); g = oof.loc[ids, 'paper_id'].to_numpy()
    p_weak = np.array([r['unprotected']['p_replicate'] for r in A])
    p_strong_map, st = R6.strong_tool(A, y); p_strong = np.array([p_strong_map[c] for c in ids])
    out = {'grid': GRID.tolist(), 'base_error': float(min(yy.mean(), 1 - yy.mean())), 'tools': {}}
    for name, p in (('weak', p_weak), ('strong', p_strong)):
        c = curve(p, yy); lo, hi = band(p, yy, g)
        bud = {}
        for e in BUDGETS:
            k = 1 - e; err = curve(p, yy, np.array([k]))[0]
            bud[str(e)] = {'coverage': k, 'risk': float(err), 'wrong_share_of_all': float(err * k)}
        out['tools'][name] = {'risk': c.tolist(), 'lo': lo.tolist(), 'hi': hi.tolist(), 'budgets': bud}
    # agent operating points
    S = json.loads((ROOT / 'results/agent_summary.json').read_text()); Rs = json.loads((ROOT / 'results/r6_summary.json').read_text())
    pts = []
    for m, v in S.items():
        for arm in ('no_tool', 'placebo', 'unprotected', 'contract_aware'):
            s = v['A'].get(arm)
            if s and s['auto'] > 0:
                pts.append({'model': m, 'tool': 'weak', 'arm': arm, 'coverage': s['auto'], 'risk': s['auto_error'] / s['auto']})
            elif s:
                pts.append({'model': m, 'tool': 'weak', 'arm': arm, 'coverage': 0.0, 'risk': None})
        r = Rs.get(m, {})
        for arm in ('v3',):
            s = r.get('A', {}).get(arm)
            if s:
                pts.append({'model': m, 'tool': 'weak', 'arm': 'r6_' + arm, 'coverage': s['auto'],
                            'risk': (s['wrong'] / s['auto']) if s['auto'] > 0 else None})
        for arm, s in r.get('S', {}).items():
            if isinstance(s, dict):
                pts.append({'model': m, 'tool': 'strong', 'arm': arm, 'coverage': s['auto'],
                            'risk': (s['wrong'] / s['auto']) if s['auto'] > 0 else None})
    out['agent_points'] = pts; out['synthetic_tool'] = st
    (ROOT / 'results/budget_analysis.json').write_text(json.dumps(out, indent=2))
    for name, t in out['tools'].items():
        print(name, {e: (round(b['risk'], 3), round(b['wrong_share_of_all'], 3)) for e, b in t['budgets'].items()},
              'risk@10%', round(t['risk'][1], 3), 'risk@100%', round(t['risk'][-1], 3))
    for p_ in pts:
        print(p_['model'][:12], p_['tool'], p_['arm'].ljust(15), round(p_['coverage'], 3), None if p_['risk'] is None else round(p_['risk'], 3))


if __name__ == '__main__':
    main()
