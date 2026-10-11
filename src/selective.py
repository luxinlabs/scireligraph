#!/usr/bin/env python3
"""Selective prediction: if an agent acts only on its most confident verifier outputs, how accurate is it?

Two confidence definitions are reported:
  abs05   |p - 0.5|, decision threshold 0.5 (standard; primary)
  median  |p - median(p)|, decision threshold median(p) (robust to a shifted prediction centre)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]
oof = pd.read_csv(ROOT / 'results/oof_predictions_seedavg.csv'); y = oof.y.to_numpy()
groups = oof.paper_id.to_numpy(); uniq = np.unique(groups); pos = {g: np.flatnonzero(groups == g) for g in uniq}
rng = np.random.default_rng(3)
MODELS = ['B1 Original-study metadata (LR)', 'M2 Metadata + graph + dependency correction', 'D Metadata + replication design (launch-time)']
COVS = [0.1, 0.2, 0.3, 0.5, 1.0]


def sel_acc(p, yy, centre, cov):
    conf = np.abs(p - centre); k = max(1, int(round(cov * len(p)))); idx = np.argsort(-conf, kind='stable')[:k]
    return ((p[idx] >= centre) == yy[idx]).mean(), (p[idx] >= centre).mean()


out = []
boots = [np.concatenate([pos[g] for g in rng.choice(uniq, len(uniq))]) for _ in range(2000)]
for m in tqdm(MODELS, desc='selective'):
    p = oof[m].to_numpy()
    for conf_def in ['abs05', 'median']:
        centre = 0.5 if conf_def == 'abs05' else float(np.median(p))
        for cov in COVS:
            acc, frac_pos = sel_acc(p, y, centre, cov)
            bs = [sel_acc(p[ii], y[ii], centre, cov)[0] for ii in boots]
            out.append({'model': m, 'confidence': conf_def, 'coverage': cov, 'n': int(round(cov * len(p))), 'accuracy': float(acc),
                        'frac_pred_replicate': float(frac_pos), 'ci': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]})
            tqdm.write(f'{m[:30]:30s} {conf_def:6s} cov {cov:.1f} acc {acc:.3f} [{out[-1]["ci"][0]:.3f},{out[-1]["ci"][1]:.3f}] pos {frac_pos:.2f}')
(ROOT / 'results/selective.json').write_text(json.dumps(out, indent=2, default=float))
