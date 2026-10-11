#!/usr/bin/env python3
"""Figures for the paper: (1) AUROC with paper-clustered 95% CIs, (2) reliability diagram."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / 'results', ROOT / 'paper' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
S = json.loads((RES / 'summary.json').read_text())['models']

ORDER = [('B0 Prevalence', 'B0 prevalence'), ('B1 Original-study metadata (LR)', 'B1 metadata (LR)'),
         ('B1g Original-study metadata (GBT)', 'B1g metadata (GBT)'), ('B2 Title/abstract TF-IDF (LR)', 'B2 text TF-IDF'),
         ('B4 Metadata + flat citation evidence', 'B4 + flat evidence'), ('M1 Metadata + evidence graph', 'M1 + graph'),
         ('M2 Metadata + graph + dependency correction', 'M2 + dependency'), ('G-only Graph + dependency (no metadata)', 'Graph only'),
         ('NC M2 with permuted graph (negative control)', 'M2 permuted (NC)'),
         ('S-M2 Graph at replication-launch cutoff', 'M2 @ launch cutoff'),
         ('D Metadata + replication design (launch-time)', 'B1 + planned design'),
         ('Dx Metadata + achieved replication size (post-launch)', 'D + achieved size (leak)'),
         ('L-M2 Graph + dependency, NO cutoff (leaky)', 'M2, no cutoff (leaky)'), ('LG-only Graph, NO cutoff (leaky)', 'Graph only, leaky')]
fig, ax = plt.subplots(figsize=(3.4, 3.15))
for i, (k, lab) in enumerate(ORDER):
    m = S[k]['pooled']['auroc']; lo, hi = S[k]['ci95']['auroc']
    col = '#b2182b' if ('leaky' in k or k.startswith('Dx')) else ('#2166ac' if k.startswith(('M', 'G-', 'S-')) else '#555555')
    ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt='o', ms=3, color=col, capsize=2, lw=1)
ax.axvline(0.5, color='k', lw=0.6, ls=':')
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([l for _, l in ORDER]); ax.invert_yaxis()
ax.set_xlabel('AUROC (paper-clustered 95% CI)')
fig.tight_layout(); fig.savefig(FIG / 'auroc.pdf'); fig.savefig(FIG / 'auroc.png', dpi=200)

oof = pd.read_csv(RES / 'oof_predictions_seedavg.csv'); y = oof.y.to_numpy()
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for k, lab, c in [('B1 Original-study metadata (LR)', 'B1 metadata', '#555555'),
                  ('M2 Metadata + graph + dependency correction', 'M2 graph+dep.', '#2166ac'),
                  ('L-M2 Graph + dependency, NO cutoff (leaky)', 'M2 leaky', '#b2182b')]:
    p = oof[k].to_numpy(); q = np.quantile(p, np.linspace(0, 1, 6)); idx = np.clip(np.searchsorted(q, p, side='right') - 1, 0, 4)
    xs = [p[idx == b].mean() for b in range(5)]; ys = [y[idx == b].mean() for b in range(5)]
    ax.plot(xs, ys, 'o-', ms=3, lw=1, color=c, label=lab)
ax.plot([0, 1], [0, 1], 'k:', lw=0.6); ax.set_xlim(0.2, 0.9); ax.set_ylim(0.1, 1.0)
ax.set_xlabel('Mean predicted P(replicate) (quintiles)'); ax.set_ylabel('Observed replication rate'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / 'calibration.pdf'); fig.savefig(FIG / 'calibration.png', dpi=200)
print('figures written to', FIG)
