#!/usr/bin/env python3
"""Sensitivity of the main intervals to model-fitting variability (reviewer request).

The paper's CIs bootstrap seed-averaged out-of-fold predictions, so the models are fitted once. Here every bootstrap
replicate resamples PAPERS with replacement (all copies of a resampled paper keep the ORIGINAL paper id as their group,
so a paper and its copies always fall in the same fold and never straddle train and test), then re-runs the full protocol on the resample: stratified paper-grouped 5-fold CV with
nested grouped tuning of C. The percentile interval therefore includes fitting and hyperparameter-selection variability.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_experiments as E  # noqa: E402

N_BOOT = 200
SPECS = {'B1': {'kind': 'lr', 'num': E.A_NUM, 'cat': E.A_CAT},
         'M2': {'kind': 'lr', 'num': E.A_NUM + E.FLAT + E.GRAPH + E.DEP, 'cat': E.A_CAT},
         'B0': {'kind': 'prior'}}


def one(b, X, y, groups):
    rng = np.random.default_rng(1000 + b)
    uniq = np.unique(groups); pick = rng.choice(uniq, len(uniq), replace=True)
    idx, g = [], []
    for p in pick:
        pos = np.flatnonzero(groups == p); idx.extend(pos); g.extend([p] * len(pos))
    idx, g = np.array(idx), np.array(g); Xb, yb = X.iloc[idx].reset_index(drop=True), y[idx]
    preds = {k: np.zeros(len(yb)) for k in SPECS}
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=b).split(Xb, yb, g):
        for k, spec in SPECS.items():
            preds[k][te] = E.fit_predict(spec, Xb.iloc[tr], yb[tr], g[tr], Xb.iloc[te], b)
    return {k: (roc_auc_score(yb, p), brier_score_loss(yb, p)) for k, p in preds.items()}


def main():
    df, G = E.load()
    X = pd.concat([df, G['strict']], axis=1); y = df.replicated.to_numpy(); groups = df.paper_id.to_numpy()
    res = Parallel(n_jobs=-1, verbose=0)(delayed(one)(b, X, y, groups) for b in range(N_BOOT))
    q = lambda a: [float(np.quantile(a, .025)), float(np.quantile(a, .975))]
    auc = {k: np.array([r[k][0] for r in res]) for k in SPECS}; bri = {k: np.array([r[k][1] for r in res]) for k in SPECS}
    out = {'n_boot': N_BOOT,
           'B1_auroc_ci': q(auc['B1']), 'M2_auroc_ci': q(auc['M2']), 'M2_minus_B1_auroc_ci': q(auc['M2'] - auc['B1']),
           'B1_brier_skill_ci': q(1 - bri['B1'] / bri['B0']), 'B1_minus_B0_brier_ci': q(bri['B1'] - bri['B0']),
           'B1_auroc_gt_05_share': float((auc['B1'] > 0.5).mean()),
           'B1_auroc_median': float(np.median(auc['B1'])), 'M2_auroc_median': float(np.median(auc['M2'])),
           'note': 'each resample holds ~63% distinct papers, so refitted AUROCs are biased low and the interval is conservative'}
    (ROOT / 'results' / 'refit_bootstrap.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
