#!/usr/bin/env python3
"""Is the LLM-judge problem ranking or calibration?

Platt scaling (logistic regression on logit p) and isotonic regression are fitted ONLY on training folds of
stratified, paper-grouped 5-fold CV (20 repetitions, same seeds as the harness) and applied to held-out papers.
Seed-averaged recalibrated predictions are scored with the same paper-clustered bootstrap as everything else.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'
SEEDS, N_BOOT = list(range(20)), 2000
MODELS = {'gpt-5.4-mini-2026-03-17': 'Mini', 'gpt-6.1-sol': 'Sol', 'claude-haiku-4-5-20251001': 'Haiku', 'claude-opus-5-5': 'Opus'}


def llm_preds(model, condition):
    rows = [json.loads(l) for l in (R / 'llm' / f'{model}.jsonl').read_text().splitlines() if l.strip()]
    d = pd.DataFrame(rows); d = d[(d.condition == condition) & d.parsed.notna()].sort_values('ts').drop_duplicates('claim_id', keep='last')
    return d.set_index('claim_id').parsed.map(lambda o: float(o['p_replicate']))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def recalibrate(p, y, g):
    out = {'platt': np.zeros((len(SEEDS), len(y))), 'isotonic': np.zeros((len(SEEDS), len(y)))}
    for si, seed in enumerate(SEEDS):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(p, y, g):
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p[tr])[:, None], y[tr])
            out['platt'][si, te] = lr.predict_proba(logit(p[te])[:, None])[:, 1]
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip').fit(p[tr], y[tr])
            out['isotonic'][si, te] = iso.predict(p[te])
    return {k: v.mean(0) for k, v in out.items()}


def main():
    oof = pd.read_csv(R / 'oof_predictions_seedavg.csv').set_index('claim_id')
    y_all = oof.y; res = {}
    for model, tag in MODELS.items():
        for cond in ['forecast_id', 'forecast_deid']:
            raw = llm_preds(model, cond).reindex(oof.index)
            assert raw.notna().all(), 'missing LLM predictions'
            p, y, g = raw.to_numpy(), y_all.to_numpy(), oof.paper_id.to_numpy()
            rc = recalibrate(p, y, g)
            P = {'raw': p, 'platt': rc['platt'], 'isotonic': rc['isotonic'],
                 'B1': oof['B1 Original-study metadata (LR)'].to_numpy(), 'B0': oof['B0 Prevalence'].to_numpy()}
            uniq = np.unique(g); pos = {k: np.flatnonzero(g == k) for k in uniq}; rng = np.random.default_rng(13)
            B = {k: [] for k in P}
            for _ in tqdm(range(N_BOOT), desc=f'{tag} {cond}', leave=False):
                ii = np.concatenate([pos[k] for k in rng.choice(uniq, len(uniq))])
                for k, v in P.items():
                    B[k].append((roc_auc_score(y[ii], v[ii]), brier_score_loss(y[ii], v[ii])))
            B = {k: np.array(v) for k, v in B.items()}
            ci = lambda a: [float(np.quantile(a, .025)), float(np.quantile(a, .975))]
            entry = {}
            for k in ['raw', 'platt', 'isotonic']:
                entry[k] = {'auroc': float(roc_auc_score(y, P[k])), 'brier': float(brier_score_loss(y, P[k])),
                            'bss_vs_B0': float(1 - brier_score_loss(y, P[k]) / brier_score_loss(y, P['B0'])),
                            'd_brier_vs_B1': float(brier_score_loss(y, P[k]) - brier_score_loss(y, P['B1'])),
                            'd_brier_vs_B1_ci': ci(B[k][:, 1] - B['B1'][:, 1]),
                            'd_brier_vs_B0_ci': ci(B[k][:, 1] - B['B0'][:, 1]),
                            'pred_range': [float(P[k].min()), float(P[k].max())]}
            entry['B1_brier'] = float(brier_score_loss(y, P['B1'])); entry['B0_brier'] = float(brier_score_loss(y, P['B0']))
            res[f'{model}|{cond}'] = entry
            print(f"{tag:4s} {cond:14s} " + '  '.join(f"{k}: AUROC {entry[k]['auroc']:.3f} Brier {entry[k]['brier']:.4f} "
                                                   f"dB1 {entry[k]['d_brier_vs_B1']:+.4f} {[round(x, 4) for x in entry[k]['d_brier_vs_B1_ci']]}"
                                                   for k in ['raw', 'platt', 'isotonic']), flush=True)
    (R / 'llm_recalibration.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
