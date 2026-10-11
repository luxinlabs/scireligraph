#!/usr/bin/env python3
"""Score the LLM-as-judge runs with the same protocol as the harness (paper-clustered bootstrap, paired vs. B1).

Outputs results/llm_summary.json. Claims whose response failed to parse are reported, never imputed.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'
N_BOOT = 2000


def latest(model_file):
    rows = [json.loads(l) for l in model_file.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df = df[df.parsed.notna()].sort_values('ts').drop_duplicates(['claim_id', 'condition'], keep='last')
    return df, rows


def boot_idx(groups, rng):
    uniq = np.unique(groups); pos = {g: np.flatnonzero(groups == g) for g in uniq}
    return [np.concatenate([pos[g] for g in rng.choice(uniq, len(uniq))]) for _ in range(N_BOOT)]


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def main():
    oof = pd.read_csv(R / 'oof_predictions_seedavg.csv').set_index('claim_id')
    b1 = 'B1 Original-study metadata (LR)'
    out = {}
    for f in sorted((R / 'llm').glob('*.jsonl')):
        model = f.stem; df, rows = latest(f)
        res = {'calls_logged': len(rows), 'cost_usd': float(sum(r.get('cost_usd') or 0 for r in rows)),
               'parse_or_api_failures': int(sum(1 for r in rows if r.get('parsed') is None)),
               'mean_reasoning_tokens': float(np.nanmean([r['reasoning_tokens'] for r in rows if r.get('reasoning_tokens') is not None] or [np.nan])),
               'params': next((r.get('params') for r in rows if r.get('params') is not None), None)}
        fc = {}
        for cond in ['forecast_id', 'forecast_deid']:
            d = df[df.condition == cond].copy()
            d['p'] = d.parsed.map(lambda o: float(o['p_replicate']))
            d = d[d.claim_id.isin(oof.index)]
            fc[cond] = d.set_index('claim_id').p
        common = sorted(set(fc['forecast_id'].index) & set(fc['forecast_deid'].index))
        if len(common) >= 10:
            y = oof.loc[common, 'y'].to_numpy(); g = oof.loc[common, 'paper_id'].to_numpy()
            P = {'id': fc['forecast_id'].loc[common].to_numpy(), 'deid': fc['forecast_deid'].loc[common].to_numpy(),
                 'B1': oof.loc[common, b1].to_numpy(), 'B0': oof.loc[common, 'B0 Prevalence'].to_numpy()}
            pt = {k: {'auroc': safe_auc(y, v), 'brier': brier_score_loss(y, np.clip(v, 0, 1))} for k, v in P.items()}
            idxs = boot_idx(g, np.random.default_rng(11)); B = {k: [] for k in P}
            for ii in tqdm(idxs, desc=f'bootstrap {model}', leave=False):
                for k, v in P.items():
                    B[k].append((safe_auc(y[ii], v[ii]), brier_score_loss(y[ii], np.clip(v[ii], 0, 1))))
            B = {k: np.array(v) for k, v in B.items()}
            ci = lambda a: [float(np.nanquantile(a, .025)), float(np.nanquantile(a, .975))]
            res['n_forecast'] = len(common)
            res['forecast'] = {k: {**{m: float(x) for m, x in pt[k].items()}, 'auroc_ci': ci(B[k][:, 0]), 'brier_ci': ci(B[k][:, 1])} for k in P}
            for a, b in [('id', 'B1'), ('deid', 'B1'), ('id', 'deid')]:
                dd = B[a] - B[b]
                res[f'delta_{a}_vs_{b}'] = {'d_auroc': float(pt[a]['auroc'] - pt[b]['auroc']), 'd_auroc_ci': ci(dd[:, 0]),
                                            'd_brier': float(pt[a]['brier'] - pt[b]['brier']), 'd_brier_ci': ci(dd[:, 1])}
            def top10(p, yy):
                top = np.argsort(-np.abs(p - 0.5), kind='stable')[: max(1, round(0.1 * len(p)))]
                return ((p[top] >= .5) == yy[top]).mean(), ((p >= .5) == yy).mean(), (p[top] >= .5).mean()
            for k in ('id', 'deid', 'B1'):
                a10, aall, pos = top10(P[k], y)
                bs = np.array([top10(P[k][ii], y[ii])[:2] for ii in idxs])
                res.setdefault('selective_top10', {})[k] = {'acc_top10': float(a10), 'acc_all': float(aall), 'frac_pos_top10': float(pos),
                                                             'acc_top10_ci': ci(bs[:, 0]), 'gain_top10_minus_all_ci': ci(bs[:, 0] - bs[:, 1]),
                                                             'pred_spread_sd': float(P[k].std())}
        # memorization probe
        pr = df[df.condition == 'probe'].copy()
        pr['aware'] = pr.parsed.map(lambda o: o.get('aware_of_replication')); pr['recalled'] = pr.parsed.map(lambda o: o.get('recalled_outcome'))
        real, dec = pr[~pr.claim_id.str.startswith('decoy_')], pr[pr.claim_id.str.startswith('decoy_')]
        real = real[real.claim_id.isin(oof.index)].copy(); real['y'] = oof.loc[real.claim_id, 'y'].to_numpy()
        named = real[real.recalled.isin(['replicated', 'not_replicated'])]
        def wilson(k, n, z=1.96):
            if n == 0: return [None, None]
            ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
            return [float(c - h), float(c + h)]
        res['probe'] = {
            'aware_yes_real_ci': wilson(int((real.aware == 'yes').sum()), len(real)),
            'aware_yes_decoy_ci': wilson(int((dec.aware == 'yes').sum()), len(dec)),
            'n_real': len(real), 'n_decoy': len(dec),
            'aware_yes_real': float((real.aware == 'yes').mean()) if len(real) else None,
            'aware_yes_decoy': float((dec.aware == 'yes').mean()) if len(dec) else None,
            'named_outcome_real': len(named),
            'named_outcome_accuracy': float(((named.recalled == 'replicated').astype(int) == named.y).mean()) if len(named) else None,
            # majority-guess baseline on the same claims: accuracy of always naming the more common outcome
            'named_outcome_majority_baseline': float(max(named.y.mean(), 1 - named.y.mean())) if len(named) else None,
            'named_outcome_decoy': int(dec.recalled.isin(['replicated', 'not_replicated']).sum()) if len(dec) else None}
        out[model] = res
        print(json.dumps({model: res}, indent=1, default=float))
    (R / 'llm_summary.json').write_text(json.dumps(out, indent=2, default=float))


if __name__ == '__main__':
    main()
