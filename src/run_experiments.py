#!/usr/bin/env python3
"""E1-E5: leakage-controlled replication forecasting on SCORE (274 claims / 164 papers).

Protocol: stratified, paper-grouped 5-fold CV repeated over SEEDS; all imputation, scaling, text
vectorisation and the L2 strength (inner grouped 3-fold, log-loss) are fitted inside the training fold.
Uncertainty: paper-clustered bootstrap over seed-averaged out-of-fold predictions.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'data/processed'
RES = ROOT / 'results'
SEEDS = list(range(20))
N_BOOT = 2000
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

man = json.loads((P / 'manifest.json').read_text())
T = man['feature_tiers']
A_NUM = ['a_log10_n', 'a_neglog10_p', 'a_p_marginal', 'a_p_exact', 'a_abs_r', 'a_power50', 'a_interaction', 'a_pub_year']
A_CAT = ['a_stat_type', 'a_analysis', 'a_field']
FLAT, GRAPH, DEP, D, D_CAT, DX = T['G_flat'], T['G_graph'], T['G_dependency'], T['D'], T['D_cat'], T['D_achieved']


def load():
    df = pd.read_csv(P / 'claims.csv')
    df['text'] = pd.read_csv(P / 'claims_text.csv').text.fillna('')
    graphs = {k: pd.read_csv(P / f'graph_{k}.csv').drop(columns='claim_id') for k in ['strict', 'leaky', 'launch']}
    return df, graphs


def tabular_lr(num, cat):
    pre = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), num),
        ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), cat)])
    return Pipeline([('pre', pre), ('lr', LogisticRegression(max_iter=5000))])


def tabular_hgb(num, cat):
    pre = ColumnTransformer([('num', 'passthrough', num),
                             ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5, sparse_output=False), cat)])
    # fixed a-priori small-data settings; not tuned on held-out folds
    return Pipeline([('pre', pre), ('m', HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150,
                                                                       min_samples_leaf=20, l2_regularization=1.0, random_state=0))])


def text_lr():
    pre = ColumnTransformer([('txt', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                                                     stop_words='english'), 'text')])
    return Pipeline([('pre', pre), ('lr', LogisticRegression(max_iter=5000))])


def fit_predict(spec, Xtr, ytr, gtr, Xte, seed):
    kind = spec['kind']
    if kind == 'prior':
        return np.full(len(Xte), ytr.mean())
    if kind == 'hgb':
        m = tabular_hgb(spec['num'], spec['cat']).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    m = text_lr() if kind == 'text' else tabular_lr(spec['num'], spec['cat'])
    inner = list(GroupKFold(n_splits=3).split(Xtr, ytr, gtr))
    gs = GridSearchCV(m, {'lr__C': CS}, scoring='neg_log_loss', cv=inner, refit=True).fit(Xtr, ytr)
    return gs.predict_proba(Xte)[:, 1]


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {'auroc': roc_auc_score(y, p), 'brier': brier_score_loss(y, p), 'log_loss': log_loss(y, p),
            'ap': average_precision_score(y, p), 'ece': ece(y, p)}


def run(df, X, y, groups, spec, desc=None):
    """Returns per-seed metrics and the seed-averaged OOF prediction vector."""
    allp, per_seed = np.zeros((len(SEEDS), len(y))), []
    for si, seed in enumerate(tqdm(SEEDS, desc=desc, leave=False, disable=desc is None)):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in cv.split(X, y, groups):
            allp[si, te] = fit_predict(spec, X.iloc[tr], y[tr], groups[tr], X.iloc[te], seed)
        per_seed.append(metrics(y, allp[si]))
    return pd.DataFrame(per_seed), allp.mean(0)


def cluster_boot_idx(groups, rng):
    uniq = np.unique(groups); pick = rng.choice(uniq, len(uniq), replace=True)
    pos = {g: np.flatnonzero(groups == g) for g in uniq}
    return np.concatenate([pos[g] for g in pick])


def main():
    df, G = load()
    y = df.replicated.to_numpy(); groups = df.paper_id.to_numpy()
    Xbase = pd.concat([df, G['strict']], axis=1)
    Xleak = pd.concat([df, G['leaky']], axis=1)
    Xlaunch = pd.concat([df, G['launch']], axis=1)
    # Negative control: graph+dependency features permuted across PAPERS within publication-year strata
    rng = np.random.default_rng(12345)
    Xperm = Xbase.copy()
    pap = df.groupby('paper_id').a_pub_year.first()
    perm_map = {}
    for yr, ids in pap.groupby(pap).groups.items():
        ids = list(ids); perm_map.update(dict(zip(ids, rng.permutation(ids))))
    gtab = pd.concat([df[['paper_id']], G['strict']], axis=1).groupby('paper_id').first()
    for c in GRAPH + DEP: Xperm[c] = df.paper_id.map(lambda p: gtab.loc[perm_map[p], c]).to_numpy()

    specs = {
        'B0 Prevalence': ('base', {'kind': 'prior'}),
        'B1 Original-study metadata (LR)': ('base', {'kind': 'lr', 'num': A_NUM, 'cat': A_CAT}),
        'B1g Original-study metadata (GBT)': ('base', {'kind': 'hgb', 'num': A_NUM, 'cat': A_CAT}),
        'B2 Title/abstract TF-IDF (LR)': ('base', {'kind': 'text'}),
        'B4 Metadata + flat citation evidence': ('base', {'kind': 'lr', 'num': A_NUM + FLAT, 'cat': A_CAT}),
        'M1 Metadata + evidence graph': ('base', {'kind': 'lr', 'num': A_NUM + FLAT + GRAPH, 'cat': A_CAT}),
        'M2 Metadata + graph + dependency correction': ('base', {'kind': 'lr', 'num': A_NUM + FLAT + GRAPH + DEP, 'cat': A_CAT}),
        'G-only Graph + dependency (no metadata)': ('base', {'kind': 'lr', 'num': FLAT + GRAPH + DEP, 'cat': []}),
        'NC M2 with permuted graph (negative control)': ('perm', {'kind': 'lr', 'num': A_NUM + FLAT + GRAPH + DEP, 'cat': A_CAT}),
        'L-B4 Flat evidence, NO cutoff (leaky)': ('leak', {'kind': 'lr', 'num': A_NUM + FLAT, 'cat': A_CAT}),
        'L-M2 Graph + dependency, NO cutoff (leaky)': ('leak', {'kind': 'lr', 'num': A_NUM + FLAT + GRAPH + DEP, 'cat': A_CAT}),
        'LG-only Graph, NO cutoff (leaky)': ('leak', {'kind': 'lr', 'num': FLAT + GRAPH + DEP, 'cat': []}),
        'S-M2 Graph at replication-launch cutoff': ('launch', {'kind': 'lr', 'num': A_NUM + FLAT + GRAPH + DEP, 'cat': A_CAT}),
        'D Metadata + replication design (launch-time)': ('base', {'kind': 'lr', 'num': A_NUM + D, 'cat': A_CAT + D_CAT}),
        'Dx Metadata + achieved replication size (post-launch)': ('base', {'kind': 'lr', 'num': A_NUM + D + DX, 'cat': A_CAT + D_CAT}),
    }
    frames = {'base': Xbase, 'leak': Xleak, 'launch': Xlaunch, 'perm': Xperm}
    RES.mkdir(exist_ok=True)
    summary, oof = {}, {}
    for name, (fr, spec) in tqdm(specs.items(), desc='configurations', total=len(specs)):
        per_seed, pbar = run(df, frames[fr], y, groups, spec, desc=name[:28])
        oof[name] = pbar
        summary[name] = {'mean': per_seed.mean().to_dict(), 'sd': per_seed.std().to_dict(), 'pooled': metrics(y, pbar)}
        tqdm.write(f"{name:48s} AUROC {per_seed.auroc.mean():.3f}±{per_seed.auroc.std():.3f}  Brier {per_seed.brier.mean():.4f}  LL {per_seed.log_loss.mean():.4f}")

    # Paper-clustered bootstrap CIs (seed-averaged OOF) + paired deltas
    rng = np.random.default_rng(7)
    comps = [('B1 Original-study metadata (LR)', 'B0 Prevalence'),
             ('B4 Metadata + flat citation evidence', 'B1 Original-study metadata (LR)'),
             ('M1 Metadata + evidence graph', 'B4 Metadata + flat citation evidence'),
             ('M2 Metadata + graph + dependency correction', 'M1 Metadata + evidence graph'),
             ('M2 Metadata + graph + dependency correction', 'B1 Original-study metadata (LR)'),
             ('M2 Metadata + graph + dependency correction', 'NC M2 with permuted graph (negative control)'),
             ('L-M2 Graph + dependency, NO cutoff (leaky)', 'M2 Metadata + graph + dependency correction'),
             ('LG-only Graph, NO cutoff (leaky)', 'G-only Graph + dependency (no metadata)'),
             ('D Metadata + replication design (launch-time)', 'B1 Original-study metadata (LR)'),
             ('Dx Metadata + achieved replication size (post-launch)', 'D Metadata + replication design (launch-time)'),
             ('B2 Title/abstract TF-IDF (LR)', 'B0 Prevalence')]
    boots = {n: [] for n in specs}; dboots = {c: [] for c in comps}
    for _ in tqdm(range(N_BOOT), desc='bootstrap'):
        idx = cluster_boot_idx(groups, rng)
        if len(np.unique(y[idx])) < 2: continue
        m = {n: metrics(y[idx], oof[n][idx]) for n in specs}
        for n in specs: boots[n].append(m[n])
        for a, b in comps: dboots[(a, b)].append({k: m[a][k] - m[b][k] for k in ('auroc', 'brier', 'log_loss')})
    for n in specs:
        bt = pd.DataFrame(boots[n])
        summary[n]['ci95'] = {k: [float(bt[k].quantile(.025)), float(bt[k].quantile(.975))] for k in bt}
    deltas = []
    for (a, b), rows in dboots.items():
        bt = pd.DataFrame(rows); pt = {k: metrics(y, oof[a])[k] - metrics(y, oof[b])[k] for k in bt}
        deltas.append({'a': a, 'b': b, **{f'd_{k}': pt[k] for k in bt},
                       **{f'd_{k}_ci': [float(bt[k].quantile(.025)), float(bt[k].quantile(.975))] for k in bt},
                       'p_auroc_le0': float((bt.auroc <= 0).mean()),
                       'p_two_sided': float(min(1.0, 2 * min((bt.auroc <= 0).mean(), (bt.auroc >= 0).mean())))})
    # Holm correction across the pre-specified comparison family
    order = np.argsort([d['p_two_sided'] for d in deltas]); m_ = len(deltas); run_max = 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, min(1.0, (m_ - rank) * deltas[i]['p_two_sided'])); deltas[i]['p_holm'] = run_max

    # Sensitivity: alternative replication criteria (B1 and M2, strict)
    sens = {}
    for lab in ['alt_analyst', 'alt_rep_wthn', 'alt_bayes_rep']:
        ok = df[lab].notna().to_numpy(); yl = df[lab].to_numpy()[ok].astype(int)
        for name in ['B0 Prevalence', 'B1 Original-study metadata (LR)', 'M2 Metadata + graph + dependency correction']:
            fr, spec = specs[name]
            ps, _ = run(df[ok].reset_index(drop=True), frames[fr][ok].reset_index(drop=True), yl, groups[ok], spec)
            sens[f'{lab} | {name}'] = {'n': int(ok.sum()), 'rate': float(yl.mean()), **ps.mean().to_dict()}
            tqdm.write(f'sens {lab} {name} {ps.auroc.mean():.3f}')

    # Sensitivity: paper-weighted metrics (each paper weight 1, split over its claims), as in the SCORE analysis
    w = 1.0 / df.groupby('paper_id').paper_id.transform('size').to_numpy()
    paper_weighted = {}
    for name in specs:
        pp = np.clip(oof[name], 1e-6, 1 - 1e-6)
        paper_weighted[name] = {'auroc': float(roc_auc_score(y, pp, sample_weight=w)),
                                'brier': float(brier_score_loss(y, pp, sample_weight=w)),
                                'base_rate': float(np.average(y, weights=w))}

    # Subgroups (seed-averaged OOF from M2 and B1)
    sub = []
    for col in ['a_field', 'repli_type']:
        for val, g in df.groupby(col):
            ix = g.index.to_numpy()
            if len(np.unique(y[ix])) < 2 or len(ix) < 15: continue
            for name in ['B1 Original-study metadata (LR)', 'M2 Metadata + graph + dependency correction']:
                sub.append({'group': col, 'value': val, 'n': len(ix), 'rate': float(y[ix].mean()), 'model': name, **metrics(y[ix], oof[name][ix])})

    # Descriptive single-feature AUROCs (full sample; direction-free), strict and leaky
    feat = []
    for tag, X in [('strict', Xbase), ('leaky', Xleak)]:
        for c in A_NUM + FLAT + GRAPH + DEP + D + DX:
            if tag == 'leaky' and c not in FLAT + GRAPH + DEP: continue
            v = X[c].to_numpy(dtype=float); ok = ~np.isnan(v)
            if ok.sum() < 30 or np.nanstd(v) == 0: continue
            a = roc_auc_score(y[ok], v[ok])
            feat.append({'cutoff': tag, 'feature': c, 'n': int(ok.sum()), 'auroc': a, 'auroc_abs': max(a, 1 - a),
                         'mean_rep': float(np.nanmean(v[ok][y[ok] == 1])), 'mean_nonrep': float(np.nanmean(v[ok][y[ok] == 0]))})

    # Leakage audit: how much graph evidence each cutoff admits
    audit = {k: {c: float(G[k][c].mean()) for c in ['g_log_cites', 'g_has_evidence', 'g_selfcite_frac', 'g_log_indep_groups']} for k in G}

    pd.DataFrame(oof).assign(claim_id=df.claim_id, paper_id=df.paper_id, y=y).to_csv(RES / 'oof_predictions_seedavg.csv', index=False)
    pd.DataFrame(deltas).to_json(RES / 'paired_deltas.json', orient='records', indent=2)
    pd.DataFrame(sub).to_csv(RES / 'subgroups.csv', index=False)
    pd.DataFrame(feat).sort_values(['cutoff', 'auroc_abs'], ascending=[True, False]).to_csv(RES / 'feature_auroc.csv', index=False)
    (RES / 'summary.json').write_text(json.dumps({'n_claims': int(len(df)), 'n_papers': int(df.paper_id.nunique()),
        'positives': int(y.sum()), 'seeds': len(SEEDS), 'n_boot': N_BOOT, 'models': summary, 'sensitivity': sens,
        'graph_audit': audit, 'paper_weighted': paper_weighted}, indent=2, default=float))
    print(json.dumps(deltas, indent=1, default=float))
    print(json.dumps(audit, indent=1))


if __name__ == '__main__':
    main()
