#!/usr/bin/env python3
"""Build the claim-level modelling table from the frozen SCORE analyst package + OpenAlex cache.

Every feature carries an availability tier (see FEATURE_TIERS in the output manifest):
  A  original-study statistics and paper metadata (known at original publication)
  T  title/abstract/keywords text (known at original publication)
  G  bibliographic evidence graph, recomputed for each evidence cutoff
  D  replication design (known when the replication is launched, before its outcome)
Replication results (p-values, effects, success flags) are used ONLY as labels.
"""
import hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, pyreadr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/analysis_package/analyst data.RData'
OA = ROOT / 'data/openalex'
OUT = ROOT / 'data/processed'

# Global strict cutoff: SCORE replications were not launched before 2019 (earliest project start in
# rr_project_dates is checked below), so evidence dated <= 2018-12-31 cannot reflect any SCORE outcome.
STRICT_CUTOFF = '2018-12-31'
LEAKY_CUTOFF = '2100-01-01'  # everything OpenAlex holds as of the access date


def analysis_family(s):
    s = str(s).lower()
    if 'regress' in s or 'equation' in s or 'probit' in s or 'logit' in s or 'panel' in s: return 'regression'
    if 'anova' in s or 't-test' in s or 'mean comparison' in s or 'ancova' in s: return 'mean_comparison'
    if 'correlation' in s: return 'correlation'
    return 'other'


def oa_stem(pid):
    """Case-safe cache filename: SCORE paper ids differ only by case (e.g. 'RJb' vs 'rjb') and macOS/Windows
    filesystems are case-insensitive, so the raw id alone would make two papers share one file."""
    return f"{pid}_{hashlib.sha1(pid.encode()).hexdigest()[:8]}"


def load_openalex(pid):
    wf, cf = OA / 'works' / f'{oa_stem(pid)}.json', OA / 'citing' / f'{oa_stem(pid)}.jsonl'
    w = json.loads(wf.read_text()) if wf.exists() else None
    citing = [json.loads(l) for l in cf.read_text().splitlines() if l.strip()] if cf.exists() else []
    return w, citing


def authors(work):
    return {a['author']['id'] for a in (work.get('authorships') or []) if a.get('author') and a['author'].get('id')}


def primary_field(work):
    t = work.get('topics') or []
    return t[0]['field']['id'] if t and t[0].get('field') else None


def entropy(items):
    c = Counter(i for i in items if i)
    n = sum(c.values())
    return -sum(v / n * math.log(v / n) for v in c.values()) if n else 0.0


def graph_features(w, citing, cutoff, pub_year):
    """Claim-centred evidence graph G(c,t): original paper + citing works dated <= cutoff.

    Edges: cites (citer->original, citer->citer), shares-author (dependency), shares-reference (coupling).
    """
    cut_year = int(cutoff[:4])
    C = [x for x in citing if x.get('publication_date') and x['publication_date'] <= cutoff]
    k = len(C)
    orig_auth = authors(w)
    orig_refs = set(w.get('referenced_works') or [])
    self_mask = [bool(authors(x) & orig_auth) for x in C]
    others = [x for x, s in zip(C, self_mask) if not s]
    # dependency clusters: union-find over non-self citers linked by any shared author
    parent = list(range(len(others)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    first_seen = {}
    for i, x in enumerate(others):
        for a in authors(x):
            if a in first_seen: parent[find(i)] = find(first_seen[a])
            else: first_seen[a] = i
    n_groups = len({find(i) for i in range(len(others))})
    # citer-citer citation density (echo-chamber structure)
    ids = {x['id'] for x in C}
    internal = sum(len(ids & set(x.get('referenced_works') or [])) for x in C)
    density = internal / (k * (k - 1) / 2) if k > 1 else 0.0
    # bibliographic coupling of citers with the original paper
    coup = [len(orig_refs & set(x.get('referenced_works') or [])) / max(1, len(orig_refs | set(x.get('referenced_works') or [])))
            for x in C] if orig_refs else []
    ofield = primary_field(w)
    fields = [primary_field(x) for x in C]
    venues = [((x.get('primary_location') or {}).get('source') or {}).get('id') for x in C]
    years = max(1, cut_year - int(pub_year) + 1)
    return {
        # flat evidence
        'g_log_cites': math.log1p(k),
        'g_log_cites_per_year': math.log1p(k / years),
        'g_log_refs': math.log1p(len(orig_refs)),
        # graph / dependency-aware evidence
        'g_selfcite_frac': (sum(self_mask) / k) if k else 0.0,
        'g_log_indep_groups': math.log1p(n_groups),
        'g_indep_ratio': (n_groups / len(others)) if others else 0.0,
        'g_log_citing_authors': math.log1p(len(set().union(*[authors(x) for x in others])) if others else 0),
        'g_venue_entropy': entropy(venues),
        'g_field_count': len({f for f in fields if f}),
        'g_cross_field_frac': (sum(1 for f in fields if f and f != ofield) / k) if k else 0.0,
        'g_internal_density': density,
        'g_coupling_mean': float(np.mean(coup)) if coup else 0.0,
        'g_has_evidence': float(k > 0),
    }


FLAT_G = ['g_log_cites', 'g_log_cites_per_year', 'g_log_refs']
GRAPH_G = ['g_venue_entropy', 'g_field_count', 'g_cross_field_frac', 'g_internal_density', 'g_coupling_mean', 'g_has_evidence']
DEP_G = ['g_selfcite_frac', 'g_log_indep_groups', 'g_indep_ratio', 'g_log_citing_authors']


def main():
    r = pyreadr.read_r(str(RAW))
    ro, oo, pm, rb = r['repli_outcomes'], r['orig_outcomes'], r['paper_metadata'], r['repli_binary']
    dates = r['rr_project_dates']
    rep = ro[ro.repli_version_of_record & ~ro.is_covid].copy()

    # ---------- E0 validation against Tyner et al. (2026) ----------
    checks = {'n_claims': (rep.claim_id.nunique(), 274), 'n_papers': (rep.paper_id.nunique(), 164),
              'n_replicated': (int(rep.repli_score_criteria_met.sum()), 151), 'rows': (len(rep), 274)}
    for k, (got, want) in checks.items():
        assert got == want, f'E0 failed: {k} = {got}, expected {want}'

    df = rep[['claim_id', 'paper_id', 'report_id', 'rr_id', 'repli_score_criteria_met', 'repli_type',
              'repli_sample_size_value', 'combined_power_50', 'combined_power_75', 'is_manylabs',
              'power_for_effect_size']].copy()
    df['replicated'] = df.pop('repli_score_criteria_met').astype(int)
    alt = rb.set_index('report_id')[['repli_binary_analyst', 'repli_binary_rep_wthn', 'repli_binary_bayes_rep']]
    for c in alt.columns:
        df['alt_' + c.replace('repli_binary_', '')] = df.report_id.map(alt[c]).map({True: 1, False: 0})

    # ---------- Tier A: original-study statistics ----------
    o = oo.set_index('claim_id').loc[df.claim_id]
    df['a_log10_n'] = np.log10(o.orig_sample_size_value.clip(lower=1)).to_numpy()
    p = o.orig_p_value.clip(lower=1e-300).to_numpy()
    df['a_neglog10_p'] = np.minimum(-np.log10(p), 50)
    df['a_p_marginal'] = ((p > 0.01) & (p <= 0.05)).astype(float)
    df.loc[np.isnan(p), 'a_p_marginal'] = np.nan
    df['a_p_exact'] = (o.orig_p_value_type.to_numpy() == 'exact').astype(float)
    df['a_abs_r'] = np.abs(o.orig_conv_r.to_numpy())
    df['a_power50'] = o.orig_power_for_50_effect.to_numpy()
    inter = o.orig_stat_interaction.str.lower().to_numpy()
    df['a_interaction'] = np.where(pd.isna(inter), np.nan, (inter == 'interaction').astype(float))
    df['a_stat_type'] = o.orig_stat_type.fillna('missing').replace({'chi_squared': 'other', 'non-parametric': 'other', 'delta_gamma_squared': 'other'}).to_numpy()
    df['a_analysis'] = [analysis_family(x) for x in o.orig_analysis_type]
    pmeta = pm.set_index('paper_id').loc[df.paper_id]
    df['a_pub_year'] = pmeta.pub_year.to_numpy()
    df['a_field'] = pmeta.COS_pub_category.to_numpy()
    df['journal'] = pmeta.publication_standard.to_numpy()
    df['doi'] = pmeta.DOI.to_numpy()
    # ---------- Tier T: text ----------
    df['text'] = (pmeta.title.fillna('') + '. ' + pmeta.abstract.fillna('') + ' ' + pmeta.keywords.fillna('')).to_numpy()
    # ---------- Tier D: replication design known at launch (planned, from the preregistered power target) ----------
    df['d_new_data'] = (df.repli_type == 'new data').astype(float)
    tier_col = {'50% for 100%': 'orig_sample_size_50_for_100', '90% for 75%': 'orig_sample_size_90_for_75',
                '90% for 50%': 'orig_sample_size_90_for_50'}
    tiers = df.power_for_effect_size.astype(str).to_numpy()
    planned = np.array([o.iloc[i][tier_col[t]] if t in tier_col else np.nan for i, t in enumerate(tiers)], dtype=float)
    df['d_log10_planned_n'] = np.log10(np.clip(planned, 1, None))
    df['d_power_target'] = tiers
    # ---------- Tier D*: achieved replication size/power (known only after data collection; sensitivity only) ----------
    df['dx_log10_rep_n'] = np.log10(df.repli_sample_size_value.clip(lower=1))
    df['dx_power50'] = df.combined_power_50
    d = dates.set_index('rr_id')
    df['launch_date'] = df.rr_id.map(d.started)
    earliest = pd.to_datetime(dates.started).min()
    assert earliest > pd.Timestamp(STRICT_CUTOFF), f'strict cutoff after a replication start ({earliest})'

    # ---------- Tier G: evidence graph per cutoff ----------
    cov = {'papers_found': 0, 'citing_works': 0}
    per_paper = {}
    for pid, year in df.groupby('paper_id').a_pub_year.first().items():
        w, citing = load_openalex(pid)
        if w is None: per_paper[pid] = None; continue
        cov['papers_found'] += 1; cov['citing_works'] += len(citing)
        per_paper[pid] = (w, citing, year)
    tables = {}
    for name in ['strict', 'leaky', 'launch']:
        rows = []
        for _, row in df.iterrows():
            rec = per_paper[row.paper_id]
            if name == 'launch':
                cutoff = str(row.launch_date) if isinstance(row.launch_date, str) and row.launch_date else STRICT_CUTOFF
                cutoff = (pd.Timestamp(cutoff) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                cutoff = STRICT_CUTOFF if name == 'strict' else LEAKY_CUTOFF
            if rec is None:
                rows.append({k: np.nan for k in FLAT_G + GRAPH_G + DEP_G})
            else:
                eff_cut = cutoff if name != 'leaky' else '2026-12-31'
                rows.append(graph_features(rec[0], rec[1], eff_cut, rec[2]))
        tables[name] = pd.DataFrame(rows, index=df.index)

    OUT.mkdir(parents=True, exist_ok=True)
    base = df.drop(columns=['text'])
    base.to_csv(OUT / 'claims.csv', index=False)
    df[['claim_id', 'text']].to_csv(OUT / 'claims_text.csv', index=False)
    for name, t in tables.items():
        pd.concat([df[['claim_id']], t], axis=1).to_csv(OUT / f'graph_{name}.csv', index=False)

    # baseline/ schema-compatible export for the original starter script
    pd.DataFrame({'claim_id': df.claim_id, 'paper_id': df.paper_id, 'replicated': df.replicated,
                  'original_n': o.orig_sample_size_value.to_numpy(), 'original_p': o.orig_p_value.to_numpy(),
                  'original_effect_size': df.a_abs_r, 'publication_year': df.a_pub_year,
                  'discipline': df.a_field, 'journal': df.journal}).to_csv(OUT / 'score_claims.csv', index=False)

    sha = hashlib.sha256((ROOT / 'data/raw/analysis_package.zip').read_bytes()).hexdigest() if (ROOT / 'data/raw/analysis_package.zip').exists() else None
    manifest = {
        'source': 'OSF g5sny "analysis data and code package 15 Dec 2025 1708.zip" (https://osf.io/download/5xfdy/)',
        'source_sha256': sha,
        'openalex_access': (OA / 'ACCESS_DATE.txt').read_text().strip() if (OA / 'ACCESS_DATE.txt').exists() else None,
        'label': 'repli_score_criteria_met (SCORE primary criterion), version-of-record, non-COVID',
        'validation': {k: {'observed': int(g), 'published': w} for k, (g, w) in checks.items()},
        'n_journals': int(df.journal.nunique()),
        'earliest_replication_start': str(earliest.date()),
        'strict_cutoff': STRICT_CUTOFF,
        'launch_cutoff_available': int(df.launch_date.notna().sum()),
        'openalex_coverage': {**cov, 'papers_total': int(df.paper_id.nunique())},
        'strict_citing_works': int(sum(1 for v in per_paper.values() if v for x in v[1] if (x.get('publication_date') or '9') <= STRICT_CUTOFF)),
        'feature_tiers': {'A': [c for c in df.columns if c.startswith('a_')], 'T': ['text'],
                          'G_flat': FLAT_G, 'G_graph': GRAPH_G, 'G_dependency': DEP_G,
                          'D': ['d_new_data', 'd_log10_planned_n'], 'D_cat': ['d_power_target'],
                          'D_achieved': ['dx_log10_rep_n', 'dx_power50']},
        'missingness': {c: int(df[c].isna().sum()) for c in df.columns if c[:2] in ('a_', 'd_', 'dx')},
        'planned_vs_achieved_n_median_ratio': float(np.nanmedian(df.repli_sample_size_value / planned)),
        'alt_label_rates': {c: float(df[c].mean()) for c in df.columns if c.startswith('alt_')},
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == '__main__':
    main()
