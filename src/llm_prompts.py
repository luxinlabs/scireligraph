#!/usr/bin/env python3
"""Prompts for the LLM-as-judge verifier (forecast) and the memorization probe.

Only Tier A/T information (available at original publication) enters the forecast prompt.
The probe asks directly whether the model knows the SCORE replication and its outcome.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, pyreadr

ROOT = Path(__file__).resolve().parents[1]

FORECAST_SYSTEM = (
    "You are an expert in research methodology and replicability. Assume today's date is {cutoff}. "
    "You will see a published social or behavioural science finding. Estimate the probability that an independent, "
    "high-powered replication of the focal claim, conducted after {cutoff}, would succeed (a statistically significant "
    "effect in the same direction as the original). Base your estimate only on the information provided and general "
    "methodological knowledge. Do not use any knowledge of specific replication attempts of this paper. "
    'Respond with JSON only: {{"p_replicate": <number between 0 and 1>, "rationale": "<at most 25 words>"}}')

PROBE_SYSTEM = (
    "Answer from your own knowledge only. Respond with JSON only: "
    '{"aware_of_replication": "yes"|"no"|"unsure", "recalled_outcome": "replicated"|"not_replicated"|"mixed"|"unknown", '
    '"confidence": <number between 0 and 1>, "source": "<at most 20 words>"}')


def records():
    """One record per SCORE claim, built from the frozen archive (no replication information)."""
    r = pyreadr.read_r(str(ROOT / 'data/raw/analysis_package/analyst data.RData'))
    claims = pd.read_csv(ROOT / 'data/processed/claims.csv')
    oo = r['orig_outcomes'].set_index('claim_id').loc[claims.claim_id]
    pm = r['paper_metadata'].set_index('paper_id').loc[claims.paper_id]
    out = []
    for i, c in claims.reset_index(drop=True).iterrows():
        o, p = oo.iloc[i], pm.iloc[i]
        def v(x, fmt='{}'):
            return 'not reported' if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)
        stats = {
            'analysis': v(o.orig_analysis_type), 'test statistic': v(o.orig_stat_type),
            'reported evidence': v(o.orig_effect_size_text),
            'p-value': v(o.orig_p_value, '{:.3g}') + (f' ({o.orig_p_value_type})' if isinstance(o.orig_p_value_type, str) else ''),
            'sample size': v(o.orig_sample_size_value, '{:.0f}') + (f' {o.orig_sample_size_units}' if isinstance(o.orig_sample_size_units, str) else ''),
            'effect size (r)': v(o.orig_conv_r, '{:.3f}'), 'interaction effect': v(o.orig_stat_interaction)}
        out.append({'claim_id': c.claim_id, 'paper_id': c.paper_id, 'title': p.title, 'citation': p.citation,
                    'journal': p.publication_standard, 'year': int(p.pub_year), 'field': p.COS_pub_category,
                    'abstract': p.abstract if isinstance(p.abstract, str) else '', 'stats': stats,
                    'launch_date': c.launch_date if isinstance(c.launch_date, str) else None})
    return out


def decoy_records(n=100, seed=2026):
    """Probe decoys: papers SCORE sampled but never attempted to replicate/reproduce (no replication, reproduction or
    hybrid entry anywhere in the archive), non-COVID, drawn to match the journal mix of the 164 replicated papers.
    Correct probe answer for a decoy: no replication attempt."""
    r = pyreadr.read_r(str(ROOT / 'data/raw/analysis_package/analyst data.RData'))
    pm = r['paper_metadata']
    touched = (set(r['rr_type_key'].paper_id) | set(r['repli_all_cases'].paper_id) | set(r['repli_outcomes'].paper_id)
               | set(r['rr_internal_tracking'].paper_id) | set(r['orig_outcomes'].paper_id))
    claims = pd.read_csv(ROOT / 'data/processed/claims.csv')
    jw = claims.drop_duplicates('paper_id').journal.value_counts(normalize=True)
    pool = pm[~pm.paper_id.isin(touched) & ~pm.is_covid & pm.publication_standard.isin(jw.index)]
    w = pool.publication_standard.map(jw) / pool.groupby('publication_standard').paper_id.transform('size')
    pick = pool.sample(n=n, weights=w, random_state=seed)
    return [{'claim_id': f'decoy_{p.paper_id}', 'paper_id': p.paper_id, 'title': p.title, 'citation': p.citation,
             'journal': p.publication_standard, 'year': int(p.pub_year), 'decoy': True} for p in pick.itertuples()]


def forecast_messages(rec, cutoff='2018-12-31'):
    body = (f"Title: {rec['title']}\nCitation: {rec['citation']}, {rec['journal']} ({rec['year']}); field: {rec['field']}\n"
            f"Abstract: {rec['abstract']}\n\nFocal claim (original statistics):\n"
            + '\n'.join(f'- {k}: {v}' for k, v in rec['stats'].items()))
    return [{'role': 'system', 'content': FORECAST_SYSTEM.format(cutoff=cutoff)}, {'role': 'user', 'content': body}]


def forecast_deid_messages(rec, cutoff='2018-12-31'):
    """De-identified forecast: field and original statistics only (no title, authors, journal, year, abstract or
    table locator), so a correct answer cannot come from recalling this specific paper."""
    stats = {k: v for k, v in rec['stats'].items() if k != 'reported evidence'}
    body = (f"Field: {rec['field']}\n\nFocal claim (original statistics):\n" + '\n'.join(f'- {k}: {v}' for k, v in stats.items()))
    return [{'role': 'system', 'content': FORECAST_SYSTEM.format(cutoff=cutoff)}, {'role': 'user', 'content': body}]


def probe_messages(rec):
    body = (f"Paper: \"{rec['title']}\" ({rec['citation']}, {rec['journal']}, {rec['year']}).\n"
            "Was a claim from this paper the subject of a replication attempt in DARPA's SCORE program / the Center for "
            "Open Science's large-scale replication project of social and behavioural science papers? If so, did it replicate?")
    return [{'role': 'system', 'content': PROBE_SYSTEM}, {'role': 'user', 'content': body}]


if __name__ == '__main__':
    recs = records()
    print(json.dumps(forecast_messages(recs[0]), indent=1)[:2500])
    print(json.dumps(probe_messages(recs[0]), indent=1))
