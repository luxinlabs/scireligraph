#!/usr/bin/env python3
"""Validate and summarise cached reviewer follow-ups, without API calls.

Run from any directory: python src/followup.py
Requires numpy and scikit-learn. Writes results/followup_summary.json and the
LaTeX tables paper/tables/followup_{numbers,agents,judges}.tex.
All successful calls must match the original prompt hash and response model.
Agent comparisons use the intersection across both arms and all three runs.
Paper-clustered percentile bootstrap intervals condition on the observed runs;
they do not estimate uncertainty over arbitrary future runs or model updates.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'paper/tables'
TABLES.mkdir(parents=True, exist_ok=True)
MODELS = [('gpt-5.4-mini-2026-03-17', 'Mini', 'GPT-5.4 mini'),
          ('gpt-6.1-sol', 'Sol', 'GPT-6.1 sol'),
          ('claude-haiku-4-5-20251001', 'Haiku', 'Haiku 4.5'),
          ('claude-opus-5-5', 'Opus', 'Opus 5.5')]
B, SEED = 2000, 20261115
INPUT_HASHES = {}


def read_jsonl(path):
    INPUT_HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


claim_path = ROOT / 'data/processed/claims.csv'
INPUT_HASHES[str(claim_path.relative_to(ROOT))] = hashlib.sha256(claim_path.read_bytes()).hexdigest()
claims = {r['claim_id']: r for r in csv.DictReader(claim_path.open())}
assert len(claims) == 274
Y = {c: int(float(r['replicated'])) for c, r in claims.items()}
assert sum(Y.values()) == 151
assert len({r['paper_id'] for r in claims.values()}) == 164
RESPONSE_IDS = set()


def valid_index(rows, kind, expected_model):
    """Reject duplicates, malformed values and inconsistent cached parses."""
    result = {}
    for r in rows:
        if r.get('parsed') is None:
            continue
        key = (r['claim_id'], r['condition'])
        assert key not in result, ('duplicate successful condition', key)
        assert r['claim_id'] in claims
        assert r['paper_id'] == claims[r['claim_id']]['paper_id']
        assert r['model'] == r['response_model'] == expected_model
        assert r['response_id'] not in RESPONSE_IDS, 'reused response ID'
        RESPONSE_IDS.add(r['response_id'])
        text = r['raw']
        parsed = json.loads(text[text.find('{'):text.rfind('}') + 1])
        assert parsed == r['parsed'], ('raw/parsed mismatch', key)
        if kind == 'agent':
            assert parsed['decision'] in ('accept', 'reject', 'escalate')
        else:
            assert isinstance(parsed['p_replicate'], (int, float)) and 0 <= parsed['p_replicate'] <= 1
        result[key] = r
    return result


def samples(ids):
    groups = np.array([claims[c]['paper_id'] for c in ids])
    unique = np.unique(groups)
    lookup = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(SEED)
    for _ in range(B):
        yield np.concatenate([lookup[g] for g in rng.choice(unique, len(unique), replace=True)])


def interval(values):
    return np.percentile(values, [2.5, 97.5]).tolist()


def pct_range(vals):
    lo, hi = min(vals) * 100, max(vals) * 100
    return f'{lo:.1f}' if round(lo, 1) == round(hi, 1) else f'{lo:.1f}--{hi:.1f}'


def signed(x, places=3):
    return f'{x:+.{places}f}'.replace('-', '$-$')


out = {'protocol': {'bootstrap_resamples': B, 'seed': SEED,
                   'interval': 'paper-clustered percentile; conditional on observed runs',
                   'agent_subset': 'same claims in both arms across original and two repeat runs',
                   'judge_subset': 'complete two-sample coverage of all 274 claims required',
                   'status': 'exploratory reviewer follow-up; not preregistered'},
       'agents': {}, 'judges': {}, 'incomplete_judges': {}}
macros = {}
agent_rows, judge_rows = [], []
all_agent_cost, all_judge_cost = 0., 0.
agent_success = agent_analysed = judge_success = 0
for model, tag, name in MODELS:
    original_rows = [r for r in read_jsonl(ROOT / 'results/agent' / f'{model}.jsonl')
                     if r.get('condition') in ('A_unprotected', 'A_contract_aware')]
    repeat_rows = read_jsonl(ROOT / 'results/repeats/agent' / f'{model}.jsonl')
    orig = valid_index(original_rows, 'agent', model)
    rep = valid_index(repeat_rows, 'agent', model)
    all_agent_cost += sum(r.get('cost_usd', 0.) for r in repeat_rows)
    agent_success += len(rep)
    arms = {}
    for arm in ('unprotected', 'contract_aware'):
        runs = []
        for k in range(3):
            data = orig if k == 0 else rep
            condition = f'A_{arm}' + (f'_r{k}' if k else '')
            run = {c: r for (c, cond), r in data.items() if cond == condition}
            for c, r in run.items():
                base = orig[(c, f'A_{arm}')]
                assert r['prompt_sha'] == base['prompt_sha'], ('prompt changed', model, condition, c)
                assert r['params'] == base['params'], ('settings changed', model, condition, c)
            runs.append(run)
        arms[arm] = runs
    ids = sorted(set.intersection(*(set(run) for runs in arms.values() for run in runs)))
    assert ids
    yy = np.array([Y[c] for c in ids])
    errors, stats = {}, {}
    for arm, runs in arms.items():
        decisions = np.array([[run[c]['parsed']['decision'] for c in ids] for run in runs])
        wrong = ((decisions == 'accept') & (yy == 0)) | ((decisions == 'reject') & (yy == 1))
        errors[arm] = wrong
        stats[arm] = {'available_claims_by_run': [len(run) for run in runs],
                      'error_rate_by_run': wrong.mean(axis=1).tolist(),
                      'rejection_rate_by_run': (decisions == 'reject').mean(axis=1).tolist(),
                      'escalation_rate_by_run': (decisions == 'escalate').mean(axis=1).tolist(),
                      'three_run_decision_agreement': float(np.all(decisions == decisions[0], axis=0).mean())}
    delta = (errors['contract_aware'].astype(float) - errors['unprotected'].astype(float)).mean(axis=0)
    ci = interval([float(delta[idx].mean()) for idx in samples(ids)])
    out['agents'][model] = {'n_common': len(ids), 'claim_ids': ids, 'arms': stats,
                            'mean_delta_error': float(delta.mean()), 'mean_delta_error_ci': ci,
                            'successful_repeat_calls': len(rep), 'analysed_repeat_calls': len(ids) * 4,
                            'failed_repeat_calls': len(repeat_rows) - len(rep),
                            'repeat_dates': [min(r['ts'] for r in rep.values()), max(r['ts'] for r in rep.values())]}
    agent_analysed += len(ids) * 4
    macros[f'fu{tag}N'] = str(len(ids))
    macros[f'fu{tag}UWrong'] = pct_range(stats['unprotected']['error_rate_by_run'])
    macros[f'fu{tag}CWrong'] = pct_range(stats['contract_aware']['error_rate_by_run'])
    macros[f'fu{tag}CReject'] = pct_range(stats['contract_aware']['rejection_rate_by_run'])
    macros[f'fu{tag}Delta'] = signed(100 * delta.mean(), 1)
    macros[f'fu{tag}DeltaCi'] = '[' + ', '.join(signed(100 * v, 1) for v in ci) + ']'
    agent_rows.append(f'{name} & \\fu{tag}N{{}} & \\fu{tag}UWrong{{}} & \\fu{tag}CWrong{{}} & \\fu{tag}Delta{{}} \\fu{tag}DeltaCi{{}} \\\\')

    original_j = [r for r in read_jsonl(ROOT / 'results/llm' / f'{model}.jsonl') if r.get('condition') == 'forecast_id']
    repeat_j = read_jsonl(ROOT / 'results/repeats/llm' / f'{model}.jsonl')
    base = valid_index(original_j, 'judge', model)
    strong = valid_index(repeat_j, 'judge', model)
    all_judge_cost += sum(r.get('cost_usd', 0.) for r in repeat_j)
    judge_success += len(strong)
    expected = {(c, f'forecast_id_strong_s{k}') for c in claims for k in (1, 2)}
    for (c, cond), r in strong.items():
        assert (c, cond) in expected
        assert r['prompt_sha'] == base[(c, 'forecast_id')]['prompt_sha'], ('judge prompt changed', model, c)
    complete = set(strong) == expected
    if not complete:
        out['incomplete_judges'][model] = {'successful_calls': len(strong),
              'complete_paired_claims': sum(all((c, f'forecast_id_strong_s{k}') in strong for k in (1, 2)) for c in claims),
              'failed_calls': len(repeat_j) - len(strong),
              'exclusion': 'does not cover all 274 claims with two samples; no aggregate performance reported'}
        continue
    ids = sorted(claims)
    yy = np.array([Y[c] for c in ids])
    low = np.array([base[(c, 'forecast_id')]['parsed']['p_replicate'] for c in ids])
    high = np.array([[strong[(c, f'forecast_id_strong_s{k}')]['parsed']['p_replicate'] for c in ids] for k in (1, 2)])
    assert all(r['params'] == {'reasoning_effort': 'medium'} for r in strong.values())
    avg = high.mean(axis=0)
    boot_avg, boot_delta, boot_brier_delta = [], [], []
    for idx in samples(ids):
        a = roc_auc_score(yy[idx], avg[idx])
        boot_avg.append(a)
        boot_delta.append(a - roc_auc_score(yy[idx], low[idx]))
        boot_brier_delta.append(float(np.mean((avg[idx] - yy[idx]) ** 2 - (low[idx] - yy[idx]) ** 2)))
    delta_ci = interval(boot_delta)
    out['judges'][model] = {'n': len(ids), 'samples_per_claim': 2,
        'auroc_original': float(roc_auc_score(yy, low)),
        'auroc_medium_samples': [float(roc_auc_score(yy, v)) for v in high],
        'auroc_medium_average': float(roc_auc_score(yy, avg)),
        'auroc_medium_average_ci': interval(boot_avg),
        'delta_auroc': float(roc_auc_score(yy, avg) - roc_auc_score(yy, low)), 'delta_auroc_ci': delta_ci,
        'brier_original_raw': float(np.mean((low - yy) ** 2)),
        'brier_medium_average_raw': float(np.mean((avg - yy) ** 2)),
        'delta_brier_raw_ci': interval(boot_brier_delta),
        'settings': {'reasoning_effort': 'medium', 'max_completion_tokens': 8000,
                     'original_max_completion_tokens': 2000},
        'note': 'joint change in effort, token cap and averaging; intervals unadjusted and exploratory'}
    j = out['judges'][model]
    for suffix, val in [('Low', j['auroc_original']), ('One', j['auroc_medium_samples'][0]),
                        ('Two', j['auroc_medium_samples'][1]), ('Avg', j['auroc_medium_average'])]:
        macros[f'fu{tag}Judge{suffix}'] = f'{val:.3f}'
    macros[f'fu{tag}JudgeBrierLow'] = f"{j['brier_original_raw']:.4f}"
    macros[f'fu{tag}JudgeBrierAvg'] = f"{j['brier_medium_average_raw']:.4f}"
    macros[f'fu{tag}JudgeDelta'] = signed(j['delta_auroc'])
    macros[f'fu{tag}JudgeDeltaCi'] = '[' + ', '.join(signed(v) for v in delta_ci) + ']'
    judge_rows.append(f'{name} & \\fu{tag}JudgeLow{{}} & \\fu{tag}JudgeAvg{{}} & \\fu{tag}JudgeDeltaCi{{}} \\\\')

out['accounting'] = {'successful_agent_repeat_calls': agent_success,
                     'analysed_agent_repeat_calls': agent_analysed,
                     'successful_stronger_judge_calls': judge_success,
                     'analysed_stronger_judge_calls': len(out['judges']) * 274 * 2,
                     'agent_repeat_cost_usd': all_agent_cost,
                     'stronger_judge_cost_usd': all_judge_cost}
out['input_sha256'] = INPUT_HASHES
macros['fuAgentSuccess'] = f'{agent_success:,}'
macros['fuAgentAnalysed'] = f'{agent_analysed:,}'
macros['fuJudgeAnalysed'] = f'{len(out["judges"]) * 274 * 2:,}'
macros['fuAgentCost'] = f'{all_agent_cost:.2f}'
macros['fuJudgeCost'] = f'{all_judge_cost:.2f}'
(ROOT / 'results/followup_summary.json').write_text(json.dumps(out, indent=2) + '\n')
(TABLES / 'followup_numbers.tex').write_text('% Generated by src/followup.py; do not edit.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items()))
(TABLES / 'followup_agents.tex').write_text(r'''% Generated by src/followup.py.
\begin{table}[t]\centering
\caption{Scenario-A repeatability on common claims. U/C: bare/contract error ranges (\%) over three runs. $\overline{\Delta}$: mean paired error difference (pp), with paper-clustered 95\% CI conditional on these runs.}\label{tab:repeatability}
\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrccc@{}}\toprule
Agent & $n$ & U (\%) & C (\%) & $\overline{\Delta}$ [95\% CI] \\\midrule
''' + '\n'.join(agent_rows) + '\n' + r'''\bottomrule\end{tabular}\end{table}
''')
(TABLES / 'followup_judges.tex').write_text(r'''% Generated by src/followup.py.
\begin{table}[t]\centering
\caption{Identified-input judge follow-up (274 claims): original AUROC vs.\ the mean probability of two medium-reasoning samples with a larger token cap. Paired 95\% CIs use 2,000 paper-clustered resamples.}\label{tab:strongjudges}
\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccc@{}}\toprule
Judge & Original & Medium, mean & $\Delta$AUROC 95\% CI \\\midrule
''' + '\n'.join(judge_rows) + '\n' + r'''\bottomrule\end{tabular}\end{table}
''')
print(json.dumps({'agents': {m: {k: v for k, v in a.items() if k not in ['claim_ids','arms']} for m,a in out['agents'].items()},
                  'judges': out['judges'], 'incomplete_judges': out['incomplete_judges'], 'accounting': out['accounting']}, indent=2))
