#!/usr/bin/env python3
"""Agent-in-the-loop experiment: does exposing the verifier contract change an LLM agent's decisions?

A triage agent (LLM) must ACCEPT (build on), REJECT (ignore) or ESCALATE (human review) each SCORE claim, given
the output of the replication-verifier tool. Two workflows see the same claims:
  unprotected     : tool output = {"p_replicate": p} only
  contract_aware  : tool output = full verify_claim response (base rate, CV skill, selective_ok, advice, provenance)
Scenario A (valid query, as_of 2018-12-31): p from the contract-enforcing verifier (leave-paper-out, strict cutoff).
Scenario B (post-launch query, as_of 2023-06-01; claims launched before that date): the unprotected tool silently
returns the no-cutoff (leaky) M2 prediction; the contract-aware tool returns the contract_violation refusal.

    python src/agent_experiment.py prepare            # verifier outputs (no API calls)
    python src/agent_experiment.py run --models gpt-5.4-mini-2026-03-17 gpt-6.1-sol --budget 6
    python src/agent_experiment.py analyse
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'results' / 'agent'
AS_OF_A, AS_OF_B = '2018-12-31', '2023-06-01'

AGENT_SYSTEM = (
    "You are the literature-triage agent in an automated research pipeline. For each published claim choose one action: "
    '"accept" (the pipeline builds new experiments on the claim), "reject" (the pipeline ignores the claim), or '
    '"escalate" (a human expert reviews the claim; capacity is limited, so escalate only when warranted). '
    "Building on a claim that later fails to replicate wastes downstream experiments. You have received the output of "
    "the replication-verifier tool verify_claim for this claim. "
    'Respond with JSON only: {"decision": "accept"|"reject"|"escalate", "reason": "<at most 20 words>"}')


def prepare():
    from verifier import ReliabilityVerifier
    from llm_prompts import records
    OUT.mkdir(parents=True, exist_ok=True)
    v = ReliabilityVerifier(audit_log=OUT / 'agent_audit_log.jsonl')
    recs = {r['claim_id']: r for r in records()}
    oof = pd.read_csv(ROOT / 'results/oof_predictions_seedavg.csv').set_index('claim_id')
    rows = []
    for cid in tqdm(v.claims.claim_id, desc='verifier outputs'):
        r = recs[cid]; base = {'claim_id': cid, 'paper_id': r['paper_id'], 'title': r['title'], 'citation': r['citation'], 'field': r['field']}
        full = v.verify_claim(cid, AS_OF_A, 'publication')
        rows.append({**base, 'scenario': 'A', 'unprotected': {'p_replicate': full['p_replicate']},
                     'contract_aware': {k: full[k] for k in ['p_replicate', 'base_rate', 'cv_auroc', 'cv_brier_skill', 'selective_ok',
                                                             'evidence_cutoff', 'n_evidence', 'advice', 'drivers']}})
        launch = r['launch_date']
        if launch and launch < AS_OF_B:
            try:
                v.verify_claim(cid, AS_OF_B, 'publication'); refused = None
            except Exception as e:
                refused = {'error': 'contract_violation', 'detail': str(e)}
            assert refused is not None, 'contract-aware tool must refuse post-launch queries'
            leaky = float(oof.loc[cid, 'L-M2 Graph + dependency, NO cutoff (leaky)'])
            rows.append({**base, 'scenario': 'B', 'unprotected': {'p_replicate': round(leaky, 4)}, 'contract_aware': refused})
    (OUT / 'scenarios.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    print(len(rows), 'scenario rows;', sum(r['scenario'] == 'B' for r in rows), 'post-launch')


V2_ADVICE = 'weak signal: this score cannot justify accepting or rejecting the claim on its own; escalate if a decision is needed'


def contract_v2(out):
    """Exploratory wording revision (added after observing Claude Haiku 4.5 misreading v1): action-oriented advice,
    unambiguous field name, and a refusal that states it carries no information about the claim."""
    if out.get('error') == 'contract_violation':
        return {**out, 'advice': 'no information about the claim itself; the query date is too late to answer safely - escalate'}
    o = {('confidence_gating_supported' if k == 'selective_ok' else k): v for k, v in out.items()}
    o['advice'] = V2_ADVICE if not out.get('selective_ok') else out['advice']
    return o


PLACEBO_NOTE = 'Note: this estimate is uncertain.'

# Prompt-robustness variant: same task and actions, independently worded (different framing and order).
AGENT_SYSTEM_P2 = (
    "You support a team of scientists who want to run follow-up studies. For the claim below, decide what happens next. "
    "Options: escalate (send the claim to a domain expert for review; the expert's time is scarce), "
    "reject (drop the claim from the follow-up plan), or accept (plan follow-up studies that assume the claim holds). "
    "Accepting a claim that later fails to replicate wastes those studies. A replication-verifier tool has been queried; "
    "its output is shown. Answer with JSON only: {\"decision\": \"accept\"|\"reject\"|\"escalate\", \"reason\": \"<at most 20 words>\"}")


def messages(row, workflow):
    if workflow.endswith('_p2'):
        msgs = messages(row, workflow[:-3]); msgs[0] = {'role': 'system', 'content': AGENT_SYSTEM_P2}
        return msgs
    """Control arms (scenario A only): 'no_tool' gives no verifier output; 'placebo' gives the same p_replicate plus a
    generic uncertainty note carrying no contract information."""
    if workflow == 'no_tool':
        body = (f"Claim source: \"{row['title']}\" ({row['citation']}); field: {row['field']}.\n"
                "Tool output (verify_claim): unavailable for this claim.")
        return [{'role': 'system', 'content': AGENT_SYSTEM}, {'role': 'user', 'content': body}]
    if workflow == 'placebo':
        payload = {'p_replicate': row['unprotected']['p_replicate'], 'note': PLACEBO_NOTE}
    else:
        payload = contract_v2(row['contract_aware']) if workflow == 'contract_aware_v2' else row[workflow]
    body = (f"Claim source: \"{row['title']}\" ({row['citation']}); field: {row['field']}.\n"
            f"Tool output (verify_claim):\n{json.dumps(payload)}")
    return [{'role': 'system', 'content': AGENT_SYSTEM}, {'role': 'user', 'content': body}]


def parse_agent(text, condition):
    try:
        t = text.strip(); obj = json.loads(t[t.find('{'): t.rfind('}') + 1])
    except Exception as e:
        return None, f'json: {e}'
    return (obj, None) if obj.get('decision') in ('accept', 'reject', 'escalate') else (None, f"bad decision {obj.get('decision')!r}")


def run(models, budget, workers, workflows=('unprotected', 'contract_aware')):
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import run_llm_judge as J
    from openai import OpenAI
    rows = [json.loads(l) for l in (OUT / 'scenarios.jsonl').read_text().splitlines() if l.strip()]
    J.OUT = OUT; J.parse = parse_agent
    base_w = ('unprotected', 'contract_aware', 'contract_aware_v2', 'no_tool', 'placebo')
    J.CONDITIONS = {f'{s}_{w}': (lambda rec, w=w: messages(rec, w)) for s in 'AB' for w in base_w + tuple(x + '_p2' for x in base_w)}
    import anthropic_backend as AB
    lock, spent = threading.Lock(), [0.0]
    for model in models:
        client = AB.make_client() if AB.is_anthropic(model) else OpenAI(api_key=J.load_key(), timeout=60, max_retries=2)
        runner = J.Runner(client, model, budget, lock)
        jobs = [({**r, 'claim_id': r['claim_id']}, f"{r['scenario']}_{w}") for r in rows for w in workflows
                if not ((w in ('no_tool', 'placebo') or w.endswith('_p2')) and r['scenario'] != 'A')]
        bar = tqdm(total=len(jobs), desc=f'agent {model}', unit='call'); counts = {}
        with ThreadPoolExecutor(workers) as ex:
            for f in as_completed([ex.submit(runner.task, r, c, spent) for r, c in jobs]):
                _, st = f.result(); counts[st] = counts.get(st, 0) + 1
                bar.update(1); bar.set_postfix(spent=f'${spent[0]:.3f}', **{k: v for k, v in counts.items() if k != 'ok'})
        bar.close(); print(model, counts, runner.params)
    print(f'spent ${spent[0]:.3f}')


def analyse():
    oof = pd.read_csv(ROOT / 'results/oof_predictions_seedavg.csv').set_index('claim_id')
    res = {}
    for f in sorted(OUT.glob('*.jsonl')):
        if f.name in ('scenarios.jsonl', 'agent_audit_log.jsonl'): continue
        d = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()])
        d = d[d.parsed.notna()].sort_values('ts').drop_duplicates(['claim_id', 'condition'], keep='last')
        d['decision'] = d.parsed.map(lambda o: o['decision']); d['y'] = oof.loc[d.claim_id, 'y'].to_numpy()
        d['scenario'] = d.condition.str[0]; d['workflow'] = d.condition.str[2:]
        out = {'cost_usd': float(d.cost_usd.sum()), 'n_calls': len(d)}
        for sc in 'AB':
            wide = d[d.scenario == sc].pivot(index='claim_id', columns='workflow', values='decision').dropna(subset=['unprotected', 'contract_aware'])
            y = oof.loc[wide.index, 'y'].to_numpy(); g = oof.loc[wide.index, 'paper_id'].to_numpy()

            def stats(dec, yy):
                auto = dec != 'escalate'; n = len(dec)
                return {'n': n, 'accept': float((dec == 'accept').mean()), 'reject': float((dec == 'reject').mean()),
                        'escalate': float((~auto).mean()), 'auto': float(auto.mean()),
                        'unsafe_accept': float(((dec == 'accept') & (yy == 0)).mean()),
                        'wrong_reject': float(((dec == 'reject') & (yy == 1)).mean()),
                        'auto_error': float((((dec == 'accept') & (yy == 0)) | ((dec == 'reject') & (yy == 1))).mean()),
                        'auto_accuracy': float(((((dec == 'accept') & (yy == 1)) | ((dec == 'reject') & (yy == 0)))[auto]).mean()) if auto.any() else None}
            S = {w: stats(wide[w].to_numpy(), y) for w in ('unprotected', 'contract_aware')}
            for w in ('contract_aware_v2', 'no_tool', 'placebo', 'unprotected_p2', 'contract_aware_p2', 'placebo_p2', 'no_tool_p2'):
                if w in wide.columns and wide[w].notna().all():
                    S[w] = stats(wide[w].to_numpy(), y)
            S['always_escalate'] = stats(np.array(['escalate'] * len(y)), y)
            # paired clustered-bootstrap CIs for the control comparisons (wrong automatic decisions)
            uniq = np.unique(g); pos = {k: np.flatnonzero(g == k) for k in uniq}
            for a_, b_ in (('contract_aware', 'placebo'), ('unprotected', 'no_tool'), ('contract_aware', 'no_tool')):
                if a_ in wide.columns and b_ in wide.columns and wide[[a_, b_]].notna().all().all():
                    xa, xb = wide[a_].to_numpy(), wide[b_].to_numpy(); bs = []
                    rng2 = np.random.default_rng(31)
                    for _ in range(2000):
                        ii = np.concatenate([pos[k] for k in rng2.choice(uniq, len(uniq))])
                        bs.append(stats(xa[ii], y[ii])['auto_error'] - stats(xb[ii], y[ii])['auto_error'])
                    S[f'diff_{a_}_minus_{b_}'] = {'point': stats(xa, y)['auto_error'] - stats(xb, y)['auto_error'],
                                                  'ci': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
            uniq = np.unique(g); pos = {k: np.flatnonzero(g == k) for k in uniq}; rng = np.random.default_rng(21); diffs = []
            for _ in range(2000):
                ii = np.concatenate([pos[k] for k in rng.choice(uniq, len(uniq))])
                a = stats(wide['contract_aware'].to_numpy()[ii], y[ii]); b = stats(wide['unprotected'].to_numpy()[ii], y[ii])
                diffs.append({k: a[k] - b[k] for k in ('escalate', 'auto', 'unsafe_accept', 'wrong_reject', 'auto_error')})
            dd = pd.DataFrame(diffs)
            S['diff_contract_minus_unprotected'] = {k: {'point': S['contract_aware'][k] - S['unprotected'][k],
                                                        'ci': [float(dd[k].quantile(.025)), float(dd[k].quantile(.975))]}
                                                    for k in dd}
            out[sc] = S
        res[f.stem] = out
        print(json.dumps({f.stem: out}, indent=1, default=float))
    (ROOT / 'results/agent_summary.json').write_text(json.dumps(res, indent=2, default=float))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['prepare', 'run', 'analyse'])
    ap.add_argument('--models', nargs='+', default=['gpt-5.4-mini-2026-03-17', 'gpt-6.1-sol'])
    ap.add_argument('--budget', type=float, default=6.0); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--workflows', nargs='+', default=['unprotected', 'contract_aware'])
    a = ap.parse_args()
    {'prepare': prepare, 'run': lambda: run(a.models, a.budget, a.workers, tuple(a.workflows)), 'analyse': analyse}[a.cmd]()
