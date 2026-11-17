#!/usr/bin/env python3
"""Follow-up runs requested by Reviewer 1 (camera-ready, Nov 2026). Original results are never touched.

W3  Agent decisions were sampled once. We repeat the main arms (scenario A; bare number vs. original contract)
    for all four agents, REPS extra times, with the same prompts and default decoding as the original run.
    Replicate 0 is the original run (results/agent/<model>.jsonl); replicates 1..REPS live in results/repeats/agent/.
W5  LLM judges were run in their weakest setting (one sample, lowest reasoning). We re-run the identified forecast
    condition with a higher reasoning setting and K samples per claim (averaged), in results/repeats/llm/.

    python src/r1_followup.py pilot                 # 3 claims per job, to measure cost
    python src/r1_followup.py agents --budget 6.5
    python src/r1_followup.py judges --budget 8.5
    python src/r1_followup.py analyse               # -> results/r1_followup.json, paper/tables/r1_numbers.tex
"""
import argparse, json, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_llm_judge as J  # noqa: E402
JUDGE_PARSE = J.parse
import anthropic_backend as AB  # noqa: E402
import agent_experiment as AE  # noqa: E402

MODELS = ['gpt-5.4-mini-2026-03-17', 'gpt-6.1-sol', 'claude-haiku-4-5-20251001', 'claude-opus-5-5']
TAG = {'gpt-5.4-mini-2026-03-17': 'Mini', 'gpt-6.1-sol': 'Sol', 'claude-haiku-4-5-20251001': 'Haiku', 'claude-opus-5-5': 'Opus'}
REPS, K = 2, 2
AGENT_OUT, JUDGE_OUT = ROOT / 'results/repeats/agent', ROOT / 'results/repeats/llm'
ARMS = ('unprotected', 'contract_aware')
# "Stronger" judge setting: medium reasoning effort (OpenAI, Opus 5.5) / extended thinking (Haiku 4.5, which has no effort control).
STRONG = {'openai': {'reasoning_effort': 'medium'},
          'claude-opus-5-5': {'max_tokens': 16000, 'output_config': {'effort': 'medium'}},
          'claude-haiku-4-5-20251001': {'max_tokens': 6000, 'thinking': {'type': 'enabled', 'budget_tokens': 3000}}}


class StrongRunner(J.Runner):
    def _call(self, msgs):
        if AB.is_anthropic(self.model):
            saved = AB.SETTINGS[self.model]; AB.SETTINGS[self.model] = STRONG[self.model]
            try:
                resp, extra = AB.call(self.client, self.model, msgs)
            finally:
                AB.SETTINGS[self.model] = saved
            self.params = extra; return resp, extra
        extra = STRONG['openai']
        resp = self.client.chat.completions.create(model=self.model, messages=msgs, response_format={'type': 'json_object'},
                                                   max_completion_tokens=8000, **extra)
        self.params = extra; return resp, extra


def client_for(model):
    from openai import OpenAI
    return AB.make_client() if AB.is_anthropic(model) else OpenAI(api_key=J.load_key(), timeout=120, max_retries=2)


def execute(jobs_by_model, runner_cls, budget, workers=6):
    lock, spent = threading.Lock(), [0.0]
    for model, jobs in jobs_by_model.items():
        runner = runner_cls(client_for(model), model, budget, lock); counts = {}
        bar = tqdm(total=len(jobs), desc=model, unit='call')
        with ThreadPoolExecutor(workers) as ex:
            for f in as_completed([ex.submit(runner.task, r, c, spent) for r, c in jobs]):
                _, st = f.result(); counts[st] = counts.get(st, 0) + 1
                bar.update(1); bar.set_postfix(spent=f'${spent[0]:.3f}', **{k: v for k, v in counts.items() if k != 'ok'})
        bar.close(); print(model, counts, runner.params)
    print(f'spent ${spent[0]:.3f}'); return spent[0]


def agent_jobs(limit=None):
    J.OUT = AGENT_OUT; J.parse = AE.parse_agent; AGENT_OUT.mkdir(parents=True, exist_ok=True)
    J.CONDITIONS = {f'A_{w}_r{k}': (lambda rec, w=w: AE.messages(rec, w)) for w in ARMS for k in range(1, REPS + 1)}
    rows = [json.loads(l) for l in (AE.OUT / 'scenarios.jsonl').read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r['scenario'] == 'A'][:limit]
    return {m: [(r, c) for r in rows for c in J.CONDITIONS] for m in MODELS}


def judge_jobs(limit=None):
    from llm_prompts import records, forecast_messages
    J.OUT = JUDGE_OUT; J.parse = JUDGE_PARSE; JUDGE_OUT.mkdir(parents=True, exist_ok=True)
    # the sample index is part of the system prompt only through the condition name; identical messages would share a
    # cache key, so each sample gets its own condition and the Runner key (claim, condition, sha) stays unique.
    J.CONDITIONS = {f'forecast_id_strong_s{k}': forecast_messages for k in range(1, K + 1)}
    recs = records()[:limit]
    return {m: [(r, c) for r in recs for c in J.CONDITIONS] for m in MODELS}


def load(path, prefix):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    return [r for r in rows if r['condition'].startswith(prefix) and r.get('parsed') is not None]


def boot_ci(stat, claims, groups, B=2000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(B):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]); vals.append(stat(s))
    return np.nanpercentile(vals, [2.5, 97.5])


def analyse():
    from sklearn.metrics import roc_auc_score
    claims = pd.read_csv(ROOT / 'data/processed/claims.csv').set_index('claim_id')
    y, grp = claims.replicated, claims.paper_id
    out, N = {'agents': {}, 'judges': {}}, {}
    pct = lambda x: f'{100 * x:.0f}\\%'; pct1 = lambda x: f'{100 * x:.1f}\\%'
    # ---------- W3: agent replicates ----------
    for m in MODELS:
        runs = {w: [] for w in ARMS}
        orig = load(AE.OUT / f'{m}.jsonl', 'A_')
        rep = load(AGENT_OUT / f'{m}.jsonl', 'A_')
        for w in ARMS:
            d0 = {r['claim_id']: r['parsed']['decision'] for r in orig if r['condition'] == f'A_{w}'}
            runs[w].append(d0)
            for k in range(1, REPS + 1):
                runs[w].append({r['claim_id']: r['parsed']['decision'] for r in rep if r['condition'] == f'A_{w}_r{k}'})
        res = {}
        for w in ARMS:
            stats = []
            for d in runs[w]:
                s = pd.Series(d); yy = y.loc[s.index]
                wrong = ((s == 'accept') & (yy == 0)) | ((s == 'reject') & (yy == 1))
                stats.append({'n': len(s), 'escalate': float((s == 'escalate').mean()), 'reject': float((s == 'reject').mean()),
                              'wrong': float(wrong.mean())})
            common = set.intersection(*[set(d) for d in runs[w]])
            agree = np.mean([len({d[c] for d in runs[w]}) == 1 for c in common])
            res[w] = {'runs': stats, 'n_common': len(common), 'all_runs_agree': float(agree)}
        res['delta_wrong'] = [c['wrong'] - u['wrong'] for u, c in zip(res['unprotected']['runs'], res['contract_aware']['runs'])]
        out['agents'][m] = res
        t = TAG[m]
        for w, wt in (('unprotected', 'U'), ('contract_aware', 'C')):
            wr = [r['wrong'] for r in res[w]['runs']]; es = [r['escalate'] for r in res[w]['runs']]
            N[f'rr{t}{wt}WrongMin'], N[f'rr{t}{wt}WrongMax'] = pct1(min(wr)), pct1(max(wr))
            N[f'rr{t}{wt}EscMin'], N[f'rr{t}{wt}EscMax'] = pct(min(es)), pct(max(es))
            N[f'rr{t}{wt}Agree'] = pct(res[w]['all_runs_agree'])
        dw = res['delta_wrong']; N[f'rr{t}DMin'], N[f'rr{t}DMax'] = f'{100 * min(dw):+.1f}', f'{100 * max(dw):+.1f}'
        rj = [r['reject'] for r in res['contract_aware']['runs']]; N[f'rr{t}CRejMin'], N[f'rr{t}CRejMax'] = pct(min(rj)), pct(max(rj))
    N['rrRuns'] = str(REPS + 1)
    # ---------- W5: stronger judges ----------
    order = list(claims.index)  # claim order from the processed dataset; no raw SCORE download needed
    for m in MODELS:
        base = {r['claim_id']: r['parsed']['p_replicate'] for r in load(ROOT / 'results/llm' / f'{m}.jsonl', 'forecast_id')
                if r['condition'] == 'forecast_id'}
        st = load(JUDGE_OUT / f'{m}.jsonl', 'forecast_id_strong')
        if not st:
            continue
        S = pd.DataFrame([(r['claim_id'], r['condition'], r['parsed']['p_replicate']) for r in st], columns=['c', 'k', 'p'])
        P = S.pivot_table(index='c', columns='k', values='p').reindex(order).dropna()
        ids = list(P.index.intersection(list(base)))
        if len(ids) < len(claims):  # incomplete runs (e.g. the three-claim Claude pilots) are not reported
            out['judges_incomplete'] = {**out.get('judges_incomplete', {}), m: len(ids)}
            continue
        yy, gg = y.loc[ids].values, grp.loc[ids].values
        p_low, p_s1, p_avg = np.array([base[c] for c in ids]), P.loc[ids].iloc[:, 0].values, P.loc[ids].mean(1).values
        auc = lambda p, s=None: roc_auc_score(yy[s], p[s]) if s is not None else roc_auc_score(yy, p)
        brier = lambda p: float(np.mean((p - yy) ** 2))
        ci_avg = boot_ci(lambda s: auc(p_avg, s), ids, gg)
        ci_d = boot_ci(lambda s: auc(p_avg, s) - auc(p_low, s), ids, gg, seed=1)
        cost = float(sum(r.get('cost_usd', 0) for r in st))
        sd = float(P.loc[ids].std(1, ddof=1).mean())
        out['judges'][m] = {'n': len(ids), 'auc_low': auc(p_low), 'auc_strong_single': auc(p_s1), 'auc_strong_avg': auc(p_avg),
                            'auc_strong_avg_ci': list(ci_avg), 'delta_avg_minus_low': auc(p_avg) - auc(p_low), 'delta_ci': list(ci_d),
                            'brier_low': brier(p_low), 'brier_strong_avg': brier(p_avg), 'mean_within_claim_sd': sd, 'cost_usd': cost,
                            'params': st[0].get('params')}
        t = TAG[m]; j = out['judges'][m]
        N[f'sj{t}Low'], N[f'sj{t}One'], N[f'sj{t}Avg'] = f"{j['auc_low']:.3f}", f"{j['auc_strong_single']:.3f}", f"{j['auc_strong_avg']:.3f}"
        N[f'sj{t}AvgCi'] = f"[{ci_avg[0]:.2f}, {ci_avg[1]:.2f}]"
        N[f'sj{t}D'] = f"{j['delta_avg_minus_low']:+.3f}".replace('-', '$-$')
        N[f'sj{t}DCi'] = f"[{ci_d[0]:+.3f}, {ci_d[1]:+.3f}]".replace('-', '$-$')
        N[f'sj{t}BrierLow'], N[f'sj{t}BrierAvg'] = f"{j['brier_low']:.4f}", f"{j['brier_strong_avg']:.4f}"
        N[f'sj{t}Sd'] = f'{sd:.3f}'
    N['sjK'] = str(K)
    agent_cost = sum(r.get('cost_usd', 0) for m in MODELS for r in load(AGENT_OUT / f'{m}.jsonl', 'A_'))
    N['rrCost'] = f'{agent_cost:.2f}'; N['sjCost'] = f"{sum(j['cost_usd'] for j in out['judges'].values()):.2f}"
    (ROOT / 'results/r1_followup.json').write_text(json.dumps(out, indent=1))
    (ROOT / 'paper/tables/r1_numbers.tex').write_text(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in N.items()))
    print(json.dumps(out, indent=1)[:4000])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['pilot', 'agents', 'judges', 'analyse'])
    ap.add_argument('--budget', type=float, default=5.0); ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    if a.cmd == 'pilot':
        execute(agent_jobs(limit=3), J.Runner, a.budget, a.workers)
        execute(judge_jobs(limit=3), StrongRunner, a.budget, a.workers)
    elif a.cmd == 'agents':
        execute(agent_jobs(), J.Runner, a.budget, a.workers)
    elif a.cmd == 'judges':
        execute(judge_jobs(), StrongRunner, a.budget, a.workers)
    else:
        analyse()
