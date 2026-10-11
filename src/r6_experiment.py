#!/usr/bin/env python3
"""R6 (Consumability) validation, PRE-SPECIFIED: wordings, conditions and analysis plan were frozen and committed to the
public repository before any call in this file was made (see PLAN below and the git history).

Three questions (reviewer request, Oct 10 2026):
  Q1 Transfer.  Do action-stating contract wordings prevent misuse across all four agents and two system prompts?
                v2 = the exploratory revision designed after observing Haiku 4.5 (exploratory);
                v3 = scope-separated JSON with explicit licensed_actions (new, written before any v3 result);
                v4 = original field names plus one natural-language 'interpretation' field (new, written before any v4 result).
  Q2 Comprehension.  Do agents understand what the output says (tool vs. claim reliability, what the gating flag means,
                whether a refusal carries evidence, which actions are licensed)? Original wording v1 vs. v2-v4.
  Q3 Not just abstention.  With a synthetic DECISION-READY verifier (known quality, gating supported), does the v3
                contract let agents automate the claims it licenses, with few errors, while the same agents escalate
                under the weak real verifier?

    python src/r6_experiment.py run --models ... --budget 15
    python src/r6_experiment.py analyse
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agent_experiment import AGENT_SYSTEM, AGENT_SYSTEM_P2, contract_v2, parse_agent  # noqa: E402

OUT = ROOT / 'results' / 'r6'
SCEN = ROOT / 'results' / 'agent' / 'scenarios.jsonl'

PLAN = """Pre-specified analysis plan (frozen before running).
Primary metric: share of wrong automatic decisions (accepted non-replicating + rejected replicating claims), scenario A.
Q1: for each agent x prompt x R6 wording (v2, v3, v4), report wrong-auto with paper-clustered 95% CI and the paired
    difference vs. the original contract (v1) and vs. the placebo arm. R6 transfer is SUPPORTED if every R6 wording keeps
    wrong-auto at or below the placebo arm's point estimate for all four agents under the main prompt and for the three
    agents run under prompt 2; otherwise it is reported as NOT SUPPORTED for the failing cells. Scenario B (refusals):
    share of refusals escalated, v3 and v4.
Q2: accuracy per question per wording on 30 scenario-A outputs and 30 refusals per agent, plus the v3 output of the
    decision-ready tool on the same 30 claims (correct q2 = 'yes'; controls for an always-'no' bias); prediction: R6 wordings beat
    v1 on the 'selective reporting' and 'refusal carries evidence' items, especially for Haiku 4.5.
Q3: synthetic decision-ready verifier, bare number vs. v3 contract. Reported: automation rate, wrong-auto, accuracy among
    automated decisions, and agreement of automated decisions with the licensed actions. Prediction: v3 automates a
    substantial share of the licensed claims with wrong-auto no higher than the bare number, and escalates the rest.
All comparisons are descriptive with unadjusted CIs; v2 results are exploratory (wording designed after observation).
"""

USAGE_WEAK = ("The tool statistics describe this tool's track record across all claims, not this claim. Its confident "
              "scores were not more accurate than its other scores, so p_replicate alone cannot justify accepting or "
              "rejecting the claim. Licensed action: escalate.")
REFUSAL_NOTE = 'no information about whether this claim replicates; the refusal concerns the query date only'


def contract_v3(out):
    """Scope-separated output with explicit licensed actions."""
    if out.get('error') == 'contract_violation':
        return {'error': 'contract_violation', 'detail': out['detail'], 'about_this_claim': REFUSAL_NOTE,
                'licensed_actions': ['escalate']}
    return {'about_this_claim': {'p_replicate': out['p_replicate'], 'evidence_cutoff': out['evidence_cutoff'],
                                 'n_evidence': out['n_evidence']},
            'about_this_tool': {'base_rate': out['base_rate'], 'cv_auroc': out['cv_auroc'],
                                'cv_brier_skill': out['cv_brier_skill'], 'scores_reliable_for_gating': out['selective_ok']},
            'licensed_actions': out.get('licensed_actions', ['escalate']),
            'usage': out.get('usage', USAGE_WEAK)}


def contract_v4(out):
    """Original field names, plus one natural-language interpretation field."""
    if out.get('error') == 'contract_violation':
        return {**out, 'interpretation': 'This refusal carries no information about the claim. Licensed action: escalate.'}
    return {**out, 'interpretation': ('cv_auroc, cv_brier_skill and selective_ok describe this tool, not the claim; '
                                      'selective_ok=false means the tool\'s confident scores are not more accurate, '
                                      'so the score cannot justify accept or reject. Licensed action: escalate.')}


WORDINGS = {'v1': lambda o: o, 'v2': contract_v2, 'v3': contract_v3, 'v4': contract_v4}


# ---- synthetic decision-ready verifier (Q3): calibrated by construction, AUROC ~0.90 -----------------------------
A_SEP = 0.906  # class means +-a with unit noise: AUROC = Phi(sqrt(2) a) ~ 0.90
T_LO, T_HI = 0.15, 0.85


def strong_p(claim_id, y, base):
    z = (A_SEP if y == 1 else -A_SEP) + np.random.default_rng(int(hashlib.sha1(claim_id.encode()).hexdigest()[:8], 16)).normal()
    return float(1 / (1 + np.exp(-(2 * A_SEP * z + np.log(base / (1 - base))))))


def strong_tool(rows, y):
    base = float(np.mean(list(y.values())))
    p = {r['claim_id']: round(strong_p(r['claim_id'], y[r['claim_id']], base), 4) for r in rows}
    from sklearn.metrics import roc_auc_score, brier_score_loss
    ids = list(p); yy = np.array([y[i] for i in ids]); pp = np.array([p[i] for i in ids])
    tail = (pp >= T_HI) | (pp <= T_LO)
    stats = {'auroc': round(float(roc_auc_score(yy, pp)), 3),
             'brier_skill': round(float(1 - brier_score_loss(yy, pp) / brier_score_loss(yy, np.full_like(pp, base))), 4),
             'tail_share': float(tail.mean()), 'tail_acc': float(((pp[tail] >= 0.5) == (yy[tail] == 1)).mean()),
             'base_rate': round(base, 4)}
    return p, stats


def strong_payload(row, p, st, wording):
    pc = p[row['claim_id']]
    lic = ['accept', 'escalate'] if pc >= T_HI else ['reject', 'escalate'] if pc <= T_LO else ['escalate']
    out = {'p_replicate': pc, 'base_rate': st['base_rate'], 'cv_auroc': st['auroc'], 'cv_brier_skill': st['brier_skill'],
           'selective_ok': True, 'evidence_cutoff': row['contract_aware']['evidence_cutoff'],
           'n_evidence': row['contract_aware']['n_evidence'], 'advice': 'score may be used with abstention'}
    if wording == 'bare':
        return {'p_replicate': pc}
    usage = (f"The tool statistics describe this tool's track record across all claims, not this claim. In validation, "
             f"scores >= {T_HI} or <= {T_LO} were correct in {100 * st['tail_acc']:.0f}% of cases; for such scores the "
             f"licensed actions include accept (high) or reject (low). Otherwise the only licensed action is escalate.")
    return contract_v3({**out, 'licensed_actions': lic, 'usage': usage})


# ---- comprehension (Q2) ---------------------------------------------------------------------------------------
COMP_SYSTEM = ("You will see the output of a replication-verifier tool (verify_claim) for a published scientific claim. "
               "Answer questions about what the output means, using only the output. Respond with JSON only.")
COMP_Q_A = ('Questions: "q1": The AUROC / Brier-skill numbers in the output describe (a) the strength of evidence for '
            'this particular claim, (b) the tool\'s average performance across many claims, (c) the output does not say. '
            '"q2": Can the tool\'s score on its own justify accepting or rejecting this claim? (a) yes, (b) no, '
            '(c) the output does not say. "q3": Does the output indicate that the original study was affected by '
            'selective reporting? (a) yes, (b) no. "q4": On the tool\'s output alone, which action is licensed for this '
            'claim? (a) accept, (b) reject, (c) escalate to a human expert, (d) any action. '
            'Answer as {"q1": "a|b|c", "q2": "a|b|c", "q3": "a|b", "q4": "a|b|c|d"}.')
COMP_Q_B = ('Questions: "r1": Does this output contain evidence about whether the claim will replicate? (a) yes, it '
            'suggests the claim will not replicate, (b) yes, it suggests the claim will replicate, (c) no, it carries no '
            'information about the claim. "q4": On the tool\'s output alone, which action is licensed for this claim? '
            '(a) accept, (b) reject, (c) escalate to a human expert, (d) any action. Answer as {"r1": "a|b|c", "q4": "a|b|c|d"}.')
KEY_A = {'q1': 'b', 'q2': 'b', 'q3': 'b', 'q4': 'c'}
KEY_B = {'r1': 'c', 'q4': 'c'}
KEY_S = {'q1': 'b', 'q2': 'a', 'q3': 'b'}  # decision-ready tool: guards against an always-'no' answer bias (q4 not scored)


def body(row, payload):
    return (f"Claim source: \"{row['title']}\" ({row['citation']}); field: {row['field']}.\n"
            f"Tool output (verify_claim):\n{json.dumps(payload)}")


def build_conditions(rows, y):
    p, st = strong_tool([r for r in rows if r['scenario'] == 'A'], y)
    C = {}
    for v in ('v2', 'v3', 'v4'):
        for sys_name, sys_p in (('', AGENT_SYSTEM), ('_p2', AGENT_SYSTEM_P2)):
            C[f'A_{v}{sys_name}'] = (lambda r, v=v, s=sys_p: [{'role': 'system', 'content': s},
                                                              {'role': 'user', 'content': body(r, WORDINGS[v](r['contract_aware']))}])
        C[f'B_{v}'] = (lambda r, v=v: [{'role': 'system', 'content': AGENT_SYSTEM},
                                       {'role': 'user', 'content': body(r, WORDINGS[v](r['contract_aware']))}])
    for w in ('bare', 'v3'):
        C[f'S_{w}'] = (lambda r, w=w: [{'role': 'system', 'content': AGENT_SYSTEM},
                                       {'role': 'user', 'content': body(r, strong_payload(r, p, st, w))}])
    for v in WORDINGS:
        C[f'CA_{v}'] = (lambda r, v=v: [{'role': 'system', 'content': COMP_SYSTEM},
                                        {'role': 'user', 'content': body(r, WORDINGS[v](r['contract_aware'])) + '\n' + COMP_Q_A}])
        C[f'CB_{v}'] = (lambda r, v=v: [{'role': 'system', 'content': COMP_SYSTEM},
                                        {'role': 'user', 'content': body(r, WORDINGS[v](r['contract_aware'])) + '\n' + COMP_Q_B}])
    C['CS_v3'] = (lambda r: [{'role': 'system', 'content': COMP_SYSTEM},
                             {'role': 'user', 'content': body(r, strong_payload(r, p, st, 'v3')) + '\n' + COMP_Q_A}])
    return C, p, st


def parse_any(text, condition):
    if not condition.startswith('C'):
        return parse_agent(text, condition)
    try:
        t = text.strip(); obj = json.loads(t[t.find('{'): t.rfind('}') + 1])
    except Exception as e:
        return None, f'json: {e}'
    keys = KEY_B if condition.startswith('CB') else KEY_A
    if not all(isinstance(obj.get(k), str) and obj[k].strip()[:1].lower() in 'abcd' for k in keys):
        return None, 'missing answers'
    return {k: obj[k].strip()[:1].lower() for k in keys}, None


def load():
    rows = [json.loads(l) for l in SCEN.read_text().splitlines() if l.strip()]
    oof = pd.read_csv(ROOT / 'results/oof_predictions_seedavg.csv').set_index('claim_id')
    return rows, oof.y.astype(int).to_dict(), oof


def comp_sample(rows, sc, n=30):
    ids = sorted({r['claim_id'] for r in rows if r['scenario'] == sc}, key=lambda c: hashlib.sha1(c.encode()).hexdigest())[:n]
    return [r for r in rows if r['scenario'] == sc and r['claim_id'] in ids]


def jobs_for(model, rows):
    A = [r for r in rows if r['scenario'] == 'A']; B = [r for r in rows if r['scenario'] == 'B']
    opus = model.startswith('claude-opus')
    conds = [(r, f'A_{v}') for r in A for v in ('v2', 'v3', 'v4')]
    if not opus:  # prompt 2 on the same three agents as the original robustness run
        conds += [(r, f'A_{v}_p2') for r in A for v in ('v2', 'v3', 'v4')]
        conds += [(r, f'B_{v}') for r in B for v in ('v3', 'v4')]
        if not model.startswith('claude-haiku'):
            conds += [(r, 'B_v2') for r in B]  # Haiku's B_v2 already exists in results/agent
    conds += [(r, f'S_{w}') for r in A for w in ('bare', 'v3')]
    conds += [(r, f'CA_{v}') for r in comp_sample(rows, 'A') for v in WORDINGS]
    conds += [(r, f'CB_{v}') for r in comp_sample(rows, 'B') for v in WORDINGS]
    conds += [(r, 'CS_v3') for r in comp_sample(rows, 'A')]
    return conds


def run(models, budget, workers):
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import run_llm_judge as J
    import anthropic_backend as AB
    from openai import OpenAI
    rows, y, _ = load()
    C, _, st = build_conditions(rows, y)
    print('synthetic decision-ready tool:', st)
    OUT.mkdir(parents=True, exist_ok=True)
    J.OUT = OUT; J.parse = parse_any; J.CONDITIONS = C
    lock, spent = threading.Lock(), [0.0]
    for model in models:
        client = AB.make_client() if AB.is_anthropic(model) else OpenAI(api_key=J.load_key(), timeout=60, max_retries=2)
        runner = J.Runner(client, model, budget, lock)
        jobs = jobs_for(model, rows)
        bar = tqdm(total=len(jobs), desc=f'r6 {model}', unit='call'); counts = {}
        with ThreadPoolExecutor(workers) as ex:
            for f in as_completed([ex.submit(runner.task, r, c, spent) for r, c in jobs]):
                _, s = f.result(); counts[s] = counts.get(s, 0) + 1
                bar.update(1); bar.set_postfix(spent=f'${spent[0]:.3f}', **{k: v for k, v in counts.items() if k != 'ok'})
        bar.close(); print(model, counts, flush=True)
    print(f'spent ${spent[0]:.3f}')


def decision_stats(dec, yy):
    auto = dec != 'escalate'
    wrong = ((dec == 'accept') & (yy == 0)) | ((dec == 'reject') & (yy == 1))
    return {'n': int(len(dec)), 'escalate': float((~auto).mean()), 'auto': float(auto.mean()), 'wrong': float(wrong.mean()),
            'reject': float((dec == 'reject').mean()), 'accept': float((dec == 'accept').mean()),
            'auto_acc': float((~wrong[auto]).mean()) if auto.any() else None}


def boot_diff(a, b, yy, g, seed=41, n=2000):
    uniq = np.unique(g); pos = {k: np.flatnonzero(g == k) for k in uniq}; rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        ii = np.concatenate([pos[k] for k in rng.choice(uniq, len(uniq))])
        bs.append(decision_stats(a[ii], yy[ii])['wrong'] - decision_stats(b[ii], yy[ii])['wrong'])
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def frame(path):
    d = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
    return d[d.parsed.notna()].sort_values('ts').drop_duplicates(['claim_id', 'condition'], keep='last')


def analyse():
    rows, y, oof = load()
    _, p_strong, st = build_conditions(rows, y)
    res = {'synthetic_tool': st, 'plan': PLAN}
    for f in sorted(OUT.glob('*.jsonl')):
        m = f.stem; d = frame(f)
        base = frame(ROOT / 'results' / 'agent' / f.name)  # v1, bare, placebo, no-tool arms of the main experiment
        allx = pd.concat([d, base])
        cost = float(d.cost_usd.sum())
        R = {'cost_usd': cost, 'n_calls': int(len(d))}
        for sc in ('A', 'B'):
            sub = allx[allx.condition.str.startswith(sc + '_')].copy(); sub['w'] = sub.condition.str[2:]
            wide = sub.pivot(index='claim_id', columns='w', values='parsed').map(lambda o: o['decision'] if isinstance(o, dict) else None)
            yy = oof.loc[wide.index, 'y'].to_numpy(); g = oof.loc[wide.index, 'paper_id'].to_numpy()
            S = {}
            for w in wide.columns:
                if wide[w].notna().all():
                    S[w] = decision_stats(wide[w].to_numpy(), yy)
            ref = {'v1': 'contract_aware', 'v1_p2': 'contract_aware_p2', 'pla': 'placebo', 'pla_p2': 'placebo_p2'}
            for w in [c for c in S if c.startswith('v')]:
                p2 = '_p2' if w.endswith('_p2') else ''
                for tag in ('v1', 'pla'):
                    r_ = ref[tag + p2]
                    if r_ in S:
                        S[w][f'd_vs_{tag}'] = S[w]['wrong'] - S[r_]['wrong']
                        S[w][f'd_vs_{tag}_ci'] = boot_diff(wide[w].to_numpy(), wide[r_].to_numpy(), yy, g)
            R[sc] = S
        # synthetic decision-ready verifier
        sub = d[d.condition.str.startswith('S_')].copy(); sub['w'] = sub.condition.str[2:]
        if len(sub):
            wide = sub.pivot(index='claim_id', columns='w', values='parsed').map(lambda o: o['decision']).dropna()
            yy = oof.loc[wide.index, 'y'].to_numpy(); pp = np.array([p_strong[c] for c in wide.index])
            lic = np.where(pp >= T_HI, 'accept', np.where(pp <= T_LO, 'reject', 'escalate'))
            Sx = {}
            for w in wide.columns:
                dec = wide[w].to_numpy(); s = decision_stats(dec, yy)
                s['auto_in_licensed'] = float((dec[lic != 'escalate'] != 'escalate').mean())
                s['auto_outside_licensed'] = float((dec[lic == 'escalate'] != 'escalate').mean())
                s['agrees_with_license'] = float(((dec == lic) | ((lic != 'escalate') & (dec == 'escalate'))).mean())
                Sx[w] = s
            Sx['licensed_share'] = float((lic != 'escalate').mean())
            R['S'] = Sx
        # comprehension
        Cq = {}
        for kind, key in (('CA', KEY_A), ('CB', KEY_B), ('CS', KEY_S)):
            for v in (WORDINGS if kind != 'CS' else ['v3']):
                c = d[d.condition == f'{kind}_{v}']
                if len(c):
                    Cq[f'{kind}_{v}'] = {q: float(np.mean([o[q] == a for o in c.parsed])) for q, a in key.items()} | {'n': int(len(c))}
        R['comprehension'] = Cq
        res[m] = R
    (ROOT / 'results/r6_summary.json').write_text(json.dumps(res, indent=2, default=float))
    for m, R in res.items():
        if m in ('synthetic_tool', 'plan'): continue
        print(f'\n== {m}  ${R["cost_usd"]:.2f}  {R["n_calls"]} calls')
        for sc in ('A', 'B'):
            for w, s in sorted(R.get(sc, {}).items()):
                extra = '  '.join(f"{k} {s[k]:+.3f} [{s[k + '_ci'][0]:+.3f},{s[k + '_ci'][1]:+.3f}]" for k in ('d_vs_v1', 'd_vs_pla') if k in s)
                print(f"  {sc} {w:18s} esc {s['escalate']:.2f} wrong {s['wrong']:.3f} rej {s['reject']:.2f}  {extra}")
        for w, s in R.get('S', {}).items():
            print('  S', w, s if not isinstance(s, dict) else {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})
        for k, v in R['comprehension'].items():
            print('  ', k, {q: round(x, 2) for q, x in v.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['run', 'analyse', 'plan'])
    ap.add_argument('--models', nargs='+', default=['gpt-5.4-mini-2026-03-17', 'gpt-6.1-sol', 'claude-haiku-4-5-20251001', 'claude-opus-5-5'])
    ap.add_argument('--budget', type=float, default=15.0); ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    {'run': lambda: run(a.models, a.budget, a.workers), 'analyse': analyse, 'plan': lambda: print(PLAN)}[a.cmd]()
