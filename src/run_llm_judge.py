#!/usr/bin/env python3
"""LLM-as-judge verifier + memorization probe, with caching, resume, budget cap and a live progress bar.

    python src/run_llm_judge.py --models gpt-5.4-mini-2026-03-17 gpt-6.1-sol --limit 5      # pilot
    python src/run_llm_judge.py --models gpt-5.4-mini-2026-03-17 gpt-6.1-sol --budget 20    # full run

Conditions per claim: forecast_id (identified), forecast_deid (statistics only), probe (memorization).
Probe decoys: 100 journal-matched SCORE-sampled papers that were never replicated (probe condition only).
Every response (raw text, usage, cost, request params) is appended to results/llm/<model>.jsonl; a task whose
cached record parsed successfully is never re-sent, so re-running resumes without double billing.
The API key is read from the environment or from ../.env and is never logged.
"""
import argparse, hashlib, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from llm_prompts import records, decoy_records, forecast_messages, forecast_deid_messages, probe_messages  # noqa: E402
import anthropic_backend as AB  # noqa: E402

OUT = ROOT / 'results' / 'llm'
# USD per 1M tokens (standard tier), from platform.openai.com/docs/models/<model>, retrieved 2026-10-08
PRICES = {'gpt-6.1-sol': (2.00, 10.00), 'gpt-5.4-mini-2026-03-17': (0.75, 4.50), **AB.PRICES}
CONDITIONS = {'forecast_id': forecast_messages, 'forecast_deid': forecast_deid_messages, 'probe': probe_messages}
MAX_COMPLETION_TOKENS = 2000


def load_key():
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    for env in [ROOT / '.env', ROOT.parent / '.env']:
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip().removeprefix('export ').strip()
                if line.startswith('OPENAI_API_KEY='):
                    return line.split('=', 1)[1].strip().strip('"\'')
    sys.exit('OPENAI_API_KEY not found in environment or .env')


def sha(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:16]


def parse(text, condition):
    """Strict JSON parse; returns (value_dict, error)."""
    try:
        t = text.strip()
        if t.startswith('```'):
            t = t.strip('`').split('\n', 1)[1] if '\n' in t else t
        obj = json.loads(t[t.find('{'): t.rfind('}') + 1])
    except Exception as e:
        return None, f'json: {e}'
    if condition.startswith('forecast'):
        p = obj.get('p_replicate')
        if not isinstance(p, (int, float)) or not 0 <= p <= 1:
            return None, f'bad p_replicate: {p!r}'
    else:
        if obj.get('aware_of_replication') not in ('yes', 'no', 'unsure'):
            return None, f"bad aware_of_replication: {obj.get('aware_of_replication')!r}"
    return obj, None


class Runner:
    def __init__(self, client, model, budget, spent_lock):
        self.client, self.model, self.budget, self.lock = client, model, budget, spent_lock
        self.file = OUT / f'{model}.jsonl'
        self.done = {}
        if self.file.exists():
            for line in self.file.read_text().splitlines():
                r = json.loads(line)
                if r.get('parsed') is not None:
                    self.done[(r['claim_id'], r['condition'], r['prompt_sha'])] = r
        self.params = None  # discovered on first call (reasoning-effort support differs by model)

    def _call(self, msgs):
        if AB.is_anthropic(self.model):
            resp, extra = AB.call(self.client, self.model, msgs); self.params = extra
            return resp, extra
        candidates = [{'reasoning_effort': e} for e in ('none', 'minimal', 'low')] + [{}]
        if self.params is not None:
            candidates = [self.params]
        last = None
        for extra in candidates:
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=msgs, response_format={'type': 'json_object'},
                                                           max_completion_tokens=MAX_COMPLETION_TOKENS, **extra)
                self.params = extra
                return resp, extra
            except Exception as e:  # unsupported parameter -> try the next setting; other errors propagate after retries
                last = e
                if 'reasoning_effort' not in str(e) and 'unsupported' not in str(e).lower():
                    raise
        raise last

    def task(self, rec, condition, spent):
        msgs = CONDITIONS[condition](rec)
        key = (rec['claim_id'], condition, sha(msgs))
        if key in self.done:
            return 0.0, 'cached'
        with self.lock:
            if spent[0] >= self.budget:
                return 0.0, 'budget'
        for attempt in range(6):
            try:
                resp, extra = self._call(msgs)
                break
            except Exception as e:
                if attempt == 5:
                    self._write({'claim_id': rec['claim_id'], 'condition': condition, 'prompt_sha': key[2], 'parsed': None,
                                 'error': f'api: {type(e).__name__}: {str(e)[:300]}'})
                    return 0.0, 'error'
                time.sleep(2 ** attempt)
        u = resp.usage
        out_tok = u.completion_tokens
        reasoning = getattr(getattr(u, 'completion_tokens_details', None), 'reasoning_tokens', None)
        pi, po = PRICES[self.model]
        cost = u.prompt_tokens / 1e6 * pi + out_tok / 1e6 * po
        text = resp.choices[0].message.content or ''
        parsed, err = parse(text, condition)
        self._write({'claim_id': rec['claim_id'], 'paper_id': rec['paper_id'], 'decoy': bool(rec.get('decoy')),
                     'condition': condition, 'prompt_sha': key[2],
                     'model': self.model, 'response_model': resp.model, 'response_id': resp.id, 'params': extra,
                     'finish_reason': resp.choices[0].finish_reason, 'raw': text, 'parsed': parsed, 'error': err,
                     'prompt_tokens': u.prompt_tokens, 'completion_tokens': out_tok, 'reasoning_tokens': reasoning,
                     'cost_usd': cost, 'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        with self.lock:
            spent[0] += cost
        return cost, 'ok' if parsed is not None else 'parse_error'

    def _write(self, rec):
        with self.lock, open(self.file, 'a') as f:
            f.write(json.dumps(rec) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--limit', type=int, default=None, help='first N claims only (pilot)')
    ap.add_argument('--budget', type=float, default=20.0, help='hard stop in USD across all models in this run')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    for m in args.models:
        if m not in PRICES:
            sys.exit(f'no price recorded for {m}; add it to PRICES first')
    from openai import OpenAI
    oa_client = None; an_client = None
    OUT.mkdir(parents=True, exist_ok=True)
    recs = records()[: args.limit] if args.limit else records()
    decoys = decoy_records()[: max(2, args.limit // 2)] if args.limit else decoy_records()
    lock, spent, summary = threading.Lock(), [0.0], {}
    for model in args.models:
        if AB.is_anthropic(model):
            an_client = an_client or AB.make_client(); client = an_client
        else:
            oa_client = oa_client or OpenAI(api_key=load_key(), timeout=60, max_retries=2); client = oa_client
        runner = Runner(client, model, args.budget, lock)
        jobs = [(r, c) for r in recs for c in CONDITIONS] + [(d, 'probe') for d in decoys]
        counts = {}
        bar = tqdm(total=len(jobs), desc=model, unit='call')
        with ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(runner.task, r, c, spent) for r, c in jobs]
            for f in as_completed(futs):
                cost, status = f.result()
                counts[status] = counts.get(status, 0) + 1
                bar.update(1); bar.set_postfix(spent=f'${spent[0]:.3f}', **{k: v for k, v in counts.items() if k != 'ok'})
        bar.close()
        summary[model] = {**counts, 'params': runner.params}
        if spent[0] >= args.budget:
            print(f'BUDGET REACHED (${spent[0]:.2f} >= ${args.budget:.2f}); remaining calls were skipped.')
    print(json.dumps({'spent_usd_this_run': round(spent[0], 4), 'per_model': summary}, indent=1))


if __name__ == '__main__':
    main()
