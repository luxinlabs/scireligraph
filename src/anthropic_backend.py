#!/usr/bin/env python3
"""Anthropic (Claude) backend for the LLM-judge and agent experiments.

Adapts Messages API responses to the fields the shared Runner logs (prompt/completion tokens, text, finish reason),
so caching, budget caps, progress bars and audit logging are identical across providers.
Refusal fallbacks are deliberately NOT enabled: a fallback would silently answer with a different model and
contaminate per-model measurements. A refusal is logged as a failed (unparsed) call.
"""
import os, sys
from types import SimpleNamespace
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]

# USD per 1M tokens (standard tier) and cutoffs, from platform.claude.com model pages, retrieved 2026-10-09.
PRICES = {'claude-haiku-4-5-20251001': (1.00, 5.00), 'claude-opus-5-5': (4.00, 20.00)}
# Per-model request settings: lowest reasoning effort each model offers.
#  - Claude Opus 5.5: thinking is always on; effort is the only control -> 'low'.
#  - Claude Haiku 4.5: no effort parameter; thinking off unless budget_tokens is sent -> omit both.
SETTINGS = {'claude-opus-5-5': {'max_tokens': 8000, 'output_config': {'effort': 'low'}},
            'claude-haiku-4-5-20251001': {'max_tokens': 1024}}


def load_key():
    if os.environ.get('ANTHROPIC_API_KEY'):
        return os.environ['ANTHROPIC_API_KEY']
    for env in [ROOT / '.env', ROOT.parent / '.env']:
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip().removeprefix('export ').strip()
                if line.startswith('ANTHROPIC_API_KEY='):
                    return line.split('=', 1)[1].strip().strip('"\'')
    sys.exit('ANTHROPIC_API_KEY not found in environment or .env')


def make_client():
    return anthropic.Anthropic(api_key=load_key(), timeout=120.0, max_retries=2)


def is_anthropic(model):
    return model.startswith('claude-')


def call(client, model, msgs):
    """msgs: [{'role':'system',...}, {'role':'user',...}] -> (normalised response, params)."""
    system = '\n'.join(m['content'] for m in msgs if m['role'] == 'system')
    convo = [m for m in msgs if m['role'] != 'system']
    params = SETTINGS[model]
    resp = client.messages.create(model=model, system=system, messages=convo, **params)
    text = '' if resp.stop_reason == 'refusal' else ''.join(b.text for b in resp.content if b.type == 'text')
    norm = SimpleNamespace(
        model=resp.model, id=resp.id,
        usage=SimpleNamespace(prompt_tokens=resp.usage.input_tokens, completion_tokens=resp.usage.output_tokens,
                              completion_tokens_details=None),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=resp.stop_reason)])
    return norm, {k: v for k, v in params.items() if k != 'max_tokens'}
