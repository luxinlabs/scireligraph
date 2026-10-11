#!/usr/bin/env python3
"""Mutation testing of the verifier contract: each mutant removes one guarantee; the suite must kill it."""
import json, os, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MUTANTS = ['leaky_cutoff', 'no_leave_paper_out', 'no_stage_gating', 'no_refusal', 'no_audit']
res = {}
for m in [''] + MUTANTS:
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'], cwd=ROOT,
                       env={**os.environ, 'MUTANT': m}, capture_output=True, text=True)
    failed = sorted(set(re.findall(r'FAILED tests/test_contract.py::(\w+)', r.stdout)))
    summ = r.stdout.strip().splitlines()[-1]
    res[m or 'original'] = {'killed': bool(failed) if m else None, 'failing_tests': failed, 'summary': summ}
    print(f"{m or 'original':20s} {summ:40s} {failed}", flush=True)
(ROOT / 'results/mutation.json').write_text(json.dumps(res, indent=2))
