#!/usr/bin/env python3
"""Stage-2 re-check, parallel and isolated: each worker owns a copy of src/ and tests/ (data, results and .venv are
symlinked), so the project's verifier.py is never modified. Only tests/test_schema.py is run, because every mutant here
already passed the original contract tests in stage 1 (results/mutmut_reverify.json)."""
import json, os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'
args = [a for a in sys.argv[1:] if not a.startswith('--')]; SCRATCH = Path(args[0]); W = int(args[1]) if len(args) > 1 else 4
# stage 2 (default): survivors of stage 1 vs tests/test_schema.py; stage 3 (--stage3): survivors of stage 2 vs tests/test_boundary.py
STAGE3 = '--stage3' in sys.argv
if STAGE3:
    prev = json.loads((R / 'mutmut_reverify_stage2.json').read_text()); ids = [k for k, v in prev.items() if not v.get('killed')]
    TEST_FILE, OUT_NAME = 'tests/test_boundary.py', 'mutmut_reverify_stage3.json'
else:
    prev = json.loads((R / 'mutmut_reverify.json').read_text()); ids = [k for k, v in prev.items() if not v['killed']]
    TEST_FILE, OUT_NAME = 'tests/test_schema.py', 'mutmut_reverify_stage2.json'
env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
dirs = []
for w in range(W):
    d = SCRATCH / f'mw{w}'; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    shutil.copytree(ROOT / 'src', d / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'tests', d / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy(ROOT / '.mutmut-cache', d / '.mutmut-cache')
    for name in ('data', 'results', '.venv'): (d / name).symlink_to(ROOT / name)
    dirs.append(d)

def check(args):
    w, i = args; d = dirs[w]
    shutil.copy(R / 'verifier.py.bak', d / 'src/verifier.py')
    subprocess.run([str(ROOT / '.venv/bin/mutmut'), 'apply', i], cwd=d, check=True, capture_output=True)
    try:
        r = subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'pytest', '-x', '-q', '-p', 'no:cacheprovider', TEST_FILE],
                           cwd=d, env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return i, {'killed': None, 'timeout': True, 'exit_code': None, 'failing': []}
    failed = [l.split('::')[1].split(' ')[0] for l in r.stdout.splitlines() if l.startswith(('FAILED', 'ERROR'))]
    return i, {'killed': r.returncode != 0, 'exit_code': r.returncode, 'failing': failed}

out = {}
queues = [[(w, i) for i in ids[w::W]] for w in range(W)]
def run_queue(q):
    return [check(a) for a in q]
with ThreadPoolExecutor(W) as ex, tqdm(total=len(ids), desc='stage-2 re-check') as bar:
    futs = []
    for q in queues:
        def job(q=q):
            res = []
            for a in q:
                res.append(check(a)); bar.update(1)
            return res
        futs.append(ex.submit(job))
    for f in futs:
        for i, v in f.result(): out[i] = v
(R / OUT_NAME).write_text(json.dumps(out, indent=1))
k = sum(1 for v in out.values() if v['killed']); t = sum(1 for v in out.values() if v.get('timeout')); print(f'{len(out)} stage-2 mutants: {k} killed, {t} timeout, {len(out) - k - t} survive')
