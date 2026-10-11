#!/usr/bin/env python3
"""Independently re-check every mutant mutmut recorded as surviving/suspicious.

For each: restore pristine verifier.py, `mutmut apply <id>`, delete __pycache__, run the contract tests with bytecode
writing disabled, record the exit code, restore and checksum-verify. Writes results/mutmut_reverify.json.
"""
import hashlib, json, os, shutil, sqlite3, subprocess, sys
from pathlib import Path
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'
SRC, BAK = ROOT / 'src/verifier.py', R / 'verifier.py.bak'
want = hashlib.sha1(BAK.read_bytes()).hexdigest()
# stage 1 (default): every mutant mutmut recorded as surviving; stage 2 (--stage2): mutants still surviving after stage 1
STAGE2 = '--stage2' in sys.argv
if STAGE2:
    prev = json.loads((R / 'mutmut_reverify.json').read_text()); ids = [int(k) for k, v in prev.items() if not v['killed']]
else:
    ids = [i for i, st in sqlite3.connect(ROOT / '.mutmut-cache').execute('select id, status from Mutant')
           if st in ('bad_survived', 'ok_suspicious')]
env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
out = {}
try:
    for i in tqdm(ids, desc='re-verify survivors'):
        shutil.copy(BAK, SRC)
        for d in (ROOT / 'src/__pycache__', ROOT / 'tests/__pycache__'): shutil.rmtree(d, ignore_errors=True)
        subprocess.run([str(ROOT / '.venv/bin/mutmut'), 'apply', str(i)], cwd=ROOT, check=True, capture_output=True)
        changed = hashlib.sha1(SRC.read_bytes()).hexdigest() != want
        r = subprocess.run([sys.executable, '-m', 'pytest', '-x', '-q', '-p', 'no:cacheprovider', 'tests'], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=600)
        failed = [l.split('::')[1].split(' ')[0] for l in r.stdout.splitlines() if l.startswith('FAILED') or l.startswith('ERROR')]
        out[str(i)] = {'mutation_applied': changed, 'exit_code': r.returncode, 'killed': r.returncode != 0, 'failing': failed,
                       'last_line': r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}
finally:
    shutil.copy(BAK, SRC)
    for d in (ROOT / 'src/__pycache__', ROOT / 'tests/__pycache__'): shutil.rmtree(d, ignore_errors=True)
    assert hashlib.sha1(SRC.read_bytes()).hexdigest() == want, 'verifier.py NOT restored'
(R / ('mutmut_reverify_stage2.json' if STAGE2 else 'mutmut_reverify.json')).write_text(json.dumps(out, indent=1))
k = sum(v['killed'] for v in out.values()); print(f'{len(out)} re-checked: {k} actually killed, {len(out) - k} truly survive; source restored OK')
