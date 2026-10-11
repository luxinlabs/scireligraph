#!/usr/bin/env python3
"""Summarise mutation analysis of src/verifier.py.

Three numbers are reported, never mixed:
  tool_raw      : mutmut 2.5.1's own verdicts (it under-counted kills in our environment; see reverify)
  verified_v1   : every mutmut 'survivor' re-run independently against the original 9 contract tests
  verified_v2   : mutants surviving v1 re-run against the 10 tests added after the analysis (tests/test_schema.py)
Remaining survivors are classified by the source region they mutate.
"""
import json, re, sqlite3, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'
con = sqlite3.connect(ROOT / '.mutmut-cache')
rows = dict(con.execute('select id, status from Mutant').fetchall())
total = len(rows)
tool_killed = sum(1 for s in rows.values() if s == 'ok_killed')
v1 = json.loads((R / 'mutmut_reverify.json').read_text())
v1_killed = tool_killed + sum(1 for v in v1.values() if v['killed'])
v2 = json.loads((R / 'mutmut_reverify_stage2.json').read_text()) if (R / 'mutmut_reverify_stage2.json').exists() else {}
v2_killed = sum(1 for v in v2.values() if v.get('killed')); v2_timeout = sum(1 for v in v2.values() if v.get('timeout'))
v3 = json.loads((R / 'mutmut_reverify_stage3.json').read_text()) if (R / 'mutmut_reverify_stage3.json').exists() else {}
v3_killed = sum(1 for v in v3.values() if v.get('killed'))
if v3:
    survivors = [k for k, v in v3.items() if not v.get('killed') and not v.get('timeout')]
else:
    survivors = [k for k, v in v2.items() if not v.get('killed') and not v.get('timeout')] if v2 else [k for k, v in v1.items() if not v['killed']]

src = (R / 'verifier.py.bak').read_text().splitlines()


def region(mid):
    d = subprocess.run([str(ROOT / '.venv/bin/mutmut'), 'show', str(mid)], cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r'@@ -(\d+)', d); line = int(m.group(1)) + 3 if m else 0  # hunk starts 3 lines before the change
    changed = '\n'.join(l for l in d.splitlines() if l.startswith('+') and not l.startswith('+++'))
    if line >= 160 or '__main__' in changed or 'sys.argv' in changed: return 'cli_entry_point'
    if line <= 26: return 'import_path_setup'
    if re.search(r'CS = |max_iter|n_splits|min_frequency|add_indicator|inner = ', changed): return 'model_hyperparameters'
    if re.search(r"'(XX)?(num|cat|pre|lr)(XX)?'", changed) and ('make_pipeline' in changed or 'OneHotEncoder' in changed): return 'pipeline_step_names'
    if 'drivers' in changed or 'contrib' in changed or 'argsort' in changed or 'toarray' in changed or 'split(' in changed: return 'explanation_drivers'
    if '_cache' in changed or 'G = None' in changed: return 'caching'
    return 'other'

classes = {}
detail = {}
for k in survivors:
    c = region(k); classes[c] = classes.get(c, 0) + 1; detail[k] = c
triage = json.loads((R / 'mutmut_triage.json').read_text()) if (R / 'mutmut_triage.json').exists() else {}
if triage:
    classes = {}
    for k in survivors:
        c = triage.get(k, {}).get('class', detail[k]); classes[c] = classes.get(c, 0) + 1
final_killed = v1_killed + v2_killed
summary = {'tool': 'mutmut 2.5.1', 'target': 'src/verifier.py', 'total': total,
           'tool_raw': {'killed': tool_killed, 'score': tool_killed / total},
           'verified_v1': {'tests': 9, 'killed': v1_killed, 'score': v1_killed / total, 'tool_missed_kills': v1_killed - tool_killed},
           'verified_v2': ({'tests': 19, 'killed': final_killed, 'timeout': v2_timeout, 'score': final_killed / total,
                            'score_counting_timeouts_as_killed': (final_killed + v2_timeout) / total,
                            'score_excluding_equivalent': final_killed / (total - classes.get('equivalent', 0))} if v2 else None),
           'verified_v3': ({'tests': 23, 'killed': final_killed + v3_killed, 'score': (final_killed + v3_killed) / total,
                            'score_excluding_equivalent': (final_killed + v3_killed) / (total - classes.get('equivalent', 0))} if v3 else None),
           'remaining_survivors': len(survivors), 'survivor_classes': classes, 'survivor_detail': detail}
(R / 'mutmut_summary.json').write_text(json.dumps(summary, indent=2)); print(json.dumps({k: v for k, v in summary.items() if k != 'survivor_detail'}, indent=2))
