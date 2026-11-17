#!/usr/bin/env python3
"""Per-condition accounting of the pre-specified R6 runs (results/r6/*.jsonl) -> results/r6_call_accounting.csv.
Condition codes (src/r6_experiment.py): A_v* = scenario-A decisions under wording v2-v4 (_p2: prompt 2); B_v* = refusals
(scenario B); S_bare / S_v3 = synthetic decision-ready tool; CA_* / CB_* / CS_v3 = comprehension questions."""
from pathlib import Path
import json, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = [{'model': f.stem, 'condition': json.loads(l)['condition']}
        for f in sorted((ROOT / 'results/r6').glob('*.jsonl')) for l in f.open() if l.strip()]
t = pd.crosstab(pd.DataFrame(rows).condition, pd.DataFrame(rows).model, margins=True, margins_name='total')
t.to_csv(ROOT / 'results/r6_call_accounting.csv')
print(t.to_string())
