#!/usr/bin/env python3
"""X2: null distribution for the graph block. Graph+dependency features are permuted across papers within
publication-year strata K times; M2 is re-evaluated on the same CV seeds each time."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as E

K, SEEDS = 30, list(range(5))
E.SEEDS = SEEDS
df, G = E.load(); y = df.replicated.to_numpy(); groups = df.paper_id.to_numpy()
X = pd.concat([df, G['strict']], axis=1)
spec = {'kind': 'lr', 'num': E.A_NUM + E.FLAT + E.GRAPH + E.DEP, 'cat': E.A_CAT}
b1 = {'kind': 'lr', 'num': E.A_NUM, 'cat': E.A_CAT}
real, _ = E.run(df, X, y, groups, spec); base, _ = E.run(df, X, y, groups, b1)
pap = df.groupby('paper_id').a_pub_year.first()
gtab = pd.concat([df[['paper_id']], G['strict']], axis=1).groupby('paper_id').first()
rng = np.random.default_rng(2026); null = []
for k in tqdm(range(K), desc='permutations'):
    pm = {}
    for _, ids in pap.groupby(pap).groups.items():
        ids = list(ids); pm.update(dict(zip(ids, rng.permutation(ids))))
    Xp = X.copy()
    src = df.paper_id.map(pm)
    for c in E.GRAPH + E.DEP: Xp[c] = gtab.loc[src, c].to_numpy()
    ps, _ = E.run(df, Xp, y, groups, spec); null.append(ps.mean().to_dict())
nd = pd.DataFrame(null)
res = {'K': K, 'seeds': len(SEEDS), 'real_auroc': float(real.auroc.mean()), 'real_brier': float(real.brier.mean()),
       'b1_auroc': float(base.auroc.mean()), 'b1_brier': float(base.brier.mean()),
       'null_auroc_mean': float(nd.auroc.mean()), 'null_auroc_q': [float(nd.auroc.quantile(q)) for q in (.025, .5, .975)],
       'null_brier_mean': float(nd.brier.mean()),
       'p_auroc': float((1 + (nd.auroc >= real.auroc.mean()).sum()) / (K + 1)),
       'p_brier': float((1 + (nd.brier <= real.brier.mean()).sum()) / (K + 1))}
(E.RES / 'permutation_null.json').write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
