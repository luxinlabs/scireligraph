#!/usr/bin/env python3
"""Claim-reliability verifier tool that enforces the contract (R1-R5) at call time.

    v = ReliabilityVerifier()
    v.verify_claim('Ej3y_single-trace', as_of='2018-12-31', stage='publication')

Contract enforced on every call:
  R1  evidence for every claim is recomputed from the raw OpenAlex cache, filtered to `as_of`;
      calls whose `as_of` is on/after the claim's replication launch are refused (outcome may be public).
  R2  labels are the SCORE primary criterion validated at build time (manifest).
  R3  the model is refit without ANY claim from the queried claim's paper (leave-paper-out).
  R4  the output carries the base rate, cross-validated skill and the selective-accuracy verdict.
  R5  every call is appended to an audit log with inputs, outputs and provenance hashes.
"""
import hashlib, json, sys, time
from datetime import date
from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from build_dataset import graph_features, oa_stem, FLAT_G, GRAPH_G, DEP_G  # noqa: E402

A_NUM = ['a_log10_n', 'a_neglog10_p', 'a_p_marginal', 'a_p_exact', 'a_abs_r', 'a_power50', 'a_interaction', 'a_pub_year']
A_CAT = ['a_stat_type', 'a_analysis', 'a_field']
D_NUM, D_CAT = ['d_new_data', 'd_log10_planned_n'], ['d_power_target']
CONFIG = {'publication': {'name': 'M2 Metadata + graph + dependency correction', 'tiers': ['A', 'G'],
                          'num': A_NUM + FLAT_G + GRAPH_G + DEP_G, 'cat': A_CAT},
          'launch': {'name': 'D Metadata + replication design (launch-time)', 'tiers': ['A', 'D'],
                     'num': A_NUM + D_NUM, 'cat': A_CAT + D_CAT}}
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


class ContractViolation(ValueError):
    """Raised when a call would break the verifier contract."""


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class ReliabilityVerifier:
    def __init__(self, root=ROOT, audit_log=None, evidence=None, claims=None):
        self.root = Path(root)
        self.claims = (claims if claims is not None else pd.read_csv(self.root / 'data/processed/claims.csv')).reset_index(drop=True)
        self.manifest = json.loads((self.root / 'data/processed/manifest.json').read_text())
        self.summary = json.loads((self.root / 'results/summary.json').read_text())['models']
        self.selective = json.loads((self.root / 'results/selective.json').read_text())
        self.evidence = evidence if evidence is not None else self._load_evidence()
        self.audit_log = Path(audit_log) if audit_log else self.root / 'results/audit_log.jsonl'
        self._cache = {}

    # ---------- evidence ----------
    def _load_evidence(self):
        ev = {}
        for pid in self.claims.paper_id.unique():
            w = json.loads((self.root / f'data/openalex/works/{oa_stem(pid)}.json').read_text())
            cf = self.root / f'data/openalex/citing/{oa_stem(pid)}.jsonl'
            citing = [json.loads(l) for l in cf.read_text().splitlines() if l.strip()] if cf.exists() else []
            ev[pid] = (w, citing)
        return ev

    def _evidence_cutoff(self, as_of):
        return as_of  # mutation hook: a leaky verifier would return the access date here

    def _graph_table(self, as_of):
        cut = self._evidence_cutoff(as_of)
        rows = []
        for _, r in self.claims.iterrows():
            w, citing = self.evidence[r.paper_id]
            rows.append(graph_features(w, citing, cut, r.a_pub_year))
        return pd.DataFrame(rows, index=self.claims.index)

    def _n_evidence(self, paper_id, as_of):
        cut = self._evidence_cutoff(as_of)
        return sum(1 for x in self.evidence[paper_id][1] if (x.get('publication_date') or '9999') <= cut)

    def _check_before_launch(self, as_of, launch):
        if as_of >= launch:  # mutation hook: refusal guard
            raise ContractViolation(f'as_of {as_of} is on/after replication launch {launch}: evidence may reflect the outcome')

    # ---------- model ----------
    def _training_mask(self, paper_id):
        return (self.claims.paper_id != paper_id).to_numpy()  # mutation hook: leave-paper-out

    def _features(self, stage):
        return CONFIG[stage]['num'], CONFIG[stage]['cat']  # mutation hook: stage gating

    def _fit(self, X, y, groups, num, cat):
        pre = ColumnTransformer([
            ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), num),
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), cat)])
        pipe = Pipeline([('pre', pre), ('lr', LogisticRegression(max_iter=5000))])
        inner = list(GroupKFold(n_splits=3).split(X, y, groups))
        return GridSearchCV(pipe, {'lr__C': CS}, scoring='neg_log_loss', cv=inner).fit(X, y).best_estimator_

    # ---------- public API ----------
    def verify_claim(self, claim_id, as_of, stage='publication'):
        if stage not in CONFIG:
            raise ContractViolation(f'unknown stage {stage!r}; expected one of {sorted(CONFIG)}')
        try:
            as_of = date.fromisoformat(str(as_of)).isoformat()
        except ValueError as e:
            raise ContractViolation(f'as_of must be an ISO date: {e}')
        hit = self.claims.index[self.claims.claim_id == claim_id]
        if len(hit) != 1:
            raise ContractViolation(f'unknown claim_id {claim_id!r}')
        i = hit[0]; row = self.claims.loc[i]
        launch = row.launch_date if isinstance(row.launch_date, str) else self.manifest['earliest_replication_start']
        self._check_before_launch(as_of, launch)

        num, cat = self._features(stage)
        G = self._cache.get(as_of)
        if G is None:
            G = self._cache[as_of] = self._graph_table(as_of)
        X = pd.concat([self.claims, G], axis=1)
        tr = self._training_mask(row.paper_id)
        y = self.claims.replicated.to_numpy()
        model = self._fit(X[tr], y[tr], self.claims.paper_id.to_numpy()[tr], num, cat)
        p = float(model.predict_proba(X.loc[[i]])[0, 1])

        # drivers: per-feature contribution to the logit (coefficient x transformed value)
        pre, lr = model.named_steps['pre'], model.named_steps['lr']
        z = pre.transform(X.loc[[i]]); z = z.toarray() if hasattr(z, 'toarray') else z
        contrib = z[0] * lr.coef_[0]; names = pre.get_feature_names_out()
        top = np.argsort(-np.abs(contrib))[:3]
        drivers = [{'feature': names[k].split('__', 1)[-1], 'logit_contribution': round(float(contrib[k]), 3)} for k in top]

        cfg = CONFIG[stage]; cv = self.summary[cfg['name']]
        sel = {s['coverage']: s for s in self.selective if s['model'] == cfg['name'] and s.get('confidence', 'abs05') == 'abs05'}
        out = {
            'claim_id': claim_id, 'stage': stage, 'as_of': as_of,
            'p_replicate': round(p, 4), 'base_rate': round(float(y[tr].mean()), 4),
            'cv_auroc': round(cv['mean']['auroc'], 3),
            'cv_brier_skill': round(1 - cv['mean']['brier'] / self.summary['B0 Prevalence']['mean']['brier'], 4),
            # gating is supported only if the most confident decile is reliably MORE accurate than the full output:
            # lower 95% bound of top-decile accuracy above overall accuracy (paper-clustered bootstrap)
            'selective_ok': bool(0.1 in sel and 1.0 in sel and sel[0.1]['ci'][0] > sel[1.0]['accuracy']),
            'evidence_cutoff': as_of, 'n_evidence': self._n_evidence(row.paper_id, as_of), 'tiers_used': cfg['tiers'],
            'n_train_claims': int(tr.sum()), 'drivers': drivers,
            'provenance': {'archive_sha256': self.manifest['source_sha256'], 'openalex_access': self.manifest['openalex_access'],
                           'model_config': _sha({'stage': stage, 'num': num, 'cat': cat, 'C_grid': CS}),
                           'chosen_C': float(lr.C)},
        }
        out['advice'] = ('weak signal: do not gate actions on this score' if not out['selective_ok']
                         else 'score may be used with abstention')
        self._audit(out)
        return out

    def _audit(self, out):
        rec = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **out, 'record_sha': _sha(out)}
        with open(self.audit_log, 'a') as f:
            f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    v = ReliabilityVerifier()
    print(json.dumps(v.verify_claim(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'publication'), indent=2))
