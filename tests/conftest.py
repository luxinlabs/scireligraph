"""Shared fixtures. If MUTANT is set, a deliberately broken verifier is installed (see src/mutation_test.py)."""
import copy, os, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import verifier as V  # noqa: E402

MUTANTS = {
    'leaky_cutoff': ('_evidence_cutoff', lambda self, as_of: '2026-12-31'),
    'no_leave_paper_out': ('_training_mask', lambda self, pid: (self.claims.paper_id == self.claims.paper_id).to_numpy()),
    'no_stage_gating': ('_features', lambda self, stage: (V.CONFIG[stage]['num'] + V.D_NUM, V.CONFIG[stage]['cat'])),
    'no_refusal': ('_check_before_launch', lambda self, as_of, launch: None),
    'no_audit': ('_audit', lambda self, out: None),
}
if os.environ.get('MUTANT'):
    name, fn = MUTANTS[os.environ['MUTANT']]
    setattr(V.ReliabilityVerifier, name, fn)

_BASE = {}


@pytest.fixture(scope='session')
def base():
    if not _BASE:
        v = V.ReliabilityVerifier(audit_log=os.devnull)
        _BASE.update(claims=v.claims, evidence=v.evidence)
    return _BASE


@pytest.fixture
def make(base, tmp_path):
    def _make(claims=None, evidence=None, design_dates=None):
        return V.ReliabilityVerifier(audit_log=tmp_path / 'audit.jsonl',
                                     claims=(claims if claims is not None else base['claims']).copy(),
                                     evidence=evidence if evidence is not None else base['evidence'],
                                     design_dates=design_dates)
    return _make


@pytest.fixture
def deep_evidence(base):
    return lambda: copy.deepcopy(base['evidence'])
