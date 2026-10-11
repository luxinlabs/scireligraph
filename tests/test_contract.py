"""Contract tests for the claim-reliability verifier (R1, R3, R4, R5 and stage gating)."""
import json
import numpy as np
import pytest
import verifier as V

CLAIM = 'Ej3y_single-trace'
AS_OF = '2017-12-31'


def fake_work(i, date, author='https://openalex.org/A_FAKE'):
    return {'id': f'https://openalex.org/W_FAKE{i}', 'publication_date': date,
            'authorships': [{'author': {'id': f'{author}{i}'}}], 'referenced_works': [],
            'primary_location': {'source': {'id': f'S_FAKE{i}'}}, 'topics': []}


def test_r1_future_evidence_is_invisible(make, deep_evidence):
    """Metamorphic: adding citing works dated after as_of to EVERY paper must not change the output."""
    ref = make().verify_claim(CLAIM, AS_OF)
    ev = deep_evidence()
    for pid, (w, citing) in ev.items():
        citing.extend(fake_work(k, d) for k, d in enumerate(['2018-01-01', '2019-06-30', '2025-03-01'] * 5))
    out = make(evidence=ev).verify_claim(CLAIM, AS_OF)
    assert out['p_replicate'] == ref['p_replicate']
    assert out['n_evidence'] == ref['n_evidence']


def test_r1_past_evidence_is_visible(make, deep_evidence, base):
    """Positive control for the metamorphic test: evidence dated before as_of IS counted."""
    pid = base['claims'].set_index('claim_id').loc[CLAIM, 'paper_id']
    ref = make().verify_claim(CLAIM, AS_OF)
    ev = deep_evidence(); ev[pid][1].extend(fake_work(k, '2016-05-01') for k in range(7))
    assert make(evidence=ev).verify_claim(CLAIM, AS_OF)['n_evidence'] == ref['n_evidence'] + 7


def test_r1_refuses_on_or_after_launch(make, base):
    launch = base['claims'].set_index('claim_id').loc[CLAIM, 'launch_date']
    with pytest.raises(V.ContractViolation):
        make().verify_claim(CLAIM, launch)


@pytest.mark.parametrize('kwargs', [{'as_of': 'yesterday'}, {'as_of': AS_OF, 'stage': 'after-outcome'}])
def test_rejects_malformed_calls(make, kwargs):
    with pytest.raises(V.ContractViolation):
        make().verify_claim(CLAIM, **kwargs)


def test_r3_own_paper_labels_never_used(make, base):
    """Flipping the labels of every claim from the queried paper must not change its prediction."""
    ref = make().verify_claim(CLAIM, AS_OF)
    c = base['claims'].copy(); pid = c.set_index('claim_id').loc[CLAIM, 'paper_id']
    c.loc[c.paper_id == pid, 'replicated'] = 1 - c.loc[c.paper_id == pid, 'replicated']
    assert make(claims=c).verify_claim(CLAIM, AS_OF)['p_replicate'] == ref['p_replicate']


def test_stage_gating_publication_ignores_launch_tier(make, base):
    """At the publication stage, replication-design (Tier D) values must have no influence."""
    ref = make().verify_claim(CLAIM, AS_OF, 'publication')
    c = base['claims'].copy(); rng = np.random.default_rng(0)
    c['d_new_data'] = rng.integers(0, 2, len(c)); c['d_log10_planned_n'] = rng.normal(3, 1, len(c))
    assert make(claims=c).verify_claim(CLAIM, AS_OF, 'publication')['p_replicate'] == ref['p_replicate']


def test_r4_output_is_decision_relevant(make, base):
    out = make().verify_claim(CLAIM, AS_OF)
    pid = base['claims'].set_index('claim_id').loc[CLAIM, 'paper_id']
    assert out['base_rate'] == round(float(base['claims'][base['claims'].paper_id != pid].replicated.mean()), 4)
    assert 0 < out['p_replicate'] < 1 and isinstance(out['selective_ok'], bool)
    assert out['selective_ok'] or 'do not gate' in out['advice']


def test_r5_every_call_is_audited(make, tmp_path):
    v = make(); out = v.verify_claim(CLAIM, AS_OF)
    recs = [json.loads(l) for l in (tmp_path / 'audit.jsonl').read_text().splitlines()]
    assert len(recs) == 1 and recs[0]['p_replicate'] == out['p_replicate']
    assert recs[0]['provenance']['archive_sha256'] == v.manifest['source_sha256']
    assert {'ts', 'as_of', 'stage', 'evidence_cutoff', 'record_sha'} <= set(recs[0])
