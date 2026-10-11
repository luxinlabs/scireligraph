"""Output-schema and decision-rule tests (added after mutation analysis exposed gaps)."""
import hashlib, json
import pytest
import verifier as V

CLAIM, AS_OF = 'Ej3y_single-trace', '2017-12-31'
KEYS = {'claim_id', 'stage', 'as_of', 'p_replicate', 'base_rate', 'cv_auroc', 'cv_brier_skill', 'selective_ok',
        'evidence_cutoff', 'labels_as_of', 'n_evidence', 'tiers_used', 'n_train_claims', 'drivers', 'provenance', 'advice'}


@pytest.fixture(scope='module')
def out_pub(base, tmp_path_factory):
    # a synthetic early registration date makes the launch stage answerable, so its output can be checked
    v = V.ReliabilityVerifier(audit_log=tmp_path_factory.mktemp('a') / 'log.jsonl', claims=base['claims'].copy(),
                              evidence=base['evidence'], design_dates={CLAIM: '2000-01-01'})
    return v, v.verify_claim(CLAIM, AS_OF, 'publication'), v.verify_claim(CLAIM, AS_OF, 'launch')


def test_schema_exact(out_pub):
    _, pub, lau = out_pub
    for o in (pub, lau):
        assert set(o) == KEYS
        assert set(o['provenance']) == {'archive_sha256', 'openalex_access', 'model_config', 'chosen_C'}
        assert all(set(d) == {'feature', 'logit_contribution'} for d in o['drivers']) and len(o['drivers']) == 3
        assert o['claim_id'] == CLAIM and o['as_of'] == AS_OF and o['evidence_cutoff'] == AS_OF
    assert pub['tiers_used'] == ['A', 'G'] and lau['tiers_used'] == ['A', 'D'] and pub['stage'] == 'publication'


def test_cv_fields_match_harness(out_pub):
    v, pub, lau = out_pub
    for o, cfg in ((pub, V.CONFIG['publication']), (lau, V.CONFIG['launch'])):
        cv = v.summary[cfg['name']]['mean']
        assert o['cv_auroc'] == round(cv['auroc'], 3)
        assert o['cv_brier_skill'] == round(1 - cv['brier'] / v.summary['B0 Prevalence']['mean']['brier'], 4)


def test_selective_rule(out_pub):
    """selective_ok must equal the documented rule: lower 95% bound of top-decile accuracy > overall accuracy."""
    v, pub, lau = out_pub
    for o, cfg in ((pub, V.CONFIG['publication']), (lau, V.CONFIG['launch'])):
        sel = {s['coverage']: s for s in v.selective if s['model'] == cfg['name'] and s['confidence'] == 'abs05'}
        assert o['selective_ok'] == (sel[0.1]['ci'][0] > sel[1.0]['accuracy'])
        assert o['advice'] == ('weak signal: do not gate actions on this score' if not o['selective_ok'] else 'score may be used with abstention')


def test_selective_rule_true_branch(out_pub):
    """Rule also yields True (and permissive advice) when the bound clears overall accuracy."""
    v, _, _ = out_pub
    name = V.CONFIG['publication']['name']
    patched = [dict(s, ci=[0.99, 1.0]) if s['model'] == name and s['confidence'] == 'abs05' and s['coverage'] == 0.1 else s for s in v.selective]
    v.selective = patched
    o = v.verify_claim(CLAIM, AS_OF, 'publication')
    assert o['selective_ok'] is True and o['advice'] == 'score may be used with abstention'


def test_n_evidence_and_train_count(out_pub, base):
    v, pub, _ = out_pub
    pid = base['claims'].set_index('claim_id').loc[CLAIM, 'paper_id']
    expect = sum(1 for x in base['evidence'][pid][1] if x.get('publication_date') and x['publication_date'] <= AS_OF)
    assert pub['n_evidence'] == expect
    assert pub['n_train_claims'] == int((base['claims'].paper_id != pid).sum())


def test_launch_boundary_and_day_before(base, tmp_path):
    v = V.ReliabilityVerifier(audit_log=tmp_path / 'l.jsonl', claims=base['claims'].copy(), evidence=base['evidence'])
    launch = base['claims'].set_index('claim_id').loc[CLAIM, 'launch_date']
    import datetime as dt
    day_before = (dt.date.fromisoformat(launch) - dt.timedelta(days=1)).isoformat()
    assert v.verify_claim(CLAIM, day_before)['as_of'] == day_before
    with pytest.raises(V.ContractViolation, match='on/after replication launch'):
        v.verify_claim(CLAIM, launch)


def test_unknown_claim_and_messages(base, tmp_path):
    v = V.ReliabilityVerifier(audit_log=tmp_path / 'l.jsonl', claims=base['claims'].copy(), evidence=base['evidence'])
    with pytest.raises(V.ContractViolation, match='unknown claim_id'):
        v.verify_claim('no-such-claim', AS_OF)
    with pytest.raises(V.ContractViolation, match='unknown stage'):
        v.verify_claim(CLAIM, AS_OF, 'later')
    with pytest.raises(V.ContractViolation, match='ISO date'):
        v.verify_claim(CLAIM, 'soon')


def test_audit_record_integrity(base, tmp_path):
    v = V.ReliabilityVerifier(audit_log=tmp_path / 'l.jsonl', claims=base['claims'].copy(), evidence=base['evidence'])
    o = v.verify_claim(CLAIM, AS_OF)
    rec = json.loads((tmp_path / 'l.jsonl').read_text().splitlines()[-1])
    assert rec['record_sha'] == V._sha(o) and len(rec['record_sha']) == 16
    assert {k: rec[k] for k in o} == json.loads(json.dumps(o))
    assert rec['provenance']['model_config'] == V._sha({'stage': 'publication', 'num': V.CONFIG['publication']['num'],
                                                        'cat': V.CONFIG['publication']['cat'], 'C_grid': V.CS})
    assert len(rec['ts']) >= 24 and rec['ts'][4] == '-' and 'T' in rec['ts']


def test_sha_is_order_independent():
    assert V._sha({'a': 1, 'b': 2}) == V._sha({'b': 2, 'a': 1})


def test_feature_lists_are_complete():
    from build_dataset import FLAT_G, GRAPH_G, DEP_G
    assert V.CONFIG['publication']['num'] == V.A_NUM + FLAT_G + GRAPH_G + DEP_G and V.CONFIG['publication']['cat'] == V.A_CAT
    assert V.CONFIG['launch']['num'] == V.A_NUM + V.D_NUM and V.CONFIG['launch']['cat'] == V.A_CAT + V.D_CAT
    assert V.D_NUM == ['d_new_data', 'd_log10_planned_n'] and V.D_CAT == ['d_power_target']
    assert len(V.A_NUM) == 8 and len(V.A_CAT) == 3
