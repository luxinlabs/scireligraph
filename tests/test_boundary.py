"""Boundary and oracle tests added after mutation analysis exposed residual gaps (cutoff-day inclusion,
missing-launch fallback, default paths, evidence loading checked against the raw cache)."""
import datetime as dt, json
import pytest
import verifier as V
from build_dataset import oa_stem

CLAIM, AS_OF = 'Ej3y_single-trace', '2017-12-31'


def fake(i, date):
    return {'id': f'https://openalex.org/W_EDGE{i}', 'publication_date': date,
            'authorships': [{'author': {'id': f'https://openalex.org/A_EDGE{i}'}}], 'referenced_works': [],
            'primary_location': {'source': {'id': f'S_EDGE{i}'}}, 'topics': []}


def test_evidence_dated_exactly_on_cutoff_is_admitted(make, deep_evidence, base):
    """R1 boundary: E(c,t) is evidence dated <= t, so a work dated t counts and a work dated t+1 does not."""
    pid = base['claims'].set_index('claim_id').loc[CLAIM, 'paper_id']
    ref = make().verify_claim(CLAIM, AS_OF)['n_evidence']
    nxt = (dt.date.fromisoformat(AS_OF) + dt.timedelta(days=1)).isoformat()
    ev = deep_evidence(); ev[pid][1].extend([fake(1, AS_OF), fake(2, nxt)])
    assert make(evidence=ev).verify_claim(CLAIM, AS_OF)['n_evidence'] == ref + 1


def test_missing_launch_date_falls_back_to_earliest_replication_start(make, base):
    """Claims without a recorded launch date are refused from the earliest replication start in the archive onward."""
    c = base['claims']; nolaunch = c[c.launch_date.isna()].claim_id.iloc[0]
    v = make(); earliest = v.manifest['earliest_replication_start']
    day_before = (dt.date.fromisoformat(earliest) - dt.timedelta(days=1)).isoformat()
    assert v.verify_claim(nolaunch, day_before)['as_of'] == day_before
    with pytest.raises(V.ContractViolation, match=earliest):
        v.verify_claim(nolaunch, earliest)


def test_default_audit_log_path_value(tmp_path, base):
    v = V.ReliabilityVerifier(claims=base['claims'].copy(), evidence=base['evidence'])
    assert v.audit_log == V.ROOT / 'results' / 'audit_log.jsonl'


def test_evidence_loading_matches_raw_cache(base):
    """Independent oracle: the verifier's loaded evidence equals the raw cache files for sampled papers."""
    v = V.ReliabilityVerifier.__new__(V.ReliabilityVerifier); v.root = V.ROOT; v.claims = base['claims']
    loaded = v._load_evidence()
    for pid in list(base['claims'].paper_id.unique())[:5]:
        f = V.ROOT / 'data/openalex/citing' / f'{oa_stem(pid)}.jsonl'
        raw = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
        assert len(loaded[pid][1]) == len(raw) > 0 or len(raw) == 0
        assert [x['id'] for x in loaded[pid][1]] == [x['id'] for x in raw]
