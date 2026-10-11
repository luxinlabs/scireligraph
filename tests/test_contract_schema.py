"""The tool's answers and refusals conform to the reusable epistemic-contract schema (contract/)."""
import json
import jsonschema
import pytest
import verifier as V

SCHEMA = json.loads((V.ROOT / 'contract' / 'epistemic_contract.schema.json').read_text())
CLAIM, AS_OF = 'Ej3y_single-trace', '2017-12-31'


@pytest.mark.parametrize('stage', ['publication', 'launch'])
def test_answer_conforms(make, stage):
    out = make(design_dates={CLAIM: '2000-01-01'}).verify_claim(CLAIM, AS_OF, stage)
    jsonschema.validate(out, SCHEMA, format_checker=jsonschema.FormatChecker())
    assert out['evidence_cutoff'] == out['as_of']


def test_refusal_conforms(make, base):
    launch = base['claims'].set_index('claim_id').loc[CLAIM, 'launch_date']
    with pytest.raises(V.ContractViolation) as e:
        make().verify_claim(CLAIM, launch)
    jsonschema.validate({'error': 'contract_violation', 'detail': str(e.value)}, SCHEMA)
