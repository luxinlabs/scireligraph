# Epistemic contract — reusable specification

`epistemic_contract.schema.json` specifies what a contract-compliant claim-verifier response must contain
(an answer or a refusal). Any verifier tool can adopt it; the SciReliGraph MCP tool is one implementation,
and `tests/test_contract_schema.py` checks the tool against it.

| Requirement | What the response must make checkable | Test relation (tool-agnostic) |
|---|---|---|
| R1 temporal validity | `as_of`, `evidence_cutoff`; refusal after the outcome may be public | Metamorphic: evidence dated after `as_of` must not change the output; evidence dated exactly `as_of` must count; refusal at/after the launch date |
| R2 label validity | (build time) labels reconcile to published counts | Assertion against published totals before any model is fitted |
| R3 honest generalisation | `n_train_claims` excludes the queried paper | Isolation: flipping every label of the queried paper must not change its prediction |
| R4 decision-relevant output | `base_rate`, `cv_auroc`, `cv_brier_skill`, `selective_ok`, `advice` | The gating verdict equals the documented rule; advice is consistent with it |
| R5 auditability | `provenance` (archive hash, access date, config hash) | Every call appends an audit record whose hash matches the response |
| R6 consumability (proposed) | `advice` states the licensed action | Evaluate with the consuming agents: no-tool, placebo and contract arms |

To adopt it for another verifier: implement the response shape, wire the five test relations to your tool's
evidence store and label source, and run mutation analysis on the result to check the tests' adequacy.
