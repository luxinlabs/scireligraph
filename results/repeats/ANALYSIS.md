# Reviewer follow-up analysis (`src/followup.py`)

The camera-ready paper now reports the cached follow-ups for Reviewer 1 W3
(repeated agent decisions) and W5 (stronger judge settings), also relevant to
Reviewer 2 m1. These are exploratory additions; they are separate from the
pre-specified R6 wording study. No new API calls were made during this revision.

## Reproduce

From the submission-package directory, run:

```sh
python src/followup.py
```

The script requires NumPy and scikit-learn. It reads the original caches in
`results/agent/` and `results/llm/`, the repeats in `results/repeats/`, and
`data/processed/claims.csv`. It writes `results/followup_summary.json`
and the three LaTeX tables `paper/tables/followup_*.tex` (not tracked).
The JSON records SHA-256 hashes of every input, individual-run results, claim
subsets, coverage, costs, and paper-clustered bootstrap intervals. The original result tables and caches are unchanged.

The validation checks successful-cache uniqueness, distinct response IDs,
returned model names, paper/claim alignment, raw JSON versus cached parses,
legal probabilities/actions, identical prompt hashes, and identical agent
request settings across repeats. Invalid/API-error calls are never counted as
agent decisions or judge predictions.

## Agent repeatability: W3 substantially addressed within scenario A

The original run and two additional runs use the same prompt and request
settings for the bare-number and original-contract arms in scenario A.
All 274 claims are now complete for all four agents across both arms and all
three runs. Opus's initially missing responses were subsequently recovered;
the log retains 86 earlier failed attempts, which are excluded from decisions.
All 4,384 successful additional agent calls enter the matched analysis. Costs
include all successful additional calls ($6.18).

Haiku's contract-arm wrong-decision rate is 52.6--54.0%, versus 17.2--17.9%
with the bare number. The other three models retain lower wrong-decision
rates with the contract in each observed run. These are empirical run ranges,
not confidence intervals. The paper and summary also supply bootstrap CIs for the
mean paired difference, resampling papers while retaining all observed runs.
These intervals are conditional on the three runs. They do not model the
population of future runs or unobserved provider changes.

Remaining limits: no repeats of scenario B, placebos/unavailable controls,
second prompt, or revised wordings; possible service drift between original
and repeat batches remains unmeasured. The original cost analysis
and its intervals describe the original experiment, not a pooled repeat study.

## Stronger judges: W5 partially addressed

Both GPT models have two medium-reasoning samples on all 274 identified-input
claims. The completion-token cap also rises from 2,000 to 8,000. Predictions
are averaged as probabilities before computing AUROC and raw Brier scores.
A paired paper-clustered bootstrap (2,000 resamples, fixed seed) evaluates the
AUROC difference from the original one-sample setting. The comparison changes
reasoning, token budget and averaging jointly; it does not isolate any one.
Intervals are exploratory and unadjusted for multiple comparisons.

Neither GPT model's AUROC-difference interval excludes zero. Mini's raw Brier
score improves, while sol's does not. These raw scores are distinct from the
main paper's Platt-calibrated Brier scores. No claim is made that larger
reasoning budgets cannot help.

Each Claude model has only three paired claims (six successful calls); their
pilot scores are excluded from aggregate performance reporting. The judge
follow-up has 1,108 successful calls, including 12 Claude pilot calls, of which
1,096 GPT calls are analysed. Total logged cost, including pilots, is $2.19.
The Haiku log contains 240 API credit failures; the full Claude comparison
remains incomplete. No aggregate inference is drawn from the three-claim pilots.

## Remaining review requests

| Request | Status after this revision |
| --- | --- |
| R1 W3 / R2 m1: repeated agent sampling | Reported for scenario-A main arms; broader repeated sampling remains open. |
| R1 W5: stronger judge configuration | Reported for both GPT models; Claude coverage, multiple prompt designs, and an isolated reasoning-effect comparison remain open. |
| R2 m1: length-matched placebo | Not run; acknowledged limitation. |
| R2 M5: conflicting system policy and tool licence | Not run; discussed as future work. |
| R2 M6 / R3 C4: two-hop handoff | Not run; discussed as future work. |
| R2 m4: open-weight model | Not run; acknowledged limitation. |
| R2 M3 / R3 C4: additional tool/domain | Claims scoped; no second tool evaluated. |

This analysis is the evidence source for the camera-ready follow-up tables.
`src/r1_followup.py analyse` gives a simpler summary (`results/r1_followup.json`)
that reports judge results only for models with complete runs; the paper uses
`src/followup.py`. The additional caches are public in
`results/repeats/` (see its README).
