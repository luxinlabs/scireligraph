# Reviewer follow-up caches (Nov 2026)

Raw cached responses from `src/r1_followup.py`. The original caches in `results/agent/` and `results/llm/` are unchanged;
each record carries the prompt hash, model, request parameters, raw response, parsed output and cost.

## `agent/`: repeated agent decisions (Reviewer 1, W3)

- Scenario-A bare-number (`A_unprotected_r{1,2}`) and original-contract (`A_contract_aware_r{1,2}`) arms.
- Two extra runs per arm for all four agents: 274 claims × 2 arms × 2 runs = 1,096 calls per model.
- Run 0 is the original `results/agent/<model>.jsonl`.
- **Complete for all four models.** The Opus 5.5 file also keeps 86 calls that first failed on an API credit limit;
  they were re-run successfully, and failed records (`"parsed": null`) are ignored by the analysis.
- Cost: US$6.18.

## `llm/`: stronger LLM-judge setting (Reviewer 1, W5)

- Identified-input forecast with medium reasoning and two samples per claim (`forecast_id_strong_s{1,2}`).
- **Complete only for GPT-5.4 mini and GPT-6.1 sol** (548 calls each).
- **Incomplete for the Claude models:** 3 pilot claims each, plus 240 Haiku 4.5 calls that failed on an API credit
  limit. These files are not used for any reported result.

Reproduce the summaries with `python src/r1_followup.py analyse`.
