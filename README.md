<div align="center">

# SciReliGraph

### Epistemic Contracts for Scientific Agent Tools

**Reproducibility package for the paper**<br>
*Can Agents Trust a Replication Verifier? Epistemic Contracts for Scientific Agent Tools*

Luxin Zhang · Independent Researcher

[![Workshop](https://img.shields.io/badge/SE4AgenticAI-2026-1f6feb)](https://cs.uwaterloo.ca/~palencar/se4agenticai/)
[![Venue](https://img.shields.io/badge/IEEE%20Big%20Data-2026-0b3d91)](https://cs.uwaterloo.ca/~palencar/se4agenticai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-3776ab)](requirements.txt)
[![Tool](https://img.shields.io/badge/MCP-tool-6f42c1)](src/mcp_server.py)

Accepted at the [**2nd IEEE Big Data International Workshop on Software Engineering for Agentic AI (SE4AgenticAI 2026)**](https://cs.uwaterloo.ca/~palencar/se4agenticai/),<br>
a workshop of the 2026 IEEE International Conference on Big Data · December 14–17, 2026 · Phoenix, Arizona

</div>

---

## Overview

AI agents that do science can hand the question *"can this finding be trusted?"* to a verifier tool. Such tools are
usually judged on accuracy alone, not on the guarantees an agent relies on when it acts on the answer.

This repository contains everything behind the paper:

- **An epistemic contract** for agent tools: checkable clauses on what an output may reflect (*provenance*), which
  decisions it licenses (*licence*), how it is audited (*accountability*), and whether the agents that read it act as
  intended (*consumer*). The contract is stated as six requirements, R1–R6.
- **SciReliGraph**, a replication verifier exposed as an [MCP](https://modelcontextprotocol.io) tool that enforces
  the contract on every call, with metamorphic, isolation and refusal tests and mutation analysis.
- **An evaluation** on 274 claims with real replication outcomes from the DARPA SCORE program, covering statistical
  verifiers, LLM judges with a decoy-based memorization probe, and four LLM agents in the loop.

## Key results

| Research question | Finding |
|---|---|
| **RQ1:** Can the contract be enforced and tested? | Yes. The runtime clauses are enforced on every call; contract tests reach a verified automated mutation score of **61%**. Feature-availability typing caught **two leaks** in our own pipeline. |
| **RQ2:** Does a compliant verifier earn the licence to gate decisions? | No. The deployed verifier discriminates only modestly (AUROC **0.620**), and its confident outputs are not measurably more accurate. LLM judges rank comparably, but their temporal validity cannot be certified. |
| **RQ3:** Do agents act on the contract as intended? | Only partly. A bare probability led most agents into near-chance automatic decisions. The contract's reliability fields helped three agents, but Claude Haiku 4.5 misread them, raising its errors from **18% to 54%**; this recurred in all three repeated runs. Stating the permitted action removed the misuse in every tested condition. |

## Repository layout

| Path | Contents |
|---|---|
| [`src/`](src/) | Data acquisition and dataset build, verifier library (`verifier.py`), MCP server (`mcp_server.py`), experiments, LLM-judge and agent studies, mutation tooling, and the generators for every table, number and figure in the paper |
| [`tests/`](tests/) | 27 contract, schema and boundary tests |
| [`contract/`](contract/) | Reusable JSON Schema for an epistemic-contract response, with tool-agnostic test relations |
| [`data/processed/`](data/processed/) | Claim table, graph features and a provenance manifest (source archive SHA-256, access dates) |
| [`results/`](results/) | All metrics, out-of-fold predictions, cached LLM-judge and agent responses, and mutation analysis |
| [`results/repeats/`](results/repeats/) | Reviewer follow-up runs: repeated agent decisions and a stronger LLM-judge setting |
| [`baseline/`](baseline/) | The original starter baseline |
| [`run_all.sh`](run_all.sh) | The full offline pipeline |

The paper's LaTeX source is not part of this repository.

## Getting started

**Requirements:** Python 3.13 and the pinned packages in [`requirements.txt`](requirements.txt).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. Download the inputs (not stored in this repository)
.venv/bin/python src/get_score_data.py     # SCORE analyst package from OSF, SHA-256 checked
.venv/bin/python src/fetch_openalex.py     # OpenAlex records and citing works

# 2. Run the contract tests
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests

# 3. Run the full offline pipeline (no API calls)
./run_all.sh
```

## Reproducing the paper

Every result table, every number quoted in the text and Fig. 2 are generated from `results/`; none is typed by hand. The
generators write LaTeX tables and macros to `paper/tables/` and figures to `paper/figures/` (both created locally and
git-ignored):

| Script | Produces |
|---|---|
| `src/make_tables.py` | Forecasting, LLM-judge, contract-test, agent and control-arm tables |
| `src/make_numbers.py` | Every number in the prose, as LaTeX macros (`numbers.tex`) |
| `src/make_r6.py` | The R6 validation table and the risk–coverage figure (Fig. 2) |
| `src/make_figures.py` | Supplementary AUROC and calibration figures |
| `src/followup.py` | Validation of the reviewer follow-up caches, and the repeatability and stronger-judge tables |

All LLM analyses re-run from the cached responses in `results/llm/`, `results/agent/`, `results/r6/` and
`results/repeats/`, so no API access is needed to reproduce the reported results.

## Using the verifier as an MCP tool

`src/mcp_server.py` serves one tool over stdio:

```text
verify_claim(claim_id, as_of, stage="publication")
```

It returns the probability that a SCORE claim replicates, using only evidence dated on or before `as_of`, together with
the base rate, cross-validated skill, a gating verdict (`selective_ok`), advice on how the score may be used, and provenance hashes. Calls
on or after a replication's launch date are refused. Start it with:

```bash
.venv/bin/python src/mcp_server.py
```

The response format is specified in [`contract/epistemic_contract.schema.json`](contract/epistemic_contract.schema.json),
which any verifier tool can adopt.

## Running the LLM experiments

Only needed to collect new responses. The experiments read `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` from the
environment.

- **LLM judges and agents:** `src/run_llm_judge.py`, `src/agent_experiment.py`
- **Pre-specified R6 study:** `src/r6_experiment.py`. Its wordings, conditions and success criterion were committed
  before any call was made (commit [`8990cf7`](https://github.com/luxinlabs/scireligraph/commit/8990cf7448dc37f54f92fe55fa97ea8985ca9ab5)).
- **Reviewer follow-ups:** `src/r1_followup.py` (two extra runs of the main agent arms, and LLM judges at medium
  reasoning). Coverage and costs are documented in [`results/repeats/README.md`](results/repeats/README.md) and the
  analysis in [`results/repeats/ANALYSIS.md`](results/repeats/ANALYSIS.md).

## Citation

If you use this code, the contract schema or the results, please cite:

```bibtex
@inproceedings{zhang2026epistemic,
  title     = {Can Agents Trust a Replication Verifier? Epistemic Contracts for Scientific Agent Tools},
  author    = {Zhang, Luxin},
  booktitle = {2nd IEEE Big Data International Workshop on Software Engineering for Agentic AI (SE4AgenticAI 2026),
               2026 IEEE International Conference on Big Data},
  year      = {2026},
  note      = {Code and data: \url{https://github.com/luxinlabs/scireligraph}}
}
```

## Data and licences

- **Code:** MIT; see [`LICENSE`](LICENSE).
- **SCORE replication materials:** CC0 1.0 (OSF projects [g5sny](https://osf.io/g5sny/) and [dtzx4](https://osf.io/dtzx4/)).
- **OpenAlex:** CC0.
- `data/processed/` and `results/` are derived from these sources.

## AI-assistance disclosure

Parts of the pipeline, analysis code and manuscript were drafted or edited with AI systems (Claude, Anthropic; Codex,
OpenAI). The author reviewed and verified all content.

## Contact

Luxin Zhang · [luxin.zhang92@outlook.com](mailto:luxin.zhang92@outlook.com)
