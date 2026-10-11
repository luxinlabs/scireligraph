#!/bin/bash
# Full offline pipeline (no LLM calls). Usage: ./run_all.sh  (logs to results/run_all.log)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
step(){ echo; echo "=== [$(date +%H:%M:%S)] $*"; }
step build_dataset;      $PY src/build_dataset.py > results/build.log
step run_experiments;    $PY src/run_experiments.py
step starter_baseline;   $PY baseline/baseline.py --csv data/processed/score_claims.csv --out results/starter_baseline > /dev/null
step selective;          $PY src/selective.py
step permutation_null;   $PY src/permutation_null.py
step contract_tests;     $PY -m pytest -q -p no:cacheprovider tests
step mutation_analysis;  $PY src/mutation_test.py
step mcp_smoke;          rm -f results/audit_log.jsonl; $PY src/mcp_smoke.py
step tables_numbers_figs; $PY src/make_figures.py && $PY src/make_tables.py && $PY src/make_numbers.py
step DONE
