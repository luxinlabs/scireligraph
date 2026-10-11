# SCORE baseline input schema

**Important:** The official OSF files must be inspected and mapped explicitly. This project does **not** contain downloaded SCORE claim-level data, and no performance metrics have been computed from SCORE.

Create a UTF-8 CSV with one row per unique original claim:

| Column | Type | Description |
|---|---|---|
| `claim_id` | string | Unique original scientific claim identifier |
| `paper_id` | string | Original paper identifier; used for grouped cross-validation |
| `replicated` | integer 0/1 | SCORE primary success criterion, NOT a robustness/reproducibility label |
| `original_n` | number | Original study sample size, when known |
| `original_p` | number | Original-study p-value, when known |
| `original_effect_size` | number | Harmonized original effect size, only when comparable |
| `publication_year` | integer | Original paper publication year |
| `discipline` | string | Original research discipline |
| `journal` | string | Original journal |

Only original-study metadata available at forecast time is permitted. Never include replication sample size, replication p-value, replication effect, later citations, replication report text, success classification as features, or any other post-outcome information.

Before modeling, document the source OSF file and column for every mapped field, verify 274 unique claims and 164 paper groups if the full dataset is available, and check whether the chosen SCORE outcome definition reproduces the published 151/274 success count. If the dataset is a subset, document inclusion and exclusions.

Run `python baseline.py --csv score_claims.csv --out results`.

The script fits prevalence-only and logistic regression baselines using five-fold stratified, **paper-grouped** cross-validation. It exports per-fold metrics and out-of-fold predictions. This is an exploratory baseline, not a final locked external test. Calibration, hyperparameter tuning, and feature selection must be nested within training folds for a publication-grade result. For statistical uncertainty, use a paper-clustered bootstrap of out-of-fold predictions.
