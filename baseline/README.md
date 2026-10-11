# SciReliGraph — SCORE metadata baseline

This reproducible starter evaluates whether original-study metadata predicts later replication outcomes. No downloaded SCORE data are included.

Official SCORE resources:
- https://osf.io/dtzx4/overview
- https://osf.io/bzfgy/
- https://osf.io/g5sny/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13456834/

## Run

```bash
pip install -r requirements.txt
python baseline.py --csv score_claims.csv --out results
```

Read `schema.md` before mapping original SCORE tables. The pipeline deliberately refuses missing outcomes and does not guess column names or silently convert labels.

Outputs: `results/summary.json`, `results/fold_metrics.csv`, `results/out_of_fold_predictions.csv`.
