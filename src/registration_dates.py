#!/usr/bin/env python3
"""Design-availability dates for Tier D (R1). SCORE fixes a replication's design (power target, planned sample size)
in its preregistration; the archive dates that as `reg_date` in `rr_project_dates`. Writes
data/processed/registration_dates.csv (claim_id, reg_date); claims without a registration date get no row, and the
verifier then treats their design date as unknown."""
from pathlib import Path
import pandas as pd, pyreadr

ROOT = Path(__file__).resolve().parents[1]


def main():
    r = pyreadr.read_r(str(ROOT / 'data/raw/analysis_package/analyst data.RData'))
    reg = r['rr_project_dates'].dropna(subset=['reg_date']).set_index('rr_id').reg_date
    c = pd.read_csv(ROOT / 'data/processed/claims.csv')
    out = pd.DataFrame({'claim_id': c.claim_id, 'reg_date': c.rr_id.map(reg)}).dropna()
    out.to_csv(ROOT / 'data/processed/registration_dates.csv', index=False)
    s = pd.to_datetime(c.set_index('claim_id').launch_date.reindex(out.claim_id).to_numpy())
    g = pd.to_datetime(out.reg_date.to_numpy())
    ok = ~pd.isna(s)
    print(f'{len(out)}/{len(c)} claims with a registration date; registered after project start: '
          f'{(g[ok] > s[ok]).mean():.1%}; median lag {pd.Series((g[ok] - s[ok]).days).median():.0f} days')


if __name__ == '__main__':
    main()
