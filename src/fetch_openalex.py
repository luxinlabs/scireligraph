#!/usr/bin/env python3
"""Fetch OpenAlex records for the 164 SCORE original papers and every work citing them.

Raw JSON is cached under data/openalex/ so the graph can be rebuilt offline. Time filtering
is NOT done here; it happens in build_dataset.py so that strict and leaky cutoffs share the
same raw evidence.
"""
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd, pyreadr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import oa_stem

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'openalex'
SEL_CITING = 'id,doi,publication_date,authorships,primary_location,referenced_works,topics'

def get(url, tries=8):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SciReliGraph-research/0.1'})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            print(f'  HTTP {e.code} retry {i}', file=sys.stderr); time.sleep(2 * (i + 1))
        except Exception as e:
            print(f'  {e} retry {i}', file=sys.stderr); time.sleep(2 * (i + 1))
    raise RuntimeError('failed ' + url)

def main():
    r = pyreadr.read_r(str(ROOT / 'data/raw/analysis_package/analyst data.RData'))
    ro = r['repli_outcomes']
    papers = ro[ro.repli_version_of_record & ~ro.is_covid].paper_id.unique()
    meta = r['paper_metadata'].set_index('paper_id').loc[papers]
    (OUT / 'works').mkdir(parents=True, exist_ok=True); (OUT / 'citing').mkdir(exist_ok=True)
    for pid, doi in meta.DOI.items():
        wf = OUT / 'works' / f'{oa_stem(pid)}.json'
        if not wf.exists():
            w = get('https://api.openalex.org/works/doi:' + urllib.parse.quote(doi.strip().lower(), safe='/'))
            wf.write_text(json.dumps(w))
        w = json.loads(wf.read_text())
        cf = OUT / 'citing' / f'{oa_stem(pid)}.jsonl'
        if w is None or cf.exists(): continue
        wid = w['id'].rsplit('/', 1)[-1]
        cursor, rows = '*', []
        while cursor:
            d = get(f'https://api.openalex.org/works?filter=cites:{wid}&per-page=200&cursor={cursor}&select={SEL_CITING}')
            rows += d['results']; cursor = d['meta'].get('next_cursor')
            if not d['results']: break
        cf.write_text('\n'.join(json.dumps(x) for x in rows))
        print(pid, wid, w.get('cited_by_count'), len(rows), flush=True)
    (OUT / 'ACCESS_DATE.txt').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z') + '\n')

if __name__ == '__main__':
    main()
