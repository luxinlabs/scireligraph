#!/usr/bin/env python3
"""Check every *cited* bibliography entry against an authoritative source and write results/citation_check.{json,md}.

DOI -> Crossref (title similarity); arXiv id -> arXiv abs page (citation_title meta); otherwise a URL in the entry must
resolve (HTTP 200); remaining entries are searched on Crossref by title. Similarity >= 0.85 counts as verified.
"""
import difflib, json, re, time, urllib.parse, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; P = ROOT / 'paper'
bib = (P / 'references.bib').read_text()
entries = {m.group(2): m.group(3) for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', bib, re.S)}
aux = (P / 'main.aux').read_text() if (P / 'main.aux').exists() else ''
cited = sorted({k for g in re.findall(r'\\citation\{([^}]*)\}', aux) for k in g.split(',')})
norm = lambda s: re.sub(r'[^a-z0-9 ]', '', re.sub(r'\\[a-z]+|[{}]', '', (s or '').lower()))
def field(b, f):
    m = re.search(rf'\b{f}\s*=\s*\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}', b, re.S); return m.group(1).strip() if m else None
def get(url, js=True):
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 citation-check'}), timeout=30) as r:
                d = r.read(); return (json.loads(d), r.status) if js else (d.decode('utf-8', 'ignore'), r.status)
        except Exception as e:
            err = e; time.sleep(2)
    return None, str(err)
out = []
cited = [k for k in cited if not k.startswith('IEEEexample:')]  # BibTeX style-control entry, not a reference
for k in cited:
    b = entries.get(k, ''); title = field(b, 'title'); doi = field(b, 'doi')
    blob = ' '.join(filter(None, [field(b, 'note'), field(b, 'howpublished'), field(b, 'eprint')]))  # eprint: identifier kept unprinted
    arx = re.search(r'(\d{4}\.\d{4,5})', blob or ''); url = re.search(r'https?://[^\s}]+', blob or '')
    res = {'key': k, 'title': title, 'method': None, 'found_title': None, 'similarity': None, 'verified': False}
    if doi:
        d, st = get('https://api.crossref.org/works/' + urllib.parse.quote(doi, safe='/'))
        ft = (d['message']['title'][0] + (': ' + d['message']['subtitle'][0] if d['message'].get('subtitle') else '')) if d else None; res.update(method='crossref-doi', found_title=ft)
    elif arx:
        h, st = get('https://arxiv.org/abs/' + arx.group(1), js=False)
        m = re.search(r'name="citation_title" content="([^"]*)"', h or ''); ft = m.group(1) if m else None
        res.update(method='arxiv', found_title=ft)
    elif url:
        h, st = get(url.group(0), js=False); res.update(method=f'url ({st})', found_title=title if h else None)
    else:
        d, st = get('https://api.crossref.org/works?rows=5&query.bibliographic=' + urllib.parse.quote(norm(title)))
        items = d['message']['items'] if d else []
        best = max(items, key=lambda it: difflib.SequenceMatcher(None, norm(title), norm((it.get('title') or [''])[0])).ratio(), default=None)
        res.update(method='crossref-search', found_title=(best.get('title') or [None])[0] if best else None)
    if res['found_title']:
        res['similarity'] = round(difflib.SequenceMatcher(None, norm(title), norm(res['found_title'])).ratio(), 2)
        res['verified'] = res['similarity'] >= 0.85
    out.append(res); print(f"{'OK ' if res['verified'] else '?? '} {k:28s} {res['method']:16s} {res['similarity']}")
(ROOT / 'results' / 'citation_check.json').write_text(json.dumps(out, indent=1))
md = ['| Key | Method | Similarity | Verified |', '|---|---|---|---|'] + [f"| {r['key']} | {r['method']} | {r['similarity']} | {'yes' if r['verified'] else '**manual**'} |" for r in out]
(ROOT / 'results' / 'citation_check.md').write_text('\n'.join(md) + '\n')
print(sum(r['verified'] for r in out), 'of', len(out), 'verified automatically')
