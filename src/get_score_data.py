#!/usr/bin/env python3
"""Download the publication-frozen SCORE analyst package (OSF g5sny, CC0) and verify it against the manifest checksum.

    python src/get_score_data.py      # -> data/raw/analysis_package.zip and data/raw/analysis_package/
"""
import hashlib, json, sys, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://osf.io/download/5xfdy/'
RAW = ROOT / 'data' / 'raw'
ZIP = RAW / 'analysis_package.zip'


def main():
    want = json.loads((ROOT / 'data/processed/manifest.json').read_text())['source_sha256']
    RAW.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        print('downloading', URL)
        urllib.request.urlretrieve(URL, ZIP)
    got = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    if got != want:
        sys.exit(f'checksum mismatch: got {got}, manifest {want}. The OSF file may have been updated; '
                 'results are only guaranteed for the manifest version.')
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(RAW / 'analysis_package', [m for m in z.namelist() if not m.startswith('__MACOSX')])
    print('ok: sha256 matches manifest;', ZIP.stat().st_size, 'bytes extracted to', RAW / 'analysis_package')


if __name__ == '__main__':
    main()
