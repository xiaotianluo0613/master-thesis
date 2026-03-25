#!/usr/bin/env python3
import json
import re
from pathlib import Path

root = Path('output/type_volume_stitched')
files = sorted([p for p in root.glob('*.txt') if p.name != 'INDEX.json'])

patterns = {
    'date_numeric': re.compile(r'\b\d{1,2}[\./-]\d{1,2}[\./-]\d{2,4}\b'),
    'date_month_sv': re.compile(r'\b\d{1,2}\s*(?:de|den)?\s*(januari|februari|mars|april|maj|juni|juli|augusti|september|oktober|november|december|jan|feb|mar|apr|jun|jul|aug|sep|okt|nov|dec|febr|sept|oct)\b', re.I),
    'year_4d': re.compile(r'\b(1[6-9]\d{2}|190\d)\b'),
    'no_marker': re.compile(r'\b(N:o|No|Nr|n:o|no|nr)\s*\d+\b'),
    'person_report': re.compile(r'\b(har anm[aä]lt|hafva anm[aä]lt|rapporterat|anh[aå]llen|efterlys)\b', re.I),
    'legal_formula': re.compile(r'\b(Anno|uti|§|Kongl\.?|Hofr[aä]tt|Ransakning|Protocoll|dombok)\b', re.I),
    'address_marker': re.compile(r'\b(vid|uti|i huset|rote|socken|gatan|torget|hamnkanalen)\b', re.I),
}

def top_lines(text, regex, n=3):
    out = []
    for line in text.splitlines():
        if regex.search(line):
            out.append(line.strip())
            if len(out) >= n:
                break
    return out

summary = {"files": []}
for p in files:
    txt = p.read_text(encoding='utf-8', errors='ignore')
    row = {
        'file': p.name,
        'counts': {k: len(r.findall(txt)) for k, r in patterns.items()},
        'examples': {k: top_lines(txt, patterns[k], 3) for k in ['date_month_sv', 'no_marker', 'person_report', 'legal_formula']}
    }
    summary['files'].append(row)

out = Path('output/boundary_pattern_summary.json')
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Wrote {out}')
