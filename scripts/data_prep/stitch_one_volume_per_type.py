#!/usr/bin/env python3
"""
Stitch one full representative volume per document type for manual pattern exploration.

For each target type:
- Select one representative volume from fingerprints CSV
- Parse ALL XML files in that volume
- Extract ALTO text lines and stitch into one readable TXT with file separators

Outputs:
- output/type_volume_stitched/{TYPE}__{VOLUME_ID}.txt
- output/type_volume_stitched/INDEX.json
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lxml import etree

ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v) if v not in (None, "") else default
    except Exception:
        return default


def safe_int(v: str, default: int = 0) -> int:
    try:
        return int(float(v)) if v not in (None, "") else default
    except Exception:
        return default


def extract_text_lines(xml_path: Path) -> List[str]:
    """Extract ALTO TextLine content preserving line structure."""
    lines: List[str] = []
    try:
        tree = etree.parse(str(xml_path))
        for tl in tree.xpath('//alto:TextLine', namespaces=ALTO_NS):
            toks = tl.xpath('./alto:String/@CONTENT', namespaces=ALTO_NS)
            line = ' '.join(t.strip() for t in toks if t and t.strip())
            if line:
                lines.append(line)
    except Exception:
        return []
    return lines


def select_representative_volume(rows: List[Dict]) -> Optional[Dict]:
    """
    Select a representative row for a type.
    Preference:
      1) high avg_pc_score
      2) lower blank_page_ratio
      3) medium xml count (readable but complete)
    """
    if not rows:
        return None

    # Keep rows with some pages
    candidates = [r for r in rows if safe_int(r.get('xml_file_count', '0')) > 0]
    if not candidates:
        return None

    def score(r: Dict) -> Tuple[float, float, float]:
        pc = safe_float(r.get('avg_pc_score', '0'))
        blank = safe_float(r.get('blank_page_ratio', '0'))
        pages = safe_int(r.get('xml_file_count', '0'))

        # Prefer around 200-500 pages as readable exploration size
        target = 300
        page_penalty = abs(pages - target)

        return (pc, -blank, -page_penalty)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def stitch_volume(volume_dir: Path) -> Tuple[str, int, int]:
    """Stitch all XML files in a volume into one text with separators."""
    xml_files = sorted(volume_dir.glob('*.xml'))
    out_lines: List[str] = []
    total_lines = 0

    for i, xf in enumerate(xml_files, 1):
        lines = extract_text_lines(xf)
        total_lines += len(lines)

        out_lines.append('=' * 100)
        out_lines.append(f'FILE {i}/{len(xml_files)}: {xf.name}')
        out_lines.append('=' * 100)
        if lines:
            out_lines.extend(lines)
        else:
            out_lines.append('[No text extracted]')
        out_lines.append('')

    stitched = '\n'.join(out_lines)
    return stitched, len(xml_files), total_lines


def main():
    parser = argparse.ArgumentParser(description='Stitch one full volume per type for pattern exploration')
    parser.add_argument('--fingerprints', default='output/comprehensive_volume_fingerprints.csv')
    parser.add_argument('--transcriptions-root', default='Filedrop-7hXHfBBqt2nJHQuk/home/dgxuser/erik/data/transcriptions')
    parser.add_argument('--out-dir', default='output/type_volume_stitched')
    parser.add_argument('--types', default='', help='Comma-separated types; default = all except Empty_or_Blank')
    args = parser.parse_args()

    fp = Path(args.fingerprints)
    root = Path(args.transcriptions_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    with fp.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    all_types = sorted({r['document_type'] for r in rows})

    if args.types.strip():
        target_types = [t.strip() for t in args.types.split(',') if t.strip()]
    else:
        target_types = [t for t in all_types if t != 'Empty_or_Blank']

    by_type: Dict[str, List[Dict]] = {t: [] for t in target_types}
    for r in rows:
        t = r.get('document_type', '')
        if t in by_type:
            by_type[t].append(r)

    index = {
        'types_requested': target_types,
        'outputs': []
    }

    for t in target_types:
        rep = select_representative_volume(by_type[t])
        if not rep:
            index['outputs'].append({'type': t, 'status': 'no_candidate'})
            continue

        volume_id = rep['volume_id']
        volume_dir = root / volume_id
        if not volume_dir.exists():
            index['outputs'].append({'type': t, 'status': 'missing_volume_dir', 'volume_id': volume_id})
            continue

        stitched, xml_count, text_line_count = stitch_volume(volume_dir)

        out_file = out_dir / f'{t}__{volume_id}.txt'
        header = [
            f'TYPE: {t}',
            f'VOLUME_ID: {volume_id}',
            f'YEAR: {rep.get("year", "")}',
            f'AVG_PC_SCORE: {rep.get("avg_pc_score", "")}',
            f'BLANK_PAGE_RATIO: {rep.get("blank_page_ratio", "")}',
            f'XML_FILE_COUNT: {xml_count}',
            f'TOTAL_TEXT_LINES: {text_line_count}',
            '',
            '#' * 100,
            'STITCHED FULL VOLUME CONTENT (ALL XML FILES)',
            '#' * 100,
            ''
        ]
        out_file.write_text('\n'.join(header) + stitched, encoding='utf-8')

        index['outputs'].append({
            'type': t,
            'status': 'ok',
            'volume_id': volume_id,
            'year': rep.get('year', ''),
            'avg_pc_score': rep.get('avg_pc_score', ''),
            'blank_page_ratio': rep.get('blank_page_ratio', ''),
            'xml_file_count': xml_count,
            'total_text_lines': text_line_count,
            'output_file': str(out_file),
        })

        print(f'✅ {t:<15} -> {volume_id} ({xml_count} files)')

    (out_dir / 'INDEX.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n📄 Index written: {out_dir / "INDEX.json"}')


if __name__ == '__main__':
    main()
