#!/usr/bin/env python3
"""
Generate full-content examples for each document type.

Unlike output/document_type_samples.txt (short 600-char snippets), this script writes
full ALTO page content for sampled volumes per type and also reports potential
record/date boundary cues so you can inspect segmentation patterns.
"""

import csv
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import textwrap

from lxml import etree

CONTENT_PATTERN = re.compile(r'CONTENT="([^"]*)"')
ALTO_NS = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}

# Broad date/boundary cues (historical Swedish + OCR variants)
BOUNDARY_PATTERNS = [
    re.compile(r'\b\d{1,2}\s+(?:jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec)[a-zåäö]*\b', re.IGNORECASE),
    re.compile(r'\b(?:januari|februari|mars|april|maj|juni|juli|augusti|september|oktober|november|december)\b', re.IGNORECASE),
    re.compile(r'\b(?:anno|år)\s*(?:1[6-9]\d{2}|20\d{2})\b', re.IGNORECASE),
    re.compile(r'\b(?:no|n:o|nr)\s*\d+\b', re.IGNORECASE),
    re.compile(r'\bgöteborg\b', re.IGNORECASE),
]


def extract_full_text_from_alto(xml_path: Path) -> str:
    """Extract full text from one ALTO XML page/file, preserving line breaks when possible."""
    try:
        # Preferred: parse ALTO TextLine structure for natural line breaks
        tree = etree.parse(str(xml_path))
        lines = []

        for text_line in tree.xpath('//alto:TextLine', namespaces=ALTO_NS):
            tokens = text_line.xpath('./alto:String/@CONTENT', namespaces=ALTO_NS)
            line = ' '.join(t.strip() for t in tokens if t and t.strip())
            if line:
                lines.append(line)

        if lines:
            return '\n'.join(lines)

        # Fallback: simple regex extraction
        content = xml_path.read_text(encoding='utf-8', errors='ignore')
        tokens = CONTENT_PATTERN.findall(content)
        return ' '.join(t.strip() for t in tokens if t and t.strip())
    except Exception:
        return ""


def _wrap_multiline(text: str, width: int = 110) -> str:
    """Wrap long lines while preserving existing paragraph/line boundaries."""
    wrapped_lines = []
    for line in text.splitlines():
        raw = line.rstrip()
        if not raw:
            wrapped_lines.append('')
            continue
        wrapped_lines.extend(textwrap.wrap(raw, width=width, break_long_words=False, replace_whitespace=False))
    return '\n'.join(wrapped_lines)


def pick_most_contentful_xml(volume_dir: Path, preferred_xml: str = None) -> Path:
    """
    Pick an XML file with substantial text content.

    Strategy:
    1) Try preferred XML first (if provided and substantial)
    2) Scan all XML files and pick the one with largest extracted text length
    """
    if not volume_dir.exists():
        return None

    xml_files = sorted(volume_dir.glob('*.xml'))
    if not xml_files:
        return None

    # 1) Preferred file shortcut
    if preferred_xml:
        pref = volume_dir / preferred_xml
        if pref.exists():
            pref_text = extract_full_text_from_alto(pref)
            if len(pref_text) >= 800:  # treat as substantial page
                return pref

    # 2) Search all files for the most content-rich one
    best_file = None
    best_len = -1
    for xf in xml_files:
        txt = extract_full_text_from_alto(xf)
        tlen = len(txt)
        if tlen > best_len:
            best_len = tlen
            best_file = xf

    return best_file


def find_boundary_cues(text: str, max_hits: int = 25) -> List[Tuple[int, str]]:
    """Return list of (char_position, matched_text) for potential boundaries."""
    hits = []
    for pattern in BOUNDARY_PATTERNS:
        for m in pattern.finditer(text):
            hits.append((m.start(), m.group(0)))

    # Deduplicate by (pos, cue), sort by position
    hits = sorted(set(hits), key=lambda x: x[0])
    return hits[:max_hits]


def choose_examples_by_type(rows: List[Dict], per_type: int) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['document_type']].append(r)

    # Stable selection: sort by volume_id, then take first N
    chosen = {}
    for doc_type, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x['volume_id'])
        chosen[doc_type] = items_sorted[:per_type]
    return chosen


def main():
    parser = argparse.ArgumentParser(description='Generate full-content examples for each document type')
    parser.add_argument('--fingerprints-csv',
                        default='output/comprehensive_volume_fingerprints.csv',
                        help='Path to comprehensive fingerprints CSV')
    parser.add_argument('--transcriptions-root',
                        default='Filedrop-7hXHfBBqt2nJHQuk/home/dgxuser/erik/data/transcriptions',
                        help='Root folder containing volume directories')
    parser.add_argument('--output',
                        default='output/document_type_samples_full.txt',
                        help='Output TXT path')
    parser.add_argument('--per-type', type=int, default=3,
                        help='Number of examples per type')

    args = parser.parse_args()

    csv_path = Path(args.fingerprints_csv)
    root = Path(args.transcriptions_root)
    out_path = Path(args.output)

    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')
    if not root.exists():
        raise FileNotFoundError(f'Transcriptions root not found: {root}')

    rows = []
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    chosen = choose_examples_by_type(rows, args.per_type)

    # Type counts for header
    type_counts = defaultdict(int)
    for r in rows:
        type_counts[r['document_type']] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        f.write('=' * 100 + '\n')
        f.write('DOCUMENT TYPE FULL EXAMPLES - COMPLETE PAGE CONTENT\n')
        f.write('=' * 100 + '\n')
        f.write('This file contains full extracted text from selected ALTO XML files (no 600-char truncation).\n')
        f.write('It also includes boundary-cue matches to help inspect potential record/date segmentation patterns.\n\n')

        for doc_type in sorted(chosen.keys()):
            f.write('\n' + '=' * 100 + '\n')
            f.write(f'TYPE: {doc_type} (Total volumes: {type_counts[doc_type]})\n')
            f.write('=' * 100 + '\n\n')

            for idx, row in enumerate(chosen[doc_type], 1):
                volume_id = row['volume_id']
                source_xml = row.get('title_page_source', 'N/A')
                year = row.get('year', 'N/A')
                title = row.get('volume_title', 'N/A')

                volume_dir = root / volume_id
                xml_path = pick_most_contentful_xml(
                    volume_dir,
                    preferred_xml=(source_xml if source_xml != 'N/A' else None)
                )

                full_text = extract_full_text_from_alto(xml_path) if xml_path else ''
                boundaries = find_boundary_cues(full_text)
                full_text_wrapped = _wrap_multiline(full_text, width=110) if full_text else ''

                f.write(f'--- Sample {idx} ---\n')
                f.write(f'Volume: {volume_id}\n')
                f.write(f'Year: {year}\n')
                f.write(f'Source XML: {xml_path.name if xml_path else "N/A"}\n')
                f.write(f'Title: {title}\n')
                f.write(f'Text chars: {len(full_text)} | Text words: {len(full_text.split())}\n')
                f.write('\nBoundary cue matches (position -> cue):\n')
                if boundaries:
                    for pos, cue in boundaries:
                        f.write(f'  - {pos:>6} -> {cue}\n')
                else:
                    f.write('  - None detected by current patterns\n')

                f.write('\nFULL CONTENT:\n')
                f.write('-' * 100 + '\n')
                if full_text_wrapped:
                    f.write(full_text_wrapped + '\n')
                else:
                    f.write('[No text extracted]\n')
                f.write('-' * 100 + '\n\n')

    print(f'✅ Wrote full examples: {out_path}')


if __name__ == '__main__':
    main()
