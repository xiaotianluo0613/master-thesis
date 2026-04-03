#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def export_file(src: Path, queries_only: bool = False) -> Path:
    data = json.loads(src.read_text(encoding='utf-8'))
    queries = data.get('queries', [])
    out = src.with_name(f"{src.stem}.queries.txt") if queries_only else src.with_suffix('.txt')

    if queries_only:
        lines = [q.get('query', '').strip().replace('\n', ' ') for q in queries if q.get('query')]
        out.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        return out

    lines = [
        f"source_file: {src.name}",
        f"model: {data.get('metadata', {}).get('model', 'unknown')}",
        f"total_queries: {len(queries)}",
        "",
    ]

    for i, q in enumerate(queries, 1):
        query_text = q.get('query', '').strip().replace('\n', ' ')
        layer = q.get('layer', 'na')
        volume_id = q.get('volume_id', 'na')
        rel_n = q.get('num_relevant', 'na')
        lines.append(f"[{i}] layer={layer} volume={volume_id} num_relevant={rel_n}")
        lines.append(query_text)
        lines.append("")

    out.write_text('\n'.join(lines), encoding='utf-8')
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Export generated query JSON files to TXT')
    parser.add_argument('--inputs', nargs='+', required=True, help='Input JSON files')
    parser.add_argument('--queries-only', action='store_true',
                        help='Write only one query per line to *.queries.txt files')
    args = parser.parse_args()

    for inp in args.inputs:
        src = Path(inp)
        if not src.exists():
            print(f"Skip missing: {src}")
            continue
        out = export_file(src, queries_only=args.queries_only)
        print(f"Wrote: {out}")


if __name__ == '__main__':
    main()
