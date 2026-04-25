#!/usr/bin/env python3
"""
Convert manually-annotated CSV back to evaluation-ready JSON.

Input:  data/test_human_candidates_annotated.csv
        (same format as test_human_candidates.csv but with 'relevant' column
         filled in as 'yes'/'no' / 'y'/'n' / '1'/'0' / 'true'/'false')

Output: data/test_human_queries.json
        (same format as global_val_queries.json — ready for evaluate_comparison.py)

Usage:
    python scripts/pipeline/import_human_annotations.py \
        --input  data/test_human_candidates_annotated.csv \
        --output data/test_human_queries.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


POSITIVE_VALUES = {"yes", "y", "1", "true", "ja", "j", "relevant"}


def parse_relevant(value: str) -> bool | None:
    v = value.strip().lower()
    if not v:
        return None  # blank = unannotated
    return v in POSITIVE_VALUES


def main():
    parser = argparse.ArgumentParser(description="Import human annotations from CSV")
    parser.add_argument("--input", required=True, help="Annotated CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Allow queries with 0 relevant chunks (default: error)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    # Group rows by query_id, preserving query text
    queries_map = {}  # query_id -> {"query": ..., "relevant_chunks": [...]}
    unannotated = defaultdict(list)
    total_rows = 0

    with open(args.input, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            qid = row["query_id"].strip()
            chunk_id = row["chunk_id"].strip()
            relevant_raw = row.get("relevant", "").strip()

            if qid not in queries_map:
                queries_map[qid] = {
                    "query_id": qid,
                    "query": row["query"].strip(),
                    "relevant_chunks": [],
                    "num_candidates": 0,
                }

            queries_map[qid]["num_candidates"] += 1
            rel = parse_relevant(relevant_raw)

            if rel is None:
                unannotated[qid].append(chunk_id)
            elif rel:
                queries_map[qid]["relevant_chunks"].append(chunk_id)

    # Sort queries by query_id
    queries = sorted(queries_map.values(), key=lambda q: q["query_id"])

    # Validation
    errors = []
    for q in queries:
        q["num_relevant"] = len(q["relevant_chunks"])
        if q["num_relevant"] == 0 and not args.allow_missing:
            errors.append(f"  {q['query_id']}: 0 relevant chunks (may be incomplete annotation)")

    if unannotated:
        print(f"Warning: {sum(len(v) for v in unannotated.values())} unannotated rows across "
              f"{len(unannotated)} queries")
        for qid, cids in list(unannotated.items())[:5]:
            print(f"  {qid}: {len(cids)} blank rows")

    if errors:
        print(f"\nErrors ({len(errors)} queries have 0 relevant chunks):")
        for e in errors[:10]:
            print(e)
        print("\nFix annotations or re-run with --allow-missing to proceed anyway.")
        sys.exit(1)

    # Write output
    output = {
        "metadata": {
            "source": args.input,
            "total_queries": len(queries),
            "total_rows": total_rows,
            "avg_relevant_per_query": round(
                sum(q["num_relevant"] for q in queries) / max(len(queries), 1), 2
            ),
        },
        "queries": queries,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nImported {len(queries)} queries, {total_rows} total candidates")
    print(f"Avg relevant per query: {output['metadata']['avg_relevant_per_query']}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
