#!/usr/bin/env python3
"""
Sample chunk groups for synthetic test query generation.

Selects groups NOT present in any train or val query set to ensure
zero overlap. Samples proportionally from each layer.

Output format mirrors layer*_chunks_grouped.json so it can be fed
directly to generate_n_to_n_queries_layered.py.

Usage:
    python scripts/pipeline/sample_test_chunks.py \
        --layer-chunks data/layer1_chunks_grouped.json \
                       data/layer2_chunks_grouped.json \
                       data/layer3_chunks_grouped.json \
                       data/layer4_chunks_grouped.json \
        --val-queries  data/layer1_val_queries.json \
                       data/layer2_val_queries.json \
                       data/layer3_val_queries.json \
                       data/layer4_val_queries.json \
        --train-queries data/layer1_train_queries.json \
                        data/layer2_train_queries.json \
                        data/layer3_train_queries.json \
                        data/layer4_train_queries.json \
        --counts 34 17 17 17 \
        --seed 42 \
        --output data/test_synthetic_chunk_groups.json
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def collect_used_dates(query_files):
    """Collect all group 'date' values used in given query files."""
    used = set()
    for path in query_files:
        p = Path(path)
        if not p.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for q in data.get("queries", []):
            if "date" in q:
                used.add(q["date"])
    return used


def load_layer_chunks(path):
    """Load chunks from one layer file, grouped by group_id."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]

    groups = defaultdict(list)
    for c in chunks:
        gid = c.get("group_id") or c.get("date") or c["chunk_id"]
        groups[gid].append(c)

    # Build group metadata: use first chunk for date/layer/doc_type
    group_list = []
    for gid, group_chunks in groups.items():
        rep = group_chunks[0]
        group_list.append({
            "group_id": gid,
            "date": rep.get("date", gid),
            "layer": rep.get("pair_group", ""),
            "document_type": rep.get("document_type", ""),
            "chunks": group_chunks,
        })

    return group_list


def main():
    parser = argparse.ArgumentParser(
        description="Sample chunk groups for synthetic test queries, excluding train/val groups"
    )
    parser.add_argument("--layer-chunks", nargs="+", required=True,
                        help="Chunk files for layers 1-4 (in order)")
    parser.add_argument("--val-queries", nargs="+", required=True,
                        help="Val query files for layers 1-4 (in order)")
    parser.add_argument("--train-queries", nargs="*", default=[],
                        help="Train query files (optional; excluded by default to maximise available groups)")
    parser.add_argument("--counts", nargs="+", type=int, default=[34, 17, 17, 17],
                        help="Number of groups to sample per layer (default: 34 17 17 17)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/test_synthetic_chunk_groups.json")
    args = parser.parse_args()

    random.seed(args.seed)

    # Collect all used dates from train + val
    print("Collecting used group dates from train and val sets...")
    all_query_files = list(args.val_queries) + list(args.train_queries)
    used_dates = collect_used_dates(all_query_files)
    print(f"  Total used group dates: {len(used_dates)}")

    if len(args.layer_chunks) != len(args.counts):
        raise ValueError(f"--layer-chunks ({len(args.layer_chunks)} files) must match "
                         f"--counts ({len(args.counts)} values)")

    sampled_chunks = []
    summary = []

    for layer_idx, (chunk_path, count) in enumerate(zip(args.layer_chunks, args.counts), start=1):
        print(f"\nLayer {layer_idx}: {chunk_path}")
        groups = load_layer_chunks(chunk_path)
        print(f"  Total groups: {len(groups)}")

        # Exclude groups used in train or val
        available = [g for g in groups if g["date"] not in used_dates]
        print(f"  Available (not in train/val): {len(available)}")

        if len(available) < count:
            print(f"  Warning: only {len(available)} available, requested {count}. Using all.")
            count = len(available)

        sampled = random.sample(available, count)
        print(f"  Sampled: {count} groups ({sum(len(g['chunks']) for g in sampled)} chunks)")

        for g in sampled:
            sampled_chunks.extend(g["chunks"])

        summary.append({
            "layer": f"layer{layer_idx}",
            "total_groups": len(groups),
            "available_groups": len(available),
            "sampled_groups": count,
            "sampled_chunks": sum(len(g["chunks"]) for g in sampled),
        })

    # Write output in same format as layer*_chunks_grouped.json
    output = {
        "metadata": {
            "description": "Chunk groups sampled for synthetic test query generation",
            "seed": args.seed,
            "excluded_dates": len(used_dates),
            "summary": summary,
            "total_chunks": len(sampled_chunks),
        },
        "chunks": sampled_chunks,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(sampled_chunks)} chunks ({sum(s['sampled_groups'] for s in summary)} groups) -> {args.output}")
    print("\nSummary:")
    for s in summary:
        print(f"  {s['layer']}: {s['sampled_groups']} groups, {s['sampled_chunks']} chunks "
              f"({s['available_groups']} available out of {s['total_groups']} total)")


if __name__ == "__main__":
    main()
