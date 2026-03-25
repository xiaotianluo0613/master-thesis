#!/usr/bin/env python3
"""
Split queries into train and validation sets.

- 330 train queries (1 positive sampled per query during mining)
- 87 val queries (ALL positives kept for N-to-N evaluation)
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split queries into train/val sets")
    parser.add_argument("--queries", default="data/queries_layer1_n2n_pilot_final_v3.json")
    parser.add_argument("--train-output", default="data/train_queries.json")
    parser.add_argument("--val-output", default="data/val_queries.json")
    parser.add_argument("--val-size", type=int, default=87)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.queries, encoding="utf-8") as f:
        data = json.load(f)

    queries = data["queries"]
    total = len(queries)

    # Group by date to avoid leaking the same document group across splits
    from collections import defaultdict
    groups = defaultdict(list)
    for q in queries:
        groups[q["date"]].append(q)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    # Fill val set by groups until we reach val_size
    val_queries = []
    train_queries = []
    for key in group_keys:
        if len(val_queries) < args.val_size:
            val_queries.extend(groups[key])
        else:
            train_queries.extend(groups[key])

    # Trim val to exact size if slightly over
    val_queries = val_queries[:args.val_size]
    # Remaining go to train
    val_dates = {q["date"] for q in val_queries}
    train_queries = [q for q in queries if q["date"] not in val_dates]

    print(f"Total queries: {total}")
    print(f"Train: {len(train_queries)}")
    print(f"Val:   {len(val_queries)}")
    print(f"Overlap check: {len(set(q['date'] for q in train_queries) & set(q['date'] for q in val_queries))} shared dates (should be 0)")

    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)

    train_out = {"metadata": {**data.get("metadata", {}), "split": "train", "size": len(train_queries)}, "queries": train_queries}
    val_out = {"metadata": {**data.get("metadata", {}), "split": "val", "size": len(val_queries), "eval_note": "all positives kept"}, "queries": val_queries}

    with open(args.train_output, "w", encoding="utf-8") as f:
        json.dump(train_out, f, ensure_ascii=False, indent=2)
    with open(args.val_output, "w", encoding="utf-8") as f:
        json.dump(val_out, f, ensure_ascii=False, indent=2)

    print(f"✅ Train saved to {args.train_output}")
    print(f"✅ Val saved to   {args.val_output}")


if __name__ == "__main__":
    main()
