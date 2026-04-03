#!/usr/bin/env python3
"""
Convert bge_negatives.json to FlagEmbedding training JSONL format.

Input:  output/bge_negatives.json
Output: output/bge_training_data.jsonl

FlagEmbedding format (one JSON object per line):
  {"query": "...", "pos": ["..."], "neg": ["...", "...", ...]}
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/bge_negatives.json")
    parser.add_argument("--output", default="output/bge_training_data.jsonl")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    examples = data["examples"]
    print(f"Loaded {len(examples)} examples from {args.input}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for e in examples:
            query = e["query"]
            positive = e["positive"]
            negatives = [n["text"] for n in e["hard_negatives"]]

            if not query or not positive or not negatives:
                skipped += 1
                continue

            record = {
                "query": query,
                "pos": [positive],
                "neg": negatives,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"✅ Written {written} examples to {args.output}")
    if skipped:
        print(f"   Skipped {skipped} examples with missing fields")

    # Verify first line
    with open(args.output, encoding="utf-8") as f:
        first = json.loads(f.readline())
    print(f"\nSample:")
    print(f"  query: {first['query'][:80]}")
    print(f"  pos:   {first['pos'][0][:80]}")
    print(f"  neg[0]: {first['neg'][0][:80]}")
    print(f"  num negatives: {len(first['neg'])}")


if __name__ == "__main__":
    main()
