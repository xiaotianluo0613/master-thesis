#!/usr/bin/env python3
"""
GPL Scoring — MarginMSE training data preparation.

For each (query, positive, negative) triple:
  - Score (query, positive) with bge-reranker-v2-m3
  - Score (query, negative) with bge-reranker-v2-m3
  - Compute margin = score(positive) - score(negative)

Output: JSONL where each line is a training example with margin score.
"""

import argparse
import json
import time
from pathlib import Path

from FlagEmbedding import FlagReranker


def load_negatives(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data['examples'])} examples from {path}")
    return data["examples"]


def score_pairs(reranker: FlagReranker, pairs: list, batch_size: int) -> list:
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        batch_scores = reranker.compute_score(
            batch,
            normalize=True,
            max_length=args.max_passage_len,
        )
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/gpl_negatives.json")
    parser.add_argument("--output", default="output/gpl_training_data.jsonl")
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-query-len", type=int, default=512)
    parser.add_argument("--max-passage-len", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading reranker: {args.reranker}")
    reranker = FlagReranker(args.reranker, use_fp16=True)
    print("Reranker loaded.")

    examples = load_negatives(args.input)

    # Build all pairs for batched scoring
    pos_pairs = [[e["query"], e["positive"]] for e in examples]
    neg_pairs = [[e["query"], e["negative"]] for e in examples]

    print(f"\nScoring {len(pos_pairs)} positive pairs...")
    start = time.time()
    pos_scores = score_pairs(reranker, pos_pairs, args.batch_size)
    print(f"Done in {time.time()-start:.1f}s")

    print(f"Scoring {len(neg_pairs)} negative pairs...")
    start = time.time()
    neg_scores = score_pairs(reranker, neg_pairs, args.batch_size)
    print(f"Done in {time.time()-start:.1f}s")

    # Write output JSONL
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for e, ps, ns in zip(examples, pos_scores, neg_scores):
            margin = ps - ns
            if margin <= 0:
                # Negative scored higher than positive — unreliable example
                skipped += 1
                continue
            record = {
                "query": e["query"],
                "pos": e["positive"],
                "neg": e["negative"],
                "pos_score": round(ps, 6),
                "neg_score": round(ns, 6),
                "margin": round(margin, 6),
                "query_id": e["query_id"],
                "positive_chunk_id": e["positive_chunk_id"],
                "negative_chunk_id": e["negative_chunk_id"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n✅ Written {written} examples to {args.output}")
    print(f"   Skipped {skipped} examples where margin <= 0")

    # Quick stats
    margins = []
    with open(args.output, encoding="utf-8") as f:
        for line in f:
            margins.append(json.loads(line)["margin"])
    if margins:
        avg = sum(margins) / len(margins)
        print(f"   Avg margin: {avg:.4f}")
        print(f"   Min margin: {min(margins):.4f}")
        print(f"   Max margin: {max(margins):.4f}")


if __name__ == "__main__":
    main()
