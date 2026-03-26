#!/usr/bin/env python3
"""
BGE-M3 Integration Scoring — compute teacher scores for m3_kd_loss training.

For each (query, positive) and (query, negative) pair:
  - Score using BGE-M3 integration score (dense + sparse + multi-vector combined)

Adds pos_scores and neg_scores to the training JSONL for use with m3_kd_loss.

Input:  output/bge_training_data.jsonl  (from convert_to_flagembedding_format.py)
Output: output/bge_training_data_scored.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/bge_training_data.jsonl")
    parser.add_argument("--output", default="output/bge_training_data_scored.jsonl")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading BGE-M3: {args.model}")
    model = BGEM3FlagModel(args.model, use_fp16=True)
    print("Model loaded.")

    examples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {args.input}")

    # Build all pairs: (query, pos) and (query, neg)
    # Each example has 1 pos and up to 7 negs
    pos_pairs = [[e["query"], e["pos"][0]] for e in examples]
    neg_pairs_flat = []
    neg_counts = []
    for e in examples:
        neg_counts.append(len(e["neg"]))
        for neg in e["neg"]:
            neg_pairs_flat.append([e["query"], neg])

    def score_pairs(pairs):
        scores = []
        for i in range(0, len(pairs), args.batch_size):
            batch = pairs[i:i + args.batch_size]
            queries = [p[0] for p in batch]
            passages = [p[1] for p in batch]
            sentence_pairs = [[q, p] for q, p in zip(queries, passages)]
            result = model.compute_score(
                sentence_pairs,
                weights_for_different_modes=[0.4, 0.2, 0.4],  # dense, sparse, colbert
            )
            batch_scores = result["colbert+sparse+dense"]
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
            if (i // args.batch_size + 1) % 10 == 0:
                print(f"  {i + len(batch)}/{len(pairs)} scored...")
        return scores

    print(f"\nScoring {len(pos_pairs)} positive pairs...")
    start = time.time()
    pos_scores = score_pairs(pos_pairs)
    print(f"Done in {time.time()-start:.1f}s")

    print(f"Scoring {len(neg_pairs_flat)} negative pairs...")
    start = time.time()
    neg_scores_flat = score_pairs(neg_pairs_flat)
    print(f"Done in {time.time()-start:.1f}s")

    # Reconstruct per-example neg scores
    neg_scores_per_example = []
    idx = 0
    for count in neg_counts:
        neg_scores_per_example.append(neg_scores_flat[idx:idx + count])
        idx += count

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for e, ps, ns in zip(examples, pos_scores, neg_scores_per_example):
            record = {
                "query": e["query"],
                "pos": e["pos"],
                "neg": e["neg"],
                "pos_scores": [round(ps, 6)],
                "neg_scores": [round(s, 6) for s in ns],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Written {len(examples)} examples to {args.output}")

    # Quick stats
    all_pos = pos_scores
    all_neg = neg_scores_flat
    print(f"   Avg pos score: {sum(all_pos)/len(all_pos):.4f}")
    print(f"   Avg neg score: {sum(all_neg)/len(all_neg):.4f}")


if __name__ == "__main__":
    main()
