#!/usr/bin/env python3
"""
Evaluate and compare retrieval models on a val/test query set.

Metrics: nDCG@k, Recall@k, MAP@k  (at each k in --k-values)
Corpus:  chunked JSON with "chunks" list
Queries: query JSON with "queries" list, each having "relevant_chunks"

Usage:
    python scripts/pipeline/evaluate_comparison.py \
        --chunks  data/layer1_chunks_grouped.json \
        --queries data/layer1_val_queries.json \
        --models  BAAI/bge-m3 output/models/layer1-bge-m3-unified \
        --k-values 10 100 \
        --output  output/layer1_eval_results.json
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel


def load_corpus(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text_with_prefix"] if "text_with_prefix" in c else c["text"] for c in chunks]
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunk_ids, texts


def load_queries(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]
    print(f"Loaded {len(queries)} queries from {path}")
    return queries


def encode(model, texts, batch_size, desc=""):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out = model.encode(batch, batch_size=batch_size, max_length=512)
        all_embeddings.append(out["dense_vecs"])
        if desc and (i // batch_size + 1) % 10 == 0:
            print(f"  {desc}: {i + len(batch)}/{len(texts)}")
    return np.vstack(all_embeddings)


def compute_ndcg(relevant_ids: set, retrieved: list, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, rid in enumerate(retrieved[:k], start=1)
        if rid in relevant_ids
    )
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant_ids), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(relevant_ids: set, retrieved: list, k: int) -> float:
    hits = sum(1 for rid in retrieved[:k] if rid in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def compute_map(relevant_ids: set, retrieved_full: list) -> float:
    """Standard MAP — average precision over the full ranked list."""
    hits, precision_sum = 0, 0.0
    for rank, rid in enumerate(retrieved_full, start=1):
        if rid in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_ids) if relevant_ids else 0.0


def evaluate_model(model_path: str, chunk_ids: list, chunk_texts: list,
                   queries: list, batch_size: int, k_values: list):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    model = BGEM3FlagModel(model_path, use_fp16=False)

    print("  Encoding corpus...")
    corpus_embs = encode(model, chunk_texts, batch_size, desc="corpus")
    print("  Encoding queries...")
    query_embs = encode(model, [q["query"] for q in queries], batch_size, desc="queries")

    corpus_norm = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-9)
    query_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-9)
    scores = query_norm @ corpus_norm.T  # (num_queries, num_chunks)

    max_k = max(k_values)
    per_k = {k: {"ndcg": [], "recall": []} for k in k_values}
    map_scores = []

    for i, q in enumerate(queries):
        relevant_ids = set(q["relevant_chunks"])
        # Full ranking for MAP
        all_indices = np.argsort(scores[i])[::-1]
        retrieved_full = [chunk_ids[idx] for idx in all_indices]
        map_scores.append(compute_map(relevant_ids, retrieved_full))

        # Top-k for nDCG and Recall
        retrieved_topk = retrieved_full[:max_k]
        for k in k_values:
            per_k[k]["ndcg"].append(compute_ndcg(relevant_ids, retrieved_topk, k))
            per_k[k]["recall"].append(compute_recall(relevant_ids, retrieved_topk, k))

    result = {"model": model_path, "MAP": round(float(np.mean(map_scores)), 4)}
    for k in k_values:
        result[f"nDCG@{k}"]   = round(float(np.mean(per_k[k]["ndcg"])), 4)
        result[f"Recall@{k}"] = round(float(np.mean(per_k[k]["recall"])), 4)

    return result


def print_table(results: list, k_values: list):
    metrics = ["MAP"] + \
              [f"nDCG@{k}" for k in k_values] + \
              [f"Recall@{k}" for k in k_values]
    header = f"{'Model':<40}" + "".join(f"{m:>12}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        name = r["model"].split("/")[-1]
        row = f"{name:<40}" + "".join(f"{r.get(m, 0.0):>12.4f}" for m in metrics)
        print(row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval models")
    parser.add_argument("--chunks",    required=True)
    parser.add_argument("--queries",   required=True)
    parser.add_argument("--models",    nargs="+", default=["BAAI/bge-m3"])
    parser.add_argument("--k-values",  nargs="+", type=int, default=[10, 100])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output",    default="output/eval_results.json")
    args = parser.parse_args()

    chunk_ids, chunk_texts = load_corpus(args.chunks)
    queries = load_queries(args.queries)

    results = []
    for model_path in args.models:
        result = evaluate_model(model_path, chunk_ids, chunk_texts,
                                queries, args.batch_size, args.k_values)
        results.append(result)

    print_table(results, args.k_values)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
