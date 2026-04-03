#!/usr/bin/env python3
"""
Evaluation script — compare baseline BGE-M3 vs fine-tuned models.

Metrics: nDCG@10, MRR@10
Val set: data/val_queries.json (87 queries, all positives kept)
Corpus:  data/layer1_pilot_pairs_550_grouped_3_4.json (550 chunks)
"""

import argparse
import json
import math
import numpy as np
from FlagEmbedding import BGEM3FlagModel


def load_corpus(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(chunks)} chunks")
    return chunk_ids, texts


def load_val_queries(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]
    print(f"Loaded {len(queries)} val queries")
    return queries


def encode(model, texts, batch_size, desc=""):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out = model.encode(batch, batch_size=batch_size, max_length=512)
        all_embeddings.append(out["dense_vecs"])
        if desc and (i // batch_size + 1) % 5 == 0:
            print(f"  {desc}: {i + len(batch)}/{len(texts)}")
    return np.vstack(all_embeddings)


def compute_ndcg_at_k(relevant_ids: set, retrieved_ids: list, k: int) -> float:
    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)
    # Ideal DCG
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr_at_k(relevant_ids: set, retrieved_ids: list, k: int) -> float:
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_model(model_path: str, chunk_ids: list, chunk_texts: list, queries: list, batch_size: int, k: int = 10):
    print(f"\nEvaluating: {model_path}")
    model = BGEM3FlagModel(model_path, use_fp16=False)

    print("  Encoding corpus...")
    corpus_embs = encode(model, chunk_texts, batch_size, desc="corpus")

    print("  Encoding queries...")
    query_texts = [q["query"] for q in queries]
    query_embs = encode(model, query_texts, batch_size, desc="queries")

    # Normalize
    corpus_norm = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-9)
    query_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-9)

    # Compute similarities
    scores = query_norm @ corpus_norm.T  # (num_queries, num_chunks)

    ndcg_scores = []
    mrr_scores = []

    for i, q in enumerate(queries):
        relevant_ids = set(q["relevant_chunks"])
        sim = scores[i]
        top_k_indices = np.argsort(sim)[::-1][:k]
        retrieved = [chunk_ids[idx] for idx in top_k_indices]

        ndcg_scores.append(compute_ndcg_at_k(relevant_ids, retrieved, k))
        mrr_scores.append(compute_mrr_at_k(relevant_ids, retrieved, k))

    return {
        "model": model_path,
        f"nDCG@{k}": round(float(np.mean(ndcg_scores)), 4),
        f"MRR@{k}": round(float(np.mean(mrr_scores)), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/layer1_pilot_pairs_550_grouped_3_4.json")
    parser.add_argument("--queries", default="data/val_queries.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="output/eval_results.json")
    args = parser.parse_args()

    chunk_ids, chunk_texts = load_corpus(args.chunks)
    queries = load_val_queries(args.queries)

    models = [
        "BAAI/bge-m3",                    # baseline
        "output/models/bge-m3-unified",   # BGE official approach
        "output/models/bge-m3-gpl",       # GPL approach
    ]

    results = []
    for model_path in models:
        result = evaluate_model(model_path, chunk_ids, chunk_texts, queries, args.batch_size, args.k)
        results.append(result)
        print(f"  nDCG@{args.k}: {result[f'nDCG@{args.k}']:.4f}  MRR@{args.k}: {result[f'MRR@{args.k}']:.4f}")

    print("\n" + "="*60)
    print(f"{'Model':<35} {'nDCG@10':>10} {'MRR@10':>10}")
    print("-"*60)
    for r in results:
        name = r["model"].split("/")[-1]
        print(f"{name:<35} {r[f'nDCG@{args.k}']:>10.4f} {r[f'MRR@{args.k}']:>10.4f}")
    print("="*60)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
