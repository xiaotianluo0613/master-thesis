#!/usr/bin/env python3
"""
Build test set candidate pools for both human and synthetic test sets.

For each query, retrieves top-k passages from:
  - BGE-M3 baseline (BAAI/bge-m3)
  - L4 fine-tuned model

Merges and deduplicates candidates, tracking which model retrieved each.
Outputs JSON (for evaluation pipeline) and CSV (for human annotation).

Usage:
    # Human test set
    python scripts/pipeline/build_test_candidates.py \
        --corpus data/global_val_chunks.json \
        --queries data/human_queries.txt \
        --queries-format txt \
        --baseline-model BAAI/bge-m3 \
        --finetuned-model output/models/layer4-bge-m3-lora-dense-b4-merged \
        --output-json data/test_human_candidates.json \
        --output-csv  data/test_human_candidates.csv

    # Synthetic test set (queries-format json)
    python scripts/pipeline/build_test_candidates.py \
        --corpus data/global_val_chunks.json \
        --queries data/test_synthetic_queries_raw.json \
        --queries-format json \
        --baseline-model BAAI/bge-m3 \
        --finetuned-model output/models/layer4-bge-m3-lora-dense-b4-merged \
        --output-json data/test_synthetic_candidates.json \
        --output-csv  data/test_synthetic_candidates.csv
"""

import argparse
import csv
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
    # Build metadata lookup
    chunk_meta = {c["chunk_id"]: c for c in chunks}
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunk_ids, texts, chunk_meta


def load_queries_txt(path: str):
    """Load plain-text queries (one per non-empty line). Returns list of dicts."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    queries = [{"query": line.strip(), "source_chunks": []} for line in lines if line.strip()]
    for i, q in enumerate(queries):
        q["query_id"] = f"hq_{i+1:03d}"
    print(f"Loaded {len(queries)} human queries from {path}")
    return queries


def load_queries_json(path: str):
    """Load queries from JSON (same format as val/train queries). Returns list of dicts."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]
    # Ensure query_id field exists
    for i, q in enumerate(queries):
        if "query_id" not in q:
            q["query_id"] = f"sq_{i+1:03d}"
        # source_chunks = relevant_chunks from query generation (the generating group's chunks)
        if "source_chunks" not in q:
            q["source_chunks"] = q.get("relevant_chunks", [])
    print(f"Loaded {len(queries)} synthetic queries from {path}")
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


def retrieve_top_k(model, chunk_ids, chunk_texts, queries, batch_size, k):
    """Encode corpus and queries, return {query_id: [(rank, chunk_id, score), ...]}."""
    print("  Encoding corpus...")
    corpus_embs = encode(model, chunk_texts, batch_size, desc="corpus")
    print("  Encoding queries...")
    query_texts = [q["query"] for q in queries]
    query_embs = encode(model, query_texts, batch_size, desc="queries")

    corpus_norm = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-9)
    query_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-9)
    scores = query_norm @ corpus_norm.T  # (num_queries, num_chunks)

    results = {}
    for i, q in enumerate(queries):
        top_indices = np.argsort(scores[i])[::-1][:k]
        results[q["query_id"]] = [
            (rank + 1, chunk_ids[idx], float(scores[i][idx]))
            for rank, idx in enumerate(top_indices)
        ]
    return results


def merge_candidates(baseline_results, finetuned_results, queries, chunk_meta):
    """
    Merge top-k from both models per query.
    For each chunk, track which model(s) retrieved it and the rank from each.
    Returns dict: {query_id: [candidate_dict, ...]} sorted by best rank.
    """
    merged = {}
    for q in queries:
        qid = q["query_id"]
        baseline_hits = {cid: (rank, score) for rank, cid, score in baseline_results.get(qid, [])}
        ft_hits = {cid: (rank, score) for rank, cid, score in finetuned_results.get(qid, [])}

        all_chunk_ids = set(baseline_hits) | set(ft_hits)
        candidates = []
        for cid in all_chunk_ids:
            retrieved_by = []
            rank_b, rank_ft = None, None
            if cid in baseline_hits:
                retrieved_by.append("baseline")
                rank_b = baseline_hits[cid][0]
            if cid in ft_hits:
                retrieved_by.append("l4")
                rank_ft = ft_hits[cid][0]

            # Best rank across models (lower is better)
            best_rank = min(r for r in [rank_b, rank_ft] if r is not None)
            meta = chunk_meta.get(cid, {})
            candidates.append({
                "chunk_id": cid,
                "document_type": meta.get("document_type", ""),
                "text": meta.get("text_without_prefix", meta.get("text", "")),
                "retrieved_by": retrieved_by,
                "rank_baseline": rank_b,
                "rank_l4": rank_ft,
                "_best_rank": best_rank,
                "relevant": None,
            })

        # Sort by best rank, then assign candidate_no
        candidates.sort(key=lambda c: c["_best_rank"])
        for no, c in enumerate(candidates, start=1):
            c["candidate_no"] = no
            del c["_best_rank"]

        merged[qid] = candidates

    return merged


def save_json(queries, merged_candidates, baseline_model, finetuned_model, top_k, output_path):
    output = {
        "metadata": {
            "corpus": "global_val_chunks.json",
            "baseline_model": baseline_model,
            "finetuned_model": finetuned_model,
            "top_k_per_model": top_k,
        },
        "queries": []
    }
    for q in queries:
        qid = q["query_id"]
        entry = {
            "query_id": qid,
            "query": q["query"],
            "source_chunks": q.get("source_chunks", []),
            "layer": q.get("layer", ""),
            "candidates": merged_candidates.get(qid, []),
        }
        output["queries"].append(entry)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {output_path}  ({len(output['queries'])} queries)")


def save_csv(queries, merged_candidates, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([
            "query_id", "query", "candidate_no", "chunk_id",
            "document_type", "retrieved_by", "rank_baseline", "rank_l4", "text", "relevant"
        ])
        for q in queries:
            qid = q["query_id"]
            for c in merged_candidates.get(qid, []):
                writer.writerow([
                    qid,
                    q["query"],
                    c["candidate_no"],
                    c["chunk_id"],
                    c["document_type"],
                    "+".join(c["retrieved_by"]),
                    c.get("rank_baseline") or "",
                    c.get("rank_l4") or "",
                    c["text"],
                    "",  # blank for annotation
                ])
    print(f"Saved CSV:  {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build test set candidate pools")
    parser.add_argument("--corpus", required=True, help="Path to global_val_chunks.json")
    parser.add_argument("--queries", required=True, help="Path to queries file")
    parser.add_argument("--queries-format", choices=["txt", "json"], default="txt")
    parser.add_argument("--baseline-model", default="BAAI/bge-m3")
    parser.add_argument("--finetuned-model",
                        default="output/models/layer4-bge-m3-lora-dense-b4-merged")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k per model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    chunk_ids, chunk_texts, chunk_meta = load_corpus(args.corpus)

    if args.queries_format == "txt":
        queries = load_queries_txt(args.queries)
    else:
        queries = load_queries_json(args.queries)

    # --- Baseline model ---
    print(f"\n{'='*60}")
    print(f"Running baseline: {args.baseline_model}")
    baseline_model = BGEM3FlagModel(args.baseline_model, use_fp16=False)
    baseline_results = retrieve_top_k(
        baseline_model, chunk_ids, chunk_texts, queries, args.batch_size, args.top_k
    )
    del baseline_model  # free GPU memory

    # --- Fine-tuned model ---
    print(f"\n{'='*60}")
    print(f"Running fine-tuned: {args.finetuned_model}")
    ft_model = BGEM3FlagModel(args.finetuned_model, use_fp16=False)
    ft_results = retrieve_top_k(
        ft_model, chunk_ids, chunk_texts, queries, args.batch_size, args.top_k
    )
    del ft_model

    # --- Merge candidates ---
    print("\nMerging candidates...")
    merged = merge_candidates(baseline_results, ft_results, queries, chunk_meta)

    total_candidates = sum(len(v) for v in merged.values())
    print(f"Total candidates across {len(queries)} queries: {total_candidates} "
          f"(avg {total_candidates/len(queries):.1f} per query)")

    save_json(queries, merged, args.baseline_model, args.finetuned_model, args.top_k,
              args.output_json)
    save_csv(queries, merged, args.output_csv)


if __name__ == "__main__":
    main()
