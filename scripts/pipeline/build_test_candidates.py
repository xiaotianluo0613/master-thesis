#!/usr/bin/env python3
"""
Build test set candidate pools for both human and synthetic test sets.

Two-stage retrieval per model:
  Stage 1 — Dense top-100 + Sparse top-100, union → up to 200 candidates
  Stage 2 — Integration re-score (dense 0.4 + sparse 0.2 + ColBERT 0.4) → keep top-20

Merge baseline + L4 top-20 → ~30 candidates per query.

For human queries only: --gemini-prefilter sends each query's ~30 candidates to
Gemini Flash, which returns ~10 relevant ones to reduce manual annotation burden.

Usage:
    # Human test set (with Gemini pre-filter)
    python scripts/pipeline/build_test_candidates.py \
        --corpus data/global_val_chunks.json \
        --queries data/human_queries.txt \
        --queries-format txt \
        --baseline-model BAAI/bge-m3 \
        --finetuned-model output/models/layer4-bge-m3-lora-dense-b4-merged \
        --output-json data/test_human_candidates.json \
        --output-csv  data/test_human_candidates.csv \
        --gemini-prefilter

    # Synthetic test set
    python scripts/pipeline/build_test_candidates.py \
        --corpus data/global_val_chunks.json \
        --queries data/test_synthetic_queries_raw.json \
        --queries-format json \
        --baseline-model BAAI/bge-m3 \
        --finetuned-model output/models/layer4-bge-m3-lora-dense-b4-merged \
        --output-json data/test_synthetic_candidates.json \
        --output-csv  data/test_synthetic_candidates.csv

Environment (only needed with --gemini-prefilter):
    GEMINI_API_KEY
"""

import argparse
import csv
import json
import os
import random
import re
import time
import numpy as np
from pathlib import Path

import requests
from FlagEmbedding import BGEM3FlagModel


# ── Corpus / query loading ─────────────────────────────────────────────────

def load_corpus(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text_with_prefix"] if "text_with_prefix" in c else c["text"] for c in chunks]
    chunk_meta = {c["chunk_id"]: c for c in chunks}
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunk_ids, texts, chunk_meta


def load_queries_txt(path: str):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    queries = [{"query": line.strip(), "source_chunks": []} for line in lines if line.strip()]
    for i, q in enumerate(queries):
        q["query_id"] = f"hq_{i+1:03d}"
    print(f"Loaded {len(queries)} human queries from {path}")
    return queries


def load_queries_json(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]
    for i, q in enumerate(queries):
        if "query_id" not in q:
            q["query_id"] = f"sq_{i+1:03d}"
        if "source_chunks" not in q:
            q["source_chunks"] = q.get("relevant_chunks", [])
    print(f"Loaded {len(queries)} synthetic queries from {path}")
    return queries


# ── Encoding ───────────────────────────────────────────────────────────────

def encode_corpus(model, texts, batch_size):
    """Encode corpus with dense + sparse in one pass."""
    dense_list, sparse_list = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out = model.encode(batch, batch_size=batch_size, max_length=512,
                           return_dense=True, return_sparse=True)
        dense_list.append(out["dense_vecs"])
        sparse_list.extend(out["lexical_weights"])
        if (i // batch_size + 1) % 10 == 0:
            print(f"  corpus: {i + len(batch)}/{len(texts)}")
    return np.vstack(dense_list), sparse_list


def encode_queries(model, query_texts, batch_size):
    """Encode queries with dense + sparse in one pass."""
    dense_list, sparse_list = [], []
    for i in range(0, len(query_texts), batch_size):
        batch = query_texts[i:i + batch_size]
        out = model.encode(batch, batch_size=batch_size, max_length=512,
                           return_dense=True, return_sparse=True)
        dense_list.append(out["dense_vecs"])
        sparse_list.extend(out["lexical_weights"])
    return np.vstack(dense_list), sparse_list


# ── Sparse similarity ──────────────────────────────────────────────────────

def sparse_scores_for_query(query_weights: dict, all_passage_weights: list) -> np.ndarray:
    """Compute sparse dot-product similarity between one query and all passages."""
    scores = np.zeros(len(all_passage_weights), dtype=np.float32)
    for token_id, qw in query_weights.items():
        for pi, pw in enumerate(all_passage_weights):
            if token_id in pw:
                scores[pi] += qw * pw[token_id]
    return scores


def sparse_scores_vectorized(query_weights: dict, all_passage_weights: list) -> np.ndarray:
    """Vectorized sparse similarity: accumulate per token_id across all passages."""
    scores = np.zeros(len(all_passage_weights), dtype=np.float32)
    for token_id, qw in query_weights.items():
        for pi, pw in enumerate(all_passage_weights):
            v = pw.get(token_id)
            if v:
                scores[pi] += qw * v
    return scores


# ── Two-stage retrieval ────────────────────────────────────────────────────

def integration_rescore(model, query_text: str, candidate_texts: list,
                        batch_size: int = 32) -> list:
    """Re-score candidates using dense+sparse+ColBERT integration (0.4/0.2/0.4)."""
    pairs = [[query_text, t] for t in candidate_texts]
    scores = []
    for i in range(0, len(pairs), batch_size):
        result = model.compute_score(
            pairs[i:i + batch_size],
            weights_for_different_modes=[0.4, 0.2, 0.4],
        )
        batch_scores = result["colbert+sparse+dense"]
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)
    return scores


def retrieve_top_k(model, chunk_ids, chunk_texts, queries, batch_size, dense_k, final_k):
    """
    Two-stage retrieval:
      1. Dense top-dense_k + Sparse top-dense_k, union → up to 2×dense_k candidates
      2. Integration re-score → keep top-final_k
    Returns {query_id: [(rank, chunk_id, score), ...]}
    """
    print("  Encoding corpus (dense + sparse)...")
    corpus_dense, corpus_sparse = encode_corpus(model, chunk_texts, batch_size)

    print("  Encoding queries (dense + sparse)...")
    query_texts_list = [q["query"] for q in queries]
    query_dense, query_sparse = encode_queries(model, query_texts_list, batch_size)

    # Normalise dense vecs for cosine similarity
    corpus_norm = corpus_dense / (np.linalg.norm(corpus_dense, axis=1, keepdims=True) + 1e-9)
    query_norm = query_dense / (np.linalg.norm(query_dense, axis=1, keepdims=True) + 1e-9)
    all_dense_scores = query_norm @ corpus_norm.T  # (num_queries, num_chunks)

    results = {}
    for i, q in enumerate(queries):
        # Stage 1a: dense top-k
        dense_top = set(np.argsort(all_dense_scores[i])[::-1][:dense_k].tolist())

        # Stage 1b: sparse top-k
        sparse_sc = sparse_scores_vectorized(query_sparse[i], corpus_sparse)
        sparse_top = set(np.argsort(sparse_sc)[::-1][:dense_k].tolist())

        # Union
        shortlist_indices = list(dense_top | sparse_top)
        shortlist_ids = [chunk_ids[idx] for idx in shortlist_indices]
        shortlist_texts = [chunk_texts[idx] for idx in shortlist_indices]

        # Stage 2: integration re-score
        int_scores = integration_rescore(model, q["query"], shortlist_texts)
        ranked = sorted(zip(shortlist_ids, int_scores), key=lambda x: x[1], reverse=True)

        results[q["query_id"]] = [
            (rank + 1, cid, float(score))
            for rank, (cid, score) in enumerate(ranked[:final_k])
        ]

        if (i + 1) % 10 == 0 or (i + 1) == len(queries):
            print(f"  Retrieved {i+1}/{len(queries)} queries "
                  f"(shortlist avg: {len(shortlist_ids)})")

    return results


# ── Candidate merging ──────────────────────────────────────────────────────

def merge_candidates(baseline_results, finetuned_results, queries, chunk_meta):
    """Merge top-k from both models, dedup, track which model retrieved each."""
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

        candidates.sort(key=lambda c: c["_best_rank"])
        for no, c in enumerate(candidates, start=1):
            c["candidate_no"] = no
            del c["_best_rank"]

        merged[qid] = candidates

    return merged


# ── Gemini pre-filter (human queries only) ─────────────────────────────────

GEMINI_PREFILTER_PROMPT = """\
You are helping a historian annotate search results for 19th century Swedish archive documents.

Given the query below and a list of numbered archive passages, identify which passages are \
RELEVANT. A passage is relevant if it would help answer the query, even partially.

Return ONLY a JSON array of the relevant candidate numbers. Example: [1, 3, 7]
If none are relevant, return: []

Query: {query}

Passages:
{passages}"""


def call_gemini(prompt: str, api_key: str, model: str = "gemini-2.5-flash",
                max_retries: int = 6) -> str:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models"
           f"/{model}:generateContent?key={api_key}")
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
    }
    attempt = 0
    rate_limit_hits = 0
    while True:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 429:
                rate_limit_hits += 1
                wait = min(900, int(60 * (2 ** min(rate_limit_hits - 1, 6))))
                print(f"  Rate limit, waiting {wait}s...")
                time.sleep(wait + random.uniform(0, 2))
                continue
            r.raise_for_status()
            data = r.json()
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                raise
            print(f"  Gemini retry {attempt}/{max_retries}: {e}")
            time.sleep(min(60, 5 * attempt))


def gemini_prefilter(queries, merged_candidates, api_key, gemini_model, min_candidates=10):
    """
    For each query, send all candidates to Gemini Flash and keep only relevant ones.
    Always guarantees at least min_candidates by padding with top-ranked candidates
    that Gemini didn't pick. Gemini judgment is not ground truth — just reduces
    manual annotation burden.
    """
    print(f"\nGemini pre-filter on {len(queries)} queries (min {min_candidates} per query)...")
    for qi, q in enumerate(queries):
        qid = q["query_id"]
        candidates = merged_candidates.get(qid, [])
        if not candidates:
            continue

        passages = "\n\n".join(
            f"[{c['candidate_no']}] ({c['document_type']})\n{c['text'][:800]}"
            for c in candidates
        )
        prompt = GEMINI_PREFILTER_PROMPT.format(query=q["query"], passages=passages)

        try:
            response = call_gemini(prompt, api_key, gemini_model)
            match = re.search(r"\[[\d,\s]*\]", response)
            relevant_nos = set(json.loads(match.group())) if match else set()
        except Exception as e:
            print(f"  Error on {qid}: {e} — keeping all candidates")
            relevant_nos = {c["candidate_no"] for c in candidates}

        filtered = [c for c in candidates if c["candidate_no"] in relevant_nos]

        # Pad up to min_candidates with top-ranked candidates not already included
        if len(filtered) < min_candidates:
            filtered_nos = {c["candidate_no"] for c in filtered}
            for c in candidates:
                if len(filtered) >= min_candidates:
                    break
                if c["candidate_no"] not in filtered_nos:
                    filtered.append(c)
                    filtered_nos.add(c["candidate_no"])

        merged_candidates[qid] = filtered
        print(f"  [{qi+1}/{len(queries)}] {qid}: {len(candidates)} → {len(filtered)} candidates")

    return merged_candidates


# ── Output ─────────────────────────────────────────────────────────────────

def save_json(queries, merged_candidates, baseline_model, finetuned_model,
              dense_k, final_k, output_path):
    output = {
        "metadata": {
            "corpus": "global_val_chunks.json",
            "baseline_model": baseline_model,
            "finetuned_model": finetuned_model,
            "dense_shortlist_k": dense_k,
            "final_k_per_model": final_k,
            "retrieval": "dense+sparse union shortlist → integration rescore (0.4/0.2/0.4)",
        },
        "queries": [],
    }
    for q in queries:
        qid = q["query_id"]
        output["queries"].append({
            "query_id": qid,
            "query": q["query"],
            "source_chunks": q.get("source_chunks", []),
            "layer": q.get("layer", ""),
            "candidates": merged_candidates.get(qid, []),
        })

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
            "document_type", "retrieved_by", "rank_baseline", "rank_l4", "text", "relevant",
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
                    "",
                ])
    print(f"Saved CSV:  {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build test set candidate pools")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--queries-format", choices=["txt", "json"], default="txt")
    parser.add_argument("--baseline-model", default="BAAI/bge-m3")
    parser.add_argument("--finetuned-model",
                        default="output/models/layer4-bge-m3-lora-dense-b4-merged")
    parser.add_argument("--dense-k", type=int, default=100,
                        help="Top-k per retrieval method (dense and sparse) for shortlist")
    parser.add_argument("--final-k", type=int, default=20,
                        help="Keep top-k after integration re-scoring")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gemini-prefilter", action="store_true",
                        help="Use Gemini Flash to pre-filter candidates (human queries only)")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--min-candidates", type=int, default=10,
                        help="Minimum candidates per query after Gemini filter (padded by rank)")
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
        baseline_model, chunk_ids, chunk_texts, queries,
        args.batch_size, args.dense_k, args.final_k,
    )
    del baseline_model

    # --- Fine-tuned model ---
    print(f"\n{'='*60}")
    print(f"Running fine-tuned: {args.finetuned_model}")
    ft_model = BGEM3FlagModel(args.finetuned_model, use_fp16=False)
    ft_results = retrieve_top_k(
        ft_model, chunk_ids, chunk_texts, queries,
        args.batch_size, args.dense_k, args.final_k,
    )
    del ft_model

    # --- Merge ---
    print("\nMerging candidates...")
    merged = merge_candidates(baseline_results, ft_results, queries, chunk_meta)

    total = sum(len(v) for v in merged.values())
    print(f"Total candidates: {total} across {len(queries)} queries "
          f"(avg {total/len(queries):.1f} per query)")

    # --- Gemini pre-filter (human queries only) ---
    if args.gemini_prefilter:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set — required for --gemini-prefilter")
        merged = gemini_prefilter(queries, merged, api_key, args.gemini_model, args.min_candidates)
        total_after = sum(len(v) for v in merged.values())
        print(f"After Gemini filter: {total_after} candidates "
              f"(avg {total_after/len(queries):.1f} per query)")

    save_json(queries, merged, args.baseline_model, args.finetuned_model,
              args.dense_k, args.final_k, args.output_json)
    save_csv(queries, merged, args.output_csv)


if __name__ == "__main__":
    main()
