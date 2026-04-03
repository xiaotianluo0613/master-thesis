#!/usr/bin/env python3
"""
N-to-N evaluation with:
1) Sliding-window chunking per model budget
2) Parent-level max pooling over child-window scores

Why:
- Different models have different max context sizes.
- We index child windows, then aggregate to parent chunks with max pooling.
"""

import json
import argparse
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Install: pip install sentence-transformers faiss-cpu")
    raise


def sliding_windows_words(text: str, window_words: int, stride_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= window_words:
        return [' '.join(words)]

    windows = []
    i = 0
    while i < len(words):
        chunk = words[i:i + window_words]
        if not chunk:
            break
        windows.append(' '.join(chunk))
        if i + window_words >= len(words):
            break
        i += stride_words
    return windows


def build_child_index(chunks: List[Dict], model: SentenceTransformer,
                      window_words: int, overlap_ratio: float) -> Tuple[faiss.IndexFlatIP, List[str]]:
    """Build child-window index and return (index, child_parent_ids)."""
    stride = max(1, int(window_words * (1.0 - overlap_ratio)))

    child_texts: List[str] = []
    child_parent_ids: List[str] = []

    for c in chunks:
        parent_id = c['chunk_id']
        text = c.get('text_without_prefix', c.get('text', ''))
        windows = sliding_windows_words(text, window_words=window_words, stride_words=stride)
        for w in windows:
            child_texts.append(w)
            child_parent_ids.append(parent_id)

    emb = model.encode(
        child_texts,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb.astype('float32'))
    return idx, child_parent_ids


def retrieve_parent_rank(query: str, model: SentenceTransformer, child_index: faiss.IndexFlatIP,
                         child_parent_ids: List[str], top_k_children: int = 200) -> List[Tuple[str, float]]:
    """Retrieve child windows then aggregate to parent by max score (max pooling)."""
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, inds = child_index.search(q.astype('float32'), top_k_children)

    parent_max = {}
    for s, i in zip(scores[0], inds[0]):
        if i < 0:
            continue
        pid = child_parent_ids[i]
        if pid not in parent_max or s > parent_max[pid]:
            parent_max[pid] = float(s)

    ranked = sorted(parent_max.items(), key=lambda x: x[1], reverse=True)
    return ranked


def precision_at_k(ranked_ids: List[str], relevant: set, k: int) -> float:
    r = ranked_ids[:k]
    return len(set(r) & relevant) / k if k else 0.0


def recall_at_k(ranked_ids: List[str], relevant: set, k: int) -> float:
    r = ranked_ids[:k]
    return len(set(r) & relevant) / len(relevant) if relevant else 0.0


def average_precision(ranked_ids: List[str], relevant: set) -> float:
    if not relevant:
        return 0.0
    hit = 0
    ps = []
    for i, rid in enumerate(ranked_ids, 1):
        if rid in relevant:
            hit += 1
            ps.append(hit / i)
    return sum(ps) / len(relevant) if ps else 0.0


def evaluate(queries_file: str, chunks_file: str, output_file: str,
             model_name: str = 'BAAI/bge-m3', window_words: int = 350,
             overlap_ratio: float = 0.2, k_values: List[int] = [5, 10]):
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device='cpu')

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)['chunks']
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)['queries']

    print(f"Chunks: {len(chunks)} | Queries: {len(queries)}")
    print(f"Building child index with sliding windows: window_words={window_words}, overlap={overlap_ratio}")
    t0 = time.time()
    child_index, child_parent_ids = build_child_index(chunks, model, window_words, overlap_ratio)
    print(f"Child vectors: {child_index.ntotal} | Build time: {time.time()-t0:.1f}s")

    metrics = {k: {'p': [], 'r': [], 'ap': []} for k in k_values}

    for i, q in enumerate(queries, 1):
        ranked = retrieve_parent_rank(q['query'], model, child_index, child_parent_ids, top_k_children=300)
        ranked_ids = [x[0] for x in ranked]
        relevant = set(q.get('relevant_chunks', []))

        ap = average_precision(ranked_ids, relevant)
        for k in k_values:
            metrics[k]['p'].append(precision_at_k(ranked_ids, relevant, k))
            metrics[k]['r'].append(recall_at_k(ranked_ids, relevant, k))
            metrics[k]['ap'].append(ap)

        if i % 50 == 0:
            print(f"Processed {i}/{len(queries)}")

    out = {
        'model': model_name,
        'method': 'sliding-window + parent max-pooling',
        'window_words': window_words,
        'overlap_ratio': overlap_ratio,
        'queries_evaluated': len(queries),
        'results': {
            str(k): {
                'precision': float(np.mean(metrics[k]['p'])),
                'recall': float(np.mean(metrics[k]['r'])),
                'map': float(np.mean(metrics[k]['ap'])),
            }
            for k in k_values
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_file}")
    for k in k_values:
        print(f"K={k}: P={out['results'][str(k)]['precision']:.4f} R={out['results'][str(k)]['recall']:.4f} MAP={out['results'][str(k)]['map']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', default='data/queries_daily_n_to_n.json')
    parser.add_argument('--chunks', default='data/30002051_chunks_split_prefixed.json')
    parser.add_argument('--output', default='data/n_to_n_results_sliding_maxpool.json')
    parser.add_argument('--model', default='BAAI/bge-m3')
    parser.add_argument('--window-words', type=int, default=350)
    parser.add_argument('--overlap', type=float, default=0.2)
    args = parser.parse_args()

    evaluate(
        queries_file=args.queries,
        chunks_file=args.chunks,
        output_file=args.output,
        model_name=args.model,
        window_words=args.window_words,
        overlap_ratio=args.overlap,
    )


if __name__ == '__main__':
    main()
