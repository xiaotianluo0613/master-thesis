#!/usr/bin/env python3
"""
Simple N-to-N Query Evaluation
Reuses baseline BGE-M3 approach but with multi-relevant metrics
"""

import json
import numpy as np
from pathlib import Path
import time
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Install: pip install sentence-transformers faiss-cpu")
    raise


def load_baseline_index(chunks_file, use_prefix=False):
    """Load chunks and build index (reusing baseline approach)"""
    print("Loading BGE-M3 model...")
    model = SentenceTransformer("BAAI/bge-m3", device='cpu')
    
    print(f"Loading chunks from {chunks_file}...")
    with open(chunks_file) as f:
        chunks = json.load(f)['chunks']
    
    text_field = 'text' if use_prefix else 'text_without_prefix'
    texts = [c[text_field] for c in chunks]
    
    print(f"Encoding {len(chunks)} chunks (batch_size=8)...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    print(f"✅ Encoding took {time.time()-start:.1f}s")
    
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    print(f"✅ Index ready with {index.ntotal} vectors\n")
    return model, index, chunks


def search(query_text, model, index, chunks, k=20):
    """Search for top-k chunks"""
    query_emb = model.encode([query_text], normalize_embeddings=True)
    scores, indices = index.search(query_emb.astype('float32'), k)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            'chunk_id': chunks[idx]['chunk_id'],
            'score': float(score),
            'rank': len(results) + 1
        })
    return results


def calculate_precision_at_k(retrieved_ids, relevant_ids, k):
    """Precision@K: fraction of top-k that are relevant"""
    retrieved_k = retrieved_ids[:k]
    relevant_in_k = len(set(retrieved_k) & set(relevant_ids))
    return relevant_in_k / k if k > 0 else 0.0


def calculate_recall_at_k(retrieved_ids, relevant_ids, k):
    """Recall@K: fraction of relevant items found in top-k"""
    retrieved_k = retrieved_ids[:k]
    relevant_in_k = len(set(retrieved_k) & set(relevant_ids))
    return relevant_in_k / len(relevant_ids) if len(relevant_ids) > 0 else 0.0


def calculate_average_precision(retrieved_ids, relevant_ids):
    """Average Precision (for MAP)"""
    if not relevant_ids:
        return 0.0
    
    precisions = []
    num_relevant_found = 0
    
    for k, retrieved_id in enumerate(retrieved_ids, 1):
        if retrieved_id in relevant_ids:
            num_relevant_found += 1
            precision_at_k = num_relevant_found / k
            precisions.append(precision_at_k)
    
    return sum(precisions) / len(relevant_ids) if precisions else 0.0


def calculate_ndcg_at_k(retrieved_ids, relevant_ids, k):
    """Normalized Discounted Cumulative Gain at K"""
    # DCG
    dcg = 0.0
    for i, retrieved_id in enumerate(retrieved_ids[:k], 1):
        if retrieved_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 1)
    
    # IDCG (ideal: all relevant at top)
    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(queries_file, chunks_file, output_file, k_values=[1, 3, 5, 10]):
    """Main evaluation"""
    
    # Load model and index
    model, index, chunks = load_baseline_index(chunks_file, use_prefix=False)
    
    # Load queries
    print(f"Loading queries from {queries_file}...")
    with open(queries_file) as f:
        queries = json.load(f)['queries']
    
    print(f"✅ Loaded {len(queries)} queries")
    print(f"\nEvaluating at K = {k_values}")
    print("=" * 80)
    
    # Storage for metrics
    metrics = {k: {
        'precision': [],
        'recall': [],
        'ap': [],
        'ndcg': []
    } for k in k_values}
    
    # Evaluate each query
    start_time = time.time()
    for i, query_entry in enumerate(queries, 1):
        query_text = query_entry['query']
        relevant_chunk_ids = set(query_entry['relevant_chunks'])
        
        # Retrieve
        max_k = max(k_values)
        results = search(query_text, model, index, chunks, k=max_k)
        retrieved_ids = [r['chunk_id'] for r in results]
        
        # Calculate metrics for each k
        for k in k_values:
            precision = calculate_precision_at_k(retrieved_ids, relevant_chunk_ids, k)
            recall = calculate_recall_at_k(retrieved_ids, relevant_chunk_ids, k)
            ndcg = calculate_ndcg_at_k(retrieved_ids, relevant_chunk_ids, k)
            
            metrics[k]['precision'].append(precision)
            metrics[k]['recall'].append(recall)
            metrics[k]['ndcg'].append(ndcg)
        
        # AP (for MAP)
        ap = calculate_average_precision(retrieved_ids, relevant_chunk_ids)
        for k in k_values:
            metrics[k]['ap'].append(ap)
        
        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i}/{len(queries)} ({elapsed:.1f}s)")
    
    # Calculate means
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - N-to-N Queries")
    print("=" * 80)
    
    results_summary = {}
    for k in k_values:
        mean_precision = np.mean(metrics[k]['precision'])
        mean_recall = np.mean(metrics[k]['recall'])
        mean_ndcg = np.mean(metrics[k]['ndcg'])
        map_score = np.mean(metrics[k]['ap'])
        
        results_summary[k] = {
            'precision': float(mean_precision),
            'recall': float(mean_recall),
            'ndcg': float(mean_ndcg),
            'map': float(map_score)
        }
        
        print(f"\nK = {k}:")
        print(f"  Precision@{k}: {mean_precision:.4f}")
        print(f"  Recall@{k}:    {mean_recall:.4f}")
        print(f"  nDCG@{k}:      {mean_ndcg:.4f}")
        print(f"  MAP@{k}:       {map_score:.4f}")
    
    # Save results
    output_data = {
        'model': 'BGE-M3',
        'queries_evaluated': len(queries),
        'k_values': k_values,
        'results': results_summary,
        'total_time_seconds': time.time() - start_time
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print("=" * 80)
    
    # Compare with baseline
    baseline_file = "output/baseline_bge_m3_results.json"
    if Path(baseline_file).exists():
        print("\n" + "=" * 80)
        print("BASELINE COMPARISON - 1-to-1 Queries")
        print("=" * 80)
        with open(baseline_file) as f:
            baseline = json.load(f)
        print(f"\nBaseline (1-to-1):")
        print(f"  MRR:       {baseline.get('mrr', 'N/A'):.4f}")
        print(f"  Hit@5:     {baseline.get('hit_rate@5', 'N/A'):.4f}")
        print(f"  Hit@10:    {baseline.get('hit_rate@10', 'N/A'):.4f}")
        print(f"  Queries:   {baseline.get('queries_evaluated', 'N/A')}")
        print()
        print(f"N-to-N (current):")
        print(f"  MAP@5:     {results_summary[5]['map']:.4f}")
        print(f"  Recall@5:  {results_summary[5]['recall']:.4f}")
        print(f"  Recall@10: {results_summary[10]['recall']:.4f}")
        print(f"  Queries:   {len(queries)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', default='data/queries_daily_n_to_n.json')
    parser.add_argument('--chunks', default='data/30002051_chunks_split_prefixed.json')
    parser.add_argument('--output', default='data/n_to_n_results.json')
    args = parser.parse_args()
    
    evaluate(args.queries, args.chunks, args.output)
