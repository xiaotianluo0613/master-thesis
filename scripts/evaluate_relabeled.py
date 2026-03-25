#!/usr/bin/env python3
"""
Evaluate multiple models on re-labeled queries

Tests: BGE-M3, mContriever, mDPR (independent from retrieval models)
Metrics: Precision@K, Recall@K, MAP, nDCG@K

Note: We do NOT test BM25 or mE5-small since they were used for retrieval!
"""

import json
import numpy as np
import time
import argparse

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Install: pip install sentence-transformers faiss-cpu rank-bm25")
    raise


def evaluate_model(model_name, model, index, chunks, queries, k_values=[1, 3, 5, 10]):
    """Evaluate a single model on re-labeled queries"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*80}")
    
    metrics = {k: {'precision': [], 'recall': [], 'ap': [], 'ndcg': []} 
               for k in k_values}
    
    for i, query_entry in enumerate(queries, 1):
        query_text = query_entry['query']
        relevant_chunk_ids = set(query_entry['relevant_chunks'])
        
        if not relevant_chunk_ids:
            continue
        
        # Retrieve
        if model_name == 'BM25':
            tokenized_query = query_text.lower().split()
            scores = index.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[-max(k_values):][::-1]
            retrieved_ids = [chunks[idx]['chunk_id'] for idx in top_indices]
        else:
            query_emb = model.encode([query_text], normalize_embeddings=True)
            scores, top_indices = index.search(query_emb.astype('float32'), max(k_values))
            retrieved_ids = [chunks[idx]['chunk_id'] for idx in top_indices[0]]
        
        # Calculate metrics
        for k in k_values:
            retrieved_k = retrieved_ids[:k]
            relevant_in_k = len(set(retrieved_k) & relevant_chunk_ids)
            
            # Precision@K
            precision = relevant_in_k / k
            metrics[k]['precision'].append(precision)
            
            # Recall@K
            recall = relevant_in_k / len(relevant_chunk_ids)
            metrics[k]['recall'].append(recall)
            
            # nDCG@K
            dcg = sum(1.0 / np.log2(i + 2) for i, rid in enumerate(retrieved_k) 
                     if rid in relevant_chunk_ids)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_chunk_ids), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[k]['ndcg'].append(ndcg)
        
        # Average Precision (for MAP)
        ap = 0.0
        num_relevant_found = 0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_chunk_ids:
                num_relevant_found += 1
                ap += num_relevant_found / rank
        ap = ap / len(relevant_chunk_ids) if relevant_chunk_ids else 0.0
        
        for k in k_values:
            metrics[k]['ap'].append(ap)
        
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(queries)}")
    
    # Calculate means
    results = {}
    for k in k_values:
        results[k] = {
            'precision': float(np.mean(metrics[k]['precision'])),
            'recall': float(np.mean(metrics[k]['recall'])),
            'ndcg': float(np.mean(metrics[k]['ndcg'])),
            'map': float(np.mean(metrics[k]['ap']))
        }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', default='data/queries_relabeled_multimodel.json',
                       help='Re-labeled queries file')
    parser.add_argument('--chunks', default='data/30002051_chunks_split_prefixed.json',
                       help='Chunks file')
    parser.add_argument('--output', default='data/relabeled_evaluation_results.json',
                       help='Output results file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("EVALUATING 3 INDEPENDENT MODELS (not used in retrieval)")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    with open(args.queries) as f:
        queries_data = json.load(f)
    queries = queries_data['queries']
    
    with open(args.chunks) as f:
        chunks = json.load(f)['chunks']
    
    # Filter to queries with relevant labels
    queries = [q for q in queries if q.get('num_relevant', 0) > 0]
    print(f"✅ Loaded {len(queries)} queries with relevance labels")
    print(f"   Avg relevant chunks per query: {np.mean([q['num_relevant'] for q in queries]):.2f}")
    print(f"\nNOTE: BM25 and mE5-small were used for RETRIEVAL (not tested here)")
    print(f"      Testing: BGE-M3, mContriever, mDPR (independent models)\n")
    
    texts = [c.get('text_without_prefix', c.get('text', '')) for c in chunks]
    
    # Evaluate each model
    all_results = {}
    
    # 1. BGE-M3
    print("1️⃣  Loading BGE-M3...")
    model = SentenceTransformer('BAAI/bge-m3', device='cpu')
    embeddings = model.encode(texts, batch_size=8, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    all_results['bge-m3'] = evaluate_model('BGE-M3', model, index, chunks, queries)
    
    # 2. mContriever
    print("\n2️⃣  Loading mContriever...")
    model = SentenceTransformer('facebook/mcontriever-msmarco', device='cpu')
    embeddings = model.encode(texts, batch_size=8, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    all_results['mcontriever'] = evaluate_model('mContriever', model, index, chunks, queries)
    
    # 3. mDPR (use msmarco-bert as approximation)
    print("\n3️⃣  Loading mDPR...")
    model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5', device='cpu')
    embeddings = model.encode(texts, batch_size=8, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    all_results['mdpr'] = evaluate_model('mDPR', model, index, chunks, queries)
    
    # Print comparison
    print("\n" + "="*80)
    print("RESULTS COMPARISON (3 INDEPENDENT TEST MODELS)")
    print("="*80)
    
    for k in [1, 3, 5, 10]:
        print(f"\nK = {k}:")
        print(f"{'Model':<15} {'Precision':<12} {'Recall':<12} {'nDCG':<12} {'MAP':<12}")
        print("-"*80)
        for model_name in ['bge-m3', 'mcontriever', 'mdpr']:
            results = all_results[model_name][k]
            print(f"{model_name.upper():<15} {results['precision']:<12.4f} {results['recall']:<12.4f} {results['ndcg']:<12.4f} {results['map']:<12.4f}")
    
    # Save results
    output_data = {
        'metadata': {
            'queries_file': args.queries,
            'chunks_file': args.chunks,
            'num_queries': len(queries),
            'avg_relevant_per_query': float(np.mean([q['num_relevant'] for q in queries])),
            'models_evaluated': list(all_results.keys())
        },
        'results': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()
