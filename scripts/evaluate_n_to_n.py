#!/usr/bin/env python3
"""
Evaluate N-to-N Query Generation with Precision, Recall, MAP, and nDCG

Supports both:
- 1-to-1 queries (1 relevant chunk per query) 
- N-to-N queries (multiple relevant chunks per query)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Set
import time
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Install: pip install sentence-transformers faiss-cpu")
    raise


class MultiRelevanceEvaluator:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """Initialize with BGE-M3 model."""
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
        
        self.index = None
        self.chunks = []
        self.chunk_id_to_idx = {}
    
    def load_data(self, chunks_file: str, queries_file: str):
        """Load chunks and queries."""
        print(f"\nLoading data...")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            self.chunks = chunks_data['chunks']
        
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
            queries = queries_data['queries']
        
        print(f"✅ Loaded {len(self.chunks)} chunks and {len(queries)} queries")
        
        # Normalize query format
        for q in queries:
            if 'relevant_chunks' not in q:
                # 1-to-1 format (original)
                q['relevant_chunks'] = [q.get('relevant_chunk', q.get('chunk_id'))]
            # Convert to set for fast lookup
            q['relevant_set'] = set(q['relevant_chunks'])
        
        return queries
    
    def build_index(self, use_prefix: bool = True):
        """Build FAISS index."""
        print(f"\n{'='*80}")
        print("BUILDING VECTOR INDEX")
        print(f"{'='*80}")
        
        text_field = 'text' if use_prefix else 'text_without_prefix'
        texts = [chunk[text_field] for chunk in self.chunks]
        
        print(f"Encoding {len(texts)} chunks...")
        start = time.time()
        
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"✅ Encoding complete: {time.time()-start:.2f}s")
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.chunk_id_to_idx = {chunk['chunk_id']: i for i, chunk in enumerate(self.chunks)}
        
        print(f"✅ Index built: {self.index.ntotal} vectors")
        print(f"{'='*80}\n")
    
    def search(self, query: str, k: int = 20):
        """Search for top-k chunks."""
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = [
            (self.chunks[idx]['chunk_id'], float(score))
            for idx, score in zip(indices[0], scores[0])
        ]
        
        return results
    
    def calculate_metrics(self, retrieved: List[str], relevant: Set[str], k: int) -> Dict:
        """Calculate precision, recall, AP for one query."""
        retrieved_k = retrieved[:k]
        
        # Precision@K: proportion of retrieved that are relevant
        num_relevant_retrieved = len(set(retrieved_k) & relevant)
        precision = num_relevant_retrieved / k if k > 0 else 0.0
        
        # Recall@K: proportion of relevant that are retrieved
        recall = num_relevant_retrieved / len(relevant) if len(relevant) > 0 else 0.0
        
        # Average Precision
        num_relevant_found = 0
        sum_precisions = 0.0
        
        for i, doc_id in enumerate(retrieved_k, 1):
            if doc_id in relevant:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / i
                sum_precisions += precision_at_i
        
        ap = sum_precisions / len(relevant) if len(relevant) > 0 else 0.0
        
        # nDCG@K
        dcg = 0.0
        idcg = 0.0
        
        for i, doc_id in enumerate(retrieved_k, 1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 1)
        
        # Ideal DCG: all relevant docs at top
        for i in range(1, min(len(relevant), k) + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'ap': ap,
            'ndcg': ndcg
        }
    
    def evaluate(self, queries: List[Dict], k_values: List[int] = [5, 10, 20]):
        """Evaluate with Precision, Recall, MAP, nDCG."""
        print(f"{'='*80}")
        print("EVALUATING RETRIEVAL PERFORMANCE")
        print(f"{'='*80}")
        print(f"Queries: {len(queries)}")
        print(f"K values: {k_values}")
        
        # Analyze query types
        num_single = sum(1 for q in queries if len(q['relevant_set']) == 1)
        num_multi = len(queries) - num_single
        avg_relevant = np.mean([len(q['relevant_set']) for q in queries])
        
        print(f"Single-relevance queries: {num_single}")
        print(f"Multi-relevance queries: {num_multi}")
        print(f"Avg relevant chunks per query: {avg_relevant:.2f}")
        print()
        
        max_k = max(k_values)
        
        results = {
            'queries_evaluated': len(queries),
            'num_single_relevance': num_single,
            'num_multi_relevance': num_multi,
            'avg_relevant_per_query': avg_relevant
        }
        
        # Initialize metric storage
        for k in k_values:
            results[f'precision@{k}'] = []
            results[f'recall@{k}'] = []
            results[f'ndcg@{k}'] = []
        
        results['ap_scores'] = []
        
        print("Processing queries...")
        start = time.time()
        
        for i, query_data in enumerate(queries):
            query_text = query_data['query']
            relevant_set = query_data['relevant_set']
            
            # Search
            search_results = self.search(query_text, k=max_k)
            retrieved_ids = [chunk_id for chunk_id, _ in search_results]
            
            # Calculate metrics for each k
            for k in k_values:
                metrics = self.calculate_metrics(retrieved_ids, relevant_set, k)
                results[f'precision@{k}'].append(metrics['precision'])
                results[f'recall@{k}'].append(metrics['recall'])
                results[f'ndcg@{k}'].append(metrics['ndcg'])
            
            # AP for MAP
            metrics_full = self.calculate_metrics(retrieved_ids, relevant_set, max_k)
            results['ap_scores'].append(metrics_full['ap'])
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(queries)} queries...")
        
        elapsed = time.time() - start
        print(f"✅ Evaluation complete: {elapsed:.2f}s")
        print()
        
        # Aggregate metrics
        for k in k_values:
            results[f'precision@{k}_mean'] = np.mean(results[f'precision@{k}'])
            results[f'recall@{k}_mean'] = np.mean(results[f'recall@{k}'])
            results[f'ndcg@{k}_mean'] = np.mean(results[f'ndcg@{k}'])
        
        results['map'] = np.mean(results['ap_scores'])
        
        return results
    
    def print_results(self, results: Dict):
        """Pretty print results."""
        print(f"{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Queries Evaluated: {results['queries_evaluated']}")
        print(f"  Single-relevance: {results['num_single_relevance']}")
        print(f"  Multi-relevance: {results['num_multi_relevance']}")
        print(f"  Avg relevant per query: {results['avg_relevant_per_query']:.2f}")
        print()
        
        print(f"📊 Overall Metrics:")
        print(f"  MAP: {results['map']:.4f}")
        print()
        
        k_values = sorted([int(k.split('@')[1].split('_')[0]) 
                          for k in results.keys() if 'precision@' in k and '_mean' in k])
        
        print(f"  {'Metric':<15} " + "  ".join(f"@{k:<3}" for k in k_values))
        print(f"  {'-'*15} " + "  ".join("-"*5 for _ in k_values))
        
        print(f"  {'Precision':<15} " + "  ".join(f"{results[f'precision@{k}_mean']:.3f}" for k in k_values))
        print(f"  {'Recall':<15} " + "  ".join(f"{results[f'recall@{k}_mean']:.3f}" for k in k_values))
        print(f"  {'nDCG':<15} " + "  ".join(f"{results[f'ndcg@{k}_mean']:.3f}" for k in k_values))
        
        print(f"{'='*80}\n")
    
    def compare_with_baseline(self, n_to_n_results: Dict, baseline_results: Dict):
        """Compare N-to-N with 1-to-1 baseline."""
        print(f"{'='*80}")
        print("COMPARISON: N-to-N vs 1-to-1 Baseline")
        print(f"{'='*80}\n")
        
        print(f"{'Metric':<20} {'N-to-N':<12} {'1-to-1':<12} {'Diff':<10}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*10}")
        
        # MAP vs MRR comparison
        map_score = n_to_n_results['map']
        mrr_score = baseline_results.get('mrr', 0)
        print(f"{'MAP/MRR':<20} {map_score:<12.4f} {mrr_score:<12.4f} {map_score-mrr_score:+.4f}")
        
        # Precision/Recall vs Hit Rate
        for k in [5, 10, 20]:
            prec = n_to_n_results.get(f'precision@{k}_mean', 0)
            rec = n_to_n_results.get(f'recall@{k}_mean', 0)
            hit = baseline_results.get(f'hit_rate@{k}', 0)
            
            print(f"{'Precision@' + str(k):<20} {prec:<12.4f}")
            print(f"{'Recall@' + str(k):<20} {rec:<12.4f}")
            print(f"{'Hit Rate@' + str(k) + ' (1-to-1)':<20} {'':<12} {hit:<12.4f}")
            print()
        
        print(f"{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate N-to-N queries")
    parser.add_argument('--chunks', type=str, 
                        default='data/30002051_chunks_split_prefixed.json')
    parser.add_argument('--queries', type=str,
                        default='data/queries_n_to_n.json')
    parser.add_argument('--baseline', type=str,
                        default='output/baseline_bge_m3_results.json',
                        help='Baseline 1-to-1 results for comparison')
    parser.add_argument('--output', type=str,
                        default='output/n_to_n_results.json')
    parser.add_argument('--no-prefix', action='store_true')
    
    args = parser.parse_args()
    
    evaluator = MultiRelevanceEvaluator()
    
    queries = evaluator.load_data(args.chunks, args.queries)
    evaluator.build_index(use_prefix=not args.no_prefix)
    
    results = evaluator.evaluate(queries, k_values=[5, 10, 20])
    evaluator.print_results(results)
    
    # Compare with baseline if available
    if Path(args.baseline).exists():
        with open(args.baseline, 'r') as f:
            baseline_results = json.load(f)
        evaluator.compare_with_baseline(results, baseline_results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON
    results_json = {}
    for k, v in results.items():
        if isinstance(v, list):
            results_json[k] = v
        elif isinstance(v, (np.floating, float)):
            results_json[k] = float(v)
        else:
            results_json[k] = v
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
