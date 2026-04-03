#!/usr/bin/env python3
"""
Baseline Retrieval Test with BGE-M3
Evaluates query-chunk retrieval performance on historical Swedish police reports.

Features:
- Uses BGE-M3 multilingual embeddings
- FAISS vector store with cosine similarity
- Includes distractor chunks (all non-relevant chunks)
- Metrics: MRR and Hit Rate@K
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time
from collections import defaultdict

# Check for dependencies
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError as e:
    print(f"Missing dependencies. Install with:")
    print("pip install sentence-transformers faiss-cpu")
    raise e


class BGEBaseline:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """Initialize BGE-M3 model and vector store."""
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
        
        self.index = None
        self.chunks = []
        self.chunk_id_to_idx = {}
        
    def load_data(self, chunks_file: str, queries_file: str) -> Tuple[List[Dict], List[Dict]]:
        """Load chunks and queries from JSON files."""
        print(f"\nLoading data...")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            chunks = chunks_data['chunks']
        
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
            queries = queries_data['queries']
        
        print(f"✅ Loaded {len(chunks)} chunks and {len(queries)} queries")
        return chunks, queries
    
    def build_index(self, chunks: List[Dict], use_prefix: bool = True):
        """
        Build FAISS index with all chunks (including distractors).
        
        Args:
            chunks: List of chunk dictionaries
            use_prefix: If True, use text with prefix; if False, use text_without_prefix
        """
        print(f"\n{'='*80}")
        print("BUILDING VECTOR INDEX")
        print(f"{'='*80}")
        
        self.chunks = chunks
        
        # Extract text based on preference
        text_field = 'text' if use_prefix else 'text_without_prefix'
        texts = [chunk[text_field] for chunk in chunks]
        
        print(f"Using field: '{text_field}'")
        print(f"Encoding {len(texts)} chunks...")
        start = time.time()
        
        # Encode in batches
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        
        elapsed = time.time() - start
        print(f"✅ Encoding complete: {elapsed:.2f}s ({len(texts)/elapsed:.1f} chunks/sec)")
        
        # Build FAISS index with cosine similarity (inner product with normalized vectors)
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for normalized vectors = cosine
        self.index.add(embeddings.astype('float32'))
        
        # Create chunk_id lookup
        self.chunk_id_to_idx = {chunk['chunk_id']: i for i, chunk in enumerate(chunks)}
        
        print(f"✅ Index built: {self.index.ntotal} vectors")
        print(f"{'='*80}\n")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for top-k relevant chunks.
        
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return chunk_ids and scores
        results = [
            (self.chunks[idx]['chunk_id'], float(score))
            for idx, score in zip(indices[0], scores[0])
        ]
        
        return results
    
    def evaluate(self, queries: List[Dict], k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate retrieval performance.
        
        Metrics:
        - MRR (Mean Reciprocal Rank)
        - Hit Rate@K (percentage of queries with relevant chunk in top-K)
        """
        print(f"{'='*80}")
        print("EVALUATING RETRIEVAL PERFORMANCE")
        print(f"{'='*80}")
        print(f"Queries: {len(queries)}")
        print(f"K values: {k_values}")
        print()
        
        max_k = max(k_values)
        results = {
            'mrr': 0.0,
            'queries_evaluated': 0,
            'queries_with_results': 0,
        }
        
        # Initialize metrics for each k
        for k in k_values:
            results[f'hit_rate@{k}'] = 0.0
        
        reciprocal_ranks = []
        hit_rate_at_k = {k: [] for k in k_values}
        
        # Per-query-type metrics
        query_type_metrics = defaultdict(lambda: {
            'count': 0,
            'reciprocal_ranks': []
        })
        
        print("Processing queries...")
        start = time.time()
        
        for i, query_data in enumerate(queries):
            query_text = query_data['query']
            relevant_chunk = query_data['relevant_chunk']
            query_type = query_data['query_type']
            
            # Search
            search_results = self.search(query_text, k=max_k)
            retrieved_ids = [chunk_id for chunk_id, _ in search_results]
            
            # Find rank of relevant chunk
            try:
                rank = retrieved_ids.index(relevant_chunk) + 1  # 1-indexed
                reciprocal_rank = 1.0 / rank
                query_type_metrics[query_type]['reciprocal_ranks'].append(reciprocal_rank)
            except ValueError:
                # Relevant chunk not in top-k
                rank = None
                reciprocal_rank = 0.0
            
            reciprocal_ranks.append(reciprocal_rank)
            
            # Calculate metrics for each k
            for k in k_values:
                retrieved_k = retrieved_ids[:k]
                
                # Hit Rate@K (binary: did we find the relevant chunk?)
                hit = 1.0 if relevant_chunk in retrieved_k else 0.0
                hit_rate_at_k[k].append(hit)
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(queries)} queries...")
            
            query_type_metrics[query_type]['count'] += 1
        
        elapsed = time.time() - start
        print(f"✅ Evaluation complete: {elapsed:.2f}s ({len(queries)/elapsed:.1f} queries/sec)")
        print()
        
        # Aggregate metrics
        results['mrr'] = np.mean(reciprocal_ranks)
        results['queries_evaluated'] = len(queries)
        results['queries_with_results'] = sum(1 for rr in reciprocal_ranks if rr > 0)
        
        for k in k_values:
            results[f'hit_rate@{k}'] = np.mean(hit_rate_at_k[k])
        
        # Per-query-type MRR
        results['per_query_type'] = {}
        for qtype, metrics in query_type_metrics.items():
            results['per_query_type'][qtype] = {
                'count': metrics['count'],
                'mrr': np.mean(metrics['reciprocal_ranks']) if metrics['reciprocal_ranks'] else 0.0
            }
        
        return results
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results."""
        print(f"{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Queries Evaluated: {results['queries_evaluated']}")
        print(f"Queries with Results: {results['queries_with_results']} " +
              f"({results['queries_with_results']/results['queries_evaluated']*100:.1f}%)")
        print()
        print(f"📊 Overall Metrics:")
        print(f"  MRR: {results['mrr']:.4f}")
        print()
        
        # Hit Rate
        k_values = [int(k.split('@')[1]) for k in results.keys() if 'hit_rate@' in k]
        k_values.sort()
        
        print(f"  {'Metric':<15} " + "  ".join(f"@{k:<3}" for k in k_values))
        print(f"  {'-'*15} " + "  ".join("-"*5 for _ in k_values))
        print(f"  {'Hit Rate':<15} " + "  ".join(f"{results[f'hit_rate@{k}']:.3f}" for k in k_values))
        print()
        
        # Per-query-type
        print(f"📈 Per Query Type:")
        for qtype, metrics in results['per_query_type'].items():
            print(f"  {qtype:<20} Count: {metrics['count']:<4} MRR: {metrics['mrr']:.4f}")
        
        print(f"{'='*80}\n")
    
    def sample_results(self, queries: List[Dict], n: int = 5):
        """Show sample retrieval results."""
        print(f"{'='*80}")
        print(f"SAMPLE RESULTS (n={n})")
        print(f"{'='*80}\n")
        
        import random
        samples = random.sample(queries, min(n, len(queries)))
        
        for i, query_data in enumerate(samples, 1):
            query_text = query_data['query']
            relevant_chunk = query_data['relevant_chunk']
            query_type = query_data['query_type']
            
            print(f"[{i}] {query_type.upper()}")
            print(f"Query: {query_text}")
            print(f"Expected: {relevant_chunk}")
            print()
            
            results = self.search(query_text, k=5)
            print("Top 5 Results:")
            for rank, (chunk_id, score) in enumerate(results, 1):
                marker = "✅" if chunk_id == relevant_chunk else "  "
                print(f"  {marker} {rank}. {chunk_id} (score: {score:.4f})")
            print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BGE-M3 Baseline Retrieval Evaluation")
    parser.add_argument('--chunks', type=str, 
                        default='data/30002051_chunks_split_prefixed.json',
                        help='Path to chunks JSON file')
    parser.add_argument('--queries', type=str,
                        default='data/generated_queries_complete.json',
                        help='Path to queries JSON file')
    parser.add_argument('--no-prefix', action='store_true',
                        help='Use text without prefix (remove metadata header)')
    parser.add_argument('--output', type=str,
                        default='output/baseline_bge_m3_results.json',
                        help='Output file for results')
    parser.add_argument('--sample', type=int, default=5,
                        help='Number of sample results to display')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("BGE-M3 BASELINE EVALUATION")
    print(f"{'='*80}")
    print(f"Chunks: {args.chunks}")
    print(f"Queries: {args.queries}")
    print(f"Use prefix: {not args.no_prefix}")
    print(f"{'='*80}\n")
    
    # Initialize
    baseline = BGEBaseline()
    
    # Load data
    chunks, queries = baseline.load_data(args.chunks, args.queries)
    
    # Build index with ALL chunks (including distractors)
    baseline.build_index(chunks, use_prefix=not args.no_prefix)
    
    # Evaluate
    results = baseline.evaluate(queries, k_values=[5, 10, 20])
    
    # Print results
    baseline.print_results(results)
    
    # Show samples
    baseline.sample_results(queries, n=args.sample)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
