#!/usr/bin/env python3
"""
Re-label queries with LLM-as-judge using multi-model pooling (Option 1)

Strategy:
1. Retrieve top-10 from BM25 + mE5-small (lexical + neural)
2. Pool unique results (deduplicate)
3. Use Gemini to judge relevance of each pooled chunk
4. Create expanded dataset with multiple relevant labels
5. Test on BGE-M3, mContriever, mDPR (independent models)

This avoids circular evaluation!
"""

import json
import numpy as np
from pathlib import Path
import time
from collections import defaultdict
import argparse
import os
import requests
from typing import List, Dict, Set
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Install: pip install sentence-transformers faiss-cpu rank-bm25")
    raise


class MultiModelRetriever:
    """Retrieve from multiple models and pool results"""
    
    def __init__(self, chunks_file: str, baseline_queries_file: str = 'data/generated_queries_complete.json',
                 allowed_chunk_ids: Set[str] = None):
        print("=" * 80)
        print("LOADING RETRIEVAL MODELS (BM25 + mE5-small)")
        print("=" * 80)
        
        if allowed_chunk_ids is not None:
            # Use explicitly provided chunk IDs (e.g. a sampled subset)
            chunk_ids_to_use = allowed_chunk_ids
            print(f"\n🎯 Using {len(chunk_ids_to_use)} provided chunk IDs (sampled subset)")
        else:
            # Default: load baseline chunk IDs for fair comparison
            print("\n🎯 Loading baseline chunk IDs for fair comparison...")
            with open(baseline_queries_file) as f:
                baseline_queries = json.load(f)['queries']
            chunk_ids_to_use = set(q['chunk_id'] for q in baseline_queries)
            print(f"✅ Using {len(chunk_ids_to_use)} baseline chunks (same as 1-to-1)")
        
        # Load and filter chunks
        with open(chunks_file) as f:
            all_chunks = json.load(f)['chunks']
        self.chunks = [c for c in all_chunks if c['chunk_id'] in chunk_ids_to_use]
        print(f"✅ Filtered to {len(self.chunks)} chunks (from {len(all_chunks)} total)")
        
        self.chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks)}
        self.texts = [c.get('text_without_prefix', c.get('text', '')) for c in self.chunks]
        
        # Initialize retrieval models (NOT the test models!)
        self.models = {}
        self.indices = {}
        
        # mE5-small (neural retrieval)
        print("\n1️⃣  Loading mE5-small (neural)...")
        self.models['me5-small'] = SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')
        embeddings = self.models['me5-small'].encode(
            self.texts, batch_size=8, show_progress_bar=True, normalize_embeddings=True
        )
        self.indices['me5-small'] = faiss.IndexFlatIP(embeddings.shape[1])
        self.indices['me5-small'].add(embeddings.astype('float32'))
        print("✅ mE5-small ready")
        
        # BM25 (lexical retrieval)
        print("\n2️⃣  Building BM25 index (lexical)...")
        tokenized = [text.lower().split() for text in self.texts]
        self.indices['bm25'] = BM25Okapi(tokenized)
        print("✅ BM25 ready")
        
        print("\n" + "=" * 80)
        print("RETRIEVAL MODELS LOADED (BM25 + mE5-small)")
        print("NOTE: We will TEST on BGE-M3, mContriever, mDPR (independent models)")
        print("=" * 80)
    
    def retrieve(self, query: str, model_name: str, k: int = 10) -> List[str]:
        """Retrieve top-k chunk IDs from a specific model"""
        
        if model_name == 'bm25':
            tokenized_query = query.lower().split()
            scores = self.indices['bm25'].get_scores(tokenized_query)
            top_indices = np.argsort(scores)[-k:][::-1]
        else:
            query_emb = self.models[model_name].encode([query], normalize_embeddings=True)
            scores, top_indices = self.indices[model_name].search(query_emb.astype('float32'), k)
            top_indices = top_indices[0]
        
        return [self.chunks[idx]['chunk_id'] for idx in top_indices]
    
    def pool_results(self, query: str, k: int = 10) -> List[Dict]:
        """Retrieve from BM25 + mE5-small and pool unique results"""
        
        pooled_chunk_ids = set()
        model_rankings = {}
        
        # Only retrieve from BM25 and mE5-small (NOT the test models!)
        for model_name in ['bm25', 'me5-small']:
            chunk_ids = self.retrieve(query, model_name, k)
            model_rankings[model_name] = chunk_ids
            pooled_chunk_ids.update(chunk_ids)
        
        # Get chunk objects
        pooled_chunks = []
        for chunk_id in pooled_chunk_ids:
            idx = self.chunk_id_to_idx[chunk_id]
            chunk = self.chunks[idx].copy()
            chunk['retrieved_by'] = [m for m, ids in model_rankings.items() if chunk_id in ids]
            pooled_chunks.append(chunk)
        
        return pooled_chunks


class GithubModelsJudge:
    """Use GitHub Models (gpt-4o-mini) to judge relevance — no rate limit issues"""
    
    def __init__(self, token: str, model: str = "gpt-4o-mini"):
        self.token = token
        self.model = model
        self.url = "https://models.inference.ai.azure.com/chat/completions"
    
    def judge(self, query: str, chunk_text: str, max_retries: int = 3, debug: bool = False) -> bool:
        """Judge if chunk is relevant to query"""
        
        system_prompt = (
            "You are an expert historian and an impartial judge evaluating a search engine. "
            "You will be given a user query (in modern Swedish) and a retrieved historical Swedish court/police document. "
            "Your task is to determine if the document contains information that helps answer the query "
            "or is clearly part of the same legal case/event. "
            "Please consider that the document may contain historical spelling, HTR OCR errors, and archaic words. "
            'Respond ONLY with a valid JSON object: {"relevant": true} or {"relevant": false}'
        )
        
        user_prompt = f"Query: {query}\n\nDocument: {chunk_text}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 429:
                    wait_time = 15 * (attempt + 1)
                    print(f"⏳ Rate limit (429), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if not response.ok:
                    print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
                    time.sleep(5)
                    continue
                
                result = response.json()
                text = result['choices'][0]['message']['content'].strip()
                
                if debug:
                    print(f"    🔍 Raw response: {repr(text)}")
                
                if '{' in text and '}' in text:
                    json_text = text[text.index('{'):text.rindex('}')+1]
                    parsed = json.loads(json_text)
                    is_rel = parsed.get('relevant', False)
                    if debug:
                        print(f"    📋 Parsed: {is_rel}")
                    return is_rel
                
                is_rel = 'true' in text.lower()
                if debug:
                    print(f"    📋 Fallback parse: {is_rel}")
                return is_rel
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ Failed after {max_retries} attempts: {e}")
                    return False
                time.sleep(2 ** attempt)
        
        return False


def relabel_dataset(queries_file: str, chunks_file: str, output_file: str,
                   github_token: str, delay: float = 1.0, max_queries: int = None,
                   judge_model: str = "gpt-4o-mini",
                   sample_chunks: int = None, seed: int = 42):
    """Main re-labeling pipeline"""
    
    import random
    
    print("=" * 80)
    print("LLM-BASED RE-LABELING WITH MULTI-MODEL POOLING")
    print("=" * 80)
    
    # Load queries
    print(f"\nLoading queries from {queries_file}...")
    with open(queries_file) as f:
        queries_data = json.load(f)
    queries = queries_data['queries']
    
    # --- Chunk sampling (reduces test set size) ---
    all_baseline_chunk_ids = list(set(q['chunk_id'] for q in queries))
    if sample_chunks and sample_chunks < len(all_baseline_chunk_ids):
        random.seed(seed)
        sampled_ids = set(random.sample(all_baseline_chunk_ids, sample_chunks))
        queries = [q for q in queries if q['chunk_id'] in sampled_ids]
        print(f"✅ Sampled {len(sampled_ids)} chunks (seed={seed}) → {len(queries)} queries")
    else:
        sampled_ids = None  # use all
        print(f"✅ Using all {len(all_baseline_chunk_ids)} baseline chunks → {len(queries)} queries")
    
    # Limit for testing
    if max_queries and max_queries < len(queries):
        queries = queries[:max_queries]
        print(f"⚠️  Further limited to {max_queries} queries for testing")
    
    # Initialize retriever (only index the chunks we actually need)
    retriever = MultiModelRetriever(chunks_file, allowed_chunk_ids=sampled_ids)
    
    # Initialize judge
    judge = GithubModelsJudge(github_token, model=judge_model)
    print(f"🤖 Judge model: {judge_model} (via GitHub Models)")
    
    # Re-label
    print("\n" + "=" * 80)
    print("STARTING RE-LABELING PROCESS")
    print("=" * 80)
    
    relabeled_queries = []
    stats = {
        'total_queries': len(queries),
        'total_judgments': 0,
        'total_relevant': 0,
        'avg_pooled_per_query': 0,
        'avg_relevant_per_query': 0
    }
    
    start_time = time.time()
    
    for i, query_entry in enumerate(queries, 1):
        query_text = query_entry['query']
        original_chunk_id = query_entry['chunk_id']
        
        print(f"\n[{i}/{len(queries)}] Processing query: {query_text[:60]}...")
        
        # Pool results from all models
        pooled_chunks = retriever.pool_results(query_text, k=10)
        print(f"  📊 Pooled {len(pooled_chunks)} unique chunks from 2 models (BM25 + mE5-small)")
        
        # Judge each chunk
        relevant_chunk_ids = []
        debug_mode = (i <= 2)  # Debug first 2 queries
        for j, chunk in enumerate(pooled_chunks, 1):
            chunk_text = chunk.get('text_without_prefix', chunk.get('text', ''))
            if debug_mode:
                print(f"  [{j}/{len(pooled_chunks)}] Judging chunk: {chunk['chunk_id']}")
            is_relevant = judge.judge(query_text, chunk_text, debug=debug_mode)
            
            if is_relevant:
                relevant_chunk_ids.append(chunk['chunk_id'])
                if debug_mode:
                    print(f"    ✅ RELEVANT!")
            elif debug_mode:
                print(f"    ❌ Not relevant")
            
            stats['total_judgments'] += 1
            time.sleep(delay)
        
        # Create new entry
        relabeled_entry = {
            'query': query_text,
            'query_type': query_entry.get('query_type', 'unknown'),
            'original_chunk_id': original_chunk_id,
            'relevant_chunks': relevant_chunk_ids,
            'num_relevant': len(relevant_chunk_ids),
            'pooled_chunks': [c['chunk_id'] for c in pooled_chunks],
            'num_pooled': len(pooled_chunks)
        }
        
        relabeled_queries.append(relabeled_entry)
        stats['total_relevant'] += len(relevant_chunk_ids)
        
        print(f"  ✅ Found {len(relevant_chunk_ids)} relevant chunks (out of {len(pooled_chunks)} pooled)")
        
        # Progress update
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(queries) - i) / rate
            print(f"\n  Progress: {i}/{len(queries)} ({elapsed/60:.1f}m elapsed, {remaining/60:.1f}m remaining)")
    
    # Calculate stats
    stats['avg_pooled_per_query'] = stats['total_judgments'] / len(queries)
    stats['avg_relevant_per_query'] = stats['total_relevant'] / len(queries)
    stats['total_time_seconds'] = time.time() - start_time
    
    # Save results
    output_data = {
        'metadata': {
            'method': 'multi-model-pooling-llm-judge',
            'retrieval_models_used': ['bm25', 'me5-small'],
            'test_models_planned': ['bge-m3', 'mcontriever', 'mdpr'],
            'judge_model': f'{judge.model} (GitHub Models)',
            'k_per_model': 10,
            'original_queries_file': queries_file,
            'chunks_file': chunks_file,
            'sample_chunks': sample_chunks,
            'sample_seed': seed if sample_chunks else None,
            'num_chunks_used': len(sampled_ids) if sampled_ids else len(all_baseline_chunk_ids)
        },
        'statistics': stats,
        'queries': relabeled_queries
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("RE-LABELING COMPLETE")
    print("=" * 80)
    print(f"\n📊 Statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Total judgments: {stats['total_judgments']}")
    print(f"  Total relevant: {stats['total_relevant']}")
    print(f"  Avg pooled per query: {stats['avg_pooled_per_query']:.1f}")
    print(f"  Avg relevant per query: {stats['avg_relevant_per_query']:.1f}")
    print(f"  Total time: {stats['total_time_seconds']/60:.1f} minutes")
    print(f"\n💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Re-label queries with multi-model pooling + LLM judge')
    parser.add_argument('--queries', default='data/generated_queries_complete.json',
                       help='Input queries file')
    parser.add_argument('--chunks', default='data/30002051_chunks_split_prefixed.json',
                       help='Chunks file')
    parser.add_argument('--output', default='data/queries_relabeled_multimodel.json',
                       help='Output file for re-labeled queries')
    parser.add_argument('--github-token', help='GitHub token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--judge-model', default='gpt-4o-mini',
                       help='GitHub Models model name (default: gpt-4o-mini)')
    parser.add_argument('--delay', type=float, default=4.0,
                       help='Delay between API calls (seconds)')
    parser.add_argument('--max-queries', type=int, default=None,
                       help='Max number of queries to process (for testing)')
    parser.add_argument('--sample-chunks', type=int, default=None,
                       help='Randomly sample N chunks from baseline (e.g. 50 for ~50%% test set)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for chunk sampling (default: 42)')
    
    args = parser.parse_args()
    
    # Get GitHub token
    try:
        import subprocess
        github_token = args.github_token or os.environ.get('GITHUB_TOKEN')
        if not github_token:
            result = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True, check=True)
            github_token = result.stdout.strip()
    except Exception:
        github_token = None
    
    if not github_token:
        print("❌ Error: GitHub token required")
        print("   Set via --github-token, GITHUB_TOKEN env var, or run 'gh auth login'")
        exit(1)
    
    relabel_dataset(
        args.queries,
        args.chunks,
        args.output,
        github_token,
        args.delay,
        args.max_queries,
        args.judge_model,
        args.sample_chunks,
        args.seed
    )


if __name__ == '__main__':
    main()
