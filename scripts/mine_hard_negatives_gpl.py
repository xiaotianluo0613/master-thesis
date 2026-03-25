#!/usr/bin/env python3
"""
Hard Negative Mining for GPL (Generative Pseudo Labelling) Approach.

For each query:
  - Retrieve top-10 chunks using BGE-M3
  - Filter out true positives and same-group chunks
  - Select the BOTTOM 1 from remaining (avoids false negatives at top)
  - Randomly sample 1 positive per query for training

Output: output/gpl_negatives.json
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class HardNegativeMinerGPL:
    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 32):
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.index = None
        self.chunks: List[Dict] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        print(f"Model loaded. Embedding dimension: {self.dimension}")

    def load_data(self, chunks_file: str, queries_file: str) -> Tuple[List[Dict], List[Dict]]:
        print("\nLoading data...")
        with open(chunks_file, encoding="utf-8") as f:
            chunks = json.load(f)["chunks"]
        with open(queries_file, encoding="utf-8") as f:
            queries = json.load(f)["queries"]
        print(f"✅ Loaded {len(chunks)} chunks and {len(queries)} queries")
        return chunks, queries

    def build_index(self, chunks: List[Dict]):
        print("\nBuilding FAISS index...")
        self.chunks = chunks
        texts = [c.get("text_without_prefix") or c.get("text", "") for c in chunks]
        start = time.time()
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print(f"✅ Encoded {len(texts)} chunks in {time.time()-start:.1f}s")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype("float32"))
        self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        print(f"✅ Index built: {self.index.ntotal} vectors")

    def _group_id(self, chunk: Dict) -> Optional[str]:
        """Extract group id from chunk. Falls back to date field."""
        return chunk.get("group_id") or chunk.get("date")

    def _positive_group_ids(self, positive_ids: List[str]) -> Set[str]:
        """Get all group IDs associated with the positive chunks."""
        group_ids = set()
        for cid in positive_ids:
            idx = self.chunk_id_to_idx.get(cid)
            if idx is not None:
                gid = self._group_id(self.chunks[idx])
                if gid:
                    group_ids.add(gid)
        return group_ids

    def retrieve(self, query_text: str, k: int) -> List[Tuple[str, float]]:
        emb = self.model.encode(
            [query_text], convert_to_numpy=True, normalize_embeddings=True
        )
        scores, indices = self.index.search(emb.astype("float32"), k)
        return [(self.chunks[i]["chunk_id"], float(s)) for i, s in zip(indices[0], scores[0])]

    def mine_for_query(self, query_data: Dict, retrieval_k: int = 10) -> Optional[Dict]:
        query_text = query_data["query"]
        positive_ids = set(query_data.get("relevant_chunks", []))
        positive_group_ids = self._positive_group_ids(list(positive_ids))

        candidates = self.retrieve(query_text, k=retrieval_k)

        # Filter positives and same-group chunks
        filtered = [
            (cid, score) for cid, score in candidates
            if cid not in positive_ids
            and self._group_id(self.chunks[self.chunk_id_to_idx[cid]]) not in positive_group_ids
        ]

        if not filtered:
            return None

        # Select bottom 1 from filtered candidates (safest against false negatives)
        hard_neg_id, hard_neg_score = filtered[-1]
        hard_neg_chunk = self.chunks[self.chunk_id_to_idx[hard_neg_id]]
        hard_neg_text = hard_neg_chunk.get("text_without_prefix") or hard_neg_chunk.get("text", "")

        # Randomly sample 1 positive for training
        all_positive_ids = list(positive_ids)
        sampled_pos_id = random.choice(all_positive_ids)
        sampled_pos_idx = self.chunk_id_to_idx.get(sampled_pos_id)
        if sampled_pos_idx is None:
            return None
        pos_chunk = self.chunks[sampled_pos_idx]
        pos_text = pos_chunk.get("text_without_prefix") or pos_chunk.get("text", "")

        return {
            "query_id": f"{query_data.get('date', 'unknown')}_q{query_data.get('query_index', 0)}",
            "query": query_text,
            "query_type": query_data.get("query_type", "unknown"),
            "layer": query_data.get("layer", "unknown"),
            "date": query_data.get("date", "unknown"),
            "volume_id": query_data.get("volume_id", "unknown"),
            "positive_chunk_id": sampled_pos_id,
            "positive": pos_text,
            "all_positive_ids": list(positive_ids),
            "negative_chunk_id": hard_neg_id,
            "negative": hard_neg_text,
            "negative_rank_in_retrieved": len(filtered),  # bottom of filtered list
            "negative_similarity_score": hard_neg_score,
            "num_candidates_after_filter": len(filtered),
        }

    def mine_all(self, queries: List[Dict], retrieval_k: int = 10) -> List[Dict]:
        print(f"\nMining hard negatives for {len(queries)} queries (retrieval_k={retrieval_k})...")
        results = []
        skipped = 0
        for i, q in enumerate(queries):
            record = self.mine_for_query(q, retrieval_k=retrieval_k)
            if record:
                results.append(record)
            else:
                skipped += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(queries)} processed, {skipped} skipped...")
        print(f"✅ Mining complete: {len(results)} examples, {skipped} skipped")
        return results

    def compute_stats(self, examples: List[Dict]) -> Dict:
        neg_scores = [e["negative_similarity_score"] for e in examples if e["negative_similarity_score"] is not None]
        return {
            "total_examples": len(examples),
            "avg_negative_similarity": float(np.mean(neg_scores)) if neg_scores else 0.0,
            "min_negative_similarity": float(np.min(neg_scores)) if neg_scores else 0.0,
            "max_negative_similarity": float(np.max(neg_scores)) if neg_scores else 0.0,
        }

    def save(self, examples: List[Dict], stats: Dict, output_path: str, model_name: str, retrieval_k: int):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "metadata": {
                "approach": "GPL",
                "model": model_name,
                "retrieval_k": retrieval_k,
                "negative_selection": "bottom-1 from filtered candidates",
                "positive_selection": "random 1 from relevant_chunks",
                "total_examples": len(examples),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "stats": stats,
            "examples": examples,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(examples)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives for GPL approach")
    parser.add_argument("--chunks", default="data/layer1_pilot_pairs_550_grouped_3_4.json")
    parser.add_argument("--queries", default="data/queries_layer1_n2n_pilot_final_v2.json")
    parser.add_argument("--output", default="output/gpl_negatives.json")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--retrieval-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    miner = HardNegativeMinerGPL(model_name=args.model, batch_size=args.batch_size)
    chunks, queries = miner.load_data(args.chunks, args.queries)
    miner.build_index(chunks)
    examples = miner.mine_all(queries, retrieval_k=args.retrieval_k)
    stats = miner.compute_stats(examples)

    print(f"\n📊 Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    miner.save(examples, stats, args.output, args.model, args.retrieval_k)


if __name__ == "__main__":
    main()
