#!/usr/bin/env python3
"""
Hard Negative Mining for BGE-M3 Official Approach.

For each query:
  - Retrieve top-200 chunks using BGE-M3
  - Filter out true positives and same-group chunks
  - Skip rank 1 (likely false negative), take from ranks 2-200
  - Select 7 hard negatives
  - Randomly sample 1 positive per query for training

Output: output/bge_negatives.json (ready for teacher scoring step)
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


class HardNegativeMinerBGE:
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
        return chunk.get("group_id") or chunk.get("date")

    def _positive_group_ids(self, positive_ids: List[str]) -> Set[str]:
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

    def mine_for_query(self, query_data: Dict, retrieval_k: int = 200, num_negatives: int = 7) -> Optional[Dict]:
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

        # Skip rank 1 (most similar — likely false negative), take from rank 2 onward
        safe_candidates = filtered[1:] if len(filtered) > 1 else filtered

        if not safe_candidates:
            return None

        # Take top num_negatives from safe candidates
        selected = safe_candidates[:num_negatives]

        if len(selected) < num_negatives:
            # Warn but don't skip — use what we have
            pass

        neg_chunks = []
        for cid, score in selected:
            chunk = self.chunks[self.chunk_id_to_idx[cid]]
            text = chunk.get("text_without_prefix") or chunk.get("text", "")
            neg_chunks.append({
                "chunk_id": cid,
                "text": text,
                "similarity_score": score,
            })

        # Randomly sample 1 positive for training
        sampled_pos_id = random.choice(list(positive_ids))
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
            "hard_negatives": neg_chunks,
            "num_negatives_found": len(neg_chunks),
            "num_candidates_after_filter": len(filtered),
        }

    def mine_all(self, queries: List[Dict], retrieval_k: int = 200, num_negatives: int = 7) -> List[Dict]:
        print(f"\nMining hard negatives for {len(queries)} queries (retrieval_k={retrieval_k}, num_negatives={num_negatives})...")
        results = []
        skipped = 0
        partial = 0
        for i, q in enumerate(queries):
            record = self.mine_for_query(q, retrieval_k=retrieval_k, num_negatives=num_negatives)
            if record:
                if record["num_negatives_found"] < num_negatives:
                    partial += 1
                results.append(record)
            else:
                skipped += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(queries)} processed, {skipped} skipped, {partial} partial...")
        print(f"✅ Mining complete: {len(results)} examples, {skipped} skipped, {partial} with fewer than {num_negatives} negatives")
        return results

    def compute_stats(self, examples: List[Dict], num_negatives: int) -> Dict:
        full = sum(1 for e in examples if e["num_negatives_found"] == num_negatives)
        top1_scores = [e["hard_negatives"][0]["similarity_score"] for e in examples if e["hard_negatives"]]
        return {
            "total_examples": len(examples),
            "examples_with_full_negatives": full,
            "avg_top1_negative_similarity": float(np.mean(top1_scores)) if top1_scores else 0.0,
            "min_top1_negative_similarity": float(np.min(top1_scores)) if top1_scores else 0.0,
            "max_top1_negative_similarity": float(np.max(top1_scores)) if top1_scores else 0.0,
        }

    def save(self, examples: List[Dict], stats: Dict, output_path: str, model_name: str, retrieval_k: int, num_negatives: int):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "metadata": {
                "approach": "BGE-M3-official",
                "model": model_name,
                "retrieval_k": retrieval_k,
                "num_negatives": num_negatives,
                "negative_selection": "ranks 2 to retrieval_k after filtering positives",
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
    parser = argparse.ArgumentParser(description="Mine hard negatives for BGE-M3 official approach")
    parser.add_argument("--chunks", default="data/layer1_pilot_pairs_550_grouped_3_4.json")
    parser.add_argument("--queries", default="data/queries_layer1_n2n_pilot_final_v2.json")
    parser.add_argument("--output", default="output/bge_negatives.json")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--retrieval-k", type=int, default=200)
    parser.add_argument("--num-negatives", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    miner = HardNegativeMinerBGE(model_name=args.model, batch_size=args.batch_size)
    chunks, queries = miner.load_data(args.chunks, args.queries)
    miner.build_index(chunks)
    examples = miner.mine_all(queries, retrieval_k=args.retrieval_k, num_negatives=args.num_negatives)
    stats = miner.compute_stats(examples, args.num_negatives)

    print(f"\n📊 Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    miner.save(examples, stats, args.output, args.model, args.retrieval_k, args.num_negatives)


if __name__ == "__main__":
    main()
