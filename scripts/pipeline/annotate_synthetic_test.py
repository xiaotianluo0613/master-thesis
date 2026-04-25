#!/usr/bin/env python3
"""
LLM-based relevance annotation for synthetic test set candidates.

For each (query, candidate_chunk) pair:
  - source chunks (from query generation group) → automatically relevant: true
  - all other retrieved candidates → judged by Gemini Flash

Assembles final relevant_chunks list per query and writes evaluation-ready JSON
in the same format as global_val_queries.json.

Usage:
    python scripts/pipeline/annotate_synthetic_test.py \
        --candidates data/test_synthetic_candidates.json \
        --output     data/test_synthetic_queries.json \
        --model      gemini-2.5-flash \
        --resume

Environment:
    GEMINI_API_KEY  required
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests


RELEVANCE_PROMPT = """\
[SYSTEM]
You are a historian evaluating search results for 19th century Swedish archive documents.
Your task: decide if the passage is relevant to the query.

A passage is RELEVANT if it would help a historian answer the query, even partially.
A passage is NOT_RELEVANT if it is about unrelated people, places, or events.

Answer ONLY with one of these two tokens on the first line:
RELEVANT
NOT_RELEVANT

Then on the next line, write:
Reason: <one brief sentence explaining why>

[USER]
Query: {query}

Archive passage:
{text}
"""


def get_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not set")
    return key


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
                exp = min(rate_limit_hits - 1, 6)
                wait = min(900, int(60 * (2 ** exp)))
                jitter = random.uniform(0.0, 2.0)
                print(f"  Rate limit, waiting {wait}s...")
                time.sleep(wait + jitter)
                continue
            r.raise_for_status()
            data = r.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"Gemini empty response: {data}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                raise RuntimeError(f"Gemini empty text: {data}")
            return text
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                raise
            print(f"  API retry {attempt}/{max_retries}: {e}")
            time.sleep(min(900, 5 * attempt))

    raise RuntimeError("No response after retries")


def parse_gemini_response(response: str) -> tuple[bool, str]:
    """Parse RELEVANT/NOT_RELEVANT + reason from Gemini output."""
    lines = response.strip().splitlines()
    decision_line = lines[0].strip().upper() if lines else ""
    relevant = "NOT_RELEVANT" not in decision_line and "RELEVANT" in decision_line

    reason = ""
    for line in lines[1:]:
        if line.lower().startswith("reason:"):
            reason = line[len("reason:"):].strip()
            break

    return relevant, reason


def main():
    parser = argparse.ArgumentParser(description="LLM annotation for synthetic test candidates")
    parser.add_argument("--candidates", required=True,
                        help="Path to test_synthetic_candidates.json")
    parser.add_argument("--output", required=True,
                        help="Path to output test_synthetic_queries.json")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--resume", action="store_true",
                        help="Skip queries already present in output file")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N queries")
    args = parser.parse_args()

    api_key = get_gemini_key()

    with open(args.candidates, encoding="utf-8") as f:
        candidates_data = json.load(f)
    queries = candidates_data["queries"]
    print(f"Loaded {len(queries)} queries from {args.candidates}")

    # Load existing output for resume
    done_ids = set()
    existing_queries = []
    if args.resume and Path(args.output).exists():
        with open(args.output, encoding="utf-8") as f:
            existing = json.load(f)
        existing_queries = existing.get("queries", [])
        done_ids = {q["query_id"] for q in existing_queries}
        print(f"Resume: {len(done_ids)} queries already annotated")

    annotated_queries = list(existing_queries)

    def save_checkpoint():
        out = {
            "metadata": {
                "candidates_file": args.candidates,
                "model": args.model,
                "total_queries": len(annotated_queries),
            },
            "queries": annotated_queries,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    pending = [q for q in queries if q["query_id"] not in done_ids]
    print(f"Queries to annotate: {len(pending)}")

    total_auto = 0
    total_llm = 0
    total_relevant = 0

    for qi, q in enumerate(pending):
        qid = q["query_id"]
        source_chunks = set(q.get("source_chunks", []))
        candidates = q.get("candidates", [])

        annotations = []
        relevant_chunks = []

        for c in candidates:
            cid = c["chunk_id"]
            if cid in source_chunks:
                # Auto-positive: source chunk
                annotations.append({
                    "chunk_id": cid,
                    "relevant": True,
                    "source": "auto",
                    "reason": "source chunk for this query",
                })
                relevant_chunks.append(cid)
                total_auto += 1
            else:
                # LLM judgment
                prompt = RELEVANCE_PROMPT.format(
                    query=q["query"],
                    text=c.get("text", "")[:2000],  # truncate very long texts
                )
                try:
                    response = call_gemini(prompt, api_key, args.model)
                    rel, reason = parse_gemini_response(response)
                except Exception as e:
                    print(f"  Error annotating {qid}/{cid}: {e}")
                    rel, reason = False, f"annotation error: {e}"

                annotations.append({
                    "chunk_id": cid,
                    "relevant": rel,
                    "source": "llm",
                    "reason": reason,
                })
                if rel:
                    relevant_chunks.append(cid)
                total_llm += 1

        total_relevant += len(relevant_chunks)

        annotated_queries.append({
            "query_id": qid,
            "query": q["query"],
            "layer": q.get("layer", ""),
            "source_chunks": list(source_chunks),
            "relevant_chunks": relevant_chunks,
            "num_relevant": len(relevant_chunks),
            "annotations": annotations,
        })

        if (qi + 1) % args.save_every == 0 or (qi + 1) == len(pending):
            save_checkpoint()
            print(f"  [{qi+1}/{len(pending)}] Saved. "
                  f"Auto: {total_auto}, LLM: {total_llm}, "
                  f"Avg relevant: {total_relevant / len(annotated_queries):.2f}")

    save_checkpoint()
    print(f"\nDone. {len(annotated_queries)} queries annotated.")
    print(f"  Auto-positives: {total_auto}")
    print(f"  LLM-judged: {total_llm}")
    print(f"  Avg relevant chunks per query: "
          f"{total_relevant / max(len(annotated_queries), 1):.2f}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
