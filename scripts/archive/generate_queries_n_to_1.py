#!/usr/bin/env python3
"""
Generate N-to-1 queries using GitHub Models (default: gpt-4o-mini).
This is the unified replacement for provider-named scripts.
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import requests


BASE_EXPERT_ROLE = (
    "You are an expert Swedish historian and archivist researching 19th-century crime in Sweden."
)


def build_n_to_1_system_prompt() -> str:
    return (
        f"{BASE_EXPERT_ROLE} "
        "Mode: N-to-1. Task: Generate precise queries based SOLELY on this segment. Focus on specific actors and events.\n\n"
        "Your task is to read a historical record segment and generate exactly THREE realistic user search queries in modern Swedish.\n\n"
        "Generate three distinct angles:\n"
        "1) actors/entities\n"
        "2) event/crime\n"
        "3) legal/social consequence\n\n"
        "CRITICAL CONSTRAINTS:\n"
        "- Modernize language and correct OCR noise where possible.\n"
        "- Each query must contain concrete details from the segment.\n"
        "- Return ONLY valid JSON with keys: case_summary, queries.\n\n"
        "EXAMPLE OUTPUT FORMAT:\n"
        "{\n"
        "  \"case_summary\": \"Brief summary\",\n"
        "  \"queries\": [\"q1\", \"q2\", \"q3\"]\n"
        "}"
    )


def get_github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def call_chat(prompt: str, token: str, model: str) -> str:
    url = "https://models.inference.ai.azure.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_n_to_1_system_prompt()},
            {"role": "user", "content": f"Generate JSON output for this segment:\n\n{prompt}"},
        ],
        "temperature": 0.5,
        "max_tokens": 500,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code == 429:
        raise RuntimeError("Rate limited (429)")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def parse_response(response: str) -> Dict:
    cleaned = re.sub(r"```json\s*|\s*```", "", response).strip()
    return json.loads(cleaned)


def select_chunks(chunks: List[Dict], limit: int) -> List[Dict]:
    selected = []
    for c in chunks:
        if (not c.get("is_split")) or c.get("sub_chunk_index") == 0:
            selected.append(c)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate N-to-1 queries with a unified model backend")
    parser.add_argument("--chunks", default="data/30002051_chunks_split_prefixed.json")
    parser.add_argument("--output", default="data/generated_queries_n_to_1_unified.json")
    parser.add_argument("--num-chunks", type=int, default=0)
    parser.add_argument("--queries-per-chunk", type=int, default=3)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    token = get_github_token()
    data = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    chunks = select_chunks(data["chunks"], args.num_chunks)

    rows = []
    for i, c in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] {c['chunk_id']}")
        text = c.get("text", "")
        try:
            res = parse_response(call_chat(text, token, args.model))
            queries = res.get("queries", [])[: args.queries_per_chunk]
            qtypes = ["actors", "event_crime", "legal_social_consequence"]
            for j, q in enumerate(queries):
                rows.append(
                    {
                        "query": " ".join(str(q).split()),
                        "query_type": qtypes[j] if j < len(qtypes) else "n_to_1",
                        "chunk_id": c["chunk_id"],
                        "date": c.get("date"),
                        "case_summary": res.get("case_summary", ""),
                        "relevant_chunk": c["chunk_id"],
                        "method": "n_to_1",
                        "model": args.model,
                    }
                )
        except Exception as e:
            print(f"  failed: {e}")
        time.sleep(args.delay)

    out = {
        "metadata": {
            "method": "n_to_1_unified",
            "model": args.model,
            "queries_per_chunk": args.queries_per_chunk,
            "chunks_processed": len(chunks),
            "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "queries": rows,
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.output} ({len(rows)} queries)")


if __name__ == "__main__":
    main()
