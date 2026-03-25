#!/usr/bin/env python3
"""
Layered N-to-N Query Generation Script
======================================

Generates layered, few-shot N-to-N research queries using LLMs (GPT-4o-mini or Gemini 2.5 Flash) for Swedish social history research.

Features:
- For each daily grouped record (N segments), generates 3 professional Swedish research queries from different angles.
- Prompt structure: system role, task, guidelines, output format, layer-specific bias, and 2 few-shot examples.
- Supports both OpenAI (via Azure/GitHub) and Gemini providers.
- Robust to rate limits, empty responses, and supports resumable output.
- Output is a JSON file with queries, metadata, and failed dates.

Inputs:
- Chunks file: JSON with grouped text segments (e.g., data/layer1_pilot_pairs_550_grouped_3_4.json)
- Few-shot examples: data/n_to_n_fewshot_examples.json
- Layer pools: output/data_pools/train_layer*_pool.txt

Usage Example:
    python scripts/generate_n_to_n_queries_layered.py \
        --provider gemini \
        --model gemini-2.5-flash \
        --chunks data/layer1_pilot_pairs_550_grouped_3_4.json \
        --output data/queries_layer1_n2n_pilot_q3_gemini25flash.json \
        --temperature 1.0 \
        --queries-per-day 3 \
        --disable-baseline-filter \
        --resume \
        --save-every 1

Environment Variables:
- GEMINI_API_KEY (for Gemini)
- GITHUB_TOKEN (for Azure OpenAI)

Output:
- JSON file with generated queries, metadata, and failed dates.

See --help for all options.
"""

import argparse
import json
import os
import random
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


import requests


def get_github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        raise ValueError("GitHub token not found. Set GITHUB_TOKEN or run 'gh auth login'.")


def get_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    raise ValueError("Gemini API key not found. Set GEMINI_API_KEY.")


def call_chat(
    messages: List[Dict],
    token: str,
    model: str = "gpt-4o-mini",
    temperature: float = 1.0,
    max_retries: int = 6,
    rate_limit_base_wait: int = 60,
    rate_limit_max_wait: int = 900,
) -> str:
    url = "https://models.inference.ai.azure.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }

    attempt = 0
    rate_limit_hits = 0
    while True:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 429:
                rate_limit_hits += 1
                retry_after = r.headers.get("Retry-After")
                if retry_after and str(retry_after).isdigit():
                    wait = min(rate_limit_max_wait, int(retry_after))
                else:
                    # Exponential backoff with cap + tiny jitter to avoid sync retries.
                    exp = min(rate_limit_hits - 1, 6)
                    wait = min(rate_limit_max_wait, int(rate_limit_base_wait * (2 ** exp)))
                jitter = random.uniform(0.0, 2.0)
                print(f"⏳ Rate limit hit, waiting {wait:.0f}s...")
                time.sleep(wait + jitter)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                raise
            print(f"⚠️ API retry {attempt}/{max_retries} after error: {e}")
            time.sleep(min(rate_limit_max_wait, 5 * attempt))

    raise RuntimeError("No response from model after retries")


def call_chat_gemini(
    messages: List[Dict],
    api_key: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 1.0,
    max_retries: int = 6,
    rate_limit_base_wait: int = 60,
    rate_limit_max_wait: int = 900,
) -> str:
    # Gemini accepts one concatenated prompt; keep role markers for clarity.
    prompt = "\n\n".join([f"[{m.get('role', 'user').upper()}]\n{m.get('content', '')}" for m in messages])
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 4096,
        },
    }

    attempt = 0
    rate_limit_hits = 0
    while True:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 429:
                rate_limit_hits += 1
                retry_after = r.headers.get("Retry-After")
                if retry_after and str(retry_after).isdigit():
                    wait = min(rate_limit_max_wait, int(retry_after))
                else:
                    exp = min(rate_limit_hits - 1, 6)
                    wait = min(rate_limit_max_wait, int(rate_limit_base_wait * (2 ** exp)))
                jitter = random.uniform(0.0, 2.0)
                print(f"⏳ Rate limit hit, waiting {wait:.0f}s...")
                time.sleep(wait + jitter)
                continue

            r.raise_for_status()
            data = r.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"Gemini empty response: {data}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join([p.get("text", "") for p in parts]).strip()
            if not text:
                raise RuntimeError(f"Gemini empty text response: {data}")
            return text
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                raise
            print(f"⚠️ API retry {attempt}/{max_retries} after error: {e}")
            time.sleep(min(rate_limit_max_wait, 5 * attempt))


def load_pool_ids(pool_file: Path) -> Set[str]:
    if not pool_file.exists():
        return set()
    return {ln.strip() for ln in pool_file.read_text(encoding="utf-8").splitlines() if ln.strip()}


def detect_layer(volume_id: str, layer_pools: Dict[str, Set[str]]) -> str:
    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        if volume_id in layer_pools[layer]:
            return layer
    return "layer1"  # safe fallback


def reconstruct_daily_reports(chunks: List[Dict]) -> Dict[str, Dict]:
    grouped = defaultdict(lambda: {"date": None, "chunk_ids": [], "sub_chunks": [], "volume_id": None})
    for c in chunks:
        date = c.get("date") or (f"VOLUME-{c.get('volume_id')}" if c.get("volume_id") else "unknown-date")
        grouped[date]["date"] = date
        grouped[date]["chunk_ids"].append(c["chunk_id"])
        grouped[date]["sub_chunks"].append(c)
        if not grouped[date]["volume_id"]:
            grouped[date]["volume_id"] = c.get("volume_id")

    def _page_order_key(sub_chunk: Dict, fallback_pos: int) -> int:
        # 1) explicit sub-chunk index from upstream split pipeline
        sci = sub_chunk.get("sub_chunk_index")
        if sci is not None:
            try:
                return int(sci)
            except Exception:
                pass

        # 2) parse trailing numeric page id from source XML filename, e.g. *_00037.xml
        sf = str(sub_chunk.get("source_file", ""))
        if sf:
            stem = Path(sf).stem
            m = re.search(r"(\d+)$", stem)
            if m:
                return int(m.group(1))

        # 3) parse trailing numeric page id from chunk_id as fallback
        cid = str(sub_chunk.get("chunk_id", ""))
        m = re.search(r"(\d+)$", cid)
        if m:
            return int(m.group(1))

        # 4) preserve original input order if no numeric cue exists
        return fallback_pos

    for d, g in grouped.items():
        indexed = list(enumerate(g["sub_chunks"]))
        indexed.sort(key=lambda it: _page_order_key(it[1], it[0]))
        g["sub_chunks"] = [x for _, x in indexed]
        segments = [s.get("text_without_prefix", s.get("text", "")) for s in g["sub_chunks"]]
        g["segments"] = [s for s in segments if s]
    return dict(grouped)


def parse_queries(response: str) -> List[Dict]:
    """Parse labeled query output into list of {query, query_type} dicts."""
    type_map = {1: "entity", 2: "entity", 3: "social_pattern"}
    results = []
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        for idx in [1, 2, 3]:
            prefix = f"Query {idx}:"
            if line.startswith(prefix):
                text = line[len(prefix):].strip()
                if text:
                    results.append({"query": text, "query_type": type_map[idx], "query_index": idx})
                break
    return results


def build_prompt_messages(
    segments: List[str],
    layer_key: str,
    fewshot: Dict,
) -> List[Dict]:
    layer_info = fewshot[layer_key]
    layer_bias = layer_info.get("bias", "")
    raw_examples = layer_info.get("examples", [])
    entity_ex = [e["query"] for e in raw_examples if e.get("type") == "entity"]
    social_ex = [e["query"] for e in raw_examples if e.get("type") == "social_pattern"]

    # Keep manageable input length
    joined = "\n\n".join([f"[Segment {i+1}]\n{txt[:1800]}" for i, txt in enumerate(segments[:8])])

    system_prompt = (
        "# Role\n"
        "You are a senior historian in Swedish social history, specializing in 16th-19th century Swedish legal documents like court records, police reports, and protocols. "
        "You understand that historians search archives in two distinct ways:\n"
        "- Academic historians search for social patterns, events, and group behaviors\n"
        "- Genealogists search for specific named individuals, places, and dates\n\n"
        "Your task is to generate queries that simulate both search behaviors."
    )

    user_prompt = (
        "# Task\n"
        "Read these document segments and generate 3 search queries in SWEDISH:\n"
        "- 2 entity-type queries (genealogist search behavior)\n"
        "- 1 social pattern query (academic historian search behavior)\n\n"
        "# Guidelines for Entity Queries (2 queries)\n"
        "1. Ask about the existence or general record of a person, place, or object across the archive — NOT about specific details within a single document.\n"
        "   Write as a researcher who does not yet know which document contains the answer.\n"
        "2. Vary the entity across queries — do not always pick the most prominent person.\n"
        "   Sometimes pick a witness, a secondary actor, a location, or a specific object.\n"
        "3. Include the person's role or location alongside their name (e.g., not just 'Andersson' but 'snickare Andersson i Majorna').\n"
        "4. Vary the question form — do NOT always start with 'Finns det'. Use different forms across queries:\n"
        "   - Existence: 'Finns det uppgifter om...'\n"
        "   - Descriptive: 'Vad är känt om...'\n"
        "   - Relational: 'Vilka personer omnämns i samband med...'\n"
        "   - Record-seeking: 'Förekommer [name/place] i arkivet?'\n"
        "5. AVOID: specific event details, phrases implying a single document ('i detta mål', 'i detta protokoll', 'under denna period').\n"
        "   GOOD: 'Vad är känt om snickare Andersson i Majorna?'\n"
        "   BAD: 'Vad berättade C. G. Westerberg om stölden på Salutorget?'\n\n"
        "# Guidelines for Social Pattern Query (1 query)\n"
        "1. Write a short, open-ended question about a recurring social situation, crime type, or group behavior.\n"
        "2. Write as a historian who does not yet know the answer — curious, exploratory.\n"
        "3. Keep it general: most social pattern queries should NOT reference specific streets or names from the source.\n"
        "   Only occasionally (1 in 3) may you include a specific location from the text.\n\n"
        "# Output Format\n"
        "Output exactly three lines in SWEDISH. Each line starts with the label, then the query.\n"
        "Each query MUST be a full natural language question — NOT a keyword phrase.\n"
        "Keep each query concise: one clear question, similar length to the few-shot examples.\n"
        "Query 1: <your entity query here>\n"
        "Query 2: <your entity query here>\n"
        "Query 3: <your social pattern query here>\n\n"
        f"# Layer Specific Emphasis ({layer_key} / {layer_info.get('label','')})\n"
        f"{layer_bias}\n\n"
        "# Few-shot Examples\n"
        "Entity examples:\n"
        + "".join([f"- {q}\n" for q in entity_ex]) +
        "Social pattern examples:\n"
        + "".join([f"- {q}\n" for q in social_ex]) + "\n"
        "# Input Segments\n"
        f"{joined}\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def write_output(
    out_path: Path,
    generated: List[Dict],
    failed: List[str],
    args,
    total_days_processed: int,
) -> None:
    out = {
        "metadata": {
            "method": "layered-baseprompt-fewshot-n2n",
            "provider": args.provider,
            "model": args.model,
            "temperature": args.temperature,
            "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dry_run": args.dry_run,
            "baseline_filter_disabled": args.disable_baseline_filter,
            "total_days_processed": total_days_processed,
            "successful_days": len(generated) // 3,
            "total_queries": len(generated),
            "failed_days": len(failed),
            "fewshot_file": args.fewshot,
            "pool_dir": args.pool_dir,
            "queries_per_call": 3,
            "query_types": ["entity", "entity", "social_pattern"],
        },
        "queries": generated,
        "failed_dates": failed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate improved layered N-to-N queries")
    parser.add_argument("--chunks", default="data/30002051_chunks_split_prefixed.json")
    parser.add_argument("--baseline-queries", default="data/generated_queries_complete.json")
    parser.add_argument("--fewshot", default="data/n_to_n_fewshot_examples.json")
    parser.add_argument("--pool-dir", default="output/data_pools")
    parser.add_argument("--output", default="data/queries_daily_n_to_n_layered.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--provider", choices=["github", "gemini"], default="github",
                        help="LLM provider backend.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--delay", type=float, default=4.0)
    parser.add_argument("--max-retries", type=int, default=6,
                        help="Max retries for non-429 request errors.")
    parser.add_argument("--rate-limit-base-wait", type=int, default=60,
                        help="Initial backoff seconds for 429 responses.")
    parser.add_argument("--rate-limit-max-wait", type=int, default=900,
                        help="Max backoff seconds for 429 responses.")
    parser.add_argument("--queries-per-day", type=int, default=1,
                        help="How many distinct queries to generate per grouped day/volume.")
    parser.add_argument("--disable-baseline-filter", action="store_true",
                        help="Do not restrict chunks to baseline chunk_ids (useful for small prompt tests).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts and output records without calling the model API.")
    parser.add_argument("--prompt-output", default="data/layered_prompt_test_prompts.json",
                        help="Where to save built prompts in dry-run mode.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing --output file if it exists.")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Write checkpoint to --output every N groups (default: 1).")
    args = parser.parse_args()

    token = None
    gemini_key = None
    if not args.dry_run:
        if args.provider == "github":
            token = get_github_token()
        else:
            gemini_key = get_gemini_key()

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]
    baseline = None
    if not args.disable_baseline_filter:
        with open(args.baseline_queries, "r", encoding="utf-8") as f:
            baseline = json.load(f)["queries"]
    with open(args.fewshot, "r", encoding="utf-8") as f:
        fewshot = json.load(f)

    # Restrict to baseline chunk IDs for fairness (optional)
    if baseline is not None:
        baseline_ids = {q["chunk_id"] for q in baseline}
        chunks = [c for c in chunks if c["chunk_id"] in baseline_ids]

    reports = reconstruct_daily_reports(chunks)
    dates = sorted(reports.keys())
    if args.max_days > 0:
        dates = dates[: args.max_days]

    # Layer pools
    pool_dir = Path(args.pool_dir)
    layer_pools = {
        "layer1": load_pool_ids(pool_dir / "train_layer1_pool.txt"),
        "layer2": load_pool_ids(pool_dir / "train_layer2_pool.txt"),
        "layer3": load_pool_ids(pool_dir / "train_layer3_pool.txt"),
        "layer4": load_pool_ids(pool_dir / "train_layer4_pool.txt"),
    }

    out_path = Path(args.output)
    generated: List[Dict] = []
    failed: List[str] = []
    prompts_dump = []

    done_dates = set()
    if args.resume and out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            generated = prev.get("queries", [])
            failed = prev.get("failed_dates", [])
            for q in generated:
                done_dates.add(q.get("date"))
            print(f"🔁 Resuming from {out_path}: existing queries={len(generated)} failed_dates={len(failed)}")
        except Exception as e:
            print(f"⚠️ Could not load resume file {out_path}: {e}")

    for i, d in enumerate(dates, 1):
        rep = reports[d]
        volume_id = rep.get("volume_id")
        layer = detect_layer(volume_id, layer_pools) if volume_id else "layer1"

        print(f"[{i}/{len(dates)}] {d} volume={volume_id} layer={layer}")

        if d in done_dates:
            continue

        try:
            msgs = build_prompt_messages(rep["segments"], layer, fewshot)
            if args.dry_run:
                prompts_dump.append({
                    "date": d,
                    "volume_id": volume_id,
                    "layer": layer,
                    "messages": msgs,
                    "num_segments": len(rep["segments"]),
                    "num_relevant": len(rep["chunk_ids"]),
                })
                parsed = [
                    {"query": f"[DRY-RUN] {d} ({layer}) entity 1", "query_type": "entity", "query_index": 1},
                    {"query": f"[DRY-RUN] {d} ({layer}) entity 2", "query_type": "entity", "query_index": 2},
                    {"query": f"[DRY-RUN] {d} ({layer}) social_pattern", "query_type": "social_pattern", "query_index": 3},
                ]
            else:
                if args.provider == "github":
                    raw = call_chat(
                        msgs, token=token, model=args.model, temperature=args.temperature,
                        max_retries=max(1, args.max_retries),
                        rate_limit_base_wait=max(1, args.rate_limit_base_wait),
                        rate_limit_max_wait=max(1, args.rate_limit_max_wait),
                    )
                else:
                    raw = call_chat_gemini(
                        msgs, api_key=gemini_key, model=args.model, temperature=args.temperature,
                        max_retries=max(1, args.max_retries),
                        rate_limit_base_wait=max(1, args.rate_limit_base_wait),
                        rate_limit_max_wait=max(1, args.rate_limit_max_wait),
                    )
                parsed = parse_queries(raw)
                if not parsed:
                    raise ValueError(f"Could not parse any queries from response: {raw[:200]}")

            case_summary = rep["segments"][0][:220] + "..." if rep["segments"] else ""
            for item in parsed:
                generated.append({
                    "query": item["query"],
                    "query_type": item["query_type"],
                    "query_index": item["query_index"],
                    "layer": layer,
                    "date": d,
                    "volume_id": volume_id,
                    "case_summary": case_summary,
                    "relevant_chunks": rep["chunk_ids"],
                    "num_relevant": len(rep["chunk_ids"]),
                })
            done_dates.add(d)

            if not args.dry_run:
                time.sleep(args.delay)

        except Exception as e:
            print(f"❌ failed {d}: {e}")

        if d not in done_dates and d not in failed:
            failed.append(d)

        if args.save_every > 0 and (i % args.save_every == 0):
            write_output(out_path, generated, failed, args, total_days_processed=len(dates))

    write_output(out_path, generated, failed, args, total_days_processed=len(dates))

    if args.dry_run:
        prompt_out = Path(args.prompt_output)
        prompt_out.parent.mkdir(parents=True, exist_ok=True)
        prompt_out.write_text(json.dumps({"prompts": prompts_dump}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("✅ Layered N-to-N generation complete")
    print(f"Output: {out_path}")
    if args.dry_run:
        print(f"Prompt dump: {args.prompt_output}")
    print(f"Generated: {len(generated)} | Failed: {len(failed)}")


if __name__ == "__main__":
    main()
