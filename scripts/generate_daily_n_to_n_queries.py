#!/usr/bin/env python3
"""
N-to-N Query Generation: Daily Report Reconstruction

Reconstructs full daily reports from sub-chunks and generates 3 complex queries:
1. Macro/Log Style: Broad question about overall day's events
2-3. Cross-Reference Style: Questions connecting entities from beginning and end of report

Each query maps to ALL sub-chunks from that day.
Uses GitHub Models API with gpt-4o-mini (same as 1-to-1 baseline).
"""

import json
import time
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import requests
import argparse

def get_github_token() -> str:
    """Get GitHub token from environment or gh CLI."""
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        return token
    
    try:
        result = subprocess.run(
            ['gh', 'auth', 'token'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise ValueError(
            "GitHub token not found. Either:\n"
            "1. Set GITHUB_TOKEN environment variable, or\n"
            "2. Run 'gh auth login' to authenticate with GitHub CLI"
        )

def call_github_models(prompt: str, token: str, model: str = "gpt-4o-mini") -> dict:
    """Call GitHub Models API with retry logic (same as baseline)."""
    url = "https://models.inference.ai.azure.com/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": model,
        "temperature": 0.7,
        "max_tokens": 800,
    }
    
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 429:
                wait_time = 60 * (attempt + 1)
                print(f"⏳ Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            result = response.json()
            
            if 'choices' not in result or not result['choices']:
                print(f"⚠️  No choices in response: {result}")
                return None
                
            text = result['choices'][0]['message']['content']
            return {"text": text}
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(5)
    
    return None


def reconstruct_daily_reports(chunks: List[Dict]) -> Dict[str, Dict]:
    """
    Reconstruct full daily reports by grouping sub-chunks by date.
    
    Returns:
        Dict mapping date -> {
            'date': str,
            'full_text': str (concatenated text_without_prefix),
            'chunk_ids': List[str] (all sub-chunk IDs),
            'sub_chunks': List[Dict] (ordered by sub_chunk_index)
        }
    """
    daily_reports = defaultdict(lambda: {
        'date': None,
        'sub_chunks': [],
        'chunk_ids': []
    })
    
    for chunk in chunks:
        date = chunk['date']
        daily_reports[date]['date'] = date
        daily_reports[date]['sub_chunks'].append(chunk)
        daily_reports[date]['chunk_ids'].append(chunk['chunk_id'])
    
    # Sort sub-chunks by index and concatenate text
    for date, report in daily_reports.items():
        # Sort by sub_chunk_index if available (handle None values)
        report['sub_chunks'].sort(key=lambda c: c.get('sub_chunk_index') if c.get('sub_chunk_index') is not None else -1)
        
        # Concatenate text_without_prefix
        full_text = ' '.join([
            c.get('text_without_prefix', c.get('text', ''))
            for c in report['sub_chunks']
        ])
        report['full_text'] = full_text
    
    return dict(daily_reports)


def generate_daily_queries(
    daily_report: Dict,
    github_token: str,
    delay: float = 5.0
) -> Tuple[List[str], str]:
    """
    Generate 3 complex queries for a daily report.
    If report is too long, split into consecutive batches and generate queries for each.
    
    Returns:
        (queries: List[str], case_summary: str)
    """
    date = daily_report['date']
    sub_chunks = daily_report['sub_chunks']
    
    # Batch sub-chunks to fit within token limit
    max_chars = 8000  # ~2000 tokens per batch
    
    batches = []
    current_batch = []
    current_size = 0
    
    for chunk in sub_chunks:
        text = chunk.get('text_without_prefix', chunk.get('text', ''))
        chunk_size = len(text)
        
        # If adding this chunk exceeds limit, start new batch
        if current_size + chunk_size > max_chars and current_batch:
            batches.append(current_batch)
            current_batch = [chunk]
            current_size = chunk_size
        else:
            current_batch.append(chunk)
            current_size += chunk_size
    
    # Add last batch
    if current_batch:
        batches.append(current_batch)
    
    if len(batches) > 1:
        print(f"  ℹ️  Split {date} into {len(batches)} batches ({len(sub_chunks)} total chunks)")
    
    # Generate queries for each batch
    all_queries = []
    for batch_idx, batch in enumerate(batches, 1):
        batch_text = ' '.join([c.get('text_without_prefix', c.get('text', '')) for c in batch])
        
        batch_prefix = f" (batch {batch_idx}/{len(batches)})" if len(batches) > 1 else ""
        queries = generate_queries_for_text(date + batch_prefix, batch_text, github_token)
        
        if queries:
            all_queries.extend(queries)
        
        # Delay between batches
        if batch_idx < len(batches):
            time.sleep(delay)
    
    # Case summary from first batch
    first_text = ' '.join([c.get('text_without_prefix', c.get('text', '')) for c in batches[0]])
    case_summary = first_text[:200] + "..."
    
    return all_queries, case_summary


def generate_queries_for_text(
    date_label: str,
    text: str,
    github_token: str
) -> List[str]:
    """
    Generate 3 queries for a text batch.
    
    Returns:
        List of 3 query strings
    """
    prompt = f"""You are an expert historian specializing in 18th and 19th-century Swedish legal and crime history. I will provide you with a concatenated text containing all police and court records from a single day. 

Your task is to generate exactly 3 complex search queries in modern Swedish that a contemporary historian might use to search for these specific records in a digital archive. 

CRITICAL RULES:
1. The queries MUST be answerable by the provided text, but absolutely MUST NOT contain meta-phrases like "in this text", "according to the document", "in this case", or "during this time period".
2. Query 1 (Macro/Log Style): Generate a broad question summarizing the overall events, crime phenomena, or general police actions of this specific date.
3. Query 2 & 3 (Cross-Reference Style): Generate complex questions that connect multiple specific entities (e.g., victim + criminal + stolen goods + location). The question MUST force the retrieval system to combine details from the *beginning* of the text with details from the *end* of the text.
4. DO NOT generate simple, micro-level questions focusing on only one isolated person or one single detail, as these cannot justify retrieving the entire day's record.
5. You MUST include specific named entities (people, locations, specific stolen items) in Query 2 and 3.

Output ONLY a valid JSON array of 3 strings containing the Swedish queries. Do not output any markdown formatting or explanations.

Text from {date_label}:
{text}"""
    
    print(f"📡 Calling GitHub Models API for {date_label}...")
    response = call_github_models(prompt, github_token)
    
    if not response or 'text' not in response:
        print(f"❌ Failed to get response for {date_label}")
        return []
    
    response_text = response['text'].strip()
    
    # Clean markdown code blocks if present
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        response_text = '\n'.join(lines[1:-1])
    
    try:
        queries = json.loads(response_text)
        if not isinstance(queries, list) or len(queries) != 3:
            print(f"⚠️  Invalid query format for {date_label}: expected 3 queries, got {len(queries) if isinstance(queries, list) else 'non-list'}")
            return []
        
        print(f"✅ Generated 3 queries for {date_label}")
        return queries
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error for {date_label}: {e}")
        print(f"Response: {response_text[:200]}...")
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate N-to-N queries from reconstructed daily reports")
    parser.add_argument("chunks_file", help="Path to chunks JSON file")
    parser.add_argument("--output", default="data/queries_daily_n_to_n.json", help="Output JSON file")
    parser.add_argument("--baseline-queries", default="data/generated_queries_complete.json", 
                       help="Path to 1-to-1 queries JSON to match dates")
    parser.add_argument("--max-days", type=int, default=0, help="Maximum number of days to process (0=all)")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between API calls (seconds)")
    
    args = parser.parse_args()
    
    # Get GitHub token
    try:
        github_token = get_github_token()
        print("✅ GitHub token found")
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Load chunks
    print(f"📂 Loading chunks from {args.chunks_file}...")
    with open(args.chunks_file) as f:
        data = json.load(f)
    
    chunks = data['chunks']
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Load baseline queries to get dates to match
    print(f"📂 Loading baseline queries from {args.baseline_queries}...")
    with open(args.baseline_queries) as f:
        baseline_data = json.load(f)
    
    baseline_queries = baseline_data['queries']
    print(f"✅ Loaded {len(baseline_queries)} baseline queries")
    
    # Get ONLY the chunk IDs that have baseline queries (for fair comparison)
    baseline_chunk_ids = set(q['chunk_id'] for q in baseline_queries)
    print(f"🎯 Found {len(baseline_chunk_ids)} unique chunks with baseline queries")
    
    # Filter chunks to only include baseline chunks
    baseline_chunks = [c for c in chunks if c['chunk_id'] in baseline_chunk_ids]
    print(f"✅ Using {len(baseline_chunks)} chunks (same as 1-to-1 baseline)")
    
    # Reconstruct daily reports from ONLY baseline chunks
    print(f"🔧 Reconstructing daily reports from baseline chunks...")
    daily_reports = reconstruct_daily_reports(baseline_chunks)
    print(f"✅ Using {len(daily_reports)} daily reports (matching baseline)")
    
    # Sort by date
    sorted_dates = sorted(daily_reports.keys())
    
    # Limit to max_days if specified
    if args.max_days > 0:
        sorted_dates = sorted_dates[:args.max_days]
        print(f"🎯 Processing first {args.max_days} of {len(daily_reports)} days")
    else:
        print(f"🎯 Processing all {len(sorted_dates)} days")
    
    # Generate queries
    all_queries = []
    failed_dates = []
    
    for i, date in enumerate(sorted_dates, 1):
        print(f"\n[{i}/{len(sorted_dates)}] Processing {date}...")
        
        daily_report = daily_reports[date]
        queries, case_summary = generate_daily_queries(daily_report, github_token, args.delay)
        
        if not queries:
            failed_dates.append(date)
            continue
        
        # Create query entries - assign type based on position within each set of 3
        for query_idx, query_text in enumerate(queries):
            # Every 3 queries: first is macro_log, rest are cross_reference
            position_in_set = query_idx % 3
            query_type = "macro_log" if position_in_set == 0 else "cross_reference"
            
            query_entry = {
                "query": query_text,
                "query_type": query_type,
                "date": date,
                "case_summary": case_summary,
                "relevant_chunks": daily_report['chunk_ids'],  # ALL sub-chunks from this day
                "num_relevant": len(daily_report['chunk_ids'])
            }
            all_queries.append(query_entry)
    
    # Save results
    output = {
        "metadata": {
            "model": "gpt-4o-mini (GitHub Models)",
            "total_days_processed": len(sorted_dates),
            "successful_days": len(sorted_dates) - len(failed_dates),
            "failed_days": len(failed_dates),
            "total_queries": len(all_queries),
            "queries_per_day": 3,
            "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": args.chunks_file,
            "baseline_queries_file": args.baseline_queries,
            "max_days": args.max_days
        },
        "queries": all_queries
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Statistics:")
    print(f"   • Days processed: {len(sorted_dates)}")
    print(f"   • Successful: {len(sorted_dates) - len(failed_dates)}")
    print(f"   • Failed: {len(failed_dates)}")
    print(f"   • Total queries: {len(all_queries)}")
    print(f"   • Queries per day: 3 (1 macro + 2 cross-reference)")
    if all_queries:
        print(f"   • Average relevant chunks per query: {sum(q['num_relevant'] for q in all_queries) / len(all_queries):.1f}")
    print(f"\n💾 Output saved to: {output_path}")
    
    if failed_dates:
        print(f"\n⚠️  Failed dates: {', '.join(failed_dates[:10])}")
        if len(failed_dates) > 10:
            print(f"   ... and {len(failed_dates) - 10} more")


if __name__ == "__main__":
    main()
