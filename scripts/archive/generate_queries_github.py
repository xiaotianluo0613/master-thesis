#!/usr/bin/env python3
"""
Generate synthetic queries using GitHub Models API (free with GitHub account).
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict
import os
import requests


def get_github_token() -> str:
    """Get GitHub token from environment or gh CLI."""
    # Try environment variable first
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        return token
    
    # Try gh CLI
    try:
        import subprocess
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


def call_github_models(prompt: str, token: str, model: str = "gpt-4o") -> str:
    """
    Call GitHub Models API.
    
    Available models:
    - gpt-4o, gpt-4o-mini (OpenAI)
    - Phi-3-medium-128k-instruct (Microsoft)
    - Mistral-large, Mistral-small (Mistral)
    - Meta-Llama-3.1-405B-Instruct (Meta)
    
    Args:
        prompt: The prompt to send
        token: GitHub authentication token
        model: Model to use
        
    Returns:
        Generated text response
    """
    url = f"https://models.inference.ai.azure.com/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": """You are an expert Swedish historian and archivist researching 19th-century crime in Sweden. Your task is to read a fragmented historical police report and generate exactly THREE realistic user search queries in modern Swedish.

STEP 1: Case Summary (Chain-of-Thought) Before generating queries, briefly summarize the core event, victims, suspects, and stolen items in 1-2 sentences. This will help you ground the queries in facts.

STEP 2: Query Generation Generate THREE distinct search queries that a real historian or genealogist would type into a search engine to find this specific document.

Query 1 (Style A - Thematic/Exploratory): A broad search about the type of crime, stolen objects, or phenomenon during a specific time. (e.g., 'Vilka cykelstölder anmäldes våren 1898?')

Query 2 (Style B - Entity Tracking): A specific search focused on a victim, suspect, or specific address mentioned in the text. (e.g., 'Vilka brott begick Johan Teodor Bäckman?')

Query 3 (Style C - Cross-Reference): A highly specific search combining an entity with an object or location. (e.g., 'I vilken pantbank hittades byxorna som stals från Haga Nygata?')

CRITICAL CONSTRAINTS:
- Modernize language: The text may contain historical Swedish or OCR errors (e.g., 'sterbhus' instead of 'dödsbo', 'f' instead of 'v'). Formulate the queries in standard, modern Swedish.
- Entity-Enforced: Every query MUST contain specific details from the text (names, streets, objects, or dates). Do NOT generate philosophical questions.
- Format: Return ONLY a valid JSON object. No markdown, no introductory text.

EXAMPLE OUTPUT FORMAT:
{
  "case_summary": "Brief summary of the case in 1-2 sentences",
  "queries": [
    "Query 1 text here",
    "Query 2 text here",
    "Query 3 text here"
  ]
}"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": model,
        "temperature": 0.7,
        "max_tokens": 800
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return result['choices'][0]['message']['content'].strip()


def generate_queries_for_chunk(chunk: Dict, token: str, model: str, num_queries: int = 3) -> List[Dict]:
    """Generate multiple queries for a single chunk."""
    date = chunk['date']
    text = chunk['text']  # Use full text with prefix for better context
    
    prompt = f"""Generate the JSON output for this historical record:

{text}"""
    
    try:
        response = call_github_models(prompt, token, model)
        
        # Parse JSON response
        import re
        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*|\s*```', '', response)
        
        result = json.loads(response)
        
        queries = []
        for i, query_text in enumerate(result.get('queries', [])[:3], 1):
            query_type = ['thematic', 'entity_tracking', 'cross_reference'][i-1]
            queries.append({
                'query': query_text,
                'query_type': query_type,
                'chunk_id': chunk['chunk_id'],
                'date': chunk['date'],
                'case_summary': result.get('case_summary', ''),
                'relevant_chunk': chunk['chunk_id']
            })
        
        return queries
        
    except json.JSONDecodeError as e:
        print(f"   ⚠️  Failed to parse JSON response: {e}")
        print(f"   Raw response: {response[:200]}...")
        return []
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []


def generate_queries(
    chunks_file: str,
    output_file: str,
    num_chunks: int = 50,
    queries_per_chunk: int = 3,
    delay: float = 5.0,
    model: str = "gpt-4o-mini",
    resume: bool = True,
    batch_size: int = 10
):
    """Generate synthetic queries from chunks using GitHub Models.
    
    Args:
        chunks_file: Path to input chunks JSON file
        output_file: Path to output queries JSON file
        num_chunks: Number of chunks to process (0 = all)
        queries_per_chunk: Number of queries to generate per chunk
        delay: Delay in seconds between API calls (recommended: 5+ for reliability)
        model: GitHub Models model to use
        resume: If True, skip already processed chunks
        batch_size: Save progress every N chunks
    """
    print(f"🔑 Getting GitHub token...")
    token = get_github_token()
    print("✅ Token obtained")
    print(f"🤖 Using model: {model}")
    print(f"⏱️  Rate limit protection: {delay}s delay between requests")
    
    print(f"\n📖 Loading chunks from {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data['chunks']
    print(f"   Total chunks available: {len(chunks)}")
    
    # Load existing queries if resuming
    existing_chunk_ids = set()
    all_queries = []
    if resume and Path(output_file).exists():
        print(f"\n🔄 Resuming from existing file: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            all_queries = existing_data.get('queries', [])
            existing_chunk_ids = {q['chunk_id'] for q in all_queries}
        print(f"   Already processed: {len(existing_chunk_ids)} chunks")
        print(f"   Existing queries: {len(all_queries)}")
    
    # Select chunks to process
    selected_chunks = []
    for chunk in chunks:
        # Skip already processed chunks
        if chunk['chunk_id'] in existing_chunk_ids:
            continue
        # Only process first sub-chunk or unsplit chunks
        if not chunk['is_split'] or chunk.get('sub_chunk_index') == 0:
            selected_chunks.append(chunk)
        if num_chunks > 0 and len(selected_chunks) >= num_chunks:
            break
    
    print(f"   Selected {len(selected_chunks)} chunks for query generation")
    print(f"   Generating {queries_per_chunk} queries per chunk = ~{len(selected_chunks) * queries_per_chunk} total queries")
    
    if len(selected_chunks) == 0:
        print("\n✅ No new chunks to process!")
        return
    
    # Process chunks with batching
    for i, chunk in enumerate(selected_chunks, 1):
        print(f"\n[{i}/{len(selected_chunks)}] Processing chunk {chunk['chunk_id'][:40]}...")
        
        try:
            queries = generate_queries_for_chunk(chunk, token, model, queries_per_chunk)
            all_queries.extend(queries)
            
            if len(queries) > 0:
                print(f"   ✅ Generated {len(queries)} queries")
                print(f"   📝 Summary: {queries[0].get('case_summary', 'N/A')[:80]}...")
                for q in queries:
                    print(f"      [{q['query_type']}] {q['query'][:80]}...")
            else:
                print(f"   ❌ Failed to generate queries")
            
            # Save progress periodically
            if i % batch_size == 0:
                print(f"\n💾 Saving progress... ({len(all_queries)} queries so far)")
                save_queries(all_queries, output_file, chunks_file, model, queries_per_chunk, i)
            
            # Rate limiting delay
            if i < len(selected_chunks):
                print(f"   ⏳ Waiting {delay}s to avoid rate limits...")
                time.sleep(delay)
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"   ⚠️  Rate limit hit! Waiting 60s before retry...")
                time.sleep(60)
                # Retry once
                try:
                    queries = generate_queries_for_chunk(chunk, token, model, queries_per_chunk)
                    all_queries.extend(queries)
                    print(f"   ✅ Retry successful: {len(queries)} queries")
                except Exception as retry_err:
                    print(f"   ❌ Retry failed: {retry_err}")
            else:
                print(f"   ❌ HTTP Error: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
    
    # Save final results
    print(f"\n💾 Saving final results: {len(all_queries)} queries to {output_file}")
    save_queries(all_queries, output_file, chunks_file, model, queries_per_chunk, len(selected_chunks))
    
    print("\n✅ Query generation complete!")
    print(f"   Total queries: {len(all_queries)}")
    print(f"   Output: {output_file}")


def save_queries(queries: List[Dict], output_file: str, chunks_file: str, 
                 model: str, queries_per_chunk: int, chunks_processed: int):
    """Save queries to output file."""
    output_data = {
        'metadata': {
            'source_file': chunks_file,
            'model': model,
            'chunks_processed': chunks_processed,
            'queries_per_chunk': queries_per_chunk,
            'total_queries': len(queries),
            'generation_date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'queries': queries
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic queries using GitHub Models API (free)',
        epilog='Authenticate with: gh auth login'
    )
    parser.add_argument('chunks_file', help='Input chunks JSON file')
    parser.add_argument('--output', '-o', default='generated_queries.json',
                        help='Output queries JSON file')
    parser.add_argument('--num-chunks', '-n', type=int, default=0,
                        help='Number of chunks to process (0 = all, default: 0)')
    parser.add_argument('--queries-per-chunk', '-q', type=int, default=3,
                        help='Queries per chunk (default: 3)')
    parser.add_argument('--delay', '-d', type=float, default=5.0,
                        help='Delay between API calls in seconds (default: 5.0 for reliability)')
    parser.add_argument('--model', '-m', default='gpt-4o',
                        choices=['gpt-4o', 'gpt-4o-mini', 'Phi-3-medium-128k-instruct', 
                                'Mistral-large', 'Mistral-small', 'Meta-Llama-3.1-405B-Instruct'],
                        help='GitHub Models model to use (default: gpt-4o)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh instead of resuming from existing output')
    parser.add_argument('--batch-size', '-b', type=int, default=10,
                        help='Save progress every N chunks (default: 10)')
    
    args = parser.parse_args()
    
    generate_queries(
        args.chunks_file,
        args.output,
        args.num_chunks,
        args.queries_per_chunk,
        args.delay,
        args.model,
        resume=not args.no_resume,
        batch_size=args.batch_size
    )
