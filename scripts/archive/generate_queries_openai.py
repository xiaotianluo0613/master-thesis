#!/usr/bin/env python3
"""
Generate synthetic queries using OpenAI API directly (paid, more reliable).
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict
import os


BASE_EXPERT_ROLE = (
    "You are an expert Swedish historian and archivist researching 19th-century crime in Sweden."
)


def build_n_to_1_system_prompt() -> str:
    return (
        f"{BASE_EXPERT_ROLE} "
        "Mode: N-to-1. Task: Generate precise queries based SOLELY on this segment. Focus on specific actors and events.\n\n"
        "Your task is to read a fragmented historical police report and generate exactly THREE realistic user search queries in modern Swedish.\n\n"
        "STEP 1: Case Summary (Chain-of-Thought) Before generating queries, briefly summarize the core event, victims, suspects, and stolen items in 1-2 sentences. This will help you ground the queries in facts.\n\n"
        "STEP 2: Query Generation Generate THREE distinct search queries that a real historian or genealogist would type into a search engine to find this specific document.\n\n"
        "Query 1 (Style A - Thematic/Exploratory): A broad search about the type of crime, stolen objects, or phenomenon during a specific time. (e.g., 'Vilka cykelstölder anmäldes våren 1898?')\n\n"
        "Query 2 (Style B - Entity Tracking): A specific search focused on a victim, suspect, or specific address mentioned in the text. (e.g., 'Vilka brott begick Johan Teodor Bäckman?')\n\n"
        "Query 3 (Style C - Cross-Reference): A highly specific search combining an entity with an object or location. (e.g., 'I vilken pantbank hittades byxorna som stals från Haga Nygata?')\n\n"
        "Ensure the 3 queries for the same group approach the text from different angles (e.g., one focusing on the actors, one on the event/crime, and one on the legal/social consequence).\n\n"
        "CRITICAL CONSTRAINTS:\n"
        "- Modernize language: The text may contain historical Swedish or OCR errors (e.g., 'sterbhus' instead of 'dödsbo', 'f' instead of 'v'). Formulate the queries in standard, modern Swedish.\n"
        "- Entity-Enforced: Every query MUST contain specific details from the text (names, streets, objects, or dates). Do NOT generate philosophical questions.\n"
        "- Format: Return ONLY a valid JSON object. No markdown, no introductory text.\n\n"
        "EXAMPLE OUTPUT FORMAT:\n"
        "{\n"
        "  \"case_summary\": \"Brief summary of the case in 1-2 sentences\",\n"
        "  \"queries\": [\n"
        "    \"Query 1 text here\",\n"
        "    \"Query 2 text here\",\n"
        "    \"Query 3 text here\"\n"
        "  ]\n"
        "}"
    )


def get_openai_key() -> str:
    """Get OpenAI API key from environment."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenAI API key not found.\n"
            "Set it with: export OPENAI_API_KEY='your-key-here'\n"
            "Get your key from: https://platform.openai.com/api-keys"
        )
    return api_key


def call_openai(prompt: str, api_key: str, model: str = "gpt-4o") -> str:
    """
    Call OpenAI API directly.
    
    Args:
        prompt: The prompt to send
        api_key: OpenAI API key
        model: Model to use (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
        
    Returns:
        Generated text response
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                                "content": build_n_to_1_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=800
    )
    
    return response.choices[0].message.content.strip()


def generate_queries_for_chunk(chunk: Dict, api_key: str, model: str, num_queries: int = 3) -> List[Dict]:
    """Generate multiple queries for a single chunk."""
    date = chunk['date']
    text = chunk['text']  # Use full text with prefix for better context
    
    prompt = f"""Generate the JSON output for this historical record:

{text}"""
    
    try:
        response = call_openai(prompt, api_key, model)
        
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
    num_chunks: int = 0,
    queries_per_chunk: int = 3,
    delay: float = 0.5,
    model: str = "gpt-4o",
    resume: bool = True,
    batch_size: int = 10
):
    """Generate synthetic queries from chunks using OpenAI API.
    
    Args:
        chunks_file: Path to input chunks JSON file
        output_file: Path to output queries JSON file
        num_chunks: Number of chunks to process (0 = all)
        queries_per_chunk: Number of queries to generate per chunk
        delay: Delay in seconds between API calls (0.5s is safe for paid API)
        model: OpenAI model to use
        resume: If True, skip already processed chunks
        batch_size: Save progress every N chunks
    """
    print(f"🔑 Getting OpenAI API key...")
    api_key = get_openai_key()
    print("✅ API key obtained")
    print(f"🤖 Using model: {model}")
    print(f"⏱️  Delay between requests: {delay}s")
    
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
    
    print(f"   Selected {len(selected_chunks)} new chunks for processing")
    print(f"   Generating {queries_per_chunk} queries per chunk = ~{len(selected_chunks) * queries_per_chunk} new queries")
    
    if len(selected_chunks) == 0:
        print("\n✅ No new chunks to process!")
        return
    
    # Estimate cost
    avg_tokens_per_chunk = 800  # ~500 input + 300 output
    total_tokens = len(selected_chunks) * avg_tokens_per_chunk
    cost_gpt4o = (total_tokens / 1_000_000) * 6.25  # $2.50 input + $10 output average
    cost_gpt4o_mini = (total_tokens / 1_000_000) * 0.30  # $0.15 input + $0.60 output average
    
    print(f"\n💰 Estimated cost:")
    print(f"   GPT-4o: ~${cost_gpt4o:.2f}")
    print(f"   GPT-4o-mini: ~${cost_gpt4o_mini:.2f}")
    print(f"   Estimated time: ~{(len(selected_chunks) * delay) / 60:.1f} minutes")
    
    # Process chunks with batching
    start_time = time.time()
    for i, chunk in enumerate(selected_chunks, 1):
        print(f"\n[{i}/{len(selected_chunks)}] Processing chunk {chunk['chunk_id'][:40]}...")
        
        try:
            queries = generate_queries_for_chunk(chunk, api_key, model, queries_per_chunk)
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
                elapsed = time.time() - start_time
                queries_count = len([q for q in all_queries if q['chunk_id'] not in existing_chunk_ids])
                print(f"\n💾 Saving progress... ({queries_count} new queries, {elapsed/60:.1f} min elapsed)")
                save_queries(all_queries, output_file, chunks_file, model, queries_per_chunk, 
                           len(existing_chunk_ids) + i)
            
            # Rate limiting delay
            if i < len(selected_chunks):
                time.sleep(delay)
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            # Save progress on error
            print(f"   💾 Saving progress due to error...")
            save_queries(all_queries, output_file, chunks_file, model, queries_per_chunk,
                       len(existing_chunk_ids) + i)
    
    # Save final results
    elapsed = time.time() - start_time
    print(f"\n💾 Saving final results: {len(all_queries)} total queries to {output_file}")
    save_queries(all_queries, output_file, chunks_file, model, queries_per_chunk, 
                len(existing_chunk_ids) + len(selected_chunks))
    
    print("\n✅ Query generation complete!")
    print(f"   New queries generated: {len(selected_chunks) * queries_per_chunk}")
    print(f"   Total queries in file: {len(all_queries)}")
    print(f"   Time elapsed: {elapsed/60:.1f} minutes")
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
        description='Generate synthetic queries using OpenAI API (paid)',
        epilog='Set OPENAI_API_KEY environment variable before running'
    )
    parser.add_argument('chunks_file', help='Input chunks JSON file')
    parser.add_argument('--output', '-o', default='generated_queries_openai.json',
                        help='Output queries JSON file')
    parser.add_argument('--num-chunks', '-n', type=int, default=0,
                        help='Number of chunks to process (0 = all, default: 0)')
    parser.add_argument('--queries-per-chunk', '-q', type=int, default=3,
                        help='Queries per chunk (default: 3)')
    parser.add_argument('--delay', '-d', type=float, default=0.5,
                        help='Delay between API calls in seconds (default: 0.5)')
    parser.add_argument('--model', '-m', default='gpt-4o',
                        choices=['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
                        help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh instead of resuming from existing file')
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
        not args.no_resume,
        args.batch_size
    )
