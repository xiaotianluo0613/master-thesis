#!/usr/bin/env python3
"""
Generate synthetic queries from police report chunks using OpenAI API.
Works with OpenAI, Azure OpenAI, or any OpenAI-compatible API.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict
import os

# Try importing OpenAI library
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  OpenAI library not installed. Install with: pip install openai")


def get_api_key() -> str:
    """
    Get API key from environment variable.
    
    Returns:
        OpenAI API key
    """
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError(
            "OpenAI API key not found.\n"
            "Set OPENAI_API_KEY environment variable:\n"
            "  export OPENAI_API_KEY='your-api-key-here'\n\n"
            "Or get a key from: https://platform.openai.com/api-keys"
        )
    return key


def call_openai_api(prompt: str, client: OpenAI, model: str = "gpt-4o-mini") -> str:
    """
    Call OpenAI API to generate text.
    
    Args:
        prompt: The prompt to send
        client: OpenAI client instance
        model: Model to use (gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc.)
        
    Returns:
        Generated text response
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Du är en expert på svensk historia och historiska dokument. Du hjälper till att generera realistiska sökfrågor på svenska för historiska polisrapporter från 1800-talet."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()


def generate_queries_for_chunk(chunk: Dict, client: OpenAI, model: str, num_queries: int = 3) -> List[Dict]:
    """
    Generate multiple queries for a single chunk.
    
    Args:
        chunk: Chunk dictionary with text and metadata
        client: OpenAI client instance
        model: Model name to use
        num_queries: Number of queries to generate per chunk
        
    Returns:
        List of query dictionaries
    """
    # Extract key information
    date = chunk['date']
    text_sample = chunk['text_without_prefix'][:500]  # First 500 chars
    
    prompt = f"""Baserat på följande historiska polisrapport från Göteborg {date}, generera {num_queries} olika sökfrågor som en forskare skulle kunna ställa.

Rapport (utdrag):
{text_sample}

Generera {num_queries} olika typer av frågor:
1. En faktabaserad fråga (vem, vad, var)
2. En temporal fråga (när, hur länge)
3. En analytisk fråga (varför, hur)

Format: Skriv varje fråga på en ny rad, numrerad 1., 2., 3.
Frågorna ska vara på svenska och naturliga för en historiker eller forskare.
"""
    
    try:
        response = call_openai_api(prompt, client, model)
        
        # Parse response into individual queries
        queries = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove numbering (1., 2., etc.)
            if line and (line[0].isdigit() or line.startswith('-')):
                query_text = line.split('.', 1)[-1].strip() if '.' in line else line[2:].strip()
                if query_text:
                    queries.append({
                        'query': query_text,
                        'chunk_id': chunk['chunk_id'],
                        'date': chunk['date'],
                        'relevant_chunk': chunk['chunk_id']
                    })
        
        return queries[:num_queries]  # Ensure we don't exceed requested number
        
    except Exception as e:
        print(f"Error generating queries for chunk {chunk['chunk_id']}: {e}")
        return []


def generate_queries(
    chunks_file: str,
    output_file: str,
    num_chunks: int = 50,
    queries_per_chunk: int = 3,
    delay: float = 1.0,
    model: str = "gpt-4o-mini"
):
    """
    Generate synthetic queries from chunks.
    
    Args:
        chunks_file: Path to chunks JSON file
        output_file: Path to output queries JSON
        num_chunks: Number of chunks to process
        queries_per_chunk: Number of queries per chunk
        delay: Delay between API calls (seconds)
        model: OpenAI model to use
    """
    if not HAS_OPENAI:
        print("❌ OpenAI library not installed.")
        print("Install with: pip install openai")
        sys.exit(1)
    
    print(f"🔑 Getting API key...")
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)
    print("✅ API key obtained")
    print(f"🤖 Using model: {model}")
    
    print(f"\n📖 Loading chunks from {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data['chunks']
    print(f"   Total chunks available: {len(chunks)}")
    
    # Sample diverse chunks (prefer unsplit or first sub-chunks)
    selected_chunks = []
    for chunk in chunks:
        if not chunk['is_split'] or chunk.get('sub_chunk_index') == 0:
            selected_chunks.append(chunk)
        if len(selected_chunks) >= num_chunks:
            break
    
    print(f"   Selected {len(selected_chunks)} chunks for query generation")
    print(f"   Generating {queries_per_chunk} queries per chunk = ~{len(selected_chunks) * queries_per_chunk} total queries")
    
    all_queries = []
    
    for i, chunk in enumerate(selected_chunks, 1):
        print(f"\n[{i}/{len(selected_chunks)}] Processing chunk {chunk['chunk_id'][:30]}...")
        
        queries = generate_queries_for_chunk(chunk, client, model, queries_per_chunk)
        all_queries.extend(queries)
        
        print(f"   Generated {len(queries)} queries")
        for q in queries:
            print(f"      - {q['query'][:80]}...")
        
        # Rate limiting
        if i < len(selected_chunks):
            time.sleep(delay)
    
    # Save results
    output_data = {
        'metadata': {
            'source_file': chunks_file,
            'model': model,
            'chunks_processed': len(selected_chunks),
            'queries_per_chunk': queries_per_chunk,
            'total_queries': len(all_queries),
            'generation_date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'queries': all_queries
    }
    
    print(f"\n💾 Saving {len(all_queries)} queries to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Query generation complete!")
    print(f"   Total queries: {len(all_queries)}")
    print(f"   Output: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic queries from chunks using OpenAI API',
        epilog='Set OPENAI_API_KEY environment variable before running.'
    )
    parser.add_argument('chunks_file', help='Input chunks JSON file')
    parser.add_argument('--output', '-o', default='generated_queries.json',
                        help='Output queries JSON file')
    parser.add_argument('--num-chunks', '-n', type=int, default=50,
                        help='Number of chunks to process (default: 50)')
    parser.add_argument('--queries-per-chunk', '-q', type=int, default=3,
                        help='Queries per chunk (default: 3)')
    parser.add_argument('--delay', '-d', type=float, default=1.0,
                        help='Delay between API calls in seconds (default: 1.0)')
    parser.add_argument('--model', '-m', default='gpt-4o-mini',
                        help='OpenAI model to use (default: gpt-4o-mini)')
    
    args = parser.parse_args()
    
    generate_queries(
        args.chunks_file,
        args.output,
        args.num_chunks,
        args.queries_per_chunk,
        args.delay,
        args.model
    )
