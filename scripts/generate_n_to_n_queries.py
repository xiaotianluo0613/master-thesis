#!/usr/bin/env python3
"""
N-to-N Query Generation for Multi-Document Retrieval Evaluation

Generates queries that can match multiple relevant chunks:
- Temporal queries: "What happened in Gothenburg in January 1898?" → multiple reports from that period
- Thematic queries: "What fraud cases were reported in 1898?" → multiple fraud reports
- Entity queries: "What crimes involved stolen watches?" → multiple watch theft cases

This enables evaluation of Precision, Recall, MAP, and nDCG.
"""

import json
import time
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import requests

def call_gemini(prompt: str, api_key: str, model: str = "gemini-2.0-flash-exp") -> dict:
    """Call Gemini API with retry logic."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 8192,
        },
        "safetySettings": [
            {"category": cat, "threshold": "BLOCK_NONE"}
            for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                       "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
    }
    
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 429:
                wait_time = 60 * (attempt + 1)
                print(f"Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            result = response.json()
            
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Extract JSON from markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            return json.loads(text)
            
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(5)
    
    raise Exception("Failed after 3 attempts")


def group_chunks_by_criteria(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """Group chunks by various criteria for N-to-N query generation."""
    
    groups = {
        'temporal': defaultdict(list),  # By month
        'thematic': defaultdict(list),  # By keywords
        'entities': defaultdict(list),  # By named entities
    }
    
    # Crime type keywords
    crime_keywords = {
        'bedrägeri': 'fraud',
        'stöld': 'theft',
        'inbrott': 'burglary',
        'misshandel': 'assault',
        'fylleri': 'drunkenness',
        'lösdriveri': 'vagrancy'
    }
    
    # Common entities
    entity_keywords = {
        'ur': 'watch',
        'cykel': 'bicycle',
        'pengar': 'money',
        'klocka': 'clock',
        'portmonnä': 'wallet'
    }
    
    for chunk in chunks:
        date = chunk.get('date', '')
        text = chunk.get('text', '').lower()
        
        # Group by month (YYYY-MM)
        if len(date) >= 7:
            month_key = date[:7]  # e.g., "1898-Jan"
            groups['temporal'][month_key].append(chunk)
        
        # Group by crime type
        for keyword, crime_type in crime_keywords.items():
            if keyword in text:
                groups['thematic'][crime_type].append(chunk)
        
        # Group by entities
        for keyword, entity in entity_keywords.items():
            if keyword in text:
                groups['entities'][entity].append(chunk)
    
    return groups


def generate_n_to_n_queries(groups: Dict, api_key: str, max_queries: int = 50) -> List[Dict]:
    """Generate N-to-N queries where each query matches multiple chunks."""
    
    queries = []
    query_count = 0
    
    print(f"\n{'='*80}")
    print("GENERATING N-TO-N QUERIES")
    print(f"{'='*80}\n")
    
    # Temporal queries (month-based)
    print("Generating temporal queries (N-to-1)...")
    for month, chunks in list(groups['temporal'].items())[:15]:
        if len(chunks) < 2:  # Skip if only 1 chunk
            continue
        
        if query_count >= max_queries:
            break
        
        chunk_ids = [c['chunk_id'] for c in chunks]
        sample_texts = [c['text'][:300] for c in chunks[:3]]  # Sample from first 3
        
        prompt = f"""Du är en expert på svenska polisrapporter från 1800-talet.

Baserat på dessa rapporter från {month}:

{chr(10).join(f"Rapport {i+1}: {text}..." for i, text in enumerate(sample_texts))}

Generera EN temporal fråga som skulle matcha FLERA av dessa rapporter från samma tidsperiod.

Frågan ska:
- Fråga om händelser under denna månad/period
- Vara tillräckligt bred för att matcha flera rapporter
- Vara realistisk för en forskare

Svara i JSON-format:
{{
  "query": "Din fråga här",
  "query_type": "temporal",
  "date_range": "{month}",
  "explanation": "Varför denna fråga matchar flera rapporter"
}}"""
        
        try:
            result = call_gemini(prompt, api_key)
            result['relevant_chunks'] = chunk_ids
            result['num_relevant'] = len(chunk_ids)
            queries.append(result)
            query_count += 1
            
            print(f"  Generated temporal query: {len(chunk_ids)} relevant chunks")
            time.sleep(5)
            
        except Exception as e:
            print(f"  Error generating temporal query: {e}")
            continue
    
    # Thematic queries (crime type)
    print("\nGenerating thematic queries (N-to-1)...")
    for crime_type, chunks in list(groups['thematic'].items())[:10]:
        if len(chunks) < 2:
            continue
        
        if query_count >= max_queries:
            break
        
        chunk_ids = [c['chunk_id'] for c in chunks]
        sample_texts = [c['text'][:300] for c in chunks[:3]]
        
        prompt = f"""Du är en expert på svenska polisrapporter från 1800-talet.

Baserat på dessa {crime_type}-rapporter:

{chr(10).join(f"Rapport {i+1}: {text}..." for i, text in enumerate(sample_texts))}

Generera EN tematisk fråga som skulle matcha FLERA av dessa {crime_type}-rapporter.

Frågan ska:
- Fråga om {crime_type}-brott generellt
- Vara tillräckligt bred för att matcha flera rapporter
- Vara realistisk för en forskare

Svara i JSON-format:
{{
  "query": "Din fråga här",
  "query_type": "thematic",
  "theme": "{crime_type}",
  "explanation": "Varför denna fråga matchar flera rapporter"
}}"""
        
        try:
            result = call_gemini(prompt, api_key)
            result['relevant_chunks'] = chunk_ids
            result['num_relevant'] = len(chunk_ids)
            queries.append(result)
            query_count += 1
            
            print(f"  Generated thematic query for {crime_type}: {len(chunk_ids)} relevant chunks")
            time.sleep(5)
            
        except Exception as e:
            print(f"  Error generating thematic query: {e}")
            continue
    
    # Entity queries
    print("\nGenerating entity-based queries (N-to-1)...")
    for entity, chunks in list(groups['entities'].items())[:10]:
        if len(chunks) < 2:
            continue
        
        if query_count >= max_queries:
            break
        
        chunk_ids = [c['chunk_id'] for c in chunks]
        sample_texts = [c['text'][:300] for c in chunks[:3]]
        
        prompt = f"""Du är en expert på svenska polisrapporter från 1800-talet.

Baserat på dessa rapporter som involverar {entity}:

{chr(10).join(f"Rapport {i+1}: {text}..." for i, text in enumerate(sample_texts))}

Generera EN fråga som skulle matcha FLERA av dessa rapporter om {entity}.

Frågan ska:
- Fråga om {entity} i brottssammanhang
- Vara tillräckligt bred för att matcha flera rapporter
- Vara realistisk för en forskare

Svara i JSON-format:
{{
  "query": "Din fråga här",
  "query_type": "entity",
  "entity": "{entity}",
  "explanation": "Varför denna fråga matchar flera rapporter"
}}"""
        
        try:
            result = call_gemini(prompt, api_key)
            result['relevant_chunks'] = chunk_ids
            result['num_relevant'] = len(chunk_ids)
            queries.append(result)
            query_count += 1
            
            print(f"  Generated entity query for {entity}: {len(chunk_ids)} relevant chunks")
            time.sleep(5)
            
        except Exception as e:
            print(f"  Error generating entity query: {e}")
            continue
    
    return queries


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Generate N-to-N queries")
    parser.add_argument('chunks_file', type=str, help='Path to chunks JSON')
    parser.add_argument('--output', type=str, default='data/queries_n_to_n.json',
                       help='Output file for queries')
    parser.add_argument('--max-queries', type=int, default=50,
                       help='Maximum number of queries to generate')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        return
    
    # Load chunks
    print(f"Loading chunks from {args.chunks_file}...")
    with open(args.chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
        chunks = chunks_data['chunks']
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Group chunks
    print("\nGrouping chunks by criteria...")
    groups = group_chunks_by_criteria(chunks)
    
    print(f"  Temporal groups: {len(groups['temporal'])}")
    print(f"  Thematic groups: {len(groups['thematic'])}")
    print(f"  Entity groups: {len(groups['entities'])}")
    
    # Generate queries
    queries = generate_n_to_n_queries(groups, api_key, max_queries=args.max_queries)
    
    # Save results
    output_data = {
        'metadata': {
            'source_file': args.chunks_file,
            'total_chunks': len(chunks),
            'total_queries': len(queries),
            'query_types': list(set(q.get('query_type') for q in queries)),
            'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': 'gemini-2.0-flash-exp'
        },
        'queries': queries
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Generated {len(queries)} N-to-N queries")
    print(f"   Saved to: {output_path}")
    print(f"   Average relevant chunks per query: {sum(q['num_relevant'] for q in queries) / len(queries):.1f}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
