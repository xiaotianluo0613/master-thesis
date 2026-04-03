#!/usr/bin/env python3
"""
Analyze the JSONL dataset to understand document relationships and structure.

Helps determine if documents are continuous pages of a larger text
or independent records.
"""

import json
from pathlib import Path
from collections import defaultdict


def analyze_dataset(jsonl_path: Path) -> None:
    """
    Analyze the JSONL dataset to understand its structure.
    
    Args:
        jsonl_path: Path to input JSONL file
    """
    documents = []
    
    # Read all documents
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                documents.append(entry)
            except json.JSONDecodeError:
                continue
    
    print("\n" + "=" * 80)
    print("DATASET ANALYSIS")
    print("=" * 80)
    
    # Basic statistics
    print(f"\nTotal documents: {len(documents)}")
    
    # Text statistics
    text_lengths = [len(doc.get('text', '')) for doc in documents]
    non_empty = sum(1 for length in text_lengths if length > 0)
    
    print(f"Non-empty documents: {non_empty}/{len(documents)}")
    print(f"Empty documents: {len(documents) - non_empty}")
    
    if text_lengths:
        print(f"\nText length statistics:")
        print(f"  Total characters: {sum(text_lengths):,}")
        print(f"  Average per document: {sum(text_lengths) // non_empty if non_empty > 0 else 0:,}")
        print(f"  Min: {min(l for l in text_lengths if l > 0) if non_empty > 0 else 0}")
        print(f"  Max: {max(text_lengths)}")
    
    # Metadata analysis
    print(f"\nVolumes found:")
    volumes = defaultdict(int)
    for doc in documents:
        volume = doc.get('metadata', {}).get('volume', 'Unknown')
        volumes[volume] += 1
    
    for volume, count in sorted(volumes.items()):
        print(f"  {volume}: {count} documents")
    
    # Document continuity analysis
    print(f"\n" + "-" * 80)
    print("DOCUMENT CONTINUITY ANALYSIS")
    print("-" * 80)
    print("\nFirst 10 non-empty documents:")
    
    count = 0
    for i, doc in enumerate(documents):
        if doc.get('text', '').strip():
            count += 1
            text = doc['text']
            # Show first and last 100 chars to identify continuity
            first_part = text[:100].replace('\n', ' ')[:80]
            last_part = text[-100:].replace('\n', ' ')[-80:]
            
            print(f"\n[{count}] ID: {doc['id']}")
            print(f"    First: {first_part}...")
            if len(text) > 200:
                print(f"    Last:  ...{last_part}")
            
            if count >= 10:
                break
    
    # Look for numbered sections (reports)
    print(f"\n" + "-" * 80)
    print("DOCUMENT TYPES")
    print("-" * 80)
    
    report_pattern = set()
    for doc in documents:
        text = doc.get('text', '').strip()
        # Check for "Rapport" or numbered reports
        if 'Rapport' in text or 'No ' in text[:200]:
            report_pattern.add(doc['id'])
    
    title_pattern = set()
    for doc in documents:
        text = doc.get('text', '').strip()
        # Check if looks like a title page
        if len(text) < 200 and 'Polisens' in text:
            title_pattern.add(doc['id'])
    
    print(f"\nLikely title/cover pages: {len(title_pattern)} documents")
    if title_pattern:
        print(f"  Examples: {list(title_pattern)[:5]}")
    
    print(f"\nLikely report documents: {len(report_pattern)} documents")
    if report_pattern:
        print(f"  Examples: {list(report_pattern)[:5]}")
    
    # Check for continuity markers
    print(f"\n" + "-" * 80)
    print("POTENTIAL CONTINUITY ISSUES")
    print("-" * 80)
    
    continuation_risk = []
    for i, doc in enumerate(documents[:-1]):
        text = doc.get('text', '').strip()
        next_text = documents[i+1].get('text', '').strip()
        
        if not text or not next_text:
            continue
        
        # Check if current doc ends mid-sentence (no punctuation)
        last_char = text[-1] if text else ''
        first_word_next = next_text.split()[0] if next_text.split() else ''
        
        # Look for incomplete sentences
        if last_char not in '.!?:;,' and first_word_next and first_word_next[0].islower():
            continuation_risk.append((doc['id'], documents[i+1]['id']))
    
    if continuation_risk:
        print(f"\nPotential mid-sentence breaks (first 5):")
        for curr_id, next_id in continuation_risk[:5]:
            print(f"  {curr_id} -> {next_id}")
    else:
        print("\nNo obvious mid-sentence breaks detected.")
        print("Documents likely represent independent records/pages.")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("""
Each XML file represents an OCR of a single document/page image.
The documents appear to be from a police report archive from 1868 (Gothenburg).
They can be:
  1. Individual report pages (often multi-page reports)
  2. Title/cover pages
  3. Continuation pages of reports from previous images

To determine which pages belong together:
  - Look for "Rapport No X" markers
  - Check for numbered reports (No 1, No 2, etc.)
  - Examine text continuity between sequential documents
  - Look at document IDs for patterns
    """)
    print("=" * 80 + "\n")


def main():
    """Main function."""
    workspace_root = Path(__file__).parent
    jsonl_file = workspace_root / "output.jsonl"
    
    if not jsonl_file.exists():
        print(f"Error: {jsonl_file} not found")
        return
    
    analyze_dataset(jsonl_file)


if __name__ == "__main__":
    main()
