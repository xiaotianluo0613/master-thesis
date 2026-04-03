#!/usr/bin/env python3
"""
Split large chunks into smaller sub-chunks using sliding window with overlap.

This script takes chunks from process_police_volume.py and splits any chunk
exceeding a token limit into smaller sub-chunks with overlapping text.

Metadata handling:
- Parent chunk metadata (date, volume_id, source_xmls) preserved
- New field: sub_chunk_index (e.g., 0, 1, 2 for parts of same parent)
- New field: is_split (True if chunk was split)
- Updated chunk_id: "vol_30002051_date_1898-Jan-03_sub_0"
"""

import json
import re
from pathlib import Path
from typing import List, Dict
import argparse


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for Swedish text.
    Swedish: approximately 1.3 words = 1 token
    """
    words = len(text.split())
    return int(words * 1.3)


def split_text_sliding_window(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Split text into chunks using sliding window with overlap.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
    
    Returns:
        List of text chunks with overlap
    """
    # Convert tokens to approximate words
    max_words = int(max_tokens / 1.3)
    overlap_words = int(overlap_tokens / 1.3)
    
    words = text.split()
    
    if len(words) <= max_words:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        # Take max_words from current position
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        
        # If we've reached the end, break
        if end >= len(words):
            break
        
        # Move forward by (max_words - overlap_words) for next chunk
        start += (max_words - overlap_words)
    
    return chunks


def split_chunks(input_file: Path, output_file: Path, max_tokens: int = 512, overlap_tokens: int = 50):
    """
    Split large chunks into smaller sub-chunks.
    
    Args:
        input_file: Path to input JSON with chunks
        output_file: Path to output JSON with split chunks
        max_tokens: Maximum tokens per chunk INCLUDING prefix (default 512)
        overlap_tokens: Overlap between consecutive sub-chunks (default 50)
    """
    print(f"📖 Loading chunks from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_chunks = data['chunks']
    print(f"   Original chunks: {len(original_chunks)}")
    
    # Calculate prefix overhead (approximately 24 tokens for Swedish text)
    prefix_overhead = 24
    content_max_tokens = max_tokens - prefix_overhead
    print(f"   Max tokens per chunk: {max_tokens} (content: {content_max_tokens}, prefix: ~{prefix_overhead})")
    
    def correct_date_for_prefix(date_str: str) -> str:
        """Fix OCR year errors in dates for prefix only - all should be 1898."""
        if not date_str.startswith('1898'):
            import re
            return re.sub(r'^\d{4}', '1898', date_str)
        return date_str
    
    new_chunks = []
    split_count = 0
    sub_chunk_count = 0
    
    for i, chunk in enumerate(original_chunks):
        tokens = estimate_tokens(chunk['text'])
        
        if tokens <= content_max_tokens:
            # Chunk fits within limit - add prefix and keep as is
            corrected_date = correct_date_for_prefix(chunk['date'])
            prefix = (
                f"Källa: Göteborgs Polisens Detektiva Afdelnings Rapporter. "
                f"Rapportens datum: {corrected_date}. "
                f"Text: "
            )
            final_text = prefix + chunk['text']
            
            chunk['text'] = final_text
            chunk['text_without_prefix'] = chunk['text']  # For unsplit chunks, original is in 'text'
            chunk['char_count'] = len(final_text)
            chunk['word_count'] = len(final_text.split())
            chunk['is_split'] = False
            chunk['sub_chunk_index'] = None
            chunk['parent_chunk_id'] = None
            new_chunks.append(chunk)
        else:
            # Split chunk into sub-chunks (using content_max_tokens for splitting)
            split_count += 1
            sub_texts = split_text_sliding_window(chunk['text'], content_max_tokens, overlap_tokens)
            
            print(f"   Splitting chunk {i} ({chunk['date']}): {tokens} tokens → {len(sub_texts)} sub-chunks")
            
            for sub_idx, sub_text in enumerate(sub_texts):
                # Add context prefix with corrected date
                corrected_date = correct_date_for_prefix(chunk['date'])
                prefix = (
                    f"Källa: Göteborgs Polisens Detektiva Afdelnings Rapporter. "
                    f"Rapportens datum: {corrected_date}. "
                )
                if len(sub_texts) > 1:
                    prefix += f"(Detta är del {sub_idx + 1} av {len(sub_texts)} från samma rapport). "
                prefix += "Text: "
                
                final_text = prefix + sub_text
                
                sub_chunk = {
                    'chunk_id': f"{chunk['chunk_id']}_sub_{sub_idx}",
                    'parent_chunk_id': chunk['chunk_id'],
                    'volume_id': chunk['volume_id'],
                    'date': chunk['date'],
                    'source_xmls': chunk['source_xmls'],
                    'is_split': True,
                    'sub_chunk_index': sub_idx,
                    'total_sub_chunks': len(sub_texts),
                    'text': final_text,
                    'text_without_prefix': sub_text,
                    'char_count': len(final_text),
                    'word_count': len(final_text.split()),
                    'year': chunk['year']
                }
                new_chunks.append(sub_chunk)
                sub_chunk_count += 1
    
    # Update output data
    output_data = {
        'volume_id': data['volume_id'],
        'year': data['year'],
        'original_chunk_count': len(original_chunks),
        'split_chunk_count': split_count,
        'total_chunks': len(new_chunks),
        'total_words': sum(c['word_count'] for c in new_chunks),
        'max_tokens': max_tokens,
        'overlap_tokens': overlap_tokens,
        'chunks': new_chunks
    }
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Splitting complete:")
    print(f"   Original chunks: {len(original_chunks)}")
    print(f"   Chunks split: {split_count}")
    print(f"   New sub-chunks created: {sub_chunk_count}")
    print(f"   Total final chunks: {len(new_chunks)}")
    print(f"   Saved to: {output_file}")
    
    # Show token distribution
    token_counts = [estimate_tokens(c['text']) for c in new_chunks]
    print(f"\n📊 Token distribution after splitting:")
    print(f"   Min: {min(token_counts)} tokens")
    print(f"   Max: {max(token_counts)} tokens")
    print(f"   Mean: {sum(token_counts)/len(token_counts):.0f} tokens")
    print(f"   Chunks over {max_tokens}: {sum(1 for t in token_counts if t > max_tokens)}")


def main():
    parser = argparse.ArgumentParser(
        description='Split large chunks into smaller sub-chunks with sliding window overlap'
    )
    parser.add_argument('input', help='Input JSON file with chunks')
    parser.add_argument('--output', '-o', help='Output JSON file (default: input_split.json)')
    parser.add_argument('--max-tokens', '-m', type=int, default=512,
                       help='Maximum tokens per chunk (default: 512)')
    parser.add_argument('--overlap', '-l', type=int, default=50,
                       help='Overlap tokens between chunks (default: 50)')
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        return
    
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"{input_file.stem}_split.json"
    
    print(f"\n{'='*80}")
    print(f"🔪 Splitting Large Chunks")
    print(f"{'='*80}\n")
    print(f"Max tokens per chunk: {args.max_tokens}")
    print(f"Overlap tokens: {args.overlap}")
    print()
    
    split_chunks(input_file, output_file, args.max_tokens, args.overlap)


if __name__ == '__main__':
    main()
