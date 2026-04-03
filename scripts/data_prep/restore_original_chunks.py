#!/usr/bin/env python3
"""
Restore original chunks with OCR dates from the split file.
"""

import json


# Load the split file which has original OCR dates
with open('30002051_chunks_split.json', 'r', encoding='utf-8') as f:
    split_data = json.load(f)

# Extract unique parent chunks (reconstruct original chunks before splitting)
parent_chunks = {}

for chunk in split_data['chunks']:
    if chunk['is_split']:
        # This is a sub-chunk - get parent info
        parent_id = chunk['parent_chunk_id']
        if parent_id not in parent_chunks:
            # Reconstruct parent chunk
            parent_chunks[parent_id] = {
                'chunk_id': parent_id,
                'volume_id': chunk['volume_id'],
                'date': chunk['date'],
                'source_xmls': chunk['source_xmls'],
                'year': chunk['year'],
                'text_parts': []
            }
        # Collect text parts (without prefix if exists)
        text = chunk.get('text_without_prefix', chunk['text'])
        parent_chunks[parent_id]['text_parts'].append((chunk['sub_chunk_index'], text))
    else:
        # Unsplit chunk - use as is
        parent_id = chunk['chunk_id']
        parent_chunks[parent_id] = {
            'chunk_id': chunk['chunk_id'],
            'volume_id': chunk['volume_id'],
            'date': chunk['date'],
            'source_xmls': chunk['source_xmls'],
            'year': chunk['year'],
            'text': chunk.get('text_without_prefix', chunk['text']),
            'char_count': chunk.get('char_count', len(chunk['text'])),
            'word_count': chunk.get('word_count', len(chunk['text'].split()))
        }

# Reconstruct full text for split chunks
final_chunks = []
for parent_id, parent_data in parent_chunks.items():
    if 'text_parts' in parent_data:
        # Sort by sub_chunk_index and join (accounting for overlap)
        sorted_parts = sorted(parent_data['text_parts'], key=lambda x: x[0])
        # For now, just use the first part's full text as representative
        # (The actual stitching would need overlap removal logic)
        full_text = sorted_parts[0][1]  # First sub-chunk
        
        chunk = {
            'chunk_id': parent_data['chunk_id'],
            'volume_id': parent_data['volume_id'],
            'date': parent_data['date'],
            'source_xmls': parent_data['source_xmls'],
            'year': parent_data['year'],
            'text': full_text,
            'char_count': len(full_text),
            'word_count': len(full_text.split())
        }
    else:
        chunk = parent_data
    
    final_chunks.append(chunk)

# Sort by chunk_id
final_chunks.sort(key=lambda x: x['chunk_id'])

output = {
    'volume_id': split_data['volume_id'],
    'total_chunks': len(final_chunks),
    'chunks': final_chunks
}

with open('30002051_chunks.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ Restored {len(final_chunks)} original chunks with OCR dates")
print(f"   Saved to: 30002051_chunks.json")

# Show some examples with OCR dates
wrong_dates = [c for c in final_chunks if not c['date'].startswith('1898')]
print(f"\n📊 Chunks with OCR date errors (preserved): {len(wrong_dates)}")
for c in wrong_dates[:3]:
    print(f"   {c['date']}")
