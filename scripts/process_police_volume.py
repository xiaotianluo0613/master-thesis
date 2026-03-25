#!/usr/bin/env python3
"""
Police Reports Volume Processor

Processes a police report volume (e.g., 30002051) through three phases:
1. Parse & Stitch: Extract text from all XML files
2. Split by Cases: Divide into case chunks based on "No X" patterns  
3. Add Context: Enrich chunks with volume metadata

Output: JSON file with chunks ready for RAG pipeline

Usage:
    python3 process_police_volume.py <volume_dir> --year <year> [--output chunks.json]
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import argparse
import json
import re
from typing import List, Dict, Tuple

def extract_text_from_xml(xml_file):
    """Extract text lines from a single XML file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        lines = []
        # Process by TextLine elements to preserve line structure
        for text_line in root.findall('.//{*}TextLine'):
            # Get all String elements (words) in this line
            words = []
            for string_elem in text_line.findall('.//{*}String'):
                content = string_elem.get('CONTENT', '')
                if content:
                    words.append(content)
            
            if words:
                # Join words with spaces to form a line
                line_text = ' '.join(words)
                lines.append(line_text)
        
        return lines
    except Exception as e:
        print(f"Warning: Error reading {xml_file.name}: {e}", file=sys.stderr)
        return []

def stitch_volume(volume_dir: Path) -> Tuple[str, List[Dict], str]:
    """
    Phase 1: Stitch all XML files in volume together.
    
    Returns:
        full_text: Complete stitched text
        source_map: List of {xml_file, start_line, end_line, line_count}
        volume_id: Volume identifier
    """
    xml_files = sorted(volume_dir.glob('*.xml'))
    
    if not xml_files:
        raise ValueError(f"No XML files found in {volume_dir}")
    
    volume_id = volume_dir.name
    print(f"📖 Phase 1: Stitching volume {volume_id}")
    print(f"   Found {len(xml_files)} XML files")
    
    full_text_lines = []
    source_map = []
    total_lines = 0
    
    for xml_file in xml_files:
        lines = extract_text_from_xml(xml_file)
        
        if lines:
            # Add FILE marker
            marker = f"<FILE:{xml_file.name}>"
            full_text_lines.append(marker)
            
            # Add content lines
            full_text_lines.extend(lines)
            
            source_map.append({
                'xml_file': xml_file.name,
                'start_line': total_lines,
                'end_line': total_lines + len(lines) + 1,  # +1 for marker
                'line_count': len(lines)
            })
            
            total_lines += len(lines) + 1  # +1 for marker
    
    full_text = '\n'.join(full_text_lines)
    
    print(f"   ✓ Extracted {total_lines:,} lines")
    print(f"   ✓ {len(full_text):,} chars, {len(full_text.split()):,} words")
    
    return full_text, source_map, volume_id

def find_case_boundaries(text: str) -> List[Tuple[int, str]]:
    """
    Phase 2: Find daily report boundaries using date patterns.
    
    Strategy: Find "Göteborg den DD Month YYYY" patterns which indicate
    the start of each daily police report. Each daily report can contain
    multiple cases, and some cases may span across multiple daily reports.
    
    Note: Ignores "S: D." and "S. d." (samma dag/same day) patterns - these
    are continuation markers, not new date boundaries.
    
    Returns list of (position, date_id) tuples in document order
    """
    boundaries = []
    
    # Pattern: Match various date formats with OCR errors and abbreviations
    # Examples:
    # - "Göteborg den 3 Januari 1898."
    # - "Göteborg den 7. Januari 1898." (period after day)
    # - "Göteborg den 15 Jan. 1898." (abbreviated month)
    # - "Göteborg 17 Januari 1898." (missing "den")
    # - "dröteborg den 8 Januari 1898." (OCR error: G->d)
    # Swedish months can vary: Januari/Januarii, Sept/September, etc.
    # We only match full dates, NOT abbreviations like "S: D." or "S. d."
    
    # Pattern parts:
    # - (dröteborg|Göteborg): Handle OCR error
    # - (?: den)?: Optional " den" (may be missing, space included in optional group)
    # - \s+: Required space between city/den and day number
    # - (\d{1,2})\.?: Day with optional period
    # - ([A-Za-zåäö]+\.?): Month (full or abbreviated with optional period)
    # - (\d{4})\.: Year with period
    pattern = r'(?:dröteborg|Göteborg)(?: den)?\s+(\d{1,2})\.? ([A-Za-zåäö]+\.?) (\d{4})\.'
    
    for match in re.finditer(pattern, text, re.IGNORECASE):
        day = match.group(1)
        month = match.group(2)
        year = match.group(3)
        position = match.start()
        
        # Create a date ID like "1898-01-03" (normalized)
        date_id = f"{year}-{month[:3]}-{day.zfill(2)}"
        
        boundaries.append((position, date_id))
    
    return boundaries

def get_source_xmls(start_line: int, end_line: int, source_map: List[Dict]) -> List[str]:
    """Get list of XML files that contain lines in range [start_line, end_line]."""
    xml_files = []
    
    for source in source_map:
        # Check if this source overlaps with our range
        if source['end_line'] > start_line and source['start_line'] < end_line:
            xml_files.append(source['xml_file'])
    
    return xml_files

def split_into_cases(full_text: str, source_map: List[Dict], volume_id: str) -> List[Dict]:
    """
    Phase 2: Split stitched text into daily report chunks.
    
    Each chunk includes:
    - chunk_id: Unique identifier
    - volume_id: Source volume
    - date: Date from "Göteborg den..." pattern
    - source_xmls: List of XML files containing this daily report
    - text: Daily report text content
    - char_count: Number of characters
    - word_count: Number of words
    """
    print(f"\n🔍 Phase 2: Splitting by daily reports (dates)")
    
    boundaries = find_case_boundaries(full_text)
    
    if not boundaries:
        print(f"   ⚠️  No date boundaries found")
        return []
    
    print(f"   Found {len(boundaries)} daily report boundaries")
    
    chunks = []
    
    for i, (position, date_id) in enumerate(boundaries):
        # Find next boundary or end of text
        if i + 1 < len(boundaries):
            next_position = boundaries[i + 1][0]
        else:
            next_position = len(full_text)
        
        # Extract text FROM this date boundary TO next boundary (or EOF)
        # This ensures each chunk starts exactly where the date appears
        report_text = full_text[position:next_position].strip()
        
        # Find which file this date appears in by looking backward for FILE marker
        # Search from start of document up to current position (not just previous boundary)
        lookback_text = full_text[0:position]
        
        # Find the last FILE marker before this date
        file_pattern = r'<FILE:([^>]+)>'
        lookback_files = re.findall(file_pattern, lookback_text)
        
        # Extract FILE markers that appear in the report text itself
        report_files = re.findall(file_pattern, report_text)
        
        # Combine: starting file (from lookback) + files within report
        source_xmls = []
        if lookback_files:
            source_xmls.append(lookback_files[-1])  # Last file before date
        source_xmls.extend(report_files)  # Files within the report
        
        # Remove markers from text for clean output
        clean_text = re.sub(file_pattern + r'\n?', '', report_text)
        
        chunk = {
            'chunk_id': f'vol_{volume_id}_date_{date_id}',
            'volume_id': volume_id,
            'date': date_id,
            'source_xmls': source_xmls,
            'text': clean_text,
            'char_count': len(clean_text),
            'word_count': len(clean_text.split())
        }
        
        chunks.append(chunk)
    
    print(f"   ✓ Created {len(chunks)} daily report chunks")
    print(f"   ✓ Avg words/report: {sum(c['word_count'] for c in chunks) / len(chunks):.0f}")
    
    return chunks

def process_volume(volume_dir: Path, year: int, output_file: Path = None):
    """Main processing pipeline."""
    
    print(f"\n{'='*80}")
    print(f"🚔 Processing Police Report Volume")
    print(f"{'='*80}\n")
    
    # Phase 1: Stitch
    full_text, source_map, volume_id = stitch_volume(volume_dir)
    
    # Phase 2: Split by cases
    chunks = split_into_cases(full_text, source_map, volume_id)
    
    if not chunks:
        print("\n⚠️  No chunks created. Volume may not follow expected format.")
        return
    
    # Add year to each chunk
    for chunk in chunks:
        chunk['year'] = year
    
    # Save output
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'volume_id': volume_id,
                'year': year,
                'total_chunks': len(chunks),
                'total_words': sum(c['word_count'] for c in chunks),
                'chunks': chunks
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"✅ SUCCESS")
        print(f"{'='*80}")
        print(f"   Output: {output_file}")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Total words: {sum(c['word_count'] for c in chunks):,}")
        
        # Show sample chunk
        print(f"\n{'='*80}")
        print(f"📋 SAMPLE CHUNK (First case)")
        print(f"{'='*80}\n")
        sample = chunks[0]
        print(f"Chunk ID: {sample['chunk_id']}")
        print(f"Date: {sample['date']}")
        print(f"Year: {sample['year']}")
        print(f"Source XMLs: {', '.join(sample['source_xmls'][:5])}{'...' if len(sample['source_xmls']) > 5 else ''}")
        print(f"Words: {sample['word_count']}")
        print(f"\nText (first 800 chars):\n{sample['text'][:800]}...")
        print()

def main():
    parser = argparse.ArgumentParser(
        description='Process police report volume into daily report chunks for RAG pipeline'
    )
    parser.add_argument('volume_dir', help='Path to volume directory containing XML files')
    parser.add_argument('--year', '-y', type=int, required=True, help='Year of the volume (e.g., 1898)')
    parser.add_argument('--output', '-o', help='Output JSON file', default=None)
    
    args = parser.parse_args()
    
    volume_dir = Path(args.volume_dir)
    
    if not volume_dir.exists():
        print(f"Error: Directory {volume_dir} does not exist")
        sys.exit(1)
    
    if not args.output:
        args.output = f"{volume_dir.name}_chunks.json"
    
    output_file = Path(args.output)
    
    process_volume(volume_dir, args.year, output_file)

if __name__ == '__main__':
    main()
