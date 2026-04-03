#!/usr/bin/env python3
"""
Volume Stitcher: Parse all XML files in a volume and stitch text together.
Tracks source files for each piece of text.

Usage:
    python3 stitch_volume.py <volume_dir> [--output stitched.txt]
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import argparse
import json

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

def stitch_volume(volume_dir, output_file=None):
    """Stitch all XML files in a volume together."""
    volume_path = Path(volume_dir)
    
    if not volume_path.exists():
        print(f"Error: Directory {volume_dir} does not exist")
        sys.exit(1)
    
    # Get all XML files sorted by filename
    xml_files = sorted(volume_path.glob('*.xml'))
    
    if not xml_files:
        print(f"Error: No XML files found in {volume_dir}")
        sys.exit(1)
    
    volume_id = volume_path.name
    print(f"📖 Stitching volume: {volume_id}")
    print(f"   Found {len(xml_files)} XML files")
    
    # Stitch all pages with file markers
    full_text_lines = []
    source_map = []  # Track which XML each line came from
    total_lines = 0
    
    for xml_file in xml_files:
        lines = extract_text_from_xml(xml_file)
        
        if lines:
            # Add file marker before content
            marker = f"<FILE:{xml_file.name}>"
            full_text_lines.append(marker)
            
            # Add lines to full text
            full_text_lines.extend(lines)
            
            # Track source
            source_map.append({
                'xml_file': xml_file.name,
                'start_line': total_lines,
                'end_line': total_lines + len(lines) + 1,  # +1 for marker
                'line_count': len(lines)
            })
            
            total_lines += len(lines) + 1  # +1 for marker line
    
    # Join all text
    full_text = '\n'.join(full_text_lines)
    
    print(f"   ✓ Extracted {total_lines:,} text lines")
    print(f"   ✓ Total characters: {len(full_text):,}")
    print(f"   ✓ Total words: {len(full_text.split()):,}")
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        
        # Save stitched text WITHOUT headers (for processing)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"   ✓ Saved stitched text to: {output_path}")
        
        # Save source map as JSON
        map_file = output_path.with_suffix('.map.json')
        with open(map_file, 'w', encoding='utf-8') as f:
            json.dump({
                'volume_id': volume_id,
                'total_xml_files': len(xml_files),
                'total_lines': total_lines,
                'total_chars': len(full_text),
                'total_words': len(full_text.split()),
                'source_map': source_map
            }, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Saved source map to: {map_file}")
        
        # Show first 2000 characters as preview
        print(f"\n{'='*80}")
        print("📄 PREVIEW (first 2000 characters):")
        print(f"{'='*80}\n")
        print(full_text[:2000])
        if len(full_text) > 2000:
            print("\n[... content continues ...]")
    
    return full_text, source_map, volume_id

def main():
    parser = argparse.ArgumentParser(description='Stitch all XML files in a volume')
    parser.add_argument('volume_dir', help='Path to volume directory containing XML files')
    parser.add_argument('--output', '-o', help='Output file for stitched text', default=None)
    
    args = parser.parse_args()
    
    if not args.output:
        volume_name = Path(args.volume_dir).name
        args.output = f"{volume_name}_stitched.txt"
    
    stitch_volume(args.volume_dir, args.output)

if __name__ == '__main__':
    main()
