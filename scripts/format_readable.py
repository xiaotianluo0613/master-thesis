#!/usr/bin/env python3
"""
Convert JSONL output to a human-readable format for data exploration.

Displays each document with its metadata and text in an easy-to-read format,
helping understand document boundaries and relationships.
"""

import json
import sys
from pathlib import Path


def format_jsonl_readable(jsonl_path: Path, output_path: Path) -> None:
    """
    Convert JSONL file to readable text format.
    
    Args:
        jsonl_path: Path to input JSONL file
        output_path: Path to output readable text file
    """
    with open(jsonl_path, 'r', encoding='utf-8') as inf, \
         open(output_path, 'w', encoding='utf-8') as outf:
        
        for idx, line in enumerate(inf, 1):
            try:
                entry = json.loads(line)
                
                # Write document header with metadata
                outf.write('=' * 80 + '\n')
                outf.write(f"ID: {entry.get('id', 'N/A')}\n")
                outf.write(f"Volume: {entry.get('metadata', {}).get('volume', 'N/A')}\n")
                outf.write(f"Text Length: {len(entry.get('text', ''))} characters\n")
                outf.write('=' * 80 + '\n')
                
                # Write text
                text = entry.get('text', '')
                if text:
                    outf.write(text)
                else:
                    outf.write('[EMPTY - No text content]\n')
                
                # Add spacing between documents
                outf.write('\n\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {idx}: {e}", file=sys.stderr)
                continue
    
    print(f"✓ Formatted {idx} documents")
    print(f"✓ Output saved to: {output_path}")


def main():
    """Main function."""
    workspace_root = Path(__file__).parent
    jsonl_file = workspace_root / "output.jsonl"
    readable_file = workspace_root / "output_readable.txt"
    
    if not jsonl_file.exists():
        print(f"Error: {jsonl_file} not found", file=sys.stderr)
        sys.exit(1)
    
    format_jsonl_readable(jsonl_file, readable_file)


if __name__ == "__main__":
    main()
