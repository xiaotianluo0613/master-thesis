#!/usr/bin/env python3
"""
Convert JSONL format to readable text for 30002022 dataset.
"""

import json
from pathlib import Path


def format_jsonl_readable_30002022():
    """
    Convert output_30002022.jsonl to output_30002022_readable.txt
    """
    workspace_root = Path(__file__).parent
    input_file = workspace_root / "output_30002022.jsonl"
    output_file = workspace_root / "output_30002022_readable.txt"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return
    
    with open(input_file, "r", encoding="utf-8") as inf:
        with open(output_file, "w", encoding="utf-8") as outf:
            for line in inf:
                data = json.loads(line)
                doc_id = data.get("id", "UNKNOWN")
                text = data.get("text", "")
                metadata = data.get("metadata", {})
                volume = metadata.get("volume", "UNKNOWN")
                
                # Format output
                outf.write(f"{'='*80}\n")
                outf.write(f"ID: {doc_id} | VOLUME: {volume}\n")
                outf.write(f"Text Length: {len(text)} characters\n")
                outf.write(f"{'='*80}\n\n")
                outf.write(text)
                outf.write("\n\n")
    
    print(f"Successfully created {output_file}")
    
    # Get file size
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    format_jsonl_readable_30002022()
