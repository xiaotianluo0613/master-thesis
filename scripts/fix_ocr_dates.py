#!/usr/bin/env python3
"""
Fix OCR date errors in chunk files.
All dates should be in 1898 based on volume metadata.
"""

import json
import sys
import re
from pathlib import Path


def fix_date_string(date_str: str) -> str:
    """
    Fix OCR errors in date string by replacing wrong year with 1898.
    
    Args:
        date_str: Date string in format YYYY-MMM-DD
        
    Returns:
        Corrected date string with year 1898
    """
    # Replace any 4-digit year that's not 1898 with 1898
    pattern = r'^\d{4}'
    if not date_str.startswith('1898'):
        return re.sub(pattern, '1898', date_str)
    return date_str


def fix_chunk_dates(input_file: str, output_file: str = None) -> None:
    """
    Fix OCR date errors in chunks JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (defaults to overwriting input)
    """
    if output_file is None:
        output_file = input_file
    
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data['chunks']
    fixed_count = 0
    
    print("\nFixing OCR date errors...")
    for chunk in chunks:
        original_date = chunk['date']
        
        if not original_date.startswith('1898'):
            # Fix the date field
            corrected_date = fix_date_string(original_date)
            chunk['date'] = corrected_date
            
            # Fix the year field
            chunk['year'] = 1898
            
            # Fix chunk_id if it contains the date
            if 'date_' in chunk['chunk_id']:
                chunk['chunk_id'] = chunk['chunk_id'].replace(
                    f"date_{original_date}",
                    f"date_{corrected_date}"
                )
            
            # Fix parent_chunk_id if present
            if 'parent_chunk_id' in chunk and chunk['parent_chunk_id']:
                chunk['parent_chunk_id'] = chunk['parent_chunk_id'].replace(
                    f"date_{original_date}",
                    f"date_{corrected_date}"
                )
            
            # Fix the prefix in the text field if present
            if 'text' in chunk and 'Rapportens datum:' in chunk['text']:
                chunk['text'] = chunk['text'].replace(
                    f"Rapportens datum: {original_date}",
                    f"Rapportens datum: {corrected_date}"
                )
            
            fixed_count += 1
            print(f"  Fixed: {original_date} → {corrected_date} (chunk {chunk['chunk_id']})")
    
    print(f"\nFixed {fixed_count} chunks with wrong dates")
    print(f"Saving to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("Done!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 fix_ocr_dates.py <input_file.json> [output_file.json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    fix_chunk_dates(input_file, output_file)
