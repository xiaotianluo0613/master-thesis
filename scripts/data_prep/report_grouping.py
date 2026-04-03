#!/usr/bin/env python3
"""
Group documents by report number to show multi-page reports.

This helps understand which XML files belong to the same police report.
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def extract_first_report_number(text: str) -> str:
    """
    Extract the FIRST report number from text (report header).
    
    Looks for "Rapport No X" or "No X." pattern that appears like a header,
    NOT just any "No X" which might be a house number.
    
    Strategy: Look for patterns that indicate a report header:
    - "Rapport No X"
    - "No X." (followed by period, appears early in text)
    - Usually appears in first 200 characters
    """
    text_start = text[:300]
    
    # First try: "Rapport No X" - most reliable
    match = re.search(r'Rapport\s+No\s+(\d+)', text_start, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Second try: "No X." with period (report header style)
    match = re.search(r'^\s*No\s+(\d+)\s*[\.\!]', text_start, re.MULTILINE)
    if match:
        return match.group(1)
    
    # Third try: "No X" at very start (but avoid if it's far into text)
    match = re.search(r'\bNo\s+(\d+)\b', text[:150])
    if match:
        return match.group(1)
    
    return None


def group_by_report(jsonl_path: Path) -> None:
    """
    Group documents by report number and show continuity.
    
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
    print("POLICE REPORTS - GROUPED BY REPORT NUMBER")
    print("=" * 80)
    
    # Group documents by report number
    reports = defaultdict(list)
    ungrouped = []
    
    for i, doc in enumerate(documents):
        text = doc.get('text', '').strip()
        if not text:
            continue
        
        report_num = extract_first_report_number(text)
        
        if report_num:
            reports[report_num].append((i, doc))
        else:
            ungrouped.append((i, doc))
    
    # Display report summaries
    print(f"\nFound {len(reports)} distinct report numbers")
    print(f"Documents in multi-page reports: {sum(len(v) for v in reports.values() if len(v) > 1)}")
    print(f"Likely continuation/ungrouped documents: {len(ungrouped)}")
    
    # Show sample reports
    print(f"\n" + "-" * 80)
    print("REPORT SUMMARIES (showing first 15 reports)")
    print("-" * 80)
    
    for report_num in sorted(reports.keys(), key=lambda x: int(x))[:15]:
        pages = reports[report_num]
        print(f"\nRapport No {report_num}:")
        print(f"  Pages: {len(pages)}")
        
        for page_idx, (doc_idx, doc) in enumerate(pages, 1):
            text = doc.get('text', '')
            first_line = text.split('\n')[0][:60]
            print(f"    Page {page_idx}: ID {doc['id']} - {first_line}...")
    
    # Show continuity example
    print(f"\n" + "-" * 80)
    print("CONTINUITY EXAMPLE - Rapport No 1")
    print("-" * 80)
    
    if '1' in reports:
        pages = reports['1']
        print(f"\nReport 1 spans {len(pages)} documents:")
        
        for page_idx, (doc_idx, doc) in enumerate(pages, 1):
            text = doc.get('text', '')
            # Show end of page and beginning of next
            lines = text.split('\n')
            print(f"\n--- Document {page_idx}: {doc['id']} ---")
            print(f"Last line: {lines[-2] if len(lines) > 1 else ''}")
            
            if page_idx < len(pages):
                next_text = pages[page_idx][1].get('text', '')
                next_lines = next_text.split('\n')
                print(f"Next page first line: {next_lines[0] if next_lines else ''}")
    
    print("\n" + "=" * 80)
    print("HOW TO USE THIS INFORMATION:")
    print("=" * 80)
    print("""
1. Multi-page reports: Documents with the same "Rapport No X" belong together
   
2. Sequential reading: Documents should be read in order of their ID number
   to reconstruct full reports
   
3. Document boundaries: When text ends mid-sentence and the next document
   starts with lowercase, they likely continue from each other
   
4. Ungrouped documents: May be continuation pages, title pages, or 
   independent short reports

Example: Rapport No 1 spans documents 30002021_00004 and 30002021_00005
Read them in sequence to get the complete report.
    """)
    print("=" * 80 + "\n")


def main():
    """Main function."""
    workspace_root = Path(__file__).parent
    jsonl_file = workspace_root / "output.jsonl"
    
    if not jsonl_file.exists():
        print(f"Error: {jsonl_file} not found")
        return
    
    group_by_report(jsonl_file)


if __name__ == "__main__":
    main()
