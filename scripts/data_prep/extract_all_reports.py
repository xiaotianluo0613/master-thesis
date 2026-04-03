#!/usr/bin/env python3
"""
Advanced report extractor that handles multiple reports per document page.

Since some pages contain multiple short case reports, this properly
extracts each report as a separate unit.
"""

import json
import re
import sys
from pathlib import Path


def extract_all_reports_from_text(text: str, doc_id: str) -> list:
    """
    Extract all individual reports from a single document.
    
    Some pages contain multiple short reports, each starting with 'No X.'
    
    Args:
        text: The full text from a document
        doc_id: The document ID
        
    Returns:
        List of dicts with 'report_num', 'text', 'source_doc', 'page'
    """
    if not text.strip():
        return []
    
    reports = []
    
    # Find all 'No X.' pattern positions (report headers)
    pattern = r'\bNo\s+(\d+)\s*[\.\!]'
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        # No report header found - this is a continuation page
        return []
    
    # Extract each report from its header to the next header (or end of text)
    for i, match in enumerate(matches):
        report_num = match.group(1)
        start_pos = match.start()
        
        # Find end position (start of next report or end of text)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)
        
        report_text = text[start_pos:end_pos].strip()
        
        reports.append({
            'report_num': report_num,
            'text': report_text,
            'source_doc': doc_id,
            'page_offset': i + 1  # Which report on this page
        })
    
    return reports


def main():
    """Extract all reports from JSONL file."""
    workspace_root = Path(__file__).parent
    jsonl_file = workspace_root / "output.jsonl"
    
    if not jsonl_file.exists():
        print(f"Error: {jsonl_file} not found")
        sys.exit(1)
    
    # Load all documents
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        docs = [json.loads(line) for line in f]
    
    # Extract all reports
    all_reports = []
    for doc in docs:
        reports = extract_all_reports_from_text(doc['text'], doc['id'])
        all_reports.extend(reports)
    
    print("\n" + "="*80)
    print("COMPLETE REPORT EXTRACTION")
    print("="*80)
    
    print(f"\nTotal documents: {len(docs)}")
    print(f"Total distinct reports extracted: {len(all_reports)}")
    
    # Get unique report numbers
    unique_reports = set(r['report_num'] for r in all_reports)
    print(f"Unique report numbers: {len(unique_reports)}")
    print(f"Report numbers: {sorted(map(int, unique_reports))}")
    
    print(f"\n" + "-"*80)
    print("REPORT SUMMARY")
    print("-"*80)
    
    # Group by report number
    by_number = {}
    for report in all_reports:
        num = report['report_num']
        if num not in by_number:
            by_number[num] = []
        by_number[num].append(report)
    
    # Show reports with multiple occurrences
    multi_occurrences = {num: reports for num, reports in by_number.items() if len(reports) > 1}
    
    if multi_occurrences:
        print(f"\nReports appearing MULTIPLE times:")
        for num in sorted(map(int, multi_occurrences.keys()), key=int):
            num = str(num)
            reports = multi_occurrences[num]
            print(f"\n  Report No {num}: appears {len(reports)} time(s)")
            for report in reports:
                text_preview = report['text'][:80].replace('\n', ' ')
                print(f"    → {report['source_doc']} (page {report['page_offset']}): {text_preview}...")
    
    print(f"\n" + "-"*80)
    print("FIRST 15 REPORTS")
    print("-"*80)
    
    sorted_reports = sorted(all_reports, key=lambda r: int(r['report_num']))
    for i, report in enumerate(sorted_reports[:15]):
        text_preview = report['text'][:100].replace('\n', ' ')
        print(f"\n{i+1}. Report No {report['report_num']} (from {report['source_doc']}):")
        print(f"   {text_preview}...")
    
    print(f"\n" + "="*80)
    print(f"ANSWER: Your archive contains {len(all_reports)} individual case reports")
    print(f"        in {len(docs)} document pages")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
