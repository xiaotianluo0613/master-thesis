#!/usr/bin/env python3
"""
Interactive viewer for complete reports.

Reads documents by report number and displays them as a continuous text.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict


def load_documents(jsonl_path: Path) -> dict:
    """Load all documents from JSONL file."""
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Create dict by ID for quick lookup
    return {doc['id']: doc for doc in documents}, documents


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


def group_documents_by_report(documents: list) -> dict:
    """Group documents by report number."""
    reports = defaultdict(list)
    
    for i, doc in enumerate(documents):
        text = doc.get('text', '').strip()
        if not text:
            continue
        
        report_num = extract_first_report_number(text)
        if report_num:
            reports[report_num].append(doc)
    
    return reports


def display_report(report_num: str, reports: dict) -> None:
    """Display a complete report."""
    if report_num not in reports:
        print(f"Report No {report_num} not found.")
        return
    
    pages = reports[report_num]
    
    print("\n" + "=" * 80)
    print(f"RAPPORT No {report_num} - Complete Report ({len(pages)} pages)")
    print("=" * 80 + "\n")
    
    for page_num, doc in enumerate(pages, 1):
        print(f"--- Page {page_num} (from {doc['id']}) ---\n")
        print(doc.get('text', ''))
        print()
    
    print("\n" + "=" * 80)
    print(f"End of Rapport No {report_num}")
    print("=" * 80 + "\n")


def list_reports(reports: dict) -> None:
    """List all available reports."""
    print("\n" + "=" * 80)
    print("AVAILABLE REPORTS")
    print("=" * 80 + "\n")
    
    for report_num in sorted(reports.keys(), key=lambda x: int(x)):
        num_pages = len(reports[report_num])
        print(f"Report No {report_num:3s} - {num_pages} page(s)")
    
    print(f"\nTotal: {len(reports)} reports\n")


def main():
    """Interactive report viewer."""
    workspace_root = Path(__file__).parent
    jsonl_file = workspace_root / "output.jsonl"
    
    if not jsonl_file.exists():
        print(f"Error: {jsonl_file} not found")
        sys.exit(1)
    
    doc_dict, documents = load_documents(jsonl_file)
    reports = group_documents_by_report(documents)
    
    print("\n" + "=" * 80)
    print("INTERACTIVE REPORT VIEWER")
    print("=" * 80)
    print("""
Commands:
  list                - Show all available reports
  read <report_num>   - Read complete report (e.g., 'read 1' or 'read 2')
  exit                - Exit viewer
    """)
    print("=" * 80)
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if not command:
                continue
            elif command == 'exit' or command == 'quit':
                print("Goodbye!")
                break
            elif command == 'list':
                list_reports(reports)
            elif command.startswith('read '):
                report_num = command.split()[1]
                display_report(report_num, reports)
            else:
                print("Unknown command. Try 'list', 'read <num>', or 'exit'")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # If arguments provided, use them directly (non-interactive mode)
    if len(sys.argv) > 1:
        workspace_root = Path(sys.argv[0]).parent
        jsonl_file = workspace_root / "output.jsonl"
        
        if not jsonl_file.exists():
            print(f"Error: {jsonl_file} not found")
            sys.exit(1)
        
        doc_dict, documents = load_documents(jsonl_file)
        reports = group_documents_by_report(documents)
        
        if sys.argv[1] == 'list':
            list_reports(reports)
        else:
            display_report(sys.argv[1], reports)
    else:
        main()
