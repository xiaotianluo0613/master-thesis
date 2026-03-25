#!/usr/bin/env python3
"""
Comprehensive statistical analysis of 30002022 dataset.
"""

import json
import statistics
from pathlib import Path
from collections import Counter


def analyze_dataset_30002022():
    """
    Perform statistical analysis on output_30002022.jsonl
    """
    workspace_root = Path(__file__).parent
    input_file = workspace_root / "output_30002022.jsonl"
    output_file = workspace_root / "analysis_30002022.txt"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return
    
    # Read all documents
    documents = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    
    # Calculate statistics
    total_docs = len(documents)
    empty_docs = sum(1 for doc in documents if not doc["text"].strip())
    non_empty_docs = total_docs - empty_docs
    
    # Character counts
    char_counts = [len(doc["text"]) for doc in documents]
    total_chars = sum(char_counts)
    
    # Word counts
    word_counts = [len(doc["text"].split()) for doc in documents]
    total_words = sum(word_counts)
    
    # Line counts
    line_counts = [len(doc["text"].split('\n')) for doc in documents]
    total_lines = sum(line_counts)
    
    # Exclude empty documents for average calculations
    non_empty_char_counts = [len(doc["text"]) for doc in documents if doc["text"].strip()]
    non_empty_word_counts = [len(doc["text"].split()) for doc in documents if doc["text"].strip()]
    non_empty_line_counts = [len(doc["text"].split('\n')) for doc in documents if doc["text"].strip()]
    
    # Statistics
    avg_chars_per_doc = statistics.mean(non_empty_char_counts) if non_empty_char_counts else 0
    median_chars = statistics.median(non_empty_char_counts) if non_empty_char_counts else 0
    stdev_chars = statistics.stdev(non_empty_char_counts) if len(non_empty_char_counts) > 1 else 0
    
    avg_words_per_doc = statistics.mean(non_empty_word_counts) if non_empty_word_counts else 0
    median_words = statistics.median(non_empty_word_counts) if non_empty_word_counts else 0
    
    avg_lines_per_doc = statistics.mean(non_empty_line_counts) if non_empty_line_counts else 0
    median_lines = statistics.median(non_empty_line_counts) if non_empty_line_counts else 0
    
    # Min/Max
    min_chars = min(non_empty_char_counts) if non_empty_char_counts else 0
    max_chars = max(non_empty_char_counts) if non_empty_char_counts else 0
    
    min_words = min(non_empty_word_counts) if non_empty_word_counts else 0
    max_words = max(non_empty_word_counts) if non_empty_word_counts else 0
    
    # Quartiles
    sorted_chars = sorted(non_empty_char_counts)
    q1_chars = sorted_chars[len(sorted_chars)//4] if sorted_chars else 0
    q3_chars = sorted_chars[3*len(sorted_chars)//4] if sorted_chars else 0
    
    # Categorize documents by size
    very_short = sum(1 for c in non_empty_char_counts if c < 100)
    short = sum(1 for c in non_empty_char_counts if 100 <= c < 500)
    medium = sum(1 for c in non_empty_char_counts if 500 <= c < 2000)
    long = sum(1 for c in non_empty_char_counts if 2000 <= c < 5000)
    very_long = sum(1 for c in non_empty_char_counts if c >= 5000)
    
    # Report generation
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET ANALYSIS: 30002022\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total documents: {total_docs}\n")
        f.write(f"Non-empty documents: {non_empty_docs}\n")
        f.write(f"Empty documents: {empty_docs}\n")
        f.write(f"Empty document rate: {empty_docs/total_docs*100:.1f}%\n\n")
        
        f.write("TOTAL CONTENT\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total characters: {total_chars:,}\n")
        f.write(f"Total words (estimated): {total_words:,}\n")
        f.write(f"Total lines: {total_lines:,}\n\n")
        
        f.write("CHARACTER COUNT STATISTICS (excluding empty documents)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average characters per document: {avg_chars_per_doc:.0f}\n")
        f.write(f"Median characters per document: {median_chars:.0f}\n")
        f.write(f"Standard deviation: {stdev_chars:.0f}\n")
        f.write(f"Minimum: {min_chars:,}\n")
        f.write(f"Maximum: {max_chars:,}\n")
        f.write(f"Range: {max_chars - min_chars:,}\n")
        f.write(f"Q1 (25th percentile): {q1_chars:,}\n")
        f.write(f"Q3 (75th percentile): {q3_chars:,}\n")
        f.write(f"IQR (Interquartile Range): {q3_chars - q1_chars:,}\n\n")
        
        f.write("WORD COUNT STATISTICS (excluding empty documents)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average words per document: {avg_words_per_doc:.0f}\n")
        f.write(f"Median words per document: {median_words:.0f}\n")
        f.write(f"Minimum: {min_words:,}\n")
        f.write(f"Maximum: {max_words:,}\n\n")
        
        f.write("LINE COUNT STATISTICS (excluding empty documents)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average lines per document: {avg_lines_per_doc:.1f}\n")
        f.write(f"Median lines per document: {median_lines:.0f}\n\n")
        
        f.write("DOCUMENT SIZE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Very short (< 100 chars): {very_short:3d} documents ({very_short/non_empty_docs*100:5.1f}%)\n")
        f.write(f"Short (100-500 chars): {short:3d} documents ({short/non_empty_docs*100:5.1f}%)\n")
        f.write(f"Medium (500-2000 chars): {medium:3d} documents ({medium/non_empty_docs*100:5.1f}%)\n")
        f.write(f"Long (2000-5000 chars): {long:3d} documents ({long/non_empty_docs*100:5.1f}%)\n")
        f.write(f"Very long (5000+ chars): {very_long:3d} documents ({very_long/non_empty_docs*100:5.1f}%)\n\n")
        
        f.write("DATA QUALITY NOTES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Completeness: {non_empty_docs/total_docs*100:.1f}% of documents contain text\n")
        f.write(f"Average document density: {avg_chars_per_doc/median_chars:.2f}x median\n")
        f.write(f"Document consistency: {'Good' if stdev_chars < avg_chars_per_doc else 'High variance'}\n\n")
        
        # Sample documents
        f.write("SAMPLE DOCUMENTS\n")
        f.write("-" * 80 + "\n")
        
        # Shortest non-empty
        shortest_doc = min([d for d in documents if d["text"].strip()], key=lambda d: len(d["text"]))
        f.write(f"Shortest document: {shortest_doc['id']} ({len(shortest_doc['text'])} chars)\n")
        f.write(f"Preview: {shortest_doc['text'][:100]}...\n\n")
        
        # Longest
        longest_doc = max([d for d in documents if d["text"].strip()], key=lambda d: len(d["text"]))
        f.write(f"Longest document: {longest_doc['id']} ({len(longest_doc['text'])} chars)\n")
        f.write(f"Preview: {longest_doc['text'][:100]}...\n\n")
        
        # Average size
        avg_doc = sorted([d for d in documents if d["text"].strip()], 
                         key=lambda d: abs(len(d["text"]) - avg_chars_per_doc))[0]
        f.write(f"Near-average document: {avg_doc['id']} ({len(avg_doc['text'])} chars)\n")
        f.write(f"Preview: {avg_doc['text'][:100]}...\n")
    
    print(f"Analysis complete: {output_file}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("QUICK SUMMARY - 30002022 Dataset")
    print("=" * 80)
    print(f"Documents: {total_docs} total, {non_empty_docs} with content ({empty_docs} empty)")
    print(f"Content: {total_chars:,} characters, {total_words:,} words")
    print(f"Average per document: {avg_chars_per_doc:.0f} chars, {avg_words_per_doc:.0f} words")
    print(f"Size range: {min_chars:,} - {max_chars:,} characters")
    print(f"Distribution: Very short={very_short}, Short={short}, Medium={medium}, Long={long}, Very long={very_long}")
    print()


if __name__ == "__main__":
    analyze_dataset_30002022()
