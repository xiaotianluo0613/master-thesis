#!/usr/bin/env python3
"""
Subset Selection Helper: Analyze document types to choose best subset for pipeline.

For each document type, provides:
- Average OCR quality (PC score)
- Volume count and page count
- Text quality metrics (noise, hyphen ratio, blank pages)
- 2 real text examples from actual volumes

Usage:
    python3 subset_selector.py comprehensive_volume_fingerprints.csv <data_dir>
"""

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import statistics
import random

def safe_float(value, default=0.0):
    """Safely convert to float."""
    try:
        return float(value) if value and value != '' else default
    except ValueError:
        return default

def safe_int(value, default=0):
    """Safely convert to int."""
    try:
        return int(value) if value and value != '' else default
    except ValueError:
        return default

def extract_text_sample(xml_file, max_chars=500):
    """Extract a text sample from XML file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract text from TextLine elements
        text_lines = []
        for text_line in root.findall('.//{*}TextLine'):
            unicode_elem = text_line.find('.//{*}Unicode')
            if unicode_elem is not None and unicode_elem.text:
                text_lines.append(unicode_elem.text.strip())
        
        full_text = ' '.join(text_lines)
        
        # Return sample
        if len(full_text) > max_chars:
            return full_text[:max_chars] + '...'
        return full_text if full_text else "[No text found]"
    
    except Exception as e:
        return f"[Error reading file: {e}]"

def analyze_by_type(csv_file, data_dir):
    """Analyze metrics by document type."""
    
    # Load fingerprints
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Group by type
    by_type = defaultdict(list)
    for row in data:
        doc_type = row['document_type']
        by_type[doc_type].append(row)
    
    # Analyze each type
    results = {}
    
    for doc_type, rows in by_type.items():
        # Calculate metrics
        pc_scores = [safe_float(r['avg_pc_score']) for r in rows if safe_float(r['avg_pc_score']) > 0]
        noise_ratios = [safe_float(r['noise_ratio']) for r in rows]
        hyphen_ratios = [safe_float(r['hyphen_ratio']) for r in rows]
        blank_ratios = [safe_float(r['blank_page_ratio']) for r in rows]
        page_word_p50s = [safe_float(r['page_word_p50']) for r in rows if safe_float(r['page_word_p50']) > 0]
        
        total_volumes = len(rows)
        total_pages = sum(safe_int(r['xml_file_count']) for r in rows)
        
        # Quality tiers
        high_quality = sum(1 for pc in pc_scores if pc >= 0.8)
        medium_quality = sum(1 for pc in pc_scores if 0.5 <= pc < 0.8)
        low_quality = sum(1 for pc in pc_scores if pc < 0.5)
        
        results[doc_type] = {
            'total_volumes': total_volumes,
            'total_pages': total_pages,
            'avg_pc_score': statistics.mean(pc_scores) if pc_scores else 0,
            'median_pc_score': statistics.median(pc_scores) if pc_scores else 0,
            'avg_noise_ratio': statistics.mean(noise_ratios) if noise_ratios else 0,
            'avg_hyphen_ratio': statistics.mean(hyphen_ratios) if hyphen_ratios else 0,
            'avg_blank_ratio': statistics.mean(blank_ratios) if blank_ratios else 0,
            'median_page_words': statistics.median(page_word_p50s) if page_word_p50s else 0,
            'high_quality_pct': high_quality / len(pc_scores) * 100 if pc_scores else 0,
            'medium_quality_pct': medium_quality / len(pc_scores) * 100 if pc_scores else 0,
            'low_quality_pct': low_quality / len(pc_scores) * 100 if pc_scores else 0,
            'rows': rows
        }
    
    return results

def get_text_examples(doc_type_rows, data_dir, num_examples=2):
    """Get text examples from volumes of this type."""
    
    # Filter high-quality volumes
    high_quality = [r for r in doc_type_rows 
                   if safe_float(r['avg_pc_score']) >= 0.8 
                   and safe_int(r['xml_file_count']) > 0]
    
    if not high_quality:
        high_quality = doc_type_rows  # Fallback to any volume
    
    # Randomly sample volumes
    samples = random.sample(high_quality, min(num_examples, len(high_quality)))
    
    examples = []
    for sample in samples:
        volume_id = sample['volume_id']
        volume_title = sample['volume_title']
        pc_score = safe_float(sample['avg_pc_score'])
        
        # Find XML files for this volume
        volume_dir = Path(data_dir) / volume_id
        if volume_dir.exists():
            xml_files = sorted(volume_dir.glob('*.xml'))
            if xml_files:
                # Get text from first non-empty file
                text_sample = None
                for xml_file in xml_files[:5]:  # Check first 5 files
                    text = extract_text_sample(xml_file)
                    if text and text != "[No text found]":
                        text_sample = text
                        break
                
                if not text_sample:
                    text_sample = "[No text in first 5 pages]"
                
                examples.append({
                    'volume_id': volume_id,
                    'title': volume_title,
                    'pc_score': pc_score,
                    'text': text_sample
                })
    
    return examples

def print_report(results, data_dir):
    """Print comprehensive selection report."""
    
    print("\n" + "="*100)
    print("📊 DOCUMENT TYPE ANALYSIS FOR SUBSET SELECTION")
    print("="*100)
    
    # Sort by average PC score (quality)
    sorted_types = sorted(results.items(), 
                         key=lambda x: x[1]['avg_pc_score'], 
                         reverse=True)
    
    for doc_type, metrics in sorted_types:
        print(f"\n{'='*100}")
        print(f"📁 Document Type: {doc_type}")
        print(f"{'='*100}")
        
        # Basic stats
        print(f"\n📊 Dataset Size:")
        print(f"   Volumes: {metrics['total_volumes']:,}")
        print(f"   Pages: {metrics['total_pages']:,}")
        print(f"   Avg pages/volume: {metrics['total_pages']/metrics['total_volumes']:.0f}")
        
        # Quality metrics
        print(f"\n✨ OCR Quality:")
        print(f"   Average PC Score: {metrics['avg_pc_score']:.3f}")
        print(f"   Median PC Score: {metrics['median_pc_score']:.3f}")
        print(f"   High Quality (≥0.8): {metrics['high_quality_pct']:.1f}%")
        print(f"   Medium Quality (0.5-0.8): {metrics['medium_quality_pct']:.1f}%")
        print(f"   Low Quality (<0.5): {metrics['low_quality_pct']:.1f}%")
        
        # Text quality
        print(f"\n📝 Text Quality:")
        print(f"   Noise Ratio: {metrics['avg_noise_ratio']:.4f} ({metrics['avg_noise_ratio']*100:.2f}%)")
        print(f"   Hyphen Ratio: {metrics['avg_hyphen_ratio']:.4f} ({metrics['avg_hyphen_ratio']*100:.2f}%)")
        print(f"   Blank Page Ratio: {metrics['avg_blank_ratio']:.3f} ({metrics['avg_blank_ratio']*100:.1f}%)")
        
        # Length
        print(f"\n📏 Document Length:")
        print(f"   Median words/page: {metrics['median_page_words']:.0f}")
        print(f"   Est. tokens/page: {metrics['median_page_words']*0.75:.0f} (assuming 0.75 tokens/word)")
        
        # Recommendation
        quality_score = metrics['avg_pc_score']
        if quality_score >= 0.9:
            recommendation = "⭐⭐⭐ EXCELLENT - Highly recommended for pipeline"
        elif quality_score >= 0.8:
            recommendation = "⭐⭐ GOOD - Suitable for pipeline"
        elif quality_score >= 0.6:
            recommendation = "⭐ FAIR - May need additional filtering"
        else:
            recommendation = "⚠️  POOR - Not recommended without heavy preprocessing"
        
        print(f"\n💡 Recommendation: {recommendation}")
        
        # Text examples
        print(f"\n📖 Text Examples (2 random high-quality volumes):")
        examples = get_text_examples(metrics['rows'], data_dir)
        
        for i, example in enumerate(examples, 1):
            print(f"\n   Example {i}:")
            print(f"   Volume: {example['volume_id']} - {example['title'][:80]}")
            print(f"   PC Score: {example['pc_score']:.3f}")
            print(f"   Text: {example['text'][:400]}")
            if len(example['text']) > 400:
                print(f"         ...")
        
        if not examples:
            print("   [No examples available - data directory may be missing]")
    
    # Summary recommendation
    print(f"\n{'='*100}")
    print("🎯 SUMMARY RECOMMENDATIONS")
    print(f"{'='*100}")
    
    # Recommend top 3 by quality
    top_3 = sorted_types[:3]
    print(f"\nTop 3 highest quality document types:")
    for i, (doc_type, metrics) in enumerate(top_3, 1):
        print(f"   {i}. {doc_type:20} - PC Score: {metrics['avg_pc_score']:.3f}, "
              f"{metrics['total_volumes']:,} volumes, "
              f"{metrics['total_pages']:,} pages")
    
    print(f"\n💡 For pipeline development, consider:")
    print(f"   1. Start with highest quality types (PC > 0.9) for initial testing")
    print(f"   2. Include medium quality (PC 0.7-0.9) for robustness testing")
    print(f"   3. Balance volume count vs quality (need enough data)")
    print(f"   4. Consider document length for chunking strategy")
    print()

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 subset_selector.py <fingerprints.csv> <data_dir>")
        print("\nExample:")
        print("  python3 subset_selector.py comprehensive_volume_fingerprints.csv ./30002021")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    data_dir = sys.argv[2]
    
    print(f"\n🔍 Analyzing document types from {csv_file}...")
    print(f"📂 Reading text examples from {data_dir}")
    
    results = analyze_by_type(csv_file, data_dir)
    print_report(results, data_dir)

if __name__ == '__main__':
    main()
