#!/usr/bin/env python3
"""
Analyze comprehensive volume fingerprints and create visualizations.
Generates insights for domain adaptation thesis work.

Usage:
    python3 analyze_fingerprints.py comprehensive_volume_fingerprints.csv
"""

import csv
import sys
from collections import defaultdict
import statistics

def load_data(csv_file):
    """Load fingerprints CSV into memory."""
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def convert_numeric(row, key):
    """Safely convert string to numeric, return 0 if empty/invalid."""
    try:
        val = row[key]
        if not val or val == '':
            return 0
        return float(val)
    except (ValueError, KeyError):
        return 0

def analyze_by_document_type(data):
    """Analyze metrics grouped by document type."""
    by_type = defaultdict(list)
    
    for row in data:
        doc_type = row['document_type']
        by_type[doc_type].append(row)
    
    print("\n" + "="*80)
    print("DOCUMENT TYPE ANALYSIS")
    print("="*80)
    
    results = []
    for doc_type in sorted(by_type.keys(), key=lambda x: -len(by_type[x])):
        volumes = by_type[doc_type]
        count = len(volumes)
        
        # Calculate statistics
        total_pages = sum(convert_numeric(v, 'xml_file_count') for v in volumes)
        avg_pages = total_pages / count if count > 0 else 0
        
        word_counts = [convert_numeric(v, 'actual_total_words') for v in volumes]
        avg_words_per_vol = statistics.mean(word_counts) if word_counts else 0
        
        page_medians = [convert_numeric(v, 'page_word_p50') for v in volumes if convert_numeric(v, 'page_word_p50') > 0]
        avg_page_median = statistics.mean(page_medians) if page_medians else 0
        
        pc_scores = [convert_numeric(v, 'avg_pc_score') for v in volumes if convert_numeric(v, 'avg_pc_score') > 0]
        avg_pc = statistics.mean(pc_scores) if pc_scores else 0
        
        hyphen_ratios = [convert_numeric(v, 'hyphen_ratio') for v in volumes if convert_numeric(v, 'hyphen_ratio') > 0]
        avg_hyphen = statistics.mean(hyphen_ratios) if hyphen_ratios else 0
        
        blank_ratios = [convert_numeric(v, 'blank_page_ratio') for v in volumes]
        avg_blank = statistics.mean(blank_ratios) if blank_ratios else 0
        
        results.append({
            'type': doc_type,
            'count': count,
            'total_pages': total_pages,
            'avg_pages': avg_pages,
            'avg_words_per_vol': avg_words_per_vol,
            'avg_page_median_words': avg_page_median,
            'avg_pc_score': avg_pc,
            'avg_hyphen_ratio': avg_hyphen,
            'avg_blank_ratio': avg_blank
        })
    
    # Print summary table
    print(f"\n{'Type':<20} {'Vols':>6} {'Pages':>8} {'Avg Pages':>10} {'Med Page Words':>15} {'PC Score':>10} {'Hyphen':>8} {'Blank%':>8}")
    print("-"*100)
    
    for r in results:
        print(f"{r['type']:<20} {r['count']:>6} {r['total_pages']:>8} {r['avg_pages']:>10.1f} {r['avg_page_median_words']:>15.0f} {r['avg_pc_score']:>10.3f} {r['avg_hyphen_ratio']:>8.4f} {r['avg_blank_ratio']:>8.2%}")
    
    return results

def analyze_reports_temporal(data):
    """Analyze Police Reports by year."""
    reports = [row for row in data if row['document_type'] == 'Reports']
    
    if not reports:
        print("\n⚠ No Police Reports found")
        return
    
    print("\n" + "="*80)
    print(f"POLICE REPORTS TEMPORAL ANALYSIS ({len(reports)} volumes)")
    print("="*80)
    
    by_year = defaultdict(list)
    for r in reports:
        year = r.get('year', '')
        if year:
            by_year[int(year)].append(r)
    
    print(f"\n{'Year':>6} {'Vols':>6} {'Avg Pages':>10} {'Avg Words':>12} {'Med Page Words':>15} {'PC Score':>10} {'Hyphen':>8}")
    print("-"*80)
    
    for year in sorted(by_year.keys()):
        vols = by_year[year]
        count = len(vols)
        
        avg_pages = statistics.mean([convert_numeric(v, 'xml_file_count') for v in vols])
        avg_words = statistics.mean([convert_numeric(v, 'actual_total_words') for v in vols])
        med_page_words = statistics.mean([convert_numeric(v, 'page_word_p50') for v in vols])
        pc_scores = [convert_numeric(v, 'avg_pc_score') for v in vols if convert_numeric(v, 'avg_pc_score') > 0]
        avg_pc = statistics.mean(pc_scores) if pc_scores else 0
        avg_hyphen = statistics.mean([convert_numeric(v, 'hyphen_ratio') for v in vols])
        
        print(f"{year:>6} {count:>6} {avg_pages:>10.0f} {avg_words:>12.0f} {med_page_words:>15.0f} {avg_pc:>10.3f} {avg_hyphen:>8.4f}")
    
    # Suggest train/test split
    years = sorted(by_year.keys())
    if len(years) > 1:
        mid_year = years[len(years)//2]
        train_vols = sum(len(by_year[y]) for y in years if y < mid_year)
        test_vols = sum(len(by_year[y]) for y in years if y >= mid_year)
        
        print(f"\n💡 Suggested temporal split:")
        print(f"   Train: {years[0]}-{mid_year-1} ({train_vols} volumes)")
        print(f"   Test:  {mid_year}-{years[-1]} ({test_vols} volumes)")

def analyze_length_distributions(data):
    """Analyze page length distributions across document types."""
    print("\n" + "="*80)
    print("PAGE LENGTH DISTRIBUTION ANALYSIS")
    print("="*80)
    
    by_type = defaultdict(list)
    for row in data:
        doc_type = row['document_type']
        p50 = convert_numeric(row, 'page_word_p50')
        p95 = convert_numeric(row, 'page_word_p95')
        if p50 > 0:
            by_type[doc_type].append({'p50': p50, 'p95': p95})
    
    print(f"\n{'Type':<20} {'Vols':>6} {'P50 Mean':>10} {'P50 StdDev':>12} {'P95 Mean':>10} {'P95 StdDev':>12} {'Fit 512?':<10}")
    print("-"*95)
    
    for doc_type in sorted(by_type.keys(), key=lambda x: -len(by_type[x])):
        vols = by_type[doc_type]
        p50s = [v['p50'] for v in vols]
        p95s = [v['p95'] for v in vols]
        
        p50_mean = statistics.mean(p50s)
        p50_std = statistics.stdev(p50s) if len(p50s) > 1 else 0
        p95_mean = statistics.mean(p95s)
        p95_std = statistics.stdev(p95s) if len(p95s) > 1 else 0
        
        # Rough estimate: 1 word ≈ 1.3 tokens
        fits_512 = "✓ Yes" if p95_mean * 1.3 < 512 else "✗ No"
        
        print(f"{doc_type:<20} {len(vols):>6} {p50_mean:>10.0f} {p50_std:>12.0f} {p95_mean:>10.0f} {p95_std:>12.0f} {fits_512:<10}")
    
    print("\n💡 Chunking recommendations:")
    print("   - Types with 'Fit 512? ✓ Yes': Use whole-page embeddings")
    print("   - Types with 'Fit 512? ✗ No': Use sliding window or split pages")

def analyze_quality_metrics(data):
    """Analyze OCR quality and text characteristics."""
    print("\n" + "="*80)
    print("QUALITY & LINGUISTIC CHARACTERISTICS")
    print("="*80)
    
    # Overall statistics
    all_pc = [convert_numeric(row, 'avg_pc_score') for row in data if convert_numeric(row, 'avg_pc_score') > 0]
    all_hyphen = [convert_numeric(row, 'hyphen_ratio') for row in data if convert_numeric(row, 'hyphen_ratio') > 0]
    all_noise = [convert_numeric(row, 'noise_ratio') for row in data if convert_numeric(row, 'noise_ratio') > 0]
    
    print(f"\n📊 Overall Archive Statistics:")
    print(f"   PC Score (OCR confidence):  mean={statistics.mean(all_pc):.3f}, median={statistics.median(all_pc):.3f}")
    print(f"   Hyphen ratio (¬):           mean={statistics.mean(all_hyphen):.5f}, median={statistics.median(all_hyphen):.5f}")
    print(f"   Noise ratio:                mean={statistics.mean(all_noise):.5f}, median={statistics.median(all_noise):.5f}")
    
    # Quality distribution
    high_quality = sum(1 for pc in all_pc if pc > 0.8)
    medium_quality = sum(1 for pc in all_pc if 0.5 <= pc <= 0.8)
    low_quality = sum(1 for pc in all_pc if pc < 0.5)
    
    print(f"\n   Quality tiers (by PC score):")
    print(f"   - High (>0.8):    {high_quality:>4} volumes ({high_quality/len(all_pc)*100:.1f}%)")
    print(f"   - Medium (0.5-0.8): {medium_quality:>4} volumes ({medium_quality/len(all_pc)*100:.1f}%)")
    print(f"   - Low (<0.5):     {low_quality:>4} volumes ({low_quality/len(all_pc)*100:.1f}%)")
    
    # Reports vs others
    reports = [row for row in data if row['document_type'] == 'Reports']
    others = [row for row in data if row['document_type'] != 'Reports']
    
    reports_pc = [convert_numeric(r, 'avg_pc_score') for r in reports if convert_numeric(r, 'avg_pc_score') > 0]
    others_pc = [convert_numeric(r, 'avg_pc_score') for r in others if convert_numeric(r, 'avg_pc_score') > 0]
    
    if reports_pc and others_pc:
        print(f"\n   Police Reports vs Other Documents:")
        print(f"   - Reports PC score: {statistics.mean(reports_pc):.3f}")
        print(f"   - Others PC score:  {statistics.mean(others_pc):.3f}")
        print(f"   - Difference:       {abs(statistics.mean(reports_pc) - statistics.mean(others_pc)):.3f}")

def create_training_recommendations(data):
    """Generate recommendations for training set design."""
    print("\n" + "="*80)
    print("TRAINING SET RECOMMENDATIONS")
    print("="*80)
    
    reports = [row for row in data if row['document_type'] == 'Reports']
    
    if not reports:
        print("\n⚠ No Police Reports found for recommendations")
        return
    
    print(f"\n📚 Police Reports Dataset: {len(reports)} volumes")
    
    total_pages = sum(convert_numeric(r, 'xml_file_count') for r in reports)
    total_words = sum(convert_numeric(r, 'actual_total_words') for r in reports)
    
    print(f"   Total pages: {total_pages:,}")
    print(f"   Total words: {total_words:,}")
    print(f"   Avg words/page: {total_words/total_pages:.0f}")
    
    # Quality stratification
    by_quality = defaultdict(list)
    for r in reports:
        pc = convert_numeric(r, 'avg_pc_score')
        if pc > 0.5:
            by_quality['high'].append(r)
        elif pc > 0.2:
            by_quality['medium'].append(r)
        else:
            by_quality['low'].append(r)
    
    print(f"\n   Quality stratification:")
    for quality in ['high', 'medium', 'low']:
        count = len(by_quality[quality])
        if count > 0:
            print(f"   - {quality.capitalize():8} quality: {count:2} volumes")
    
    # Hard negative candidates
    print(f"\n💡 Hard Negative Mining:")
    print(f"   Recommended similar document types for hard negatives:")
    
    # Find document types with similar page lengths
    reports_med_len = statistics.mean([convert_numeric(r, 'page_word_p50') for r in reports])
    
    candidates = []
    by_type = defaultdict(list)
    for row in data:
        if row['document_type'] != 'Reports':
            by_type[row['document_type']].append(row)
    
    for doc_type, vols in by_type.items():
        med_lens = [convert_numeric(v, 'page_word_p50') for v in vols if convert_numeric(v, 'page_word_p50') > 0]
        if med_lens:
            avg_len = statistics.mean(med_lens)
            diff = abs(avg_len - reports_med_len)
            if diff < 100:  # Similar length
                candidates.append((doc_type, len(vols), avg_len, diff))
    
    candidates.sort(key=lambda x: x[3])
    for doc_type, count, avg_len, diff in candidates[:3]:
        print(f"   - {doc_type:20} ({count:3} vols, avg page length: {avg_len:.0f} words, diff: {diff:.0f})")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_fingerprints.py <fingerprints.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    print(f"\n🔍 Loading data from {csv_file}...")
    data = load_data(csv_file)
    print(f"   Loaded {len(data)} volumes")
    
    # Run all analyses
    analyze_by_document_type(data)
    analyze_reports_temporal(data)
    analyze_length_distributions(data)
    analyze_quality_metrics(data)
    create_training_recommendations(data)
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
