#!/usr/bin/env python3
"""
Temporal Quality Analyzer: Analyze which document types in which time periods
have the best quality metrics.

Usage:
    python3 temporal_quality_analyzer.py comprehensive_volume_fingerprints.csv
"""

import csv
import sys
from collections import defaultdict
import statistics

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

def get_period(year):
    """Convert year to period (e.g., 1850s, 1860s)."""
    if year < 1600 or year > 1900:
        return "Invalid"
    decade = (year // 10) * 10
    return f"{decade}s"

def analyze_temporal_quality(csv_file):
    """Analyze quality metrics by document type and time period."""
    
    # Load data
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Filter valid years (1600-1900)
    data = [row for row in data if 1600 <= safe_int(row['year']) <= 1900]
    
    # Group by type and period
    type_period_data = defaultdict(lambda: defaultdict(list))
    
    for row in data:
        doc_type = row['document_type']
        year = safe_int(row['year'])
        period = get_period(year)
        
        if period != "Invalid":
            type_period_data[doc_type][period].append(row)
    
    # Calculate metrics for each type-period combination
    results = []
    
    for doc_type, periods in type_period_data.items():
        for period, rows in periods.items():
            if len(rows) < 3:  # Skip periods with too few samples
                continue
            
            # Calculate metrics
            pc_scores = [safe_float(r['avg_pc_score']) for r in rows if safe_float(r['avg_pc_score']) > 0]
            noise_ratios = [safe_float(r['noise_ratio']) for r in rows]
            hyphen_ratios = [safe_float(r['hyphen_ratio']) for r in rows]
            blank_ratios = [safe_float(r['blank_page_ratio']) for r in rows]
            
            if not pc_scores:
                continue
            
            avg_pc = statistics.mean(pc_scores)
            avg_noise = statistics.mean(noise_ratios)
            avg_hyphen = statistics.mean(hyphen_ratios)
            avg_blank = statistics.mean(blank_ratios)
            
            # Quality score (higher is better)
            # PC score is positive, others are negative
            quality_score = avg_pc - (avg_noise * 10) - (avg_blank * 2)
            
            total_volumes = len(rows)
            total_pages = sum(safe_int(r['xml_file_count']) for r in rows)
            
            results.append({
                'doc_type': doc_type,
                'period': period,
                'volumes': total_volumes,
                'pages': total_pages,
                'avg_pc_score': avg_pc,
                'avg_noise_ratio': avg_noise,
                'avg_hyphen_ratio': avg_hyphen,
                'avg_blank_ratio': avg_blank,
                'quality_score': quality_score,
                'rows': rows
            })
    
    return results

def print_analysis(results):
    """Print comprehensive temporal quality analysis."""
    
    print("\n" + "="*120)
    print("📊 TEMPORAL QUALITY ANALYSIS: Document Type Performance by Time Period")
    print("="*120)
    
    # Overall best combinations
    print("\n🏆 TOP 20 BEST TYPE-PERIOD COMBINATIONS (by Quality Score)")
    print("="*120)
    print(f"{'Rank':<6}{'Type':<20}{'Period':<10}{'Volumes':<10}{'Pages':<12}"
          f"{'PC Score':<12}{'Noise%':<10}{'Blank%':<10}{'Quality':<10}")
    print("-"*120)
    
    sorted_results = sorted(results, key=lambda x: x['quality_score'], reverse=True)
    
    for i, r in enumerate(sorted_results[:20], 1):
        print(f"{i:<6}{r['doc_type']:<20}{r['period']:<10}{r['volumes']:<10}{r['pages']:<12}"
              f"{r['avg_pc_score']:<12.3f}{r['avg_noise_ratio']*100:<10.2f}"
              f"{r['avg_blank_ratio']*100:<10.1f}{r['quality_score']:<10.3f}")
    
    # Analysis by document type
    print(f"\n{'='*120}")
    print("📁 ANALYSIS BY DOCUMENT TYPE")
    print("="*120)
    
    # Group by type
    by_type = defaultdict(list)
    for r in results:
        by_type[r['doc_type']].append(r)
    
    # Sort types by average quality
    type_avg_quality = []
    for doc_type, type_results in by_type.items():
        avg_quality = statistics.mean([r['quality_score'] for r in type_results])
        type_avg_quality.append((doc_type, avg_quality, type_results))
    
    type_avg_quality.sort(key=lambda x: x[1], reverse=True)
    
    for doc_type, avg_quality, type_results in type_avg_quality:
        print(f"\n{'─'*120}")
        print(f"📄 {doc_type} (Avg Quality: {avg_quality:.3f})")
        print(f"{'─'*120}")
        
        # Sort periods within type
        type_results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Show top 5 periods
        print(f"\n  {'Period':<12}{'Volumes':<10}{'Pages':<12}{'PC Score':<12}"
              f"{'Noise%':<10}{'Blank%':<10}{'Quality':<10}{'★'}")
        print(f"  {'-'*110}")
        
        for r in type_results[:5]:
            stars = '★★★' if r['quality_score'] > 0.95 else '★★' if r['quality_score'] > 0.90 else '★'
            print(f"  {r['period']:<12}{r['volumes']:<10}{r['pages']:<12}"
                  f"{r['avg_pc_score']:<12.3f}{r['avg_noise_ratio']*100:<10.2f}"
                  f"{r['avg_blank_ratio']*100:<10.1f}{r['quality_score']:<10.3f}{stars}")
        
        if len(type_results) > 5:
            print(f"  ... ({len(type_results)-5} more periods)")
        
        # Best period for this type
        best = type_results[0]
        print(f"\n  💡 Best period: {best['period']} "
              f"({best['volumes']} vols, PC={best['avg_pc_score']:.3f})")
    
    # Temporal trends
    print(f"\n{'='*120}")
    print("📈 TEMPORAL TRENDS: Quality Over Time")
    print("="*120)
    
    # Group by period across all types
    by_period = defaultdict(list)
    for r in results:
        by_period[r['period']].append(r)
    
    # Calculate average quality per period
    period_stats = []
    for period, period_results in by_period.items():
        avg_pc = statistics.mean([r['avg_pc_score'] for r in period_results])
        avg_noise = statistics.mean([r['avg_noise_ratio'] for r in period_results])
        avg_blank = statistics.mean([r['avg_blank_ratio'] for r in period_results])
        total_volumes = sum([r['volumes'] for r in period_results])
        
        period_stats.append({
            'period': period,
            'avg_pc': avg_pc,
            'avg_noise': avg_noise,
            'avg_blank': avg_blank,
            'total_volumes': total_volumes
        })
    
    # Sort by period
    period_stats.sort(key=lambda x: x['period'])
    
    print(f"\n{'Period':<12}{'Volumes':<12}{'Avg PC Score':<15}{'Avg Noise%':<15}{'Avg Blank%':<15}{'Trend'}")
    print("-"*120)
    
    for ps in period_stats:
        trend = '📈' if ps['avg_pc'] > 0.95 else '➡️' if ps['avg_pc'] > 0.90 else '📉'
        print(f"{ps['period']:<12}{ps['total_volumes']:<12}{ps['avg_pc']:<15.3f}"
              f"{ps['avg_noise']*100:<15.2f}{ps['avg_blank']*100:<15.1f}{trend}")
    
    # Recommendations
    print(f"\n{'='*120}")
    print("🎯 RECOMMENDATIONS FOR SUBSET SELECTION")
    print("="*120)
    
    # Find best combinations with substantial data
    substantial = [r for r in sorted_results if r['volumes'] >= 5 and r['pages'] >= 1000]
    
    print(f"\n✅ Best Type-Period Combinations (≥5 volumes, ≥1000 pages):\n")
    
    for i, r in enumerate(substantial[:10], 1):
        print(f"   {i}. {r['doc_type']:20} {r['period']:10} - "
              f"{r['volumes']:3} vols, {r['pages']:6} pages, "
              f"PC={r['avg_pc_score']:.3f}, Quality={r['quality_score']:.3f}")
    
    print(f"\n💡 Strategy Suggestions:")
    print(f"   1. For HIGH QUALITY training: Use top-ranked combinations (Quality > 0.95)")
    print(f"   2. For TEMPORAL ROBUSTNESS: Sample from multiple periods within same type")
    print(f"   3. For DOMAIN ADAPTATION: Mix high-quality + medium-quality periods")
    print(f"   4. Police Reports best period: ", end="")
    
    # Find best Reports period
    reports = [r for r in results if r['doc_type'] == 'Reports']
    if reports:
        reports.sort(key=lambda x: x['quality_score'], reverse=True)
        best_reports = reports[0]
        print(f"{best_reports['period']} ({best_reports['volumes']} vols, PC={best_reports['avg_pc_score']:.3f})")
    else:
        print("No data")
    
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 temporal_quality_analyzer.py <fingerprints.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    print(f"\n🔍 Analyzing temporal quality patterns from {csv_file}...")
    
    results = analyze_temporal_quality(csv_file)
    print(f"   Found {len(results)} type-period combinations\n")
    
    print_analysis(results)

if __name__ == '__main__':
    main()
