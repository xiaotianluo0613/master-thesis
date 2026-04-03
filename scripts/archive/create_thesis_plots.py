#!/usr/bin/env python3
"""
Create 4 core plots for thesis raw dataset description.
Aligned with NotebookLM analysis requirements.

Usage:
    python3 create_thesis_plots.py comprehensive_volume_fingerprints.csv
"""

import csv
import sys
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set academic style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10

def load_data(csv_file):
    """Load comprehensive fingerprints CSV."""
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

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


# ============================================================================
# Plot 1: Scale & Temporal Distribution (Timeline Histogram)
# ============================================================================
def plot_temporal_distribution(data, output_dir):
    """
    Plot 1: 数据规模与时间跨度
    Timeline histogram showing pages per year (1600-1900)
    """
    # Calculate totals
    total_volumes = len(data)
    total_pages = sum(safe_int(row['xml_file_count']) for row in data)
    total_words = sum(safe_float(row['actual_total_words']) for row in data)
    
    print(f"\n📊 Dataset Scale:")
    print(f"   Total Volumes: {total_volumes:,}")
    print(f"   Total Pages: {total_pages:,}")
    print(f"   Total Words: {total_words:,.0f}")
    
    # Prepare year data (filter 1600-1900)
    year_pages = defaultdict(int)
    for row in data:
        year = safe_int(row['year'])
        if 1600 <= year <= 1900:
            pages = safe_int(row['xml_file_count'])
            year_pages[year] += pages
    
    years = sorted(year_pages.keys())
    pages = [year_pages[y] for y in years]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(years, pages, width=1.0, color='steelblue', edgecolor='black', linewidth=0.3, alpha=0.8)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Pages')
    ax.set_title('Temporal Distribution of Archive Pages (1600-1900)')
    ax.grid(axis='y', alpha=0.3)
    
    # Add text annotation
    ax.text(0.02, 0.98, 
            f'Total: {total_volumes:,} volumes, {total_pages:,} pages, {total_words/1e6:.1f}M words',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_temporal_distribution.png', bbox_inches='tight')
    print(f"   ✓ Saved: 1_temporal_distribution.png")
    plt.close()


# ============================================================================
# Plot 2: Document Type Distribution (with explanations)
# ============================================================================
def plot_document_type_distribution(data, output_dir):
    """
    Plot 2: 档案类型构成
    Horizontal bar chart with type explanations
    """
    # Type explanations in Swedish + English
    type_explanations = {
        'Reports': 'Police crime reports (Polisrapporter)',
        'Court_Records': 'Court proceedings (Domstolshandlingar)',
        'Protocols': 'Official protocols (Protokoll)',
        'Court_Book': 'Court record books (Domboken)',
        'Registers': 'Indexes and registers (Register)',
        'Legal': 'Legal documents (Juridiska handlingar)',
        'District': 'District records (Häradshandlingar)',
        'City': 'City documents (Stadsdokument)',
        'Other': 'Miscellaneous documents',
        'Empty_or_Blank': 'Empty/blank pages'
    }
    
    # Calculate stats by type
    type_stats = defaultdict(lambda: {'volumes': 0, 'pages': 0})
    for row in data:
        doc_type = row['document_type']
        type_stats[doc_type]['volumes'] += 1
        type_stats[doc_type]['pages'] += safe_int(row['xml_file_count'])
    
    # Sort by page count
    sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['pages'], reverse=True)
    
    total_pages = sum(s['pages'] for _, s in sorted_types)
    
    # Prepare data
    labels = []
    pages = []
    percentages = []
    
    for doc_type, stats in sorted_types:
        explanation = type_explanations.get(doc_type, doc_type)
        labels.append(f"{doc_type}\n({explanation})")
        pages.append(stats['pages'])
        pct = stats['pages'] / total_pages * 100
        percentages.append(pct)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(labels, pages, color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Color Reports differently
    for i, (doc_type, _) in enumerate(sorted_types):
        if doc_type == 'Reports':
            bars[i].set_color('crimson')
            bars[i].set_alpha(0.9)
    
    # Add percentage labels
    for i, (p, pct) in enumerate(zip(pages, percentages)):
        ax.text(p + max(pages)*0.01, i, f'{pct:.1f}%', 
                va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Number of Pages')
    ax.set_title('Distribution of Document Types in Archive')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_document_type_distribution.png', bbox_inches='tight')
    print(f"   ✓ Saved: 2_document_type_distribution.png")
    plt.close()


# ============================================================================
# Plot 3: OCR Quality & Noise (HTR Confidence Distribution)
# ============================================================================
def plot_ocr_quality_noise(data, output_dir):
    """
    Plot 3: 数据质量与噪音挑战
    HTR confidence distribution with quality thresholds
    """
    # Filter valid PC scores
    valid_pc_scores = [safe_float(row['avg_pc_score']) 
                       for row in data 
                       if safe_float(row['avg_pc_score']) > 0]
    
    blank_ratios = [safe_float(row['blank_page_ratio']) for row in data]
    hyphen_ratios = [safe_float(row['hyphen_ratio']) for row in data]
    
    # Calculate stats
    pc_median = statistics.median(valid_pc_scores) if valid_pc_scores else 0
    pc_mean = statistics.mean(valid_pc_scores) if valid_pc_scores else 0
    high_blank_count = sum(1 for b in blank_ratios if b > 0.1)
    hyphen_mean = statistics.mean([h for h in hyphen_ratios if h > 0])
    
    print(f"\n📊 Quality Metrics:")
    print(f"   PC Score Median: {pc_median:.3f}")
    print(f"   PC Score Mean: {pc_mean:.3f}")
    print(f"   Volumes with >10% blank pages: {high_blank_count}")
    print(f"   Average hyphen ratio: {hyphen_mean:.5f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: HTR Confidence Distribution
    ax1.hist(valid_pc_scores, bins=50, color='steelblue', edgecolor='black', 
             linewidth=0.5, alpha=0.7, density=True)
    
    # Add KDE
    from scipy import stats as sp_stats
    kde = sp_stats.gaussian_kde(valid_pc_scores)
    x_range = np.linspace(0, 1, 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    # Quality thresholds
    ax1.axvline(0.8, color='green', linestyle='--', linewidth=2, alpha=0.7, label='High Quality (0.8)')
    ax1.axvline(0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Medium Quality (0.5)')
    
    ax1.axvline(pc_median, color='red', linestyle='-', linewidth=2, alpha=0.5, label=f'Median ({pc_median:.3f})')
    
    ax1.set_xlabel('HTR Confidence Score (PC)')
    ax1.set_ylabel('Density')
    ax1.set_title('HTR Quality Distribution')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Blank Page Ratio Distribution
    ax2.hist([b*100 for b in blank_ratios if b > 0], bins=30, 
             color='coral', edgecolor='black', linewidth=0.5, alpha=0.7)
    ax2.axvline(10, color='red', linestyle='--', linewidth=2, label='10% Threshold')
    ax2.set_xlabel('Blank Page Ratio (%)')
    ax2.set_ylabel('Number of Volumes')
    ax2.set_title('Blank Page Distribution')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_ocr_quality_noise.png', bbox_inches='tight')
    print(f"   ✓ Saved: 3_ocr_quality_noise.png")
    plt.close()


# ============================================================================
# Plot 4: Page Length Distribution (Chunking Motivation)
# ============================================================================
def plot_page_length_distribution(data, output_dir):
    """
    Plot 4: 页面长度分布与切块动机
    Box plot showing page length by document type with 512-token limit
    """
    # Prepare data by type
    by_type = defaultdict(list)
    for row in data:
        doc_type = row['document_type']
        p50 = safe_float(row['page_word_p50'])
        if p50 > 0:
            by_type[doc_type].append(p50)
    
    # Sort by median length
    type_medians = [(t, statistics.median(vals)) for t, vals in by_type.items()]
    type_medians.sort(key=lambda x: -x[1])
    
    sorted_types = [t[0] for t in type_medians]
    data_to_plot = [by_type[t] for t in sorted_types]
    
    # Calculate Reports median
    reports_median = statistics.median(by_type.get('Reports', [0]))
    print(f"\n📊 Page Length:")
    print(f"   Police Reports median: {reports_median:.0f} words/page")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bp = ax.boxplot(data_to_plot, labels=sorted_types, patch_artist=True,
                    showfliers=False, widths=0.6)
    
    # Color boxes
    for patch, doc_type in zip(bp['boxes'], sorted_types):
        color = 'crimson' if doc_type == 'Reports' else 'steelblue'
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add 512-token limit line (≈380 Swedish words)
    token_limit = 380
    ax.axhline(token_limit, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(len(sorted_types) - 0.5, token_limit + 20, 
            'Approx. 512 Tokens Limit (~380 words)', 
            color='red', fontweight='bold', ha='right', fontsize=10)
    
    ax.set_xlabel('Document Type')
    ax.set_ylabel('Page Word Count (Median per Volume)')
    ax.set_title('Page Length Distribution by Document Type')
    ax.set_xticklabels(sorted_types, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_page_length_distribution.png', bbox_inches='tight')
    print(f"   ✓ Saved: 4_page_length_distribution.png")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 create_thesis_plots.py <fingerprints.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = Path('thesis_plots')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📊 Creating 4 thesis plots from {csv_file}...")
    print(f"   Output directory: {output_dir}/\n")
    
    data = load_data(csv_file)
    print(f"   Loaded {len(data)} volumes\n")
    
    print("   Generating plots...")
    plot_temporal_distribution(data, output_dir)
    plot_document_type_distribution(data, output_dir)
    plot_ocr_quality_noise(data, output_dir)
    plot_page_length_distribution(data, output_dir)
    
    print(f"\n✓ All 4 thesis plots saved to {output_dir}/")
    print("\nGenerated files:")
    print("   1. 1_temporal_distribution.png - Scale & temporal distribution (1600-1900)")
    print("   2. 2_document_type_distribution.png - Document types with explanations")
    print("   3. 3_ocr_quality_noise.png - HTR quality & blank page distribution")
    print("   4. 4_page_length_distribution.png - Page length by type with 512-token limit\n")

if __name__ == '__main__':
    main()
