#!/usr/bin/env python3
"""
Create visualization plots from comprehensive volume fingerprints.
Generates publication-quality figures for domain adaptation thesis.

Usage:
    python3 visualize_fingerprints.py comprehensive_volume_fingerprints.csv
"""

import csv
import sys
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

def load_data(csv_file):
    """Load fingerprints CSV."""
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def convert_numeric(row, key):
    """Safely convert to numeric."""
    try:
        val = row[key]
        if not val or val == '':
            return 0
        return float(val)
    except (ValueError, KeyError):
        return 0

def plot_document_type_distribution(data, output_dir):
    """Bar chart of document type counts."""
    type_counts = defaultdict(int)
    for row in data:
        type_counts[row['document_type']] += 1
    
    # Sort by count descending
    types = sorted(type_counts.items(), key=lambda x: -x[1])
    labels = [t[0] for t in types]
    counts = [t[1] for t in types]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, counts, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Highlight Reports
    for i, label in enumerate(labels):
        if label == 'Reports':
            bars[i].set_color('crimson')
            bars[i].set_edgecolor('darkred')
            bars[i].set_linewidth(1.5)
    
    ax.set_xlabel('Number of Volumes')
    ax.set_title('Document Type Distribution in Archive (N=3,315 volumes)')
    ax.invert_yaxis()
    
    # Add count labels
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax.text(count + 20, i, f'{count} ({count/sum(counts)*100:.1f}%)', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'document_type_distribution.png', bbox_inches='tight')
    print(f"   ✓ Saved: document_type_distribution.png")
    plt.close()

def plot_reports_temporal(data, output_dir):
    """Timeline of Police Reports with page counts."""
    reports = [row for row in data if row['document_type'] == 'Reports']
    reports.sort(key=lambda x: int(x['year']) if x['year'] else 0)
    
    years = [int(r['year']) for r in reports if r['year']]
    pages = [convert_numeric(r, 'xml_file_count') for r in reports if r['year']]
    words = [convert_numeric(r, 'actual_total_words') for r in reports if r['year']]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Page counts
    ax1.bar(years, pages, color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel('Number of Pages')
    ax1.set_title('Police Reports Archive: Temporal Distribution (1868-1900)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add temporal split line
    mid_year = years[len(years)//2]
    ax1.axvline(mid_year - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(mid_year - 7, max(pages) * 0.9, 'Suggested\nTrain/Test\nSplit', 
             color='red', fontweight='bold', ha='center')
    
    # Word counts
    ax2.bar(years, words, color='darkgreen', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Total Words')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axvline(mid_year - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Format x-axis
    ax2.set_xticks(years[::2])  # Show every other year
    ax2.set_xticklabels(years[::2], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reports_temporal_distribution.png', bbox_inches='tight')
    print(f"   ✓ Saved: reports_temporal_distribution.png")
    plt.close()

def plot_page_length_boxplot(data, output_dir):
    """Box plots of page length distributions by document type."""
    by_type = defaultdict(list)
    
    for row in data:
        doc_type = row['document_type']
        p50 = convert_numeric(row, 'page_word_p50')
        if p50 > 0:
            by_type[doc_type].append(p50)
    
    # Sort by median length
    types_sorted = sorted(by_type.keys(), 
                          key=lambda x: statistics.median(by_type[x]) if by_type[x] else 0,
                          reverse=True)
    
    # Prepare data
    data_to_plot = [by_type[t] for t in types_sorted]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bp = ax.boxplot(data_to_plot, labels=types_sorted, patch_artist=True,
                    showfliers=True, whis=1.5)
    
    # Color boxes
    colors = ['crimson' if t == 'Reports' else 'steelblue' for t in types_sorted]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add 512-token threshold line (assuming 1 word ≈ 1.3 tokens)
    threshold_words = 512 / 1.3
    ax.axhline(threshold_words, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(len(types_sorted) - 0.5, threshold_words + 20, 
            '512-token threshold (~394 words)', 
            color='red', fontweight='bold', ha='right')
    
    ax.set_ylabel('Median Page Length (words)')
    ax.set_title('Page Length Distribution by Document Type (Median per volume)')
    ax.set_xticklabels(types_sorted, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'page_length_distribution.png', bbox_inches='tight')
    print(f"   ✓ Saved: page_length_distribution.png")
    plt.close()

def plot_quality_distribution(data, output_dir):
    """Histogram of OCR quality (PC scores)."""
    all_scores = [convert_numeric(row, 'avg_pc_score') for row in data 
                  if convert_numeric(row, 'avg_pc_score') > 0]
    
    reports_scores = [convert_numeric(row, 'avg_pc_score') for row in data 
                      if row['document_type'] == 'Reports' and convert_numeric(row, 'avg_pc_score') > 0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Overall distribution
    ax.hist(all_scores, bins=50, alpha=0.6, color='steelblue', 
            edgecolor='black', linewidth=0.5, label='All Documents')
    
    # Reports overlay
    if reports_scores:
        ax.hist(reports_scores, bins=20, alpha=0.8, color='crimson', 
                edgecolor='darkred', linewidth=1, label='Police Reports')
    
    # Quality thresholds
    ax.axvline(0.8, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(0.82, ax.get_ylim()[1] * 0.9, 'High Quality\n(PC > 0.8)', 
            color='green', fontweight='bold')
    
    ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(0.52, ax.get_ylim()[1] * 0.8, 'Medium Quality\n(PC > 0.5)', 
            color='orange', fontweight='bold')
    
    ax.set_xlabel('Average PC Score (OCR Confidence)')
    ax.set_ylabel('Number of Volumes')
    ax.set_title('OCR Quality Distribution Across Archive')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ocr_quality_distribution.png', bbox_inches='tight')
    print(f"   ✓ Saved: ocr_quality_distribution.png")
    plt.close()

def plot_quality_vs_length(data, output_dir):
    """Scatter plot: OCR quality vs page length."""
    # Prepare data by document type
    by_type = defaultdict(lambda: {'pc': [], 'length': []})
    
    for row in data:
        doc_type = row['document_type']
        pc = convert_numeric(row, 'avg_pc_score')
        length = convert_numeric(row, 'page_word_p50')
        
        if pc > 0 and length > 0:
            by_type[doc_type]['pc'].append(pc)
            by_type[doc_type]['length'].append(length)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each type
    colors = {'Reports': 'crimson', 'Court_Records': 'steelblue', 
              'Protocols': 'green', 'Other': 'gray'}
    
    for doc_type in ['Reports', 'Court_Records', 'Protocols']:
        if doc_type in by_type:
            ax.scatter(by_type[doc_type]['length'], by_type[doc_type]['pc'], 
                      alpha=0.6, s=50, c=colors.get(doc_type, 'gray'),
                      edgecolors='black', linewidths=0.5,
                      label=f'{doc_type} (n={len(by_type[doc_type]["pc"])})')
    
    # Plot "Other" with smaller size
    if 'Other' in by_type:
        ax.scatter(by_type['Other']['length'], by_type['Other']['pc'],
                  alpha=0.2, s=20, c='lightgray', edgecolors='none',
                  label=f'Other (n={len(by_type["Other"]["pc"])})')
    
    ax.set_xlabel('Median Page Length (words)')
    ax.set_ylabel('Average PC Score (OCR Confidence)')
    ax.set_title('OCR Quality vs Page Length by Document Type')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_vs_length.png', bbox_inches='tight')
    print(f"   ✓ Saved: quality_vs_length.png")
    plt.close()

def plot_linguistic_features(data, output_dir):
    """Violin plots of linguistic features by document type."""
    by_type = defaultdict(lambda: {'hyphen': [], 'noise': []})
    
    for row in data:
        doc_type = row['document_type']
        hyphen = convert_numeric(row, 'hyphen_ratio')
        noise = convert_numeric(row, 'noise_ratio')
        
        if hyphen > 0:
            by_type[doc_type]['hyphen'].append(hyphen * 100)  # Convert to percentage
        if noise > 0:
            by_type[doc_type]['noise'].append(noise * 100)
    
    # Top 5 most common types
    top_types = ['Other', 'Court_Records', 'Protocols', 'Registers', 'Reports']
    top_types = [t for t in top_types if t in by_type]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Hyphen ratios
    hyphen_data = [by_type[t]['hyphen'] for t in top_types]
    parts1 = ax1.violinplot(hyphen_data, positions=range(len(top_types)), 
                            showmeans=True, showmedians=True)
    
    for pc in parts1['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.7)
    
    ax1.set_xticks(range(len(top_types)))
    ax1.set_xticklabels(top_types, rotation=45, ha='right')
    ax1.set_ylabel('Hyphen Ratio (%)')
    ax1.set_title('Historical Hyphenation (¬) by Document Type')
    ax1.grid(axis='y', alpha=0.3)
    
    # Noise ratios
    noise_data = [by_type[t]['noise'] for t in top_types]
    parts2 = ax2.violinplot(noise_data, positions=range(len(top_types)),
                            showmeans=True, showmedians=True)
    
    for pc in parts2['bodies']:
        pc.set_facecolor('darkgreen')
        pc.set_alpha(0.7)
    
    ax2.set_xticks(range(len(top_types)))
    ax2.set_xticklabels(top_types, rotation=45, ha='right')
    ax2.set_ylabel('Noise Ratio (%)')
    ax2.set_title('OCR Noise by Document Type')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'linguistic_features.png', bbox_inches='tight')
    print(f"   ✓ Saved: linguistic_features.png")
    plt.close()

def plot_reports_quality_timeline(data, output_dir):
    """Quality metrics over time for Police Reports."""
    reports = [row for row in data if row['document_type'] == 'Reports']
    reports.sort(key=lambda x: int(x['year']) if x['year'] else 0)
    
    years = [int(r['year']) for r in reports if r['year']]
    pc_scores = [convert_numeric(r, 'avg_pc_score') for r in reports if r['year']]
    hyphen_ratios = [convert_numeric(r, 'hyphen_ratio') * 100 for r in reports if r['year']]
    blank_ratios = [convert_numeric(r, 'blank_page_ratio') * 100 for r in reports if r['year']]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # PC Score
    ax1.plot(years, pc_scores, marker='o', color='steelblue', linewidth=2, markersize=6)
    ax1.fill_between(years, pc_scores, alpha=0.3, color='steelblue')
    ax1.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='High Quality Threshold')
    ax1.set_ylabel('PC Score')
    ax1.set_title('Police Reports Quality Metrics Over Time (1868-1900)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Hyphen Ratio
    ax2.plot(years, hyphen_ratios, marker='s', color='crimson', linewidth=2, markersize=6)
    ax2.fill_between(years, hyphen_ratios, alpha=0.3, color='crimson')
    ax2.set_ylabel('Hyphen Ratio (%)')
    ax2.grid(alpha=0.3)
    
    # Blank Page Ratio
    ax3.plot(years, blank_ratios, marker='^', color='orange', linewidth=2, markersize=6)
    ax3.fill_between(years, blank_ratios, alpha=0.3, color='orange')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Blank Page Ratio (%)')
    ax3.grid(alpha=0.3)
    
    # Format x-axis
    ax3.set_xticks(years[::2])
    ax3.set_xticklabels(years[::2], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reports_quality_timeline.png', bbox_inches='tight')
    print(f"   ✓ Saved: reports_quality_timeline.png")
    plt.close()

def plot_archive_summary(data, output_dir):
    """Multi-panel summary figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Document type pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    type_counts = defaultdict(int)
    for row in data:
        type_counts[row['document_type']] += 1
    
    # Group small categories
    threshold = 50
    main_types = {k: v for k, v in type_counts.items() if v >= threshold}
    other_count = sum(v for k, v in type_counts.items() if v < threshold)
    if other_count > 0:
        main_types['Other (small)'] = other_count
    
    colors_pie = ['crimson' if 'Report' in k else 'steelblue' for k in main_types.keys()]
    ax1.pie(main_types.values(), labels=main_types.keys(), autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    ax1.set_title('Document Types', fontweight='bold')
    
    # 2. Quality histogram
    ax2 = fig.add_subplot(gs[0, 1:])
    all_pc = [convert_numeric(row, 'avg_pc_score') for row in data 
              if convert_numeric(row, 'avg_pc_score') > 0]
    ax2.hist(all_pc, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(0.8, color='green', linestyle='--', linewidth=2)
    ax2.set_xlabel('PC Score')
    ax2.set_ylabel('Count')
    ax2.set_title('OCR Quality Distribution', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Page length by type
    ax3 = fig.add_subplot(gs[1, :])
    by_type_length = defaultdict(list)
    for row in data:
        p50 = convert_numeric(row, 'page_word_p50')
        if p50 > 0:
            by_type_length[row['document_type']].append(p50)
    
    top_5_types = sorted(by_type_length.items(), 
                         key=lambda x: len(x[1]), reverse=True)[:5]
    
    positions = range(len(top_5_types))
    bp = ax3.boxplot([t[1] for t in top_5_types], 
                     labels=[t[0] for t in top_5_types],
                     patch_artist=True, showfliers=False)
    
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    
    ax3.axhline(512/1.3, color='red', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Page Length (words)')
    ax3.set_title('Page Length Distribution (Top 5 Types)', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Archive temporal distribution by type
    ax4 = fig.add_subplot(gs[2, :])
    
    # Get volumes with years
    volumes_with_years = [(row['document_type'], int(row['year'])) 
                          for row in data if row['year'] and row['year'].isdigit()]
    
    # Group by year and type
    year_type_counts = defaultdict(lambda: defaultdict(int))
    for doc_type, year in volumes_with_years:
        year_type_counts[year][doc_type] += 1
    
    # Get sorted years and top 5 types
    all_years = sorted(year_type_counts.keys())
    type_totals = defaultdict(int)
    for year_data in year_type_counts.values():
        for doc_type, count in year_data.items():
            type_totals[doc_type] += count
    
    top_types = sorted(type_totals.items(), key=lambda x: -x[1])[:5]
    type_names = [t[0] for t in top_types]
    
    # Prepare data for stacked bar
    type_data = {t: [] for t in type_names}
    for year in all_years:
        for t in type_names:
            type_data[t].append(year_type_counts[year][t])
    
    # Plot stacked bars
    colors_map = {'Reports': 'crimson', 'Court_Records': 'steelblue', 
                  'Protocols': 'green', 'Other': 'gray', 'Registers': 'orange',
                  'Legal': 'purple', 'District': 'brown', 'Court_Book': 'pink'}
    
    bottom = np.zeros(len(all_years))
    for doc_type in type_names:
        color = colors_map.get(doc_type, 'lightgray')
        ax4.bar(all_years, type_data[doc_type], bottom=bottom, 
               label=doc_type, color=color, edgecolor='black', linewidth=0.3, alpha=0.8)
        bottom += np.array(type_data[doc_type])
    
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Number of Volumes')
    ax4.set_title('Archive Temporal Distribution (Top 5 Document Types)', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    
    # Overall title
    fig.suptitle('Historical Swedish Police Archive: Comprehensive Overview (3,315 volumes)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_dir / 'archive_summary.png', bbox_inches='tight')
    print(f"   ✓ Saved: archive_summary.png")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_fingerprints.py <fingerprints.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📊 Creating visualizations from {csv_file}...")
    print(f"   Output directory: {output_dir}/\n")
    
    data = load_data(csv_file)
    print(f"   Loaded {len(data)} volumes\n")
    
    print("   Generating plots...")
    plot_document_type_distribution(data, output_dir)
    plot_page_length_boxplot(data, output_dir)
    plot_quality_distribution(data, output_dir)
    plot_quality_vs_length(data, output_dir)
    plot_linguistic_features(data, output_dir)
    plot_archive_summary(data, output_dir)
    
    print(f"\n✓ All visualizations saved to {output_dir}/")
    print("\nGenerated files:")
    print("   1. document_type_distribution.png - Bar chart of document types")
    print("   2. page_length_distribution.png - Page length box plots")
    print("   3. ocr_quality_distribution.png - OCR quality histogram")
    print("   4. quality_vs_length.png - Scatter plot quality vs length")
    print("   5. linguistic_features.png - Hyphen/noise violin plots")
    print("   6. archive_summary.png - Multi-panel overview figure\n")

if __name__ == '__main__':
    main()
