#!/usr/bin/env python3
"""
Comprehensive Volume Fingerprinter for Historical Swedish Archives
Extracts detailed metadata from ALTO XML volumes for domain adaptation research.

Usage:
    python3 comprehensive_volume_fingerprinter.py <transcriptions_dir> [--sample N]
    
Example:
    # Full scan (all volumes)
    python3 comprehensive_volume_fingerprinter.py ./Filedrop-7hXHfBBqt2nJHQuk/home/dgxuser/erik/data/transcriptions
    
    # Test on 50 volumes
    python3 comprehensive_volume_fingerprinter.py ./Filedrop-7hXHfBBqt2nJHQuk/home/dgxuser/erik/data/transcriptions --sample 50
"""

import csv
import re
import argparse
from pathlib import Path
from collections import defaultdict
import statistics
import time

# Regex patterns
CONTENT_PATTERN = re.compile(r'CONTENT="([^"]*)"')
PC_PATTERN = re.compile(r'PC="([^"]*)"')
# Historical corpus range guard (avoid OCR artifacts like 2000)
YEAR_PATTERN = re.compile(r'\b(1[6-9]\d{2}|190\d|1910)\b')

# Document type keywords (Swedish)
KEYWORDS = {
    'Rapporter': 'Reports',
    'Rapport': 'Reports',
    'Protocoll': 'Protocols',
    'Protokoll': 'Protocols',
    'Domböcker': 'Court_Book',
    'Dombok': 'Court_Book',
    'Journaler': 'Registers',
    'Register': 'Registers',
    'Index': 'Registers',
    'Härads': 'District',
    'Kommuns': 'City',
    'Städ': 'City',
    'Rätt': 'Court_Records',
    'Domstol': 'Court_Records',
    'Laga': 'Legal',
    'Målbok': 'Law',
}


class VolumeFingerprinter:
    """Extract comprehensive metadata from archive volumes."""
    
    def __init__(self, base_path, sample_size=None):
        self.base_path = Path(base_path)
        self.sample_size = sample_size
        self.results = []
        
    def extract_title_and_type(self, volume_path):
        """Extract volume title and document type from first files with content."""
        xml_files = sorted(volume_path.glob('*.xml'))[:30]
        
        for xml_file in xml_files:
            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                matches = CONTENT_PATTERN.findall(content)
                
                # Look for substantial content (not just metadata)
                if matches and len(matches) > 5:
                    title_text = ' '.join(matches[:50])
                    
                    # Match keywords for document type
                    doc_type = 'Other'
                    for keyword, category in KEYWORDS.items():
                        if keyword.lower() in title_text.lower():
                            doc_type = category
                            break
                    
                    return title_text[:200], doc_type, xml_file.name
                    
            except Exception as e:
                continue
        
        return "Unknown", "Empty_or_Blank", "N/A"
    
    def find_year(self, volume_path):
        """Find year from first files with content."""
        xml_files = sorted(volume_path.glob('*.xml'))[:10]
        
        for xml_file in xml_files:
            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                matches = CONTENT_PATTERN.findall(content)
                text = ' '.join(matches)
                
                years = YEAR_PATTERN.findall(text)
                if years:
                    return int(years[0])
                    
            except Exception:
                continue
        
        return None
    
    def analyze_volume_comprehensive(self, volume_path):
        """Extract comprehensive metrics from all XML files in volume."""
        xml_files = sorted(volume_path.glob('*.xml'))
        
        if not xml_files:
            return None
        
        # Collect data from all pages
        all_pc_scores = []
        page_char_counts = []
        page_word_counts = []
        total_chars = 0
        total_words = 0
        hyphen_count = 0
        noise_char_count = 0
        blank_pages = 0
        
        for xml_file in xml_files:
            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract PC scores
                pc_scores = PC_PATTERN.findall(content)
                for pc_str in pc_scores:
                    try:
                        all_pc_scores.append(float(pc_str))
                    except ValueError:
                        continue
                
                # Extract text content
                matches = CONTENT_PATTERN.findall(content)
                page_text = ' '.join(matches)
                
                # Page-level metrics
                page_chars = len(page_text)
                page_words = len(page_text.split())
                
                page_char_counts.append(page_chars)
                page_word_counts.append(page_words)
                
                if page_chars < 50:
                    blank_pages += 1
                
                # Volume-level accumulation
                total_chars += page_chars
                total_words += page_words
                hyphen_count += page_text.count('¬')
                
                # Noise ratio: characters outside expected Swedish text
                # Keep: letters, numbers, Swedish chars, basic punctuation, hyphen marker
                clean_text = re.sub(r'[a-zA-ZåäöÅÄÖ0-9\s,.!?;:()\-\'\"\¬]', '', page_text)
                noise_char_count += len(clean_text)
                
            except Exception as e:
                # Skip corrupted files
                page_char_counts.append(0)
                page_word_counts.append(0)
                blank_pages += 1
                continue
        
        # Calculate statistics
        metrics = {
            'xml_file_count': len(xml_files),
            'actual_total_chars': total_chars,
            'actual_total_words': total_words,
        }
        
        # PC score statistics
        if all_pc_scores:
            metrics['avg_pc_score'] = round(statistics.mean(all_pc_scores), 4)
            metrics['std_pc_score'] = round(statistics.stdev(all_pc_scores), 4) if len(all_pc_scores) > 1 else 0.0
        else:
            metrics['avg_pc_score'] = 0.0
            metrics['std_pc_score'] = 0.0
        
        # Linguistic markers
        metrics['hyphen_ratio'] = round(hyphen_count / total_chars, 6) if total_chars > 0 else 0.0
        metrics['noise_ratio'] = round(noise_char_count / total_chars, 6) if total_chars > 0 else 0.0
        
        # Page length distributions (percentiles)
        if page_char_counts:
            sorted_chars = sorted(page_char_counts)
            sorted_words = sorted(page_word_counts)
            n = len(sorted_chars)
            
            metrics['page_char_p50'] = sorted_chars[n // 2]
            metrics['page_char_p95'] = sorted_chars[int(n * 0.95)] if n > 1 else sorted_chars[0]
            metrics['page_char_max'] = max(page_char_counts)
            
            metrics['page_word_p50'] = sorted_words[n // 2]
            metrics['page_word_p95'] = sorted_words[int(n * 0.95)] if n > 1 else sorted_words[0]
        else:
            metrics['page_char_p50'] = 0
            metrics['page_char_p95'] = 0
            metrics['page_char_max'] = 0
            metrics['page_word_p50'] = 0
            metrics['page_word_p95'] = 0
        
        # Blank page ratio
        metrics['blank_page_ratio'] = round(blank_pages / len(xml_files), 4) if xml_files else 0.0
        
        return metrics
    
    def process_volume(self, volume_path):
        """Process single volume and extract all metadata."""
        volume_id = volume_path.name
        
        print(f"Processing {volume_id}...")
        
        # Basic metadata
        title, doc_type, title_source = self.extract_title_and_type(volume_path)
        year = self.find_year(volume_path)
        
        # Comprehensive metrics
        metrics = self.analyze_volume_comprehensive(volume_path)
        
        if metrics is None:
            print(f"  Skipping {volume_id} - no XML files")
            return None
        
        # Combine all metadata
        result = {
            'volume_id': volume_id,
            'volume_title': title,
            'document_type': doc_type,
            'year': year if year else '',
            'title_page_source': title_source,
            **metrics
        }
        
        return result
    
    def scan_archive(self):
        """Scan all volumes in archive."""
        print(f"\nScanning archive at: {self.base_path}")
        print(f"Sample size: {'ALL' if self.sample_size is None else self.sample_size}\n")
        
        # Get all volume directories
        volume_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]
        volume_dirs.sort()
        
        if self.sample_size:
            import random
            random.seed(42)  # Reproducible sampling
            volume_dirs = random.sample(volume_dirs, min(self.sample_size, len(volume_dirs)))
            volume_dirs.sort()
        
        print(f"Found {len(volume_dirs)} volumes to process\n")
        
        start_time = time.time()
        
        for i, volume_dir in enumerate(volume_dirs, 1):
            result = self.process_volume(volume_dir)
            
            if result:
                self.results.append(result)
            
            # Progress indicator
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(volume_dirs) - i) / rate
                print(f"  Progress: {i}/{len(volume_dirs)} ({i/len(volume_dirs)*100:.1f}%) - "
                      f"Est. remaining: {remaining/60:.1f} min")
        
        total_time = time.time() - start_time
        print(f"\n✓ Completed {len(self.results)} volumes in {total_time/60:.1f} minutes")
    
    def save_results(self, output_file='comprehensive_volume_fingerprints.csv'):
        """Save results to CSV file."""
        if not self.results:
            print("No results to save")
            return
        
        # Sort by volume_id
        sorted_results = sorted(self.results, key=lambda x: x['volume_id'])
        
        # Define column order
        fieldnames = [
            'volume_id',
            'volume_title',
            'document_type',
            'year',
            'title_page_source',
            'xml_file_count',
            'actual_total_chars',
            'actual_total_words',
            'avg_pc_score',
            'std_pc_score',
            'hyphen_ratio',
            'noise_ratio',
            'page_char_p50',
            'page_char_p95',
            'page_char_max',
            'page_word_p50',
            'page_word_p95',
            'blank_page_ratio'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_results)
        
        print(f"\n✓ Results saved to: {output_file}")
        print(f"  Total volumes: {len(sorted_results)}")
        
        # Summary statistics
        doc_types = defaultdict(int)
        for result in sorted_results:
            doc_types[result['document_type']] += 1
        
        print(f"\n📊 Document Type Distribution:")
        for doc_type, count in sorted(doc_types.items(), key=lambda x: -x[1]):
            print(f"  {doc_type:20} {count:4} ({count/len(sorted_results)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Extract comprehensive metadata from historical archive volumes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full scan
  python3 comprehensive_volume_fingerprinter.py ./transcriptions
  
  # Test on 50 volumes
  python3 comprehensive_volume_fingerprinter.py ./transcriptions --sample 50
  
  # Uppmax usage
  python3 comprehensive_volume_fingerprinter.py /proj/your_project/transcriptions
        """
    )
    parser.add_argument('base_path', help='Path to transcriptions directory')
    parser.add_argument('--sample', type=int, help='Sample N random volumes (for testing)')
    parser.add_argument('--output', default='comprehensive_volume_fingerprints.csv',
                       help='Output CSV file (default: comprehensive_volume_fingerprints.csv)')
    
    args = parser.parse_args()
    
    # Validate path
    base_path = Path(args.base_path)
    if not base_path.exists():
        print(f"❌ Error: Path does not exist: {base_path}")
        return 1
    
    # Run fingerprinting
    fingerprinter = VolumeFingerprinter(base_path, sample_size=args.sample)
    fingerprinter.scan_archive()
    fingerprinter.save_results(args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())
