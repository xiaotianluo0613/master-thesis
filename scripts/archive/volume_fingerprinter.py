#!/usr/bin/env python3
"""
Volume Fingerprinter: Global Data Panorama Scanner

Scans a large archive of ALTO XML files organized in subfolders (Volumes)
and extracts "fingerprints" - compact summaries including:
- Title detection (Rapporter, Protocoll, Domböcker)
- Year extraction
- File count and PC scores
- Logic pattern detection (Case Numbers)
- Character count estimation

Designed for memory efficiency using iterparse and sampling techniques.
Safe error handling for corrupt/missing files.
"""

import csv
import os
import sys
import random
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import re

from lxml import etree

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
TITLE_KEYWORDS = {'Rapporter', 'Protocoll', 'Domböcker'}
YEAR_PATTERN = re.compile(r'(?:år\s*)?(\d{4})')
CASE_PATTERN = re.compile(r'No\s+\d+')


class VolumeFingerprinter:
    """Scanner for extracting volume fingerprints from ALTO XML archive."""
    
    def __init__(self, root_path: Path, output_csv: str = "volume_fingerprints_summary.csv"):
        """
        Initialize the fingerprinter.
        
        Args:
            root_path: Root directory containing volume subfolders
            output_csv: Output CSV filename
        """
        self.root_path = Path(root_path)
        self.output_csv = output_csv
        self.results = []
        
    def extract_text_from_alto(self, filepath: Path) -> str:
        """
        Extract text from ALTO XML file using streaming parser.
        Returns limited text for efficiency (first 2000 chars).
        
        Args:
            filepath: Path to ALTO XML file
            
        Returns:
            Extracted text or empty string on error
        """
        try:
            text_parts = []
            context = etree.iterparse(str(filepath), events=('end',), tag=f"{{{ALTO_NS['alto']}}}String")
            
            for event, elem in context:
                content = elem.get('CONTENT', '').strip()
                if content:
                    text_parts.append(content)
                
                # Limit to first 2000 chars for efficiency
                if sum(len(t) for t in text_parts) > 2000:
                    break
                
                # Clear element to free memory
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
            
            return ' '.join(text_parts)
            
        except Exception as e:
            logger.warning(f"Error extracting text from {filepath}: {e}")
            return ""
    
    def get_pc_score(self, filepath: Path) -> Optional[float]:
        """
        Extract PC (Probability of Correctness) score from ALTO file header.
        Uses iterparse to avoid loading full tree.
        
        Args:
            filepath: Path to ALTO XML file
            
        Returns:
            PC score or None on error
        """
        try:
            context = etree.iterparse(str(filepath), events=('start',), tag=f"{{{ALTO_NS['alto']}}}Page")
            
            for event, elem in context:
                pc_value = elem.get('PC')
                if pc_value:
                    try:
                        return float(pc_value)
                    except (ValueError, TypeError):
                        return None
                
                # Clear to free memory
                elem.clear()
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting PC score from {filepath}: {e}")
            return None
    
    def find_title_page(self, volume_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Scan first 3 XML files for title and extract full title text.
        Returns (file_path, full_title, year_extracted, keyword_found).
        
        Args:
            volume_path: Path to volume folder
            
        Returns:
            Tuple of (title_file, full_title, year, keyword) or (None, None, None, None)
        """
        xml_files = sorted(volume_path.glob("*.xml"))[:3]  # First 3 files
        
        for xml_file in xml_files:
            text = self.extract_text_from_alto(xml_file)
            
            # Check for title keywords
            for keyword in TITLE_KEYWORDS:
                if keyword in text:
                    # Extract year
                    year_match = YEAR_PATTERN.search(text)
                    year = year_match.group(1) if year_match else "N/A"
                    
                    # Extract full title - get first 200 characters of text
                    full_title = text[:200].strip()
                    # Replace newlines with spaces for CSV compatibility
                    full_title = ' '.join(full_title.split())
                    
                    return (xml_file.name, full_title, year, keyword)
        
        return (None, None, None, None)
    
    def get_xml_file_count(self, volume_path: Path) -> int:
        """Count total XML files in volume."""
        try:
            return len(list(volume_path.glob("*.xml")))
        except Exception as e:
            logger.warning(f"Error counting XML files in {volume_path}: {e}")
            return 0
    
    def get_average_pc_score(self, volume_path: Path, sample_size: int = 5) -> float:
        """
        Extract average PC score from first N XML files.
        
        Args:
            volume_path: Path to volume folder
            sample_size: Number of files to sample
            
        Returns:
            Average PC score or 0.0 if none found
        """
        xml_files = sorted(volume_path.glob("*.xml"))[:sample_size]
        pc_scores = []
        
        for xml_file in xml_files:
            pc = self.get_pc_score(xml_file)
            if pc is not None:
                pc_scores.append(pc)
        
        return sum(pc_scores) / len(pc_scores) if pc_scores else 0.0
    
    def estimate_total_chars(self, volume_path: Path, sample_size: int = 5) -> int:
        """
        Estimate total characters by sampling and extrapolating.
        
        Args:
            volume_path: Path to volume folder
            sample_size: Number of files to sample
            
        Returns:
            Estimated total characters
        """
        xml_files = sorted(volume_path.glob("*.xml"))
        total_files = len(xml_files)
        
        if total_files == 0:
            return 0
        
        sample_files = random.sample(xml_files, min(sample_size, total_files))
        sample_chars = []
        
        for xml_file in sample_files:
            text = self.extract_text_from_alto(xml_file)
            sample_chars.append(len(text))
        
        if not sample_chars:
            return 0
        
        avg_chars_per_file = sum(sample_chars) / len(sample_chars)
        return int(avg_chars_per_file * total_files)
    
    def get_logic_score(self, volume_path: Path, sample_size: int = 3) -> float:
        """
        Check case number pattern in random sample of files.
        
        Args:
            volume_path: Path to volume folder
            sample_size: Number of random files to check
            
        Returns:
            Logic score: proportion of files with case pattern (0.0-1.0)
        """
        xml_files = list(volume_path.glob("*.xml"))
        
        if not xml_files:
            return 0.0
        
        # Exclude potential title pages (first 3 files)
        non_title_files = xml_files[3:] if len(xml_files) > 3 else xml_files
        
        if not non_title_files:
            return 0.0
        
        sample_files = random.sample(non_title_files, min(sample_size, len(non_title_files)))
        matches = 0
        
        for xml_file in sample_files:
            text = self.extract_text_from_alto(xml_file)
            if CASE_PATTERN.search(text):
                matches += 1
        
        return matches / len(sample_files)
    
    def fingerprint_volume(self, volume_path: Path) -> Optional[Dict]:
        """
        Generate complete fingerprint for a single volume.
        
        Args:
            volume_path: Path to volume folder
            
        Returns:
            Dictionary with fingerprint data or None on error
        """
        try:
            volume_id = volume_path.name
            logger.info(f"Processing volume: {volume_id}")
            
            # Get title page info
            title_file, full_title, year, keyword = self.find_title_page(volume_path)
            
            # Get statistics
            xml_count = self.get_xml_file_count(volume_path)
            avg_pc = self.get_average_pc_score(volume_path)
            logic_score = self.get_logic_score(volume_path)
            est_chars = self.estimate_total_chars(volume_path)
            
            return {
                'volume_id': volume_id,
                'volume_title': full_title or 'N/A',
                'volume_title_keyword': keyword or 'N/A',
                'year': year or 'N/A',
                'title_page_source': title_file or 'N/A',
                'xml_file_count': xml_count,
                'average_pc_score': f"{avg_pc:.3f}",
                'logic_pattern_score': f"{logic_score:.3f}",
                'estimated_total_chars': est_chars
            }
            
        except Exception as e:
            logger.error(f"Error processing volume {volume_path}: {e}")
            return None
    
    def scan_all_volumes(self, root_path: Optional[Path] = None) -> None:
        """
        Scan all volume subfolders in root directory.
        
        Args:
            root_path: Root directory to scan (uses self.root_path if None)
        """
        if root_path is None:
            root_path = self.root_path
        
        if not root_path.exists():
            logger.error(f"Root path does not exist: {root_path}")
            return
        
        # Find all volume subfolders
        volume_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
        
        if not volume_dirs:
            logger.warning(f"No subdirectories found in {root_path}")
            return
        
        logger.info(f"Found {len(volume_dirs)} volume folders to process")
        
        for idx, volume_dir in enumerate(volume_dirs, 1):
            logger.info(f"[{idx}/{len(volume_dirs)}] Processing {volume_dir.name}")
            
            fingerprint = self.fingerprint_volume(volume_dir)
            if fingerprint:
                self.results.append(fingerprint)
        
        logger.info(f"Completed processing {len(self.results)} volumes successfully")
    
    def save_results(self, output_path: Optional[str] = None) -> None:
        """
        Save fingerprints to CSV file.
        
        Args:
            output_path: Output file path (uses self.output_csv if None)
        """
        if output_path is None:
            output_path = self.output_csv
        
        if not self.results:
            logger.warning("No results to save")
            return
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'volume_id',
                    'volume_title',
                    'volume_title_keyword',
                    'year',
                    'title_page_source',
                    'xml_file_count',
                    'average_pc_score',
                    'logic_pattern_score',
                    'estimated_total_chars'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            
            logger.info(f"Results saved to {output_path}")
            logger.info(f"Total volumes processed: {len(self.results)}")
            
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
    
    def print_summary(self) -> None:
        """Print summary statistics of all scanned volumes."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("VOLUME FINGERPRINTS SUMMARY")
        print("="*80)
        print(f"Total volumes scanned: {len(self.results)}")
        print()
        
        # Calculate statistics
        total_files = sum(int(r['xml_file_count']) for r in self.results)
        avg_pc_scores = [float(r['average_pc_score']) for r in self.results]
        logic_scores = [float(r['logic_pattern_score']) for r in self.results]
        total_est_chars = sum(int(r['estimated_total_chars']) for r in self.results)
        
        years = [r['year'] for r in self.results if r['year'] != 'N/A']
        keywords = [r['volume_title_keyword'] for r in self.results if r['volume_title_keyword'] != 'N/A']
        
        print(f"Total estimated characters: {total_est_chars:,}")
        print(f"Total XML files: {total_files:,}")
        print(f"Average PC score: {sum(avg_pc_scores)/len(avg_pc_scores):.3f}")
        print(f"Average logic pattern score: {sum(logic_scores)/len(logic_scores):.3f}")
        print()
        
        if years:
            print(f"Year range: {min(years)} - {max(years)}")
        
        if keywords:
            from collections import Counter
            keyword_counts = Counter(keywords)
            print(f"Title keywords found:")
            for kw, count in keyword_counts.most_common():
                print(f"  - {kw}: {count} volumes")
        
        print("="*80 + "\n")


def main():
    """
    Main function with test mode and batch processing options.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate fingerprints for ALTO XML archive volumes'
    )
    parser.add_argument(
        'root_path',
        help='Root path containing volume subfolders'
    )
    parser.add_argument(
        '--output',
        default='volume_fingerprints_summary.csv',
        help='Output CSV file (default: volume_fingerprints_summary.csv)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: scan only the first volume'
    )
    parser.add_argument(
        '--single',
        help='Process only a single volume folder'
    )
    
    args = parser.parse_args()
    
    root_path = Path(args.root_path)
    
    # Verify root path exists
    if not root_path.exists():
        print(f"Error: Root path does not exist: {root_path}")
        sys.exit(1)
    
    # Initialize fingerprinter
    fingerprinter = VolumeFingerprinter(root_path, args.output)
    
    # Test mode: single volume
    if args.test:
        volume_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
        if volume_dirs:
            print(f"\n=== TEST MODE: Processing single volume ===\n")
            fingerprint = fingerprinter.fingerprint_volume(volume_dirs[0])
            if fingerprint:
                print(f"Fingerprint for {volume_dirs[0].name}:")
                for key, value in fingerprint.items():
                    print(f"  {key}: {value}")
            fingerprinter.results = [fingerprint]
        else:
            print("No volume folders found")
            sys.exit(1)
    
    # Single volume mode
    elif args.single:
        single_path = Path(args.single)
        if not single_path.exists():
            print(f"Error: Path does not exist: {single_path}")
            sys.exit(1)
        
        print(f"\n=== Processing single volume: {single_path.name} ===\n")
        fingerprint = fingerprinter.fingerprint_volume(single_path)
        if fingerprint:
            print(f"Fingerprint for {single_path.name}:")
            for key, value in fingerprint.items():
                print(f"  {key}: {value}")
            fingerprinter.results = [fingerprint]
        else:
            print("Failed to generate fingerprint")
            sys.exit(1)
    
    # Batch mode: all volumes
    else:
        print(f"\n=== Processing all volumes in: {root_path} ===\n")
        fingerprinter.scan_all_volumes()
    
    # Save results
    fingerprinter.save_results()
    
    # Print summary
    fingerprinter.print_summary()


if __name__ == '__main__':
    main()
