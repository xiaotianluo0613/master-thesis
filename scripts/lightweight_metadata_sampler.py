#!/usr/bin/env python3
"""
Lightweight Archive Metadata Sampler

Randomly samples 5,000 folders from archive to quickly discover:
- Document types (Rapporter, Protocoll, Domböcker, etc.)
- Title and year metadata
- Sample text from EACH UNIQUE TYPE (not redundant)

Much faster than full scan - perfect for local exploration.
"""

import csv
import os
import sys
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Set, List
import re

from lxml import etree

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
TITLE_KEYWORDS = {'Rapporter', 'Protocoll', 'Domböcker'}
YEAR_PATTERN = re.compile(r'(?:år\s*)?(\d{4})')


class LightweightSampler:
    """Fast metadata sampler for archive exploration."""
    
    def __init__(self, root_path: Path):
        self.root_path = Path(root_path)
        self.results = []
        self.seen_types = set()  # Track unique document types
        self.sample_texts = []   # Store text samples of new types
        
    def extract_text_from_alto(self, filepath: Path, max_chars: int = 3000) -> str:
        """Extract text from ALTO file (limited for speed)."""
        try:
            text_parts = []
            total_chars = 0
            
            for event, elem in etree.iterparse(str(filepath), events=('end',), 
                                              tag=f"{{{ALTO_NS['alto']}}}String"):
                content = elem.get('CONTENT', '').strip()
                if content:
                    text_parts.append(content)
                    total_chars += len(content)
                
                # Stop early if we have enough text
                if total_chars > max_chars:
                    break
                
                # Clean up element to save memory
                elem.clear()
                for ancestor in elem.iterancestors():
                    ancestor.clear()
            
            return ' '.join(text_parts)
            
        except Exception as e:
            logger.debug(f"Error extracting text from {filepath}: {e}")
            return ""
    
    def get_pc_score(self, filepath: Path) -> Optional[float]:
        """Extract PC score from ALTO file."""
        try:
            for event, elem in etree.iterparse(str(filepath), events=('start',), 
                                              tag=f"{{{ALTO_NS['alto']}}}Page"):
                pc_value = elem.get('PC')
                if pc_value:
                    try:
                        result = float(pc_value)
                        elem.clear()
                        return result
                    except (ValueError, TypeError):
                        elem.clear()
                        return None
                
                elem.clear()
            
            return None
            
        except Exception as e:
            logger.debug(f"Error getting PC score from {filepath}: {e}")
            return None
    
    def find_title_and_type(self, volume_path: Path) -> tuple:
        """
        Scan first 3 files to find title, year, and document type.
        Returns: (title, year, document_type)
        """
        xml_files = sorted(volume_path.glob("*.xml"))[:3]
        
        for xml_file in xml_files:
            text = self.extract_text_from_alto(xml_file)
            
            # Check for title keywords
            for keyword in TITLE_KEYWORDS:
                if keyword in text:
                    # Extract year
                    year_match = YEAR_PATTERN.search(text)
                    year = year_match.group(1) if year_match else "N/A"
                    
                    # Extract title (first 150 chars)
                    title = ' '.join(text[:150].split())
                    
                    return (title, year, keyword)
        
        return (None, None, None)
    
    def get_avg_pc_score(self, volume_path: Path) -> float:
        """Average PC score from first 3 files."""
        xml_files = sorted(volume_path.glob("*.xml"))[:3]
        pc_scores = []
        
        for xml_file in xml_files:
            pc = self.get_pc_score(xml_file)
            if pc is not None:
                pc_scores.append(pc)
        
        return sum(pc_scores) / len(pc_scores) if pc_scores else 0.0
    
    def get_xml_count(self, volume_path: Path) -> int:
        """Count XML files in volume."""
        try:
            return len(list(volume_path.glob("*.xml")))
        except Exception as e:
            logger.warning(f"Error counting XMLs in {volume_path}: {e}")
            return 0
    
    def sample_volume(self, volume_path: Path) -> Optional[Dict]:
        """
        Sample a single volume.
        If document type is NEW, also save sample text.
        """
        try:
            volume_id = volume_path.name
            
            # Get metadata
            title, year, doc_type = self.find_title_and_type(volume_path)
            
            if not doc_type:
                doc_type = "Unknown"
            
            xml_count = self.get_xml_count(volume_path)
            avg_pc = self.get_avg_pc_score(volume_path)
            
            result = {
                'volume_id': volume_id,
                'type': doc_type,
                'year': year or 'N/A',
                'xml_count': xml_count,
                'avg_pc_score': f"{avg_pc:.3f}"
            }
            
            # If this is a NEW document type, sample text
            if doc_type not in self.seen_types:
                self.seen_types.add(doc_type)
                
                # Pick random XML file and extract text
                xml_files = list(volume_path.glob("*.xml"))
                if xml_files:
                    sample_file = random.choice(xml_files)
                    sample_text = self.extract_text_from_alto(sample_file, max_chars=2000)
                    
                    if sample_text:
                        self.sample_texts.append({
                            'type': doc_type,
                            'volume_id': volume_id,
                            'year': year or 'N/A',
                            'text': sample_text
                        })
                        logger.info(f"NEW TYPE FOUND: {doc_type} (from {volume_id})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error sampling {volume_path}: {e}")
            return None
    
    def run_sampling(self, sample_count: int = 5000) -> None:
        """Randomly sample N folders from archive."""
        # Get all volume directories
        all_volumes = [d for d in self.root_path.iterdir() if d.is_dir()]
        
        if not all_volumes:
            logger.error("No volume folders found")
            return
        
        # Randomly sample (or use all if less than sample_count)
        sample_size = min(sample_count, len(all_volumes))
        sampled_volumes = random.sample(all_volumes, sample_size)
        
        logger.info(f"Sampling {sample_size} folders from {len(all_volumes)} total")
        
        for idx, volume_path in enumerate(sampled_volumes, 1):
            logger.info(f"[{idx}/{sample_size}] Sampling {volume_path.name}")
            
            result = self.sample_volume(volume_path)
            if result:
                self.results.append(result)
        
        logger.info(f"Completed sampling. Found {len(self.seen_types)} unique types")
    
    def save_results(self, csv_output: str = "local_sample_panorama.csv",
                     text_output: str = "sample_texts.txt") -> None:
        """Save metadata CSV and sample texts."""
        # Save CSV
        if self.results:
            try:
                with open(csv_output, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['volume_id', 'type', 'year', 'xml_count', 'avg_pc_score']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.results)
                logger.info(f"Saved CSV: {csv_output} ({len(self.results)} volumes)")
            except Exception as e:
                logger.error(f"Error saving CSV: {e}")
        
        # Save text samples
        if self.sample_texts:
            try:
                with open(text_output, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write("SAMPLE TEXTS FROM UNIQUE DOCUMENT TYPES\n")
                    f.write("="*80 + "\n\n")
                    
                    for sample in self.sample_texts:
                        f.write(f"TYPE: {sample['type']}\n")
                        f.write(f"VOLUME: {sample['volume_id']}\n")
                        f.write(f"YEAR: {sample['year']}\n")
                        f.write("-"*80 + "\n")
                        f.write(sample['text'][:2000])  # Limit to 2000 chars per sample
                        f.write("\n\n" + "="*80 + "\n\n")
                
                logger.info(f"Saved text samples: {text_output} ({len(self.sample_texts)} types)")
            except Exception as e:
                logger.error(f"Error saving text samples: {e}")
    
    def print_summary(self) -> None:
        """Print summary of findings."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("SAMPLING RESULTS")
        print("="*80)
        print(f"Volumes sampled: {len(self.results)}")
        print(f"Unique document types found: {len(self.seen_types)}")
        print(f"Document types: {sorted(self.seen_types)}")
        print()
        
        # Type breakdown
        from collections import Counter
        types = [r['type'] for r in self.results]
        type_counts = Counter(types)
        
        print("Type distribution:")
        for doc_type, count in type_counts.most_common():
            pct = (count / len(self.results)) * 100
            print(f"  {doc_type}: {count} ({pct:.1f}%)")
        
        # Year range
        years = [r['year'] for r in self.results if r['year'] != 'N/A']
        if years:
            print(f"\nYear range: {min(years)} - {max(years)}")
        
        # PC score
        pc_scores = [float(r['avg_pc_score']) for r in self.results]
        avg_pc = sum(pc_scores) / len(pc_scores)
        print(f"Average PC score: {avg_pc:.3f}")
        
        print("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Lightweight metadata sampler for archive exploration'
    )
    parser.add_argument('root_path', help='Root path to archive')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of folders to sample (default: 5000)')
    parser.add_argument('--csv-output', default='local_sample_panorama.csv',
                       help='Output CSV file')
    parser.add_argument('--text-output', default='sample_texts.txt',
                       help='Output text samples file')
    
    args = parser.parse_args()
    
    root_path = Path(args.root_path)
    if not root_path.exists():
        print(f"Error: {root_path} not found")
        sys.exit(1)
    
    sampler = LightweightSampler(root_path)
    sampler.run_sampling(args.samples)
    sampler.save_results(args.csv_output, args.text_output)
    sampler.print_summary()


if __name__ == '__main__':
    main()
