#!/usr/bin/env python3
"""
Lightweight Archive Metadata Sampler - Version 3
Uses volume ID prefix to categorize document types.
Extracts metadata and sample text from actual content.
"""

import csv
import os
import sys
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Set, List
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

YEAR_PATTERN = re.compile(r'(?:år\s*|year\s*)?(\d{4})', re.IGNORECASE)
PC_PATTERN = re.compile(r'PC="([0-9.]+)"')


class LightweightSamplerV3:
    """Fast metadata sampler using volume ID prefix."""
    
    def __init__(self, root_path: Path):
        self.root_path = Path(root_path)
        self.results = []
        self.seen_types = set()
        self.sample_texts = []
        
    def extract_title_and_keywords(self, volume_path: Path) -> tuple:
        """Extract volume title from content and find keywords."""
        xml_files = sorted(volume_path.glob("*.xml"))
        
        title = "Unknown"
        doc_type = "Other"
        
        # Scan up to 30 files to find actual content
        for xml_file in xml_files[:30]:
            with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(6000)
            
            # Extract CONTENT fields
            matches = re.findall(r'CONTENT="([^"]*)"', content)
            
            # Only process if we found substantial text (more than just metadata)
            if matches and len(matches) > 5:
                text = ' '.join(matches)
                title = text[:150]
                
                # Keywords to look for
                keywords = {
                    'Rapporter': 'Reports',
                    'Protocoll': 'Protocols',
                    'Domböcker': 'Court_Records',
                    'Journaler': 'Journals',
                    'Register': 'Registers',
                    'Index': 'Index',
                    'Dombok': 'Court_Book',
                    'Härads': 'District',
                    'Kommuns': 'Municipality',
                    'Städ': 'City',
                    'Rätt': 'Law',
                    'Domstol': 'Court',
                    'Laga': 'Legal',
                    'Målbok': 'Case_Book'
                }
                
                # Find matching keyword
                for keyword, category in keywords.items():
                    if keyword.lower() in text.lower():
                        doc_type = category
                        break
                
                return (title, doc_type)
        
        # No content found in first 30 files
        return ("Empty_or_Blank", "Empty_or_Blank")
    
    def read_file_text_simple(self, filepath: Path, max_chars: int = 3000) -> str:
        """Read XML file as plain text."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read(max_chars * 2)
            
            # Strip XML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Clean up whitespace
            text = ' '.join(text.split())
            return text[:max_chars]
            
        except Exception as e:
            logger.debug(f"Error reading {filepath}: {e}")
            return ""
    
    def get_pc_score_simple(self, filepath: Path) -> Optional[float]:
        """Extract PC score from file."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read(5000)
            
            match = PC_PATTERN.search(text)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, TypeError):
                    return None
            
            return None
            
        except Exception as e:
            logger.debug(f"Error getting PC score from {filepath}: {e}")
            return None
    
    def find_year(self, volume_path: Path) -> str:
        """Extract year by scanning first 10 files for actual content."""
        xml_files = sorted(volume_path.glob("*.xml"))
        
        for xml_file in xml_files[:10]:
            with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(6000)
            
            # Extract CONTENT fields only
            matches = re.findall(r'CONTENT="([^"]*)"', content)
            
            # Only search if we have actual text content
            if matches and len(matches) > 5:
                text = ' '.join(matches)
                
                # Look for 4-digit years in reasonable range (1600-2100)
                year_matches = re.findall(r'\b(1[6-9]\d{2}|20[0-1]\d)\b', text)
                if year_matches:
                    # Return first valid year found
                    return year_matches[0]
        
        return "N/A"
    
    def get_avg_pc_score(self, volume_path: Path) -> float:
        """Average PC score from first 3 files."""
        xml_files = sorted(volume_path.glob("*.xml"))[:3]
        pc_scores = []
        
        for xml_file in xml_files:
            pc = self.get_pc_score_simple(xml_file)
            if pc is not None:
                pc_scores.append(pc)
        
        return sum(pc_scores) / len(pc_scores) if pc_scores else 0.0
    
    def get_xml_count(self, volume_path: Path) -> int:
        """Count XML files in volume."""
        try:
            return len(list(volume_path.glob("*.xml")))
        except Exception as e:
            logger.debug(f"Error counting XMLs in {volume_path}: {e}")
            return 0
    
    def sample_volume(self, volume_path: Path) -> Optional[Dict]:
        """Sample a single volume."""
        try:
            volume_id = volume_path.name
            
            # Extract title and find doc type from keywords
            title, doc_type = self.extract_title_and_keywords(volume_path)
            
            # Get year
            year = self.find_year(volume_path)
            
            # Get stats
            xml_count = self.get_xml_count(volume_path)
            avg_pc = self.get_avg_pc_score(volume_path)
            
            result = {
                'volume_id': volume_id,
                'title': title,
                'type': doc_type,
                'year': year,
                'xml_count': xml_count,
                'avg_pc_score': f"{avg_pc:.3f}"
            }
            
            # If this is a NEW document type, sample text
            if doc_type not in self.seen_types:
                self.seen_types.add(doc_type)
                
                xml_files = list(volume_path.glob("*.xml"))
                if xml_files:
                    # Find first file with actual content
                    for sample_file in xml_files[:15]:
                        with open(sample_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(5000)
                        
                        matches = re.findall(r'CONTENT="([^"]*)"', content)
                        if matches and len(matches) > 5:
                            sample_text = ' '.join(matches[:100])  # Join first 100 words
                            
                            if sample_text and len(sample_text) > 100:
                                self.sample_texts.append({
                                    'type': doc_type,
                                    'volume_id': volume_id,
                                    'year': year,
                                    'text': sample_text[:2000]
                                })
                                logger.info(f"NEW TYPE: {doc_type} (from {volume_id})")
                            break
            
            return result
            
        except Exception as e:
            logger.error(f"Error sampling {volume_path}: {e}")
            return None
    
    def run_sampling(self, sample_count: int = 5000) -> None:
        """Randomly sample N folders from archive."""
        all_volumes = [d for d in self.root_path.iterdir() if d.is_dir()]
        
        if not all_volumes:
            logger.error("No volume folders found")
            return
        
        sample_size = min(sample_count, len(all_volumes))
        sampled_volumes = random.sample(all_volumes, sample_size)
        
        logger.info(f"Sampling {sample_size} folders from {len(all_volumes)} total")
        
        for idx, volume_path in enumerate(sampled_volumes, 1):
            if idx % 100 == 0:
                logger.info(f"[{idx}/{sample_size}] {volume_path.name}")
            
            result = self.sample_volume(volume_path)
            if result:
                self.results.append(result)
        
        logger.info(f"Completed. Found {len(self.seen_types)} unique types")
    
    def save_results(self, csv_output: str = "local_sample_panorama.csv",
                     text_output: str = "sample_texts.txt") -> None:
        """Save metadata CSV and sample texts."""
        if self.results:
            try:
                # Sort by volume_id
                sorted_results = sorted(self.results, key=lambda x: x['volume_id'])
                
                with open(csv_output, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['volume_id', 'title', 'type', 'year', 'xml_count', 'avg_pc_score']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(sorted_results)
                logger.info(f"Saved CSV: {csv_output} ({len(self.results)} volumes)")
            except Exception as e:
                logger.error(f"Error saving CSV: {e}")
        
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
                        f.write(sample['text'][:2000])
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
        
        from collections import Counter
        types = [r['type'] for r in self.results]
        type_counts = Counter(types)
        
        print("Type distribution:")
        for doc_type, count in type_counts.most_common():
            pct = (count / len(self.results)) * 100
            print(f"  {doc_type}: {count} ({pct:.1f}%)")
        
        years = [r['year'] for r in self.results if r['year'] != 'N/A']
        if years:
            try:
                print(f"\nYear range: {min(years)} - {max(years)}")
                from statistics import mean
                avg_year = mean([int(y) for y in years])
                print(f"Average year: {avg_year:.0f}")
            except:
                pass
        
        pc_scores = [float(r['avg_pc_score']) for r in self.results]
        if pc_scores:
            avg_pc = sum(pc_scores) / len(pc_scores)
            print(f"Average PC score: {avg_pc:.3f}")
        
        print("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Lightweight metadata sampler (fast, safe version v3)'
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
    
    sampler = LightweightSamplerV3(root_path)
    sampler.run_sampling(args.samples)
    sampler.save_results(args.csv_output, args.text_output)
    sampler.print_summary()


if __name__ == '__main__':
    main()
