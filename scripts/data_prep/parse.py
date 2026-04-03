#!/usr/bin/env python3
"""
Robust ALTO XML parser to extract text and convert to JSONL format.

Handles ALTO XML namespace, Swedish hyphenation logic, and text extraction
from alto:String elements within alto:TextBlock.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from lxml import etree
from tqdm import tqdm


# Define ALTO namespace
ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}


def extract_text_from_textblock(textblock: etree._Element) -> str:
    """
    Extract text from an alto:TextBlock element.
    
    Handles Swedish hyphenation: if a line ends with '¬', strip it and merge
    with the first word of the following line without a space.
    
    Args:
        textblock: The alto:TextBlock element
        
    Returns:
        Cleaned text with hyphenation resolved
    """
    lines = []
    
    # Get all TextLine elements within this TextBlock
    textlines = textblock.findall(".//alto:TextLine", ALTO_NS)
    
    i = 0
    while i < len(textlines):
        textline = textlines[i]
        line_text = ""
        has_hyphenation = False
        
        # Iterate through all children of TextLine to maintain order
        for child in textline:
            if child.tag == f"{{{ALTO_NS['alto']}}}String":
                content = child.get("CONTENT", "").strip()
                if content:
                    line_text += content
                    
            elif child.tag == f"{{{ALTO_NS['alto']}}}SP":
                # Space element
                line_text += " "
                
            elif child.tag == f"{{{ALTO_NS['alto']}}}HYP":
                # Hyphenation marker (usually '¬')
                hyp_content = child.get("CONTENT", "")
                if hyp_content == "¬":
                    has_hyphenation = True
        
        line_text = line_text.strip()
        
        if line_text:
            # Check if this line ends with Swedish hyphenation marker
            if has_hyphenation and i + 1 < len(textlines):
                # Remove trailing space if present
                line_text = line_text.rstrip()
                
                # Look ahead to next line and extract first word
                next_textline = textlines[i + 1]
                next_first_word = ""
                
                for child in next_textline:
                    if child.tag == f"{{{ALTO_NS['alto']}}}String":
                        content = child.get("CONTENT", "").strip()
                        if content:
                            next_first_word = content
                            break  # Take only the first word
                
                # Merge with next line without space
                if next_first_word:
                    line_text = line_text + next_first_word
                    # Skip the next line since we've consumed its first word
                    i += 2
                    lines.append(line_text)
                    continue
            
            lines.append(line_text)
        
        i += 1
    
    # Join all lines with newlines
    return "\n".join(lines)


def parse_alto_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Parse a single ALTO XML file and extract text.
    
    Args:
        filepath: Path to the ALTO XML file
        
    Returns:
        Dictionary with 'id', 'text', and 'metadata' fields, or None if parsing fails
    """
    try:
        # Parse XML with namespace handling
        tree = etree.parse(str(filepath))
        root = tree.getroot()
        
        # Extract text from all TextBlock elements
        textblocks = root.findall(".//alto:TextBlock", ALTO_NS)
        
        all_text_parts = []
        for textblock in textblocks:
            text = extract_text_from_textblock(textblock)
            if text:
                all_text_parts.append(text)
        
        # Combine all text with paragraph breaks
        full_text = "\n\n".join(all_text_parts)
        
        # Get folder name as volume metadata
        folder_name = filepath.parent.name
        
        return {
            "id": filepath.stem,
            "text": full_text,
            "metadata": {
                "volume": folder_name
            }
        }
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        return None


def main():
    """
    Main function to parse all ALTO XML files and write to JSONL.
    """
    workspace_root = Path(__file__).parent
    
    # Find all XML files in subdirectories
    xml_files = sorted(workspace_root.glob("*/*.xml"))
    
    if not xml_files:
        print("No XML files found.", file=sys.stderr)
        sys.exit(1)
    
    # Output file
    output_file = workspace_root / "output.jsonl"
    
    # Parse and write to JSONL
    with open(output_file, "w", encoding="utf-8") as outf:
        for filepath in tqdm(xml_files, desc="Parsing ALTO files"):
            result = parse_alto_file(filepath)
            if result:
                outf.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Successfully wrote {len(xml_files)} files to {output_file}")


if __name__ == "__main__":
    main()
