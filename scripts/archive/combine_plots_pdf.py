#!/usr/bin/env python3
"""
Combine all thesis plots into a single PDF document
"""

from PIL import Image
from pathlib import Path
import sys

def combine_plots_to_pdf(plots_dir: str, output_file: str):
    """Combine all PNG plots into a single PDF."""
    
    plots_path = Path(plots_dir)
    output_path = Path(output_file)
    
    # Get all PNG files sorted by name
    plot_files = sorted(plots_path.glob("*.png"))
    
    if not plot_files:
        print(f"No PNG files found in {plots_dir}")
        sys.exit(1)
    
    print(f"Found {len(plot_files)} plots:")
    for pf in plot_files:
        print(f"  - {pf.name}")
    
    # Open all images
    images = []
    for plot_file in plot_files:
        img = Image.open(plot_file)
        # Convert to RGB if necessary (PDF requires RGB)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)
    
    # Save as PDF
    if images:
        # First image is the base, rest are appended
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            resolution=100.0,
            quality=95
        )
        print(f"\n✅ PDF created: {output_path}")
        print(f"   Pages: {len(images)}")
    else:
        print("No images to combine")
        sys.exit(1)


if __name__ == '__main__':
    plots_dir = 'thesis_plots'
    output_file = 'thesis_plots/all_plots.pdf'
    
    combine_plots_to_pdf(plots_dir, output_file)
