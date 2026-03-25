# Master Thesis Project - Historical Police Reports

## 📁 Project Structure

```
master_thesis/
├── 30002021/                    # Raw XML files from volume 30002021
├── Filedrop-7hXHfBBqt2nJHQuk/   # Additional data files
├── data/                        # Processed data and queries
│   ├── 30002051_chunks*.json   # Chunked police reports
│   ├── generated_queries_complete.json  # 393 synthetic queries
│   └── ...
├── scripts/                     # Python and shell scripts
│   ├── generate_queries_*.py   # Query generation scripts
│   ├── *_fingerprinter.py      # Volume analysis tools
│   ├── analyze_*.py            # Analysis scripts
│   └── *.sh                    # Shell automation scripts
├── docs/                        # Documentation
│   ├── INDEX.md                # Main documentation index
│   ├── chunking_methodology_summary.md
│   └── ...
├── logs/                        # Execution logs
├── output/                      # Generated outputs (CSV, TXT)
├── thesis_plots/                # Visualization outputs
└── visualizations/              # Additional visualizations

## 🎯 Quick Start

### Generated Queries (Ready to Use)
- **File**: `data/generated_queries_complete.json`
- **Count**: 393 queries (124 chunks)
- **Types**: thematic, entity_tracking, cross_reference
- **Quality**: 99.5% unique, verified

### Key Scripts
- `scripts/generate_queries_gemini.py` - Generate queries using Gemini API
- `scripts/generate_queries_github.py` - Generate queries using GitHub Models
- `scripts/volume_fingerprinter.py` - Analyze volume metadata
- `scripts/split_large_chunks.py` - Chunk processing

### Documentation
- `docs/INDEX.md` - Main documentation hub
- `docs/chunking_methodology_summary.md` - Chunking approach
- `docs/RATE_LIMITING_EXPLAINED.md` - API usage guidelines

## 📊 Dataset Overview
- **Volume**: 30002051 (19th century Swedish police reports)
- **Chunks**: 557 total, 208 unique parent documents
- **Language**: Historical Swedish → Modern Swedish queries
- **Time Period**: 1868-1899

## 🔄 Next Steps
1. Baseline retrieval evaluation with existing 393 queries
2. Complete remaining query generation when API limits reset
3. Full retrieval system evaluation
