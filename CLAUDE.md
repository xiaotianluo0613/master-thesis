# Master Thesis: BGE-M3 Fine-tuning for Swedish Archives

## Project Overview

Fine-tuning the BGE-M3 dense embedding model to better retrieve and understand historical documents in Swedish archives. Working with the Swedish National Archives to improve historian search capabilities.

**Deadline**: June 2026 (thesis defense)
**Supervisor**: Swedish National Archives
**Key Contact**: Thesis supervisor at Swedish National Archives

## Technical Approach

- **Model**: BGE-M3 (dense embeddings)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for parameter efficiency
- **Data Generation**: GPL (Generate Passages and Labels) paradigm
- **Data Source**: Unlabeled historical Swedish documents
- **Current Focus**: 1-to-N passage generation (N chunks → N passages) to simulate historian search behavior

## Project Structure

```
.
├── scripts/              # Data generation, evaluation, analysis scripts
├── data/                # Processed datasets and intermediate files
├── thesis_writing/      # LaTeX thesis document
├── output/              # Model outputs and evaluation results
├── logs/                # Execution logs
├── thesis_plots/        # Visualization results
└── visualizations/      # Additional plots
```

## Key Scripts

**Data Generation**:
- `generate_n_to_n_queries.py` - Main 1-to-N query generation (current focus)
- `generate_n_to_n_queries_layered.py` - Layered version for structured generation
- `relabel_with_llm.py` - LLM-based query refinement

**Evaluation**:
- `evaluate_n_to_n.py` - Evaluate 1-to-N passage quality
- `evaluate_relabeled.py` - Assess relabeled data

**Analysis**:
- `analyze_dataset.py` - Dataset statistics
- `analyze_fingerprints.py` - Temporal/document patterns

## Current Challenge

Implementing efficient 1-to-N passage generation from N document chunks to simulate how historians search (one query might match multiple relevant passages). This is critical for training robust embeddings.

## Lessons from Previous Year

Previous student's approach failed. This year's pivot to:
1. **Quality data generation** (not just quantity)
2. **LoRA fine-tuning** (parameter-efficient, faster iteration)
3. **Better evaluation** of 1-to-N behavior

## Setup Notes

- Python venv in `.venv`
- Using Claude for development guidance (switched from other AI service)
- Goal: Learn professional AI development practices while completing thesis
- Job-focused: Building portfolio-quality code
