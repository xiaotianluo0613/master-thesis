# Master Thesis: BGE-M3 Fine-tuning for Swedish Archives

## Project Overview

Fine-tuning BGE-M3 to improve dense retrieval over historical Swedish archive documents.
Working with the Swedish National Archives to help historians search 19th century legal
and police records.

**Goals**:
1. Learn how to use AI to do projects in an industry setting
2. Build a GitHub portfolio demo showcasing the full pipeline
3. Achieve a working RAG system on top of the fine-tuned model

**Deadline**: June 2026 (thesis defense)
**Supervisor**: Erik (Swedish National Archives)
**Predecessor**: Astrid's thesis — fine-tuned BGE-M3 with full FT, generic prompts, arbitrary
chunking. Result barely beat BM25. Her Future Work explicitly recommended LoRA.
**Cluster**: UPPMAX (Pelle), project `uppmax2026-1-95`
**Repo**: https://github.com/xiaotianluo0613/master-thesis (private)

---

## Current Status

See `docs/experiment_log.md` for the authoritative running record. Summary:

Pilot complete (2026-03-27). Used Layer 1 document types (Court Book + Court Records +
Reports), ~550 chunks, 417 queries, 330 train / 87 val.

| Model | nDCG@10 | MRR@10 |
|-------|---------|--------|
| BGE-M3 baseline | 0.150 | 0.273 |
| BGE unified (Approach B) | **0.191** | **0.357** (+31%) |

**GPL approach dropped. BGE unified is the one approach going forward.**
**Next**: Scale up to full Layer 1 training (~5,000 pairs) with LoRA.

---

## Technical Approach

- **Model**: BGE-M3 (`BAAI/bge-m3`)
- **Fine-tuning goal**: LoRA (PEFT). Full fine-tuning used for pilot only — FlagEmbedding's
  encoder-only m3 trainer does not yet support LoRA. When we add LoRA: either patch
  FlagEmbedding or switch to a custom PEFT training loop.
- **Training**: BGE unified — m3_kd_loss, BGE-M3 integration score as teacher
  (dense 0.4 + sparse 0.2 + ColBERT 0.4), 7 hard negatives from ranks 2–200
- **Paradigm**: GPL — query generation → hard negative mining → teacher scoring → train

### Why LoRA
1. Zero inference latency
2. Bigger batch size on 48GB VRAM (Astrid's bottleneck)
3. Prevents catastrophic forgetting
4. Astrid's own Future Work recommendation

---

## Data Design: 4-Layer Curriculum Learning

The goal is not to teach specific document types — it is to progressively teach the model
historical Swedish legal archive language, starting from the richest narrative text and adding
more specialised or noisier content layer by layer.

Each layer adds new document types on top of previous data. Query generation is
layer-specific (different bias + few-shot examples per layer).

| Layer | Types added | Purpose | Target pairs | Cumulative |
|-------|------------|---------|-------------|-----------|
| Layer 1 | Court Book + Court Records + Reports | Rich narrative, core historical language | 5,000 | 5,000 |
| Layer 2 | + District + Protocols | Geographic entities, structured bureaucratic text | 2,500 | 7,500 |
| Layer 3 | + Legal | Dense legal terminology | 2,500 | 10,000 |
| Layer 4 | + City | Noisy fragments, OCR-degraded text | 1,500 | 11,500 |

**Size justification**: GPL paper showed 10K pairs sufficient for significant improvement;
performance saturates ~50K.

### Data pools (volume IDs only — chunking not done yet for full scale)

Pools are lists of volume IDs in `output/data_pools/`. Filtering: avg_pc_score ≥ 0.95,
blank_page_ratio ≤ 0.1, valid year, excluded types: Other, Registers, Unknown.

| Pool file | Volume count |
|-----------|-------------|
| train_layer1_pool.txt | 597 |
| train_layer2_pool.txt | 538 |
| train_layer3_pool.txt | 35 |
| train_layer4_pool.txt | 16 |
| test_pool_all_seen_types.txt | 35 |

**Temporal split**: train ≤ 1875, test > 1875 (test = 1878–1900)

---

## Query Generation Design

**Only N-to-N queries** (no N-to-1). Groups of 3–4 consecutive chunks → 3 queries per
group (2 entity + 1 social pattern). Each query maps to 2–6 positive chunks.

### Core principle
Queries simulate a historian who does NOT know where the answer is:
- Must not leak specific source text details
- Must sound like a researcher searching blind
- Must require semantic matching to retrieve the answer
- General enough to retrieve multiple related chunks

### Prompt structure
1. Base prompt (historian role, task, guidelines, output format)
2. Layer-specific bias (what to emphasize per layer)
3. Few-shot examples (2 per layer, from Erik's human queries — excluded from test set)

---

## Evaluation Design

### Validation set (used during training)
- Carved from generated queries, same types as training
- Each layer of training has its own validation set
- Used to compare runs and tune hyperparameters
- Pilot val: 87 queries, N-to-N (all positives kept), 550-chunk corpus

### Test set (used ONCE at the end — never touch during training)
- Temporal split: unseen time period (1878–1900)
- Human queries: Erik's colleagues writing real historian queries (gold standard)
- Few-shot examples sourced from these human queries are already excluded
- Only evaluated when writing the thesis Results chapter

### Metrics
- **nDCG@10** (primary), **MRR@10** (secondary)
- **Max Pooling** for fair comparison across models with different token limits:
  score(parent_doc) = max score across all child chunks

### Baseline note
- MRR 0.56 (2026-03-18): 1-to-1 queries, NOT comparable to current setup
- MRR 0.273 (2026-03-27): correct N-to-N baseline

---

## Project Structure

```
.
├── CLAUDE.md                        # This file
├── scripts/
│   ├── pipeline/                    # Active pipeline — run in this order
│   │   ├── generate_n_to_n_queries_layered.py   # Step 1: query generation (Gemini)
│   │   ├── split_train_val.py                   # Step 2: train/val split
│   │   ├── mine_hard_negatives_bge.py            # Step 3: hard negatives (7, ranks 2-200)
│   │   ├── score_bge_integration.py              # Step 4: BGE-M3 integration teacher scoring
│   │   ├── convert_to_flagembedding_format.py    # Step 5: format for FlagEmbedding
│   │   └── evaluate_comparison.py               # Step 6: nDCG + MRR evaluation
│   ├── data_prep/                   # One-time corpus building scripts (done)
│   └── archive/                     # Exploratory / superseded scripts (incl. GPL)
├── slurm/                           # UPPMAX job scripts
├── data/                            # Input data (not in git — scp manually)
├── output/
│   ├── data_pools/                  # Volume ID lists per layer + test pool
│   └── models/                      # Trained model checkpoints (not in git)
├── docs/
│   ├── experiment_log.md            # Authoritative running log (most recent first)
│   ├── approach.md                  # Methodology details
│   └── chunking_methodology_summary.md
├── thesis_writing/                  # LaTeX thesis
└── thesis_plots/                    # Figures
```

---

## UPPMAX Setup

- **Storage**: `/proj/uppmax2025-2-505/xilu1878/master_thesis/`
- **GPU project**: `uppmax2026-1-95` (from April 1, 2026)
- **Python**: `module load Python/3.11.5-GCCcore-13.3.0`
- **Venv**: `source .venv/bin/activate`
- **FlagEmbedding**: source install at `/proj/uppmax2025-2-505/xilu1878/FlagEmbedding`
  - Bug fix: `dtype=torch_dtype` → `torch_dtype=torch_dtype` in runner.py
- **transformers**: must be `4.53.0`
- **Data files**: NOT in git — scp manually

---

## Future Extension: RAG Demo

Build a working RAG system on top of the fine-tuned model for GitHub showcase.
```
Historian query → fine-tuned BGE-M3 retrieves top-k chunks → LLM answers in modern Swedish → response with archive citations
```
Lives in `scripts/rag/`. Stretch goal: agentic RAG.
