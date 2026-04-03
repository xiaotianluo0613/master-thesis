# Fine-tuning BGE-M3 for Historical Swedish Archive Retrieval

Master's thesis project fine-tuning a multilingual embedding model to improve dense retrieval over 19th century Swedish legal and police archive documents, in collaboration with the **Swedish National Archives**.

---

## The Problem

Historians searching the Swedish National Archives face a hard retrieval problem:

- Documents are handwritten records transcribed via OCR — noisy, archaic Swedish
- Standard embedding models are trained on modern text and fail on this domain
- The predecessor thesis (Astrid, 2024) fine-tuned BGE-M3 with generic prompts and arbitrary chunking — the result barely beat BM25

This project improves on that with a structured data pipeline, curriculum learning, and domain-specific query generation.

---

## Approach

**Model**: [BGE-M3](https://huggingface.co/BAAI/bge-m3) — multilingual, multi-granularity embeddings
**Training**: BGE unified loss (`m3_kd_loss`) — BGE-M3 integration score as teacher (dense 0.4 + sparse 0.2 + ColBERT 0.4), 7 hard negatives per query
**Goal**: LoRA fine-tuning (zero inference overhead, prevents catastrophic forgetting)

### GPL Paradigm
```
XML pages → chunks → group (3–4 pages) → Gemini generates queries
→ hard negative mining (BGE-M3) → teacher scoring → fine-tune
```

### 4-Layer Curriculum Learning

Training progressively introduces harder document types, from richest narrative text to noisiest OCR-degraded fragments:

| Layer | Document Types | Target Pairs | Cumulative |
|-------|---------------|-------------|-----------|
| 1 | Court Book + Court Records + Reports | 5,000 | 5,000 |
| 2 | + District + Protocols | 2,500 | 7,500 |
| 3 | + Legal | 2,500 | 10,000 |
| 4 | + City | 1,500 | 11,500 |

### Query Design: N-to-N

Groups of 3–4 consecutive pages → 3 queries per group (2 entity + 1 social pattern).
Each query maps to all chunks in the group as positives.
Queries simulate a historian searching blind — no source text details leaked.

---

## Pilot Results (2026-03-27)

Evaluated on 87 N-to-N queries, 550-chunk corpus:

| Model | nDCG@10 | MRR@10 |
|-------|---------|--------|
| BGE-M3 baseline | 0.150 | 0.273 |
| BGE-M3 fine-tuned (BGE unified, 1 epoch) | **0.191** | **0.357** |

**+31% MRR** after just 1 epoch on 330 training examples. Full-scale Layer 1 training (~5,000 pairs) is in progress.

---

## Project Structure

```
.
├── scripts/
│   ├── pipeline/           # Active pipeline — run in order
│   │   ├── build_layer1_chunks.py              # Step 1: chunk XML pages, proportional sampling
│   │   ├── group_layer1_pairs_chunks_3_4.py    # Step 2: group into 3–4 page windows
│   │   ├── generate_n_to_n_queries_layered.py  # Step 3: Gemini query generation
│   │   ├── split_train_val.py                  # Step 4: train/val split
│   │   ├── mine_hard_negatives_bge.py          # Step 5: hard negatives (BGE-M3)
│   │   ├── score_bge_integration.py            # Step 6: teacher scoring
│   │   ├── convert_to_flagembedding_format.py  # Step 7: format for FlagEmbedding
│   │   └── evaluate_comparison.py              # Step 8: nDCG + MRR evaluation
│   ├── data_prep/          # One-time corpus preparation (done)
│   └── archive/            # Superseded scripts (GPL approach, exploratory)
├── slurm/                  # UPPMAX job scripts (carry all hyperparameters)
├── docs/
│   ├── experiment_log.md   # Running record of all experiments and decisions
│   └── approach.md         # Methodology details
├── output/
│   └── data_pools/         # Volume ID lists per curriculum layer
└── thesis_writing/         # LaTeX thesis
```

---

## Evaluation Design

- **Validation**: Per-layer, carved from generated queries, used during training
- **Test**: Temporal split (train ≤ 1875, test 1878–1900) + real historian queries from Erik's colleagues at the Swedish National Archives — evaluated **once** at thesis submission
- **Metric**: nDCG@10 (primary), MRR@10 (secondary)
- **Max pooling**: `score(doc) = max(chunk scores)` — fair comparison across models with different context windows

---

## Future: RAG Demo

After fine-tuning: a working RAG system for GitHub showcase.
```
Historian query → fine-tuned BGE-M3 retrieves top-k chunks → LLM answers in modern Swedish → response with archive citations
```

---

## Stack

- Python 3.11, [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), PyTorch
- Query generation: Gemini 2.5 Flash
- Compute: UPPMAX Pelle cluster (NVIDIA A100 48GB)
- Corpus: ~16,000 volumes, Swedish National Archives (not public)
