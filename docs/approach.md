# Fine-tuning Methodology

## Overview

Fine-tuning BGE-M3 for dense retrieval over 19th century Swedish archive documents (Swedish National Archives). The approach uses LoRA (parameter-efficient fine-tuning) with the BGE unified training objective, applied in four progressive layers of increasing domain specialisation.

**Research questions**:
1. Does LoRA fine-tuning improve retrieval quality over the BGE-M3 baseline on this domain?
2. Does cumulative domain expansion prevent catastrophic forgetting across layers?
3. At what point does adding more training data stop helping (saturation)?

---

## Full-Scale Approach (Layer 1–4)

### Fine-tuning Method: LoRA

**Decision**: LoRA (Low-Rank Adaptation) with target modules Q, K, V, O (dense/output projection included).

**Config**: r=16, alpha=32, dropout=0.05, lr=1e-4, batch=4, 3 epochs.

**Rationale**:
- Full fine-tuning risks catastrophic forgetting of multilingual pretraining
- LoRA trains only 0.4% of parameters; inference latency is identical to base model
- Layer 1 experiments confirmed LoRA+dense closes 96% of full FT's MAP gain

**Implementation**: FlagEmbedding patched to support LoRA (`patches/apply_lora_patch.py`). Patch adds PEFT LoRA on the base encoder before FlagEmbedding wraps it.

---

### Training Strategy: Cumulative Domain Expansion

Each layer trains on **all previous layers' data combined**, starting from the previous layer's checkpoint.

| Layer | New types | Cumulative types | Approx. training examples |
|-------|-----------|-----------------|--------------------------|
| Layer 1 | Court Book, Court Records, Reports | same | ~19,700 |
| Layer 2 | + District, Protocols | all 5 | ~30,000 |
| Layer 3 | + Legal | all 6 | ~40,000 |
| Layer 4 | + City | all 7 | ~46,000 |

**Why cumulative, not sequential**: LoRA adapters are small (few parameters) and highly susceptible to catastrophic forgetting when trained on new data alone. Replaying all previous data is cheap relative to the benefit. Each layer expands the domain distribution the adapter has seen — hence "cumulative domain expansion" rather than strict curriculum learning (easy→hard).

**Note**: "Curriculum learning" is a loose analogy here. The ordering (narrative-rich → geographic → legal → noisy) is motivated by domain difficulty and data quality, not a formal easy-to-hard curriculum.

---

### Evaluation: Global Validation Set

**Design**: One fixed validation set, sampled proportionally from all 4 layers' generated queries, with group-aware split (no query group spans train/val boundary).

**Purpose**: Same exam across all training stages — scores after Layer 1, 2, 3, 4 training are directly comparable. Any drop reveals catastrophic forgetting.

**Construction**: Built incrementally as each layer's queries become available. Layer 1 and 2 models are retroactively evaluated on the complete global val set once all 4 layers are done.

**Sampling**: ~10% of total training data, proportional by layer size (to confirm with supervisor).

**Alongside**: Each layer also has its own layer-specific val set for tracking per-layer training dynamics.

---

## Pilot Experiment (archived — 2026-03-27)

*The pilot compared GPL vs BGE unified on 550 chunks / 417 queries. BGE unified won (+31% MRR vs +22% for GPL). GPL approach dropped after pilot. The sections below document the pilot methodology for reference.*

---

## Data Preparation

### Query Generation
- Model: `gemini-2.5-flash` (production run), compared against `gemini-2.5-pro`
- Script: `scripts/generate_n_to_n_queries_layered.py`
- Output: `data/queries_layer1_n2n_pilot_final_v3.json`
- Format: N-to-N (each query has 2–6 relevant chunks)

### Train/Val Split
- Script: `scripts/split_train_val.py`
- Split: **330 train / 87 val** (seed=42, group-aware — no document group spans both splits)
- Training: 1 positive randomly sampled per query
- Validation: ALL positives kept for N-to-N evaluation

---

## Approach A: GPL + MarginMSE

Based on: *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval* (Wang et al., 2022)

### Hard Negative Mining
- Script: `scripts/mine_hard_negatives_gpl.py`
- Retrieval: BGE-M3 base, top-10 per query
- Selection: **bottom-1** from filtered candidates (avoids false negatives at top ranks)
- Filtering: removes true positives + all chunks from same document group
- Output: `output/gpl_negatives.json`

### Scoring
- Script: `scripts/score_margins_gpl.py`
- Model: `BAAI/bge-reranker-v2-m3` (cross-encoder)
- Scores (query, positive) and (query, negative) separately; outputs pos_scores/neg_scores
- Skips examples where pos_score ≤ neg_score (unreliable)
- Output: `output/gpl_training_data.jsonl`

### Training
- SLURM: `slurm/finetune_gpl.sh`
- Framework: FlagEmbedding official m3 trainer (MarginMSE not supported; using kl_div)
- Loss: `kl_div` with pre-computed reranker scores as teacher
- Hyperparameters: epochs=1 (pilot), batch_size=32, lr=2e-5, warmup_ratio=0.1, temp=0.02, train_group_size=2
- Output: `output/models/bge-m3-gpl/`

---

## Approach B: BGE-M3 Official (InfoNCE + Knowledge Distillation)

Based on: *BGE M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation* (Chen et al., 2024)

### Hard Negative Mining
- Script: `scripts/mine_hard_negatives_bge.py`
- Retrieval: BGE-M3 base, top-200 per query
- Selection: **7 negatives from ranks 2–200** (rank 1 skipped as likely false negative)
- Filtering: removes true positives + all chunks from same document group
- Output: `output/bge_negatives.json`

### Teacher Scoring
- Script: `scripts/score_bge_integration.py`
- Model: BGE-M3 integration score (dense + sparse + ColBERT, weights 0.4/0.2/0.4)
- Scores: query vs positive + all 7 negatives
- Output: `output/bge_training_data_scored.jsonl`
- Note: pre-FT scores are inverted (avg pos=0.31, avg neg=0.35) — expected, confirms need for fine-tuning

### Training
- Framework: FlagEmbedding official m3 trainer (`torchrun -m FlagEmbedding.finetune.embedder.encoder_only.m3`)
- SLURM: `slurm/finetune_bge.sh`
- Loss: `m3_kd_loss` (unified self-distillation across dense+sparse+colbert)
- Hyperparameters (pilot): epochs=1, batch_size=2, lr=1e-5, train_group_size=8, warmup_ratio=0.1, temp=0.02, fp16=True
- Output: `output/models/bge-m3-unified/`

---

## Evaluation

- Script: `scripts/evaluate_comparison.py`
- Val set: `data/val_queries.json` (87 queries, all positives)
- Models: baseline (BAAI/bge-m3), Approach A, Approach B
- Metrics: **nDCG@10**, **MRR@10**

---

## Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Mining model | BGE-M3 (not BM25) | BM25 fails on 19th century Swedish OCR with broken tokens and archaic spelling |
| Query principle | No source details leaked | Queries must require semantic matching; sound like a researcher who doesn't know where the answer is |
| Positive sampling | 1 random per query (train) | Avoids query repetition artifacts in training batches |
| Group-level filtering | Exclude same-group chunks | All chunks from same daily report share context; leakage would give false easy negatives |
| GPL negative rank | Bottom-1 of top-10 | Top ranks may be unlabeled true positives (false negatives) |
| BGE negative rank | Ranks 2–200 | Rank 1 skipped for same reason; wider pool for 7 negatives |
