# Fine-tuning Methodology: GPL vs BGE-M3 Official

## Overview

Pilot experiment comparing two BGE-M3 fine-tuning paradigms on Swedish historical archive documents (Swedish National Archives, 19th century police reports and court records).

**Research question**: Which fine-tuning paradigm produces better dense retrieval on this domain?

**Dataset**: 417 queries generated via Generative Pseudo Labelling (GPL) from 550 document chunks. Queries follow a 2 entity + 1 social pattern ratio per document group, designed so that each query requires semantic matching to find the answer and does not reveal specific details from the source text.

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
