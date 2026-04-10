# Experiment Log

Running log of experiments, results, and decisions. Most recent entry first.

---

## 2026-04-10 (evening) — L2 Fine-tune Submitted, L3/L4 Scoring Queued, Global Val Set Built

**Status**: All jobs queued on UPPMAX. Pipeline advancing on all layers simultaneously.

**Actions**:
- Verified L2 scoring: 6474 examples, scores 0.15–0.49 (mean 0.30) — clean ✅
- Split L3/L4 queries on UPPMAX: L3 1629 train / 250 val, L4 978 train / 150 val (seed 42, 0 overlap)
- All 4 layer val sets committed to git (canonical UPPMAX versions)
- Built global val set: 1150 queries (L1: 500, L2: 250, L3: 250, L4: 150), proportional to training weights
- Fixed all split scripts: `-p core` → `-p cpu` (core invalid on Pelle, cpu/gpu are valid)
- Diagnosed L2 fine-tune failure: LoRA checkpoint has no `config.json` (only adapter delta saved, not base weights)
- Created `scripts/pipeline/merge_lora_checkpoint.py`: merges LoRA adapter + base BGE-M3, copies colbert/sparse linear heads
- Merged L1 LoRA checkpoint → `output/models/layer1-bge-m3-lora-dense-b4-merged`
- Created `slurm/layer3_mine.sh`, `layer4_mine.sh`, `layer3_score.sh`, `layer4_score.sh`
- L3 mining: 6498 examples ✅ L4 mining: 3903 examples ✅ (both completed)

**Jobs running on UPPMAX**:
- `4858497` — L2 fine-tune (layer2-bge-m3-lora-dense-b4, from merged L1 checkpoint)
- `4858615` — L3 teacher scoring
- `4858616` — L4 teacher scoring

**Key decision — merge after every layer**:
LoRA saves only the adapter delta. To continue training from a LoRA checkpoint, must first merge into a full model. Going forward: add merge step to end of each finetune slurm script.

**Next**:
- When L2 fine-tune completes: check results, decide L3 fine-tune hyperparameters
- When L3/L4 scoring completes: create L3/L4 finetune scripts (need L2 merged checkpoint as starting point for L3)
- Add auto-merge step to end of layer2_finetune.sh

---

## 2026-04-10 — Layer 2 Pipeline Progress + RAG Demo Started

**Status**: Layer 2 data pipeline mostly done. L3/L4 query gen complete. RAG demo built.

**Actions**:
- Fixed `build_layer2_chunks.py`: District was crowding out Protocols (cap now proportional to volume count: ~181 District, ~2319 Protocols)
- Generated queries for all layers on local Mac (UPPMAX login node had missing fewshot file):
  - Layer 2: 1881 queries (629 groups × 3, 2 failed)
  - Layer 3: 1881 queries (627 groups × 3, 0 failed)
  - Layer 4: 1128 queries (376 groups × 3, 0 failed)
- Split Layer 2: 1629 train / 250 val (group-aware, 0 overlap)
- Layer 2 hard negative mining: 6474 examples, all with 7 full negatives (avg top-1 neg similarity: 0.470)
- Layer 2 teacher scoring: complete → `output/layer2_bge_training_data_scored.jsonl`
- L3/L4 queries pushed to git; splits + mine/score/finetune pending

**RAG Demo** (`demo/`):
- Built with Gemini for job portfolio purposes
- Stack: Streamlit + Qdrant Cloud + BGE-M3 (sentence-transformers)
- `ingest_data.py`: parses XML, encodes with BGE-M3, upserts to Qdrant
- `app.py`: chat UI, encodes query, retrieves top-3 from Qdrant, displays results
- Currently uses base BGE-M3; will swap in fine-tuned model when ready
- One volume ingested so far (`30002021`)

**Pending**:
- Submit `sbatch slurm/layer2_finetune.sh` on UPPMAX (after UPPMAX connection restores)
- Split L3/L4 queries on UPPMAX → build global val set
- Fix split scripts: `-p core` partition invalid on Pelle (run splits directly on login node)

---

## 2026-04-07 — Supervisor Meeting + Layer 2 Planning

**Status**: Supervisor meeting done. Layer 2 pipeline scripts written and committed.

**Supervisor feedback**:
- Layer 1 results well-received (vs baseline)
- Absolute figures not yet practical — expected at this stage
- Requested: global validation set covering ALL document types to simulate real use

**Decisions**:
- Global val set: proportionally sampled from all 4 layers (weights 5000:2500:2500:1500), built incrementally
- Training strategy confirmed: cumulative domain expansion (each layer trains on all previous data combined, continues from previous checkpoint)
- LoRA+dense config locked in for all layers: r=16, alpha=32, target=Q,K,V,O, lr=1e-4, batch=4, 3 epochs

**Scripts written**:
- `scripts/pipeline/build_layer2/3/4_chunks.py`
- `scripts/pipeline/prepare_layer2/3/4_data.sh`
- `slurm/layer2/3/4_query_gen.sh` (login node, nohup)
- `slurm/layer2/3/4_split.sh`, `layer2_mine.sh`, `layer2_score.sh`, `layer2_finetune.sh`
- `scripts/pipeline/build_global_val_set.py` (proportional weighted sampling)

---

## 2026-04-06 — Layer 1 LoRA Experiments Complete

**Status**: All Layer 1 experiments done. LoRA+dense is the recommended approach going forward.

**Actions**:
- Applied FlagEmbedding LoRA patch on Pelle (`patches/apply_lora_patch.py`)
- Ran 3 LoRA variants and 5-way evaluation (`layer1_eval_all.sh`):

| Run | Batch | Target | MAP | nDCG@10 | vs baseline |
|-----|-------|--------|-----|---------|-------------|
| LoRA-v1 | 8 | Q,K,V | 0.0999 | 0.1453 | +54% |
| LoRA-b4 | 4 | Q,K,V | 0.1128 | 0.1655 | +74% |
| LoRA+dense-b4 | 4 | Q,K,V,O | 0.1183 | 0.1749 | +82% |
| Full FT | 4 | all | 0.1229 | 0.1828 | +89% |

**Key findings**:
- Batch size was a confound in LoRA-v1 (b8 = 2x fewer gradient steps than full FT)
- Fixing to b4: LoRA-b4 jumps from +54% to +74% MAP
- Adding dense: further +5%, closes gap to only 4% vs full FT
- LoRA+dense trains only 0.4% of parameters, runs 2.5x faster

**Decision**: Use LoRA+dense (Q,K,V,O, r=16, alpha=32, batch=4, lr=1e-4) for Layer 2+

**Next**: Supervisor meeting 2026-04-07 11:00 → then Layer 2 data prep and training

---

## 2026-04-05 — Layer 1 Eval Results + LoRA Implementation

**Status**: Layer 1 full FT evaluated. LoRA patch written and pushed. Pending: apply patch on Pelle + submit training job.

**Actions**:
- Layer 1 full FT evaluation complete (`layer1-full-ft-v1`):
  - MAP: 0.1229, nDCG@10: 0.1828, Recall@10: 0.1647, nDCG@100: 0.2233, Recall@100: 0.2998
  - **+89% MAP vs baseline** (0.0649) on 7300-chunk corpus, 551 val queries
- Created `docs/results.md` — dedicated results tracking table for all runs
- **Lessons learned**: always write explicit plan before phase transitions; don't copy pilot hyperparameters blindly
- Wrote plan for Layer 1 LoRA training: r=16, alpha=32, target=Q+K+V, lr=1e-4, batch=8, 3 epochs
- Patched FlagEmbedding to support LoRA (`patches/flagembedding_lora.patch`):
  - `arguments.py`: added use_lora, lora_rank, lora_alpha, lora_dropout, lora_target_modules fields
  - `runner.py`: split get_model() call, inject PEFT LoRA on base encoder before FlagEmbedding wrapper
- Wrote `slurm/layer1_lora.sh` and `slurm/layer1_lora_evaluate.sh`
- All pushed to GitHub

**Pending**:
- Verify peft installed in UPPMAX venv (`python -c "import peft"`)
- Apply patch: `patch -p1 -d /proj/uppmax2025-2-505/xilu1878/FlagEmbedding < patches/flagembedding_lora.patch`
- Submit: `sbatch slurm/layer1_lora.sh`
- After training: `sbatch slurm/layer1_lora_evaluate.sh` → 3-way comparison

**Key decisions**:
- LoRA target modules: Q+K+V only (no dense/output) for clean baseline; add dense in v2
- Start from BGE-M3 base (not full FT checkpoint) for clean comparison
- Curriculum learning: each layer continues from previous layer's checkpoint (decided, not yet implemented)

**Next**: Apply patch on Pelle → submit layer1_lora.sh → evaluate → plan Layer 2

---

## 2026-04-04 — Layer 1 Query Generation Complete + GitHub Portfolio

**Status**: Layer 1 data ready. SLURM jobs being written. Waiting to run on UPPMAX.

**Actions**:
- Query generation completed: **5514 queries**, 0 failed, 0 missing positives
  - 1838 groups × 3 queries (2 entity + 1 social pattern), 3–4 positives per query
  - Fix needed: added `--disable-baseline-filter` to `prepare_layer1_data.sh` — old script was filtering new chunk IDs against pilot baseline file → 0 queries generated
- Verified query output: correct schema, all layer1, min/max positives match group sizes
- `scp`-ed `layer1_queries.json` + `layer1_chunks_grouped.json` to UPPMAX
- Wrote Layer 1 SLURM scripts: `layer1_split.sh`, `layer1_mine.sh`, `layer1_score.sh`, `layer1_finetune.sh` (all use new account `uppmax2026-1-95`)
- Polished GitHub repo for job portfolio: rewrote README, added `requirements.txt`, updated `.gitignore` (LaTeX artifacts, model weights), added `visualizations/`, deleted stale exploratory docs, made repo public

**Key decisions**:
- Pilot used 1 epoch; Layer 1 using 3 epochs — larger dataset justifies more passes
- Pilot batch size 2; Layer 1 batch size 4 — more data, more stable gradients
- Added `docs/TODO.md` as persistent task tracker across sessions
- Added `docs/experiment_log.md` update rule to memory — update at end of every session

**Layer 1 SLURM pipeline (evening)**:
- `layer1_split.sh`: ran directly on Pelle — 4962 train / 551 val
- `layer1_mine.sh` (job 4779905): ran but used old 1-positive-per-query logic
- Identified bug: N-to-N design requires one training example per positive, not random 1
- Fixed `mine_hard_negatives_bge.py`: expand each query into one example per positive chunk; negatives shared across all expansions of same query
- Re-ran mining (job 4779906): **19,716 examples** (vs 4,962 before), all with full 7 negatives
- `layer1_score.sh`: completed — `output/layer1_bge_training_data_scored.jsonl` (19,716 lines)
- `layer1_finetune.sh` (job 4779908): submitted before sleep

**Next**: Check training results in morning → evaluate → plan Layer 2

---

## 2026-04-03 — Project Restructure + Layer 1 Full-Scale Pipeline

**Status**: Layer 1 data preparation running (query generation in progress).

**Actions**:

**Project restructure (industry-standard layout)**:
- Reorganised 60+ scripts into `scripts/pipeline/`, `scripts/data_prep/`, `scripts/archive/` using `git mv` to preserve history
- Archived all GPL scripts and exploratory code — GPL approach dropped after pilot (BGE unified +31% MRR vs GPL +22%)
- Rewrote `CLAUDE.md` to reflect current state: LoRA is the goal (full FT pilot only), 4-layer curriculum design, evaluation design, UPPMAX setup, 3 project goals

**Layer 1 full-scale data pipeline (local)**:
- Wrote `scripts/pipeline/build_layer1_chunks.py`:
  - Proportional sampling: use ALL Reports first (scarce — only 321 from 2 volumes due to low OCR confidence filtering), fill remaining proportionally from Court_Book and Court_Records
  - Early stopping per type to avoid scanning all 560+ volumes — runs in seconds
  - Target: 7300 chunks from 597 Layer 1 pool volumes
- Wrote `scripts/pipeline/prepare_layer1_data.sh`: 3-step local pipeline (build chunks → group 3–4 per group → generate queries via Gemini)
- Fixed `group_layer1_pairs_chunks_3_4.py` to accept both `"chunks"` and `"pairs"` JSON keys
- Updated SLURM scripts with corrected `scripts/pipeline/` paths

**Key decisions**:
- Chunk unit: 1 XML page = 1 chunk, no token limit enforced (same as pilot, max pooling handles eval fairness)
- Proportional imbalance accepted: Reports under-represented (321 chunks vs ~7000 combined CB+CR) but used in full — quality filter excludes most Report volumes
- SLURM scripts as experiment records: Python scripts stay generic, SLURM carries all hyperparameters
- Switched to Claude Code as primary AI assistant for industry-style AI-assisted development

**Next**: Wait for Gemini query generation to complete (~several hours, uses `--resume` if interrupted). Then scp `data/layer1_queries.json` and `data/layer1_chunks_grouped.json` to UPPMAX and run SLURM jobs: mine hard negatives → teacher scoring → fine-tune Layer 1.

---

## 2026-03-27 — Pilot Evaluation Results

**Status**: Complete.

**Script**: `scripts/evaluate_comparison.py`
**Val set**: `data/val_queries.json` (87 queries, all positives, N-to-N)
**Corpus**: 550 chunks

| Model | nDCG@10 | MRR@10 |
|-------|---------|--------|
| BGE-M3 baseline | 0.1505 | 0.2726 |
| BGE-M3 unified (Approach B) | **0.1908** | **0.3573** |
| BGE-M3 GPL (Approach A) | 0.1800 | 0.3319 |

**Observations**:
- Both fine-tuned models beat baseline after just 1 epoch on 330 examples
- BGE unified outperforms GPL on both metrics
- Absolute numbers are low but expected: hard domain (19th century Swedish OCR), small pilot, N-to-N evaluation
- Relative improvement: +31% MRR for BGE unified, +22% MRR for GPL
- Baseline MRR (0.27) is lower than earlier baseline eval (0.56 on 2026-03-18) — not comparable: earlier eval used a different (likely 1-to-1) query set before N-to-N generation was implemented. Current 0.27 is the correct baseline for N-to-N evaluation.

**Next**: Investigate baseline discrepancy. Then decide: scale up data or iterate on approach.

---

## 2026-03-26 — Pilot Fine-tuning Complete

**Status**: Complete. Both models trained on UPPMAX.

**Actions**:
- Moved all heavy compute to UPPMAX (Pelle cluster) — local machine too slow
- Set up repo, venv, and data on `/proj/uppmax2025-2-505/xilu1878/master_thesis/`
- Ran hard negative mining on GPU (jobs 4713881, 4713882) — finished in seconds
  - `output/bge_negatives.json`: 330 examples, 7 negatives each, avg top1 sim=0.459
  - `output/gpl_negatives.json`: 330 examples, 1 negative each
- Scored GPL data with `bge-reranker-v2-m3` → `output/gpl_training_data.jsonl`
  - pos_score=0.69, neg_score=0.03 — strong margin, good quality
- Scored BGE data with BGE-M3 integration score (dense+sparse+colbert, weights 0.4/0.2/0.4) → `output/bge_training_data_scored.jsonl`
  - avg pos=0.31, avg neg=0.35 — inverted scores (expected pre-FT, confirms need for fine-tuning)
- Fine-tuned BGE-M3 unified (m3_kd_loss, epoch=1, full FT) → `output/models/bge-m3-unified/`
- Fine-tuned GPL approach (kl_div, epoch=1, full FT) → `output/models/bge-m3-gpl/`

**Key decisions**:
- Switched from MarginMSE to kl_div for GPL (FlagEmbedding m3 trainer does not support MarginMSE)
- Used BGE-M3 integration score as teacher (not bge-reranker-v2-m3) for BGE approach — follows Chen et al. 2024
- epoch=1 for pilot phase per supervisor advice — compare data prep approaches, not max performance
- Full fine-tuning (no LoRA) — LoRA not supported in FlagEmbedding encoder-only m3 trainer

**Infrastructure issues resolved**:
- FlagEmbedding training module requires source install (`pip install -e ".[finetune]"`)
- Bug in FlagEmbedding runner.py: `dtype=torch_dtype` → `torch_dtype=torch_dtype` (fixed manually)
- transformers version: 4.53.0 works; 4.44.0 breaks FlagEmbedding inference

**Next**: ~~Evaluation~~ — done, see below.

---

## 2026-03-25 — Hard Negative Mining

**Status**: Mining scripts written and running.

**Actions**:
- Wrote `scripts/mine_hard_negatives_gpl.py` (Approach A: top-10, bottom-1)
- Wrote `scripts/mine_hard_negatives_bge.py` (Approach B: top-200, 7 negatives)
- Wrote `scripts/split_train_val.py` (330 train / 87 val, group-aware split)
- Decided to use `data/queries_layer1_n2n_pilot_final_v3.json` as input

**Pending**: Score margins (Approach A) and teacher scores (Approach B) once mining completes.

---

## 2026-03-24 — Model Comparison PDF Sent to Supervisor

**Status**: Complete.

**Actions**:
- Ran `gemini-2.5-flash` (5 groups) and `gemini-2.5-pro` (5 groups) with same prompt
- Fixed bug: gemini-2.5-pro was copying `[entity]` format labels literally → changed to `<your entity query here>`
- Generated comparison PDF: `thesis_plots/pilot10_supervisor_examples.pdf`

**Observation**: Pro model produces more varied and natural query forms. Flash is faster and cheaper with acceptable quality for bulk generation.

---

## 2026-03-23 — Query Generation Improved

**Status**: Complete. Full run: `data/queries_layer1_n2n_pilot_final_v2.json` (417 queries).

**Changes made**:
- Fixed query ratio bug: was 2 social + 1 entity, now correctly 2 entity + 1 social pattern
- Fixed `type_map` mislabeling in parser
- Updated entity query guidelines:
  - Must not reveal specific details from source text
  - Must sound like a researcher who doesn't know where the answer is
  - Must require semantic matching to find the answer
  - Vary question form: Finns det / Vad är känt om / Vilka personer / Förekommer

**Core principle established**:
> A good query: (1) does not leak source details, (2) sounds like a blind researcher, (3) requires semantic matching.

---

## 2026-03-18 — Baseline Evaluation

**Status**: Complete.

**Result**: BGE-M3 base model achieves MRR ~0.56 on pilot dataset (no fine-tuning).
This is the baseline to beat with fine-tuning.
