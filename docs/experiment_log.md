# Experiment Log

Running log of experiments, results, and decisions. Most recent entry first.

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
- Baseline MRR (0.27) is lower than earlier baseline eval (0.56 on 2026-03-18) — likely different val set/corpus; needs investigation

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
