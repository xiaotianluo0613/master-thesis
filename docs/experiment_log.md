# Experiment Log

Running log of experiments, results, and decisions. Most recent entry first.

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
