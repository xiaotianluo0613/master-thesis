# Project TODO

Checked off when done, kept for record. Most urgent first within each section.

---

## 🔥 Now — Layer 1 Training

- [ ] `scp` layer1 data files to UPPMAX
- [ ] Write SLURM job: `split_train_val.py` (Layer 1 queries → train/val split)
- [ ] Write SLURM job: `mine_hard_negatives_bge.py` (7 negatives, ranks 2–200)
- [ ] Write SLURM job: `score_bge_integration.py` (teacher scoring)
- [ ] Write SLURM job: fine-tune BGE-M3 unified (Layer 1, full FT first, then LoRA)
- [ ] Write SLURM job: `evaluate_comparison.py` (Layer 1 val set)
- [ ] Update experiment log with Layer 1 results

---

## 🎯 Demo — RAG System (job portfolio)

- [ ] Design RAG pipeline: fine-tuned BGE-M3 retrieval → LLM answer in modern Swedish
- [ ] Build `scripts/rag/retrieve.py` — top-k chunk retrieval with fine-tuned model
- [ ] Build `scripts/rag/answer.py` — LLM call (Claude API) with retrieved context
- [ ] Build `scripts/rag/demo.py` — CLI demo: input query → answer + archive citations
- [ ] Add demo section to README with example query/answer
- [ ] (Stretch) Streamlit web UI for the demo

---

## 📐 LoRA (when FlagEmbedding supports it)

- [ ] Monitor FlagEmbedding repo for LoRA support in encoder-only m3 trainer
- [ ] OR: patch FlagEmbedding / switch to custom PEFT training loop
- [ ] Re-run Layer 1 training with LoRA, compare to full FT baseline

---

## 📚 Remaining Curriculum Layers

- [ ] Layer 2: build chunks + generate queries (District + Protocols)
- [ ] Layer 3: build chunks + generate queries (Legal)
- [ ] Layer 4: build chunks + generate queries (City — noisy OCR)
- [ ] Cumulative training: Layer 1 → 2 → 3 → 4

---

## 🎓 Thesis

- [ ] Run test set evaluation (ONCE — after all training done)
- [ ] Write Results chapter
- [ ] Write Discussion chapter
- [ ] Final submission — June 2026
