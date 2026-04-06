# Training Results

All runs tracked here. Add a row after every training + evaluation.

---

## Run Table

| Run ID | Date | Layer | Model base | Method | Train examples | Epochs | Batch size | LR | Temp | Val corpus | Val queries | MAP | nDCG@10 | nDCG@100 | Recall@10 | Recall@100 | Notes |
|--------|------|-------|-----------|--------|---------------|--------|------------|----|------|------------|-------------|-----|---------|----------|-----------|------------|-------|
| pilot-baseline | 2026-03-27 | Pilot | BAAI/bge-m3 | None (baseline) | — | — | — | — | — | 550 chunks | 87 queries | — | 0.1505 | — | — | — | N-to-N baseline |
| pilot-bge-unified | 2026-03-27 | Pilot | BAAI/bge-m3 | BGE unified full FT | 330 | 1 | 2 | 1e-5 | 0.02 | 550 chunks | 87 queries | — | 0.1908 | — | — | — | MRR@10: 0.357 (+31% vs baseline). GPL dropped after this. |
| pilot-gpl | 2026-03-27 | Pilot | BAAI/bge-m3 | GPL | 330 | 1 | 2 | 1e-5 | 0.02 | 550 chunks | 87 queries | — | 0.1800 | — | — | — | MRR@10: 0.332. Dropped — BGE unified outperforms. |
| layer1-baseline | 2026-04-05 | Layer 1 | BAAI/bge-m3 | None (baseline) | — | — | — | — | — | 7300 chunks | 551 queries | 0.0649 | 0.0967 | 0.1292 | 0.0923 | 0.1981 | Harder task: larger corpus, more queries |
| layer1-full-ft-v1 | 2026-04-05 | Layer 1 | BAAI/bge-m3 | BGE unified full FT | 19716 | 3 | 4 | 1e-5 | 0.02 | 7300 chunks | 551 queries | 0.1229 | 0.1828 | 0.2233 | 0.1647 | 0.2998 | +89% MAP vs baseline. Pilot hyperparams, unplanned. Job 4779908, runtime 2h53m. |
| layer1-lora-v1 | 2026-04-06 | Layer 1 | BAAI/bge-m3 | BGE unified LoRA | 19716 | 3 | 8 | 1e-4 | 0.02 | 7300 chunks | 551 queries | 0.0999 | 0.1453 | 0.1839 | 0.1328 | 0.2580 | r=16, alpha=32, Q+K+V. Batch=8 confound vs full FT (2x fewer steps). Job 4785689, runtime 1h8m. |
| layer1-lora-b4 | 2026-04-06 | Layer 1 | BAAI/bge-m3 | BGE unified LoRA | 19716 | 3 | 4 | 1e-4 | 0.02 | 7300 chunks | 551 queries | 0.1128 | 0.1655 | 0.2040 | 0.1496 | 0.2763 | r=16, alpha=32, Q+K+V. Same steps as full FT. +74% MAP vs baseline. |
| layer1-lora-dense-b4 | 2026-04-06 | Layer 1 | BAAI/bge-m3 | BGE unified LoRA | 19716 | 3 | 4 | 1e-4 | 0.02 | 7300 chunks | 551 queries | 0.1183 | 0.1749 | 0.2140 | 0.1567 | 0.2863 | r=16, alpha=32, Q+K+V+dense. +82% MAP. Only 4% gap vs full FT. **Recommended approach.** |

---

## Notes

- **Pilot metrics** used nDCG@10 + MRR@10 only. From Layer 1 onward: MAP + nDCG@{10,100} + Recall@{10,100}.
- **Baseline numbers differ between pilot and Layer 1** — not comparable. Different corpus size (550 vs 7300 chunks) and query count (87 vs 551). Absolute numbers drop with larger corpus.
- **layer1-full-ft-v1** trained with pilot hyperparameters (3 epochs, batch 4) without a prior plan. Serves as a full FT reference point for LoRA comparison.
- **layer1-lora-v1** used batch=8 → only half the gradient steps of full FT. Not a fair comparison.
- **layer1-lora-b4** fixes the confound (batch=4, same steps). Shows true capacity difference.
- **layer1-lora-dense-b4** adds output projection (dense) to LoRA targets. Best LoRA result — only 4% MAP gap vs full FT with 0.4% trainable params. Recommended approach for Layer 2+.
- All val splits: 10% val / 90% train, grouped by date (no document group leaks across splits).
