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

---

## Notes

- **Pilot metrics** used nDCG@10 + MRR@10 only. From Layer 1 onward: MAP + nDCG@{10,100} + Recall@{10,100}.
- **Baseline numbers differ between pilot and Layer 1** — not comparable. Different corpus size (550 vs 7300 chunks) and query count (87 vs 551). Absolute numbers drop with larger corpus.
- **layer1-full-ft-v1** trained with pilot hyperparameters (3 epochs, batch 4) without a prior plan. Serves as a full FT reference point for LoRA comparison.
- All val splits: 10% val / 90% train, grouped by date (no document group leaks across splits).
