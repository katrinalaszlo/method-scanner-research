# Phase 30 — Pairwise outcome prediction on Papers-with-Code leaderboards

Pre-registered two holdouts and three hypotheses (schema30.md, amendments 1–4).

## Data
- Leaderboard selection: 25 boards (seed-0 random subsample to 40 papers max per board), 17 tasks, 900 papers, 5000 pairs.
- Extraction: 893/900 records, 7 failures (Qwen3:8b on title + abstract, Phase 27 prompt, closed ontology `op@object`).
- Feature sets: T (year), L (PwC method lineage, 109 collections), E (nomic embedding, 768-d), S (scanner tuples, 2457 distinct).

## Results

### Board-held-out accuracy (leave-one-board-out; transfer case)
| Feature set | Acc | Per-board mean |
|---|---|---|
| chance | 0.500 | 0.500 |
| T | 0.711 | 0.711 |
| L | 0.565 | 0.566 |
| E | 0.665 | 0.665 |
| S | 0.603 | 0.603 |
| T+L | 0.712 | 0.712 |
| T+E | 0.725 | 0.726 |
| T+S | 0.710 | 0.711 |
| T+L+S | 0.711 | 0.711 |
| T+L+E | 0.718 | 0.719 |
| **T+L+E+S** | **0.712** | **0.712** |

### Within-board accuracy (pairs split 5-fold; same board at train/test time)
| Feature set | Acc | Per-board mean |
|---|---|---|
| chance | 0.500 | 0.500 |
| T | 0.711 | 0.711 |
| L | 0.614 | 0.614 |
| E | 0.790 | 0.790 |
| S | 0.873 | 0.873 |
| T+L | 0.728 | 0.729 |
| T+E | 0.816 | 0.816 |
| T+S | 0.882 | 0.882 |
| T+L+S | 0.889 | 0.889 |
| T+L+E | 0.829 | 0.829 |
| **T+L+E+S** | **0.898** | **0.898** |

## Pre-registered hypotheses

| H | Condition | Result |
|---|---|---|
| H1 | acc(T+L+S) > acc(T+L) on board-held-out | 0.711 > 0.712? **FALSE** |
| H2 | acc(S) > 0.5 on board-held-out | 0.603 > 0.5? **TRUE** |
| H3 | acc(T+L+E+S) > acc(T+L+E) on board-held-out | 0.712 > 0.718? **FALSE** |

## Interpretation

**Transfer (board-held-out):** Year (T) alone predicts 71.1% accuracy — a strong baseline. Adding embedding (E) reaches 72.5%; adding lineage (L) or structure (S) does not improve transfer performance. S by itself reaches 60.3% (H2: yes, above chance), but the combination T+L+E+S regresses to 71.2%, suggesting S overfits to the training boards and does not generalize.

**Within-board:** Structure is extremely strong (87.3% vs 60.1% board-held-out). Combined with year and lineage, T+L+S reaches 88.9%, and adding embedding (T+L+E+S) yields 89.8%. This sharp divergence — 0.603 transfer, 0.873 within-board — indicates that Method Scanner's `op@object` vocabulary captures board/task-specific structure but does not transfer across domains.

**Why S doesn't transfer:** The 2457 distinct tuples are multicollinear with the task. Papers on ImageNet use different operations than papers on COCO or Kinetics. The scanner extracts genuine operations, but their predictive power is local to each board's methodology ecosystem. Year (0.711) and abstract semantics (0.665) transfer better because they are domain-agnostic.

**Lineage (L) is weak:** 56.5% board-held-out, 61.4% within-board. The PwC method tags reach only 55% of papers; the top tag is spam-contaminated. Lineage adds little even in combination.

## Caveats

1. **7 failures in extraction:** Qwen3:8b rejected 7 abstracts (e.g., non-English, corrupted UTF-8). Evaluation covers 893/900.
2. **Subsampling bias:** Boards with >40 papers were randomly subsampled (seed 0). Large boards (e.g., ImageNet 342 papers) lost papers; this may underestimate the benefit of S within those boards.
3. **Structure input:** Scanner runs on title + abstract only (median 175 words), not the full method section. Phase 27 extraction on full-text sections reached ~25 ops/paper; abstracts yield ~13.5 ops/paper on average.
4. **Vocabulary:** 2457 tuples across 893 papers (mean 13.5 per paper) is sparse and task-specific. Denser, cross-task vocabularies might transfer better.
5. **Model:** Logistic regression with L2, C ∈ {0.01, 0.1, 1, 10}, tuned inside training on 5-fold CV. Simple model; complex learners might extract more structure signal.

## Follow-up

The structure-transfer problem is interesting: Method Scanner identifies operations, but operation vocabularies are task-local. Future work could:
- Cluster operations into coarser abstractions (e.g., "compute something @ latent representation") to reduce task-specificity.
- Test on cross-domain pairs (e.g., predict which of two papers from different tasks scores higher on their respective benchmarks).
- Run extraction on full-text method sections (⇒ denser tuples, higher cost).
- Try structure-aware embeddings that bind `op@object` to semantic space.

For now: **method structure does not predict outcomes across domains, but predicts strongly within domain.** The signal exists (H2 confirmed), but is not generalizable.
