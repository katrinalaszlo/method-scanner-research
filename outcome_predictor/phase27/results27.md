# Phase 27 results — full-operation extraction

Pre-registered in `schema27.md`. Extraction run 2026-08-25/26; evaluated 2026-08-26.

## Extraction

44 papers (27 D4RL, 17 Atari), single-pass prompt with the closed 15-type object ontology, 2500-word sections,
`qwen3:8b`, 0 failures (one JSON salvage). Median 25 operations and 15 distinct tuples per paper (Phase 24: ~4 and
3.7). 89 objects outside the ontology dropped and counted; 70% of quotes found verbatim in the section.

| | D4RL (27 methods) | Atari (17) |
|---|---|---|
| vocabulary, strict | 80 → **242** | 54 → **131** |
| tuples shared by ≥2 methods | 11 → **55** | 9 → **42** |
| tuples per method | 3.7 → 14.7 | 4.2 → 14.3 |

## Mean per-method MAE (strict vocab; relaxed in `results27.json`)

| benchmark | predictor | structures | A | B | C | D | E | D<B | D<E | C<A |
|---|---|---|---|---|---|---|---|---|---|---|
| D4RL | clipped Ridge | original | 26.3 | 24.5 | 26.5 | 22.4 | 24.1 | 22/27 | 20/27 | 9/27 |
| D4RL | clipped Ridge | **phase27** | 26.3 | 24.5 | **24.8** | **22.0** | 24.1 | 21/27 | 20/27 | **20/27** |
| D4RL | dataset-first kNN | original | 26.3 | 14.7 | 26.0 | 14.8 | 15.8 | 3/27 | 11/27 | 15/27 |
| D4RL | dataset-first kNN | phase27 | 26.3 | 14.7 | 25.6 | 14.7 | 15.8 | 9/27 | 11/27 | 18/27 |
| Atari | clipped Ridge | original | 114.8 | 111.0 | 116.3 | 87.9 | 115.4 | 11/17 | 11/17 | 9/17 |
| Atari | clipped Ridge | **phase27** | 114.8 | 111.0 | **108.9** | 88.2 | 115.4 | 11/17 | 11/17 | 10/17 |
| Atari | dataset-first kNN | original | 114.8 | 79.0 | 101.0 | 78.5 | 73.5 | 3/17 | 4/17 | 13/17 |
| Atari | dataset-first kNN | phase27 | 114.8 | 79.0 | 96.5 | **75.2** | 73.5 | 6/17 | 4/17 | 13/17 |

## Verdicts

- **H3 `MAE(C) < MAE(A)` — weakly supported once overlap exists.** Structure alone now beats the training mean
  on both benchmarks under Ridge (D4RL 24.8 vs 26.3, 20/27 methods; Atari 108.9 vs 114.8, 10/17). With the
  original ~4-tuple structures it never did. The gain is small (≈1.5 points on D4RL, ≈6 on Atari's HNS×100).
- **H1 / H2 — unchanged.** Structure + conditions is not better than before (D4RL 22.0 vs 22.4; Atari 88.2 vs
  87.9). Under the nearest-neighbour predictor the only movement is Atari D 78.5 → 75.2 (6/17 wins); lineage
  (E 73.5) is still as good or better there.
- **Fixing overlap did not change the answer.** Five times more shared tuples and four times more tuples per
  method produce the same level of prediction. The information that `operation@object` carries about outcomes
  is real but shallow: about what kind of method this is (lineage resolution), captured either by the sparse
  structure, the full structure, or the hand label, and mostly redundant with dataset identity + conditions.

## What this closes and what it leaves open

Closed: the "vocabulary is too sparse" explanation for Phases 24–26. It was true and fixing it helped C, not D.

Open: (1) whether structure predicts *changes* — the Phase 23-style question "if I swap component X, does
the score move" — which is a within-method, paired design, not a cross-method regression; (2) a structure
representation with order/graph information rather than a bag of tuples; (3) a benchmark whose outcome
variance is not dominated by dataset identity.

Files: `records/<id>.json` (ops, objects, quotes, grounding), `extract_full.py`, `eval27.py`, `results27.json`, `extract.log`.
