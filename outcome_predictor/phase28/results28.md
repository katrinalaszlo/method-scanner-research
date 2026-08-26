# Phase 28 results — paired within-method design

Pre-registered in `schema28.md`; evaluated 2026-08-26.

## Data
880 pairs (main vs variant, same paper / method / dataset) from 22 papers, 96 distinct variants
(2 variants dropped: the change-mapper returned no JSON twice). Mean |delta| = 27.9 score units
(D4RL normalized score; Atari 100×HNS). Change types assigned by the mapper: {'hyperparameter': 76, 'remove': 14, 'other': 5, 'replace': 1}.
Only 31 of 96 variants touch any operation according to the mapper; 65 have an empty affected set (mostly the
hyperparameter sweeps — TAP, ReBRAC, PLAS ε, EDAC η, SimPLe γ/rollout, COMBO β — which is the intended control group).

## Leave-one-paper-out, clipped Ridge (mean per-paper MAE on signed delta)

| model | features | mean paper MAE | row MAE | sign acc (\|δ\|≥2) | MAE on \|δ\| |
|---|---|---|---|---|---|
| A | mean delta | 39.81 | 36.25 | 0.766 | 33.46 |
| B | dataset + benchmark + change_type + main score | 28.45 | 27.21 | 0.752 | 23.97 |
| C | affected op@object tuples + count + fraction | 40.13 | 33.27 | 0.752 | 30.85 |
| D | B + C | 29.17 | 26.97 | 0.742 | 23.67 |

Wins: D<B on 12/22 papers; C<A on 14/22.

## Per-paper MAE

| paper | n | A | B | C | D |
|---|---|---|---|---|---|
| 1903.00374 | 175 | 30.2 | 28.8 | 22.1 | 23.5 |
| 2007.05929 | 26 | 33.4 | 22.3 | 28.4 | 21.3 |
| 2011.07213 | 35 | 26.8 | 9.5 | 16.4 | 9.0 |
| 2102.08363 | 26 | 26.5 | 25.0 | 24.7 | 24.8 |
| 2106.02039 | 9 | 24.1 | 14.1 | 14.8 | 13.0 |
| 2106.08909 | 21 | 18.6 | 8.5 | 11.8 | 8.5 |
| 2110.01548 | 46 | 24.5 | 20.3 | 31.3 | 25.0 |
| 2111.0021 | 118 | 88.5 | 66.5 | 88.8 | 68.2 |
| 2205.09991 | 12 | 26.4 | 10.8 | 77.2 | 21.3 |
| 2208.06193 | 6 | 16.6 | 21.1 | 45.5 | 23.9 |
| 2208.10291 | 180 | 23.7 | 16.0 | 20.1 | 17.2 |
| 2209.03993 | 18 | 23.3 | 6.5 | 12.4 | 5.6 |
| 2209.14548 | 27 | 20.3 | 16.1 | 18.8 | 13.4 |
| 2211.15657 | 8 | 12.2 | 12.4 | 20.6 | 12.7 |
| 2302.01877 | 6 | 22.8 | 13.6 | 10.9 | 11.7 |
| 2304.10573 | 9 | 20.5 | 9.0 | 8.6 | 7.2 |
| 2305.09836 | 108 | 27.5 | 20.5 | 24.7 | 21.8 |
| 2310.03022 | 12 | 19.0 | 12.3 | 8.9 | 10.7 |
| 2310.09615 | 1 | 90.1 | 71.6 | 100.3 | 80.4 |
| 2401.02644 | 21 | 19.3 | 9.2 | 9.4 | 7.6 |
| 2403.00564 | 6 | 160.7 | 110.3 | 163.5 | 112.9 |
| 2405.12399 | 10 | 120.9 | 101.5 | 123.3 | 102.0 |

## Verdicts
- **H1 `MAE(D) < MAE(B)` — not supported.** Adding which operations a change touches does not improve prediction of
  how much the score moves, beyond knowing the dataset, the main score and the mapper's coarse change type.
- **H2 `MAE(C) < MAE(A)` — not supported.** The affected-tuple features alone predict the delta no better than the
  mean delta of the training pairs.
- Sign accuracy ≈ 0.75 for every model including the mean baseline: most reported variants score lower than the
  main configuration, so sign is predictable without any feature.

## Caveats
- Change mapping is LLM-derived from short variant descriptions; the mapper labelled several genuine structural
  variants (PLAS+P perturbation layer, IDQL-1, TT uniform) as `hyperparameter` with no affected ops. That is the
  pipeline's output, not edited; it caps what C/D can see.
- Pairs are unbalanced (TAP 180, SimPLe 175, EfficientZero 118, ReBRAC 108 of 880); per-paper MAE is the headline.
- Main-vs-variant rows sometimes differ in seeds (ReBRAC 10 vs 4) or table (EfficientZero Tables 1 vs 8/9).

## Reading across Phases 24–28
Cross-method (24–27): `operation@object` carries lineage-level information about a method's score band; nothing
about its ranking within a dataset. Within-method (28): the tuples a change touches carry nothing about the size of
the change. The representation, as a bag of typed operations, does not encode what makes one variant of a method
better than another. Structure-as-tuples is a taxonomy, not a predictor — consistent with what Phase 23 found by
experiment (the surprises were hyperparameter × schedule interactions the ontology could not carry).
