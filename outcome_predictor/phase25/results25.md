# Phase 25 results — exploratory follow-up (not pre-registered as Phase 24)

Same 530 rows / 27 methods / leave-one-method-out as Phase 24. Changes fixed in `schema25.md` before running:
structure vocabulary variants (strict / relaxed / bare), Ridge with alpha ≥ 1 and predictions clipped to the training
outcome range, kNN with `dataset` as its own equal-weight distance group.

## Aggregate (mean per-method MAE; median in parentheses)

| variant | predictor | vocab | shared≥2 | A | B | C | D | E | D<B | D<E | B<A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| strict | ridge_clipped | 80 | 11 | 26.3 (24.6) | 24.5 (19.7) | 26.5 (24.7) | 22.4 (13.4) | 24.1 (18.0) | 22/27 | 20/27 | 22/27 |
| strict | knn_dataset_first | 80 | 11 | 26.3 (24.6) | 14.7 (13.4) | 26.0 (25.4) | 14.8 (13.4) | 15.8 (14.6) | 3/27 | 11/27 | 25/27 |
| relaxed | ridge_clipped | 75 | 14 | 26.3 (24.6) | 24.5 (19.7) | 26.7 (24.8) | 21.9 (13.0) | 24.1 (18.0) | 22/27 | 21/27 | 22/27 |
| relaxed | knn_dataset_first | 75 | 14 | 26.3 (24.6) | 14.7 (13.4) | 27.1 (27.5) | 14.8 (13.4) | 15.8 (14.6) | 4/27 | 11/27 | 25/27 |
| bare | ridge_clipped | 53 | 14 | 26.3 (24.6) | 24.5 (19.7) | 26.6 (24.8) | 23.5 (13.0) | 24.1 (18.0) | 20/27 | 19/27 | 22/27 |
| bare | knn_dataset_first | 53 | 14 | 26.3 (24.6) | 14.7 (13.4) | 27.6 (27.5) | 14.8 (13.4) | 15.8 (14.6) | 11/27 | 15/27 | 25/27 |

## Reading

- **Clipping fixes the Phase 24 Ridge failure.** With predictions bounded to the training range, conditions-only B drops
  from 37.9 to 24.5 and beats the mean baseline; D = 22.4 (strict) / 21.9 (relaxed) / 23.5 (bare), winning on 20–22 of 27
  methods (median gain ≈ 2–3 points). Lineage E = 24.1; D beats E on 19–21/27.
- **Dataset identity is the dominant predictor.** Giving `dataset` its own kNN distance group takes B from 27.3 to 14.7
  — the 5 nearest same-dataset rows of other methods predict a held-out method's score within ~15 points. Under that
  predictor structure adds nothing (D 14.8, ties on most methods); lineage costs a point (E 15.8).
- **Vocabulary relaxation barely matters.** Relaxed types share 14/75 tuples vs 11/80 strict; bare ops 14/53. The
  clipped-Ridge D gain is ~2.5 points regardless of variant, and kNN is indifferent. Structure alone (C) never beats A.
- **Net:** with a bounded linear model, `operation@object` structure gives a small, fairly consistent improvement over
  conditions alone and over lineage on held-out D4RL methods (≈2–3 MAE points, 20–22/27 methods). With the stronger
  nearest-neighbour predictor that improvement disappears, because same-dataset neighbours already explain most of the
  variance and structure cannot separate methods within a dataset. This is exploratory: the Phase 24 verdict stands;
  a pre-registered Phase 26 with the clipped Ridge as the primary predictor is the honest next test, on a second benchmark.

## Per-method MAE — relaxed / ridge_clipped

| method | A | B | C | D | E |
|---|---|---|---|---|---|
| AdaptDiffuser | 24.6 | 19.4 | 24.8 | 15.9 | 16.7 |
| COMBO | 28.7 | 17.8 | 28.7 | 8.5 | 18.0 |
| CQL | 35.8 | 22.6 | 35.7 | 17.2 | 22.2 |
| DecisionConvFormer | 20.7 | 11.7 | 20.9 | 8.8 | 11.4 |
| DecisionDiffuser | 21.8 | 27.1 | 22.1 | 15.1 | 20.0 |
| DecisionTransformer | 19.2 | 6.7 | 19.2 | 10.6 | 6.8 |
| Diffuser | 19.4 | 48.9 | 19.4 | 39.7 | 48.3 |
| DiffusionQL | 26.4 | 9.3 | 28.0 | 6.5 | 8.6 |
| EDAC | 30.0 | 26.4 | 30.8 | 20.7 | 23.9 |
| EDP | 24.4 | 14.2 | 24.4 | 10.8 | 14.1 |
| FisherBRC | 35.5 | 19.7 | 35.6 | 13.0 | 19.5 |
| HierarchicalDiffuser | 25.4 | 12.9 | 25.7 | 8.1 | 13.6 |
| IDQL | 20.8 | 9.9 | 21.5 | 8.7 | 8.1 |
| IQL | 17.6 | 8.4 | 17.8 | 5.0 | 8.1 |
| MOPO | 37.1 | 25.9 | 43.9 | 37.3 | 26.9 |
| MOReL | 26.3 | 89.5 | 26.3 | 89.5 | 89.5 |
| OneStepRL | 30.6 | 25.8 | 30.6 | 9.8 | 24.9 |
| PLAS | 40.3 | 25.6 | 40.1 | 22.5 | 25.9 |
| QDT | 19.5 | 9.6 | 18.4 | 6.2 | 10.7 |
| QGPO | 23.6 | 52.9 | 23.4 | 86.8 | 64.9 |
| ReBRAC | 29.1 | 20.0 | 29.1 | 17.9 | 20.9 |
| RvS | 31.9 | 78.8 | 31.8 | 72.2 | 76.2 |
| SRPO | 21.9 | 20.4 | 22.1 | 14.9 | 16.7 |
| SfBC | 21.9 | 9.2 | 21.9 | 6.7 | 9.2 |
| TAP | 21.7 | 8.9 | 23.8 | 9.9 | 9.3 |
| TD3+BC | 34.9 | 20.4 | 34.9 | 13.0 | 20.6 |
| TrajectoryTransformer | 21.1 | 19.1 | 21.3 | 15.1 | 15.1 |

## Per-method MAE — strict / knn_dataset_first

| method | A | B | C | D | E |
|---|---|---|---|---|---|
| AdaptDiffuser | 24.6 | 15.8 | 18.7 | 15.8 | 16.0 |
| COMBO | 28.7 | 13.5 | 28.7 | 13.4 | 42.8 |
| CQL | 35.8 | 17.8 | 37.2 | 17.8 | 18.4 |
| DecisionConvFormer | 20.7 | 13.4 | 17.3 | 13.4 | 25.6 |
| DecisionDiffuser | 21.8 | 15.0 | 25.4 | 14.7 | 15.0 |
| DecisionTransformer | 19.2 | 8.3 | 19.5 | 8.3 | 14.6 |
| Diffuser | 19.4 | 9.6 | 28.7 | 10.0 | 12.7 |
| DiffusionQL | 26.4 | 2.6 | 20.7 | 2.6 | 2.6 |
| EDAC | 30.0 | 20.3 | 32.7 | 20.3 | 20.3 |
| EDP | 24.4 | 10.1 | 18.9 | 10.1 | 6.5 |
| FisherBRC | 35.5 | 14.5 | 35.9 | 14.5 | 12.3 |
| HierarchicalDiffuser | 25.4 | 9.3 | 19.6 | 11.7 | 15.6 |
| IDQL | 20.8 | 8.2 | 18.2 | 8.2 | 7.0 |
| IQL | 17.6 | 11.5 | 17.2 | 11.5 | 10.4 |
| MOPO | 37.1 | 28.7 | 29.7 | 30.7 | 22.4 |
| MOReL | 26.3 | 16.8 | 28.9 | 16.8 | 16.8 |
| OneStepRL | 30.6 | 15.9 | 33.0 | 14.2 | 12.3 |
| PLAS | 40.3 | 12.8 | 46.4 | 12.8 | 17.4 |
| QDT | 19.5 | 8.3 | 29.6 | 8.3 | 15.0 |
| QGPO | 23.6 | 11.7 | 18.0 | 11.7 | 3.0 |
| ReBRAC | 29.1 | 25.7 | 27.5 | 25.7 | 37.1 |
| RvS | 31.9 | 35.4 | 36.9 | 35.4 | 35.4 |
| SRPO | 21.9 | 22.6 | 16.0 | 22.6 | 5.7 |
| SfBC | 21.9 | 11.5 | 21.5 | 12.3 | 9.6 |
| TAP | 21.7 | 9.4 | 22.1 | 9.4 | 8.8 |
| TD3+BC | 34.9 | 14.6 | 34.6 | 14.6 | 11.8 |
| TrajectoryTransformer | 21.1 | 12.5 | 20.0 | 12.5 | 12.5 |

## 25b — within-dataset ranking (`within_dataset.py`)

Per D4RL dataset (18, all ≥5 rows), Spearman between held-out predictions and actual scores across rows:

| predictor | A | B | C | D | E |
|---|---|---|---|---|---|
| relaxed / clipped Ridge (mean ρ / median) | −0.34 / −0.48 | −0.06 / −0.02 | −0.35 / −0.43 | −0.07 / −0.08 | −0.02 / −0.01 |
| strict / dataset-first kNN | −0.34 / −0.48 | 0.11 / 0.05 | −0.15 / −0.11 | 0.10 / 0.03 | 0.13 / 0.14 |

(A's negative ρ is a leave-one-out artifact: the training mean moves against the held-out method's level.)
**No model ranks methods within a dataset.** The 2–3-point D gain under clipped Ridge is a level effect
(predicting a method's typical score band), not an ordering effect. Structure alone (C) ranks no better
than the mean baseline.
