# Phase 24 results — Outcome predictor on D4RL locomotion

**Pre-registered test (schema.md frozen before evaluation). 2026-08-25.**

## Abstract

530 experiment rows from 28 papers (27 held-out methods) were assembled: exact method + exact
configuration + exact reported D4RL normalized return, each with verbatim table provenance. Structure
(`operation@object`) came from the existing Method Scanner pipeline run on every paper. Five models
were evaluated leave-one-method-out with Ridge (primary) and kNN k=5 (robustness).

**Primary hypothesis `MAE(D) < MAE(B)` — not supported on the pre-registered aggregate.** Ridge:
mean per-method MAE 59.9 (structure + conditions) vs 37.9 (conditions only); kNN: 27.3 vs 27.3.
Structure alone (C) is indistinguishable from the training-mean baseline (Ridge 26.5 vs 26.3; kNN
26.0 vs 26.3). **Secondary hypothesis `MAE(D) < MAE(E)` — not supported** (Ridge 59.9 vs 43.8; kNN
27.3 vs 27.0).

The Ridge aggregate is dominated by two held-out methods where every conditioned model extrapolates
catastrophically (MOReL, QGPO: MAE 400–600 for B, D and E alike). By per-method win count, Ridge D
beats B on 21/27 methods and E on 19/27, with a median per-method improvement of 2.8 and 2.3 points.
That pattern is not stable across predictors: kNN D beats B on 3/27. The supportable reading is that
the structure feature is inert under kNN and, under Ridge, gives a small median gain that is swamped by
the linear model's failure on methods whose condition values lie outside the training range.

## Dataset

- `d4rl_dataset.csv`: 530 rows, 27 `method_id`s, 28 `paper_id`s (27 arXiv papers + the Phase 23 sweep).
- Outcome: D4RL normalized return, −0.2 to 153.9. Training-mean baseline row MAE 27.0.
- `dataset_version`: v2 375 rows, unknown 125, v0 30.
- 18 dataset identities; 9 core (halfcheetah/hopper/walker2d × medium/medium-replay/medium-expert)
  hold 428 rows; random 54, expert 30, full-replay 18.
- Rows per method: ReBRAC 72, EDP 63, EDAC 36, OneStepRL 36, TAP 36, TD3+BC 30, Diffuser 24
  (9 paper + 15 Phase 23), DiffusionQL 18, IDQL 18, TrajectoryTransformer 18, CQL 15, FisherBRC 15,
  COMBO 12, DecisionConvFormer 12, MOPO 12, MOReL 12, RvS 12, PLAS 11, and 9 each for AdaptDiffuser,
  DecisionDiffuser, DecisionTransformer, HierarchicalDiffuser, IQL, QGPO, SRPO, SfBC; QDT 6.
- Excluded (counted in `sources/build_audit.json`): AWAC entirely (figure-only numbers); raw-return
  tables (CQL, MOPO); ablations whose varied parameter has no schema column (COMBO β, QDT α, EDAC η
  sweep, TAP β/L, DD α, HD K, PLAS ε, ReBRAC penalty-off); structural variants (BC-/BCQ-Diffusion,
  CondDiffuser, SfBC±, TAP decoder/planner variants, one-step multi-step/iterative); online
  fine-tuning and delayed-reward rows; ReBRAC FeatureNorm and 4-seed rerun; PLAS hopper-medium-expert
  (a)/(b) (ambiguous dataset); rows the extractors flagged as garbled (EDAC Table 9 cells, etc.).
- Phase 23: 15 rows (3 envs × {ddpm-20, ddim-20/10/5/2}), seed-averaged, under `method_id=Diffuser`.

## Structure

One structure per method, from the scanner (`qwen3:8b` extraction + v2 typing, no manual edits).
Vocabulary across the 27 methods: 80 tuples; only 11 tuples are shared by ≥2 methods
(`predict@distribution` ×4; `predict@trajectory`, `reconstruct@trajectory`, `improve@policy`,
`encode@input`, `sample@action`, `correct@parameter` ×3; `train@parameter`, `evaluate@value`,
`sample@trajectory`, `reconstruct@sample` ×2). Everything else is unique to one method and therefore
carries no information at held-out time. Examples:

| method | structure |
|---|---|
| Diffuser | predict@trajectory, reconstruct@trajectory, sample@trajectory |
| DecisionDiffuser | diffuse@state, inverse dynamics@action, predict@distribution, sample@state |
| DiffusionQL | improve@parameter, reconstruct@sample, sample@sample |
| CQL | correct@value, evaluate@value, minimize q-values@value |
| IQL | correct@parameter, predict@distribution, predict@value |
| TD3+BC | normalize@state, regularize@policy, scale@gradient |
| OneStepRL | evaluate@value, improve@policy |
| DecisionTransformer | append@other, decrement@reward, encode@input, interleave@sample, predict@action, select@action |
| TrajectoryTransformer | beam search@trajectory, discretize@state, reconstruct@trajectory, tokenize@state |
| MOPO | bound return@other, maximize return@other, predict@distribution |

Section detection was wrong for five papers (DD, RvS, Fisher-BRC, QGPO, Hierarchical Diffuser) and
they were re-extracted with the method section pinned before evaluation; replaced records are in
`superseded_records/`. QDT typing needed a generic parser fix in `semantic_type.py` (model emitted a
stray quote before a JSON key). TD3+BC, ReBRAC, IDQL, SRPO produced no canonical ops (raw verbs kept).

## Feature schema

See `schema.md`. 15 numeric conditions (+ per-column {present, missing, not_applicable} state),
6 categorical conditions, plus `dataset` and `dataset_version`; 8 lineage labels; alpha grid
{0.01 … 1000} by grouped CV; kNN k=5 unweighted, equal-weight group distances.

## Aggregate results

Headline aggregate = mean of per-method MAE (each held-out method counts once).

### Ridge (primary)

| model | row MAE | mean per-method MAE | median per-method MAE | RMSE | corr |
|---|---|---|---|---|---|
| A mean | 26.95 | 26.30 | 24.58 | 31.3 | −0.30 |
| B conditions | 31.09 | 37.88 | 19.66 | 72.5 | 0.20 |
| C structure | 26.86 | 26.45 | 24.59 | 31.3 | 0.00 |
| **D structure+conditions** | **40.12** | **59.90** | **13.44** | 125.0 | 0.12 |
| E lineage+conditions | 34.54 | 43.80 | 18.02 | 94.6 | 0.14 |

Wins: D<B on 21/27 methods, D<E on 19/27, B<A on 22/27, C<A on 9/27. Median(D−B) = −2.8,
median(D−E) = −2.3 points.

### kNN k=5 (robustness)

| model | row MAE | mean per-method MAE | median per-method MAE | RMSE | corr |
|---|---|---|---|---|---|
| A mean | 26.95 | 26.30 | 24.58 | 31.3 | −0.30 |
| B conditions | 27.87 | 27.27 | 24.05 | 35.2 | 0.22 |
| C structure | 26.41 | 26.02 | 25.41 | 32.6 | −0.01 |
| **D structure+conditions** | **28.10** | **27.33** | **23.51** | 35.6 | 0.19 |
| E lineage+conditions | 27.82 | 26.97 | 24.69 | 34.3 | 0.25 |

Wins: D<B on 3/27 (ties on most: adding structure did not change the 5 nearest neighbours), D<E on
6/27, B<A 15/27, C<A 15/27. Median(D−B) = 0.

(Correlation of the mean baseline is negative because the training mean shifts against the held-out
method's level; it is reported for completeness, not interpreted.)

## Per-method MAE

### Ridge

| method | n | A | B | C | D | E |
|---|---|---|---|---|---|---|
| AdaptDiffuser | 9 | 24.6 | 19.4 | 24.6 | **17.9** | 16.7 |
| COMBO | 12 | 28.7 | 17.8 | 28.7 | **8.4** | 18.0 |
| CQL | 15 | 35.8 | 22.6 | 33.9 | **17.3** | 22.2 |
| DecisionConvFormer | 12 | 20.7 | 11.7 | 20.8 | **9.1** | 11.4 |
| DecisionDiffuser | 9 | 21.8 | 27.1 | 23.5 | **19.0** | 20.0 |
| DecisionTransformer | 9 | 19.2 | 6.7 | 19.2 | 9.9 | 6.8 |
| Diffuser | 24 | 19.4 | 54.4 | 18.9 | 65.0 | 56.1 |
| DiffusionQL | 18 | 26.4 | 9.3 | 27.5 | **6.5** | 8.6 |
| EDAC | 36 | 30.0 | 26.4 | 31.8 | **21.0** | 23.9 |
| EDP | 63 | 24.4 | 14.2 | 20.9 | **10.8** | 14.1 |
| FisherBRC | 15 | 35.5 | 19.7 | 35.5 | **13.0** | 19.5 |
| HierarchicalDiffuser | 9 | 25.4 | 12.9 | 25.7 | **11.6** | 13.6 |
| IDQL | 18 | 20.8 | 9.9 | 21.3 | 8.4 | 8.1 |
| IQL | 9 | 17.6 | 8.4 | 17.7 | **5.0** | 8.1 |
| MOPO | 12 | 37.1 | 25.9 | 42.2 | 37.6 | 26.9 |
| MOReL | 12 | 26.3 | 438.6 | 26.1 | 593.1 | 592.5 |
| OneStepRL | 36 | 30.6 | 25.8 | 30.6 | **9.9** | 24.9 |
| PLAS | 11 | 40.3 | 25.6 | 40.2 | **22.9** | 25.9 |
| QDT | 6 | 19.5 | 9.6 | 18.8 | **6.1** | 10.7 |
| QGPO | 9 | 23.6 | 55.0 | 23.4 | 570.6 | 82.9 |
| ReBRAC | 72 | 29.1 | 20.0 | 29.4 | **17.5** | 20.9 |
| RvS | 12 | 31.9 | 83.8 | 30.5 | 76.7 | 80.0 |
| SRPO | 9 | 21.9 | 20.4 | 22.0 | **14.7** | 16.7 |
| SfBC | 9 | 21.9 | 9.2 | 21.9 | **6.7** | 9.2 |
| TAP | 36 | 21.7 | 8.9 | 22.8 | 10.2 | 9.3 |
| TD3+BC | 30 | 34.9 | 20.4 | 35.0 | **13.4** | 20.6 |
| TrajectoryTransformer | 18 | 21.1 | 19.1 | 21.4 | **15.1** | 15.1 |

Bold = D is the best of B/D/E. Alpha chosen for D was 1.0 for every held-out method; for B mostly 10,
for C mostly 0.01 (structure has almost no shared support, so CV cannot regularize it meaningfully).

### kNN

| method | n | A | B | C | D | E |
|---|---|---|---|---|---|---|
| AdaptDiffuser | 9 | 24.6 | 15.8 | 18.7 | 15.8 | 15.8 |
| COMBO | 12 | 28.7 | 42.8 | 28.7 | 42.8 | 42.8 |
| CQL | 15 | 35.8 | 28.2 | 37.2 | 28.2 | 33.9 |
| DecisionConvFormer | 12 | 20.7 | 32.6 | 17.3 | 30.6 | 32.6 |
| DecisionDiffuser | 9 | 21.8 | 53.1 | 25.4 | 53.1 | 53.1 |
| DecisionTransformer | 9 | 19.2 | 20.8 | 19.5 | 20.8 | 20.8 |
| Diffuser | 24 | 19.4 | 15.3 | 28.7 | 15.4 | 28.7 |
| DiffusionQL | 18 | 26.4 | 22.6 | 20.7 | 22.6 | 22.6 |
| EDAC | 36 | 30.0 | 24.7 | 32.7 | 24.7 | 24.7 |
| EDP | 63 | 24.4 | 14.0 | 18.9 | 14.0 | 14.0 |
| FisherBRC | 15 | 35.5 | 19.2 | 35.9 | 38.4 | 30.1 |
| HierarchicalDiffuser | 9 | 25.4 | 16.7 | 19.6 | 16.7 | 16.7 |
| IDQL | 18 | 20.8 | 12.3 | 18.2 | 12.3 | 7.0 |
| IQL | 9 | 17.6 | 26.5 | 17.2 | 26.5 | 43.9 |
| MOPO | 12 | 37.1 | 22.4 | 29.7 | 22.4 | 22.4 |
| MOReL | 12 | 26.3 | 23.5 | 28.9 | 23.5 | 23.5 |
| OneStepRL | 36 | 30.6 | 26.4 | 33.0 | 29.2 | 26.4 |
| PLAS | 11 | 40.3 | 22.8 | 46.4 | 22.8 | 16.4 |
| QDT | 6 | 19.5 | 18.3 | 29.6 | 18.3 | 18.3 |
| QGPO | 9 | 23.6 | 32.3 | 18.0 | 16.5 | 16.5 |
| ReBRAC | 72 | 29.1 | 48.7 | 27.5 | 48.7 | 47.6 |
| RvS | 12 | 31.9 | 39.7 | 36.9 | 39.7 | 39.7 |
| SRPO | 9 | 21.9 | 53.4 | 16.0 | 53.4 | 25.3 |
| SfBC | 9 | 21.9 | 17.0 | 21.5 | 22.8 | 22.8 |
| TAP | 36 | 21.7 | 24.0 | 22.1 | 24.0 | 19.5 |
| TD3+BC | 30 | 34.9 | 35.0 | 34.6 | 35.0 | 35.1 |
| TrajectoryTransformer | 18 | 21.1 | 28.0 | 20.0 | 19.7 | 28.0 |

## Verdicts

| comparison | Ridge (mean per-method MAE) | kNN | verdict |
|---|---|---|---|
| B vs A — do conditions help? | 37.9 vs 26.3 (worse; wins 22/27) | 27.3 vs 26.3 (worse; 15/27) | **No, not on aggregate.** Conditions help on most methods under Ridge but blow up on 3–4 methods; under kNN they are a wash. |
| C vs A — structure alone | 26.5 vs 26.3 | 26.0 vs 26.3 | **No signal.** Structure alone ≈ guessing the mean. |
| **D vs B — primary** | 59.9 vs 37.9 (worse; wins 21/27, median −2.8) | 27.3 vs 27.3 (ties; wins 3/27) | **Not supported.** |
| D vs E — mechanism vs lineage | 59.9 vs 43.8 (worse; wins 19/27, median −2.3) | 27.3 vs 27.0 (wins 6/27) | **Not supported.** |

## Prediction-vs-actual examples (Ridge D)

Held-out IQL (well-behaved case):

| dataset | actual | predicted |
|---|---|---|
| halfcheetah-medium | 47.4 | 50.4 |
| hopper-medium | 66.3 | 75.7 |
| walker2d-medium | 78.3 | 77.9 |
| hopper-medium-replay | 94.7 | 83.4 |
| hopper-medium-expert | 91.5 | 101.2 |
| walker2d-medium-expert | 109.6 | 103.6 |

Held-out Diffuser, Phase 23 sampler sweep (model predicts the dataset level roughly but is blind to
the sampler/step effect — all five configs land within 4 points of each other):

| dataset | config | actual | predicted |
|---|---|---|---|
| hopper-medium | ddpm-20 | 47.5 | 87.7 |
| hopper-medium | ddim-10 | 83.5 | 90.7 |
| hopper-medium | ddim-2 | 53.8 | 91.5 |
| walker2d-medium | ddpm-20 | 71.5 | 89.3 |
| walker2d-medium | ddim-2 | 55.4 | 93.2 |

Held-out MOReL and QGPO (catastrophic extrapolation; same failure in B and E):

| method | dataset | actual | D predicted |
|---|---|---|---|
| MOReL | halfcheetah-medium-expert | 53.3 | 776.8 |
| MOReL | hopper-medium-expert | 108.7 | 648.2 |
| QGPO | walker2d-medium | 86.0 | −854.6 |
| QGPO | hopper-medium | 98.0 | −671.6 |

MOReL is the only method with `hidden_dim=32`, `model_rollout_length` 400/500 (training range 1–5)
and `num_q_networks=not_applicable` among model-based methods; QGPO is the only method with
`guidance_scale` 2–10 (training range 0.0001–0.1) and `batch_size=4096` outside SfBC. A linear model
with alpha ≤ 10 extrapolates along those columns.

## Failures and caveats

1. **Extrapolation, not structure, decides the Ridge aggregate.** Conditions columns are supported by
   few methods each; when the held-out method sits outside the training range on any of them the
   linear model produces absurd values, for B, D and E alike. The pre-registered alpha grid could not
   prevent it (grouped CV inside training does not see the held-out extremes). Exploratory only, not
   pre-registered: dropping MOReL and QGPO gives mean per-method MAE A 26.4 / B 21.2 / C 26.6 /
   **D 18.2** / E 20.3 — a 3-point D-over-B gain that would need its own pre-registered follow-up.
2. **The structure vocabulary barely overlaps across methods.** 80 tuples, 11 shared by ≥2 methods,
   none by more than 4. A held-out method's tuples are mostly unseen at training time, so the
   multi-hot is near-empty at test. This is a property of the current extraction (short op lists,
   free verbs, many `other`-typed objects), not of the predictor. Relaxed compatibility groups
   (`semantic_ontology.json`) were not used — strict types were pre-registered.
3. **kNN distance dilution.** `dataset` is one of ~23 equally weighted categorical columns, so the
   nearest neighbours are often not on the same D4RL dataset; kNN B is no better than the mean.
   Equal weights were pre-registered; a `dataset`-first distance is a follow-up, not a Phase 24 fix.
4. **Conditions are reported unevenly.** Many cells are `missing` (e.g. hidden_dim for U-Net planners,
   batch size for CQL/EDAC/COMBO, guidance weight for DD). Missingness states are encoded, but they
   correlate with paper rather than with mechanism.
5. **Dataset-version mixing.** v0 (CQL, TD3+BC), unknown (MOPO, MOReL, COMBO, Diffuser, DD, AdaptDiffuser,
   HD, PLAS, FisherBRC, QGPO, SRPO, DT) and v2 rows share `dataset`; `dataset_version` is a feature
   but the models had no way to learn a v0/v2 offset from so few v0 rows.
6. **Row imbalance.** ReBRAC (72) and EDP (63) are 25% of rows; row-level MAE and correlation are
   weighted toward them. Mean per-method MAE is the headline for that reason.
7. **Extraction caveats carried from `sources/tables/*.json`:** seed-count conflicts (COMBO 6 vs 3,
   DQL 5 vs 6, one-step 3 vs 10), ± meaning differs (std / standard error / 95% CI), MOReL
   hyperparameters are stated for the Wu et al. datasets not D4RL, COMBO conditions derived from
   prose rules, SRPO β read from a garbled table region, QDT table labels resolved positionally.
8. **Phase 23 rows** are 3-seed means under gymnasium v4 dynamics; they are pilot rows, held out with
   Diffuser, and were not treated as independent methods.
9. **Structure is one record per paper.** Variants folded into one `method_id` (SAC-N under EDAC,
   EDP's four base learners, IDQL-A/-1, TT uniform/quantile, one-step's three operators) share that
   record even where the variant changes the mechanism slightly.

## Final questions

### 1. Primary — Is MAE(Structure + Conditions) < MAE(Conditions only)?

No. Ridge: 59.9 vs 37.9 (mean per-method MAE); kNN: 27.3 vs 27.3. Row-level: 40.1 vs 31.1 and
28.1 vs 27.9. On the pre-registered metric, adding `operation@object` structure did not reduce
prediction error on held-out D4RL methods.

### 2. Mechanism vs lineage — Is MAE(Structure + Conditions) < MAE(Lineage + Conditions)?

No. Ridge 59.9 vs 43.8; kNN 27.3 vs 27.0. The structural representation did not outperform
conventional lineage labels under the same conditions.

### 3. Stability — does an improvement hold across methods or come from a few cases?

The aggregate loss comes from two methods (MOReL, QGPO) where all conditioned Ridge models fail by
hundreds of points. Away from them, Ridge D beats B on 21/27 methods with a median gain of 2.8 points
and beats E on 19/27 (median 2.3). That per-method pattern is consistent under Ridge but absent under
kNN (D ties B on 24/27 methods, wins 3). So: no stable improvement across predictors; a modest,
consistent-but-small Ridge-only improvement that the pre-registered aggregate does not capture and
that was not itself pre-registered as the test.

### 4. Interpretation — what does the result support and not support?

Supported:
- On this D4RL evaluation, the current `operation@object` structure did not add predictive
  information beyond experimental conditions on the pre-registered metric, with either predictor.
- Structure alone carries no outcome information here (C ≈ A) — largely because the vocabulary is
  almost disjoint across methods.
- Experimental conditions themselves are a weak and unstable predictor across methods on this
  dataset: helpful on most held-out methods under Ridge, catastrophic on a few, inert under kNN.
- There is a small, Ridge-only, per-method tendency for D to beat B and E (median ≈ 2–3 points) that
  is worth a pre-registered follow-up with a bounded predictor (e.g. clipped predictions, higher
  minimum alpha, or a tree model) and a relaxed/grouped structure vocabulary.

Not supported, and not claimed:
- that mechanism structure predicts scientific outcomes in general;
- that mechanism beats lineage in ML broadly;
- that the negative result is a property of the idea rather than of this extraction — the vocabulary
  overlap problem (11 shared tuples across 27 methods) means the structure feature was mostly empty at
  held-out time, so the test says as much about extraction coverage as about predictive content.

A second benchmark and a structure representation with real cross-method overlap would both be needed
before any general claim in either direction.
