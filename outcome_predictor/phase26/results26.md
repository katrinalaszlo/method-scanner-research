# Phase 26 results — second benchmark (Atari 100k) and vocabulary-overlap test

Pre-registered in `schema26.md` before evaluation. 2026-08-25.

## 26a — Atari 100k

### Dataset

`atari100k_dataset.csv`: 475 rows, 17 methods, 17 papers, 26 games. Outcome = 100 × human-normalized
score computed from the Random/Human columns printed in each paper's own table (`raw_score`,
`random_ref`, `human_ref` kept per row). Range −2.2 to 1431.9, median 33.6 — a heavy right tail
(Krull, Boxing, Kangaroo, Road Runner are far above human for several agents), so MAE values are large and
tail-dominated. Rows per method: 26 each, except SPR 52 (with/without augmentation), EfficientZero 34
(+8 no-augmentation rows), SimPLe 25 (Frostbite has no human reference in its table).
Excluded (in `sources/build_audit.json`): SimPLe world-model/γ/rollout ablations (confounded with a
short-vs-long training difference that has no column), EfficientZero component ablations (structural),
IRIS 64-token, STORM without demonstration, DIAMOND 1-step (single seed), EZ-V2 value-target ablations,
SimPLe's 10 games outside the 26-game set.

Structure: Method Scanner on each paper (SimPLe and DER pinned to their method sections; superseded
records kept). DreamerV3 uses its existing corpus record. Strict vocabulary 54 tuples, 9 shared by ≥2
methods; relaxed 48 / 11.

Conditions: `env_steps, replay_ratio, batch_size, learning_rate, n_step, discount, hidden_dim, num_layers,
imagination_horizon, num_simulations, model_params, n_seeds, eval_episodes` + `sticky_actions,
data_augmentation, checkpoint_selection` + game. Lineage: model_free_value (DER, OTRainbow),
model_free_representation (CURL, DrQ, SPR, PlayVirtual, MLR, BBF), world_model_imagination (SimPLe,
DreamerV3, IRIS, TWM, STORM, DIAMOND, Δ-IRIS), mcts_planning (EfficientZero, EZ-V2).

### Aggregate (mean per-method MAE on HNS×100; median in parentheses)

| predictor | vocab | A | B | C | D | E | D<B | D<E | B<A |
|---|---|---|---|---|---|---|---|---|---|
| clipped Ridge (primary) | strict | 114.8 (93.6) | 111.0 (84.1) | 116.3 (100.0) | **87.9 (72.2)** | 115.4 (84.0) | 11/17 | 11/17 | 13/17 |
| clipped Ridge | relaxed | 114.8 | 111.0 | 119.2 | **88.1** | 115.4 | 11/17 | 11/17 | 13/17 |
| dataset-first kNN | strict | 114.8 (93.6) | 79.0 (55.2) | 101.0 (84.6) | 78.5 (55.2) | **73.5 (43.8)** | 3/17 | 4/17 | 16/17 |
| dataset-first kNN | relaxed | 114.8 | 79.0 | 100.8 | 79.0 | **73.5** | 3/17 | 4/17 | 16/17 |

Median per-method (D−B): Ridge −4.3, kNN 0.0. Median (D−E): Ridge −4.5, kNN 0.0.
Excluding DreamerV3 (where B and E clip at the training maximum — its replay ratio 128 and hidden 1024
are unique): Ridge A 114.4 / B 92.8 / C 116.0 / **D 85.7** / E 93.6; kNN B 77.4 / D 76.8 / **E 71.5**.

### Per-method MAE — clipped Ridge, strict

| method | A | B | C | D | E |
|---|---|---|---|---|---|
| BBF | 186.9 | 196.0 | 186.8 | **147.4** | 199.0 |
| CURL | 85.7 | 40.3 | 80.6 | 43.7 | 41.4 |
| DER | 82.8 | 39.9 | 85.5 | 40.1 | 39.9 |
| DIAMOND | 140.5 | 96.2 | 140.8 | 101.7 | 96.1 |
| Δ-IRIS | 129.3 | 87.1 | 127.2 | 91.5 | 87.1 |
| DrQ | 83.0 | 44.7 | 84.2 | **33.4** | 45.6 |
| DreamerV3 | 120.4 | 403.1 | 121.0 | **121.9** | 464.4 |
| EfficientZero | 180.8 | 205.6 | 180.8 | **185.9** | 205.7 |
| EfficientZeroV2 | 236.3 | 227.6 | 236.2 | **222.8** | 227.3 |
| IRIS | 102.8 | 89.7 | 100.0 | **69.5** | 92.1 |
| MLR | 76.7 | 46.4 | 73.7 | **43.3** | 46.9 |
| OTRainbow | 87.9 | 125.7 | 126.6 | **121.4** | 129.3 |
| PlayVirtual | 75.4 | 32.8 | 74.4 | **28.9** | 32.9 |
| SPR | 75.6 | 52.5 | 74.2 | **44.7** | 55.6 |
| STORM | 111.0 | 84.1 | 111.4 | 85.1 | 84.0 |
| SimPLe | 82.0 | 34.8 | 91.5 | 40.2 | 34.1 |
| TWM | 93.6 | 81.3 | 81.8 | **72.2** | 80.2 |

### Per-method MAE — dataset-first kNN, strict

| method | A | B | C | D | E |
|---|---|---|---|---|---|
| BBF | 186.9 | 156.6 | 216.0 | 156.6 | 156.6 |
| CURL | 85.7 | 42.7 | 32.2 | 27.4 | 28.7 |
| DER | 82.8 | 55.2 | 43.4 | 55.2 | 16.8 |
| DIAMOND | 140.5 | 98.7 | 139.2 | 98.7 | 86.6 |
| Δ-IRIS | 129.3 | 78.9 | 129.9 | 78.9 | 69.0 |
| DrQ | 83.0 | 24.4 | 28.3 | 24.4 | 25.3 |
| DreamerV3 | 120.4 | 105.3 | 106.7 | 105.3 | 105.3 |
| EfficientZero | 180.8 | 192.9 | 205.5 | 192.9 | 192.9 |
| EfficientZeroV2 | 236.3 | 217.7 | 245.2 | 217.7 | 217.7 |
| IRIS | 102.8 | 46.2 | 97.2 | 46.2 | 31.2 |
| MLR | 76.7 | 24.5 | 51.0 | 25.3 | 22.5 |
| OTRainbow | 87.9 | 85.5 | 84.6 | 85.5 | 85.5 |
| PlayVirtual | 75.4 | 19.5 | 50.7 | 19.5 | 17.2 |
| SPR | 75.6 | 29.4 | 46.6 | 29.4 | 23.3 |
| STORM | 111.0 | 77.9 | 109.4 | 77.9 | 87.4 |
| SimPLe | 82.0 | 43.8 | 49.4 | 43.2 | 43.8 |
| TWM | 93.6 | 44.1 | 82.1 | 50.2 | 40.2 |

### Verdicts (primary predictor = clipped Ridge, as pre-registered)

| comparison | clipped Ridge | dataset-first kNN | verdict |
|---|---|---|---|
| B vs A | 111.0 vs 114.8 (13/17) | 79.0 vs 114.8 (16/17) | conditions help; dataset-first kNN much more than Ridge |
| C vs A | 116.3 vs 114.8 (9/17) | 101.0 vs 114.8 (13/17) | structure alone: no signal under Ridge, weak under kNN |
| **D vs B (H1)** | **87.9 vs 111.0, 11/17, median −4.3** | 78.5 vs 79.0, 3/17, median 0 | **supported under the primary predictor; not under kNN** |
| **D vs E (H2)** | **87.9 vs 115.4, 11/17, median −4.5** | 78.5 vs 73.5, 4/17 | **supported under the primary predictor; reversed under kNN (lineage wins)** |

Same shape as D4RL under Phase 25's predictors: a linear model with conditions gains from structure on a
majority of held-out methods and on aggregate; a same-dataset nearest-neighbour model does not, and there
the conventional lineage label is the better feature. The Ridge D gain is again partly level-correction
on methods whose conditions sit at the training edge (DreamerV3, BBF) and partly a consistent small gain
on the Rainbow-family methods (DrQ, SPR, MLR, PlayVirtual, TWM, IRIS).

### Caveats

- Heavy-tailed outcome: HNS×100 up to 1432; MAE is dominated by superhuman games. Medians reported alongside.
- 17 methods, one benchmark protocol; per-game std rarely printed; seeds vary 3–50; eval episodes often unstated.
- Structure overlap is again thin (9/54 tuples shared). STORM's record has 13 tuples, most others 3–5.
- `replay_ratio` for EfficientZero (1.2) and DIAMOND (4) are derived from printed step counts, not printed as such.
- DreamerV3's structure comes from the earlier corpus record (1500-word section), not a Phase 26 run.

## 26b — vocabulary overlap (long sections)

Same 27 D4RL papers re-extracted with 2500-word method sections (`--words 2500`, same pins; needed
`num_ctx: 8192` in `app.py` / `semantic_type.py` — the default context silently truncated long prompts).
QDT's long extraction was retried once after a parse failure and succeeded.

| section | vocab | tuples | shared ≥2 | tuples/method | clipped Ridge B / D / E | D<B | kNN B / D / E | D<B |
|---|---|---|---|---|---|---|---|---|
| short (Phase 24) | strict | 80 | 11 | 3.7 | 24.5 / 22.4 / 24.1 | 22/27 | 14.7 / 14.8 / 15.8 | 3/27 |
| long | strict | 74 | 19 | 3.9 | 24.5 / 22.2 / 24.1 | 22/27 | 14.7 / 14.6 / 15.8 | 6/27 |
| short | relaxed | 75 | 14 | 3.7 | 24.5 / 21.9 / 24.1 | 22/27 | 14.7 / 14.8 / 15.8 | 4/27 |
| long | relaxed | 68 | 21 | 3.9 | 24.5 / 22.2 / 24.1 | 20/27 | 14.7 / 14.9 / 15.8 | 6/27 |
| short | bare | 53 | 14 | 3.6 | 24.5 / 23.5 / 24.1 | 20/27 | 14.7 / 14.8 / 15.8 | 11/27 |
| long | bare | 44 | 14 | 3.9 | 24.5 / 23.4 / 24.1 | 18/27 | 14.7 / 14.8 / 15.8 | 11/27 |

Reading: longer sections nearly double cross-method overlap (11→19 strict, 14→21 relaxed) but the extractor
still returns ~4 operations per paper, so each method's structure vector stays tiny and the predictor is
unchanged (D 22.2 vs 22.4). **The bottleneck is the number of operations the extraction step emits per paper,
not section length or type granularity.** Structure alone (C) never beats the mean baseline in any variant.

## Overall reading after Phases 24–26

- Two benchmarks, same pattern: under a bounded linear model, `operation@object` structure lowers held-out
  MAE beyond conditions (D4RL −2.5, Atari −23 on HNS×100; 20–22/27 and 11/17 methods) and beyond
  conventional lineage; under a same-dataset nearest-neighbour model it adds nothing, and lineage is as good
  or better. The Phase 24 pre-registered test (unclipped Ridge, equal-weight kNN) was negative; the
  Phase 26 pre-registered test (clipped Ridge primary) is positive on Atari for both H1 and H2.
- What the gain is: level correction for a held-out method's typical score band, not within-dataset
  ranking (Phase 25b ρ≈0). Structure carries "what kind of method is this" at roughly the resolution of a
  lineage label, expressed in a shared vocabulary; it does not carry "which method wins on this dataset".
- What would change the answer: an extraction that emits full operation lists (10–20 ops per paper, as the
  STORM record shows is possible) with a closed object ontology, so that held-out methods overlap the
  training vocabulary on more than 2–4 tuples.
