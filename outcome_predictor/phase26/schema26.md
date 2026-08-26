# Phase 26 — pre-registration (second benchmark: Atari 100k; vocabulary-overlap test)

Written 2026-08-25 before any Atari row was evaluated and before any long-section scanner record was inspected.

## 26a — Atari 100k outcome prediction

- Rows: one per (method, configuration, game) from the paper's own per-game table at the 100k-step budget.
  Outcome = human-normalized score `(score − random) / (human − random)` using the Random/Human reference
  values printed in the same paper's table; if a paper prints none, the reference values from the SPR
  paper (2007.05929) are used and the row is marked `reference=SPR`. Rows where a game score cannot be
  tied unambiguously to its column are not emitted. Aggregates (mean/median/IQM) are not rows.
- Holdout unit: `method_id`, leave-one-method-out. Headline: mean per-method MAE on HNS.
- Predictors, fixed from Phase 25: **clipped Ridge** (alpha grid {1, 10, 100, 1000}, grouped CV, predictions
  clipped to training range) as PRIMARY; dataset-first kNN k=5 as robustness. Structure vocabulary: strict
  `op@type` (primary) and relaxed groups (secondary), both reported.
- Models A–E as in Phase 24. Hypotheses: H1 `MAE(D) < MAE(B)`, H2 `MAE(D) < MAE(E)`; stability by per-method wins.
- Conditions (Atari): numeric — `env_steps`, `replay_ratio` (gradient updates per env step), `batch_size`,
  `learning_rate`, `n_step`, `discount`, `hidden_dim` (encoder/latent width), `num_layers`, `imagination_horizon`
  (world-model rollouts; not_applicable otherwise), `num_simulations` (MCTS; not_applicable otherwise),
  `model_params` (millions, if stated), `n_seeds`, `eval_episodes`; categorical — `sticky_actions` (yes/no/missing),
  `data_augmentation` (yes/no/missing), `checkpoint_selection` (last/best/missing), `dataset` (game).
  Lineage labels — model_free_value (Rainbow-family), contrastive/self-supervised model-free (CURL, SPR,
  PlayVirtual, MLR, BBF, DrQ → `model_free_representation`), `world_model_imagination` (SimPLe, DreamerV3, IRIS,
  TWM, STORM, DIAMOND, Delta-IRIS), `mcts_planning` (EfficientZero, EfficientZero V2). Assigned before evaluation.
- Every condition and lineage assignment lives in `build_atari.py`, written before evaluation.

## 26b — vocabulary overlap

Same 27 D4RL papers re-extracted with 2500-word sections (`--words 2500`, same pins). Report: vocabulary size,
tuples shared by ≥2 methods (strict, relaxed, bare), and Phase 25's clipped-Ridge / dataset-first-kNN table
re-run with the long-section structures. Rule: report both; no selection between short and long after seeing MAE.
