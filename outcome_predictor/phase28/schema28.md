# Phase 28 — paired within-method design (pre-registration, 2026-08-26)

Question: when a paper changes one thing about its own method and re-runs it on the same dataset, does the
`operation@object` structure say anything about how much the score moves?

## Pairs
- One pair = (paper, method, dataset): a main-configuration row and one variant row from the same table family
  (ablation column, hyperparameter sweep value, named variant method of the same paper, Phase 23 sampler swap).
  Target `delta = score(variant) − score(main)` in the benchmark's normalized units (D4RL score; Atari 100×HNS).
- Excluded: online fine-tuning, delayed-reward, raw-return rows, restated/rerun "main" columns, dataset-version
  reruns, Atari games outside the 26-game set. Listed in `build_pairs.py`.
- Hyperparameter sweeps are kept: they are the control group where the change should touch no operation.

## Change mapping (LLM, deterministic)
- For each distinct (paper, variant) the Phase 27 operation list (op, object, quote) is shown to `qwen3:8b`
  (temp 0) with the variant description taken verbatim from the extracted tables (label, config_conditions,
  notes, variant-method description). It returns `change_type` ∈ {remove, replace, add, hyperparameter, other}
  and the indices of affected operations. Outputs are stored in `changes/` and not edited.

## Features and models (leave-one-paper-out, clipped Ridge, alpha grid {1,10,100,1000})
- A: mean delta of training pairs.
- B (no structure): dataset one-hot, main score (standardized), benchmark, change_type one-hot.
- C (structure): multi-hot of affected `op@object` tuples, count of affected ops, fraction of the method's ops affected.
- D: B + C.
- Primary metric: MAE on signed delta, mean per held-out paper. Secondary: sign accuracy on pairs with |delta| ≥ 2,
  and MAE on |delta|. Hypotheses: H1 `MAE(D) < MAE(B)`; H2 `MAE(C) < MAE(A)`.
