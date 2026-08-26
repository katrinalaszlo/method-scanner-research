# Phase 25 — exploratory follow-up to Phase 24 (labeled, separate)

Written 2026-08-25 before running `run.py`. Phase 24 files are untouched; this phase tests the three
flaws recorded in `../results.md` §Failures. Same dataset (`../d4rl_dataset.csv`), same holdout
(leave-one-method-out), same headline metric (mean per-method MAE), same models A–E.

## Changes, fixed in advance

1. **Structure vocabulary variants** (Phase 24 used strict `op@type`, 11/80 tuples shared):
   - `strict`  — Phase 24 tuples (control)
   - `relaxed` — object type mapped to its compatibility group from `semantic_ontology.json`
     (represented_state / learning_signal / stochastic_object), other types unchanged; `other:` stays unique
   - `bare`    — canonical operation only (`op`), object dropped
2. **Bounded Ridge**: alpha grid `{1, 10, 100, 1000}` (min raised from 0.01), predictions clipped to
   `[min, max]` of the training outcomes. Same grouped-CV alpha selection.
3. **Dataset-first kNN**: `dataset` mismatch becomes its own distance group (equal weight with
   structure / numeric / categorical / lineage), instead of one of ~23 categorical columns. k=5 unweighted.

Every (variant × predictor) combination is run; nothing is chosen after seeing results. Verdict rule as in
Phase 24: D < B and D < E on mean per-method MAE, plus win counts.
