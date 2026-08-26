# Phase 27 — full-operation extraction (pre-registration)

Written 2026-08-25 before running `extract_full.py` on the corpus. Motivation: Phases 24–26 found the structure
feature inert mainly because the scanner emits ~4 operations per paper, so held-out methods overlap the training
vocabulary on 2–4 tuples.

## Extraction (single pass, replaces the two-pass extract+type for this phase only)

- Input: the same method sections as Phase 26b (2500 words; same `--from` pins), `qwen3:8b`, temperature 0, `num_ctx 8192`.
- Prompt asks for an exhaustive ordered list of the operations the algorithm performs at runtime (training and
  inference), 8–25 items, each with `op` (one or two words, verb first), `object` from the CLOSED ontology
  {state, latent, error, gradient, parameter, noise, input, action, reward, distribution, sample, policy, value, task,
  trajectory} — no `other` — and a short supporting quote. Items whose quote is not found in the section are kept but
  flagged `grounded=false` (reported; not filtered, to avoid post-hoc selection).
- Tuple = canonical op (`app.RAW_TO_CANONICAL`, exact then head verb; raw op kept if unmapped) `@` object.
  Pipeline of Phase 24 otherwise unchanged. Records stored in `phase27/records/`, not in `results/`.
- Pilot: 3 papers (Diffuser, IQL, SPR) to confirm the prompt returns 8–25 typed operations. The prompt may be
  adjusted on pilot *format* failures only, before any evaluation; pilot tuples are not compared with outcomes.

## Evaluation

- Same datasets (`d4rl_dataset.csv`, `atari100k_dataset.csv`), same holdout, same predictors as Phase 26
  (clipped Ridge primary, dataset-first kNN), strict and relaxed vocab. Structures swapped for Phase 27 tuples.
- Report vocabulary size, shared≥2, tuples/method, and the A–E table for both benchmarks against the Phase 24/26
  structures. Hypotheses: H1 `MAE(D) < MAE(B)`, H2 `MAE(D) < MAE(E)`, and the new one H3: `MAE(C) < MAE(A)`
  (structure alone carries signal once overlap exists). Nothing selected after seeing MAE.

### Pilot adjustment (format only, before any evaluation)
First pilot: 25 / 8 / 13 ops but 9 / 6 / 6 objects outside the closed set (loss, target, representation, mean, plan, model…),
and IQL degenerated into six identical "compute loss @ target" items. Fixes: ontology definitions and an explicit
synonym rule added to the prompt; a post-hoc synonym map (loss→error, target/Q-function/return→value,
representation→latent, mean/variance/probability→distribution, plan→trajectory, model/network→parameter,
observation/image→input, plurals) applied to the model's object; "no repeated items" rule. Pilot re-run on the same 3 papers.
Second pilot: Diffuser 24 ops / 2 invalid, IQL 8 ops / 0 invalid (faithful: its section is three loss+update pairs),
SPR 24 ops / 7 invalid after JSON salvage (malformed list; well-formed items recovered by regex, logged). Format accepted;
full run launched on the 27 D4RL + 17 Atari papers with this prompt. Invalid objects are dropped from tuples and counted.
