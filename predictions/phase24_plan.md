# Phase 24 plan — Outcome prediction from structure + conditions (the original system)

## What this phase is

Phases 1–23 built the left half of the intended system: paper → `operation@object` structure. This phase builds the right half:

```
STRUCTURE + CONDITIONS → predictor → OUTCOME
```

and tests it the way the original design specified: hold out a published paper, give the predictor its structure and conditions, hide its reported result, predict, reveal, score.

The question is not "can we predict D4RL scores". It is:

> **Does `operation@object` structure carry predictive information about outcomes beyond what the benchmark and hyperparameters already carry?**

Phase 23 showed why conditions cannot be omitted: the same transferred mechanism behaved differently on hopper and walker because of horizon and schedule — things `operation@object` does not encode.

## Benchmark: D4RL locomotion

One ecosystem, one outcome variable with a consistent meaning. Tasks: `{halfcheetah, hopper, walker2d} × {medium, medium-replay, medium-expert}` — the nine cells nearly every offline-RL paper since 2020 reports as normalized return. Papers report a mean (often ± sd over seeds) per cell, so one paper yields up to nine rows.

Target: the paper's **reported mean normalized return** per cell. Prediction error is judged against seed variance, which Phase 23 measured directly: ±1.6 on halfcheetah-medium, ±5.6 hopper-medium, ±13.6 walker2d-medium (3 seeds, DDPM-20). A predictor within seed variance is as good as re-running the paper.

## Data

### Rows
One row = (paper, task cell): structure of the method + conditions of that cell's run + reported outcome.

### Structure (existing pipeline)
Method section → extract → canonicalize → type (v2 + `trajectory`) → tuple set. Apply the delta-paper rule from Phase 21: papers that inherit a base loop (TD3+BC from TD3, Decision Diffuser from Diffuser, EDAC from SAC) get the base loop injected, with `inherited_ops` kept separate so structure-only can be run both ways.

### Conditions (new extraction target)
Per paper, per cell where it varies:

| field | type | source |
|---|---|---|
| task, dataset quality | categorical | table header |
| algorithm class (model-free / model-based / sequence-model / diffusion) | categorical | lineage label — deliberately *not* structure; it is the "what a human would say" baseline |
| planning / context horizon | int or none | method / hyperparameter table |
| ensemble size, critics count | int | hyperparameter table |
| policy extraction (weighted BC / actor / planner / return-conditioned) | categorical | method |
| gradient steps / epochs | int | hyperparameter table |
| seeds reported | int | table caption |
| year | int | arXiv id |

Extraction: same few-shot approach as `failure`, a separate prompt over the hyperparameter appendix + results section; hand-verified for the first ten papers, then spot-checked. Missing fields stay missing (no imputation in v0).

### Outcome (new extraction target)
Per cell: reported mean, sd if given, number of seeds. Extracted from the results table; hand-verified against the PDF for every paper — this is the label, it cannot be noisy.

### Papers
Target 30 papers → ~200 rows. Candidate list (arXiv ids **must be verified via the API before fetch** — Phase 7 found 6 of 30 from-memory ids were wrong papers):

BCQ, BEAR, CQL, IQL, TD3+BC, AWAC, Onestep RL, Fisher-BRC, PLAS, EDAC, MOPO, MOReL, COMBO, MBOP, Decision Transformer, Trajectory Transformer, RvS, Diffuser, Decision Diffuser, and the D4RL paper's own baselines (BC, SAC-offline, BRAC-p/v, AWR). Add later offline-RL papers reporting the same nine cells until n≈30.

### Phase 23 rows
The 45 runs collapse to 15 clean rows: Diffuser structure × {sampler, steps, env} → measured mean over 3 seeds. Include them as a separate source (`measured`, not `reported`) and report results with and without them.

## Predictor ladder

Run in this order; each must beat the previous to justify the next. All evaluated leave-one-**paper**-out (hold out every row of one paper, never just one cell — cells of one paper are not independent).

| | predictor | what it tests |
|---|---|---|
| A | global mean per task cell | how much the benchmark alone tells you |
| B | conditions only — kNN or ridge on the conditions vector | how much hyperparameters + lineage tell you |
| C | structure only — kNN on strict / relaxed tuple Jaccard, outcome = neighbours' mean in the same cell | does mechanism carry anything on its own |
| D | structure + conditions — conditions model with structure-neighbour residual, or ridge on [conditions ‖ tuple indicator vector] | **the thesis test** |
| D′ | D with `algorithm class` removed from conditions | is structure just re-encoding the lineage label |
| D″ | D with inherited ops excluded | is the signal in the paper's own contribution |

Metric: MAE per cell, reported per task and overall, with a bootstrap CI over held-out papers. Also report **rank correlation** within each cell — predicting *ordering* of methods is the useful question even if absolute error is dominated by a per-cell offset.

## Pre-registered expectations

- A will be surprisingly good: task cell explains most of the variance. Everything is judged relative to A.
- B beats A, mostly via algorithm class and year.
- C alone is weak — structure without the task cell has no scale.
- **The claim stands only if D beats B by more than the bootstrap CI, and D′ still beats B.** If D ≈ B, structure adds nothing beyond lineage + hyperparameters for this benchmark, and that is the result to publish.
- Expected failure mode to watch: D beats B because tuple sets encode the algorithm class (diffusion papers all have `sample@…`). D′ is the control for that.

## Success criteria

| criterion | target |
|---|---|
| dataset | ≥25 papers, ≥150 rows, every outcome hand-verified |
| A vs B vs C vs D reported | with LOPO CIs |
| thesis | D − B improvement > CI, and D′ − B > 0 |
| calibration | D's MAE within seed variance on ≥5 of 9 cells |

## Deliverables

1. `phase24/papers.csv` — verified ids, roles (base / delta), sources
2. `phase24/conditions.json`, `phase24/outcomes.csv` — per-row, with provenance (page/table)
3. Structure via the existing pipeline (`results/`, typed v2)
4. `phase24/predict.py` — ladder A–D″, LOPO, bootstrap
5. `phase24/results.md` — table of MAE / rank-corr per predictor per cell; verdict on the thesis; the Phase 23 rows as a held-out check
6. findings + write-up section

## Timeline (single machine, ~40 s per extraction, no GPU needed)

| step | effort |
|---|---|
| verify ids, fetch PDFs, extract + type structure (existing pipeline) | 0.5 day |
| conditions + outcome extraction prompt, hand-verify 10, run 30, spot-check | 1 day |
| predictor ladder + LOPO + bootstrap | 0.5 day |
| write-up | 0.5 day |

## Risks

- **Outcome tables disagree across papers** for the same baseline (CQL is reported anywhere from 44 to 47 on halfcheetah-medium by different authors). Use each paper's own reported number for its own method only; do not take baselines from other papers' tables.
- **Conditions extraction is the new noise source.** It will be as noisy as typing (~1 field/paper). Hand-verify; report which fields were hand-fixed.
- **n≈30 papers is small for D.** Ridge with strong regularisation, or kNN — nothing with more parameters than rows. The bootstrap CI is the honest guard.
- **Benchmark drift.** v0 vs v2 datasets; exclude v0-only rows.

## What this changes in the write-up if it works

Current claim: structural matching + one human-derived transfer experiment. New claim: `operation@object` is a feature representation that improves held-out outcome prediction beyond benchmark + hyperparameters + lineage. If it fails: structure is descriptive, not predictive, at this corpus size — also worth stating.
