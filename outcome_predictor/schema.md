# Phase 24 — Frozen feature schema

Frozen 2026-08-25, before `evaluate.py` was run on the assembled dataset. Nothing below may change
because of held-out performance. A flaw found later is recorded in `results.md` and tested in a
separately labeled follow-up, not by editing this file.

## Holdout

- Unit: `method_id`. Leave-one-method-out: every row of method K is test, every row of every other
  method is train. A method's ablations, sampler variants, dataset-version reruns and Phase 23 rows
  all share its `method_id` and leave together.
- Primary metric: MAE. Headline aggregate = **mean of per-method MAE** (each held-out method counts
  once; row-level MAE is also reported but is dominated by the methods with many ablation rows —
  ReBRAC, EDP, TAP). Secondary: RMSE, Pearson correlation.
- Hypotheses: H1 `MAE(D) < MAE(B)`; H2 `MAE(D) < MAE(E)`. Stability: count of held-out methods on
  which D beats B, and D beats E.

## Row definition

One row = one method + one exact reported configuration + one D4RL locomotion dataset + the exact
normalized score reported for that configuration (mean over the paper's seeds). Rules applied in
`build_dataset.py`:

- Only the paper's own method. Baselines reported by other papers are not rows.
- Only normalized D4RL scores. Raw-return tables excluded.
- Only offline results. Online fine-tuning rows excluded. Delayed/sparse-reward rows excluded.
- Ablation rows are included only when the varied parameter maps to a schema column below
  (e.g. TAP horizon sweep, ReBRAC batch/layers/layernorm, EDP RAT-vs-OMS checkpoint rule, DQL
  online model selection, IDQL-A/IDQL-1, Diffuser Phase 23 sampler sweep). Ablations whose varied
  parameter has no column (COMBO β, QDT α, EDAC η sweep, PLAS ε, TAP β / L, DD α, HD K) are excluded
  so that feature-identical rows with different outcomes are not introduced.
- Structural ablations (different mechanism, e.g. SfBC+Gaussian, BC-Diffusion, CondMLPDiffuser,
  TAP unconditional decoder, one-step multi-step/iterative variants) are excluded: they do not share
  the paper's scanner structure.
- Rows the extractor could not tie unambiguously to a dataset and column were never emitted
  (`sources/tables/*.json` → `problems`).
- Excluded papers: AWAC (2006.09359; locomotion numbers only in a figure, no seeds/episodes stated).
- SAC-N rows are included under `method_id=EDAC` as the η=0 configuration (EDAC with zero
  diversification is SAC-N); they differ from EDAC rows in `num_q_networks` where the paper's
  chosen N differs. PLAS+P ε-sweep rows excluded (no column); PLAS main rows included.

## Dataset identity

- `dataset` = `<env>-<quality>`, env ∈ {halfcheetah, hopper, walker2d}, quality ∈ {random, medium,
  medium-replay, medium-expert, expert, full-replay}. Printed names are normalized (`mixed`,
  `m-re`, `med-rep` → `medium-replay`; `walker` → `walker2d`; `-v2` suffix stripped into the
  version field).
- `dataset_version` ∈ {v0, v2, unknown} from the paper's own statement; `unknown` when never stated.
  Phase 23 rows are `v2` data run under gymnasium v4 dynamics (noted in `notes`).

## Structure feature

- Source: the Method Scanner record for the paper (`results/*.json`, `source: test:<arxiv_id>`),
  produced by the existing pipeline (`fulltext.py` → `app.py` extraction with `qwen3:8b` →
  `semantic_type.py --pass v2`). Tuple outputs were not edited. Five papers whose auto-detected
  section was wrong (DD, RvS, Fisher-BRC, QGPO, Hierarchical Diffuser) were re-extracted with the
  method section pinned via `--from`; pins were chosen from section headings before evaluation and
  the replaced records are kept in `superseded_records/`.
- Tuple = `<canonical_op>@<v2 object type>`, exactly as `compare.semantic_set(pass_no="v2",
  relaxed=False)`: canonical op by exact then head-verb lookup in `app.RAW_TO_CANONICAL`, raw op
  kept when unmapped; object types `other`/`unknown` become `other:<method_id>` (never shared).
- One structure per method_id (the paper's structure). Phase 23 rows use Diffuser's structure.
- Encoding (Ridge): multi-hot over the vocabulary of tuples present in the training rows; test
  tuples outside that vocabulary are dropped.
- Distance (kNN): `1 − Jaccard(tuple_set_A, tuple_set_B)`.

## Conditions

Numeric (standardized with training mean/std; absent → 0 after standardization, plus a
per-column state one-hot {present, missing, not_applicable}):

| column | definition |
|---|---|
| `batch_size` | training batch size of the main policy/model |
| `learning_rate` | learning rate of the main policy/model (actor lr where actor/critic differ) |
| `training_steps` | gradient steps of the main policy/model (epochs × steps/epoch when given that way) |
| `hidden_dim` | width of the policy/decision network (MLP hidden units or transformer embedding dim); dynamics-model widths are not used |
| `num_layers` | depth of that network (hidden layers / blocks) |
| `discount` | γ used by the method; `not_applicable` for return-conditioned supervised methods |
| `planning_horizon` | trajectory-level planning horizon; `not_applicable` for single-step policies |
| `context_length` | sequence-model context (tokens/transitions); `not_applicable` for non-sequence methods |
| `denoising_steps` | denoising network evaluations per action at inference; `0` for SRPO (no sampling at inference); `not_applicable` for non-diffusion methods |
| `guidance_scale` | guidance / energy scale at sampling; `not_applicable` for unguided methods |
| `num_candidates` | candidate actions/plans compared at inference (beam width for beam search); `1` when a single sample is taken; `not_applicable` for deterministic/Gaussian policies |
| `num_q_networks` | critic ensemble size; `not_applicable` when there is no Q-function |
| `model_rollout_length` | model-based rollout length; `not_applicable` for model-free methods |
| `expectile_tau` | IQL-style expectile; `not_applicable` otherwise |
| `n_seeds` | number of seeds the reported mean averages over, as the paper states it |

Categorical (one-hot on training vocabulary; `missing` and `not_applicable` are their own states;
unseen test categories encode as all-zeros):

| column | states |
|---|---|
| `sampler` | ddpm, ddim, dpm_solver, not_applicable, missing |
| `checkpoint_selection` | last, best_online, offline_criterion, missing |
| `critic_layernorm` | yes, no, missing |
| `policy_extraction` | rev_kl, exp_weight, easy_bcq, missing (one-step RL operators) |
| `base_learner` | td3, crr, iql, dql, missing (EDP) |
| `discretization` | uniform, quantile, missing (Trajectory Transformer) |

`dataset` and `dataset_version` are always part of the conditions group.

Missing vs not-applicable vs zero are never collapsed: `missing` = not reported; `not_applicable` =
the method has no such quantity; `0` = explicitly zero.

## Lineage (Model E)

Conventional research label, from how the paper places itself — not scanner mechanism families:

| label | methods |
|---|---|
| diffusion_planning | Diffuser, DecisionDiffuser, AdaptDiffuser, HierarchicalDiffuser |
| diffusion_policy | DiffusionQL, IDQL, SfBC, QGPO, EDP, SRPO |
| offline_q_learning | CQL, IQL |
| behavior_regularized_actor_critic | TD3+BC, ReBRAC, FisherBRC, PLAS, OneStepRL |
| uncertainty_based_offline_rl | EDAC |
| model_based_offline_rl | MOPO, MOReL, COMBO |
| return_conditioned_supervised_learning | DecisionTransformer, DecisionConvFormer, QDT, RvS |
| sequence_model_planning | TrajectoryTransformer, TAP |

One-hot on the training vocabulary; kNN distance = mismatch (0/1).

## Models

| model | Ridge features | kNN distance groups |
|---|---|---|
| A | none — training mean | — |
| B | conditions | numeric, categorical |
| C | structure | structure |
| D | structure + conditions | structure, numeric, categorical |
| E | lineage + conditions | lineage, numeric, categorical |

## Predictors

- Ridge (sklearn, intercept). Alpha grid `{0.01, 0.1, 1, 10, 100, 1000}`, chosen by
  GroupKFold(min(5, #training methods)) over `method_id` inside the training rows, lowest mean MAE.
  Same pipeline for B, C, D, E; encoders fit on training rows only.
- kNN, k = 5, unweighted mean of the 5 nearest training rows. Group distances: structure
  `1 − Jaccard`; numeric = RMS of standardized differences over columns present in both rows
  (group unavailable for a pair if no column is jointly present); categorical = mean 0/1 mismatch
  over `dataset`, `dataset_version`, the categorical conditions, and the {present, missing,
  not_applicable} state of every numeric column; lineage = 0/1 mismatch. Combined = equal-weight
  mean over the groups available for the pair. Weights are not tuned.
