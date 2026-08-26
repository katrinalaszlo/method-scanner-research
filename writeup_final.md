# Method Scanner

## Finding shared mechanisms across ML papers, and testing one of them

Kat László · August 2026 · 23 phases · 58-paper corpus · one experimental validation

---

## Abstract

Scientific methods are named by lineage — Kalman filter, JEPA, predictive coding, reinforcement learning — and lineage is a poor guide to what a method actually does. Method Scanner is a four-layer pipeline that reads a paper's method section with a local 8B model, reduces it to a set of typed operations of the form `operation@object` (`predict@state`, `correct@parameter`, `sample@trajectory`), and compares papers on those sets. Applied to 58 papers across filtering, self-supervised learning, predictive coding, world models, diffusion, reinforcement learning and meta-learning, it finds that the **object** of an operation, not the operation, is what separates mechanisms: `predict` is universal, but `predict@state`, `predict@latent` and `predict@parameter` are three different things. Nearly every lineage label in the corpus split into two mechanism families or dissolved. The pipeline's typed comparison recovers two real cross-field bridges (self-supervised latent prediction ↔ predictive coding as a learning rule; particle filters ↔ Kalman-style state estimation) and refuses the one most people assume (Kalman filtering ↔ predictive coding).

To test whether the analogies are more than similarity scores, we took a structural match the scanner produced between an unseen robotics paper (Diffuser: planning with diffusion) and an image-generation sampler (DDIM), turned it into a pre-registered quantitative prediction — *DDIM's deterministic sampler at 5–10× fewer denoising steps keeps Diffuser's D4RL return within ~5%* — and ran it: 45 episodes, three seeds, three environments, paired against the same-seed baseline. The prediction held at 2× fewer steps on every environment, landed on the 5% threshold to the decimal at 4×, and failed at 10×. The scanner did not predict one thing the data showed: on walker2d the deterministic sampler at full step count is *worse* than the stochastic one. That is reported as an open observation, not folded into the result. The scientific claim is not the speedup; it is that a pipeline reading text produced a falsifiable engineering prediction whose bounds the experiment confirmed.

---

## 1. Problem

Ask whether Kalman filtering, JEPA and predictive coding share a mechanism and you get three answers depending on who you ask: a control theorist sees a predict–update loop in all three, a self-supervised-learning researcher sees an encoder–predictor–target loop in two of them, a neuroscientist sees error propagation in one. All three are reading the same verbs — *predict*, *correct*, *update* — and disagreeing about whether the verbs mean the same thing.

The lineage labels make this worse. "Predictive coding" covers a 1999 cortical model and a 2022 backprop alternative. "Reinforcement learning" covers on-policy trust-region methods and replay-buffer Q-learning that share no update rule. "Filters" covers Kalman gain computation, particle resampling, and a learned association step. The label tells you where the paper came from, not what its inner loop does.

The goal of this project was to strip methods to their functional components and compare those, so that a structural analogy across fields is something you can measure, and — the real test — something you can act on. If the pipeline says two methods share a mechanism, a component from one should be swappable into the other, and the swap should behave the way the shared structure predicts.

---

## 2. Pipeline

Four layers. Each is derived from the one below and can be revised without re-running it; raw model output is never overwritten.

### 2.1 Extract

Input is a paper's **method section**, not its abstract. This was decided early and by experiment: the 1960 Kalman paper's abstract never says *predict* — it talks about the Wiener problem and gain computation — while the body does. Abstracts carry the contribution; method sections carry the mechanism.

Sourcing: arXiv ID → PDF → `pdftotext` → a heading walker that finds the first numbered top-level section that isn't introduction/background/related work and takes 1500 words from it (four pdftotext layouts handled: same-line, split-line, roman numerals, small-caps). It hits by heading on ~90% of papers; the rest fall back to a fixed window 20% into the text, and the fallback is recorded in the record's `source`. A `--from <regex>` pin was added later for papers where the heuristic picks the wrong section.

Extraction is done by **qwen3:8b** through Ollama (native API, `think:false`, temperature 0), with a four-example few-shot prompt that asks for `{operations[], loss, failure}`. Two things mattered here more than anything else in the prompt:

- **Examples, not rules.** Negative instructions ("do not list *derive*, *formulate*, …") were ignored by 3B–8B models. Three to four worked examples fixed the same failures.
- **A `null` example.** With only positive `failure` examples the model echoed the example's failure phrase onto papers that had none ("collapse without an asymmetric design" appeared on the World Models paper). One example with `failure: null` and a rule against reusing example wording brought the echo rate to 1 in 30, and a read-time guard catches the survivors.

A closed-vocabulary prompt proposed externally was A/B'd on six hard papers and rejected: the model emitted the vocabulary list instead of the paper (BYOL, explicitly non-contrastive, got `contrast`; DreamerV3 got `augment, add_noise`), and the `failure` field went null on all six.

Model choice was a bake-off on the first ten papers: llama3.2:3b produced nouns as operations, duplicates, and echoes; qwen3:4b could not disable its thinking mode and returned empty output; qwen3:8b was clean at ~5 s per abstract, ~40 s per method section.

### 2.2 Canonicalize

Raw verbs are mapped to a 23-item controlled vocabulary — `predict`, `correct`, `encode`, `sample`, `reconstruct`, `evaluate`, `improve`, `act`, `plan`, `transition`, … — by exact match, then by head verb (`update covariance` → `update` → `correct`), with head-verb matches flagged so they stay auditable. Unmapped operations are kept and shown.

This layer is derived at read time and never stored. Every vocabulary change applies to the whole corpus instantly and is reversible. The reason to have it at all: uncontrolled verbs (`predict` / `forecast` / `infer`; `update` / `suppress` / `associate`) score real matches as zero, and scaling the corpus before controlling the vocabulary scales noise.

### 2.3 Type

A second LLM pass reads the same method section plus the extracted operations and assigns each operation an **object** from a fixed ontology, with a supporting quote:

| object | definition |
|---|---|
| `state` | estimate of a dynamical system's hidden state (x_t, belief, track) |
| `latent` | learned representation / embedding |
| `parameter` | model weights |
| `input` | raw observation |
| `error`, `gradient` | learning signals |
| `sample`, `noise`, `distribution` | stochastic objects |
| `action` | control output |
| `value`, `policy`, `task` | added for RL and meta-learning (ontology v2) |
| `trajectory` | a state–action sequence as a whole (added for planning) |
| `other` | none of the above — leave raw, do not force |

The output is a set of tuples per paper: `predict@latent`, `correct@parameter`, `sample@sample`. The typing pass is where the pipeline's signal comes from and also where its noise comes from. Two independent passes agree on 72–84% of tuples; re-running on identical text drifts about one tuple per paper. Family-level results are stable across passes to the second decimal; individual pair scores can be one re-typing away from zero.

One systematic error needed a rule rather than a prompt: the model assigns `state` to anything vector-shaped (BYOL's projector output, ResNet's identity map). A stricter prompt did not fix it, and a two-pass agreement filter did not catch it because the model was consistently wrong. A read-time guard — keep `state` only if the supporting quote contains a temporal-dynamics marker, else demote to `latent` — removed exactly the false pairs it created (SimSiam × Neural Kalman Filtering 0.40) and none of the true ones. Two more guards followed the same pattern: `act/select/sample/store` never take `state`, and untyped objects on those verbs are filled in.

### 2.4 Compare

Jaccard similarity on each paper's tuple set, in three modes:

- **Lexical** — canonical operations only, objects ignored. What a good keyword search gives you.
- **Strict** — tuples must match exactly, `op@object`.
- **Relaxed** — objects match within compatibility groups (`state ~ latent ~ trajectory`; `error ~ gradient ~ value`; `noise ~ distribution ~ sample`). `parameter` and `input` never relax.

Within-family and between-family means, ranked cross-family pairs, and a `--query` mode that scores an unseen paper against the corpus. Families are assigned in a separate `families.json` that overrides lineage labels with mechanism labels; every reassignment records the tuples that forced it.

The three modes trade off in a specific way. Strict is high-precision and can only see a field once the ontology has that field's object — before `value` and `policy` existed, five RL papers scored 0.02 within-family, indistinguishable from the control set. Relaxed connects a new field on the day it arrives, at the cost of false positives (a Kalman state and a Dreamer latent count as the same object). Lexical runs 2–5× higher than strict everywhere and merges things that should not be merged. All three are reported; strict is the number to trust.

---

## 3. Corpus and families

58 papers, chosen by hypothesis rather than volume: five per candidate family plus five negative controls (VGG, ResNet, Transformer, word2vec, Adam), expanded where a family cohered and pruned where it did not. Every arXiv ID was verified against the API — six of the first thirty guesses from memory were wrong papers.

Final structure, strict within-family mean Jaccard:

| family | n | strict | core tuple |
|---|---|---|---|
| on-policy | 3 | 0.43 | PPO, TRPO, SAC: `evaluate@value` + `improve@policy/parameter` |
| meta-learning | 3 | 0.34 | MAML, Reptile, Meta-SGD: `sample@task` + `correct@parameter` |
| replay-off-policy | 5 | 0.21 | DQN, DDPG, DDQN, TD3, Rainbow: `store/sample@sample` + `select/act@action` |
| jepa | 5 | 0.21 | I-JEPA, V-JEPA, BYOL, SimSiam, …: `encode@latent` + `predict@latent` |
| pc-learning | 9 | 0.13 | predictive coding as a weight-update rule: `correct@parameter` |
| particle | 5 | 0.11 | `sample@sample` + `correct/predict@state` |
| diffusion | 5 | 0.07 | `sample`, `predict`, `denoise` on noise/latent |
| state-estimation | 8 | 0.06 | `predict@state` + `correct@state` |
| pc-neuro | 3 | 0.06 | predict inputs, update neural state |
| worldmodels | 6 | 0.03 | latent dynamics + decoder |
| control | 5 | 0.00 | negative control |
| a3c-outlier | 1 | — | matches nothing in either RL family |

The numbers are small in absolute terms — Jaccard on sets of three to eight tuples, with typing noise — and the ranking, the zero on the controls, and the gap between strict and lexical carry more information than any single value. The families with n=3 move by ±0.1 on one paper.

---

## 4. Findings

### 4.1 Lineage ≠ mechanism, every time it was tested

**Predictive coding is two families.** PredNet and Predictive Coding Light predict *inputs* and update *neural state*; PC Networks, PC Beyond Backprop and PC Review predict *latents* and update *parameters*. Lexically they are one family (0.55); typed, they are two (0.06 between). Split by mechanism, both halves cohere. The neuroscience half then broke again when the corpus grew — three spiking/cortical papers shared verbs with PredNet but acted on different objects — and its lexical score *rose* (0.33 → 0.42) while its strict score collapsed (0.17 → 0.02). Lexical coherence with typed incoherence is the signature of a lineage label.

**Filters are three families.** Each filtering paper describes a different piece of the loop — KalmanNet the recursion, PKF the association step, PF-Net the resampling, Backprop KF the training. Within-family 0.08 lexical until re-split into state-estimation and particle, after which particle × state-estimation became the second-strongest cross-family bridge in the corpus (§4.2).

**Reinforcement learning is two families, and not the two the textbooks say.** Five Sutton-lineage papers scored 0.02 within-family until the ontology gained `value` and `policy`. With those, PPO, TRPO and SAC typed `evaluate@value + improve@policy/parameter` and cohered at 0.43. DQN typed `store@sample, sample@sample, select@action, act@action`. The expected split was policy-gradient vs value-based. Then DDPG — policy-gradient by every textbook — typed as the DQN replay loop with an actor attached (0.36 with DQN, 0.00 with PPO), and SAC — off-policy by lineage — typed cleanly with the on-policy trio. The data split RL as **on-policy vs replay-based**. A3C matched neither and is held out.

**Latent-generative is not a family.** Five VAE-with-dynamics papers (DVBF, Deep Kalman Filters, BBVI, L-VSSF, Variational Tracking) produced five different tuple signatures and 0.00 pairwise at every corpus size. The label was dissolved; two of the papers fit no family at strict and are recorded as orphans rather than forced.

### 4.2 The bridges that survived every test

- **jepa × pc-learning**, on `predict@latent + correct@parameter`. Top strict cross-family pair from the first typed run to the last. I-JEPA × PC Networks & Inference Learning at 0.50 is the strongest pair in the corpus. Reading: I-JEPA/BYOL and predictive-coding-as-backprop-alternative are the same mechanism — predict a latent, update weights on the error.
- **particle × state-estimation**, on `predict@state + correct@state`. Differentiable Particle Filters × KalmanNet at 0.50. Two filter lineages, one loop, typed identically. This is the only bridge that survives the agreed-only filter (both typing passes concur) at pair level, and it is a positive control the design did not plan for.
- **meta-learning × pc-learning** (0.13) and **meta-learning × on-policy** (0.08; MAML's `improve@parameter` ↔ PPO/TRPO). Both invisible before `task` and `value` existed.

### 4.3 The bridge that dissolved

Lexically, KalmanNet (`predict, correct`) tied at 1.00 with three predictive-coding papers — the Kalman ↔ PC analogy everyone expects. An ablation dropping `predict` left a one-verb `correct` link that tied KalmanNet to Adam, an optimizer, as strongly as to any PC paper; a hub that connects equally to a control is not a hub. Under typing the link dissolves entirely: KalmanNet corrects `@state`, PC Review corrects `@parameter`. Same verb, different mechanism. The Kalman–PC analogy is a story about the word *correct*, not about a shared loop.

### 4.4 Delta papers do not restate the loop

DDQN, TD3 and Rainbow inherit DQN/DDPG's agent loop by citation and write only their contribution — a target rule, a clipped Q, a distributional projection. Extracted from their own text they score ~0 with each other and with their base papers, and re-extracting from the correct section did not help. Prepending the base paper's Algorithm 1 to each delta paper's section lifted the replay-off-policy family from 0.05 to 0.21 and DDIM × DDPM from 0.00 to 0.50. The control: injecting MAML's loop into Reptile — which prints its own Algorithm 1 — changed nothing (0.40 → 0.40). Injection surfaces a loop the paper omits; it does not inflate a paper that already restates one.

Two caveats. Scoring injected papers partly measures inserted text; on their own contributions alone (0–1 ops each) the family is 0.04, which is the honest description of an incremental lineage. And deciding which papers are deltas of which base is a manual step.

### 4.5 The ontology grows one object per new field

RL was invisible until `value` and `policy`. Meta-learning until `task`. Robotics planning (§5) until `trajectory`. Each addition made its field visible and disturbed nothing else. The cost of the strict layer is that it needs this; the reason to keep it is that the addition tells you something — when `trajectory` was added, Diffuser's strict score went from 0.20 to 0.00, correctly, because no corpus paper diffuses a trajectory. A new object isolates a paper until a second paper acts on the same object. Strict says "nothing here does what this does"; relaxed connects the paper to its nearest analogue on the same day.

### 4.6 Pre-registered live tests

| paper | predicted before running | result |
|---|---|---|
| Bidirectional PC for action anticipation | pc-learning on `predict@latent + correct@parameter` | wrong — typed `refine@latent + train@parameter`, a JEPA/world-model loop; "predictive coding" in the title was lineage |
| DVBF | state-estimation | wrong — no `correct@state`; a recognition model + transition + decoder |
| Snoek Bayesian optimization | no match | right, and *because* of typing — lexically it scored 0.67 with predictive coding |
| MAML | pc-learning or new | right for a generic reason (`correct@parameter`); its structure was invisible until `task` |
| Diffuser | reviewer: worldmodels / KalmanNet; author: diffusion / DDIM | author's prediction right — DDIM 1.00 lexical, 0.50 relaxed |

The pattern: the system refuses lineage guesses reliably, finds the right neighbour when the ontology has the object, and goes blind — says `@other` — when it doesn't.

---

## 5. From a match to a prediction: Diffuser and DDIM

**Diffuser** (Janner et al. 2022) plans for robot locomotion by generating whole state–action trajectories with a diffusion model and executing the first action. It was chosen as an unseen paper from a field not in the corpus. Extracted: `predict trajectory, denoise, sample` → canonical `predict, reconstruct, sample`; loss MSE; failure "planners that exploit learned models by finding adversarial examples", grounded in the text.

Against the 58-paper corpus:

| mode | nearest | family |
|---|---|---|
| lexical | **DDIM 1.00** [`predict, reconstruct, sample`]; DDPM, EDM 0.67; Differentiable PF 0.60 | diffusion 0.55 |
| strict, before `trajectory` | 0.20, four-way tie on `sample@sample` | undecidable |
| strict, after `trajectory` | 0.00 — Diffuser's tuples are all `@trajectory`, nothing in the corpus is | isolated, correctly |
| relaxed, after `trajectory` | **DDIM 0.50** [`predict, reconstruct @represented_state`]; Differentiable PF 0.29 | diffusion / jepa 0.14 |

Chronology matters here. The hypothesis was written from the **lexical** match — `predict, reconstruct, sample` shared with DDIM at 1.00 — and from reading Diffuser's tuples and section text, *before* `trajectory` existed in the ontology. At that point strict could not decide (a four-way tie on the single tuple `sample@sample`) because the typer had put Diffuser's other two operations at `@other`. `trajectory` was added afterwards; strict then went to 0.00, correctly, and relaxed went to 0.50 because the author placed `trajectory` in the `represented_state` group alongside `state` and `latent` (a plan is a state sequence). That group membership is a human decision, not something the tool discovered. The relaxed match confirmed that the lexical link survived the new object type; it did not generate the hypothesis.

The structural reading behind the hypothesis is that **Diffuser's planner is the DDPM/DDIM sampler run over τ = (s₀ a₀ s₁ a₁ …) instead of an image**, and that nothing in Diffuser's method depends on the stochastic Markov reverse chain — only on the learned ε_θ(τ, i) and the value-guidance gradient.

If that reading is right, a specific component swap should work: replace Diffuser's DDPM sampler with DDIM's deterministic σ=0 sampler and cut the number of denoising steps. Pre-registered prediction:

> *Diffuser's planning performance is insensitive to the stochasticity of the reverse chain. DDIM (σ=0) at 5–10× fewer denoising steps keeps D4RL normalized return within ~5% of the DDPM baseline.*

Null: performance depends on chain stochasticity or step count; the swap degrades return by more than 5%.

---

## 6. Experimental validation

### 6.1 Setup

| component | detail |
|---|---|
| codebase | jannerm/diffuser @ 7ea4228, released pretrained locomotion checkpoints (diffusion epoch 800k, value 160k) |
| tasks | halfcheetah-medium-v2 (planning horizon H=4), hopper-medium-v2 (H=32), walker2d-medium-v2 (H=32) |
| baseline | DDPM sampler at **T=20** — the released checkpoints were trained with 20 diffusion steps, not the 100 assumed in the draft plan |
| swap | DDIM σ=0 on the same trained ε-model over a strided subset of the trained schedule; value guidance (scale, `t_stopgrad`, `n_guide_steps`, gradient scaled by posterior variance) applied per step exactly as in the original `n_step_guided_p_sample`. ~80 lines added, routed by a `sampler` flag; nothing else in the planner changed |
| sweep | DDIM at 20, 10, 5, 2 steps (1×, 2×, 4×, 10× fewer) |
| protocol | 64 candidate plans per environment step, first action executed, 1000-step episodes (hopper and walker terminate on a fall), seeds {0, 1, 2} |
| hardware | Apple M4, PyTorch on MPS; 0.6–1.8 s per environment step at 20 denoising steps |
| deviation | dynamics run on gymnasium `*-v4` (MuJoCo 3.x) rather than the paper's `*-v2` (mujoco-py 2.0, x86-only and not viable on this machine). Datasets are the original D4RL v2 HDF5 files, so the checkpoints' normalizers match training. The DDPM baseline was re-run under the same deviation, so every comparison is internal |
| port check | DDPM-20 mean of 3 seeds: 43.2 / 47.5 / 71.5 vs paper 44.2 / 58.5 / 79.2 |
| analysis | **paired**: each DDIM run minus the DDPM-20 run of the same seed and environment, then mean ± sd over seeds. The baseline itself moves 6–25% between seeds, so unpaired means are misleading |

45 episodes. The 5% threshold was fixed before the first run and not revisited.

### 6.2 Results

Paired drop versus same-seed DDPM-20 (%, positive = worse; mean ± sd over 3 seeds):

| env (H) | DDIM-20 (1×) | DDIM-10 (2×) | DDIM-5 (4×) | DDIM-2 (10×) |
|---|---|---|---|---|
| halfcheetah (4) | +1.6 ± 4.7 | **+1.8 ± 3.8** | **+5.1 ± 0.8** | +9.7 ± 3.8 |
| hopper (32) | −7.9 ± 22 | **−75 ± 19** | −32 ± 23 | −12 ± 12 |
| walker2d (32) | **+20 ± 21** | **−5.3 ± 3.3** | −13 ± 36 | +25 ± 22 |

Scores (D4RL ×100, mean ± sd):

| env | DDPM-20 | DDIM-20 | DDIM-10 | DDIM-5 | DDIM-2 |
|---|---|---|---|---|---|
| halfcheetah | 43.2 ± 1.6 | 42.4 ± 0.6 | 42.3 ± 0.3 | 40.9 ± 1.4 | 38.9 ± 0.2 |
| hopper | 47.5 ± 5.6 | 50.5 ± 5.4 | **83.5 ± 16.4** | 62.1 ± 8.1 | 53.8 ± 11.9 |
| walker2d | 71.5 ± 13.6 | 57.8 ± 20.9 | 75.5 ± 16.4 | 77.4 ± 10.3 | 55.4 ± 24.9 |

Episode lengths, where the score on hopper and walker is decided (1000 = no fall):

| hopper | s0 | s1 | s2 | | walker2d | s0 | s1 | s2 |
|---|---|---|---|---|---|---|---|---|
| DDPM-20 | 459 | 437 | 539 | | DDPM-20 | 1000 | 724 | 1000 |
| DDIM-20 | 478 | 557 | 459 | | DDIM-20 | 1000 | 473 | 662 |
| DDIM-10 | **690** | **776** | **1000** | | DDIM-10 | 1000 | 688 | 1000 |
| DDIM-5 | 531 | 688 | 656 | | DDIM-5 | 1000 | 1000 | 811 |
| DDIM-2 | 465 | 482 | 676 | | DDIM-2 | 676 | 471 | 910 |

### 6.3 Reading the results

**At 2× fewer steps (DDIM-10) the prediction holds on all three environments.** halfcheetah and walker are within the band with tight spread (+1.8 ± 3.8, −5.3 ± 3.3). Hopper is not merely within the band: every DDIM-10 episode (690, 776, 1000 steps) outlasts every baseline episode (437–539), on every seed. Reward per step is flat across configurations; the gain is entirely that the planner stops falling.

**At 4× (DDIM-5) the prediction is borderline, and the borderline is sharp.** halfcheetah lands at −5.1 ± 0.8% — on the pre-registered threshold to the decimal, with the tightest spread in the table. Hopper is above baseline on all three seeds. Walker is above baseline on mean but with one seed's fall inside the spread.

**At 10× (DDIM-2) the prediction fails.** halfcheetah loses 9.7%; walker falls on all three seeds (−25 ± 22%). Only hopper tolerates two steps.

**On stochasticity, the prediction was right on two environments and wrong on one.** DDIM-20 versus DDPM-20 — same step count, only the chain noise removed — is a wash on halfcheetah (+1.6 ± 4.7) and hopper (−8 ± 22). On walker2d it is a loss: the deterministic sampler at full 20 steps fell early on two of three seeds (+20 ± 21%), while the same deterministic sampler at 10 steps was above baseline on all three. The effect is non-monotone in step count, so it is not "fewer steps make coarser plans". This was not predicted and is not explained by the result. It is recorded as an open observation; the natural follow-up — a sweep over the DDIM noise parameter η at 10 and 20 steps on hopper and walker — was scripted and then deferred for time.

### 6.4 What was tested, and what was found

The result to take from this experiment is not "Diffuser can be made 4× faster". It is that **a pipeline reading paper text produced a quantitative engineering prediction whose bounds the experiment confirmed.** The scanner said the two methods share a sampler over different objects; that the chain's stochasticity is not part of the shared mechanism; and that the swap would hold to about 5% at 5–10× fewer steps. The experiment found the mechanism clause true, the stochasticity clause true on two of three environments, and the quantitative clause true at the conservative end of its range (2×, 4×) and false at the aggressive end (10×). The pipeline identified where the trade-off holds and where it breaks. A prediction that had said "always works" or "never works" would have been less informative than this one, and either would have been wrong.

What the scanner could not have carried: the two surprises are horizon × schedule interactions — walker's non-monotone response to step count, hopper's stability gain — and those are hyperparameters of the paper, not mechanism. A `stochastic@trajectory` object would not have predicted walker's direction.

The practical consequence is real but secondary: DDIM-10 halves planning cost at no loss on all three environments; DDIM-5 quarters it at ≤5% on halfcheetah and a gain on hopper. Per 1000-step episode on this hardware: DDPM-20 ≈ 16 min, DDIM-10 ≈ 9, DDIM-5 ≈ 5.

### 6.5 Limitations of the experiment

- **Three seeds.** On hopper and walker the score is decided by when the agent falls, so a single seed can move 20 points and the standard deviations are large. The sign of a paired mean is more trustworthy than its magnitude; the tight halfcheetah numbers are the only ones where the magnitude is precise.
- **v4 dynamics, not v2.** Absolute scores are 1–11 points below the paper's; the baseline is re-run under the same deviation, so relative comparisons are internal, but the transfer of these exact numbers to the paper's setup is not established.
- **One planning hyperparameter set.** The value-guidance scale and `t_stopgrad` were left at the released defaults for every configuration; a DDIM-tuned guidance scale might move the 10× result.
- **Wall-clock timings** varied 2× across the run from unrelated load on the machine; returns are unaffected.

---

## 7. Method decisions that mattered

- **Few-shot over rules; a null example; reject closed vocabularies.** Small models follow examples and ignore prohibitions.
- **Method sections over abstracts.** Abstracts carry contribution, not mechanism.
- **Canonicalize before scaling.** Uncontrolled verbs zero out real matches.
- **Scale by hypothesis.** Five papers per candidate family plus controls tests a claim; a random corpus does not.
- **Derived layers, raw preserved.** Canonical mapping and guards are computed at read time; every change is corpus-wide and reversible.
- **Fix systematic model errors in code, not prompts.** The `state` over-assignment survived a stricter prompt and a two-pass agreement filter. A read-time rule removed exactly the false pairs.
- **Pre-register, snapshot, keep negative results.** Every phase wrote its target before the run and its verdict after. Roughly half the pre-registered targets failed, and those are the more useful entries in the findings log.

---

## 8. Limitations of the pipeline

- **Corpus size.** 3–9 papers per family; n=3 families move ±0.1 on one paper.
- **Typing noise.** 72–84% two-pass agreement; ~1 tuple per paper drifts on a re-run. Family structure is stable; pair scores are fragile.
- **Section finding** fails on ~10% of papers, and the fallback window can land on the wrong content (Rainbow's fell on the distributional projection).
- **The extractor summarises pseudocode** when prose follows it — TD3's injected loop collapsed to three `update` operations, twice.
- **Single model, single prompt family.** No independent typing rater yet.
- **Delta-paper injection is manual.** The pipeline cannot yet tell which papers inherit their loop by citation.

---

## 9. Conclusion

The project set out to make cross-field structural analogies measurable, and ended by acting on one. Along the way the central finding was not the one it started looking for. Kalman filtering and predictive coding do not share a loop; they share a verb. The thing that actually separates methods is the **object** an operation acts on — `predict@state` is filtering, `predict@latent` is self-supervised learning, `predict@parameter` is meta-learning — and the vocabulary of objects, not of verbs, is what has to grow as the corpus meets new fields. Every lineage label the pipeline touched split along that line.

The typed comparison recovered two real bridges (JEPA ↔ predictive coding as a learning rule; particle ↔ Kalman-style estimation), refused a false one (KalmanNet as a hub), and, given an unseen robotics paper, matched it to an image-generation sampler and produced a pre-registered engineering prediction. That prediction was then run: 45 episodes, paired, three seeds. It held where the shared mechanism said it should (2× and 4× fewer denoising steps), failed where it said it might (10×), and turned up one thing it did not anticipate (walker2d prefers the stochastic chain at full step count), which is recorded as such.

The loop closed: structural analogy → falsifiable hypothesis → experiment → confirmed bounds, plus one honest surprise. That is what the scanner is for, and it works.

---

## Appendix: where everything is

`GitHub/method_scanner/` — `app.py` (server, extractor, canonical vocabulary, read-time guards), `fulltext.py` (arXiv → section, `--from` pin), `semantic_type.py` (typing pass), `compare.py` (`--semantic --pass v2 --relaxed --own --families mechanism --query`), `phase21_inject.py` (base-loop injection), `phase22_split.py` (inherited vs own ops), `families.json`, `semantic_ontology.json`.

Evidence: `findings.md` (per-phase results and verdicts), `task_plan.md`, `progress.md`, `predictions/` (live-test records and per-phase reports), `matrices/` (every similarity matrix by phase), `results_phase*_snapshot/` (records before each re-extraction).

Experiment: `phase23/` — Diffuser clone with the DDIM patch (`ddim_patch.diff`, `diffuser/sampling/ddim.py`), `shims/` (old-gym API over gymnasium + original D4RL data), `sweep.sh`, `collect.py`, `analyze.py`, `results_collected.csv`, `results_paired.csv`, `return_vs_steps.png`, `phase23_results.md`, per-run `rollout.json` under `diffuser/logs/`.
