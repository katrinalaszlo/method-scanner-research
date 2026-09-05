# Method Scanner: finding shared mechanisms across ML papers with a local LLM

Kat Laszlo · 2026-08-24 → 2026-08-25 · 23 phases + 2 live tests + 1 experimental validation · 58-paper corpus

## Summary

Question: do Kalman filtering, JEPA, predictive coding, world models and RL share a predict-then-correct loop, and can a local 8B model see it from paper text?

Answer: yes for some pairs, no for the one everyone assumes. The shared loop is real between **self-supervised latent prediction (JEPA) and predictive-coding-as-a-learning-rule**, and between **particle filters and Kalman-style state estimation**. It is *not* real between Kalman filters and predictive coding — they share verbs (`predict`, `correct`) but act on different objects (state vs. parameters). A string-level comparison says they're the same; a typed comparison says they aren't.

The thesis this leaves: `predict` is universal — that's not the finding. The finding is that the **object** of the operation (`state`, `latent`, `parameter`, `value`, `trajectory`) is what splits fields. `operation@object` discriminates where `operation` alone can't. That turns "which methods are similar?" into "which mathematical object is being operated on?" — and almost every named "family" in ML turned out to be a lineage label wearing a mechanism's name.

## Pipeline

Four layers, each derived from the one below and revisable without re-running it:

1. **Extract.** Method section (arXiv PDF → `pdftotext` → numbered-heading finder, 1500 words) → qwen3:8b via Ollama → `{operations[], loss, failure}`. Few-shot prompt, four worked examples, one with `failure: null`. ~40 s/paper.
2. **Canonicalize.** Raw verbs → 23 canonical ops (`predict`, `correct`, `encode`, `sample`, `evaluate`, `improve`, …) by exact match, then head-verb fallback (flagged). Raw ops never overwritten.
3. **Type.** Second LLM pass assigns each op an object from a fixed ontology — `state`, `latent`, `parameter`, `input`, `sample`, `action`, `value`, `policy`, `task`, `trajectory`, … — with a supporting quote. Tuples look like `predict@latent`, `correct@parameter`.
4. **Compare.** Jaccard over op sets per paper, three ways: **lexical** (canonical ops), **strict** (op@object must match exactly), **relaxed** (objects match within groups, e.g. state ~ latent ~ trajectory). Within-family and between-family means; top pairs; `--query` for an unseen paper.

Trade-off between the layers: strict is high-precision and needs one ontology addition per new field before it can see that field; relaxed connects a new field on day one (Diffuser → DDIM 0.50) at the cost of false positives (a Kalman state and a Dreamer latent count as the same object). Lexical is what a keyword search would give. All three are reported everywhere; strict is the number to trust.

Read-time guards fix systematic typing errors without touching disk: `state` is kept only if the quote has a temporal marker, else demoted to `latent`; `act/select/sample/store` never take `state`.

## Corpus and final families

58 papers, 11 mechanism families + 1 outlier. Families were assigned from typed tuples, not from what the paper calls itself. Strict v2 within-family mean Jaccard:

| family | n | strict | core tuple |
|---|---|---|---|
| on-policy | 3 | 0.43 | PPO, TRPO, SAC: `evaluate@value` + `improve@policy/parameter` |
| meta-learning | 3 | 0.34 | MAML, Reptile, Meta-SGD: `sample@task` + `correct@parameter` |
| replay-off-policy | 5 | 0.21 | DQN, DDPG, DDQN, TD3, Rainbow: `store/sample@sample` + `select/act@action` |
| jepa | 5 | 0.21 | `encode@latent` + `predict@latent` |
| pc-learning | 9 | 0.13 | predictive coding as weight update: `correct@parameter` |
| particle | 5 | 0.11 | `sample@sample` + `correct/predict@state` |
| diffusion | 5 | 0.07 | `sample`, `predict`, `denoise` on noise/latent |
| state-estimation | 8 | 0.06 | `predict@state` + `correct@state` |
| pc-neuro | 3 | 0.06 | predict inputs, update neural state |
| worldmodels | 6 | 0.03 | latent dynamics + decoder |
| control | 5 | 0.00 | VGG, ResNet, Transformer, word2vec, Adam — negative control |
| a3c-outlier | 1 | — | matches nothing |

Lexical scores run 2–5× higher across the board (on-policy 1.00, jepa 0.45). Strict is the number to read; lexical is what you'd get from a keyword search.

## Findings

**1. Lineage ≠ mechanism. Every time.**
- *Predictive coding* is two families: PredNet/PCL+ predict inputs and update neural state; PC Networks / PC Beyond Backprop predict latents and update weights. Lexical merged them (0.55); typed split them (0.06). Re-split by mechanism, both halves cohere.
- *Filters* (Kalman, particle, PKF, Backprop KF): each paper describes a different piece of the loop. Within-family 0.08 lexical until re-split into state-estimation vs. particle.
- *RL*: "policy-learning" scored 0.02 — indistinguishable from controls — until the ontology got `value`/`policy` and the family was split. Then DDPG, labelled policy-gradient by every textbook, typed as the DQN replay loop (0.36 with DQN, 0.00 with PPO). SAC, off-policy by lineage, typed cleanly with the on-policy trio (`evaluate@value + improve@policy`). The data split RL as *on-policy vs. replay-based*, not policy-gradient vs. value-based.
- *Latent-generative* (VAE-dynamics papers: DVBF, Deep KF, BBVI, …) never cohered at any n. Five papers, five signatures. Dissolved; two members are orphans that fit nothing at strict.

**2. The bridges that survived every test.**
- **jepa × pc-learning** on `predict@latent + correct@parameter`. Top strict cross-family pair in Phases 11–22. I-JEPA × PC Networks 0.50 is the single strongest pair in the corpus. Reading: I-JEPA/BYOL and PC-as-backprop-alternative are the same mechanism — predict a latent, update weights on the error.
- **particle × state-estimation** on `predict@state + correct@state`. Differentiable Particle Filters × KalmanNet 0.50. Two filter lineages, one loop, typed identically — a positive control the design didn't plan for. The only bridge that survives the agreed-only (both typing passes concur) filter at pair level.
- **meta-learning × pc-learning** (0.13) and **meta-learning × on-policy** (0.08, MAML `improve@parameter` ↔ PPO/TRPO). Invisible before `task`/`value` existed in the ontology.

**3. The bridge that didn't: KalmanNet as a hub.** Lexically KalmanNet (`predict, correct`) tied at 1.00 with three predictive-coding papers. Ablating `predict` left a one-op `correct` link that tied KalmanNet to Adam as strongly as to any PC paper. Under typing the link dissolves: KalmanNet corrects `@state`, PC Review corrects `@parameter`. Same verb, different mechanism.

**4. False bridges come from one bug and one guard removes them.** The typer over-assigns `state` to anything vector-shaped (BYOL `project@state`, SimSiam `predict@state`). That created SimSiam × Neural Kalman Filtering 0.40 — a self-supervised projector matched to a filter — and it survived a stricter prompt *and* the two-pass agreement filter. A code-level guard (keep `state` only with a temporal marker in the quote) removed exactly that pair and none of the true ones; jepa coherence went 0.12 → 0.21. Agreement is not correctness; prompting doesn't fix systematic bias; a read-time rule does.

**5. Delta papers don't restate the loop.** DDQN, TD3, Rainbow inherit DQN/DDPG's agent loop and write only their contribution (a target rule, a clipped Q, a distributional projection). Extracted from their own text they score ~0 with each other. Prepending the base paper's Algorithm 1 lifted replay-off-policy 0.05 → 0.21 and DDIM × DDPM 0.00 → 0.50. Control: injecting MAML's loop into Reptile — which prints its own Algorithm 1 — changed nothing (0.40 → 0.40), so injection doesn't inflate papers that already restate the loop. It is a per-paper property, not per-lineage. Caveat: injected records score partly on inserted text; their own contributions are 0–1 ops each, so on own-ops-only the family is 0.04. That's what an incremental lineage is.

**6. The ontology grows one object per new field.** RL was invisible until `value`/`policy`; meta-learning until `task`; Diffuser (planning with diffusion) until `trajectory`. Each addition made the field visible and did not disturb the others. Adding `trajectory` moved Diffuser's strict score from 0.20 to **0.00** — correctly: no corpus paper diffuses a trajectory. A new object type isolates a paper until a second paper acts on the same object. Relaxed connects a new field on day one (Diffuser → DDIM 0.50); strict says "nothing here does this".

**7. Live tests, pre-registered.**

| paper | pre-registered prediction | result |
|---|---|---|
| Bidirectional PC for action anticipation (abstract) | pc-learning, on `predict@latent + correct@parameter` | wrong: `refine@latent` + `train@parameter` → jepa / world-model. "Predictive coding" in the title was lineage |
| DVBF | state-estimation | wrong: no `correct@state`; it's a recognition model + transition + decoder |
| Snoek Bayesian optimization | no match | right — and *because* of typing; lexically it scored 0.67 with predictive coding |
| MAML | pc-learning or new | right for a generic reason (`correct@parameter`); defining structure invisible until `task` added |

**Generalization test, unseen field — Diffuser** (planning with diffusion, robotics). Pre-registered: reviewer said worldmodels / KalmanNet; I said diffusion / DDIM. Result: DDIM 1.00 lexical, 0.50 relaxed, family diffusion. Strict needed `trajectory` — the addition typed Diffuser correctly and isolated it (0.00) until a second trajectory-mechanism paper enters the corpus. Two testable transfers came out of the match: DDIM's deterministic sampler at 5–10× fewer steps into Diffuser at ≤5% return loss; SMC resampling by exp(J) in place of gradient guidance (from the Differentiable Particle Filter match).

Scorecard: the system refuses lineage guesses reliably; it finds the right neighbour when the ontology has the object; it goes blind when it doesn't, and says so (`@other`).

## First experimental validation (Phase 23)

The Diffuser × DDIM match produced a falsifiable prediction: *swap DDIM's deterministic sampler into Diffuser at 5–10× fewer denoising steps and D4RL return stays within ~5%.* We implemented DDIM in the released Diffuser code and ran the pretrained T=20 locomotion checkpoints on halfcheetah / hopper / walker2d, DDPM-20 vs DDIM at 20/10/5/2 steps, three seeds, paired to the same-seed baseline (`phase23/phase23_results.md`).

| paired drop vs DDPM-20 (+ = worse) | DDIM-20 | DDIM-10 (2×) | DDIM-5 (4×) | DDIM-2 (10×) |
|---|---|---|---|---|
| halfcheetah | +1.6 ± 4.7 | +1.8 ± 3.8 | **+5.1 ± 0.8** | +9.7 ± 3.8 |
| hopper | −8 ± 22 | **−75 ± 19** | −32 ± 23 | −12 ± 12 |
| walker2d | **+20 ± 21** | −5.3 ± 3.3 | −13 ± 36 | +25 ± 22 |

**Supported at 2×, borderline at 4×, refuted at 10×.** DDIM-10 is baseline-or-better on all three envs and on hopper it is the best planner in the sweep (episodes 690/776/1000 vs a baseline that falls by step 540 every seed). The 4× number on halfcheetah landed on the pre-registered threshold to the decimal. Two things the prediction missed, both horizon × schedule interactions rather than mechanism: walker gets *worse* with the deterministic sampler at full 20 steps (2/3 seeds) while improving at 10, and hopper's gain is a stability effect, not reward rate. Neither is something the ontology could carry — `stochastic@trajectory` would not have predicted walker's direction.

What this validates: the chain from typed structural match → quantitative hypothesis → code patch → measured result closes, on a laptop, in a day. The prediction's mechanism clause (Diffuser is the DDIM sampler over τ; nothing depends on the Markov chain) held. Its quantitative clause held at the conservative end and failed at the aggressive end, which is what a pre-registered range is for. Practical payoff: DDIM-10 halves planning cost at no loss.

## What this means

1. **Taxonomy.** Mechanism families (on-policy, replay-off-policy, pc-learning, pc-neuro) are a correction to lineage labels (RL, predictive coding, filters). The correction is mechanical and auditable: every reassignment in `families.json` cites the tuples that forced it.
2. **Borrowing.** The pairs that share two typed ops are the ones where a component can move: DPF × KalmanNet (`predict@state + correct@state`) says particle and Kalman correction steps are interchangeable at the mechanism level; I-JEPA × PC Networks (`predict@latent + correct@parameter`) says a PC local update could stand in for JEPA's EMA weight update. Single-op matches — especially `correct@parameter`, which links MAML, Adam and half the corpus — say only "weights get updated" and are not borrowing signals.
3. **Hypothesis generation.** Each live test produced a concrete, falsifiable transfer (Diffuser ← DDIM sampler; bidirectional PC ← two-filter smoothing, not JEPA). The hypotheses are only as specific as the shared tuples; the pipeline is honest about that. One has now been run (Phase 23).
4. **How papers are written.** Incremental papers omit the loop they inherit. Any automated structural reading of a lineage will find its base papers coherent and its delta papers orphaned unless the base loop is supplied. That's a property of the literature, not the tool.

## Thesis

`operation@object` is a more reliable structural descriptor than `operation`. The object is the discriminator: same verb on `state` vs `parameter` vs `latent` is three mechanisms, and the ontology of objects — not the vocabulary of verbs — is what has to grow as the corpus meets new fields.

## Decisions that mattered

- **Few-shot over rules.** Negative instructions were ignored by small models; three to four worked examples fixed extraction. A closed-vocabulary prompt (external suggestion) made the model emit the vocabulary instead of the paper — rejected on a 6-paper A/B.
- **qwen3:8b over llama3.2:3b**, native Ollama API for `think:false`. Every llama failure mode gone.
- **Canonicalize before scaling.** Uncontrolled verbs score real matches as zero; scaling first scales noise.
- **Scale by hypothesis.** 5 papers per family + 5 controls tests a claim; random papers don't.
- **Derived layers, raw preserved.** Canonical ops and guards are computed at read time; every vocab or guard change applies to the whole corpus instantly and is reversible.
- **Method sections, not abstracts.** Kalman 1960's abstract never says "predict". The body does. Abstracts carry contribution, not mechanism.
- **Pre-register, snapshot, keep negative results.** Every phase has a target written before the run and a verdict after; failed targets are in the findings with the reason — roughly half the pre-registered targets failed, and those are the more useful results.

## Limitations

- **n.** 3–9 papers per family. Family means at n=3 move ±0.1 on one paper.
- **Typing noise.** Pass-1 vs pass-2 agreement 72–84%; a re-run on identical text drifts ~1 tuple/paper. Family-level structure is stable across passes; individual pair scores are one re-typing away from zero.
- **Section finder** hits ~90% by heading; misses were wrong-section (TD3) or fallback windows (Rainbow, DDQN). Pinning fixed the section, not the delta-paper problem.
- **Extractor summarises pseudocode** when prose follows it (TD3's injected loop collapsed to three `update` ops, twice).
- **One model, one prompt family.** No cross-model typing rater yet.
- **Injection is manual.** Deciding which papers are deltas of which base is a human step.
- **Phase 23:** three seeds; hopper/walker scores are dominated by fall timing; gymnasium-v4 dynamics rather than the paper's mujoco-py v2 (baseline re-run under the same deviation).

## Next

0. η sweep (does adding noise back at 10/20 steps fix walker / break hopper?) and the second Diffuser hypothesis — SMC resampling by exp(J(τ)) replacing gradient guidance, from the particle-filter match.
1. Second trajectory-mechanism paper (Decision Diffuser, Trajectory Transformer) so `@trajectory` has a strict neighbour.
2. Measure typing drift corpus-wide before any whole-corpus re-type.
3. Store `inherited_ops` vs own ops per record in the UI/CSV, not just in the comparator.
4. Claude API as an independent typing rater.
5. Compare tab in the UI (all comparison is script-only).

## Where things are

`GitHub/method_scanner/` — `app.py` (server, extractor, canonical vocab, guards), `fulltext.py` (arXiv → section, `--from` pin), `semantic_type.py`, `compare.py` (`--semantic --pass v2 --relaxed --own --families mechanism --query`), `phase21_inject.py`, `phase22_split.py`, `families.json`, `semantic_ontology.json`. Evidence: `findings.md` (per-phase results), `task_plan.md`, `progress.md`, `predictions/` (reports and live-test records), `matrices/` (every similarity matrix, per phase), `results_phase*_snapshot/` (records before each re-extraction). Experiment: `phase23/` (Diffuser clone + DDIM patch, shims, sweep, results, plot).
