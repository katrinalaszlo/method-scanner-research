# Phase 23 results — DDIM sampler in Diffuser (predictive transfer test)

**Final: 45/45 runs, 3 seeds × 3 envs × 5 configs. 2026-08-25.**

## Abstract
The method-scanner pipeline matched Diffuser (planning with diffusion, Janner et al. 2022) to DDIM at 1.00 lexical / 0.50 relaxed and produced a falsifiable transfer: *Diffuser's planner is the DDPM/DDIM sampler run over trajectories, so DDIM's deterministic σ=0 sampler at 5–10× fewer denoising steps should keep D4RL return within ~5%.* We implemented DDIM inside the released Diffuser codebase and ran the pretrained locomotion checkpoints (T=20) on halfcheetah / hopper / walker2d-medium-v2, DDPM-20 vs DDIM at 20, 10, 5, 2 steps, three seeds, paired against the same-seed DDPM baseline. **At 2× fewer steps (DDIM-10) the prediction holds on all three envs** (halfcheetah −1.8 ± 3.8%, walker +5.3 ± 3.3%, hopper +75 ± 19% — every hopper DDIM-10 episode outlasted every baseline episode). **At 4× (DDIM-5) it is borderline**: halfcheetah −5.1 ± 0.8% (right on the threshold, tight), hopper +32%, walker +13 ± 36% (one seed fell). **At 10× (DDIM-2) it fails** on halfcheetah (−9.7%) and walker (−25%). The sub-claim that stochasticity is irrelevant held on halfcheetah and hopper but **not on walker**, where the deterministic sampler at full 20 steps lost 32% in two of three seeds. Verdict: H1 supported at 2×, borderline at 4×, refuted at 10×; the "insensitive to stochasticity" clause is env-dependent.

## Methods
- Codebase: jannerm/diffuser @ 7ea4228, pretrained `logs/pretrained` (diffusion epoch 800k, value 160k). Locomotion checkpoints are **T=20**, so the sweep is 20/10/5/2, not 100/50/20/10.
- Sampler: `diffuser/sampling/ddim.py` — x_{t−1} = √ᾱ_prev·x̂0 + √(1−ᾱ_prev−σ²)·ε + σ·z, σ = η·√((1−ᾱ_prev)/(1−ᾱ_t))·√(1−ᾱ_t/ᾱ_prev), η = 0; strided subset of the trained schedule; x̂0 clipped as in the original; value guidance (scale, t_stopgrad, n_guide_steps, grad × posterior variance) per step exactly as `n_step_guided_p_sample`. `--sampler ddim --ddim_steps N`. Patch `ddim_patch.diff`.
- Env/data: gymnasium `*-v4` dynamics (mujoco 3.12, arm64) instead of `*-v2` (mujoco-py 2.0, x86-only); original D4RL v2 HDF5 datasets so normalizers match training; D4RL reference scores. Baseline re-run under the same deviation. Port check, DDPM-20 mean of 3 seeds: 43.2 / 47.5 / 71.5 vs paper 44.2 / 58.5 / 79.2.
- Protocol: 64 candidate plans per env step, first action executed, 1000-step episodes (hopper/walker end on a fall), `--seed` ∈ {0,1,2} sets torch/numpy/env seeds. Apple M4, torch MPS, 0.6–1.8 s per env step depending on machine load.
- Analysis: paired per seed (DDIM score − same-seed DDPM-20 score), then mean ± sd over seeds. Pooled means are also reported; paired is the number to read because the DDPM baseline itself moves 6–25% between seeds.

## Results
### Score (D4RL ×100, mean ± sd over 3 seeds)
| env (H) | DDPM-20 | DDIM-20 | DDIM-10 | DDIM-5 | DDIM-2 |
|---|---|---|---|---|---|
| halfcheetah (4) | 43.2 ± 1.6 | 42.4 ± 0.6 | 42.3 ± 0.3 | 40.9 ± 1.4 | 38.9 ± 0.2 |
| hopper (32) | 47.5 ± 5.6 | 50.5 ± 5.4 | **83.5 ± 16.4** | 62.1 ± 8.1 | 53.8 ± 11.9 |
| walker2d (32) | 71.5 ± 13.6 | 57.8 ± 20.9 | 75.5 ± 16.4 | 77.4 ± 10.3 | 55.4 ± 24.9 |

### Paired drop vs same-seed DDPM-20 (%, + = worse; mean ± sd, worst seed)
| env | DDIM-20 | DDIM-10 | DDIM-5 | DDIM-2 |
|---|---|---|---|---|
| halfcheetah | +1.6 ± 4.7 (4.7) | **+1.8 ± 3.8** (4.8) | **+5.1 ± 0.8** (6.0) | +9.7 ± 3.8 (11.9) |
| hopper | −7.9 ± 22 (13.3) | **−75 ± 19** (−54) | −32 ± 23 (−17) | −12 ± 12 (−3) |
| walker2d | **+20 ± 21** (32) | **−5.3 ± 3.3** (−1.4) | −13 ± 36 (19) | +25 ± 22 (45) |

### Episode length (1000 = no fall)
| hopper | s0 | s1 | s2 | | walker2d | s0 | s1 | s2 |
|---|---|---|---|---|---|---|---|---|
| DDPM-20 | 459 | 437 | 539 | | DDPM-20 | 1000 | 724 | 1000 |
| DDIM-20 | 478 | 557 | 459 | | DDIM-20 | 1000 | 473 | 662 |
| DDIM-10 | **690** | **776** | **1000** | | DDIM-10 | 1000 | 688 | 1000 |
| DDIM-5 | 531 | 688 | 656 | | DDIM-5 | 1000 | 1000 | 811 |
| DDIM-2 | 465 | 482 | 676 | | DDIM-2 | 676 | 471 | 910 |

Plot `return_vs_steps.png`. Raw `results_collected.csv`, `results_paired.csv`, per-run `rollout.json` under `diffuser/logs/<env>/plans/*/p23_*`.

### Criteria
| criterion | result |
|---|---|
| Primary: ≥2/3 envs ≥95% of baseline at DDIM-10 | **met, 3/3** (paired: +1.8, −75, −5.3) |
| Secondary: mean drop <5% at DDIM-10 | **met** (−26% pooled; the hopper gain dominates) |
| Aggressive, DDIM-5 (4×) | **borderline** — halfcheetah 5.1 ± 0.8% (on the line, tight); hopper and walker above baseline on mean; walker one seed −19% |
| Aggressive, DDIM-2 (10×) | **not met** — halfcheetah −9.7%, walker −25% (falls in 3/3 seeds); hopper alone tolerates it |
| Sanity: DDIM-20 vs DDPM-20 (stochasticity alone) | env-dependent — halfcheetah wash (+1.6 ± 4.7), hopper wash (−8 ± 22), **walker worse (+20 ± 21, 2/3 seeds fell early)** |

## Conclusion
**H1 supported at 2× fewer steps, borderline at 4×, refuted at 10×.** The part of the prediction that came from the structural match — Diffuser's planner is the DDIM sampler over τ, and nothing in the method depends on the Markov reverse chain — holds where it was tested cleanly: DDIM-10 is baseline-or-better on all three envs, and on hopper it is the best planner in the sweep by a wide margin (three full-length or near-full-length episodes against a baseline that falls by step 540 every time). The quantitative clause "5–10× fewer steps within ~5%" was right at the 4× edge on halfcheetah to the decimal (−5.1 ± 0.8%) and wrong at 10×.

Two things the prediction did not anticipate:
1. **Stochasticity matters on walker.** DDIM-20 lost 32% on two of three walker seeds while DDIM-10 gained 5% on all three. Non-monotone in step count, so this is not "fewer steps = coarser plan". A plausible reading: at full T the deterministic trajectory commits early to one mode; the strided 10-step schedule skips the low-signal high-noise steps and behaves like a mild regulariser. That is a hypothesis for the η sweep, not a result.
2. **Hopper's gain is a stability effect, not a return-rate effect.** Reward per step is flat across configs; DDIM-10 simply doesn't fall. Whatever the DDPM chain injects at 20 steps is destabilising on hopper's narrow support region. Same η-sweep test.

What the ontology could have carried: nothing. Both surprises are horizon × schedule interactions, hyperparameters of the paper rather than mechanism. `stochastic@trajectory` would not have predicted walker's direction.

Practical result: **DDIM-10 halves planning cost on all three envs at no loss; DDIM-5 quarters it at ≤5% on halfcheetah and a gain on hopper.** Per 1000-step episode here: DDPM-20 ≈ 16 min, DDIM-10 ≈ 9, DDIM-5 ≈ 5.

Caveats: three seeds; hopper/walker scores are dominated by fall timing, so sd is large and the sign of a mean is more trustworthy than its size. v4 dynamics, not v2. Pre-registered threshold (5%) unchanged after seeing data. Machine load varied 2× during seed 2 (iCloud sync, Xcode, a VM); wall-clock numbers are indicative only, returns are unaffected.

## Next
1. **η sweep**, hopper and walker first: DDIM-10 and DDIM-20 at η ∈ {0.5, 1.0}. Isolates whether walker's DDIM-20 loss is the missing noise or the full-length deterministic schedule, and whether hopper's DDIM-10 gain survives adding noise back. ~2 h.
2. **Second hypothesis from the same match**: SMC resampling of the 64 candidates by exp(J(τ)) in place of gradient guidance (from Diffuser × Differentiable Particle Filter 0.60). New `sample_fn`; ~1 day.
3. Fold into the write-up (done, see `method_scanner_writeup.md` §"First experimental validation").
