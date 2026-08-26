# Phase 23 — Predictive transfer test: DDIM sampler in Diffuser

## Hypothesis (from the Phase-final structural match, Diffuser × DDIM 1.00 lexical / 0.50 relaxed)
H1: Diffuser's planning performance is insensitive to the stochasticity of the reverse chain. Replacing the DDPM sampler with DDIM (σ=0, deterministic) at 2–10× fewer denoising steps keeps D4RL normalized return within 5% of the DDPM baseline.
H0: performance depends on stochasticity and/or step count; DDIM or fewer steps degrade return by >5%.

## Setup (what actually runs here — deviations from the reviewer's draft noted)
| component | detail |
|---|---|
| codebase | jannerm/diffuser @ 7ea4228, pretrained locomotion checkpoints (`logs/pretrained`, epoch 800k diffusion / 160k value) |
| tasks | halfcheetah-medium-v2 (H=4), hopper-medium-v2 (H=32), walker2d-medium-v2 (H=32) |
| baseline sampler | DDPM, **T=20** — the released checkpoints were trained with 20 diffusion steps, not 100 |
| replacement | DDIM σ=0, same trained ε-model, strided timestep subset; value guidance identical (scale, t_stopgrad, n_guide_steps) |
| step sweep | 20 (DDIM at full T — isolates stochastic vs deterministic), 10, 5, 2 |
| batch | 64 candidate plans per env step (config default), first action executed |
| seeds | 0, 1, 2 (reviewer asked for 10 samples; 3 seeds × 5 configs × 3 envs ≈ 4.5 h on this machine) |
| metric | D4RL normalized score from `rollout.json`; raw returns reported too |
| hardware | Apple M4, torch MPS; ~0.5 s per env step at 20 diffusion steps |
| env deviation | dynamics on gymnasium `*-v4` (mujoco 3.x bindings) instead of `*-v2` (mujoco-py 2.0, x86-only). Datasets are the original D4RL v2 HDF5 files, so normalizers match training. Baseline DDPM-20 is re-run under the same deviation, so the comparison is internally consistent; absolute scores may differ slightly from the paper (44.2 / 58.5 / 79.2). |
| rendering | disabled (no mujoco-py); does not affect returns |

## Success criteria
Primary: ≥2 of 3 envs keep ≥95% of DDPM-20 mean score at DDIM-10 (the "5–10× fewer" claim at 2× fewer is the conservative read; DDIM-5 and DDIM-2 test the aggressive end).
Secondary: mean drop across envs <5% at DDIM-10.
Sanity: DDIM-20 vs DDPM-20 isolates stochasticity alone.

## Procedure
`sweep.sh` — seeds outer, envs middle, configs inner; appends one row per run to `results.csv`; per-run logs `runs_*.log`. `analyze.py` — table, % drop, plot, verdict.
