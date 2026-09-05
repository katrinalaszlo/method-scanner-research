# Method Scanner

**Finding shared mechanisms across ML papers by ignoring what they're called.**

Kat László · August 2026 · 30 phases · 58-paper core corpus + two outcome benchmarks

---

Methods are named by lineage — *Kalman filter*, *JEPA*, *predictive coding*, *reinforcement learning* — and lineage is a poor guide to what a method actually does. Two papers from different families can run nearly the same inner loop; two under one label can share no update rule at all.

This project strips papers to their machinery and compares that instead. A local 8B model reads a paper's **method section** (not its abstract) and reduces it to typed operations of the form `operation@object` — `predict@state`, `correct@parameter`, `sample@trajectory`. Papers are then compared on those sets.

## The main finding

**The object separates mechanisms; the operation does not.** Nearly every paper in the corpus "predicts" something, so `predict` on its own carries no information. But `predict@state`, `predict@latent` and `predict@parameter` are three unrelated activities. Under that lens most lineage labels split into two mechanism families or dissolved entirely.

The pipeline recovers two real cross-field bridges (self-supervised latent prediction ↔ predictive coding as a learning rule; particle filters ↔ Kalman-style state estimation) and refuses the one most people assume holds (Kalman filtering ↔ predictive coding).

## The test

A similarity score is cheap. So one match the scanner produced — between a robotics planner (Diffuser) and an image-generation sampler (DDIM) — was turned into a pre-registered quantitative prediction: *DDIM's deterministic sampler at 5–10× fewer denoising steps keeps Diffuser's D4RL return within ~5%.*

45 episodes, three seeds, three environments, paired against the same-seed baseline. The prediction **held at 2×** on every environment, **landed on the 5% threshold to the decimal at 4×**, and **failed at 10×**. One thing the scanner did not predict showed up in the data — on walker2d the deterministic sampler is worse at full step count — and is reported as an open observation rather than folded into the result.

The claim is not the speedup. It is that a pipeline reading text produced a falsifiable engineering prediction whose bounds the experiment confirmed.

## The follow-up, and where it fails

`outcome_predictor/` asks the harder question: does the extracted structure predict *how well a method performs*, beyond experimental conditions and lineage labels? Mostly **no**, across four pre-registered phases on D4RL, Atari 100k, within-method paired changes, and 25 Papers-with-Code leaderboards. Those negative results are written up at the same length as the positive one.

## Pipeline

Four layers. Each is derived from the one below and can be revised without re-running it; raw model output is never overwritten.

| layer | what |
|---|---|
| **Extract** | arXiv ID → PDF → `pdftotext` → heading walker finds the method section → `qwen3:8b` via Ollama emits `{operations, loss, failure}` |
| **Canonicalize** | raw verbs → 23-item controlled vocabulary, by exact match then head verb. Derived at read time, never stored |
| **Type** | second LLM pass assigns each operation an `object` from a fixed ontology, with a supporting quote |
| **Compare** | similarity over the resulting tuple sets, per paper and per family |

## Where everything is

| path | what |
|---|---|
| `writeup_final.md` | **the full writeup** — problem, pipeline, corpus, findings, the Diffuser/DDIM validation, limitations |
| `findings.md` | running record of every result and rejected approach, phase by phase |
| `progress.md` | session log — what was run, what broke, what was decided |
| `task_plan.md` | phase plan |
| `app.py` | the pipeline server + UI (`:8000`) |
| `fulltext.py` | arXiv → PDF → method-section extraction |
| `semantic_type.py` | the typing pass |
| `compare.py` | similarity matrices |
| `results/`, `matrices/` | per-paper records and computed matrices |
| `phase23/` | the Diffuser/DDIM experiment — patch, sweep scripts, raw run logs, paired results |
| `outcome_predictor/` | phases 24–30, the outcome-prediction follow-up (see its own README) |

## Running it

The core pipeline is Python standard library only. It needs `pdftotext` (poppler) and [Ollama](https://ollama.com) with `qwen3:8b` pulled.

```
python3 app.py        # UI on http://localhost:8000
python3 fulltext.py   # batch method-section extraction from arXiv IDs
```

`outcome_predictor/` additionally needs `numpy`, `scipy`, `scikit-learn` and `pandas`; `phase23/` reproduces the D4RL experiment against an unmodified Diffuser checkout and its own environment (see `phase23/phase23_plan.md`).

## Status

Research log, not a maintained tool. It is public so the method, the one positive validation, and the several negative results can be read and checked.
