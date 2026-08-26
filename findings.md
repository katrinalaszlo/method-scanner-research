# Findings & Decisions

## Repo State (2026-08-24 end of day)
- `app.py` — server + extractor. qwen3:8b via native Ollama `/api/chat`, `think:false`, temp 0, `num_predict` 300
- `index.html` — tabs Extract | History. Drop zone (.csv/.txt), template download, staged batch, re-run + delete per record, CSV export
- `extract_llama.py` — original standalone script, still llama3.2:3b + openai client. Now stale relative to app.py
- `results/` — 10 records, all from final prompt + qwen3:8b. Every record has abstract, prompts, raw_response
- `results.json` — legacy backup, not read by anything
- `venv/` — Python 3.14, `openai` installed but no longer imported by app.py
- Not in git. `GitHub/` is gitignored in the wiki repo

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/` | UI |
| GET | `/api/results` | history, newest first |
| GET | `/api/results.csv` | export |
| GET | `/api/template.csv` | `paper,abstract` starter |
| GET | `/api/prompt` | current model + prompts |
| POST | `/api/extract` | `{paper, abstract}` |
| POST | `/api/batch` | CSV body, needs `abstract` column |
| POST | `/api/rerun` | `{file}` — re-extract from stored abstract, replace record |
| DELETE | `/api/results/{file}` | URL-encoded filename |

## Record Shape
```json
{"paper", "timestamp", "model", "elapsed_seconds", "abstract",
 "system_prompt", "user_prompt", "raw_response",
 "data": {"name", "operations": [], "loss", "failure"}}
```
Phase 5 adds `canonical_ops: []` alongside `data`, never replacing `operations`.

## Bake-off (10 papers, same prompt)
| model | ok | sec/paper | verdict |
|---|---|---|---|
| llama3.2:3b | 10/10 | ~2.5 | nouns as ops, dupes, failure leaks into ops, echoes example list on hard inputs |
| qwen3:4b | 0/10 | ~19 | ignores `think:false`, narrates |
| qwen3:8b | 10/10 | ~5.5 | clean verbs, reads the abstract, conservative on `failure` until 3rd example added |

## Current extraction (qwen3:8b, 3-example prompt)
| paper | ops | loss | failure |
|---|---|---|---|
| Kalman Filter (1960) | update, compute gains | mean square error | — |
| PKF | associate, update | — | — |
| V-JEPA | encode, predict | accuracy loss | collapse |
| C-JEPA | encode, intervene, forecast | contrastive loss | — |
| DALI | infer, predict, condition | — | requires explicit context variables |
| ResDreamer | reconstruct | — | multi-step error accumulation |
| Predictive Coding Light | suppress, transmit | — | — |
| Information Spreading | denoise, describe | score matching | — |
| TriSHI | compute normalized common neighbors, compute avg clustering coeff, compute community co-membership | — | — |
| LeNet-5 | train, synthesize, recognize | overall performance measure | — |

Grounding checks done: C-JEPA null failure is correct (abstract states none; llama's "non-causal correlations" was invented). "accuracy loss" / "overall performance measure" are quoted from abstracts that describe but don't name the objective. LeNet-5 abstract is mostly about Graph Transformer Networks — qwen extracted that faithfully.

## The finding that drives Phase 5
Five papers share a predict-then-correct loop — Kalman, PKF, V-JEPA, DALI, Predictive Coding Light (suppress = subtract the prediction, transmit = the error). But the emitted verbs differ: predict / forecast / infer; update / suppress / associate. String-level Jaccard scores Kalman-vs-V-JEPA at **zero**. The structure lives one level above the verbs. Hence canonicalize first, compare second, scale third.

## Recall caveat
1-3 ops per paper from abstracts. Abstracts summarize contribution, not mechanism. Method sections would multiply recall — Phase 8, only if the family claim holds at 40 papers.

## Phase 5 result (2026-08-24)
| paper | canonical | unmapped |
|---|---|---|
| Kalman Filter | correct | compute gains |
| PKF | associate, correct | — |
| V-JEPA | encode, predict | — |
| C-JEPA | encode, intervene, predict | — |
| DALI | predict, intervene | — |
| Predictive Coding Light | suppress, transmit | — |
| ResDreamer | reconstruct | — |
| Information Spreading | reconstruct | describe |
| TriSHI | — | 3× compute … |
| LeNet-5 | train, classify | synthesize |

**Criterion failed.** The predict-then-correct family does not appear: Kalman shares nothing with V-JEPA/DALI. Two causes, both real:
1. **Recall.** The 1960 Kalman abstract never says "predict" — it talks about the Wiener problem and gain computation. The predict step is in the paper body. Abstracts don't carry mechanism.
2. **Deliberate split.** `suppress`/`associate` kept apart from `correct`, so PCL and PKF can't match Kalman on the correction step by construction.

What did work: controls separate cleanly (TriSHI fully unmapped, LeNet on train/classify), and the JEPA/world-model side clusters (V-JEPA, C-JEPA, DALI all on `predict`, two on `encode`, two on `intervene`).

## Phase 8 narrow test: Kalman 1960 method section (2026-08-24)
Source: Kalman1960.pdf (unitedthc.com mirror) → `pdftotext` → lines 860-1100, the derivation leading into Theorem 3. 1338 words, math garbled but prose intact. Saved as `kalman_1960_method_section.txt`. Dry run, not in history.

| input | ops | canonical | loss | failure |
|---|---|---|---|---|
| abstract | update, compute gains | correct | mean square error | — |
| method section | estimate, update covariance | **predict** | expected quadratic loss | requires positive definite covariance matrices and linearly independent observations |

**Cause #1 confirmed.** `predict` appears from the body, not the abstract. Kalman ∩ V-JEPA is now {predict}. Loss and failure also got more specific. Cost: 38s vs 5s.

**New mechanism finding:** `update covariance` is unmapped by exact-match, but its head verb `update` → `correct`. Same for `compute gains`. A head-verb fallback (match first word when whole string misses) would map both — flagged as head-match, not exact, so it stays auditable.

## Phase 7 result: 30 papers, full-text method sections (2026-08-24)
Section finder hit 29/30 by heading (DreamerV3 fell back). 0 extraction failures. ~40s/paper.

**Canonical ops per family**
| family | ops |
|---|---|
| filters | correct 3, predict 2, reconstruct 1, encode 1 |
| jepa | encode 5, predict 4, compare 1 |
| predcoding | predict 4, correct 4, encode 1, transmit 1 |
| worldmodels | predict 4, train 3, encode 2, reconstruct 1 |
| diffusion | predict 3, reconstruct 1, intervene 1 |
| control | correct 1, classify 1, encode 1, reconstruct 1 — no two controls share anything |

**Mean Jaccard on canonical ops**
| within | | between (top) | |
|---|---|---|---|
| jepa | 0.73 | jepa × worldmodels | 0.40 |
| predcoding | 0.46 | **filters × predcoding** | **0.40** |
| worldmodels | 0.34 | jepa × predcoding | 0.32 |
| filters | 0.23 | predcoding × worldmodels | 0.23 |
| diffusion | 0.20 | | |
| control | 0.00 | | |

**predict + correct both present:** PredNet, PC Networks & Inference Learning, PC Beyond Backprop, KalmanNet. Three predictive-coding papers and one Kalman paper — that's the cross-domain bridge, and it's now two verbs wide, not one.

**Why filters within-family is low (0.23):** the raw ops are right but multi-word with a generic head — `compute filter state`, `compute data association weights`, `resample particles`. `compute` isn't in the vocab and shouldn't be; the object carries the meaning. Vocab needs object-aware rules or a second canonical layer.

**Contamination found:** `failure` = "…collapse without an asymmetric design" on BYOL, V-JEPA, **and World Models (Ha & Schmidhuber)**. That phrase is my few-shot example. BYOL/V-JEPA plausibly true; World Models is not about that at all → qwen echoes examples too when the paper is thin on failure modes. Fix: rewrite the third example so its failure phrase is unmistakably synthetic, or drop the phrase and describe the failure differently.

**Unmapped raw ops worth adding:** project (3), sample (2), augment, diffuse / reverse diffuse / add noise (diffusion's own loop), plan, execute (world-models control step), compute error (→ compare?), stop-gradient.

## Phase 9 result: re-run of the 30 with echo-proof example + vocab round 2 (2026-08-24)
Comparator: `compare.py` → `similarity_matrix.csv` (30×30). Min 2 shared ops for top pairs.

| within | | between (top) | |
|---|---|---|---|
| jepa | 0.46 | filters × predcoding | 0.21 |
| predcoding | 0.30 | filters × worldmodels | 0.20 |
| worldmodels | 0.20 | jepa × worldmodels | 0.20 |
| diffusion | 0.20 | predcoding × worldmodels | 0.18 |
| filters | 0.08 | | |
| control | 0.03 | | |

All numbers lower than Phase 7 — vocab round 2 added 7 ops, so sets are finer and overlap rarer. Expected.

**Top cross-domain pairs (≥2 shared ops)**
1. I-JEPA × World Models — 1.00 — [encode, predict, correct]
2. KalmanNet × {PredNet, PC Networks & Inference Learning, PC Beyond Backprop} — 0.67 each — [predict, correct]. Also KalmanNet × PlaNet, × World Models on the same two
3. Deep Kalman Filters × DreamerV3 × DDIM — 1.00 / 0.67 / 0.67 — [predict, reconstruct]. A filters/worldmodels/diffusion triangle on the generative loop

**Echo check:** "asymmetric design" gone from all 30. World Models failure is now "credit assignment problem" — correct. But Backprop KF's failure is "struggles at very low signal-to-noise ratios" — fragment of the *new* example. Its stored section text does not contain "signal-to-noise" → confirmed echo. The model reaches for the example when the paper is thin on failure modes; changing the phrase only changes what gets echoed. Mitigation options: (a) failure example with `null` too, (b) post-hoc grep against example strings and null them out with a flag, (c) two-pass — extract failure separately with no examples.

**Predictive Coding Light+** now fully unmapped: `excite, inhibit, maintain`. The neuroscience vocabulary doesn't overlap the ML vocabulary at all. Either add `inhibit → suppress`, or this is the actual finding — PCL's mechanism isn't described at the level where the others are.

**Filters within 0.08:** each filter paper describes a different piece (KalmanNet the loop, PKF the association, PF-Net the sampling, Backprop KF the training). Family label is by lineage, not by mechanism described. The corpus design assumed family = shared loop; for filters that's false at the method-section level.

## A/B: current prompt vs DeepSeek closed-vocab prompt (2026-08-24, 6 problem papers, qwen3:8b, dry)
| paper | current ops / failure | DeepSeek ops / failure |
|---|---|---|
| Backprop KF | compute, filter, train / echo (SNR) | estimate, predict, update, transform, combine, calculate, train / null |
| DreamerV3 | transform, predict, reconstruct / real | +encode, generate, sample, combine, calculate, **augment, add_noise**, correct (11) / null |
| PCL+ | excite, inhibit, maintain / real | encode, predict, **inhibit**, update, condition, … (10) / null |
| PKF | compute data association weights, update state / null | estimate, predict, update, associate, condition, calculate / null |
| BYOL | encode, project, predict, normalize / real | +**contrast**, update, calculate, augment, transform, compare (10) / null |
| Adam | 3 specific ops / real | update, estimate, calculate, correct / null |

Closed vocab in the prompt → the model emits the vocab list, not the paper: DreamerV3 gets `augment, add_noise` (not in its text), BYOL gets `contrast` (BYOL is explicitly non-contrastive), PCL+ still says `inhibit` despite the in-prompt rule `inhibit → suppress`. Failure went null on all 6, including 4 with documented failure modes the current prompt found. The DeepSeek prompt fixed the echo by suppressing the field entirely. **Rejected.** Keep: two null-failure examples as the Phase 10 fix.

## Phase 10 result: 30 re-run with null-failure example + strict rule (2026-08-24)
| within | | between (top) | |
|---|---|---|---|
| predcoding | 0.55 | filters × predcoding | 0.34 |
| jepa | 0.45 | jepa × predcoding | 0.29 |
| diffusion | 0.30 | filters × jepa | 0.22 |
| worldmodels | 0.22 | predcoding × worldmodels | 0.21 |
| filters | 0.15 | | |
| control | 0.00 | | |

Up from Phase 9 across the board (predcoding 0.30 → 0.55, filters 0.08 → 0.15). More literal extraction → more shared verbs. Filters × predcoding is the top cross-family pair for the third run running.

**Top pairs:** DeepKF × DreamerV3 [predict, reconstruct] 1.00; KalmanNet × PC Review [correct, predict] 1.00; I-JEPA × World Models [correct, encode, predict] 1.00; BYOL × World Models 0.75; KalmanNet × {PredNet, PCN&IL, I-JEPA} 0.67; Backprop KF × PC Beyond Backprop [predict, transmit] 0.67.

**Failure field:** 4/30 null (was 3). One echo survived — EDM got "musical noise…" — the model still reaches for the example on one paper in 30. Prose-null "no failure mentioned" on DDIM. Both now caught by a read-time guard (`guard_failure` in app.py): nulls the field, records `failure_flag`, disk untouched. Echo rate: 1/30 with guard as backstop — acceptable; two-pass extraction for `failure` stays available if it climbs.

**Stable across three runs (Phase 7, 9, 10):** filters × predcoding top or second cross-family pair; jepa the tightest family; controls near zero; KalmanNet the hub paper (predict+correct) bridging filters to PC, JEPA, and world models.

## H1 ablation: is KalmanNet a hub only because of `predict`? (2026-08-24)
`compare.py --drop predict`. KalmanNet's canonical set is exactly {predict, correct}; without predict it is {correct}.

| | with predict | without predict |
|---|---|---|
| filters × predcoding (mean) | 0.34 | 0.22 — still #1 cross-family |
| filters × jepa | 0.22 | 0.13 |
| KalmanNet in top pairs (≥2 shared) | 5 pairs at 0.67–1.00 | **none** — one op left, can't share two |
| KalmanNet in top pairs (≥1 shared) | — | ties on `correct` with Adam, PC Review, 4 more PC, World Models, I-JEPA, BYOL |

**Verdict: H1 supported.** The KalmanNet *bridge* — the two-op predict+correct match — is entirely dependent on `predict`. What survives is a one-op `correct` link, and that link is not discriminating: it ties KalmanNet to Adam (an optimizer) at 1.00 as strongly as to any predictive-coding paper. A hub that connects equally to a control is not a hub.

**But the family-level signal survives:** filters × predcoding stays the top cross-family pair without predict (0.22), carried by `correct` across PKF, KalmanNet, PC Review, PCN&IL. And the only ≥2-op cross-family pairs left are I-JEPA/BYOL × World Models on `correct, encode` — a bridge that does not depend on predict.

**Reading:** `predict` is the glue for the KalmanNet story; `correct` is the glue for the filters↔PC family story; `encode` + `correct` is the JEPA↔world-models story. Three different glues, not one. The ontology question (Level 4→5: what does "the same op" mean in different fields) starts here — KalmanNet's `correct` is a Kalman gain step, PredNet's is an error-driven state update, Adam's is a parameter step. Same verb, three mechanisms.

## Phase 11 result: semantic typology (2026-08-24)
Typing pass: 30/30, 106 (op, object_type) tuples, 86 with a quote found in the section text.

| within | lexical | strict | relaxed |
|---|---|---|---|
| jepa | 0.45 | 0.12 | 0.24 |
| predcoding | 0.55 | 0.06 | 0.09 |
| diffusion | 0.30 | 0.06 | 0.12 |
| filters | 0.15 | 0.03 | 0.07 |
| control | 0.00 | 0.00 | 0.00 |

| between (top) | lexical | strict | relaxed |
|---|---|---|---|
| filters × predcoding | 0.34 | 0.04 | 0.06 |
| jepa × predcoding | 0.29 | **0.07** | **0.11** |
| jepa × worldmodels | 0.19 | 0.05 | 0.09 |

**Top strict pairs:** I-JEPA × PC Networks & IL 0.50 [correct@parameter, predict@latent]; V-JEPA × World Models 0.40 [encode@latent, predict@latent]; BYOL × PCN&IL 0.33. **Relaxed adds:** KalmanNet × World Models 0.40 [correct@state~latent, predict@state~latent].

**KalmanNet → predictive coding bridge under semantic typing: gone.** Strict, KalmanNet shares one tuple with one PC paper each — `predict@state` with PC Beyond Backprop, `correct@state` with PredNet — and nothing with the other three. The lexical 1.00 with PC Review dissolves: that paper's `correct` is `correct@parameter` (weight update), KalmanNet's is `correct@state`. Exactly the split H1 predicted. Semantic H1 (drop predict): no ≥2-shared cross-family pair survives at all, strict or relaxed.

**The real bridge is jepa ↔ predictive coding, on `predict@latent` + `correct@parameter`.** Top cross-family under both strict and relaxed. Reading: I-JEPA/BYOL and the PC-as-learning-rule papers (PCN&IL, PC Beyond Backprop) are the same thing — predict a latent, update weights on the error. Filters aren't in that family: they predict and correct *state*, not parameters.

**Bridge tuples across ≥3 families:** predict@state and predict@latent (both span filters/jepa/predcoding/worldmodels — `predict` is the universal verb but splits by object); correct@parameter (control/jepa/predcoding — the learning-rule family); correct@state (filters/predcoding/worldmodels — the state-estimation family); encode@input; predict@distribution.

**Typing consistency — the caveat.** `predict` was typed 6 ways (state 6, latent 6, distribution 3, parameter 2, input 2, gradient 1). Some is real (PredNet predicts frames → input; DDPM predicts noise); some is the model reaching for `state` on anything vector-shaped (BYOL project@state, ResNet identity@state). `correct` split cleanly (parameter 6 / state 4). Strict numbers are lower bounds; relaxed (state~latent) is the fairer read for the represented-object question. 20/106 quotes not found verbatim — mostly math-symbol mangling, spot-checked, not fabrication.

**Predictive coding is two families wearing one name.** PredNet/PCL+ predict *inputs* and update *states* (neuroscience-style); PCN&IL / PC Beyond Backprop / PC Review predict latents and update *parameters* (PC-as-backprop-alternative). Lexical Jaccard merged them (0.55 within); semantic splits them (0.06). The family label was by lineage, again.

## Phase 12+13 result: mechanism families (2026-08-24)
`families.json` relabels 10 papers by mechanism (from Phase 11 tuples); `compare.py --families mechanism`. Lineage labels untouched in `source`.

| within | lexical | strict | relaxed | n |
|---|---|---|---|---|
| pc-learning | **0.61** (was predcoding 0.55) | 0.16 (was 0.06) | 0.24 | 3 |
| pc-neuro | 0.33 | 0.17 | 0.17 | 2 |
| state-estimation | 0.28 (was filters 0.15) | 0.11 (was 0.03) | 0.11 | 3 |
| jepa | 0.45 | 0.12 | 0.24 | 5 |
| latent-generative / particle | — | — | — | 1 each |

**The split is real.** pc-learning within-family rose under every comparator (strict 0.06 → 0.16, ~3×). state-estimation rose too (strict 0.03 → 0.11). Splitting by mechanism made families coherent under the typed view, which is the view that distinguishes mechanism — that's the test passing.

**Top cross-family, strict:** jepa × pc-learning is #1 at family level (0.12) and pair level (I-JEPA × PCN&IL 0.50, BYOL × PCN&IL 0.33) on `correct@parameter + predict@latent`. Under lineage labels this was #1 too; the re-split didn't create it, it sharpened it. Second: jepa × latent-generative (Deep Kalman Filters) 0.11 — DKF is a VAE, and it sits with self-supervised latent prediction, not with the filters it's named after.

**Relaxed adds** KalmanNet × World Models 0.40 on `correct@state~latent + predict@state~latent`: the state-estimation ↔ world-model bridge, visible only when a Kalman state and a Dreamer latent are allowed to count as the same kind of object. That's a defensible equivalence — both are recursive estimates of a hidden variable — and it's the one bridge the strict view refuses.

**Lexical pc-learning × pc-neuro 0.56 vs strict 0.03**: the two PC families share verbs and almost nothing else. Lexical Jaccard would have called them one family forever.

**Caveats:** n=2–3 per re-split family; means at that size move ±0.1 on one paper. Two families are singletons — need 2+ more papers each before they mean anything. Assignments are from one typing pass; the pass has known `state` over-assignment.

## Phase 14 result: typing reliability, expansion candidates, rules (2026-08-24)

**Track A — pass 1 vs pass 2 typing agreement** (same model, stricter prompt, temp 0; full report in `agreement_report.txt`)
- 30/30 papers, 106 tuples. Exact agreement **72%** (76/106). Pass 2 said `unknown` only 2% of the time — the model does not use the escape hatch even when told it's normal.
- `state` is sticky: 20/26 pass-1 `state` stayed `state` under a prompt that says outright "a vector or embedding is latent, not state". BYOL `project@state` survived. **Over-assignment is not a prompting problem.** Pass 2 also *created* new `state` assignments (`sample→state` ×2, `action→state` ×2).
- Most-changed papers: DreamerV3 3/3, then a cluster at 2/3 (Deep KF, DDPM, EDM, DreamerV2, I-JEPA, word2vec). Least-stable op: `sample` (2/4); most stable: `update`, `project`, `update weights` (4/4, 4/4, 3/3); `predict` 10/14.
- Diffusion papers were re-typed `distribution → input` ×3: pass 2 read "noise added to the image" as acting on the input. Both readings are defensible — this is ontology ambiguity, not model error.

**Do the findings survive the re-typing?** Strict, mechanism families:
| | pass 1 | pass 2 | agreed-only |
|---|---|---|---|
| pc-learning within | 0.16 | 0.16 | 0.16 |
| pc-neuro within | 0.17 | 0.17 | 0.20 |
| state-estimation within | 0.11 | 0.11 | 0.11 |
| jepa within | 0.12 | 0.12 | 0.11 |
| top cross-family | jepa × pc-learning 0.12 | latent-gen × state-est 0.11, jepa × pc-learning 0.08 | jepa × pc-learning 0.10 |
| ≥2-shared cross pairs | 4 (I-JEPA × PCN&IL 0.50 …) | **0** | **0** |

Family-level structure is stable across passes — the re-split families keep their within scores to the second decimal. Pair-level headline (I-JEPA × PCN&IL on `correct@parameter + predict@latent`) does **not** survive pass 2 or agreed-only: one of its two tuples got re-typed. Family means are robust; individual pair scores are one re-typing away from zero at this n.

**Track B — expansion candidates:** `candidates_phase15.csv`, 51 papers, none in corpus. 15 shortlisted (3/family), PDFs fetched, 9 with a detected method heading, 6 fallback. See task_plan Phase 15 for the list.

**Track C:** `compatibility_rules.md`. Non-negotiables: `parameter` and `input` never relax.

## Phase 15 result: 45 papers, mechanism families 4–6 each (2026-08-24)
15 added (`papers_phase15.txt`), all extracted (PF-RNN needed a 30% window — the 20% window was pure math and qwen emitted base64 garbage), all typed twice. Matrices in `matrices/phase15_*.csv`.

**Stability, Phase 13 (n=2–3) → Phase 15 (n=4–6), strict**
| family | n | lexical | strict | Δ strict | relaxed | agreed |
|---|---|---|---|---|---|---|
| pc-learning | 3→6 | 0.61→0.50 | 0.16→**0.21** | +0.05 | 0.24 | 0.21 |
| particle | 1→4 | —→0.33 | —→0.16 | new | 0.08 | 0.14 |
| jepa | 5 | 0.45 | 0.12 | 0 | 0.24 | 0.11 |
| state-estimation | 3→6 | 0.28→0.26 | 0.11→0.08 | −0.03 | 0.08 | 0.08 |
| pc-neuro | 2→5 | 0.33→0.42 | 0.17→**0.02** | **−0.15** | 0.02 | 0.02 |
| latent-generative | 1→4 | —→0.06 | —→0.00 | new | 0.03 | 0.00 |
| control | 5 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |

Typing agreement on the 15 new papers: 84% (46/55); corpus-wide 75% (124/165).

**What held:** pc-learning is the most coherent typed family and got *more* coherent with n doubled (0.16→0.21). jepa × pc-learning stays a top-2 cross-family pair in every mode (strict 0.10, agreed 0.11). Controls at zero. The state-over-assignment and the `parameter`-never-relaxes rule both behave as documented.

**What broke:**
- **pc-neuro collapsed under typing (0.17→0.02) while rising lexically (0.33→0.42).** The three spiking/cortical papers share verbs with PredNet/PCL+ but act on different objects. pc-neuro was a real family at n=2 and a name at n=5. Lexical coherence + typed incoherence is the signature of a lineage label.
- **latent-generative is not a family** (lexical 0.06, strict 0.00 at n=4). Four VAE-dynamics papers describe four different things. Drop the label or re-split.

**New bridge, and it's the expected one: particle × state-estimation.** #1 cross-family in strict (0.11), pass-2, and agreed (0.12). Differentiable Particle Filters × KalmanNet / Neural Kalman Filtering at 0.50 on `correct@state + predict@state`; PF-RNN × KalmanNet 0.40 on the same. Two filter lineages, one loop, typed identically — a positive control the design didn't plan but got. It survives every typing variant, which the jepa × PCN&IL pair did not.

**A false bridge from the state bug:** SimSiam × Neural Kalman Filtering 0.33–0.40 on `predict@state + project@state`. SimSiam's projector output is not a state. This pair is in the top 6 under strict, pass 2, and agreed — the over-assignment is consistent enough to survive the agreement filter. Agreement is not correctness.

**Phase 16 recommendation:** (1) dissolve latent-generative, re-split pc-neuro by typed tuples before adding more; (2) fix `state` at the ontology, not the prompt — require a temporal-dynamics quote for `state`, else `latent`, enforced in code post-typing; (3) then a second batch from `candidates_phase15.csv` toward n=8 for pc-learning, state-estimation, particle only — the families that cohere.

## Phase 16a: code-level `state` guard (2026-08-24)
`guard_state()` in app.py, read time, both passes, disk untouched: a `state` tuple keeps `state` only if its quote (±200 chars of section text around it) contains a temporal-dynamics marker (t−1, x_t, time step, trajectory, recurrent, filter, transition, belief, …); otherwise demoted to `latent` with `demoted_from: state`.

Demoted, pass 1: 7 of 51 `state` tuples — SimSiam `project`, SimSiam `predict`, ResNet `identity mapping`, I-JEPA `encode` (the four known-wrong ones), plus DPF `decode`, Adaptive UKF `compute Kalman gain`, Spiking PC `predict`. Pass 2: 6. Kept: 44.

| | before guard | after guard |
|---|---|---|
| jepa within, strict | 0.12 | **0.21** |
| SimSiam × Neural KF (false bridge) | 0.33–0.40, top 6 in every mode | **gone** |
| DPF × KalmanNet / Neural KF (true bridge) | 0.50 | 0.50 |
| PF-RNN × KalmanNet | 0.40 | 0.40 |
| I-JEPA × PCN&IL | 0.50 | 0.50 |
| particle × state-estimation (family, agreed) | 0.12 | 0.12 |
| jepa × pc-learning (family, strict) | 0.10 | 0.09 |

The guard removed exactly the false pair and touched none of the true ones. jepa coherence nearly doubled because its papers now agree they act on latents. One over-reach: Adaptive UKF's `compute Kalman gain@state` → `latent` — should be `parameter`; the guard can only demote to `latent`. Harmless here (it matched nothing either way).

## Live prediction test #1 (2026-08-24) — "Bidirectional Predictive Coding for Multi-Step Action Anticipation"
Abstract supplied by DeepSeek (provenance unknown — may be synthetic). Abstract only, not a method section. Stored with `source: test:` so it stays out of the corpus; `compare.py --query` added.

Extracted: `encode, forecast, refine, train` → canonical `encode, predict, correct, train`; loss contrastive; failure quoted verbatim. Typed (both passes identical): `encode@latent, forecast@state, refine@latent, train@parameter`.

| mode | closest paper | top family | pc-learning rank |
|---|---|---|---|
| lexical | I-JEPA = PC Exact Backprop = World Models (0.75) | jepa 0.47 | 2nd (0.42) |
| strict | PlaNet 0.33 [predict@state, train@parameter] | particle/worldmodels 0.10 | 5th (0.05) |
| relaxed | World Models 0.60 [correct, encode, predict @represented_state] | jepa 0.21 | 7th (0.08) |

**DeepSeek's prediction: family pc-learning, closest I-JEPA/PCN&IL on predict@latent + correct@parameter.** Family: wrong under typing — the abstract has `refine@latent` (a latent gets corrected) and `train@parameter`, not `correct@parameter`. "Predictive coding" in the title is lineage; the mechanism is encode → predict → refine a latent, i.e. jepa / world-model. That is precisely the lineage-vs-mechanism trap the project documented, and the prediction fell into it. Closest paper: I-JEPA is right in lexical and relaxed, but on `encode@latent + predict@latent` — not the tuple predicted. Qualitative hypothesis ("replace the backward predictor with I-JEPA's correction mechanism"): I-JEPA has no latent-correction step; its `correct` is an EMA weight update (`correct@parameter`). The backward predictor is a *smoother* — refine the latent using future context — whose structural relatives are PlaNet's posterior and the Kalman/particle correction step (World Models 0.60, KalmanNet 0.40, PF-RNN 0.33 under relaxed). Better hypothesis from the data: treat the backward pass as a learned smoother and borrow from state-estimation (e.g. two-filter smoothing), not from JEPA.

Caveat: `forecast@state` was kept by the guard because "temporal consistency" sits within 200 chars; "next action state" is arguably a latent. Under relaxed it makes no difference.

## Phase 16 result: families fixed, Sutton-lineage RL added (2026-08-24)
50 papers. `families.json` `_moves_phase16` logs the 9 reassignments with reasons. 5 RL papers (DQN preprint, PPO, TRPO, SAC, A3C) extracted and typed twice; provisional family `policy-learning`.

**Within-family after Track A + B (n)**
| family | lexical | strict | relaxed | agreed |
|---|---|---|---|---|
| jepa (5) | 0.45 | **0.21** | 0.24 | 0.11 |
| pc-learning (8) | 0.41 | 0.16 | 0.19 | 0.15 |
| particle (5) | 0.30 | 0.15 | 0.15 | 0.14 |
| state-estimation (8) | 0.20 | 0.07 | 0.07 | 0.07 |
| pc-neuro (3) | 0.56 | 0.06 | 0.06 | 0.07 |
| worldmodels (6) | 0.25 | 0.04 | 0.09 | 0.00 |
| **policy-learning (5)** | **0.13** | **0.02** | 0.02 | 0.02 |
| control (5) | 0.00 | 0.00 | 0.00 | 0.00 |

**Verdict: `policy-learning` does not emerge as a family in this ontology.** Within-family strict 0.02 — indistinguishable from controls. 8 of 10 RL pairs share zero tuples; the two that share anything share `train@parameter` or `correct@parameter`. It doesn't collapse into another family either: best cross link is pc-learning at 0.05, entirely on `correct@parameter` (14 hits) — the generic "weights get updated" tuple — plus `sample@sample` with particle (DQN's replay sampling) and `act@action` with worldmodels (Dreamer's actor).

**Why, mechanically:** the RL papers' defining ops are unmapped — `compute advantage estimates`, `construct surrogate loss`, `solve constrained optimization`, `improve` (policy), `evaluate` (policy), `accumulate updates`, `interact`. The vocabulary has no `evaluate`/`improve` canonical op, and the ontology has no object for value/advantage/return/policy: `compute advantage` was typed `@state` (PPO) and `@latent` (TRPO); SAC's policy was `@distribution`; `reward` exists as a type and was never assigned. Sutton-lineage RL is a real mechanism family (policy evaluation → policy improvement, both on a value/policy object); the ontology can't see it because it was built from filters, SSL, PC, world models, and diffusion, none of which have a value function. **This is an ontology gap, not a negative result about RL.**

**Stable across Phases 11–16:** jepa × pc-learning top cross-family strict (0.10); particle × state-estimation second and the only bridge that survives agreed-only at pair level (DPF × KalmanNet / Neural KF 0.50); controls zero.

## Phase 17: predictions on 3 unseen papers (2026-08-24)
Pre-registered (DeepSeek): DVBF → state-estimation; Snoek BO → no match; MAML → pc-learning or new. Papers extracted as `test:` records (out of corpus), typed, scored with `compare.py --query`. Full output `predictions/phase17_results.txt`, summary `predictions/phase17_predictions.csv`.

| paper | strict top family | top paper | shared | verdict |
|---|---|---|---|---|
| DVBF | pc-neuro 0.10 | PredNet 0.17 | `predict@input` | **wrong** — nothing above 0.10 |
| Snoek BO | control 0.04 | word2vec 0.20 | `predict@distribution` | **right** — no match |
| MAML | pc-learning 0.17 | Constrained PC 0.40 | `correct@parameter, train@parameter` | right, for a generic reason |

**Scorecard 1.5 / 3.** The negative control worked cleanly, and it worked *because of typing*: lexically Snoek BO scores 0.67 with PC Exact Backprop and PlaNet on `predict, train` — lexical Jaccard would have filed Bayesian optimization under predictive coding. Strict says no family above 0.04. That's the ontology doing its job.

**DVBF failed for a structural reason.** Typed `infer@parameter, compute transition@state, predict@input` — a recognition model, a transition, a decoder. No `correct@state`. It is not a filter in the Kalman sense (no measurement-update step); "Bayes Filters" in the title is lineage. Its real kin is Deep Kalman Filters — the latent-generative family we dissolved in Phase 16 because n=4 didn't cohere. DVBF says that dissolution may have been premature: the family exists, our four examples were just a bad sample, or the signature is `infer@latent + reconstruct@input + transition@state` and the typing pass never lands all three. Either way the pre-registered prediction (state-estimation) was a lineage guess and the system correctly refused it.

**MAML matched pc-learning on the two most generic tuples in the corpus.** Its defining structure — sample tasks, evaluate adapted loss, outer update — is invisible: `sample@other, evaluate@other`. Same gap as RL: no `task`/`objective` object, no `evaluate` op. The match is real but says only "this is a gradient-update method", which is true of half the corpus.

**Hypotheses (Track B), graded by how much of the match is specific:**
1. MAML ↔ Constrained PC on `correct@parameter + train@parameter`: *"A predictive-coding local weight update could replace MAML's inner-loop gradient step without loss, assuming adaptation needs only a few first-order steps."* Testable; weakly grounded — the shared tuples don't include the inner/outer structure that makes MAML MAML.
2. DVBF ↔ PredNet on `predict@input`: *"DVBF's decoder could be replaced by a hierarchical PredNet-style predictor."* One tuple; not worth running.
3. Snoek BO: none. Correct behaviour.

**What this says about the ontology:** two of three unseen papers were only partly visible to it — the missing objects are `task`/`objective`/`value` and the missing ops are `evaluate`/`improve`, the same gap Phase 16 found for RL. That's now the single most valuable fix.

**Pipeline:** MAML pass 2 emitted two JSON lists; parser now takes the first, which was a 1-item stub — pass 2 for MAML is unusable, pass 1 is what predictions use. `--query` now honours `--min-shared` (single-op 1.00 ties on `predict` were noise). `--test` caps sections at 1000 words.

## Phase 18 result: ontology v2, whole corpus re-typed (2026-08-25)
54 corpus papers (+2 test). v2 typing in `semantic_ops_v2` on every record; v1 intact. 26/54 papers had ≥1 tuple change (`predictions/phase18_v2_diff.txt`). Matrices `matrices/phase18_*.csv`.

| family (n) | lexical | strict v1 | **strict v2** | relaxed v2 | target | verdict |
|---|---|---|---|---|---|---|
| meta-learning (3) | 0.67 | 0.14 | **0.34** | 0.34 | >0.15, distinct | **met** — top family; meta×pc-learning 0.14 < within 0.34 |
| policy-learning (5) | 0.37 | 0.09 | **0.14** | 0.14 | >0.15 | **just missed** — see below |
| latent-generative (5) | 0.10 | 0.00 | **0.00** | 0.02 | >0.15 | **failed** — dissolution stands |
| jepa (5) | 0.45 | 0.21 | 0.21 | 0.24 | — | stable |
| pc-learning (8) | 0.40 | 0.16 | 0.15 | 0.17 | — | stable |
| particle (4) | 0.33 | 0.16 | 0.11 | 0.11 | — | down (DDPM-style `sample→state` re-typings) |
| control (5) | 0.00 | 0.00 | 0.00 | 0.00 | — | stable |

**policy-learning: the ontology now sees the Sutton loop, and the lineage is two mechanisms.** PPO, TRPO, SAC all typed `evaluate@value + improve@parameter` (SAC also `improve@policy`) — pairwise 0.40–0.50, sub-family mean ≈0.43. DQN (`store@sample, sample@sample, select@action, act@action, output@value`) and A3C (`accumulate updates@gradient, correct@parameter`) share nothing with them. Value-based / async-gradient vs policy-gradient. Family mean 0.14 is three coherent papers diluted by two outliers — the pc-neuro pattern again. Split: `policy-gradient` (PPO, TRPO, SAC) coheres above target; `value-based` (DQN) and A3C need company before they're families.

**meta-learning: real and distinct.** All three on `sample@task + correct@parameter`; MAML adds `improve@parameter`, Meta-SGD/Reptile add `compute gradient@gradient`. Nearest family is pc-learning at 0.14 on `correct@parameter` — adjacent (both are gradient-update methods), not collapsed. `task` was the missing object; with it, the family appears.

**latent-generative: not a family at this granularity.** DVBF `transition@state + predict@input`; DKF `reconstruct@input + predict@latent`; L-VSSF `forward/backward pass@state`; Variational Tracking `predict@state ×2 + transmit@state`; BBVI `parameterize@distribution + sample@sample`. Five papers, five signatures. The `infer@latent + transition@state + reconstruct@input` hypothesis never lands all three in any one paper. Dissolution stands; DVBF, DKF and Variational Tracking are each nearer to worldmodels or state-estimation than to each other.

**New cross-domain analogy (success criterion 4, met):** meta-learning × pc-learning is now the **top strict between-family pair (0.14)**, ahead of jepa × pc-learning (0.10) for the first time. Also meta-learning × policy-learning 0.08 (MAML `improve@parameter` ↔ PPO/TRPO). Both invisible before `task`/`value` existed.

**v2 side effects to watch:** Adaptive UKF's four covariance ops moved `state→error` (correct); PPO `evaluate@state→value` (correct); DreamerV3 `predict@distribution→reward` (correct). But `Dream to Control act@action→act@state` and `DDPM sample@sample→state` are regressions — the typing model still reaches for `state`, now from new directions. Particle dropped 0.16→0.11 for this reason.

## Canonical vocabulary (live, in `app.py`)
| canonical | absorbs |
|---|---|
| encode | embed, map, represent, extract features |
| predict | forecast, infer, estimate, imagine |
| correct | update, suppress, refine, adjust |
| compare | contrast, match, associate, score |
| reconstruct | decode, denoise, generate |
| mask | drop, occlude |
| intervene | condition, perturb |
| aggregate | combine, compute (multi-feature) |
| classify | recognize, label |
| train | learn, optimize, synthesize |
| transmit | propagate, pass |
Unmapped verbs stay raw and get flagged, not forced.

## DeepSeek thread (external, untrusted)
Kat is running a parallel QA conversation with DeepSeek. Its diagnosis on 2026-08-24 was wrong on two points: claimed the ban-list wasn't in the prompt (it was — rules just don't work on 3b), and rated Information Spreading's 8-verb echo as "very rich ✅" (it was the example list copied verbatim). Its "run the comparator" suggestion is right in spirit but premature before canonicalization.

## Phase 19 result: policy-learning split, latent-generative retired (2026-08-25)
58 corpus papers. Added DDPG 1509.02971, TD3 1802.09477 (policy-gradient), Double DQN 1509.06461, Rainbow 1710.02298 (value-based); v2 + v1 typed. `families.json`: policy-learning → `policy-gradient` (PPO, TRPO, SAC, DDPG, TD3) + `value-based` (DQN, A3C-flagged, DDQN, Rainbow); latent-generative removed, `_moves_phase19`. State guard v2 in `guard_state()`: `act/select/execute→action`, `sample/resample/store→sample`, never `state`. Matrices `matrices/phase19_*.csv`. Full report `predictions/phase19_results.txt`.

| family (n) | lexical | strict v2 | relaxed v2 | target | verdict |
|---|---|---|---|---|---|
| policy-gradient (5) | 0.40 | **0.15** | 0.15 | >0.30 | **failed** — PPO/TRPO/SAC core 0.40–0.50 intact; DDPG + TD3 dilute |
| value-based (4) | 0.11 | **0.03** | 0.03 | >0.20 | **failed** — four signatures, no shared loop |
| policy-gradient × value-based | 0.10 | **0.06** | — | <0.10 | met, trivially |
| meta-learning (3) | 0.67 | 0.34 | 0.34 | — | stable |
| jepa (5) | 0.45 | 0.21 | 0.24 | — | stable |
| pc-learning (8) | 0.40 | 0.15 | 0.17 | — | stable |

**DDPG is the finding.** Typed `store@sample, sample@sample, select@action, execute@action, update@parameter` — the DQN replay loop with an actor bolted on. Its strongest pair in the corpus is DQN (0.36 strict, 1.00 lexical on `act, correct, sample`), across the split. Lineage says policy-gradient; mechanism says off-policy replay. The data splits RL as **on-policy trust-region** (PPO, TRPO, SAC: `evaluate@value + improve@policy/parameter`) vs **replay-based off-policy** (DQN, DDPG), not policy-gradient vs value-based. Third time the lineage/mechanism mismatch shows (pc-neuro Phase 15, A3C Phase 18).

**Three of four new papers got bad input.** TD3 → heading "Overestimation Bias" (analysis, not Algorithm 1) → 3 tuples. DDQN and Rainbow → 20% fallback; Rainbow's window hit the C51 distributional projection (`contract/shift/softmax@distribution`), 0.00 with everything. Value-based 0.03 is partly extraction, not mechanism — the Phase 5 Kalman lesson again. Don't judge value-based until the sections are pinned.

**Latent-generative dissolution holds**: every ex-LG paper is 0.00 with every other. Reassignment: Variational Tracking → state-estimation fits (0.33 with Backprop KF). DKF → worldmodels weak (peers 0.06; nearest is Spiking PC 0.20, pc-learning — as the spec predicted). DVBF and L-VSSF: 0.00 to their new peers and to everything at strict — orphans, no family at this granularity. Homes kept, flagged.

**Top analogies unchanged by the RL expansion:** meta-learning × pc-learning 0.14, jepa × pc-learning 0.10 (I-JEPA × PC Networks 0.50 is the top paper pair), jepa × worldmodels 0.06.

## Phase 20 result: pinned re-extraction, RL re-split, orphans finalised (2026-08-25)
TD3, DDQN, Rainbow re-extracted from pinned sections (`fulltext.py --from`, old records in `results_phase19_snapshot/`). `families.json`: `policy-gradient`/`value-based` → **`on-policy`** (PPO, TRPO, SAC; A3C held out as `a3c-outlier`) / **`replay-off-policy`** (DQN, DDPG, TD3, DDQN, Rainbow); DKF → pc-learning; DVBF, L-VSSF listed as orphans; latent-generative gone. Matrices `matrices/phase20_*.csv`. Report `predictions/phase20_results.txt`.

| family (n) | lexical | strict v2 | relaxed v2 | note |
|---|---|---|---|---|
| on-policy (3) | 1.00 | **0.43** | 0.43 | target >0.30 **met** — top family; A3C held out as n=1 outlier (0.22 with it in) |
| replay-off-policy (5) | 0.28 | **0.05** | 0.05 | target >0.20 **failed** — only DDPG × DQN 0.36; TD3, Rainbow 0.00 with everyone |
| on-policy × replay | — | 0.03 | — | target <0.10 met, trivially |
| pc-learning (9, +DKF) | 0.35 | 0.13 | 0.15 | DKF peers 0.05 — weak fit as expected |

**Correct sections didn't fix the three.** TD3 (4.2 Clipped Double Q) → `update@value, estimate@value, minimize@error`. Rainbow (Integrated Agent) → same distributional projection ops as before. DDQN gained `update network@parameter, copy target network@parameter` but the typer wrote multi-word ops (`select action@value`) that don't match DQN's `select@action`. Phase 19's diagnosis (wrong section) was only half right.

**The real reason replay-off-policy doesn't cohere: delta papers.** DQN and DDPG print the whole agent loop as pseudocode; DDQN, TD3, Rainbow inherit it as background and write only their contribution. The extractor reads what's written, so the shared loop is absent from 3 of 5 members. Different failure from wrong-section: it's a property of incremental papers, and no section choice fixes it. General consequence: **mechanism families cohere when members restate the loop (jepa, pc-learning, on-policy trio) and dissolve when members are deltas on a base paper.** Candidate fix: inject the base paper's loop before extracting a delta paper.

**SAC is not a bridge.** No `store/sample` tuple; 0.40 with PPO and TRPO, 0.00 with every replay paper. Off-policy by lineage, on-policy by what its method section describes (`evaluate@value + improve@policy`). Third lineage/mechanism mismatch in RL alone.

Everything outside RL is identical to Phase 19. Top analogies unchanged: meta × pc-learning 0.13, jepa × pc-learning 0.10, meta × on-policy 0.08.

## Phase 21 result: delta-paper test — base loop injected before DDQN, TD3, Rainbow (2026-08-25)
`phase21_inject.py` prepends the base paper's Algorithm 1 (DQN for DDQN/Rainbow, DDPG for TD3) to each delta paper's pinned section, then extracts + types. Sections in `predictions/phase21_sections/`. Phase 20 records → `results_phase20_snapshot/`. Matrices `matrices/phase21_*.csv`. Report `predictions/phase21_results.txt`.

| replay-off-policy (5) | before | after | target |
|---|---|---|---|
| strict v2 | 0.05 | **0.19** | >0.20 — missed by one pair |
| relaxed v2 | 0.05 | 0.19 | |
| lexical | 0.28 | 0.63 | |
| on-policy × replay | 0.03 | 0.00 | |

**The loop appears in all three** (DDQN `store/sample@sample, compute target@value, update@parameter`; Rainbow `store@other, sample@sample, compute@error, update@parameter`; TD3 `update critic@value, update actor@policy, update target networks@parameter`). Every pair among the five rose from ~0 to 0.08–0.36; DDQN × Rainbow 0.00 → 0.33 is the biggest. Residual gap is typing noise: Rainbow `store@other`, TD3 compressing store/sample/select into three `update@` ops.

**Verdict: delta papers were the bottleneck** — not section choice (Phase 20), not the ontology. Four-fold rise from injection alone. Everything outside RL unchanged (on-policy 0.43, meta 0.34, jepa 0.21).

**Caveat, by construction:** measuring DDQN × DQN after prepending DQN's loop to DDQN partly measures inserted text. What the test does show: the extractor recovers a loop from pseudocode in one pass; the delta papers' own text doesn't carry it; and after injection the delta signal mostly disappears (DDQN's `evaluate@value` gone, DDQN × PPO/TRPO 0.17 → 0.00; TD3's clipped-double-Q target gone). Injected records are "base + one op". Fair design next: keep `inherited_ops` separate from own ops and compare both ways.

**General claim, sharpened:** families cohere when members restate the loop. jepa, pc-learning, on-policy do; the DQN/DDPG lineage doesn't (Atari/MuJoCo papers assume the loop). Whether delta-paper is RL-specific or general is testable: inject DDPM into DDIM, MAML into Reptile, and see if those families move.

## Phase 22 result: guard fix, inherited/own split, delta-paper test on diffusion + meta-learning (2026-08-25)
`guard_state()` now fills `store/sample/select/act@other` → `sample/action` at read time. `compare.py --own` scores injected records on own ops only (`phase22_split.py` writes `semantic_ops_v2_inherited`/`_own`). `phase21_inject.py` extended with DDPM → DDIM and MAML → Reptile. Matrices `matrices/phase22_*.csv`. Report `predictions/phase22_results.txt`.

| | full | own-only | target |
|---|---|---|---|
| replay-off-policy (5) | **0.21** | 0.04 | >0.20 met / >0.15 own not met |
| diffusion (5) | 0.07 | 0.02 | |
| meta-learning (3) | 0.34 | 0.10 | |
| on-policy (3, control) | 0.43 | 0.43 | |

**Rainbow fixed by rule** (`store@sample`): Rainbow × DDPG 0.20 → 0.33. **TD3 not fixed**: re-extraction of the injected text gave the same three `update` ops twice — the extractor summarises DDPG's pseudocode into its update steps. Recorded.

**Own-only 0.04: the family is the inherited loop.** DDQN, TD3, Rainbow have 0–1 own ops each (a target rule, nothing, a loss). That's what an incremental lineage is; "real vs inherited" is a false dichotomy for delta papers. Honest description: a base-loop family, two members carry the loop in their own text, three by citation.

**Delta-paper is general, per paper — not per lineage.** DDIM × DDPM **0.00 → 0.50** after injecting DDPM's Algorithms 1+2 (DDIM's section 3 never restates the training/sampling loop). Reptile × MAML **0.40 → 0.40**: Reptile prints its own Algorithm 1, so injection changed nothing. That's the control the by-construction caveat needed — prepending a base loop does not inflate a paper that already restates it. Diffusion family barely moved (0.06 → 0.07) because DDIM was one delta among four restaters.

**Final structure (strict v2):** on-policy 0.43, meta-learning 0.34, replay-off-policy 0.21, jepa 0.21, pc-learning 0.13, particle 0.11, diffusion 0.07, state-estimation 0.06, pc-neuro 0.06, worldmodels 0.03, control 0.00, a3c-outlier n=1. Lexical is saturated on injected/restated loops (on-policy 1.00); strict v2 is the score to read.

## Final live test: Diffuser (2205.09991), unseen paper, unseen field (2026-08-25)
Extract → v2 typing → query against the 58, out of corpus (`test:` source). Pre-registered: gatekeeper said worldmodels/state-estimation, KalmanNet/DreamerV3; Claude said diffusion, DDPM/DDIM. Report `predictions/final_test_diffuser.txt`.

Extracted `predict trajectory, denoise, sample` → canonical `predict, reconstruct, sample`. **Lexical: DDIM 1.00**, DDPM/EDM/DKF/DreamerV3 0.67, Differentiable PF 0.60; family diffusion 0.55. **Strict: 0.20**, one tuple (`sample@sample`) — typer put `predict trajectory` and `denoise` at `@other` because the ontology has no `trajectory` object. Relaxed: diffusion 0.16.

Assigned family **diffusion**; Claude's prediction right, gatekeeper's wrong (state-estimation 0.12 lexical, KalmanNet absent). Same failure class as `task` (Phase 18) and `value/policy` (v2): every new field costs one ontology object before the strict layer can see it. Hypotheses recorded: DDIM σ=0 sampler substitutes into Diffuser at fewer steps; SMC resampling by exp(J) substitutes for gradient guidance (the particle-filter match).

## Final ontology update: `trajectory` (2026-08-25)
`semantic_ontology.json`: `trajectory` added to `object_types_v2` and to the `represented_state` relaxed group; v1 untouched. Diffuser re-typed (old v2 kept in `semantic_ops_v2_before_trajectory`): `predict@trajectory, denoise@trajectory, sample@trajectory`. Report `predictions/final_ontology_trajectory.txt`.

| Diffuser vs corpus | before | after |
|---|---|---|
| strict v2 | 0.20 (sample@sample, 4-way tie) | **0.00** — no corpus paper has any `@trajectory` tuple |
| relaxed v2 | 0.20 ×6 | **DDIM 0.50**, Diff. PF 0.29; diffusion = jepa 0.14 family-mean |
| lexical | DDIM 1.00, diffusion 0.55 | unchanged |

**Strict criterion (>0.20) not met, and correctly so.** The only prior strict match was `sample@sample`; `sample` now takes `trajectory`, which is right. Diffuser's mechanism — a diffusion sampler whose object is a trajectory — has no corpus neighbour. A new object type isolates a paper correctly; it creates a strict match only when a second paper acts on the same object with the same verb. Nearest-paper assignment is unambiguous (DDIM); family-mean relaxed ties diffusion/jepa because `predict@represented_state` is jepa's core.

**Dry run on 11 worldmodels + diffusion records (corpus not modified):** PlaNet genuinely gains `collect data@trajectory, plan@trajectory`; Sohl-Dickstein's `reverse@trajectory` is ungrounded; 5 unrelated tuples drifted on re-run (~1/paper typing noise). PlaNet's verbs don't overlap Diffuser's, so applying would not move the strict score. Left as-is.

Final structure unchanged from Phase 22. For the write-up: relaxed connects a new field on day one; strict says "nothing here does what this does" until the corpus has a second trajectory-mechanism paper (Decision Diffuser, Trajectory Transformer, CEM-MPC).

## Phase 23: predictive transfer test — DDIM in Diffuser, seed 0 (2026-08-25)
First experimental test of a pipeline-generated hypothesis. DDIM (σ=0) implemented in the released Diffuser code (`phase23/diffuser/sampling/ddim.py`, `phase23/ddim_patch.diff`), pretrained T=20 locomotion checkpoints, gymnasium-v4 dynamics with original D4RL v2 data (port check: DDPM-20 = 41.4 / 45.2 / 77.1 vs paper 44.2 / 58.5 / 79.2). Report `phase23/phase23_results.md`, plot `phase23/return_vs_steps.png`.

| env (H) | DDPM-20 | DDIM-20 | DDIM-10 | DDIM-5 | DDIM-2 |
|---|---|---|---|---|---|
| halfcheetah (4) | 41.4 | 43.0 | 42.4 | 39.4 (−4.9%) | 39.2 (−5.3%) |
| hopper (32) | 45.2 | 48.0 | 69.5 | 53.0 | 46.5 |
| walker2d (32) | 77.1 | 79.9 | 82.9 | 81.8 | **55.0 (−29%, fell)** |

**H1 supported.** Deterministic ≥ stochastic on 3/3 envs at full steps; DDIM-10 above baseline on 3/3; DDIM-5 (4× cheaper) within 5% on 3/3; DDIM-2 (10×) fails on walker. The prediction's "5–10× fewer steps within ~5%" was right at 5×, wrong at 10× for H=32 planning. The structural reading behind it — Diffuser is the DDIM sampler over τ, nothing depends on the Markov chain — held quantitatively. One seed; seeds 1–2 running.

## Phase 23 final: 3 seeds (2026-08-25)
45/45 runs. Paired drop vs same-seed DDPM-20 (%, + = worse): halfcheetah DDIM-20/10/5/2 = +1.6 / +1.8 / **+5.1 ± 0.8** / +9.7; hopper = −8 / **−75** / −32 / −12; walker = **+20** / −5.3 / −13 / +25. Report `phase23/phase23_results.md`.

**H1 supported at 2× (DDIM-10 baseline-or-better on 3/3), borderline at 4× (halfcheetah exactly on the 5% line), refuted at 10×.** Hopper DDIM-10 is the sweep's best planner: episodes 690/776/1000 vs baseline 437–539 on every seed — a stability effect, not reward rate. **Stochasticity is not irrelevant on walker**: DDIM-20 lost 32% on 2/3 seeds while DDIM-10 gained on 3/3; non-monotone in steps, so not "coarser plan". Both surprises are horizon × schedule interactions the ontology could not have carried. Practical: DDIM-10 halves planning cost at no loss on all three envs.
