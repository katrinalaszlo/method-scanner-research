# Task Plan: Method Scanner

## Goal
Extract structured method descriptions (name, operations, loss, failure) from paper abstracts with a local LLM, then find cross-domain structural families — e.g. is Kalman filtering, JEPA, and predictive coding the same predict-then-correct loop.

## Current Phase
Phase 23 complete (DDIM in Diffuser: supported at 2×, borderline 4×, refuted 10×). Phase 24 planned: outcome prediction from structure + conditions on D4RL — the original system. Plan in predictions/phase24_plan.md

## Phases

### Phase 1: Working extractor
- [x] Ollama call, JSON slice-and-parse, save to `results/{paper}_{timestamp}.json`
- **Status:** complete

### Phase 2: Web UI
- [x] stdlib `http.server`, form + history, per-result delete
- **Status:** complete — `app.py`, `index.html`

### Phase 3: Cleanup
- [x] Migrated legacy `results.json`, deleted 9 superseded scripts
- **Status:** complete

### Phase 4: Extraction quality (2026-08-24)
- [x] Tabs: Extract | History. Current prompt viewable on Extract page
- [x] Records store `system_prompt`, `user_prompt`, `raw_response`
- [x] CSV download of history; CSV/.txt batch upload (drag-drop, staged, Extract is the only run trigger); template download; re-run per record
- [x] Prompt: rule-list → 3 worked examples (rules don't land on small models; examples do)
- [x] Model bake-off on 10 papers: llama3.2:3b vs qwen3:4b vs qwen3:8b → **qwen3:8b**
- [x] Switched to native Ollama `/api/chat` with `think: false` (OpenAI-compat endpoint can't disable qwen3 thinking → empty output)
- [x] `normalize()`: "none"/"null" strings → null, dedupe ops
- [x] Bug: `do_DELETE` didn't URL-decode filenames → records with spaces/colons undeletable. Fixed with `unquote`
- [x] All 10 records re-run on final prompt + model
- **Status:** complete

### Phase 5: Canonicalize operations
- [x] 13 canonical ops in `app.py` `CANONICAL`. Kat approved with `suppress` and `associate` kept separate so the PCL=Kalman claim is tested, not assumed
- [x] Derived at read time (`with_canonical`), never stored — vocab edits apply to all records instantly. Raw ops untouched
- [x] Static dict, exact whole-string match. Multi-word ops stay unmapped by design
- [x] UI: `canonical` row per card (solid = mapped, dashed = unmapped). CSV: `canonical_ops`, `unmapped_ops` columns. `GET /api/vocab`
- [x] Ran on the 10. **Criterion failed**: Kalman ∩ V-JEPA = ∅. Kalman's abstract yields only `update, compute gains` — no `predict`. See findings
- **Status:** complete (negative result)

### Phase 6: Compare
- [x] Jaccard on canonical op sets, computed in a script over the 30 full-text records
- [ ] Compare tab in the UI — not built; stats came from a one-off script
- [x] Success met: jepa 0.73 within, controls 0.00, filters×predcoding 0.40 is the top cross-family pair after jepa×worldmodels
- **Status:** stats done, UI not built

### Phase 7: Scale by hypothesis
- [x] 30 papers in `papers.txt`: 5 each filters / jepa / predcoding / worldmodels / diffusion + 5 controls (VGG, ResNet, Transformer, word2vec, Adam). All IDs verified against arXiv API — 6 of my first guesses were wrong papers entirely
- [x] `fulltext.py`: arXiv ID → PDF → `pdftotext` → method section (first numbered heading matching method/approach/model/…, else 20% window, 1500 words) → `POST /api/extract` with `source`
- [x] Records gained `source` field (`arxiv:ID [family] heading: …`). CSV + UI show it
- [x] Head-verb fallback in `canonicalize()`, returned as `head_matched_ops`, underlined in UI
- [x] Pipeline verified on I-JEPA (heading hit) + DDPM (fallback) before the full run
- [x] Full run — 30/30 ok, ~40s/paper
- [x] Section finder: 1/8 → 29/30 after handling roman numerals, small-caps `M ETHOD`, split-line numbers, footnote markers (monotonic numbering + intro reset)
- [x] Phase 6 stats on the 30 (abstract records excluded — different input type). See findings
- **Status:** complete

### Phase 9: Vocab round 2 + example-echo fix
- [x] Example 3 → speech enhancement, failure "musical noise artifacts at very low SNR", plus explicit "do not reuse wording" line
- [x] Vocab +7 (project, sample, augment, diffuse, reverse, plan, act) + phrase table (compute error → compare, stop-gradient → mask)
- [x] `compare.py` with `--min-shared`. `similarity_matrix.csv` written
- [x] Re-ran 30 via `/api/rerun` (source preserved). 30/30
- [x] Old echo gone. **New echo: Backprop KF failure = "very low signal-to-noise ratios"** — not in its text
- [ ] Object-aware mapping — deferred; PKF/Backprop KF still unmapped on `compute …`
- **Status:** complete, one open issue

### Phase 10: Failure-field echo
- [x] Chose (a). Added example 4: inverted-index retrieval, `failure: null` — far from every corpus family
- [x] Strict failure rule in prompt: only when the Description states it, quote its words, null is the normal case
- [x] Rejected DeepSeek's prompt twice: closed vocab in prompt (A/B: 7-11 generic ops, all failures null), and PCL as an example (corpus paper)
- [x] A/B on 6 problem papers: Backprop KF echo → "struggles with severe occlusion" (grounded). All 6 failure strings verified present in section text
- [x] Re-run 30 — 30/30
- [x] `compare.py`: all family means up vs Phase 9; filters × predcoding top cross-family for 3rd run
- [x] Read-time `guard_failure`: example-fragment echo and prose-null → null + `failure_flag`. Caught EDM echo + DDIM "no failure mentioned"
- **Status:** complete

### H1 ablation (DeepSeek-proposed, run 2026-08-24)
- [x] `--drop` flag in compare.py. Result: H1 supported — KalmanNet bridge is predict-dependent; family-level filters×PC signal survives on `correct`. See findings
- **Status:** complete

### Phase 11: Semantic typology (DeepSeek spec, 2026-08-24)
- [x] Rejected the spec's static op→object map: object type is a property of (paper, op) — that IS the H1 finding. Replaced with a per-record typing pass
- [x] `semantic_ontology.json`: 12 object types, 7 contexts, 3 compatibility groups (state~latent, error~gradient, noise~distribution~sample). Additive
- [x] `semantic_type.py`: second LLM pass, ops held fixed, assigns object_type + context + quote per op, `grounded` flag when the quote is in the text. Writes `semantic_ops` into the record (additive)
- [x] `compare.py --semantic [--strict|--relaxed]` → `semantic_similarity_matrix.csv`. Lexical path untouched. `--drop` composes with it for the semantic H1 re-test
- [x] Typing pass on 30 — 30/30 after the strict=False fix. 106 tuples, 86 grounded
- [x] Report in findings: strict/relaxed/lexical tables, KalmanNet bridge gone, jepa×predcoding the top semantic pair, bridge tuples listed
- [x] `semantic_similarity_matrix.csv` (strict) sent to Kat
- **Status:** complete

### Phase 12+13: Mechanism families
- [x] `families.json`: pc-neuro (2), pc-learning (3), state-estimation (3), latent-generative (1), particle (1). Assigned from Phase 11 tuples, not from DeepSeek's table (DKF is a VAE → latent-generative, not state-estimation)
- [x] `compare.py --families mechanism`; lineage stays default; n per family printed; singletons flagged
- [x] Re-ran lexical / strict / relaxed. Within-family up across the board under strict. jepa × pc-learning confirmed #1 cross-family. See findings
- [x] Corrected two claims in DeepSeek's recap: filters×PC "0.22 semantic" was the lexical drop-predict figure (semantic is 0.04); "proven / production-ready" overstates a 30-paper, single-pass pilot
- **Status:** complete

### Phase 14: Reliability and expansion (DeepSeek spec)
**Track A — pass-2 typing**
- [x] `semantic_type.py --pass 2`: strict-grounding prompt, `unknown` allowed, "vector/embedding is latent not state" rule, quote must contain the object word. Same model (only capable local one) → measures prompt sensitivity, not model independence. Stored as `semantic_ops_pass2`; exports `typing_pass2.json`
- [x] Fixed two pass-2 failure modes: truncation at token cap (salvage complete items, cap 600→1500), list-of-strings output (raise, log, skip)
- [x] `agreement.py`: exact agreement, unknown rate, per-paper changes, re-typing pairs, fate of pass-1 `state`, per-op agreement
- [x] Full run 30/30 (one escape-char failure fixed and re-run)
- [x] Report: 72% exact agreement, state sticky 20/26, family stats stable, pair headline fragile. `agreement_report.txt`
**Track B — expansion candidates**
- [x] `candidates_phase15.csv`: 51 candidates, 5 families, arXiv API, excludes the 30 already in corpus
- [x] Shortlist of 15 (3/family) downloaded and section-checked: 9 heading hits, 6 fallback — all usable by `fulltext.py`
**Track C — rules**
- [x] `compatibility_rules.md`: hierarchy, strict/relaxed rules, reasoning per rule tied to Phase 11–13 numbers, known weaknesses
- **Status:** complete

### Phase 15 (proposed): add 15 papers, 3 per thin family
| family | arXiv ids |
|---|---|
| pc-learning | 2106.13082, 2103.03725, 2305.13562 |
| pc-neuro | 2205.05303, 1908.08655, 2210.15752 |
| state-estimation | 2102.10021, 1703.02310, 2503.05490 |
| latent-generative | 1910.06205, 2201.01353, 1511.07367 |
| particle | 1805.11122, 1905.12885, 1508.06818 |
- Pipeline: append to papers.txt with mechanism family → `fulltext.py` → `semantic_type.py` (both passes) → add ids to families.json → compare
- ~10 min extraction + ~30 min two typing passes
- [x] 15/15 extracted (PF-RNN via 30% window after 20% failed twice), 45/45 typed both passes
- [x] compare.py family regex fixed for hyphenated names
- [x] Matrices: `matrices/phase15_{lexical,semantic_strict,semantic_relaxed,semantic_strict_pass2,semantic_strict_agreed}.csv`
- [x] Stability table + Phase 16 recommendation in findings
- **Status:** complete

### Phase 16 candidates (Kat picks)
- [x] 16b: latent-generative dissolved (→ state-estimation ×2, particle, worldmodels); pc-neuro re-split by object — two spiking papers moved to pc-learning (parameter updates), PredNet/PCL+/Dendritic stay
- [x] 16c (Track B): DQN/PPO/TRPO/SAC/A3C extracted + typed. policy-learning within strict 0.02 → not a family under current ontology. Root cause: no `evaluate`/`improve` ops, no value/advantage/policy objects. Matrices `phase16_*.csv`
- [x] Bugs: Roman numeral `XX` crash in section finder; `[?]` family tag in single-id mode (3 records' source corrected to `[policy-learning]`)

### Phase 17: predictions on unseen papers
- [x] `fulltext.py --test` (test: source, 1000-word cap, --words), `compare.py --query` with --min-shared
- [x] DVBF, Snoek BO, MAML extracted + typed + scored. Results in findings + `predictions/`
- [x] Section finder: running page headers ("Published as a conference paper…") excluded
- **Status:** complete

### Phase 18: Ontology v2 + family re-validation
- [x] Snapshot `results_phase17_snapshot/`
- [x] Vocab: `evaluate`, `improve`, `transition` (+ `act` absorbs "select action", "interact"). Types: `policy`, `value`, `task`. `value` joins learning_signal relax group
- [x] `semantic_type.py --pass v2` → `semantic_ops_v2` (v1 kept). `compare.py --pass v2` (falls back to v1 for un-retyped papers). `diff_v2.py` for the per-tuple diff
- [x] DVBF, MAML promoted from test: (flag `promoted_from_test`). Reptile 1803.02999 + Meta-SGD 1707.09835 added. families.json `_moves_phase18`: latent-generative reinstated (5), meta-learning (3)
- [x] Chain: extract 2; file filter came back empty so the whole corpus (54+2) was re-typed under v2 — better for consistency
- [x] Verdicts: meta-learning 0.34 met + distinct; policy-learning 0.14 (missed by 0.01; PPO/TRPO/SAC ≈0.43, DQN/A3C outliers); latent-generative 0.00 failed. New top between-family: meta × pc-learning 0.14
- **Status:** complete

### Phase 19: policy-gradient / value-based split
- [x] families.json: policy-gradient (PPO, TRPO, SAC + DDPG 1509.02971, TD3 1802.09477); value-based (DQN, A3C-flagged + Double DQN 1509.06461, Rainbow 1710.02298). latent-generative retired: DVBF→worldmodels, DKF→worldmodels, L-VSSF→state-est, Var. Tracking→state-est, BBVI→particle. `_moves_phase19`
- [x] State guard v2: `act/select/execute→action`, `sample/resample/store→sample` — never `state`
- [x] Chain: extract 4/4, v2 typing 4/4, v1 typing 4/4. Section finder: DDPG heading ok; TD3 wrong heading; DDQN, Rainbow fallback window
- [x] Comparator: lexical / strict v2 / relaxed v2, `matrices/phase19_*.csv`
- [x] Verdicts: policy-gradient **0.15 failed** (core trio 0.43, DDPG typed as DQN loop, TD3 mis-sectioned); value-based **0.03 failed** (n=4, 3 bad inputs); between 0.06 met trivially. Dissolution holds; DVBF, L-VSSF orphans
- [x] Report `predictions/phase19_results.txt`, findings updated
- **Status:** complete (negative result on both targets)

### Phase 20: pinned re-extraction, RL re-split, orphans
- [x] `fulltext.py --from "<regex>"` pins section start. TD3 → 4.2 (2000w), DDQN → Double DQN, Rainbow → Integrated Agent. Old records → `results_phase19_snapshot/`
- [x] Re-typed v2 + v1. Tuples barely changed — sections describe each paper's delta, not the replay loop
- [x] families.json: `on-policy` (PPO, TRPO, SAC; A3C → `a3c-outlier` n=1) / `replay-off-policy` (DQN, DDPG, TD3, DDQN, Rainbow); DKF → pc-learning; `_orphans_phase20`; `_moves_phase20`
- [x] Comparators, `matrices/phase20_*`. on-policy **0.43** met, replay-off-policy **0.05** failed, between 0.03 met. SAC not a bridge
- [x] Finding: delta papers don't restate the loop → family can't cohere by extraction alone. Report `predictions/phase20_results.txt`
- **Status:** complete

### Phase 21: delta-paper test
- [x] `phase21_inject.py`: base Algorithm 1 (DQN / DDPG) + pinned delta section → extract → v2 + v1 typing. Sections saved `predictions/phase21_sections/`. Phase 20 records → `results_phase20_snapshot/`
- [x] replay-off-policy strict **0.05 → 0.19** (target >0.20, missed by one pair); lexical 0.28 → 0.63; every pair among the five rose. On-policy trio unchanged
- [x] Verdict: delta papers were the bottleneck. Caveat: partly by construction — delta signal lost after injection. Report `predictions/phase21_results.txt`
- **Status:** complete

### Phase 22: fixes, inherited/own, delta test outside RL
- [x] Guard: `NEVER_STATE` fires on `other/unknown` too → Rainbow `store@sample`. Server restarted
- [x] TD3 re-extract via `/api/rerun`: same three `update` ops — not met, recorded
- [x] `phase22_split.py` → `semantic_ops_v2_inherited/_own`; `compare.py --own`. replay-off-policy full **0.21** (met), own **0.04** (not met — the family is the inherited loop)
- [x] DDPM→DDIM: 0.00 → 0.50. MAML→Reptile: 0.40 → 0.40 (Reptile restates its loop; injection didn't inflate). Delta-paper is general per paper, not per lineage
- [x] Report `predictions/phase22_results.txt`; matrices `phase22_*`
- **Status:** complete

### Final live test: Diffuser (unseen field)
- [x] 2205.09991 extracted (test:, 1000w), typed v2+v1, queried lexical / strict / relaxed. arXiv API 429 twice — bypassed metadata call with cached title
- [x] Lexical DDIM 1.00, family diffusion 0.55. Strict 0.20 single tuple — `trajectory` missing from ontology. Claude's pre-registered prediction right, gatekeeper's wrong
- [x] Two transfer hypotheses recorded. `predictions/final_test_diffuser.txt`
- zsh gotcha hit again: unquoted `$m` in a for-loop passed `--semantic --pass v2` as one arg → silent lexical. Caught by output shape
- **Status:** complete

### Final ontology update: `trajectory`
- [x] `trajectory` in `object_types_v2` + `represented_state` group. Diffuser re-typed (old v2 preserved). Strict 0.20 → **0.00** (no corpus `@trajectory`), relaxed DDIM 0.20 → **0.50**, lexical unchanged
- [x] Dry run on 11 worldmodels+diffusion scratch copies: PlaNet gains `collect/plan@trajectory`; 5 unrelated drifts. Not applied — no verb overlap with Diffuser
- [x] `predictions/final_ontology_trajectory.txt`. Final structure = Phase 22
- **Status:** complete (strict criterion not met, by construction — recorded)

### Phase 23: predictive transfer test — DDIM in Diffuser
- [x] Clone, DDIM sampler patch, device/py3.14 fixes, shims for gym/d4rl/mujoco_py, D4RL HDF5s, pretrained weights
- [x] Smoke: DDPM-20 and DDIM-20/10/5/2 run end to end on MPS; reference rollouts shipped with weights: HC 0.431, hopper 1.008, walker 0.798
- [x] Seed 0 complete (15 runs). `collect.py` (rollout.json scanner — sweep's hardcoded `defaults` path was wrong) → `results_collected.csv`; `analyze.py` → `results_table.csv`, `return_vs_steps.png`
- [x] Verdict seed 0: **H1 supported** — DDIM-20 ≥ DDPM-20 on 3/3; DDIM-10 3/3 above baseline; DDIM-5 3/3 within 5%; DDIM-2 walker falls (−29%). `phase23_results.md`, findings
- [x] Seeds 1–2 done, 45/45. Paired analysis (`results_paired.csv`). Final verdict: supported at 2×, borderline at 4×, refuted at 10×; walker contradicts the stochasticity clause
- [x] `phase23_results.md` final; write-up section added; copies on Desktop
- **Status:** complete

### Phase 24a (stopped): η sweep
- [x] `sweep_eta.sh`: hopper + walker, DDIM 10 and 20 steps, η ∈ {0.5, 1.0}, seeds 0–2 (24 runs, ~5–6 h). Launched detached; `analyze_eta.py` pairs against p23 DDPM-20 and DDIM-η0 of the same seed
- [ ] Questions: does noise at 20 steps fix walker's DDIM-20 loss? does noise at 10 steps break hopper's DDIM-10 gain?
- **Status:** stopped by Kat after ~5 min (too long at 5–6 h). Script + analysis ready; a cheaper version is 1 seed, hopper only, DDIM-10/20 × η {0.5, 1.0} = 4 runs ≈ 45 min

### Phase 24: outcome prediction — structure + conditions → outcome (the original system)
- [ ] Plan: `predictions/phase24_plan.md`. D4RL locomotion, ~30 papers, 9 cells each; rows = structure (existing pipeline) + conditions (new extraction) + reported outcome (hand-verified)
- [ ] Predictor ladder A (cell mean) → B (conditions) → C (structure) → D (both), D′ (no lineage label), D″ (no inherited ops); leave-one-paper-out, bootstrap CI
- [ ] Thesis test: D beats B beyond CI and D′ beats B. Phase 23's 15 measured rows held out as a check
- **Status:** planned, not started

### (old) Phase 24 candidates
- η sweep at DDIM-10 / DDIM-20, η ∈ {0.5, 1.0}, hopper + walker first
- SMC resampling by exp(J) replacing gradient guidance (particle-filter match)
- Second `@trajectory` paper for the scanner corpus

### Next candidates
- Second trajectory-mechanism paper (Decision Diffuser 2211.15657, Trajectory Transformer 2106.02039) so `@trajectory` has a strict neighbour
- Typing noise ~1 tuple/paper on re-run — measure across the corpus before any whole-corpus re-type
- TD3: inject loop after the delta or cap delta at 600w — extractor summarises pseudocode followed by prose
- Show full vs own per record in UI/CSV
- Injection as pipeline policy needs citation detection; manual for now

### (old) Phase 22 candidates (superseded)
- `inherited_ops` kept separate from own ops; compare own / own+inherited; removes the by-construction caveat
- Delta-paper test outside RL: DDPM→DDIM, MAML→Reptile. Is it an RL-lineage property or general?
- Guard `store@other` → sample (extend NEVER_STATE to `other`)

### (old) Phase 21 candidates (superseded)
- Delta-paper test: prepend DQN/DDPG Algorithm 1 to DDQN/TD3/Rainbow sections, re-type, see if replay-off-policy appears
- Typer multi-word ops: head-verb normalisation on the semantic path (`select action` → `select`)
- A3C: keep as flagged outlier or drop; trio n=3 at 0.43

### (old) Phase 20 candidates (superseded)
- Pin headings for TD3 (Algorithm 1), DDQN (Double Q-learning), Rainbow (Integrated Agent); re-extract + re-type; re-judge value-based
- Re-split RL by mechanism: on-policy (PPO, TRPO, SAC) vs replay/off-policy (DQN, DDPG, DDQN, Rainbow, TD3). SAC is the test case
- DKF / DVBF / L-VSSF orphans: shared `infer@latent` or accept between-family

### (old) Phase 19 candidates
- Split policy-learning → `policy-gradient` (PPO, TRPO, SAC) + add 2 policy-gradient papers (DDPG 1509.02971, TD3 1802.09477) and 2 value-based (Rainbow 1710.02298, Double DQN 1509.06461) to test both halves
- `state` guard v2: the guard only checks temporal markers; v2 introduced `act@state`, `sample@state` — add a rule: `act`/`sample`/`select` never take `state`
- Retire latent-generative for good; assign DVBF→worldmodels, DKF→worldmodels, Variational Tracking→state-estimation, L-VSSF→state-estimation, BBVI→particle (their Phase 16 homes)
- Write-up

### (old) Phase 18 candidates
- **Ontology v2** (was Phase 17 candidate; now confirmed by two phases): ops `evaluate`, `improve`; objects `value`, `policy`, `task`. Re-type RL ×5, Dreamer ×3, MAML, then re-test policy-learning and MAML
- Revisit latent-generative: DVBF + DKF + the 3 dissolved papers as a 5-paper test of `infer@latent + reconstruct@input + transition@state`
- Claude API as second typing rater (real model independence; seconds/paper)

### (old) Phase 17 candidates
- **Ontology v2 for RL:** add canonical ops `evaluate` (policy/value evaluation) and `improve` (policy improvement); object types `value` (value/advantage/return/Q) and `policy`. Re-type the 5 RL papers + Dreamer family only (they have actors). Re-test policy-learning. ~15 min
- Live prediction test #2 on a method section, not an abstract
- Compare tab in UI
- [x] 16a: `guard_state()` in app.py. 7 demotions, false SimSiam×NeuralKF bridge gone, jepa strict 0.12→0.21, true bridges untouched. Matrices `phase16_semantic_strict{,_agreed}.csv`
- Second batch to n=8 for pc-learning, state-estimation, particle only
- Compare tab in UI (still script-only)
- Typing consistency: second typing pass with a different seed/model and measure agreement per tuple; or hand-audit the 20 ungrounded quotes
- `state` over-assignment: tighten the ontology definition or add a `vector` type for non-dynamical embeddings
- Compare tab in the UI — still script-only

### Other candidates (unpicked)
- Object-aware mapping for generic-head ops (`compute data association weights`, `compute gradients`) — filters and Adam are underscored because of this
- Neuroscience vocab: `inhibit → suppress`, `excite`, `stimulate` — PCL+ maps to nothing
- Compare tab in the UI (matrix is script-only now)
- Second corpus round: family = shared loop, not lineage — filters family needs re-picking on that basis
- Two-pass `failure` extraction if echo rate climbs above 1/30

### Phase 8: Method sections instead of abstracts
- [x] Narrow test: Kalman 1960 body → `estimate, update covariance` → canonical `predict`. Abstract-only recall confirmed as the cause of the Phase 5 miss
- [x] Head-verb fallback — done in Phase 7
- [x] Full-text sourcing — `fulltext.py`, done in Phase 7
- **Status:** folded into Phase 7

## Decisions Made
| Decision | Rationale |
|---|---|
| qwen3:8b over llama3.2:3b | Every llama failure (nouns as ops, dupes, failure-mode leaking into ops, example-echo) gone on qwen. ~5.5s vs ~2.5s, irrelevant at this scale |
| Few-shot over rules | Negative instructions ("do not list derive/formulate…") ignored by 3b model; 2-3 worked examples fixed it. 3rd example with a real failure mode restored `failure` recall |
| Native Ollama API, not OpenAI client | Only way to set `think: false` on qwen3 |
| Extract is the only run trigger | Batch drop stages rows, nothing runs until click. Kat's call |
| Canonicalize before scaling | Uncontrolled verbs (predict/forecast/infer) make any comparator score real matches as zero. Scaling first scales noise |
| Scale by hypothesis, not volume | 5/family + controls tests the claim; random papers don't |
| Raw ops always preserved | Canonical layer is derived and revisable; raw is evidence |
| stdlib server, vanilla JS, no build | Single-purpose local tool |

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| Assumed deleted scripts recoverable from git | 1 | `GitHub/` is gitignored in wiki repo. Copied to scratchpad first |
| Server restarts silently failed — old process held :8000 | 2 | `pkill -9`, then start |
| qwen3 via OpenAI-compat: 0/10, all output spent in `<think>` | 1 | `max_tokens` raise — still empty content |
| qwen3 via OpenAI-compat `extra_body={"think": False}` | 2 | Ignored by /v1 endpoint. Switched to native `/api/chat` |
| qwen3:4b ignores `think:false`, narrates instead of JSON | 1 | Dropped 4b from consideration |
| 3rd few-shot example (theory paper → `[]`) made llama worse | 1 | Reverted to 2 examples for llama; irrelevant after qwen switch |
| `DELETE /api/results/<name with spaces>` → 404 | 1 | `unquote()` the path |
| arXiv API over `http://` returns empty body | 1 | use `https://` |
| 6 of 30 guessed arXiv IDs were unrelated papers | 1 | never trust an ID from memory — verify title via API before use |
| `semantic_type.py` died at paper 11: JSONDecodeError, raw newline inside a quoted string | 1 | `json.loads(..., strict=False)`; per-record try/except that logs `ERR` and continues (never silent). Resumed — skips already-typed records |
| `fulltext.py` crashed on TRPO: `int('XX')` — a non-numeric section token | 1 | `section_number()` returns None for anything not a digit or I–X; caller skips. SAC/A3C were never reached; re-extracted after fix |
| Whole `Desktop/Github-Wiki` vanished mid-session | 1 | Moved by the OS to `Desktop/Desktop - Kat's MacBook Air/Github-Wiki` (iCloud Desktop sync). Server kept the moved cwd but its `RESULTS_DIR` string was stale → served 0 records. Restarted from new path. Kat to confirm which location is canonical |

## Open Questions
- Canonical vocabulary: who owns it — hand-written list, or induced from the data at 40 papers?
- `loss` field: keep quoting the abstract ("overall performance measure") or force null unless a named loss?
- Info Spreading has `describe` as an op (paper-verb slipped through). One word; not worth another prompt round
