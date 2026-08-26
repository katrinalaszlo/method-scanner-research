# Progress Log

## Session: 2026-08-24 (morning)
- Phases 1-3 complete. UI built, legacy migrated, scripts pruned.

## Session: 2026-08-24 (afternoon)
- Tabs, prompt view, prompt+raw stored per record
- CSV export, CSV/.txt batch with drag-drop, template, staged run (Extract = only trigger), re-run button
- Prompt iterated: rules → 2 examples → 3 examples. Verified each on the hard trio (V-JEPA, PKF, Info Spreading) then all 10
- Bake-off llama3.2:3b / qwen3:4b / qwen3:8b. qwen3:8b wins. Native API required for `think:false`
- `normalize()` added. `do_DELETE` unquote bug fixed
- All 10 records re-run twice (qwen switch, then 3-example prompt). History is clean: 10 records, one prompt, one model
- Legacy 5 records (no abstract stored) deleted
- Test records created during verification all deleted after

### Tests run this session
| Check | Result |
|---|---|
| `POST /api/batch` with Kat's 10-row .txt | 10/10 |
| `POST /api/rerun` on PKF | replaced, `derive/formulate/apply/show` → `associate, update` |
| `DELETE` with spaces in filename | 404 before fix, 200 after |
| `GET /api/results.csv` | 12 columns, quoted multi-line cells parse |
| qwen3:8b smoke, "Failure: none" in text | `failure: null` |

### Phase 5 (same afternoon)
- Vocab approved with suppress/associate split. Built `canonicalize()` + `with_canonical()`, read-time derivation
- UI canonical row, CSV columns, `/api/vocab`
- Result on 10 papers: criterion not met (Kalman ∩ V-JEPA = ∅). Recorded in findings

### Phase 8 narrow test
- Kat picked (b). Fetched Kalman 1960 PDF, `pdftotext`, sliced Theorem-3 derivation, ran dry through `extract_method`
- Result: `predict` appears from body. Cause #1 confirmed. 38s per 1300-word input

### Phase 7 setup
- Kat: go. Head-verb fallback built (flagged, not silent)
- Candidate list drafted from memory, verified: 24/30 right, 6 wrong (4 filters, 2 predcoding). Fixed by arXiv title search
- `fulltext.py` written, tested on 2, then 30-paper run launched in background. Log: scratchpad `run30.log`

### Phase 7 run
- First run: 5/5 filters hit fallback → killed at 8, fixed finder (4 pdftotext layouts), dropped 8 records, restarted
- Second run: 30/30, 29 by heading, 0 fail
- Phase 6 stats: within jepa 0.73 / predcoding 0.46 / control 0.00; filters×predcoding 0.40; predict+correct in 3 PC papers + KalmanNet
- Found example-echo in `failure` on World Models

### Phase 9
- Kat pasted a DeepSeek spec; two items didn't match the repo (nonexistent files, llama constraint). Kept qwen3:8b, flagged
- Example 3 rewritten, vocab +7, compare.py built, 30 re-run
- Result: top pairs I-JEPA×World Models, KalmanNet×3 PC papers, DeepKF×DreamerV3×DDIM. New echo on Backprop KF

### Phase 10
- DeepSeek prompt A/B'd on 6 → rejected (vocab-menu effect, failure field killed). Recorded in findings
- DeepSeek's revised prompt: took the strict-failure rule + null example idea, rejected PCL example + its VOCAB_MAP
- Example 4 (inverted index, null failure) added. A/B on 6: echo gone, all failures grounded
- 30 re-run launched (`rerun30_p10.log`)

### Phase 10 close
- 30/30 re-run. Family means all up. One echo (EDM) + one prose-null (DDIM) → read-time guard added, flags visible in UI + CSV
- `similarity_matrix.csv` sent to Kat

### H1
- Directory moved by iCloud mid-task; server restarted from new path
- Ablation run; H1 supported. Findings updated

### Phase 11
- Repo location confirmed: `Desktop/Desktop - Kat’s MacBook Air/Github-Wiki`. CLAUDE.md tanso paths fixed (were in the wiki's GitHub/ all along)
- Ontology + typing pass + semantic comparator written. Tested typing on KalmanNet/Adam/PredNet: predict@state, bias correct@gradient, update states@state — the split H1 predicted
- Typing pass running on 30 (`semantic.log`)

### Phase 11 close
- Typing crashed at 11 (control char in JSON) → strict=False + logged skip, resumed, 30/30
- Ran lexical / strict / relaxed / semantic-H1. Findings recorded. Matrix sent

### Phase 12+13
- families.json + --families flag. Re-split validated under all three comparators
- Shell gotcha: zsh doesn't word-split unquoted $var — a loop passed "--semantic --strict" as one arg and silently ran lexical. Re-ran directly

### Phase 14
- Pass-2 typing: BYOL still `project@state` under the stricter prompt — over-assignment survives the rule
- Early agreement (5 papers): 15/20 exact, 1 unknown; pass-1 `state` → `state` 4/4
- Track B: 51 candidates, 15 shortlisted + section-checked. Track C: rules doc written

### Phase 14 close
- Pass 2 30/30. agreement.py, compare.py --pass/--agreed. Findings recorded
- Delivered: typing_pass2.json, agreement_report.txt, candidates_phase15.csv, compatibility_rules.md

### Phase 15
- Chain ran to completion despite the 10-min cap (extract → pass 1 → pass 2). PF-RNN retried with a shifted window
- Regex bug on hyphenated family names found and fixed
- 45 papers, all comparisons run, matrices saved, findings + stability recorded

### Phase 16a
- state guard implemented at read time; verified it removes only the false bridge

### Live prediction test #1
- `--query` mode in compare.py; test record kept out of corpus via `test:` source
- DeepSeek predicted pc-learning/I-JEPA; typed result says jepa/world-model, closest World Models (relaxed) / PlaNet (strict)

### Phase 16b/c
- Families fixed from typed tuples; 5 RL papers in. policy-learning fails to cohere → diagnosed as ontology gap (no value/policy objects)
- Two pipeline bugs fixed en route

### Phase 17
- 3 unseen papers scored. 1.5/3 vs pre-registered predictions. Negative control worked via typing (lexical would have failed it)
- MAML pass-2 parse quirk (two lists) — first-list salvage, still a stub; pass 1 used

### Phase 18
- Ontology v2 built and run corpus-wide. Verdicts recorded. Matrices + diff saved

### Next
Phase 19 per Kat — policy-gradient split is the clear one.

### Phase 19
- Session cut mid-chain; resumed, chain had completed on its own (extract 4/4, v2 4/4, v1 4/4)
- Ran lexical / strict v2 / relaxed v2 with mechanism families. Matrices `phase19_*`
- Both targets failed: policy-gradient 0.15, value-based 0.03. DDPG types as the DQN loop (0.36 with DQN — strongest RL pair, across the split). TD3/DDQN/Rainbow got wrong sections
- Dissolution holds; Var. Tracking fits state-est, DKF weak, DVBF + L-VSSF orphans
- Report `predictions/phase19_results.txt`; findings + plan updated

### Phase 20
- `--from` pin added to fulltext.py; TD3/DDQN/Rainbow re-extracted from correct sections, old records snapshotted
- Tuples barely moved — the sections are deltas. Diagnosis shifted from "wrong section" to "delta papers don't restate the loop"
- families.json re-split on-policy / replay-off-policy, DKF → pc-learning, orphans flagged
- on-policy 0.22 (trio 0.43), replay-off-policy 0.05, between 0.04. SAC not a bridge
- Report `predictions/phase20_results.txt`; findings, plan updated

### Phase 21
- Injection script; Phase 20 records snapshotted; 3 re-extracted + typed. Head-verb fallback on the semantic path already handled multi-word ops
- replay-off-policy 0.05 → 0.19 strict, 0.28 → 0.63 lexical. Missed target by one pair (Rainbow store@other, TD3 collapsed loop to update@)
- Delta-paper hypothesis supported; by-construction caveat recorded. Report + findings + plan updated

### Phase 22
- Guard extended to `other/unknown` objects; server restarted to apply. TD3 rerun twice-identical, not forced
- Split script + `--own` flag. Full 0.21 / own 0.04 for replay-off-policy
- DDIM injected (0.00 → 0.50 with DDPM); Reptile injected (unchanged — the control). Snapshots `results_phase21_snapshot/`
- Report, findings, plan updated

### Final live test
- Diffuser 2205.09991. arXiv API 429 on back-to-back calls; used cached title. Out-of-corpus record
- Lexical/relaxed: diffusion, DDIM 1.00. Strict: one tuple — `trajectory` not in ontology
- zsh unquoted-var gotcha recurred; reran with explicit flags. Report + findings + plan updated

### Final ontology update
- `trajectory` added (v2 + relaxed group). Diffuser force-retyped, old v2 kept in a side field
- Strict 0.20 → 0.00 (correct isolation), relaxed DDIM 0.50. Dry-run on 11 scratch copies: PlaNet gains trajectory; corpus untouched
- Report + findings + plan updated. Ready for write-up

### Phase 23 (setup)
- Diffuser cloned to `phase23/diffuser`; DDIM sampler written (`diffuser/sampling/ddim.py`, routed via `sampler=ddim`, `ddim_steps`, `ddim_eta`); patch saved `phase23/ddim_patch.diff`
- Original stack (py3.8 / mujoco-py 2.0 x86) not viable on M4. Modern stack: py3.14 venv, torch MPS, mujoco 3.12, gymnasium v4 envs, original D4RL v2 HDF5s (620 MB) + `gym`/`d4rl`/`mujoco_py` shims in `phase23/shims`
- Checkpoints are T=20, not 100 → sweep is DDPM-20 vs DDIM-20/10/5/2. ~0.5 s/env-step on MPS
- Sweep launched detached (`sweep.sh` → `results.csv`), 3 envs × 5 configs × 3 seeds ≈ 4.5 h

### Phase 23 (seed 0)
- 15/15 runs. H1 supported: deterministic sampler ≥ stochastic on all envs; 4× fewer steps within 5% everywhere; 10× fewer breaks walker
- Collector fixed (savepath is `plans/H{H}_T20_d0.997/<suffix>`, not `defaults`). Results doc + plot + findings written. Seeds 1–2 still running

### Phase 23 final
- 45/45. Paired analysis. Supported at 2×, borderline at 4×, refuted at 10×; walker contradicts stochasticity clause; hopper DDIM-10 never falls
- Results doc final, findings, plan, write-up section. Copies on Desktop. Monitor stopped

### Phase 24 (η sweep) launched
- 24 runs detached (`sweep_eta.sh`), monitor on `sweep_eta.log`, `analyze_eta.py` ready. Halfcheetah excluded (stochasticity was a wash there)
- Phase 24 stopped by Kat (too long). No runs completed. `sweep_eta.sh` / `analyze_eta.py` kept for a smaller version

### Phase 24 (outcome predictor, D4RL)
- 28 D4RL papers pulled; 27 pushed through the scanner (5 re-pinned to method sections, QDT needed a parser fix for a stray-quote key). Table/hyperparameter extraction with verbatim provenance in `outcome_predictor/sources/tables/`
- `d4rl_dataset.csv`: 530 rows, 27 methods; Phase 23 sweep folded in as 15 Diffuser pilot rows. Schema frozen (`schema.md`) before evaluation
- Leave-one-method-out, Ridge + kNN. Primary `MAE(D)<MAE(B)` not supported (Ridge 59.9 vs 37.9 mean per-method; kNN 27.3 vs 27.3). Lineage comparison not supported. Structure alone ≈ mean baseline
- Ridge aggregate dominated by MOReL/QGPO extrapolation (all conditioned models); per-method D<B on 21/27 (median −2.8) under Ridge only, 3/27 under kNN
- Root cause on the structure side: 80 tuples, 11 shared by ≥2 methods — feature near-empty at held-out time. Follow-ups (relaxed vocabulary, bounded predictor, dataset-first kNN distance) must be a new labeled phase. `outcome_predictor/results.md`

### Phase 25 (exploratory follow-up, `outcome_predictor/phase25/`)
- Pre-registered in `schema25.md`: vocab variants strict/relaxed/bare, Ridge alpha≥1 + clipped predictions, dataset-first kNN
- Clipped Ridge: B 24.5, D 21.9–23.5, E 24.1; D<B on 20–22/27. Dataset-first kNN: B 14.7, D 14.8, E 15.8 — dataset identity dominates, structure adds nothing there
- Verdict: Phase 24 negative stands; small consistent Ridge-only gain (~2–3 pts) worth a pre-registered Phase 26 on a second benchmark. `results25.md`

### Phase 26 (`outcome_predictor/phase26/`)
- 26a Atari 100k: 17 papers scanned + tables extracted with provenance; 475 rows / 17 methods / 26 games, outcome = 100×HNS from each paper's own Random/Human. Pre-registered clipped Ridge: D 87.9 < B 111.0 (11/17), D < E 115.4 (11/17). Dataset-first kNN: D 78.5 ≈ B 79.0, E 73.5 best. Same shape as D4RL under Phase 25 predictors
- 26b vocab: 2500-word sections needed `num_ctx 8192` (app.py, semantic_type.py; default context silently truncated — old server had to be killed, it was still bound to :8000). Shared tuples 11→19 but ops/paper stay ~4; MAE unchanged. Bottleneck = ops emitted per paper, not section length
- SimPLe/DER re-pinned; superseded records in phase26/superseded_records. `results26.md`
