# Phase 24 — Outcome predictor on D4RL locomotion

Tests whether Method Scanner's `operation@object` structure carries predictive information about
experimental outcomes, beyond experimental conditions and beyond conventional lineage labels.

```
published D4RL papers  ──►  structure (scanner) + conditions (paper) ──►  outcome (normalized return)
                                     leave-one-method-out  ──►  MAE per model A–E
```

## Files

| file | what |
|---|---|
| `d4rl_dataset.csv` | one row = exact method + exact configuration + exact reported normalized return, with provenance |
| `schema.md` / `schema.json` | frozen feature schema, encodings, distances, alpha grid, holdout groups |
| `build_dataset.py` | assembles the CSV from `sources/tables/*.json`, scanner records, and Phase 23 |
| `predictor.py` | Models A–E, Ridge (primary) and kNN k=5 (robustness) |
| `evaluate.py` | leave-one-method-out evaluation; writes `results.json` |
| `results.md` | results, verdicts, caveats |
| `d4rl_papers.txt` | the 28 papers pulled for this phase |
| `sources/tables/*.json` | per-paper table and hyperparameter extractions (each number carries its verbatim source line) |
| `sources/build_audit.json` | per-method counts of included / excluded rows and why |
| `sources/EXTRACT_INSTRUCTIONS.md` | the extraction protocol given to the table extractors |
| `run_scanner*.sh`, `scanner_run.log` | how the new papers were pushed through the existing scanner pipeline |
| `superseded_records/` | scanner records replaced by pinned-section re-extractions (kept for audit) |

## Reproduce

```
venv/bin/python outcome_predictor/build_dataset.py     # -> d4rl_dataset.csv
venv/bin/python outcome_predictor/evaluate.py          # -> results.json + console table
```

Structure extraction needs `app.py` on :8000 and ollama `qwen3:8b` (see `run_scanner.sh`).

## Follow-up phases (same directory)

| dir | what |
|---|---|
| `phase25/` | exploratory: clipped Ridge, dataset-first kNN, relaxed/bare vocab, within-dataset ranking (`results25.md`) |
| `phase26/` | pre-registered second benchmark Atari 100k (`atari100k_dataset.csv`, `results26.md`) + long-section vocabulary test (`vocab_results.json`) |
