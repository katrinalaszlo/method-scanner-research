# D4RL table + hyperparameter extraction (Phase 24)

You are extracting auditable experiment rows from ML papers. Paper texts live in
`/Users/katlaszlo/Desktop/Desktop - Kat’s MacBook Air/Github-Wiki/GitHub/method_scanner/papers/<arxiv_id>.layout.txt`
(pdftotext -layout, best for tables) and `<arxiv_id>.txt` (reflowed, best for prose). Read them with grep/sed/cat. Do NOT use the PDF or the web; do NOT fill anything from memory of the paper. Every number must be copied from the text file with the verbatim line it came from.

For each assigned paper write `<SCRATCH>/tables/<arxiv_id>.json` with this shape:

```json
{
  "arxiv_id": "...",
  "title": "...",
  "methods": [
    {
      "method_name": "exact name the paper uses for its own method/variant, e.g. 'Diffuser', 'EDAC', 'SAC-N', 'TT (quantile)'",
      "is_own_method": true,
      "one_paragraph_description": "what the algorithm does at runtime, in the paper's own words (paraphrase ok)",
      "conventional_lineage_hint": "the research category the paper itself places the method in, quoting the paper (e.g. 'offline RL', 'model-based offline RL', 'sequence modeling', 'diffusion-based planning', 'behavior cloning')",
      "global_conditions": {
        "<hyperparameter_name>": {"value": "...", "quote": "verbatim line", "location": "Table X / Appendix Y / Section Z"}
      },
      "per_dataset_conditions": {
        "<dataset name>": {"<hyperparameter_name>": {"value": "...", "quote": "...", "location": "..."}}
      },
      "evaluation_protocol": {"seeds": "...", "episodes_per_seed": "...", "checkpoint_selection": "last / best / ...", "quote": "...", "location": "..."},
      "dataset_version": {"value": "v0 | v2 | unknown", "quote": "verbatim line that says which D4RL version", "location": "..."},
      "rows": [
        {
          "dataset": "exact dataset name as printed (e.g. 'halfcheetah-medium-replay')",
          "outcome": 83.5,
          "std": 1.2,
          "outcome_is_mean_over_seeds": true,
          "config_label": "main | <ablation label such as 'horizon=16', 'K=1', 'w=1.2', 'N=10 samples'>",
          "config_conditions": {"<hyperparameter_name>": "value"},
          "table": "Table 1",
          "line_quote": "the verbatim line from the layout.txt containing this number",
          "notes": "any comparability caveat (e.g. score is in raw return not normalized; value comes from figure not table -> then EXCLUDE)"
        }
      ]
    }
  ],
  "problems": ["anything unreadable, ambiguous, or garbled — list it instead of guessing"]
}
```

Rules:
1. Only the paper's OWN method(s) and their explicitly-labelled variants. Do not extract baseline columns (CQL, BC, etc. reported by another paper) — EXCEPT record them in a separate top-level key `"baseline_rows_seen": ["list of baseline column names present"]` so we know they exist.
2. Only D4RL Gym-MuJoCo locomotion datasets: halfcheetah / hopper / walker2d × random / medium / medium-replay / medium-expert / expert / full-replay. Skip antmaze, kitchen, adroit, maze2d.
3. A row = one exact configuration on one dataset. If the main table has one column per method, that column is config `main`. If an ablation table re-runs the method under a different explicit setting on the same datasets (e.g. horizon sweep, guidance weight sweep, context length K sweep, number of candidates, number of critics N, denoising steps), extract those as additional rows with `config_label` and `config_conditions` filled. Only extract ablations where the varied setting is explicitly named with a value.
4. Averages across datasets ("Average"/"Total") are NOT rows. Skip them.
5. If the table in layout.txt is garbled so a number cannot be tied unambiguously to its dataset and column, put the row in `problems` and do NOT emit it. Never guess. Never fill from memory.
6. Outcome must be the normalized D4RL score (0–100 scale). If a paper reports raw returns, note it and still extract, but set `"notes": "raw return"`.
7. Hyperparameters: harvest everything the paper reports about the run configuration — network sizes, hidden dim, layers, batch size, learning rate, training steps/epochs/gradient steps, discount, horizon, denoising/diffusion steps, sampler, guidance scale/weight, number of samples/candidates, ensemble size / number of Q-networks, context length, temperature, expectile, target return, alpha, tau, BC weight, model rollout length, penalty coefficient, etc. Use snake_case names. If a value varies per dataset, put it in `per_dataset_conditions`. Always quote the line.
8. Record how many seeds / evaluation episodes the reported number averages over, with quote.
9. Find the dataset version (v0 vs v2) — grep for "-v2", "-v0", "v2", "version". If not stated, "unknown".
10. When done, print a 5-line summary: number of methods, number of rows, dataset version, main problems.

Work carefully; correctness over speed. Read the whole results section and appendix hyperparameter tables. Use `grep -n -i "hopper\|halfcheetah\|walker" <file>` to locate tables, then `sed -n` around them.
