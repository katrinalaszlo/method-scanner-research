# Phase 30 — pairwise outcome prediction on Papers-with-Code leaderboards (pre-registration, 2026-08-26)

Prior work: Wen et al. 2025 (arXiv 2506.00794) and "Teaching LMs to Forecast Research Success" (arXiv 2605.21491)
predict which of two ideas scores higher on a benchmark: fine-tuned 8B/GPT-4.1 ≈ 77%, zero-shot frontier ≈ 61% or chance,
cross-domain 45–49%. Neither decomposes the signal. Question here: how much of pairwise predictability is year trend +
lineage + abstract text, and does Method Scanner structure add anything on top — at n in the thousands.

## Data (pwc-archive snapshot, `pwc/`)
- Leaderboard = (task, dataset). Primary metric = first metric listed. Direction: lower-is-better if the metric name
  matches /error|loss|perplex|ppl|fid|mae|rmse|mse|wer|cer|distance|latency|time|params|flops|epe|abs rel|sq rel|lpips|nll|bits|kid/,
  else higher-is-better. Scores parsed as the first float in the cell.
- Boards kept: ≥ 15 distinct papers with an arXiv abstract and a parseable score; at most 3 boards per task
  (largest first) for breadth; boards taken in descending size until ≤ 900 distinct papers are covered.
- Pairs: all within-board pairs of distinct papers with different scores; label = higher-scoring paper; each pair
  presented in both orders (antisymmetric features), per board capped at 200 pairs (random, seed 0).

## Features (per paper; pair feature = x_A − x_B)
- T  year: paper_date year (float) — trend baseline.
- L  lineage: PwC method tags on the paper (`papers-with-abstracts.methods`, mapped to method collections via the
  `methods` table) — multi-hot.
- E  text: `nomic-embed-text` embedding of title + abstract (768-d).
- S  structure: Method Scanner run on the abstract with the Phase 27 full-operation prompt (closed ontology),
  strict `op@object` multi-hot. Extraction outputs not edited.
- Sets compared: chance, T, L, E, S, T+L, T+E, T+S, T+L+S, T+L+E, T+L+E+S.

## Model and evaluation
- Logistic regression (L2; C ∈ {0.01, 0.1, 1, 10} by grouped 5-fold CV inside training). Pairs duplicated in both orders.
- Two holdouts, both reported: (i) leaderboard-held-out (GroupKFold by board, 5 folds) — the transfer case;
  (ii) within-board (pairs split 5-fold at random; both orders of a pair stay in the same fold).
- Metric: accuracy; also per-board accuracy mean. H1: acc(T+L+S) > acc(T+L) on (i). H2: acc(S) > 0.5 on (i).
  H3: acc(T+L+E+S) > acc(T+L+E). Nothing chosen after seeing test accuracy.

## Amendments (recorded before any model was fit)
1. **Board selection.** The rule above ("descending size until ≤ 900 papers") selected 8 boards of 100–340 papers each,
   which makes board-held-out CV meaningless. Amended: boards with more than 40 papers are randomly subsampled to 40
   (seed 0) before the budget check. Result: 25 boards, 17 tasks, 900 papers, 5000 pairs (200 per board).
   Sub-datasets in the PwC tables are treated as their own boards (`dataset / subdataset`).
2. **Join key.** Leaderboard rows link `arxiv.org` URLs, not PwC paper URLs; papers are matched on arXiv id (version
   stripped). Rows are streamed from `evaluation-tables-*.parquet` (the flattened JSON is 9.8 GB and does not fit in memory).
3. **Lineage source.** `papers-with-abstracts.methods[].main_collection.name` is used directly (no join to the `methods`
   table). Tags exist for 500/900 papers; the snapshot's method table is partly polluted (a spam entry maps to the
   collection "2D Parallel Distributed Methods", which is by far the most frequent tag). Kept as-is: it is the
   conventional lineage baseline the snapshot offers, and pollution can only hurt L, not S.
4. **Structure input.** Scanner runs on title + abstract (median 175 words) with the Phase 27 prompt, `num_ctx` 4096.
