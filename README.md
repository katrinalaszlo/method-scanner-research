# Method Scanner

**Can you predict how a machine-learning experiment will turn out by reading the papers — before you run it?**

Kat Laszlo · August 2026 · 30 phases · 58-paper core corpus · three outcome settings

---

## The question

Machine-learning experiments are expensive. Finding out whether an idea works can cost days of compute, and most of the time the answer is no. Meanwhile, thousands of papers already describe what worked and what didn't. So:

> **If you can read what a method actually does from its paper, can you predict how well it will score — before running it?**

The project tests whether extracted descriptions of a method add predictive information beyond experimental settings, method labels, and other baselines. Plans, raw outputs, and unsuccessful experiments are preserved alongside the results. The work includes both planned tests and exploratory follow-ups.

**Across the tested benchmarks, structural features did not consistently improve outcome prediction beyond strong baselines.** This is a result about the representations, models, and datasets tested here, not evidence that predicting outcomes from method descriptions is impossible.

The work also produced an inspectable way to compare papers and a component-transfer experiment with encouraging results at 2× fewer denoising steps. Evaluation and extraction issues identified during review remain unresolved; they are described below so the reported results can be interpreted in context.

## The first approach: sort papers by name

The obvious way to predict "will this work" is to reason from what a method *is*. This is a diffusion model. That's a Q-learning method. These two are in the same family, so they'll behave alike. That's where this started: group papers by the names they go by, and predict from the group.

It broke almost immediately, in both directions. Two papers filed under completely different names produced similar extracted operation lists. And worse, **two papers under the same name could share almost nothing.** "Predictive coding" covers both a 1999 model of the visual cortex and a 2022 replacement for backpropagation. "Reinforcement learning" covers methods with no update rule in common.

A name is a family tree, not a mechanism. As a predictor it's not just weak, it's misleading — it groups things that behave differently and separates things that behave the same. The input had to change.

## The new approach: read the machinery

For the core corpus, an AI model running locally on a laptop reads a text window selected around the method section and rewrites it as a list of things the method *does*. The largest outcome experiment, Phase 30, instead uses titles and abstracts. Each item is a verb plus the thing the verb acts on:

```
predict @ state          guess where a physical system will be next
correct @ parameter      adjust the model's weights
sample @ trajectory      draw one possible path through the future
```

Papers are then compared by how much these lists overlap. Overlap is evidence of similar described operations; it does not establish that the algorithms are equivalent.

A useful finding in this corpus was that **the object makes a shared verb more informative.** Predicting a physical state, an internal representation, and model parameters can describe different computations. Typing the objects helps distinguish them, although a short operation list still omits equations, ordering, and implementation details.

## What I found

### The original analogy weakened under typing

The project began from a specific version of the question: do Kalman filtering and predictive coding share a mechanism? Control theorists, self-supervised-learning researchers and neuroscientists all describe a *predict, then correct* loop in their own field, so the analogy is widely assumed.

By verbs alone, KalmanNet ties at a perfect 1.00 with three predictive-coding papers — exactly what everyone expects. Two checks weakened that evidence:

- **An ablation.** Remove the word `predict` and the remaining one-verb `correct` link ties KalmanNet just as strongly to **Adam, an optimizer**. A hub that connects equally well to a control is not a hub.
- **Typing.** KalmanNet corrects `@state`. The predictive-coding papers correct `@parameter`. The shared verb masks a difference in the extracted objects.

The verb-only match did not survive these checks as strong evidence for a shared mechanism. That limits what this representation supports; it does not rule out a mathematical relationship between the methods.

### Family labels did not consistently match extracted structure

- **"Predictive coding" split into two groups in this corpus.** One half predicts *inputs* and updates *neural state*; the other predicts *latents* and updates *parameters*. By verbs they look like one family (0.55 similarity). Typed, they're two, with almost nothing between them (0.06). Split that way, both halves cohere.
- **"Filters" split into three groups.** Each filtering paper describes a *different piece* of the same loop — the recursion, the association step, the resampling, the training procedure. They look incoherent as a group because they aren't describing the same stage.
- **"Reinforcement learning" splits, but not where the textbooks say.** The expected division is policy-gradient vs value-based. The data splits it **on-policy vs replay-based** instead: DDPG, policy-gradient by every textbook, types as a replay loop with an actor attached (0.36 with DQN, 0.00 with PPO). SAC, off-policy by lineage, types cleanly with the on-policy group.
- **"Latent-generative" did not cohere under this representation.** Five VAE-with-dynamics papers produced five different signatures and 0.00 against each other at every corpus size. The label was dropped, and two papers are recorded as orphans rather than forced into a group.

This suggests a diagnostic worth testing further: high verb overlap with low typed overlap can flag a family label that needs closer inspection. Missing context and extraction errors can also produce low overlap.

### Two notable structural matches

- Self-supervised latent prediction ↔ predictive coding as a backprop alternative, on `predict@latent + correct@parameter`. The strongest pair in the corpus.
- Particle filters ↔ Kalman-style state estimation, on `predict@state + correct@state`. The only pair that survives requiring both typing passes to agree — an accidental positive control.

### Papers that build on other papers don't restate the loop

DDQN, TD3 and Rainbow inherit their agent loop by citation and write only their own contribution. Extracted from their own text they score ~0 against each other *and* against the papers they extend. Pasting the base paper's algorithm in front of each one fixed it (one family went 0.05 → 0.21). The control that keeps this honest: doing the same to a paper that *already* prints its full loop changed nothing (0.40 → 0.40). This control supports the interpretation that inheritance can recover missing context, though it does not establish that injection never inflates similarity.

### Five live predictions on unseen papers: three right, two wrong

The two wrong ones are the informative ones. A paper with "predictive coding" in its title ran a JEPA-style loop — the title was lineage, not mechanism. A Bayesian-optimization paper was correctly refused a match *because* of typing, after scoring 0.67 with predictive coding by verbs alone. These five cases suggest that typing can reject misleading matches and that vocabulary coverage matters. They are too few to establish reliable retrieval performance.

## Outcome prediction: no consistent gain beyond strong baselines

Given a method's extracted operations and experimental settings, can the system predict **what it will score**?

Six phases examined this question on D4RL, Atari 100k, and a collection of public leaderboards. Phase 25 was an exploratory follow-up; the phase documents record the other plans and amendments.

| phase | question asked | reported result |
|---|---|---|
| **24** | On D4RL, does structure improve prediction beyond experimental settings? | No improvement beyond the settings baseline. |
| **25** | Does changing the statistical model help? | A small gain with one predictor vanished with a stronger baseline. |
| **26** | Does the result hold on Atari 100k? | Mixed: structure helped under one predictor; family labels did as well or better under the other. |
| **27** | Does extracting more operations help? | Denser extraction did not establish a consistent advantage over the tested baselines. |
| **28** | Does the structure of a within-method change predict its score delta? | No improvement over the mean baseline in this test. |
| **30** | Can structure rank papers on public leaderboards? | Structure alone reached 87.3% on held-out pairs within boards and 60.3% on held-out boards. Year alone reached 71.1% on held-out boards; adding structure did not improve it. |

Phase 30 selected 900 papers and 5,000 pairs across 25 boards. After seven extraction failures, the saved evaluation covers **893 papers and 4,927 pairs**. It uses five-fold grouped cross-validation by board, not leave-one-board-out evaluation. See the [saved metrics](outcome_predictor/phase30/results30_partial.json) and [evaluation code](outcome_predictor/phase30/eval30.py).

**The within-board result is for additional comparisons among mostly known papers.** The split holds out pairs, not papers: in a check of the saved data, 4,922 of 4,927 test pairs contained two papers that also appeared in training pairs. The 87.3% result therefore does not establish prediction accuracy for unseen methods. A paper-disjoint evaluation is needed. Holding out boards also does not guarantee that papers or tasks are disjoint.

Task-specific vocabulary is one possible explanation for the performance gap, but these splits do not isolate it from familiarity with individual papers. The experiments support a narrower conclusion: the tested structural features did not consistently add predictive value beyond strong baselines.

## Component transfer: encouraging at 2× fewer steps

The scanner matched **Diffuser**, a robotics planning paper, with **DDIM**, an image-generation sampler. That match motivated a concrete engineering test: replace Diffuser's stochastic sampler with DDIM's deterministic sampler and reduce the denoising steps.

The earlier proposal described **5–10× fewer steps**. The [Phase 23 plan](phase23/phase23_plan.md) states a broader 2–10× hypothesis, with a primary criterion at **2× fewer steps**: at least two of three environments must retain 95% of baseline mean score. Meeting that criterion does not validate the earlier 5–10× claim.

**How it was run:**

1. Used the original Diffuser checkout and pretrained checkpoints, with changes recorded in [the patch](phase23/ddim_patch.diff).
2. Re-ran DDPM at 20 steps as the baseline, then tested DDIM at 20, 10, 5, and 2 steps.
3. Ran three environments (halfcheetah, hopper, walker2d) × three seeds × five sampler configurations: **45 episodes**.
4. Reported environment means and comparisons paired against the baseline on the same seed.
5. Used Gymnasium v4 dynamics with the original D4RL v2 datasets for both samplers. This shared environment adaptation limits direct comparison with the original paper's absolute scores.

**Observed results:**

| shortcut | outcome |
|---|---|
| **2× fewer steps** | All three environments retained at least 95% of baseline mean score, meeting the primary criterion. |
| **4× fewer steps** | Mixed: hopper and walker2d improved on average; halfcheetah's mean paired drop was **5.14%**, just outside the 5% tolerance. |
| **10× fewer steps** | Failed the tolerance on halfcheetah and walker2d; hopper improved on average. |

At full step count, deterministic DDIM also performed worse on walker2d: the mean paired drop was about 20%. Removing stochasticity was not uniformly beneficial.

The [environment means](phase23/results_table.csv), [paired results](phase23/results_paired.csv), and raw logs in `phase23/` preserve these outcomes. With only three seeds per environment and substantial variation in some conditions, this is an encouraging transfer case at 2× fewer steps, not a general validation of transfer prediction. Fewer denoising steps also do not imply an equal reduction in wall-clock runtime.

## Limits and issues to resolve

- **Extraction boundaries:** The method-section finder selects a starting heading but then takes a fixed word window. It can include later experimental results. Saved inputs need an audit for outcome leakage, and section boundaries need to be enforced before re-running affected experiments.
- **Leaderboard aggregation:** Phase 30 keeps the highest score when a paper has multiple entries, even for lower-is-better metrics. The evaluated data includes absolute relative error. The affected entries and labels need to be checked; the impact on reported accuracy has not been quantified.
- **Evaluation splits:** The within-board test reuses papers across training and test pairs. Add paper-disjoint and task-disjoint evaluations before making claims about new methods or new tasks.
- **Representation and annotation:** Core families contain only 3–9 papers. Two typing passes agree 72–84% of the time, but they use the same model with different prompts and measure prompt sensitivity, not independent annotation agreement. There is no independent human rater yet.
- **Input coverage:** Method-section selection was reported to miss about 10% of papers. Inherited loops are assigned manually, and Phase 30 uses titles and abstracts rather than method sections.
- **Scope:** These results concern the tested operation vocabulary, extraction models, predictors, and datasets. They do not establish a ceiling on all possible representations of method structure.

The metrics above are the recorded results, not corrected re-evaluations. These issues remain open in the code. Historical writeups and phase reports retain their original wording; this README qualifies their conclusions in light of the review.

## How it works

Four layers. Each is built from the one below it and can be redone without re-running the expensive parts. Records include raw model output and prompts; historical snapshots preserve selected earlier runs. The UI re-run action replaces the previous record, so archive records before re-running them if a complete history is needed.

| layer | what happens |
|---|---|
| **Extract** | arXiv ID → PDF → plain text → a parser selects a window starting near the method section (with the boundary limitations noted above) → a local `qwen3:8b` model returns the operations, the loss function, and the stated failure mode |
| **Canonicalize** | raw verbs mapped to a fixed 23-word vocabulary so *forecast* and *predict* stop counting as different. Computed on every read, never saved — changing the vocabulary updates the whole corpus instantly and reversibly |
| **Type** | a second pass assigns each operation its object (`state`, `latent`, `parameter`, `error`, …) and must quote the sentence that justifies it |
| **Compare** | overlap between papers, per pair and per family |

Three things that mattered more than anything else, all learned the hard way:

- **Examples beat rules.** Telling a small model "do not list *derive*, *formulate*" was ignored. Three or four worked examples fixed the same problem immediately.
- **Include an example where the answer is "nothing."** Given only examples where a failure mode existed, the model invented one for papers that had none, copying the example's wording onto unrelated papers. One example with an explicit empty answer cut that to 1 in 30; a check at read time catches the rest.
- **Vocabulary constraints can backfire.** An early closed-vocabulary prompt was tested and rejected: the model started describing the list instead of the paper. A paper that explicitly avoids contrastive learning got labeled `contrast`. Later phases revisited closed-ontology extraction, so this is a prompt-design observation rather than a universal rule.

## Where everything is

**Start here:**

| file | what |
|---|---|
| `writeup_final.md` | **the full writeup** — problem, method, corpus, findings, the Diffuser/DDIM experiment, limitations |
| `findings.md` | every result and every rejected approach, phase by phase, including the ones that didn't work |
| `progress.md` | session log — what ran, what broke, what got decided |
| `task_plan.md` | the phase plan |

**The pipeline:**

| file | what |
|---|---|
| `app.py` | server + web UI on `:8000`; the extractor, the vocabulary, the read-time checks |
| `fulltext.py` | arXiv → PDF → method section (`--from` pins a section by hand when the parser guesses wrong) |
| `semantic_type.py` | the typing pass |
| `compare.py` | similarity between papers and families |
| `phase21_inject.py`, `phase22_split.py` | for papers that are deltas on a base method: inherit the base loop, then separate inherited machinery from the paper's own |
| `families.json`, `semantic_ontology.json` | the family labels and the object list |

**Output:**

| path | what |
|---|---|
| `results/` | extraction records with raw model output, prompt, and typing |
| `matrices/` | every similarity matrix computed, per phase |
| `similarity_matrix.csv` | paper-vs-paper overlap on **verbs alone** |
| `semantic_similarity_matrix.csv` | the same on **verb + object** — the one that carries the finding |
| `results_phase*_snapshot/` | records as they stood before each re-extraction |
| `phase23/` | the Diffuser/DDIM experiment — patch, sweep scripts, raw logs, paired results, plot |
| `outcome_predictor/` | phases 24–30, the score-prediction work (has its own README) |
| `archive/prototypes/` | dead early scripts, kept only because `findings.md` cites them as evidence |

## Running it

The core pipeline is **Python standard library only** — no install step. It needs `pdftotext` (from poppler) and [Ollama](https://ollama.com) with `qwen3:8b` pulled.

```
python3 app.py        # web UI at http://localhost:8000
python3 fulltext.py papers.txt  # batch extraction; requires the server above
```

Roughly 40 seconds per method section on a laptop.

The score-prediction work needs real dependencies:

```
pip install -r outcome_predictor/requirements.txt
```

`phase23/` contains the robotics experiment, including the Diffuser patch and environment adaptations — see [the experiment plan](phase23/phase23_plan.md).

## Status

A research log, not a maintained tool. It is public so the methods, positive and negative results, and unresolved issues can be inspected and checked by someone else.

MIT licensed. Reuse the code, the data, or the method freely; attribution appreciated, not required. The license covers this repo's code and derived data, not the source papers, which are third-party work quoted under fair use.
