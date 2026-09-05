# Method Scanner

**Can you predict how a machine-learning experiment will turn out by reading the papers — before you run it?**

Kat Laszlo · August 2026 · 30 phases · 58-paper core corpus · three outcome benchmarks · one answer

---

## The question

Machine-learning experiments are expensive. Finding out whether an idea works can cost days of compute, and most of the time the answer is no. Meanwhile, thousands of papers already describe what worked and what didn't. So:

> **If you can read what a method actually does from its paper, can you predict how well it will score — before running it?**

If yes, that's worth a lot. It's also the kind of claim that's easy to fake, because a prediction made after seeing the result isn't a prediction. So every test in this project was written down in advance, and the results are reported whether or not they came out the way I hoped.

**The answer is no.** Six pre-registered attempts, three benchmarks, up to 900 papers and 5,000 head-to-head comparisons. A method's structure does not predict its score beyond what the experimental settings and the benchmark's identity already tell you, and what little it does know doesn't survive moving to a new task.

That's a real answer to a real question. Getting to it also produced a way of comparing papers that holds up on its own, one prediction that *did* hold, and a founding hypothesis that the tooling built to test it ended up rejecting. All of that is below.

## The first approach: sort papers by name

The obvious way to predict "will this work" is to reason from what a method *is*. This is a diffusion model. That's a Q-learning method. These two are in the same family, so they'll behave alike. That's where this started: group papers by the names they go by, and predict from the group.

It broke almost immediately, in both directions. Two papers filed under completely different names turned out to run nearly the same loop inside. And worse, **two papers under the same name could share almost nothing.** "Predictive coding" covers both a 1999 model of the visual cortex and a 2022 replacement for backpropagation. "Reinforcement learning" covers methods with no update rule in common.

A name is a family tree, not a mechanism. As a predictor it's not just weak, it's misleading — it groups things that behave differently and separates things that behave the same. The input had to change.

## The new approach: read the machinery

An AI model running locally on a laptop reads the part of a paper where the authors explain their actual method — not the abstract — and rewrites it as a list of things the method *does*. Each item is a verb plus the thing the verb acts on:

```
predict @ state          guess where a physical system will be next
correct @ parameter      adjust the model's weights
sample @ trajectory      draw one possible path through the future
```

Papers are then compared by how much these lists overlap. Same lists, same machinery, whatever the papers call themselves.

Getting this to work reliably took most of the project. It also produced the first finding: **the verb tells you nothing; the object tells you everything.** Almost every paper "predicts" something, so that word carries no information alone. But predicting *a physical state*, predicting *an internal representation*, and predicting *a model's own weights* are three unrelated activities that happen to share a word. `predict@state` is filtering. `predict@latent` is self-supervised learning. `predict@parameter` is meta-learning. The vocabulary you have to grow as you meet new fields is the vocabulary of **objects**, not verbs.

## What I found

### The original hypothesis dissolved

The project began from a specific version of the question: do Kalman filtering and predictive coding share a mechanism? Control theorists, self-supervised-learning researchers and neuroscientists all describe a *predict, then correct* loop in their own field, so the analogy is widely assumed.

By verbs alone, KalmanNet ties at a perfect 1.00 with three predictive-coding papers — exactly what everyone expects. Two checks killed it:

- **An ablation.** Remove the word `predict` and the remaining one-verb `correct` link ties KalmanNet just as strongly to **Adam, an optimizer**. A hub that connects equally well to a control is not a hub.
- **Typing.** KalmanNet corrects `@state`. The predictive-coding papers correct `@parameter`. Same verb, different object, different mechanism.

The Kalman–predictive-coding analogy is a story about the English word *correct*, not about a shared loop. The hypothesis the project was built to test is the one its own method rejected.

### Every family label split or dissolved

- **"Predictive coding" is two families.** One half predicts *inputs* and updates *neural state*; the other predicts *latents* and updates *parameters*. By verbs they look like one family (0.55 similarity). Typed, they're two, with almost nothing between them (0.06). Split that way, both halves cohere.
- **"Filters" is three families.** Each filtering paper describes a *different piece* of the same loop — the recursion, the association step, the resampling, the training procedure. They look incoherent as a group because they aren't describing the same stage.
- **"Reinforcement learning" splits, but not where the textbooks say.** The expected division is policy-gradient vs value-based. The data splits it **on-policy vs replay-based** instead: DDPG, policy-gradient by every textbook, types as a replay loop with an actor attached (0.36 with DQN, 0.00 with PPO). SAC, off-policy by lineage, types cleanly with the on-policy group.
- **"Latent-generative" isn't a family.** Five VAE-with-dynamics papers produced five different signatures and 0.00 against each other at every corpus size. The label was dropped, and two papers are recorded as orphans rather than forced into a group.

This gave a diagnostic: **looking coherent by verbs while incoherent by objects is the signature of a family name** — papers held together by shared vocabulary rather than shared mechanism.

### Two bridges survived every test

- Self-supervised latent prediction ↔ predictive coding as a backprop alternative, on `predict@latent + correct@parameter`. The strongest pair in the corpus.
- Particle filters ↔ Kalman-style state estimation, on `predict@state + correct@state`. The only pair that survives requiring both independent typing passes to agree — an accidental positive control.

### Papers that build on other papers don't restate the loop

DDQN, TD3 and Rainbow inherit their agent loop by citation and write only their own contribution. Extracted from their own text they score ~0 against each other *and* against the papers they extend. Pasting the base paper's algorithm in front of each one fixed it (one family went 0.05 → 0.21). The control that keeps this honest: doing the same to a paper that *already* prints its full loop changed nothing (0.40 → 0.40). The injection surfaces something missing; it doesn't inflate scores.

### Five live predictions on unseen papers: three right, two wrong

The two wrong ones are the informative ones. A paper with "predictive coding" in its title ran a JEPA-style loop — the title was lineage, not mechanism. A Bayesian-optimization paper was correctly refused a match *because* of typing, after scoring 0.67 with predictive coding by verbs alone. The pattern: the system reliably refuses false family matches, finds the right neighbour when its object vocabulary covers the field, and goes blank rather than guessing when it doesn't.

## The answer: no

Back to the question. Given a method's machinery and its experimental settings, can you predict **what it will score**?

Six attempts, each written down before running, on three benchmarks:

| phase | question asked | answer |
|---|---|---|
| **24** | On robot-control benchmarks (D4RL), does machinery predict the score better than the experimental settings alone? | **No.** Structure added nothing the settings didn't already carry. |
| **25** | Does fixing the statistical model change that? | **Barely.** A small consistent gain with one predictor that vanishes with a stronger one, because knowing *which benchmark* a row came from already explains most of the score. |
| **26** | Same test on a second benchmark (Atari 100k), pre-registered. | **Mixed.** Machinery helped under one predictor; the plain family label did as well or better under the other. |
| **27** | Was the extraction too thin? Re-ran it to pull ~15 items per paper instead of ~4. | **No, and this closes the excuse.** Five times more shared machinery, same answer. What structure knows about outcomes is shallow — roughly "what kind of method is this," which the hand-written label already said. |
| **28** | Forget whole methods. If a paper changes *one thing*, does the machinery of that change predict whether the score goes up or down? | **No.** No better than guessing the average. |
| **30** | 900 papers, 25 public leaderboards, ~5,000 head-to-head comparisons: can machinery pick which paper scores higher? | **Only where it can't help.** 87% accurate *within* a leaderboard it has already seen; 60% on a new one, barely above a coin flip. Publication year alone gets 71% and transfers fine. |

The last row explains the rest. The extracted vocabulary is **task-specific**: papers on image classification use different operations than papers on object detection. The scanner reads something real, but its predictive power lives inside each field's local dialect and doesn't travel.

That's the answer. It's not the one I was hoping for, and it's a good one to have: it says where the ceiling is, and it says why.

## What does work: predicting a transfer

Scores are the wrong thing to ask this method for. There is a different prediction it makes well, and it was tested the same way — written down first, then run.

The scanner was given **Diffuser**, a robotics paper that plans robot motion, from a field not in the corpus. Its top match was **DDIM**, an image-generation sampler. Different fields, different conferences, no shared citations, same extracted inner loop. (A reviewer asked to guess in advance predicted world-model or Kalman-filter papers. The scanner's answer was the one that held.)

If they really share machinery, a shortcut invented for the image paper should work in the robotics paper. So:

> **Prediction, written before running anything:** DDIM's deterministic sampler, at 5–10× fewer denoising steps, will keep Diffuser's D4RL benchmark score within about 5% of baseline.

**How it was run:**

1. Cloned the original Diffuser repository **unmodified** and reproduced its baseline first.
2. Wrote DDIM's sampler as a single **patch** (`phase23/ddim_patch.diff`) — the only change to the codebase, so nothing else could explain a difference.
3. Three simulated robots (halfcheetah, hopper, walker2d) × three random seeds × five step counts. **45 episodes.**
4. **Every run paired against the baseline on the same seed**, so comparisons are like-for-like rather than against an average.
5. Raw logs for every individual run are in `phase23/`, not just the summary table.

**Result:**

| shortcut | outcome |
|---|---|
| **2× fewer steps** | held on every robot |
| **4× fewer steps** | landed on the 5% line to the decimal |
| **10× fewer steps** | failed — as the prediction's own upper bound implied |

And one thing the scanner never predicted: on walker2d the deterministic sampler is *worse* than the random one even at full step count. Recorded as an open question, not folded into the result.

The claim is not "Diffuser can be made 4× faster." It's that a program reading English text produced a specific, falsifiable engineering prediction, and the experiment landed inside the bounds it stated in advance.

## Where the line is

- **Predicting that a component will transfer between two methods that share machinery — works.** One falsifiable prediction, bounds held.
- **Predicting the number a method will score — doesn't.** Not beyond what the settings and the benchmark already tell you, and not across tasks.

Structure answers "these are the same kind of thing, so this should work here." It does not answer "this will score 84."

**Known limits**, stated rather than buried: 3–9 papers per family, so small families move on a single paper. Two typing passes agree 72–84% of the time — family-level results are stable, individual pair scores are fragile. The method-section finder misses ~10% of papers. Deciding which papers inherit their loop from which base paper is still manual. One model, one prompt family, no independent human rater yet.

## How it works

Four layers. Each is built from the one below it and can be redone without re-running the expensive parts. Raw model output is never overwritten.

| layer | what happens |
|---|---|
| **Extract** | arXiv ID → PDF → plain text → a parser hunts down the real method section (four PDF layouts handled, ~90% hit rate) → a local `qwen3:8b` model returns the operations, the loss function, and the stated failure mode |
| **Canonicalize** | raw verbs mapped to a fixed 23-word vocabulary so *forecast* and *predict* stop counting as different. Computed on every read, never saved — changing the vocabulary updates the whole corpus instantly and reversibly |
| **Type** | a second pass assigns each operation its object (`state`, `latent`, `parameter`, `error`, …) and must quote the sentence that justifies it |
| **Compare** | overlap between papers, per pair and per family |

Three things that mattered more than anything else, all learned the hard way:

- **Examples beat rules.** Telling a small model "do not list *derive*, *formulate*" was ignored. Three or four worked examples fixed the same problem immediately.
- **Include an example where the answer is "nothing."** Given only examples where a failure mode existed, the model invented one for papers that had none, copying the example's wording onto unrelated papers. One example with an explicit empty answer cut that to 1 in 30; a check at read time catches the rest.
- **Never show the model the vocabulary list.** A closed-vocabulary prompt was tested and rejected: the model started describing the list instead of the paper. A paper that explicitly avoids contrastive learning got labeled `contrast`.

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
| `results/` | one record per paper — raw model output, prompt, typing — kept forever |
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
python3 fulltext.py   # batch: arXiv IDs in, method sections out
```

Roughly 40 seconds per method section on a laptop.

The score-prediction work needs real dependencies:

```
pip install -r outcome_predictor/requirements.txt
```

`phase23/` reproduces the robotics experiment against an unmodified Diffuser checkout in its own environment — see `phase23/phase23_plan.md`.

## Status

A research log, not a maintained tool. It's public so the question, the approach, and the answer — including the six results that came back negative — can be read and checked by someone else.

MIT licensed. Reuse the code, the data, or the method freely; attribution appreciated, not required. The license covers this repo's code and derived data, not the source papers, which are third-party work quoted under fair use.
