# Method Scanner

**Comparing machine-learning papers by what their methods actually do, instead of by what they're called.**

Kat Laszlo · August 2026 · 30 phases · 58-paper core corpus + three outcome benchmarks

---

## The problem

Methods in machine learning get named after their family tree — *Kalman filter*, *JEPA*, *predictive coding*, *reinforcement learning*. Those names tell you where a method came from, not what it does.

That turns out to matter. Two papers filed under completely different names can run nearly the same loop inside. Two papers under the *same* name can share almost nothing. "Predictive coding" covers both a 1999 model of the visual cortex and a 2022 replacement for backpropagation. "Reinforcement learning" covers methods with no update rule in common.

So if you want to know whether two methods are really related, the name is close to useless. You have to read the machinery.

## What this does

It reads the machinery automatically.

An AI model running locally on a laptop reads the section of a paper where the authors explain their actual method — not the abstract — and rewrites it as a list of things the method *does*. Each item is a verb plus the thing the verb acts on:

```
predict @ state          (guess where a physical system will be next)
correct @ parameter      (adjust the model's weights)
sample @ trajectory      (draw a possible path through the future)
```

Papers are then compared by how much these lists overlap. Same lists, same machinery — regardless of what the papers call themselves.

Doing this reliably took most of the project. Details are in `writeup_final.md`; the short version is in **How it works** below.

## The main finding

**The verb tells you nothing. The object tells you everything.**

Almost every paper in the corpus "predicts" something, so knowing that a method predicts is worthless on its own. But predicting *a physical state*, predicting *an internal representation*, and predicting *a model's own weights* are three unrelated activities that happen to share a word. Once you separate them, the picture changes completely.

Under that lens, most family names fall apart. Papers filed under one label split into two unrelated groups, or scatter entirely.

The method also finds two genuine connections across fields that nobody had drawn — self-supervised learning and predictive coding turn out to share a learning rule, and particle filters turn out to share machinery with Kalman-style state estimation — and it **rejects** the one connection most people assume is real: Kalman filtering and predictive coding. They look similar because they use the same words. They don't do the same thing.

## Testing it for real

Any tool can produce a similarity score. Scores are easy to believe and impossible to check. So the point of the project was to turn one score into something that could be *wrong*.

The scanner matched a robotics paper (**Diffuser**, which plans robot motion) to an image-generation paper (**DDIM**, which makes pictures). Different fields, different conferences, no shared citations. But according to the extracted machinery, they run the same inner loop.

If that's true, then a specific shortcut invented for the image paper should work in the robotics paper. So the prediction was written down **in advance**, with numbers attached:

> DDIM's deterministic sampler, using 5–10× fewer steps, will keep Diffuser's benchmark score within about 5%.

Then it was run: 45 episodes, three random seeds, three simulated robots, each compared against an identical baseline run.

| shortcut | result |
|---|---|
| **2× fewer steps** | held on every robot |
| **4× fewer steps** | landed on the 5% line to the decimal |
| **10× fewer steps** | failed, as the prediction's upper bound implied |

The data also showed one thing the scanner never predicted: on the walker robot, the deterministic sampler is *worse* than the random one even at full step count. That's reported as an open question rather than quietly folded into the win.

**The claim here is not "this makes Diffuser 4× faster."** The claim is that a program reading English text produced a specific, checkable engineering prediction, and the experiment landed inside the bounds it stated.

## The follow-up, and where it fails

The obvious next question: if you can read a method's machinery from its paper, can you predict **how well it will score** before running it?

Mostly **no**. Five separate attempts, each written down in advance so the answer couldn't be adjusted afterward:

| phase | question asked | answer |
|---|---|---|
| **24** | On robot-control benchmarks, does machinery predict the score better than the experimental settings alone? | **No.** Structure added nothing the settings didn't already have. |
| **25** | Does fixing the statistical model change that? | **Barely.** A small, consistent gain with one predictor; it vanishes entirely with a stronger one — because knowing *which benchmark* a row came from already explains most of the score. |
| **26** | Same test, second benchmark (Atari games), written down in advance. | **Mixed.** Machinery helped with one predictor, but the plain family label did just as well or better with the other. |
| **27** | Maybe the extraction was too thin? Re-ran it to pull ~15 items per paper instead of ~4. | **No — and this closes the excuse.** Five times more shared machinery produced the same answer. What structure knows about outcomes is shallow: roughly "what kind of method is this," which the hand-written family label already told you. |
| **28** | Forget whole methods — if a paper changes *one thing*, does the machinery of that change predict whether the score goes up or down? | **No.** Predicting from the change's structure was no better than guessing the average. |
| **30** | 900 papers across 25 public leaderboards, ~5,000 head-to-head comparisons: can machinery pick which paper scores higher? | **Only where it can't help.** 87% accurate *within* a leaderboard it has seen, but 60% on a new one — barely above a coin flip. Publication year alone gets 71% and transfers fine. |

The reason for the last one is the interesting part: the extracted vocabulary is **task-specific**. Papers about image classification use different operations than papers about object detection. The scanner is reading something real, but its predictive power is trapped inside each field's local dialect and doesn't travel.

These negative results are written up at the same length and care as the positive one. That's deliberate. A tool that only reports its wins isn't measuring anything.

## How it works

Four layers. Each is built from the one below it and can be redone without re-running the expensive parts. Raw model output is never overwritten.

| layer | what happens |
|---|---|
| **Extract** | arXiv ID → PDF → plain text → a parser hunts down the real method section (it handles four different PDF layouts and hits ~90% of the time) → a local `qwen3:8b` model returns the list of operations, the loss function, and the stated failure mode |
| **Canonicalize** | raw verbs are mapped to a fixed 23-word vocabulary, so *forecast* and *predict* stop counting as different things. Computed fresh on every read, never saved — so changing the vocabulary updates the whole corpus instantly and is always reversible |
| **Type** | a second pass assigns each operation its object (`state`, `latent`, `parameter`, `error`, …) and must quote the sentence that justifies it |
| **Compare** | overlap between papers, per pair and per family |

**Three things that mattered more than anything else**, all learned the hard way:

- **Examples beat rules.** Telling a small model "do not list *derive*, *formulate*" was simply ignored. Three or four worked examples fixed the same problem immediately.
- **You must include an example where the answer is "nothing."** Given only examples where a failure mode existed, the model invented one for papers that had none — it copied the example's wording onto unrelated papers. One example with an explicit empty answer cut that to 1 in 30, and a check at read time catches the rest.
- **Never show the model the vocabulary list.** A closed-vocabulary prompt was tested and rejected: the model started describing the list instead of the paper. A paper that explicitly avoids contrastive learning got labeled `contrast`.

The typing pass is where the signal comes from and also where the noise comes from. Two independent runs agree on 72–84% of items. Family-level results are stable to the second decimal; individual pair scores are not — one re-run can move a single pair to zero. This is stated plainly rather than hidden.

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
| `app.py` | server + web UI on `:8000`; holds the extractor, the vocabulary, and the read-time checks |
| `fulltext.py` | arXiv → PDF → method section (`--from` pins a section by hand when the parser guesses wrong) |
| `semantic_type.py` | the typing pass |
| `compare.py` | similarity between papers and families |
| `phase21_inject.py`, `phase22_split.py` | for papers that are deltas on a base method — inherit the base loop, then separate inherited machinery from the paper's own |
| `families.json`, `semantic_ontology.json` | the family labels and the object list |

**Output:**

| path | what |
|---|---|
| `results/` | one record per paper — raw model output, prompt, and typing, kept forever |
| `matrices/` | every similarity matrix computed, per phase |
| `similarity_matrix.csv` | paper-vs-paper overlap on **verbs alone** |
| `semantic_similarity_matrix.csv` | the same on **verb + object** — this is the one that carries the finding |
| `results_phase*_snapshot/` | records as they stood before each re-extraction, so nothing is lost |
| `phase23/` | the Diffuser/DDIM experiment — the patch, sweep scripts, raw logs, paired results, plot |
| `outcome_predictor/` | phases 24–30, the outcome-prediction follow-up (has its own README) |
| `archive/prototypes/` | dead early scripts, kept only because `findings.md` cites them as evidence |

## Running it

The core pipeline is **Python standard library only** — no install step. It needs `pdftotext` (from poppler) and [Ollama](https://ollama.com) with `qwen3:8b` pulled.

```
python3 app.py        # web UI at http://localhost:8000
python3 fulltext.py   # batch: arXiv IDs in, method sections out
```

Expect roughly 40 seconds per method section on a laptop.

The follow-up work needs real dependencies:

```
pip install -r outcome_predictor/requirements.txt
```

`phase23/` reproduces the robotics experiment against an unmodified Diffuser checkout in its own environment — see `phase23/phase23_plan.md`.

## Status

This is a research log, not a maintained tool. It's public so the method, the one validated prediction, and the several results that came back negative can all be read and checked by someone else.

No license — default copyright applies. If you want to reuse any of it, ask.
