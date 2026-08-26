# Semantic compatibility rules

How `compare.py --semantic` decides whether two (operation, object_type) tuples from different papers
count as the same operation. Written 2026-08-24 from the Phase 11–13 runs; revise when the data says so.

## 1. Type hierarchy

Object types come from `semantic_ontology.json`. They are what an operation acts on, assigned per
(paper, operation) by `semantic_type.py` from the paper's own method section — never by a static
verb→type map, because the same verb acts on different objects in different papers (H1 ablation:
`correct` is a state in KalmanNet, a parameter in PC Review, a gradient in Adam).

    object
    ├── represented_state        what the method is estimating or representing
    │   ├── state                hidden state of a dynamical system, tracked over time (x_t, belief, track)
    │   └── latent               learned representation / embedding / feature vector, no dynamics implied
    ├── learning_signal          what drives a parameter change
    │   ├── error                prediction error, residual
    │   └── gradient             derivative of an objective
    ├── parameter                model weights, gains, hyperparameters
    ├── stochastic_object        something drawn from or describing a distribution
    │   ├── distribution         density, posterior, score function
    │   ├── noise                noise variable or level
    │   └── sample               drawn samples, particle sets
    ├── input                    raw observation: pixels, tokens, audio, measurements
    ├── action                   control output
    ├── reward                   reward / return / value
    └── other / unknown          never matches anything

Leaves are what the typing pass assigns. Groups (indented parents) exist only for relaxed matching.

## 2. Matching rules

Two tuples (op_a, type_a), (op_b, type_b) match when op_a == op_b (after canonicalisation) AND the
types are compatible under the chosen mode.

| mode | compatible when | flag |
|---|---|---|
| strict | type_a == type_b, and neither is `other`/`unknown` | `--strict` (default) |
| relaxed | same as strict, OR both leaves sit under the same group in the hierarchy above | `--relaxed` |

Groups that relax: `represented_state` (state ~ latent), `learning_signal` (error ~ gradient),
`stochastic_object` (distribution ~ noise ~ sample). `parameter`, `input`, `action`, `reward` never relax.

`other` and `unknown` are tagged with the paper name so they can never match across papers. An
un-typed operation is evidence of nothing.

## 3. Why each rule

**state ~ latent (relaxed only).** Both are a recursive estimate of a hidden variable. A Kalman state
x_t and a Dreamer latent z_t play the same structural role: something predicted forward and corrected
by an observation. This is the equivalence that makes KalmanNet × World Models a 0.40 pair under
relaxed and a 0 under strict (Phase 12). It is NOT the default because the same relaxation also lets
BYOL's `project@state` — a mis-typing of a projection head output — match genuine state estimation.
Use relaxed to ask "same loop?"; use strict to ask "same object?".

**error ~ gradient.** In predictive-coding-as-learning-rule the prediction error *is* the gradient
signal (PC Beyond Backprop: `transmit@error`; PC Review: `compute gradients@gradient`). Merging them
lets the pc-learning family cohere under relaxed without pretending errors and gradients are the same
thing under strict.

**distribution ~ noise ~ sample.** Diffusion papers describe one process from three angles (DDPM
`predict@noise`, Score-SDE `estimate score@distribution`, DDIM `sample@sample`). Strict keeps them
apart, which is correct at the object level and wrong at the mechanism level.

**parameter never relaxes.** `correct@parameter` (a weight update) vs `correct@state` (a state update)
is the single most important distinction the project found: it is what separates pc-learning from
pc-neuro (lexical 0.56 → strict 0.03) and what dissolved the KalmanNet → PC Review "bridge". Relaxing
it would undo Phase 11.

**input never relaxes.** Predicting pixels (PredNet `predict@input`) and predicting a latent
(I-JEPA `predict@latent`) are different design choices with different failure modes (pixel
reconstruction vs representation collapse). Keeping them apart is the point of the JEPA family.

## 4. Known weaknesses

- The typing pass over-assigns `state` to any vector-shaped object (BYOL `project@state`, ResNet
  `identity@state`). Pass 2 with an explicit "a vector or embedding is latent, not state" rule still
  produced `project@state` for BYOL. Treat strict state-matches involving non-dynamical papers with
  suspicion; relaxed absorbs the error but also blurs the distinction.
- One typing model. Pass 1 vs pass 2 agreement measures prompt sensitivity, not model independence.
- Groups are hand-written from 30 papers. They should be re-derived when the corpus is larger.
