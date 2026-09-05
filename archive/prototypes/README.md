# Prototypes (archived)

Dead code. Kept because `findings.md` cites it as evidence, not because anything imports it.

These are the standalone extraction scripts from Phases 1–4, before extraction moved into `app.py`.
They are the model bake-off recorded in `findings.md`: `llama3.2:3b` produced nouns as operations,
duplicates, and example echoes; `qwen3:4b` could not disable its thinking mode and returned empty
output; `qwen3:8b` won and is what the pipeline uses.

`extract_llama.py` is flagged in `findings.md` as the original standalone script, still on
`llama3.2:3b` and the OpenAI client — stale relative to `app.py` from Phase 4 onward.

None of these are on the current path. Nothing here is maintained or expected to run.
