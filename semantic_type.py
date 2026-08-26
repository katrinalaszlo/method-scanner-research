"""Second pass: for each full-text record, type every raw operation by the object it acts on.
Ops are NOT re-extracted — the stored list is given to the model; it only assigns object_type,
context, and a supporting quote from the section text. Writes `semantic_ops` into the record.

Usage: venv/bin/python semantic_type.py            # pass 1: all arxiv records lacking semantic_ops
       venv/bin/python semantic_type.py --pass 2   # pass 2: strict grounding, `unknown` allowed -> semantic_ops_pass2 + typing_pass2.json
       venv/bin/python semantic_type.py --pass v2  # ontology v2 (policy/value/task) -> semantic_ops_v2; pass-1 prompt
       venv/bin/python semantic_type.py --force    # redo
       venv/bin/python semantic_type.py FILE.json  # one record
Pass 2 uses the same model with a stricter prompt at temperature 0 — it measures prompt sensitivity,
not model independence (no second capable local model is available).
"""
import glob
import json
import re
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY = json.load(open(os.path.join(ROOT, "semantic_ontology.json")))
TYPES = [t for t in ONTOLOGY["object_types"] if not t.startswith("_")]
TYPES_PASS2 = TYPES + ["unknown"]
TYPES_V2 = TYPES + [t for t in ONTOLOGY.get("object_types_v2", {}) if not t.startswith("_")]
DEFS_V2 = {**{k: v for k, v in ONTOLOGY["object_types"].items() if not k.startswith("_")},
           **{k: v for k, v in ONTOLOGY.get("object_types_v2", {}).items() if not k.startswith("_")}}
CONTEXTS = ONTOLOGY["contexts"]
MODEL = "qwen3:8b"
OLLAMA_CHAT = "http://localhost:11434/api/chat"

SYSTEM = "You are a careful annotator of scientific methods. Output ONLY valid JSON."
PROMPT = """Below is a method section and a list of operations already extracted from it.
For EACH operation, say what kind of object the operation acts on, using exactly one of these types:
{types}

Type definitions:
{defs}

Also give a context from: {contexts}
and a short quote (under 20 words) from the text that shows the object. If the text does not make
the object clear, use "other" and an empty quote. Do not add, remove, or rename operations.

Text:
{text}

Operations: {ops}

Output a JSON list, one item per operation, in the same order:
[{{"op": "...", "object_type": "...", "context": "...", "quote": "..."}}]
JSON:"""


PROMPT_PASS2 = """Below is a method section and a list of operations already extracted from it.
For EACH operation, identify the object the operation acts on — but ONLY if the text explicitly names
that object near where the operation is described. Use exactly one of these types:
{types}

Type definitions:
{defs}

STRICT RULES:
- Assign a type only when the text itself names the object (the word "state", "latent", "representation",
  "parameter", "weight", "gradient", "noise", "distribution", "sample", "input", "observation", "action", "reward",
  or a clear synonym). Do not infer the type from what the method is "usually" about.
- A vector, embedding, or feature is "latent", not "state". "state" is only for the hidden state of a
  dynamical system that is tracked over time.
- If the text does not make the object explicit, answer "unknown" with an empty quote. "unknown" is a
  normal answer, not a failure.
- The quote must be copied verbatim from the text, under 20 words, and must contain the object word.
- Do not add, remove, or rename operations.

Context from: {contexts}

Text:
{text}

Operations: {ops}

Output a JSON list, one item per operation, in the same order:
[{{"op": "...", "object_type": "...", "context": "...", "quote": "..."}}]
JSON:"""


def type_record(rec, pass_no=1):
    ops = (rec.get("data") or {}).get("operations") or []
    if not ops:
        return []
    template = PROMPT_PASS2 if pass_no == 2 else PROMPT
    allowed = TYPES_PASS2 if pass_no == 2 else (TYPES_V2 if pass_no == "v2" else TYPES)
    defs = DEFS_V2 if pass_no == "v2" else {t: d for t, d in ONTOLOGY["object_types"].items() if not t.startswith("_")}
    prompt = template.format(
        types=", ".join(allowed),
        defs="\n".join(f"- {t}: {d}" for t, d in defs.items()),
        contexts=", ".join(CONTEXTS),
        text=rec["abstract"],
        ops=json.dumps(ops),
    )
    body = json.dumps({
        "model": MODEL, "stream": False, "think": False,
        "options": {"temperature": 0.0, "num_predict": 1500, "num_ctx": 8192},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }).encode()
    with urllib.request.urlopen(urllib.request.Request(OLLAMA_CHAT, body, {"Content-Type": "application/json"}), timeout=600) as r:
        raw = json.load(r)["message"]["content"]
    start, end = raw.find("["), raw.rfind("]") + 1
    if start < 0:
        raise ValueError(f"no JSON list in output: {raw[:160]!r}")
    if end <= start:  # truncated before the closing bracket — keep the complete items
        end = raw.rfind("}") + 1
        if end <= start:
            raise ValueError(f"truncated with no complete item: {raw[:160]!r}")
        raw = raw[:end] + "]"
        end += 1
        print("  (truncated output — salvaged complete items)", flush=True)
    chunk = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw[start:end])  # stray backslashes from LaTeX in quotes
    chunk = re.sub(r'"\s+"(op|object_type|context|quote)"', r'"\1"', chunk)  # Phase 24: model sometimes emits `" "op"` (stray quote before a key)
    items, _ = json.JSONDecoder(strict=False).raw_decode(chunk)  # raw newlines in quotes; ignore a second list after the first
    if items and isinstance(items[0], str):  # model sometimes returns ["op", ...] — unusable
        raise ValueError(f"list of strings, not objects: {raw[:120]!r}")
    out = []
    for op, it in zip(ops, items):
        t = str(it.get("object_type", "other")).strip().lower()
        c = str(it.get("context", "unspecified")).strip().lower()
        q = str(it.get("quote", "")).strip()
        grounded = bool(q) and q.lower()[:40] in rec["abstract"].lower()
        out.append({
            "op": op,
            "object_type": t if t in allowed else ("unknown" if pass_no == 2 else "other"),
            "context": c if c in CONTEXTS else "unspecified",
            "quote": q,
            "grounded": grounded,
        })
    return out


def main(argv):
    force = "--force" in argv
    pass_no = 1
    if "--pass" in argv:
        v = argv[argv.index("--pass") + 1]
        pass_no = "v2" if v == "v2" else int(v)
    field = {1: "semantic_ops", 2: "semantic_ops_pass2", "v2": "semantic_ops_v2"}[pass_no]
    files = [a for a in argv if a.endswith(".json")] or sorted(glob.glob(os.path.join(ROOT, "results", "*.json")))
    export = []
    for f in files:
        rec = json.load(open(f))
        if not (rec.get("source") or "").startswith(("arxiv:", "test:")):
            continue
        if rec.get(field) and not force:
            if pass_no == 2:
                export.append({"paper": rec["paper"], "source": rec["source"], "tuples": rec[field]})
            continue
        try:
            sem = type_record(rec, pass_no)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"ERR {rec['paper'][:40]:42} {e}", flush=True)
            continue
        rec[field] = sem
        rec[field + "_model"] = MODEL
        json.dump(rec, open(f, "w"), indent=2)
        if pass_no == 2:
            export.append({"paper": rec["paper"], "source": rec["source"], "tuples": sem})
        tup = [f"{s['op']}:{s['object_type']}{'' if s['grounded'] else '?'}" for s in sem]
        print(f"OK  {rec['paper'][:40]:42} {tup}", flush=True)
    if pass_no == 2:
        json.dump(export, open(os.path.join(ROOT, "typing_pass2.json"), "w"), indent=2)
        print(f"wrote typing_pass2.json ({len(export)} papers)")


if __name__ == "__main__":
    main(sys.argv[1:])
