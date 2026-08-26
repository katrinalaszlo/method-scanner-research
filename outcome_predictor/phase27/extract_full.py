"""Phase 27: single-pass full-operation extraction with a closed object ontology.
Usage: venv/bin/python outcome_predictor/phase27/extract_full.py <arxiv_id> [...]   (writes records/<id>.json)"""
import json, os, re, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); P24 = os.path.dirname(HERE); ROOT = os.path.dirname(P24)
sys.path.insert(0, ROOT)
import fulltext as ft
from app import RAW_TO_CANONICAL

OBJECTS = ["state", "latent", "error", "gradient", "parameter", "noise", "input", "action", "reward", "distribution",
           "sample", "policy", "value", "task", "trajectory"]
PINS = {"2211.15657": "G ENERATIVE M ODELING WITH THE D ECISION D IFFUSER", "2112.10751": "R EINFORCEMENT L EARNING VIA S UPERVISED L EARNING",
        "2103.08050": r"^4\. Fisher-BRC", "2304.12824": r"^5\. Q-Guided Policy Optimization", "2401.02644": r"^H IERARCHICAL D IFFUSER$",
        "1903.00374": r"^W ORLD M ODELS$", "1906.05243": r"^4\. Model-based algorithms at scale"}
WORDS = 2500
ONT = json.load(open(os.path.join(ROOT, "semantic_ontology.json")))
DEFS = {**{k: v for k, v in ONT["object_types"].items() if not k.startswith("_")},
        **{k: v for k, v in ONT.get("object_types_v2", {}).items() if not k.startswith("_")}}
SYNONYM = {"loss": "error", "objective": "error", "target": "value", "q-function": "value", "q function": "value", "critic": "value",
           "return": "value", "advantage": "value", "representation": "latent", "embedding": "latent", "feature": "latent",
           "features": "latent", "mean": "distribution", "variance": "distribution", "covariance": "distribution",
           "probability": "distribution", "score": "distribution", "likelihood": "distribution", "plan": "trajectory",
           "model": "parameter", "network": "parameter", "weights": "parameter", "function": "parameter", "observation": "input",
           "image": "input", "frame": "input", "actions": "action", "states": "state", "rewards": "reward", "parameters": "parameter",
           "trajectories": "trajectory", "samples": "sample"}
SYSTEM = "You are a careful annotator of machine-learning methods. Output ONLY valid JSON."
PROMPT = """Below is the method section of a paper. List EVERY operation the described algorithm performs at runtime —
during training and during inference/acting — as an ordered list. Be exhaustive: typically 8 to 25 items. Do not
list what the authors do in the paper (propose, show, compare); list what the algorithm computes.

For each item give:
- "op": the operation as one or two words, verb first, lower case (e.g. "encode", "predict reward", "sample action",
  "update parameters", "denoise", "clip", "resample", "backpropagate").
- "object": what the operation acts on, exactly one of: {objects}
  Definitions: {defs}
  Map near-synonyms onto this list: a loss/objective -> error; a Q-function, critic, target, return or advantage -> value;
  a representation, embedding or feature -> latent; a mean, variance, probability, score or likelihood -> distribution;
  a plan -> trajectory; a model, network or weights -> parameter; an observation, image or frame -> input.
- Each item must be a distinct operation; never repeat an item.
- "quote": a supporting phrase copied verbatim from the text, under 15 words.

Text:
{text}

Output a JSON list only:
[{{"op": "...", "object": "...", "quote": "..."}}, ...]
JSON:"""


def section(aid):
    text = ft.fetch_text(aid)
    if aid in PINS:
        return ft.pinned_section(text, PINS[aid], WORDS)
    return ft.method_section(text, WORDS)


def extract(aid):
    sec, how = section(aid)
    body = json.dumps({"model": "qwen3:8b", "stream": False, "think": False,
                       "options": {"temperature": 0.0, "num_predict": 3000, "num_ctx": 8192},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": PROMPT.format(objects=", ".join(OBJECTS), text=sec, defs="; ".join(f"{k}: {v}" for k, v in DEFS.items()))}]}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"})
    raw = json.load(urllib.request.urlopen(req, timeout=900))["message"]["content"]
    s, e = raw.find("["), raw.rfind("]") + 1
    if s < 0:
        raise ValueError("no JSON list: " + raw[:200])
    chunk = raw[s:e] if e > s else raw[s:raw.rfind("}") + 1] + "]"
    chunk = re.sub(r'"\s+"(op|object|quote)"', r'"\1"', chunk)
    chunk = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", chunk)
    try:
        items, _ = json.JSONDecoder(strict=False).raw_decode(chunk)
    except json.JSONDecodeError:  # salvage well-formed items one by one (logged in the record)
        items = [json.loads(m) for m in re.findall(r'\{\s*"op"\s*:\s*"[^"]*"\s*,\s*"object"\s*:\s*"[^"]*"\s*,\s*"quote"\s*:\s*"[^"]*"\s*\}', chunk)]
        if not items:
            raise
        print("  (salvaged", len(items), "items from malformed JSON)", flush=True)
    ops = []
    low = sec.lower()
    for it in items:
        if not isinstance(it, dict) or "op" not in it:
            continue
        op = str(it["op"]).strip().lower(); obj = str(it.get("object", "")).strip().lower()
        obj = SYNONYM.get(obj, obj)  # documented synonym map (schema27.md); applied after the model's own choice
        q = str(it.get("quote", "")).strip()
        canon = RAW_TO_CANONICAL.get(op) or (RAW_TO_CANONICAL.get(op.split()[0]) if " " in op else None) or op
        ops.append({"op": op, "canonical": canon, "object": obj if obj in OBJECTS else "INVALID:" + obj,
                    "quote": q, "grounded": bool(q) and q.lower()[:30] in low})
    return {"arxiv_id": aid, "section_how": how, "section_words": len(sec.split()), "model": "qwen3:8b", "n_ops": len(ops),
            "n_invalid_object": sum(o["object"].startswith("INVALID") for o in ops), "n_grounded": sum(o["grounded"] for o in ops),
            "ops": ops, "tuples": sorted({f"{o['canonical']}@{o['object']}" for o in ops if not o["object"].startswith("INVALID")})}


if __name__ == "__main__":
    for aid in sys.argv[1:]:
        out = os.path.join(HERE, "records", aid + ".json")
        if os.path.exists(out):
            print("SKIP", aid); continue
        try:
            rec = extract(aid)
        except Exception as e:  # logged, never silent
            print("FAIL", aid, repr(e)[:160], flush=True); continue
        json.dump(rec, open(out, "w"), indent=1)
        print(f"OK {aid} ops={rec['n_ops']} invalid={rec['n_invalid_object']} grounded={rec['n_grounded']} {rec['section_how'][:30]} {rec['tuples'][:8]}", flush=True)
