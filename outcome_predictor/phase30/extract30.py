"""Phase 30: Method Scanner full-operation extraction (Phase 27 prompt, closed ontology) on the ABSTRACT of each paper.
Usage: venv/bin/python outcome_predictor/phase30/extract30.py   (writes records/<arxiv_id>.json, resumable)"""
import json, os, re, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); P24 = os.path.dirname(HERE); ROOT = os.path.dirname(P24)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(P24, "phase27"))
from app import RAW_TO_CANONICAL
from extract_full import OBJECTS, DEFS, SYNONYM, SYSTEM, PROMPT

papers = json.load(open(os.path.join(HERE, "papers.json")))
os.makedirs(os.path.join(HERE, "records"), exist_ok=True)


def extract(aid, text):
    body = json.dumps({"model": "qwen3:8b", "stream": False, "think": False,
                       "options": {"temperature": 0.0, "num_predict": 2000, "num_ctx": 4096},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": PROMPT.format(objects=", ".join(OBJECTS), text=text, defs="; ".join(f"{k}: {v}" for k, v in DEFS.items()))}]}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"})
    raw = json.load(urllib.request.urlopen(req, timeout=900))["message"]["content"]
    s, e = raw.find("["), raw.rfind("]") + 1
    if s < 0:
        raise ValueError("no JSON list: " + raw[:200])
    chunk = raw[s:e] if e > s else raw[s:raw.rfind("}") + 1] + "]"
    chunk = re.sub(r'"\s+"(op|object|quote)"', r'"\1"', chunk)
    chunk = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", chunk)
    salvaged = False
    try:
        items, _ = json.JSONDecoder(strict=False).raw_decode(chunk)
    except json.JSONDecodeError:
        items = [json.loads(m) for m in re.findall(r'\{\s*"op"\s*:\s*"[^"]*"\s*,\s*"object"\s*:\s*"[^"]*"\s*,\s*"quote"\s*:\s*"[^"]*"\s*\}', chunk)]
        if not items:
            raise
        salvaged = True
    ops, low = [], text.lower()
    for it in items:
        if not isinstance(it, dict) or "op" not in it:
            continue
        op = str(it["op"]).strip().lower(); obj = SYNONYM.get(str(it.get("object", "")).strip().lower(), str(it.get("object", "")).strip().lower())
        q = str(it.get("quote", "")).strip()
        canon = RAW_TO_CANONICAL.get(op) or (RAW_TO_CANONICAL.get(op.split()[0]) if " " in op else None) or op
        ops.append({"op": op, "canonical": canon, "object": obj if obj in OBJECTS else "INVALID:" + obj, "quote": q, "grounded": bool(q) and q.lower()[:30] in low})
    return {"arxiv_id": aid, "source": "abstract", "words": len(text.split()), "model": "qwen3:8b", "salvaged": salvaged, "n_ops": len(ops),
            "n_invalid_object": sum(o["object"].startswith("INVALID") for o in ops), "n_grounded": sum(o["grounded"] for o in ops),
            "ops": ops, "tuples": sorted({f"{o['canonical']}@{o['object']}" for o in ops if not o["object"].startswith("INVALID")})}


if __name__ == "__main__":
    for n, (aid, p) in enumerate(sorted(papers.items())):
        out = os.path.join(HERE, "records", aid + ".json")
        if os.path.exists(out): continue
        t0 = time.time()
        try:
            rec = extract(aid, p["title"] + ". " + p["abstract"])
        except Exception as e:  # logged, never silent
            print("FAIL", aid, repr(e)[:160], flush=True); continue
        json.dump(rec, open(out, "w"), indent=1)
        print(f"OK {n+1}/{len(papers)} {aid} ops={rec['n_ops']} invalid={rec['n_invalid_object']} grounded={rec['n_grounded']} {time.time()-t0:.0f}s {rec['tuples'][:5]}", flush=True)
    print("EXTRACT30_DONE")
