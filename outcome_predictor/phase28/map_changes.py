"""Phase 28: LLM change mapping. For each variant_id, show the Phase 27 op list + variant description; get affected ops."""
import csv, json, os, re, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); P24 = os.path.dirname(HERE)
OUT = os.path.join(HERE, "changes"); os.makedirs(OUT, exist_ok=True)
SYSTEM = "You are a careful annotator of machine-learning methods. Output ONLY valid JSON."
PROMPT = """A paper's method is described by this numbered list of runtime operations (op @ object, with a quote):
{ops}

The paper also reports a VARIANT of the method:
{desc}

Which operations does the variant remove, replace, or add? Answer with JSON:
{{"change_type": one of "remove" | "replace" | "add" | "hyperparameter" | "other",
  "affected": [list of operation numbers whose computation is removed, replaced or materially changed by the variant; [] if the variant only changes a numeric setting],
  "reason": "one sentence"}}
JSON:"""

pairs = list(csv.DictReader(open(os.path.join(HERE, "pairs.csv"))))
variants = {}
for p in pairs: variants.setdefault(p["variant_id"], p)
for vid, p in variants.items():
    out = os.path.join(OUT, re.sub(r"[^A-Za-z0-9._-]+", "_", vid)[:120] + ".json")
    if os.path.exists(out): continue
    rec = json.load(open(os.path.join(P24, "phase27", "records", p["paper_id"] + ".json")))
    ops = "\n".join(f"{i+1}. {o['canonical']} @ {o['object']} — \"{o['quote'][:80]}\"" for i, o in enumerate(rec["ops"]) if not o["object"].startswith("INVALID"))
    valid = [o for o in rec["ops"] if not o["object"].startswith("INVALID")]
    body = json.dumps({"model": "qwen3:8b", "stream": False, "think": False, "options": {"temperature": 0.0, "num_predict": 400, "num_ctx": 8192},
                       "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT.format(ops=ops, desc=p["description"][:1500])}]}).encode()
    try:
        raw = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"}), timeout=600))["message"]["content"]
        s, e = raw.find("{"), raw.rfind("}") + 1
        j = json.loads(raw[s:e])
        idx = [int(i) for i in j.get("affected", []) if str(i).isdigit() and 1 <= int(i) <= len(valid)]
        res = {"variant_id": vid, "paper_id": p["paper_id"], "change_type": str(j.get("change_type", "other")).lower(), "affected_idx": idx,
               "affected_tuples": sorted({f"{valid[i-1]['canonical']}@{valid[i-1]['object']}" for i in idx}), "n_ops": len(valid),
               "reason": j.get("reason", ""), "raw": raw[:1500]}
    except Exception as ex:
        res = {"variant_id": vid, "paper_id": p["paper_id"], "error": repr(ex)[:200]}
        print("FAIL", vid, res["error"], flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print("OK", vid[:60], res.get("change_type"), res.get("affected_tuples"), flush=True)
print("MAP_DONE")
