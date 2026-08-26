"""v1 -> v2 typing diff per paper. Usage: venv/bin/python diff_v2.py [--json out.json]"""
import glob, json, sys
from app import RAW_TO_CANONICAL
def canon(op):
    raw = op.strip().lower()
    return RAW_TO_CANONICAL.get(raw) or (RAW_TO_CANONICAL.get(raw.split()[0]) if " " in raw else None) or raw
rows = []
for f in sorted(glob.glob("results/*.json")):
    r = json.load(open(f))
    if not r.get("semantic_ops_v2"):
        continue
    v1 = [f"{canon(s['op'])}@{s['object_type']}" for s in r["semantic_ops"]]
    v2 = [f"{canon(s['op'])}@{s['object_type']}" for s in r["semantic_ops_v2"]]
    changed = [(a, b) for a, b in zip(v1, v2) if a != b]
    rows.append({"paper": r["paper"], "id": r["source"].split()[0], "v1": v1, "v2": v2, "changed": changed})
    print(f"{r['paper'][:44]:46} changed {len(changed)}/{len(v1)}")
    for a, b in changed:
        print(f"    {a:34} -> {b}")
if "--json" in sys.argv:
    json.dump(rows, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=2)
