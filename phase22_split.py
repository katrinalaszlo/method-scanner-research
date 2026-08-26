"""Phase 22: for injected records, split semantic_ops_v2 into inherited (head verb also in the base paper's own extracted ops) and own.
Writes semantic_ops_v2_inherited / semantic_ops_v2_own into the JSON. Idempotent."""
import glob, json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_OF = {"1509.06461": "1312.5602", "1710.02298": "1312.5602", "1802.09477": "1509.02971", "2010.02502": "2006.11239", "1803.02999": "1703.03400"}
recs = {}
for f in glob.glob(os.path.join(ROOT, "results", "*.json")):
    d = json.load(open(f))
    src = d.get("source") or ""
    if src.startswith("arxiv:"):
        recs[src.split()[0][6:]] = (f, d)
head = lambda op: op.strip().lower().split()[0]
for aid, bid in BASE_OF.items():
    f, d = recs[aid]
    if "injected" not in (d.get("source") or ""):
        print(f"skip {aid}: not an injected record"); continue
    base_heads = {head(o) for o in (recs[bid][1].get("data") or {}).get("operations") or []}
    inh = [t for t in d.get("semantic_ops_v2") or [] if head(t["op"]) in base_heads]
    own = [t for t in d.get("semantic_ops_v2") or [] if head(t["op"]) not in base_heads]
    d["semantic_ops_v2_inherited"], d["semantic_ops_v2_own"] = inh, own
    json.dump(d, open(f, "w"), indent=2, ensure_ascii=False)
    print(f"{aid} base={bid} inherited={[t['op']+'@'+t['object_type'] for t in inh]} own={[t['op']+'@'+t['object_type'] for t in own]}")
