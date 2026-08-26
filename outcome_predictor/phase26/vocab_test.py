"""Phase 26b: rebuild structures from long-section records (vocab_records/) and rerun the Phase 25 table."""
import json, os, sys, glob
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); P24 = os.path.dirname(HERE); ROOT = os.path.dirname(P24)
sys.path.insert(0, P24); sys.path.insert(0, os.path.join(P24, "phase25"))
import build_dataset as bd
from evaluate import MODELS
from run import variant, ridge25, knn25, run

def record(aid):
    recs = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "vocab_records", "*.json"))]
    recs = [r for r in recs if (r.get("source") or "").startswith(f"test:{aid} ")]
    if not recs:  # long-section extraction failed to parse for this paper -> fall back to the Phase 24 (short) record, logged
        print(f"FALLBACK short record for {aid}")
        return bd.scanner_record(aid)
    assert len(recs) == 1, (aid, len(recs)); return recs[0]

base = pd.read_csv(os.path.join(P24, "d4rl_dataset.csv"), dtype=str, keep_default_na=False); base["outcome"] = base.outcome.astype(float)
schema = json.load(open(os.path.join(P24, "schema.json")))
paper_of = {s["method_id"]: s["paper"] for s in bd.SPECS}
long_struct = {m: json.dumps(bd.structure_tuples(record(p), m)) for m, p in paper_of.items()}
short = {m: json.loads(base[base.method_id == m].method_structure.iloc[0]) for m in paper_of}
out = {"structures_long": {m: json.loads(s) for m, s in long_struct.items()}}
for label, S in (("short", {m: json.dumps(v) for m, v in short.items()}), ("long", long_struct)):
    df = base.copy(); df["method_structure"] = df.method_id.map(S)
    for v in ("strict", "relaxed", "bare"):
        d2 = df.copy(); d2["method_structure"] = d2.method_structure.map(lambda s: variant(json.loads(s), v))
        per = {m: set(json.loads(s)) for m, s in d2.groupby("method_id").method_structure.first().items()}
        vocab = set().union(*per.values()); shared = sum(1 for t in vocab if sum(t in s for s in per.values()) >= 2)
        mean_size = sum(len(s) for s in per.values()) / len(per)
        for name, fit in (("ridge_clipped", ridge25), ("knn_dataset_first", knn25)):
            r = run(d2, schema, fit); a, w = r["mean_per_method_mae"], r["wins"]
            out[f"{label}/{v}/{name}"] = {"vocab": len(vocab), "shared": shared, "mean_tuples_per_method": mean_size, **r}
            print(f"{label:5} {v:8} {name:18} vocab={len(vocab):3} shared={shared:3} tuples/method={mean_size:4.1f} | " +
                  " ".join(f"{m}={a[m]:5.1f}" for m in MODELS) + f" | D<B {w['D<B']}/27 D<E {w['D<E']}/27")
json.dump(out, open(os.path.join(HERE, "vocab_results.json"), "w"), indent=1)
