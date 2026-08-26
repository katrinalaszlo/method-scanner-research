"""Phase 26a evaluation: clipped Ridge (primary) + dataset-first kNN (robustness), strict and relaxed vocab,
leave-one-method-out on atari100k_dataset.csv. Writes results26.json and prints the table."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); P24 = os.path.dirname(HERE)
sys.path.insert(0, P24); sys.path.insert(0, os.path.join(P24, "phase25"))
from evaluate import MODELS
from run import variant, ridge25, knn25, run

df = pd.read_csv(os.path.join(HERE, "atari100k_dataset.csv"), dtype=str, keep_default_na=False); df["outcome"] = df.outcome.astype(float)
schema = json.load(open(os.path.join(HERE, "schema26.json")))
out = {"n_rows": len(df), "n_methods": int(df.method_id.nunique()), "rows_per_method": df.groupby("method_id").size().to_dict()}
n = out["n_methods"]
for v in ("strict", "relaxed"):
    d = df.copy(); d["method_structure"] = d.method_structure.map(lambda s: variant(json.loads(s), v))
    per = {m: set(json.loads(s)) for m, s in d.groupby("method_id").method_structure.first().items()}
    vocab = set().union(*per.values()); shared = sum(1 for t in vocab if sum(t in s for s in per.values()) >= 2)
    for name, fit in (("ridge_clipped", ridge25), ("knn_dataset_first", knn25)):
        r = run(d, schema, fit); out[f"{v}/{name}"] = {"vocab": len(vocab), "shared": shared, **r}
        a, w = r["mean_per_method_mae"], r["wins"]
        print(f"{v:8}{name:20} vocab={len(vocab):3} shared={shared:3} | " + " ".join(f"{m}={a[m]:6.1f}" for m in MODELS) +
              f" | D<B {w['D<B']}/{n} D<E {w['D<E']}/{n} B<A {w['B<A']} C<A {w['C<A']}")
json.dump(out, open(os.path.join(HERE, "results26.json"), "w"), indent=1)
