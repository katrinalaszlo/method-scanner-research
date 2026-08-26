"""Phase 27 evaluation: swap in full-operation structures (records/<id>.json tuples) for both benchmarks and rerun the
Phase 26 predictors (clipped Ridge primary, dataset-first kNN), strict + relaxed. Compares with the original structures."""
import json, os, sys, glob
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); P24 = os.path.dirname(HERE)
sys.path.insert(0, P24); sys.path.insert(0, os.path.join(P24, "phase25")); sys.path.insert(0, os.path.join(P24, "phase26"))
from evaluate import MODELS
from run import variant, ridge25, knn25, run

REC = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(HERE, "records", "*.json"))}
out = {"records": {k: {"n_ops": r["n_ops"], "n_invalid": r["n_invalid_object"], "n_tuples": len(r["tuples"])} for k, r in REC.items()}}
for bench, csvp, schemap in (("d4rl", os.path.join(P24, "d4rl_dataset.csv"), os.path.join(P24, "schema.json")),
                             ("atari", os.path.join(P24, "phase26", "atari100k_dataset.csv"), os.path.join(P24, "phase26", "schema26.json"))):
    base = pd.read_csv(csvp, dtype=str, keep_default_na=False); base["outcome"] = base.outcome.astype(float)
    schema = json.load(open(schemap))
    paper_of = base.groupby("method_id").paper_id.first().to_dict()
    if bench == "d4rl": paper_of["Diffuser"] = "2205.09991"
    missing = [m for m, p in paper_of.items() if p not in REC]
    if missing: print(bench, "MISSING phase27 records for", missing)
    n = base.method_id.nunique()
    for label in ("original", "phase27"):
        df = base.copy()
        if label == "phase27":
            df["method_structure"] = df.method_id.map(lambda m: json.dumps(REC[paper_of[m]]["tuples"]) if paper_of[m] in REC else None)
            df = df[df.method_structure.notna()]
        for v in ("strict", "relaxed"):
            d = df.copy(); d["method_structure"] = d.method_structure.map(lambda s: variant(json.loads(s), v))
            per = {m: set(json.loads(s)) for m, s in d.groupby("method_id").method_structure.first().items()}
            vocab = set().union(*per.values()); shared = sum(1 for t in vocab if sum(t in s for s in per.values()) >= 2)
            tpm = sum(len(s) for s in per.values()) / len(per)
            for name, fit in (("ridge_clipped", ridge25), ("knn_dataset_first", knn25)):
                r = run(d, schema, fit); a, w = r["mean_per_method_mae"], r["wins"]
                out[f"{bench}/{label}/{v}/{name}"] = {"vocab": len(vocab), "shared": shared, "tuples_per_method": tpm, **r}
                print(f"{bench:5} {label:8} {v:8} {name:18} vocab={len(vocab):3} shared={shared:3} t/m={tpm:4.1f} | " +
                      " ".join(f"{m}={a[m]:6.1f}" for m in MODELS) + f" | D<B {w['D<B']}/{n} D<E {w['D<E']}/{n} C<A {w['C<A']}/{n}", flush=True)
json.dump(out, open(os.path.join(HERE, "results27.json"), "w"), indent=1)
