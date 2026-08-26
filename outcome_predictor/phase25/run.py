"""Phase 25 runner. Usage: venv/bin/python outcome_predictor/phase25/run.py"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
HERE = os.path.dirname(os.path.abspath(__file__)); P24 = os.path.dirname(HERE); ROOT = os.path.dirname(P24)
sys.path.insert(0, P24)
import predictor as pr
from evaluate import metrics, MODELS

ONT = json.load(open(os.path.join(ROOT, "semantic_ontology.json")))
GROUP = {t: g for g, ts in ONT["compatibility_groups"].items() if not g.startswith("_") for t in ts}
ALPHA25 = [1.0, 10.0, 100.0, 1000.0]

def variant(tuples, kind):
    out = set()
    for t in tuples:
        op, obj = t.split("@", 1)
        if kind == "bare": out.add(op)
        elif kind == "relaxed": out.add(f"{op}@{GROUP.get(obj, obj)}")
        else: out.add(t)
    return json.dumps(sorted(out))

def ridge25(model, enc, train, test):
    y = train["outcome"].to_numpy(float)
    groups = pr.MODEL_GROUPS[model]
    if not groups: return np.full(len(test), y.mean())
    Xtr, Xte = enc.matrix(train, groups), enc.matrix(test, groups)
    a = pr.select_alpha(Xtr, y, train["method_id"].to_numpy(), grid=ALPHA25)
    p = Ridge(alpha=a).fit(Xtr, y).predict(Xte)
    return np.clip(p, y.min(), y.max())

def knn25(model, enc, train, test):
    y = train["outcome"].to_numpy(float)
    if not pr.MODEL_GROUPS[model]: return np.full(len(test), y.mean())
    D = pr.knn_distances(model, enc, train, test)
    if "conditions" in pr.MODEL_GROUPS[model]:
        a = train["dataset"].to_numpy().reshape(1, -1); b = test["dataset"].to_numpy().reshape(-1, 1)
        Dds = (a != b).astype(float)
        n = len(pr.MODEL_GROUPS[model]) + (1 if "conditions" in pr.MODEL_GROUPS[model] else 0)  # conditions = numeric+categorical
        # Phase 24 combined = mean over available groups; add dataset group with equal weight (approximate by re-averaging)
        D = (D * n + Dds) / (n + 1)
    preds = np.empty(len(test))
    for i in range(len(test)):
        nn = np.argsort(D[i], kind="stable")[:5]; preds[i] = y[nn].mean()
    return preds

def run(df, schema, fit):
    methods = sorted(df["method_id"].unique()); pm = {}
    for K in methods:
        tr = df[df.method_id != K].reset_index(drop=True); te = df[df.method_id == K].reset_index(drop=True)
        enc = pr.Encoder(schema["numeric_conditions"], schema["categorical_conditions"]).fit(tr)
        pm[K] = {m: metrics(te["outcome"], fit(m, enc, tr, te))["mae"] for m in MODELS}
    agg = {m: float(np.mean([pm[K][m] for K in methods])) for m in MODELS}
    med = {m: float(np.median([pm[K][m] for K in methods])) for m in MODELS}
    wins = {"D<B": sum(pm[K]["D"] < pm[K]["B"] for K in methods), "D<E": sum(pm[K]["D"] < pm[K]["E"] for K in methods),
            "B<A": sum(pm[K]["B"] < pm[K]["A"] for K in methods), "C<A": sum(pm[K]["C"] < pm[K]["A"] for K in methods)}
    return {"mean_per_method_mae": agg, "median_per_method_mae": med, "wins": wins, "per_method": pm}

def main():
    base = pd.read_csv(os.path.join(P24, "d4rl_dataset.csv"), dtype=str, keep_default_na=False)
    base["outcome"] = base["outcome"].astype(float)
    schema = json.load(open(os.path.join(P24, "schema.json")))
    out = {}
    for v in ("strict", "relaxed", "bare"):
        df = base.copy(); df["method_structure"] = df["method_structure"].map(lambda s: variant(json.loads(s), v))
        vocab = {t for s in df.groupby("method_id").method_structure.first() for t in json.loads(s)}
        shared = sum(1 for t in vocab if sum(t in json.loads(s) for s in df.groupby("method_id").method_structure.first()) >= 2)
        for name, fit in (("ridge_clipped", ridge25), ("knn_dataset_first", knn25)):
            r = run(df, schema, fit); out[f"{v}/{name}"] = r
            a, w = r["mean_per_method_mae"], r["wins"]
            print(f"{v:8}{name:20} vocab={len(vocab):3} shared={shared:3} | " + " ".join(f"{m}={a[m]:5.1f}" for m in MODELS) +
                  f" | D<B {w['D<B']}/27 D<E {w['D<E']}/27 B<A {w['B<A']} C<A {w['C<A']}")
    json.dump(out, open(os.path.join(HERE, "results25.json"), "w"), indent=1)

if __name__ == "__main__": main()
