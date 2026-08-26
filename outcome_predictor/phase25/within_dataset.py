"""Phase 25b: within-dataset ranking. For each D4RL dataset, Spearman(pred, actual) across all held-out rows on that
dataset (every row is predicted with its method held out). Rule fixed before running: all rows, datasets with >=5 rows,
report mean and median Spearman over datasets. Predictors: relaxed/ridge_clipped and strict/knn_dataset_first."""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); P24 = os.path.dirname(HERE)
sys.path.insert(0, P24); sys.path.insert(0, HERE)
import predictor as pr
from evaluate import MODELS
from run import variant, ridge25, knn25

base = pd.read_csv(os.path.join(P24, "d4rl_dataset.csv"), dtype=str, keep_default_na=False); base["outcome"] = base.outcome.astype(float)
schema = json.load(open(os.path.join(P24, "schema.json")))
out = {}
for v, name, fit in (("relaxed", "ridge_clipped", ridge25), ("strict", "knn_dataset_first", knn25)):
    df = base.copy(); df["method_structure"] = df.method_structure.map(lambda s: variant(json.loads(s), v))
    preds = {m: np.zeros(len(df)) for m in MODELS}
    for K in sorted(df.method_id.unique()):
        tr = df[df.method_id != K].reset_index(drop=True); idx = np.where(df.method_id == K)[0]; te = df.iloc[idx].reset_index(drop=True)
        enc = pr.Encoder(schema["numeric_conditions"], schema["categorical_conditions"]).fit(tr)
        for m in MODELS: preds[m][idx] = fit(m, enc, tr, te)
    res = {}
    for m in MODELS:
        rhos = []
        for ds, g in df.groupby("dataset"):
            if len(g) < 5: continue
            p = preds[m][g.index]
            rhos.append(0.0 if np.std(p) == 0 else float(spearmanr(p, g.outcome)[0]))
        res[m] = {"mean_rho": float(np.mean(rhos)), "median_rho": float(np.median(rhos)), "n_datasets": len(rhos)}
    out[f"{v}/{name}"] = res
    print(f"{v}/{name}: " + "  ".join(f"{m} rho={res[m]['mean_rho']:.2f}/{res[m]['median_rho']:.2f}" for m in MODELS) + f"  ({res['A']['n_datasets']} datasets)")
json.dump(out, open(os.path.join(HERE, "within_dataset.json"), "w"), indent=1)
