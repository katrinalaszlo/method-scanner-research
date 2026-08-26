"""Phase 24 leave-one-method-out evaluation of Models A-E (Ridge primary, kNN robustness).

Usage: venv/bin/python outcome_predictor/evaluate.py [--dataset d4rl_dataset.csv] [--schema schema.json] [--out results.json]

For every method_id K: train on every row whose method_id != K, predict every row of K.
Reports row-level MAE/RMSE/corr, per-method MAE, mean-of-per-method MAE, and D-vs-B / D-vs-E win counts.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from predictor import Encoder, fit_predict_ridge, fit_predict_knn, MODEL_GROUPS

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = list(MODEL_GROUPS)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    out = {"n": int(len(y)), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}
    if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0:
        out["corr"] = float(np.corrcoef(y, p)[0, 1])
    else:
        out["corr"] = None
    return out


def run(df, schema, predictor_kind):
    fit_predict = fit_predict_ridge if predictor_kind == "ridge" else (lambda m, e, tr, te: (fit_predict_knn(m, e, tr, te), {}))
    methods = sorted(df["method_id"].unique())
    rows = []  # one record per (held-out row, model)
    per_method = {}
    fit_info = {}
    for K in methods:
        train = df[df["method_id"] != K].reset_index(drop=True)
        test = df[df["method_id"] == K].reset_index(drop=True)
        enc = Encoder(schema["numeric_conditions"], schema["categorical_conditions"]).fit(train)
        per_method[K] = {"n_rows": int(len(test))}
        for m in MODELS:
            pred, info = fit_predict(m, enc, train, test)
            fit_info.setdefault(m, {})[K] = info
            per_method[K][m] = metrics(test["outcome"], pred)
            for i in range(len(test)):
                rows.append({
                    "method_id": K, "model": m, "paper_id": test.loc[i, "paper_id"],
                    "dataset": test.loc[i, "dataset"], "config_label": test.loc[i, "config_label"],
                    "actual": float(test.loc[i, "outcome"]), "predicted": float(pred[i]),
                })
    pred_df = pd.DataFrame(rows)
    aggregate = {}
    for m in MODELS:
        sub = pred_df[pred_df["model"] == m]
        agg = metrics(sub["actual"], sub["predicted"])
        agg["mean_per_method_mae"] = float(np.mean([per_method[K][m]["mae"] for K in methods]))
        agg["median_per_method_mae"] = float(np.median([per_method[K][m]["mae"] for K in methods]))
        aggregate[m] = agg
    wins = {
        "D_beats_B": int(sum(per_method[K]["D"]["mae"] < per_method[K]["B"]["mae"] for K in methods)),
        "D_beats_E": int(sum(per_method[K]["D"]["mae"] < per_method[K]["E"]["mae"] for K in methods)),
        "C_beats_A": int(sum(per_method[K]["C"]["mae"] < per_method[K]["A"]["mae"] for K in methods)),
        "B_beats_A": int(sum(per_method[K]["B"]["mae"] < per_method[K]["A"]["mae"] for K in methods)),
        "n_methods": len(methods),
    }
    return {"predictor": predictor_kind, "aggregate": aggregate, "per_method": per_method,
            "wins": wins, "fit_info": fit_info, "predictions": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(HERE, "d4rl_dataset.csv"))
    ap.add_argument("--schema", default=os.path.join(HERE, "schema.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.dataset, dtype=str, keep_default_na=False)
    df["outcome"] = df["outcome"].astype(float)
    schema = json.load(open(args.schema))
    if schema.get("exclude_config_labels"):
        df = df[~df["config_label"].isin(schema["exclude_config_labels"])].reset_index(drop=True)

    summary = {
        "n_rows": int(len(df)), "n_methods": int(df["method_id"].nunique()), "n_papers": int(df["paper_id"].nunique()),
        "rows_per_method": df.groupby("method_id").size().to_dict(),
        "numeric_conditions": schema["numeric_conditions"], "categorical_conditions": schema["categorical_conditions"],
    }
    out = {"summary": summary, "ridge": run(df, schema, "ridge"), "knn": run(df, schema, "knn")}
    json.dump(out, open(args.out, "w"), indent=1)

    for kind in ("ridge", "knn"):
        r = out[kind]
        print(f"\n== {kind} ==  rows={summary['n_rows']} methods={summary['n_methods']}")
        print(f"{'model':6}{'row MAE':>10}{'meanMethMAE':>13}{'RMSE':>9}{'corr':>8}")
        for m in MODELS:
            a = r["aggregate"][m]
            c = "" if a["corr"] is None else f"{a['corr']:.3f}"
            print(f"{m:6}{a['mae']:10.2f}{a['mean_per_method_mae']:13.2f}{a['rmse']:9.2f}{c:>8}")
        w = r["wins"]
        print(f"D<B on {w['D_beats_B']}/{w['n_methods']} methods; D<E on {w['D_beats_E']}/{w['n_methods']}; "
              f"B<A on {w['B_beats_A']}; C<A on {w['C_beats_A']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
