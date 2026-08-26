"""Phase 28 evaluation: leave-one-paper-out clipped Ridge on signed delta. Models A-D per schema28.md."""
import csv, glob, json, os, re
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__))
ALPHAS = [1.0, 10.0, 100.0, 1000.0]

pairs = pd.read_csv(os.path.join(HERE, "pairs.csv"))
ch = {}
for f in glob.glob(os.path.join(HERE, "changes", "*.json")):
    j = json.load(open(f))
    if "error" not in j: ch[j["variant_id"]] = j
pairs = pairs[pairs.variant_id.isin(ch)].reset_index(drop=True)
pairs["change_type"] = pairs.variant_id.map(lambda v: ch[v]["change_type"])
pairs["affected"] = pairs.variant_id.map(lambda v: ch[v]["affected_tuples"])
pairs["n_affected"] = pairs.affected.map(len)
pairs["frac_affected"] = pairs.variant_id.map(lambda v: len(ch[v]["affected_idx"]) / max(ch[v]["n_ops"], 1))
y = pairs.delta.to_numpy(float)
papers = sorted(pairs.paper_id.unique())


def onehot(col, train_idx, all_idx):
    vocab = sorted(pairs.loc[train_idx, col].unique()); ix = {v: i for i, v in enumerate(vocab)}
    X = np.zeros((len(all_idx), len(vocab)))
    for r, v in enumerate(pairs.loc[all_idx, col]):
        if v in ix: X[r, ix[v]] = 1
    return X


def feats(model, tr, te):
    idx = np.concatenate([tr, te]); blocks = []
    if model in ("B", "D"):
        ms = pairs.loc[idx, "main_score"].to_numpy(float); mu, sd = ms[:len(tr)].mean(), ms[:len(tr)].std() or 1
        blocks += [onehot("dataset", tr, idx), onehot("benchmark", tr, idx), onehot("change_type", tr, idx), ((ms - mu) / sd).reshape(-1, 1)]
    if model in ("C", "D"):
        vocab = sorted({t for a in pairs.loc[tr, "affected"] for t in a}); ix = {t: i for i, t in enumerate(vocab)}
        X = np.zeros((len(idx), len(vocab)))
        for r, a in enumerate(pairs.loc[idx, "affected"]):
            for t in a:
                if t in ix: X[r, ix[t]] = 1
        blocks += [X, pairs.loc[idx, ["n_affected", "frac_affected"]].to_numpy(float)]
    X = np.hstack(blocks); return X[:len(tr)], X[len(tr):]


def fit_predict(model, tr, te):
    if model == "A": return np.full(len(te), y[tr].mean())
    Xtr, Xte = feats(model, tr, te); g = pairs.loc[tr, "paper_id"].to_numpy()
    best, bm = None, 1e18
    for a in ALPHAS:
        m = []
        for a_tr, a_va in GroupKFold(min(5, len(set(g)))).split(Xtr, y[tr], g):
            m.append(np.mean(np.abs(Ridge(alpha=a).fit(Xtr[a_tr], y[tr][a_tr]).predict(Xtr[a_va]) - y[tr][a_va])))
        if np.mean(m) < bm: best, bm = a, np.mean(m)
    return np.clip(Ridge(alpha=best).fit(Xtr, y[tr]).predict(Xte), y[tr].min(), y[tr].max())


res = {m: {} for m in "ABCD"}; preds = {m: np.zeros(len(pairs)) for m in "ABCD"}
for p in papers:
    te = np.where(pairs.paper_id == p)[0]; tr = np.where(pairs.paper_id != p)[0]
    for m in "ABCD":
        pr = fit_predict(m, tr, te); preds[m][te] = pr
        res[m][p] = {"n": len(te), "mae": float(np.mean(np.abs(pr - y[te])))}
big = np.abs(y) >= 2
out = {"n_pairs": len(pairs), "n_papers": len(papers), "n_variants": int(pairs.variant_id.nunique()),
       "change_types": pairs.groupby("variant_id").change_type.first().value_counts().to_dict(),
       "mean_abs_delta": float(np.mean(np.abs(y)))}
print(f"pairs={len(pairs)} papers={len(papers)} variants={out['n_variants']} change_types={out['change_types']} mean|delta|={out['mean_abs_delta']:.2f}")
print(f"{'model':6}{'meanPaperMAE':>14}{'rowMAE':>9}{'signAcc(|d|>=2)':>17}{'MAE|delta|':>12}")
for m in "ABCD":
    mp = float(np.mean([res[m][p]["mae"] for p in papers])); row = float(np.mean(np.abs(preds[m] - y)))
    sign = float(np.mean(np.sign(preds[m][big]) == np.sign(y[big]))); mabs = float(np.mean(np.abs(np.abs(preds[m]) - np.abs(y))))
    out[m] = {"mean_paper_mae": mp, "row_mae": row, "sign_acc": sign, "mae_abs": mabs, "per_paper": res[m]}
    print(f"{m:6}{mp:14.2f}{row:9.2f}{sign:17.3f}{mabs:12.2f}")
out["wins"] = {"D<B": int(sum(res["D"][p]["mae"] < res["B"][p]["mae"] for p in papers)), "C<A": int(sum(res["C"][p]["mae"] < res["A"][p]["mae"] for p in papers))}
print("wins", out["wins"], "of", len(papers))
json.dump(out, open(os.path.join(HERE, "results28.json"), "w"), indent=1)
