"""Phase 30 evaluation: pairwise logistic regression, feature sets per schema30.md, two holdouts. Writes results30.json."""
import glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__))
CS = [0.01, 0.1, 1.0, 10.0]
SETS = {"chance": [], "T": ["T"], "L": ["L"], "E": ["E"], "S": ["S"], "T+L": ["T", "L"], "T+E": ["T", "E"], "T+S": ["T", "S"],
        "T+L+S": ["T", "L", "S"], "T+L+E": ["T", "L", "E"], "T+L+E+S": ["T", "L", "E", "S"]}

papers = json.load(open(os.path.join(HERE, "papers.json")))
emb = json.load(open(os.path.join(HERE, "embeddings.json")))
recs = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(HERE, "records", "*.json"))}
pairs = pd.read_csv(os.path.join(HERE, "pairs.csv"), dtype={"paper_a": str, "paper_b": str})
NO_S = "--no-S" in sys.argv  # smoke test of T/L/E while extraction runs: S-sets skipped, no records required
if NO_S:
    SETS = {k: v for k, v in SETS.items() if "S" not in v}; recs = {a: {"tuples": []} for a in papers}
elif "--partial" not in sys.argv:
    assert len(recs) == len(papers), f"records {len(recs)}/{len(papers)}; pass --partial to evaluate on the extracted subset"
pairs = pairs[pairs.paper_a.isin(recs) & pairs.paper_b.isin(recs)].reset_index(drop=True)
ids = sorted(set(pairs.paper_a) | set(pairs.paper_b)); ix = {a: i for i, a in enumerate(ids)}
years = np.array([float((papers[a]["date"] or "2019")[:4]) for a in ids])
coll = sorted({m["collection"] for a in ids for m in papers[a]["methods"] if m["collection"]})
L = np.zeros((len(ids), len(coll)))
for a in ids:
    for m in papers[a]["methods"]:
        if m["collection"]: L[ix[a], coll.index(m["collection"])] = 1
E = np.array([emb[a] for a in ids])
tup = sorted({t for a in ids for t in recs[a]["tuples"]})
S = np.zeros((len(ids), len(tup)))
for a in ids:
    for t in recs[a]["tuples"]: S[ix[a], tup.index(t)] = 1
BLOCK = {"T": (years - years.mean()).reshape(-1, 1) / years.std(), "L": L, "E": E, "S": S}
print(f"papers={len(ids)} pairs={len(pairs)} boards={pairs.board.nunique()} L-dims={len(coll)} S-dims={len(tup)} "
      f"mean tuples/paper={S.sum(1).mean():.1f} papers with any tag={int((L.sum(1)>0).sum())}")

# both orders of each pair; the second order carries negated features and flipped label
a = pairs.paper_a.map(ix).to_numpy(); b = pairs.paper_b.map(ix).to_numpy(); y = pairs.label_a_better.to_numpy()
def X_for(blocks):
    if not blocks: return np.zeros((2 * len(pairs), 1))
    F = np.hstack([BLOCK[k] for k in blocks]); d = F[a] - F[b]
    return np.vstack([d, -d])
Y = np.concatenate([y, 1 - y]); board = np.concatenate([pairs.board.to_numpy()] * 2)
pair_id = np.concatenate([np.arange(len(pairs))] * 2)


def fit_predict(X, tr, te, groups):
    if X.shape[1] == 1 and not X.any(): return np.full(len(te), 0.5)
    best, bc = -1, None
    for C in CS:
        accs = []
        for itr, iva in GroupKFold(5).split(tr, groups=groups[tr]):
            m = LogisticRegression(C=C, max_iter=2000, fit_intercept=False).fit(X[tr][itr], Y[tr][itr])
            accs.append(m.score(X[tr][iva], Y[tr][iva]))
        if np.mean(accs) > best: best, bc = np.mean(accs), C
    m = LogisticRegression(C=bc, max_iter=2000, fit_intercept=False).fit(X[tr], Y[tr])
    return m.predict_proba(X[te])[:, 1]


def holdout(name, groups, inner):
    out = {}
    for sname, blocks in SETS.items():
        X = X_for(blocks); pred = np.zeros(len(Y))
        for tr, te in GroupKFold(5).split(X, groups=groups):
            pred[te] = fit_predict(X, tr, te, inner)
        hit = (pred > 0.5) == (Y == 1)
        acc = float(hit.mean()); per = float(pd.Series(hit).groupby(board).mean().mean())
        out[sname] = {"acc": round(acc, 4), "per_board_mean": round(per, 4), "n": int(len(Y))}
        print(f"  {name:14} {sname:9} acc={acc:.3f} per-board={per:.3f}", flush=True)
    return out


rng = np.random.RandomState(0); pair_fold = rng.randint(0, 5, len(pairs)); pair_fold = np.concatenate([pair_fold] * 2)
results = {"n_papers": len(ids), "n_pairs": int(len(pairs)), "n_boards": int(pairs.board.nunique()), "L_dims": len(coll), "S_dims": len(tup),
           "board_held_out": holdout("board-held-out", board, board),
           "within_board": holdout("within-board", pair_fold, pair_id)}
bh = results["board_held_out"]
if not NO_S:
    results["H1_TLS_gt_TL"] = bh["T+L+S"]["acc"] > bh["T+L"]["acc"]
    results["H2_S_gt_chance"] = bh["S"]["acc"] > 0.5
    results["H3_TLES_gt_TLE"] = bh["T+L+E+S"]["acc"] > bh["T+L+E"]["acc"]
suffix = "_noS" if NO_S else "_partial" if "--partial" in sys.argv else ""
json.dump(results, open(os.path.join(HERE, "results30" + suffix + ".json"), "w"), indent=1)
print({k: v for k, v in results.items() if k.startswith("H")})
