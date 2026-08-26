"""Phase 24 outcome predictors: Models A-E as Ridge (primary) and kNN (robustness).

Feature groups (see schema.md — frozen before evaluation):
  structure   : multi-hot over `operation@object` tuples (vocabulary = training rows)
  conditions  : dataset one-hot + dataset_version one-hot + numeric conditions (standardized, train stats)
                + per-numeric-column state one-hot {present, missing, not_applicable}
                + categorical conditions one-hot (train vocabulary; missing / not_applicable are states)
  lineage     : one-hot conventional research label (train vocabulary)

Model   groups
  A     none (train mean)
  B     conditions
  C     structure
  D     structure + conditions
  E     lineage + conditions

Every encoder is fit on the training rows only. Ridge alpha is chosen by grouped CV (by method_id)
inside the training rows. kNN uses k=5, unweighted mean of the 5 nearest training rows, distance =
equal-weight mean over the applicable feature-group distances.
"""
import json
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

MISSING = "missing"
NOT_APPLICABLE = "not_applicable"
STATES = ["present", MISSING, NOT_APPLICABLE]

ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # frozen (schema.md)
K_NEIGHBORS = 5  # frozen

MODEL_GROUPS = {
    "A": [],
    "B": ["conditions"],
    "C": ["structure"],
    "D": ["structure", "conditions"],
    "E": ["lineage", "conditions"],
}


def parse_structure(cell):
    """method_structure column is a JSON list of 'op@object' strings."""
    if isinstance(cell, (list, set, frozenset)):
        return frozenset(cell)
    return frozenset(json.loads(cell))


def numeric_state(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s == MISSING or s == "":
            return MISSING
        if s == NOT_APPLICABLE:
            return NOT_APPLICABLE
        return "present"
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return MISSING
    return "present"


def numeric_value(v):
    if numeric_state(v) != "present":
        return np.nan
    return float(v)


def categorical_value(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return MISSING
    s = str(v).strip()
    return s if s else MISSING


# ---------------------------------------------------------------- feature encoders (fit on train only)

class Encoder:
    """Turns dataframe rows into per-group dense matrices. All vocabularies/statistics come from `fit`."""

    def __init__(self, numeric_conditions, categorical_conditions):
        self.numeric_conditions = list(numeric_conditions)
        self.categorical_conditions = list(categorical_conditions)

    def fit(self, df):
        self.vocab = sorted({t for s in df["method_structure"].map(parse_structure) for t in s})
        self.lineages = sorted(df["lineage"].map(categorical_value).unique())
        self.cat_vocab = {}
        for c in ["dataset", "dataset_version"] + self.categorical_conditions:
            self.cat_vocab[c] = sorted(df[c].map(categorical_value).unique())
        self.num_mean, self.num_std = {}, {}
        for c in self.numeric_conditions:
            vals = df[c].map(numeric_value).astype(float)
            present = vals[~vals.isna()]
            self.num_mean[c] = float(present.mean()) if len(present) else 0.0
            sd = float(present.std(ddof=0)) if len(present) > 1 else 0.0
            self.num_std[c] = sd if sd > 0 else 1.0
        return self

    # --- dense matrices for Ridge
    def structure_matrix(self, df):
        idx = {t: i for i, t in enumerate(self.vocab)}
        X = np.zeros((len(df), len(self.vocab)))
        for r, s in enumerate(df["method_structure"].map(parse_structure)):
            for t in s:
                if t in idx:
                    X[r, idx[t]] = 1.0
        return X

    def lineage_matrix(self, df):
        idx = {t: i for i, t in enumerate(self.lineages)}
        X = np.zeros((len(df), len(self.lineages)))
        for r, v in enumerate(df["lineage"].map(categorical_value)):
            if v in idx:
                X[r, idx[v]] = 1.0
        return X

    def conditions_matrix(self, df):
        blocks = []
        for c, vocab in self.cat_vocab.items():
            idx = {t: i for i, t in enumerate(vocab)}
            X = np.zeros((len(df), len(vocab)))
            for r, v in enumerate(df[c].map(categorical_value)):
                if v in idx:
                    X[r, idx[v]] = 1.0
            blocks.append(X)
        for c in self.numeric_conditions:
            vals = df[c].map(numeric_value).astype(float).to_numpy()
            z = (vals - self.num_mean[c]) / self.num_std[c]
            z = np.where(np.isnan(z), 0.0, z)  # absent -> 0 after standardization; the state one-hot carries absence
            blocks.append(z.reshape(-1, 1))
            states = df[c].map(numeric_state)
            S = np.zeros((len(df), len(STATES)))
            for r, s in enumerate(states):
                S[r, STATES.index(s)] = 1.0
            blocks.append(S)
        return np.hstack(blocks) if blocks else np.zeros((len(df), 0))

    def matrix(self, df, groups):
        parts = []
        if "structure" in groups:
            parts.append(self.structure_matrix(df))
        if "lineage" in groups:
            parts.append(self.lineage_matrix(df))
        if "conditions" in groups:
            parts.append(self.conditions_matrix(df))
        return np.hstack(parts) if parts else np.zeros((len(df), 0))

    # --- kNN group distances
    def structure_sets(self, df):
        return list(df["method_structure"].map(parse_structure))

    def categorical_table(self, df):
        """Every categorical state per row: dataset, dataset_version, categorical conditions,
        and the {present, missing, not_applicable} state of each numeric condition."""
        cols = {}
        for c in ["dataset", "dataset_version"] + self.categorical_conditions:
            cols[c] = df[c].map(categorical_value).tolist()
        for c in self.numeric_conditions:
            cols[c + "__state"] = df[c].map(numeric_state).tolist()
        return cols

    def numeric_table(self, df):
        out = {}
        for c in self.numeric_conditions:
            vals = df[c].map(numeric_value).astype(float).to_numpy()
            out[c] = (vals - self.num_mean[c]) / self.num_std[c]
        return out


# ---------------------------------------------------------------- Ridge

def select_alpha(X, y, groups, grid=ALPHA_GRID):
    """Grouped CV (by method_id) inside the training rows; pick alpha with lowest mean MAE."""
    n_groups = len(set(groups))
    if n_groups < 2 or X.shape[1] == 0:
        return grid[len(grid) // 2]
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    best, best_mae = None, float("inf")
    for a in grid:
        maes = []
        for tr, va in gkf.split(X, y, groups):
            m = Ridge(alpha=a).fit(X[tr], y[tr])
            maes.append(np.mean(np.abs(m.predict(X[va]) - y[va])))
        mae = float(np.mean(maes))
        if mae < best_mae:
            best, best_mae = a, mae
    return best


def fit_predict_ridge(model, enc, train, test):
    groups = MODEL_GROUPS[model]
    y = train["outcome"].astype(float).to_numpy()
    if not groups:  # Model A
        return np.full(len(test), y.mean()), {"alpha": None, "n_features": 0}
    Xtr = enc.matrix(train, groups)
    Xte = enc.matrix(test, groups)
    alpha = select_alpha(Xtr, y, train["method_id"].to_numpy())
    m = Ridge(alpha=alpha).fit(Xtr, y)
    return m.predict(Xte), {"alpha": alpha, "n_features": int(Xtr.shape[1])}


# ---------------------------------------------------------------- kNN

def jaccard_distance(a, b):
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def knn_distances(model, enc, train, test):
    """Return (n_test, n_train) combined distance matrix for the model's feature groups.
    Group distances:
      structure   : 1 - Jaccard(tuple sets)
      numeric     : RMS of standardized differences over columns present in BOTH rows; unavailable if none
      categorical : mean mismatch (0 same / 1 different) over categorical columns, incl. numeric-state columns
      lineage     : mismatch (0/1)
    Combined = mean over the groups that are available for that pair."""
    groups = MODEL_GROUPS[model]
    n_te, n_tr = len(test), len(train)
    dist_groups = []
    if "structure" in groups:
        S_tr, S_te = enc.structure_sets(train), enc.structure_sets(test)
        D = np.array([[jaccard_distance(a, b) for b in S_tr] for a in S_te])
        dist_groups.append(D)
    if "lineage" in groups:
        L_tr = train["lineage"].map(categorical_value).tolist()
        L_te = test["lineage"].map(categorical_value).tolist()
        D = np.array([[0.0 if a == b else 1.0 for b in L_tr] for a in L_te])
        dist_groups.append(D)
    if "conditions" in groups:
        # numeric
        N_tr, N_te = enc.numeric_table(train), enc.numeric_table(test)
        if N_tr:
            sq = np.zeros((n_te, n_tr))
            cnt = np.zeros((n_te, n_tr))
            for c in N_tr:
                a = N_te[c].reshape(-1, 1)
                b = N_tr[c].reshape(1, -1)
                both = ~np.isnan(a) & ~np.isnan(b)
                d = np.where(both, (a - b) ** 2, 0.0)
                sq += d
                cnt += both
            Dnum = np.where(cnt > 0, np.sqrt(sq / np.maximum(cnt, 1)), np.nan)
            dist_groups.append(Dnum)
        # categorical
        C_tr, C_te = enc.categorical_table(train), enc.categorical_table(test)
        mism = np.zeros((n_te, n_tr))
        for c in C_tr:
            a = np.array(C_te[c]).reshape(-1, 1)
            b = np.array(C_tr[c]).reshape(1, -1)
            mism += (a != b).astype(float)
        dist_groups.append(mism / len(C_tr))
    stack = np.stack(dist_groups)  # (g, n_te, n_tr)
    avail = ~np.isnan(stack)
    combined = np.where(avail, stack, 0.0).sum(0) / np.maximum(avail.sum(0), 1)
    return combined


def fit_predict_knn(model, enc, train, test, k=K_NEIGHBORS):
    y = train["outcome"].astype(float).to_numpy()
    if not MODEL_GROUPS[model]:
        return np.full(len(test), y.mean())
    D = knn_distances(model, enc, train, test)
    k = min(k, len(train))
    preds = np.empty(len(test))
    for i in range(len(test)):
        nn = np.argsort(D[i], kind="stable")[:k]
        preds[i] = y[nn].mean()  # unweighted, fixed before evaluation
    return preds
