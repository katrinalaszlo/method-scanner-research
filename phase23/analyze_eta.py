"""Phase 24: eta sweep analysis. Joins p24 (eta>0) runs with p23 (eta=0 DDIM and DDPM-20) of the same env/seed.
Paired drop vs same-seed DDPM-20 and vs same-seed DDIM eta=0 at the same step count."""
import glob, json, re, pandas as pd, numpy as np
rows = []
for j in sorted(glob.glob("diffuser/logs/*/plans/*/p2[34]_*/rollout.json")):
    env = j.split("/")[2]; suffix = j.split("/")[-2]; d = json.load(open(j))
    m = re.match(r"p23_(ddpm|ddim)(\d+)_s(\d+)", suffix)
    if m: rows.append(dict(env=env, sampler=m[1], steps=int(m[2]), eta=0.0, seed=int(m[3]), score=d["score"], ep_len=d["step"] + 1))
    m = re.match(r"p24_ddim(\d+)_eta([\d.]+)_s(\d+)", suffix)
    if m: rows.append(dict(env=env, sampler="ddim", steps=int(m[1]), eta=float(m[2]), seed=int(m[3]), score=d["score"], ep_len=d["step"] + 1))
df = pd.DataFrame(rows); df = df[df.env.isin(["hopper-medium-v2", "walker2d-medium-v2"])]
df.to_csv("results_eta.csv", index=False)
base = df[df.sampler == "ddpm"].set_index(["env", "seed"])["score"]
eta0 = df[(df.sampler == "ddim") & (df.eta == 0)].set_index(["env", "steps", "seed"])["score"]
d = df[df.sampler == "ddim"].copy()
d["vs_ddpm"] = [100 * (base.get((e, s), np.nan) - sc) / base.get((e, s), np.nan) for e, s, sc in zip(d.env, d.seed, d.score)]
d["vs_eta0"] = [100 * (eta0.get((e, st, s), np.nan) - sc) / eta0.get((e, st, s), np.nan) for e, st, s, sc in zip(d.env, d.steps, d.seed, d.score)]
d["env"] = d.env.str.replace("-medium-v2", "")
g = d.groupby(["env", "steps", "eta"]).agg(score=("score", "mean"), sd=("score", "std"), n=("score", "count"),
    vs_ddpm=("vs_ddpm", "mean"), vs_eta0=("vs_eta0", "mean"), full_eps=("ep_len", lambda x: int((x == 1000).sum()))).reset_index()
g["score"] = (100 * g.score).round(1); g["sd"] = (100 * g.sd).round(1); g["vs_ddpm"] = g.vs_ddpm.round(1); g["vs_eta0"] = g.vs_eta0.round(1)
print("score x100 (mean, sd, n); paired % drop vs same-seed DDPM-20 and vs same-seed DDIM eta=0 (+ = worse); full-length episodes")
print(g.to_string(index=False)); g.to_csv("results_eta_table.csv", index=False)
