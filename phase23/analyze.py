"""Phase 23 analysis: results.csv -> table (mean ± sd), % drop vs DDPM-20, plot, verdict."""
import pandas as pd, numpy as np, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import subprocess; subprocess.run([sys.executable, "collect.py"], check=True, capture_output=True)
df = pd.read_csv("results_collected.csv").dropna(subset=["score"])
df["cfg"] = df["sampler"] + "-" + df["steps"].astype(str)
order = ["ddpm-20", "ddim-20", "ddim-10", "ddim-5", "ddim-2"]
envs = ["halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"]

g = df.groupby(["env", "cfg"])["score"].agg(["mean", "std", "count"]).reset_index()
rows = []
for env in envs:
    base = g[(g.env == env) & (g.cfg == "ddpm-20")]["mean"]
    base = float(base.iloc[0]) if len(base) else np.nan
    for cfg in order:
        r = g[(g.env == env) & (g.cfg == cfg)]
        if not len(r): continue
        m, s, n = float(r["mean"].iloc[0]), float(r["std"].iloc[0]) if r["count"].iloc[0] > 1 else 0.0, int(r["count"].iloc[0])
        rows.append({"env": env.replace("-medium-v2", ""), "cfg": cfg, "score_x100": round(100 * m, 1), "sd": round(100 * s, 1), "n": n,
                     "drop_pct": round(100 * (base - m) / base, 1) if base and not np.isnan(base) else np.nan})
t = pd.DataFrame(rows)
print(t.to_string(index=False))
t.to_csv("results_table.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, env in zip(axes, envs):
    sub = t[t.env == env.replace("-medium-v2", "")]
    d = sub[sub.cfg.str.startswith("ddim")]
    ax.errorbar(d.cfg.str.replace("ddim-", "").astype(int), d.score_x100, yerr=d.sd, marker="o", label="DDIM σ=0")
    b = sub[sub.cfg == "ddpm-20"]
    if len(b):
        ax.axhline(b.score_x100.iloc[0], color="k", ls="--", label="DDPM-20 baseline")
        ax.axhline(0.95 * b.score_x100.iloc[0], color="r", ls=":", label="−5%")
    ax.set_xscale("log", base=2); ax.set_xticks([2, 5, 10, 20]); ax.set_xticklabels([2, 5, 10, 20])
    ax.set_xlabel("denoising steps"); ax.set_title(env.replace("-medium-v2", "")); ax.set_ylabel("D4RL score")
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("return_vs_steps.png", dpi=130)

## paired: each DDIM run vs the DDPM-20 run of the SAME seed and env (cancels seed-to-seed baseline drift)
base = df[df.cfg == "ddpm-20"].set_index(["env", "seed"])["score"]
dd = df[df.cfg != "ddpm-20"].copy()
dd["base"] = [base.get((e, s), np.nan) for e, s in zip(dd.env, dd.seed)]
dd["paired_drop_pct"] = 100 * (dd.base - dd.score) / dd.base
pp = dd.dropna(subset=["paired_drop_pct"]).groupby(["env", "cfg"])["paired_drop_pct"].agg(["mean", "std", "count", "max"]).reset_index()
pp["env"] = pp.env.str.replace("-medium-v2", "")
pp = pp.set_index("cfg").loc[[c for c in order if c in set(pp.cfg)]].reset_index()
print("\npaired drop vs same-seed DDPM-20 (%, + = worse; mean ± sd over seeds, worst seed)")
print(pp.round(1).to_string(index=False))
pp.round(2).to_csv("results_paired.csv", index=False)

v = t[t.cfg == "ddim-10"]
ok = (v.drop_pct < 5).sum(); tot = len(v)
print(f"\nDDIM-10: {ok}/{tot} envs within 5% of DDPM-20; mean drop {v.drop_pct.mean():.1f}%")
for cfg in ["ddim-5", "ddim-2", "ddim-20"]:
    w = t[t.cfg == cfg]
    if len(w): print(f"{cfg}: {(w.drop_pct < 5).sum()}/{len(w)} envs within 5%; mean drop {w.drop_pct.mean():.1f}%")
