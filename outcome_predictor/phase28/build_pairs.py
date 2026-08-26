"""Phase 28: build pairs.csv (main vs variant, same paper/method/dataset)."""
import csv, glob, json, os, re, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); P24 = os.path.dirname(HERE); ROOT = os.path.dirname(P24)
sys.path.insert(0, P24); sys.path.insert(0, os.path.join(P24, "phase26"))
from build_dataset import normalize_dataset
from build_atari import game

EXCLUDE = re.compile(r"online|delayed|raw return|Table 8/9 Full|4 unseen|Table 2 ablation column|v2 datasets|Table 3 raw", re.I)
# named variant methods of the same paper -> base method (same table family)
VARIANTS = {("2106.02039", "TT (uniform)"): "TT (quantile)", ("2106.08909", "One-step Easy BCQ"): "One-step Rev. KL Reg",
            ("2106.08909", "One-step Exp. Weight"): "One-step Rev. KL Reg", ("2106.08909", "Multi-step (Rev. KL Reg)"): "One-step Rev. KL Reg",
            ("2106.08909", "Iterative (Rev. KL Reg)"): "One-step Rev. KL Reg", ("2110.01548", "SAC-N"): "EDAC",
            ("2208.06193", "BC-Diffusion"): "Diffusion-QL", ("2208.06193", "BCQ-Diffusion"): "Diffusion-QL",
            ("2211.15657", "CondDiffuser"): "Decision Diffuser (DD)", ("2211.15657", "CondMLPDiffuser"): "Decision Diffuser (DD)",
            ("2209.14548", "SfBC + Gaussian"): "SfBC", ("2209.14548", "SfBC + VAE"): "SfBC", ("2209.14548", "SfBC - Planning"): "SfBC",
            ("2401.02644", "HD"): "HD-DA", ("2011.07213", "PLAS+P"): "PLAS", ("2304.10573", "IDQL-1"): "IDQL-A",
            ("2305.20081", "EDP + TD3"): "EDP (Diffusion-QL + EDP)", ("2305.20081", "EDP + CRR"): "EDP (Diffusion-QL + EDP)",
            ("2305.20081", "EDP + IQL"): "EDP (Diffusion-QL + EDP)"}
BENCH = {"outcome_predictor/sources/tables": "d4rl", "outcome_predictor/phase26/sources": "atari"}


def norm(bench, name):
    return normalize_dataset(name)[0] if bench == "d4rl" else game(name)


def hns(tab, r, g):
    refs = {game(k): v for k, v in tab.get("reference_scores", {}).items() if game(k)}
    if g not in refs or refs[g].get("human") in (None, "-"): return None
    return 100 * (float(r["outcome"]) - float(refs[g]["random"])) / (float(refs[g]["human"]) - float(refs[g]["random"]))


def score(bench, tab, r, ds):
    if bench == "d4rl":
        return None if "raw return" in (r.get("notes") or "").lower() else float(r["outcome"])
    return hns(tab, r, ds)


def describe(r, m=None):
    d = f"variant: {r['config_label']}; conditions: {json.dumps(r.get('config_conditions') or {})}; notes: {r.get('notes') or ''}"
    if m is not None:
        d += f"; variant method description: {m.get('one_paragraph_description','')[:600]}; variant settings: " + \
             "; ".join(f"{k}={v.get('value')}" for k, v in list(m.get('global_conditions', {}).items())[:12])
    return d


pairs = []
for d, bench in BENCH.items():
    for f in sorted(glob.glob(os.path.join(ROOT, d, "*.json"))):
        tab = json.load(open(f)); aid = os.path.basename(f)[:-5]
        if "methods" not in tab or aid == "2006.09359": continue
        methods = {m["method_name"]: m for m in tab["methods"]}
        def mains(m):
            out = {}
            for r in m["rows"]:
                if r["config_label"] == "main" or (r["config_label"].startswith("main") and not EXCLUDE.search(r["config_label"])):
                    ds = norm(bench, r["dataset"]); s = score(bench, tab, r, ds) if ds else None
                    if ds and s is not None and ds not in out: out[ds] = (r, s)
            return out
        for name, m in methods.items():
            base_name = VARIANTS.get((aid, name))
            base = methods.get(base_name) if base_name else m
            if base is None: continue
            bm = mains(base)
            for r in m["rows"]:
                is_variant_method = base_name is not None
                if not is_variant_method and (r["config_label"].startswith("main")): continue
                if EXCLUDE.search(r["config_label"]) or re.search(r"online fine|raw return|delayed", r.get("notes") or "", re.I): continue
                if is_variant_method and not (r["config_label"] == "main" or not r["config_label"].startswith("main")): continue
                ds = norm(bench, r["dataset"])
                if not ds or ds not in bm: continue
                s = score(bench, tab, r, ds)
                if s is None: continue
                vid = f"{aid}|{name}|{r['config_label']}"
                pairs.append({"pair_id": f"{vid}|{ds}", "variant_id": vid, "paper_id": aid, "benchmark": bench, "base_method": base["method_name"],
                              "variant_method": name, "variant_label": r["config_label"], "dataset": ds, "main_score": round(bm[ds][1], 3),
                              "variant_score": round(s, 3), "delta": round(s - bm[ds][1], 3),
                              "description": describe(r, m if is_variant_method else None),
                              "source": f"{r.get('table','')} vs main {bm[ds][0].get('table','')}"})
# Phase 23: DDIM-N vs DDPM-20, same env
p23 = defaultdict(list)
for r in csv.DictReader(open(os.path.join(ROOT, "phase23", "results_collected.csv"))):
    p23[(r["env"], r["sampler"], int(r["steps"]))].append(float(r["score"]) * 100)
for (env, sampler, steps), sc in p23.items():
    if sampler == "ddpm": continue
    ds = normalize_dataset(env)[0]; base = sum(p23[(env, "ddpm", 20)]) / 3; v = sum(sc) / 3
    vid = f"phase23|Diffuser|ddim-{steps}"
    pairs.append({"pair_id": f"{vid}|{ds}", "variant_id": vid, "paper_id": "2205.09991", "benchmark": "d4rl", "base_method": "Diffuser",
                  "variant_method": "Diffuser", "variant_label": f"ddim-{steps}", "dataset": ds, "main_score": round(base, 2),
                  "variant_score": round(v, 2), "delta": round(v - base, 2),
                  "description": f"variant: replace the stochastic DDPM reverse-process sampler (20 steps) with the deterministic DDIM sampler (eta=0) using {steps} strided denoising steps; everything else unchanged (pretrained model, guidance, horizon)",
                  "source": "phase23/results_collected.csv seed mean"})
with open(os.path.join(HERE, "pairs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys())); w.writeheader(); w.writerows(pairs)
per = defaultdict(int)
for p in pairs: per[p["paper_id"]] += 1
print(f"pairs={len(pairs)} variants={len({p['variant_id'] for p in pairs})} papers={len(per)}"); print(dict(per))
