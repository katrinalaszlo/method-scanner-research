"""Phase 24: assemble outcome_predictor/d4rl_dataset.csv from
  - outcome_predictor/sources/tables/<arxiv_id>.json   (per-paper table + hyperparameter extractions, verbatim-quoted)
  - results/*.json                                     (Method Scanner records: operations + v2 object typing)
  - phase23/diffuser/results_collected.csv             (Phase 23 DDIM sweep, seed-averaged -> 15 pilot rows)

Every inclusion/exclusion rule and every condition mapping is written here explicitly so the CSV is auditable.
Structure tuples follow compare.semantic_set(pass_no="v2", relaxed=False): canonical op (exact, then head verb)
@ v2 object type; `other`/`unknown` types become `other:<method_id>` (never shared, as in compare.py).

Usage: venv/bin/python outcome_predictor/build_dataset.py
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from app import RAW_TO_CANONICAL  # noqa: E402  (no side effects: server only starts under __main__)

TABLES = os.path.join(HERE, "sources", "tables")
MISSING, NA = "missing", "not_applicable"

NUMERIC = ["batch_size", "learning_rate", "training_steps", "hidden_dim", "num_layers", "discount",
           "planning_horizon", "context_length", "denoising_steps", "guidance_scale", "num_candidates",
           "num_q_networks", "model_rollout_length", "expectile_tau", "n_seeds"]
CATEGORICAL = ["sampler", "checkpoint_selection", "critic_layernorm", "policy_extraction", "base_learner", "discretization"]
COLUMNS = ["paper_id", "method_id", "method_structure", "lineage", "dataset", "dataset_version", "config_label",
           "outcome", "outcome_std"] + NUMERIC + CATEGORICAL + ["source", "value_type", "notes"]

ENVS = ("halfcheetah", "hopper", "walker2d")
QUALITY = {"random": "random", "r": "random", "medium": "medium", "m": "medium", "med": "medium",
           "medium-replay": "medium-replay", "m-re": "medium-replay", "med-rep": "medium-replay", "mixed": "medium-replay",
           "medium-reply": "medium-replay", "medium-expert": "medium-expert", "m-e": "medium-expert", "med-exp": "medium-expert",
           "expert": "expert", "full-replay": "full-replay"}


def normalize_dataset(name):
    """'halfcheetah-medium-replay-v2' -> ('halfcheetah-medium-replay', 'v2'); returns (None, None) if not a clean locomotion name."""
    s = name.strip().lower()
    version = None
    m = re.search(r"-(v\d)$", s)
    if m:
        version, s = m.group(1), s[: m.start()]
    env = "walker2d" if s.startswith("walker") else s.split("-")[0]
    if env not in ENVS:
        return None, None
    rest = s[len(s.split("-")[0]) + 1:]
    if rest not in QUALITY:
        return None, None
    return f"{env}-{QUALITY[rest]}", version


# ---------------------------------------------------------------- structure from scanner records

def scanner_record(arxiv_id):
    recs = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results", "*.json"))]
    recs = [r for r in recs if (r.get("source") or "").startswith(f"test:{arxiv_id} ")]
    if len(recs) != 1:
        raise SystemExit(f"expected exactly one scanner record for {arxiv_id}, found {len(recs)}")
    return recs[0]


def structure_tuples(rec, method_id):
    rows = rec.get("semantic_ops_v2")
    if not rows:
        raise SystemExit(f"record for {rec['paper'][:40]} has no semantic_ops_v2 — run semantic_type.py --pass v2")
    out = set()
    for s in rows:
        raw = s["op"].strip().lower()
        c = RAW_TO_CANONICAL.get(raw) or (RAW_TO_CANONICAL.get(raw.split()[0]) if " " in raw else None) or raw
        t = s["object_type"]
        if t in ("other", "unknown"):
            t = f"other:{method_id}"
        out.add(f"{c}@{t}")
    return sorted(out)


# ---------------------------------------------------------------- per-method specs
# Each spec: paper (arxiv id), table method names to pull rows from, lineage, base conditions,
# `include(row) -> config_label or None`, and `conditions(row, dataset) -> dict overrides`.

def base(**kw):
    d = {c: MISSING for c in NUMERIC + CATEGORICAL}
    d.update(kw)
    return d


def env_of(dataset):
    return dataset.split("-")[0]


SPECS = []


def spec(**kw):
    SPECS.append(kw)


# --- diffusion planning -------------------------------------------------------------------------
spec(method_id="Diffuser", paper="2205.09991", table_methods=["Diffuser"], lineage="diffusion_planning",
     base=base(batch_size=32, learning_rate=4e-5, training_steps=500000, discount=0.997, planning_horizon=32,
               context_length=NA, denoising_steps=20, guidance_scale=0.1, num_q_networks=NA, model_rollout_length=NA,
               expectile_tau=NA, n_seeds=150, sampler="ddpm"),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {"guidance_scale": 0.0001} if ds == "hopper-medium-expert" else {})

spec(method_id="DecisionDiffuser", paper="2211.15657", table_methods=["Decision Diffuser (DD)"], lineage="diffusion_planning",
     base=base(batch_size=32, learning_rate=2e-4, training_steps=2000000, planning_horizon=100, context_length=20,
               denoising_steps=100, num_q_networks=NA, model_rollout_length=NA, expectile_tau=NA, n_seeds=5, sampler="ddpm"),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {})

spec(method_id="AdaptDiffuser", paper="2302.01877", table_methods=["AdaptDiffuser"], lineage="diffusion_planning",
     base=base(batch_size=32, learning_rate=2e-4, training_steps=1000000, planning_horizon=32, context_length=NA,
               denoising_steps=100, num_q_networks=NA, model_rollout_length=NA, expectile_tau=NA, n_seeds=3, sampler="ddpm"),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {})

spec(method_id="HierarchicalDiffuser", paper="2401.02644", table_methods=["HD-DA"], lineage="diffusion_planning",
     base=base(planning_horizon=32, context_length=NA, num_q_networks=NA, model_rollout_length=NA, expectile_tau=NA,
               n_seeds=5, sampler="ddpm"),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {})

# --- diffusion policy ---------------------------------------------------------------------------
spec(method_id="DiffusionQL", paper="2208.06193", table_methods=["Diffusion-QL"], lineage="diffusion_policy",
     base=base(batch_size=256, learning_rate=3e-4, training_steps=2000000, hidden_dim=256, num_layers=3,
               planning_horizon=NA, context_length=NA, denoising_steps=5, guidance_scale=NA, num_candidates=1,
               num_q_networks=2, model_rollout_length=NA, expectile_tau=NA, n_seeds=5, sampler="ddpm",
               checkpoint_selection="offline_criterion"),
     include=lambda r: {"main": "main", "online model selection": "online_selection"}.get(r["config_label"]),
     conditions=lambda r, ds: {"checkpoint_selection": "best_online"} if r["config_label"] != "main" else {})

spec(method_id="IDQL", paper="2304.10573", table_methods=["IDQL-A", "IDQL-1"], lineage="diffusion_policy",
     base=base(batch_size=1024, learning_rate=3e-4, hidden_dim=256, num_layers=3, planning_horizon=NA, context_length=NA,
               denoising_steps=5, guidance_scale=NA, model_rollout_length=NA, expectile_tau=0.7, n_seeds=10, sampler="ddpm"),
     include=lambda r: "IDQL-A" if r["_method"] == "IDQL-A" else "IDQL-1",
     conditions=lambda r, ds: ({"training_steps": 3000000, "num_candidates": 128} if r["_method"] == "IDQL-A"
                               else {"training_steps": 2000000, "num_candidates": 64}))

spec(method_id="SfBC", paper="2209.14548", table_methods=["SfBC"], lineage="diffusion_policy",
     base=base(batch_size=4096, learning_rate=1e-4, planning_horizon=NA, context_length=NA, denoising_steps=15,
               guidance_scale=NA, num_candidates=32, model_rollout_length=NA, expectile_tau=NA, n_seeds=4, sampler="dpm_solver"),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {})

QGPO_S = {"halfcheetah-medium-expert": 3.0, "hopper-medium-expert": 2.0, "walker2d-medium-expert": 5.0,
          "halfcheetah-medium": 10.0, "hopper-medium": 8.0, "walker2d-medium": 10.0,
          "halfcheetah-medium-replay": 8.0, "hopper-medium-replay": 3.0, "walker2d-medium-replay": 5.0}
spec(method_id="QGPO", paper="2304.12824", table_methods=["QGPO"], lineage="diffusion_policy",
     base=base(batch_size=4096, learning_rate=1e-4, training_steps=600000, planning_horizon=NA, context_length=NA,
               denoising_steps=15, num_candidates=1, num_q_networks=2, model_rollout_length=NA, expectile_tau=NA,
               n_seeds=5, sampler="dpm_solver", checkpoint_selection="last"),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {"guidance_scale": QGPO_S[ds]})

EDP_METHODS = {"EDP (Diffusion-QL + EDP)": "dql", "EDP + TD3": "td3", "EDP + CRR": "crr", "EDP + IQL": "iql"}


def edp_include(r):
    lab = r["config_label"]
    if lab in ("main (RAT)",):
        return f"{EDP_METHODS[r['_method']]}_RAT"
    if lab in ("OMS (Best)", "OMS, K=1000"):
        return f"{EDP_METHODS[r['_method']]}_OMS"
    return None  # 'OMS, K=100, action approximation' excluded (approximation variant)


spec(method_id="EDP", paper="2305.20081", table_methods=list(EDP_METHODS), lineage="diffusion_policy",
     base=base(batch_size=256, learning_rate=3e-4, training_steps=2000000, hidden_dim=256, num_layers=3,
               planning_horizon=NA, context_length=NA, denoising_steps=15, guidance_scale=NA, num_candidates=10,
               model_rollout_length=NA, expectile_tau=NA, n_seeds=5, sampler="dpm_solver"),
     include=edp_include,
     conditions=lambda r, ds: {"base_learner": EDP_METHODS[r["_method"]],
                               "checkpoint_selection": "last" if "RAT" in r["config_label"] else "best_online",
                               **({"expectile_tau": 0.7, "num_q_networks": 2} if EDP_METHODS[r["_method"]] == "iql" else {}),
                               **({"num_q_networks": 2} if EDP_METHODS[r["_method"]] == "dql" else {})})

SRPO_BETA = {"halfcheetah-medium-expert": 0.01, "hopper-medium-expert": 0.01, "walker2d-medium-expert": 0.1,
             "halfcheetah-medium": 0.2, "hopper-medium": 0.05, "walker2d-medium": 0.05,
             "halfcheetah-medium-replay": 0.2, "hopper-medium-replay": 0.2, "walker2d-medium-replay": 0.5}
spec(method_id="SRPO", paper="2310.07297", table_methods=["SRPO"], lineage="diffusion_policy",
     base=base(batch_size=256, learning_rate=3e-4, training_steps=1000000, hidden_dim=256, num_layers=2,
               planning_horizon=NA, context_length=NA, denoising_steps=0, guidance_scale=NA, num_candidates=NA,
               model_rollout_length=NA, expectile_tau=0.7, n_seeds=6, sampler=NA, checkpoint_selection="last"),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {})

# --- offline Q-learning / actor-critic ----------------------------------------------------------
spec(method_id="CQL", paper="2006.04779", table_methods=["CQL(H)"], lineage="offline_q_learning",
     base=base(learning_rate=3e-5, training_steps=1000000, planning_horizon=NA, context_length=NA, denoising_steps=NA,
               guidance_scale=NA, num_candidates=NA, num_q_networks=2, model_rollout_length=NA, expectile_tau=NA,
               n_seeds=4, sampler=NA),
     include=lambda r: "main" if r["config_label"] == "main" and "raw" not in (r.get("notes") or "").lower() else None,
     conditions=lambda r, ds: {})

spec(method_id="IQL", paper="2110.06169", table_methods=["IQL"], lineage="offline_q_learning",
     base=base(learning_rate=3e-4, training_steps=1000000, hidden_dim=256, num_layers=2, planning_horizon=NA,
               context_length=NA, denoising_steps=NA, guidance_scale=NA, num_candidates=NA, num_q_networks=2,
               model_rollout_length=NA, expectile_tau=0.7, n_seeds=10, sampler=NA),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {})

spec(method_id="TD3+BC", paper="2106.06860", table_methods=["TD3+BC"], lineage="behavior_regularized_actor_critic",
     base=base(batch_size=256, learning_rate=3e-4, training_steps=1000000, hidden_dim=256, num_layers=2, discount=0.99,
               planning_horizon=NA, context_length=NA, denoising_steps=NA, guidance_scale=NA, num_candidates=NA,
               num_q_networks=2, model_rollout_length=NA, expectile_tau=NA, n_seeds=5, sampler=NA, checkpoint_selection="last"),
     include=lambda r: {"main": "main_v0", "main (v2 datasets)": "main_v2"}.get(r["config_label"]),
     conditions=lambda r, ds: {})

REBRAC_LABELS = {"main": ("main", {}),
                 "layernorm=off": ("no_layernorm", {"critic_layernorm": "no", "n_seeds": 4}),
                 "num_hidden_layers=2 (w/o extra layer)": ("layers_2", {"num_layers": 2, "n_seeds": 4}),
                 "batch_size=256 (w/o large batch)": ("batch_256", {"batch_size": 256, "n_seeds": 4})}
spec(method_id="ReBRAC", paper="2305.09836", table_methods=["ReBRAC"], lineage="behavior_regularized_actor_critic",
     base=base(batch_size=1024, learning_rate=1e-3, training_steps=1000000, hidden_dim=256, num_layers=3, discount=0.99,
               planning_horizon=NA, context_length=NA, denoising_steps=NA, guidance_scale=NA, num_candidates=NA,
               num_q_networks=2, model_rollout_length=NA, expectile_tau=NA, n_seeds=10, sampler=NA,
               checkpoint_selection="last", critic_layernorm="yes"),
     include=lambda r: REBRAC_LABELS[r["config_label"]][0] if r["config_label"] in REBRAC_LABELS else None,
     conditions=lambda r, ds: REBRAC_LABELS[r["config_label"]][1])

spec(method_id="FisherBRC", paper="2103.08050", table_methods=["F-BRC"], lineage="behavior_regularized_actor_critic",
     base=base(training_steps=1000000, hidden_dim=256, num_layers=3, planning_horizon=NA, context_length=NA,
               denoising_steps=NA, guidance_scale=NA, num_candidates=NA, model_rollout_length=NA, expectile_tau=NA,
               n_seeds=5, sampler=NA, checkpoint_selection="last"),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {})

spec(method_id="PLAS", paper="2011.07213", table_methods=["PLAS"], lineage="behavior_regularized_actor_critic",
     base=base(batch_size=100, learning_rate=1e-4, training_steps=500000, hidden_dim=400, num_layers=2,
               planning_horizon=NA, context_length=NA, denoising_steps=NA, guidance_scale=NA, num_candidates=NA,
               num_q_networks=2, model_rollout_length=NA, expectile_tau=NA, n_seeds=3, sampler=NA),
     include=lambda r: "main" if r["config_label"] == "main" and "(" not in r["dataset"] else None,
     conditions=lambda r, ds: {})

ONESTEP = {"One-step Easy BCQ": "easy_bcq", "One-step Rev. KL Reg": "rev_kl", "One-step Exp. Weight": "exp_weight"}
spec(method_id="OneStepRL", paper="2106.08909", table_methods=list(ONESTEP), lineage="behavior_regularized_actor_critic",
     base=base(batch_size=512, learning_rate=1e-4, training_steps=100000, hidden_dim=1024, num_layers=2,
               planning_horizon=NA, context_length=NA, denoising_steps=NA, guidance_scale=NA, num_candidates=NA,
               model_rollout_length=NA, expectile_tau=NA, n_seeds=10, sampler=NA, checkpoint_selection="last"),
     include=lambda r: ONESTEP[r["_method"]] if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {"policy_extraction": ONESTEP[r["_method"]]})

spec(method_id="EDAC", paper="2110.01548", table_methods=["SAC-N", "EDAC"], lineage="uncertainty_based_offline_rl",
     base=base(training_steps=3000000, num_layers=3, planning_horizon=NA, context_length=NA, denoising_steps=NA,
               guidance_scale=NA, num_candidates=NA, model_rollout_length=NA, expectile_tau=NA, n_seeds=4, sampler=NA),
     include=lambda r: ("SAC-N_eta0" if r["_method"] == "SAC-N" else "EDAC") if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {"num_q_networks": int(r["config_conditions"]["num_q_networks"])})

# --- model-based offline RL ---------------------------------------------------------------------
spec(method_id="MOPO", paper="2005.13239", table_methods=["MOPO"], lineage="model_based_offline_rl",
     base=base(batch_size=256, planning_horizon=NA, context_length=NA, denoising_steps=NA, guidance_scale=NA,
               num_candidates=NA, num_q_networks=2, expectile_tau=NA, n_seeds=6, sampler=NA, checkpoint_selection="last"),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {"model_rollout_length": MOPO_H[ds]})
MOPO_H = {"halfcheetah-random": 5, "hopper-random": 5, "walker2d-random": 1, "halfcheetah-medium": 1, "hopper-medium": 5,
          "walker2d-medium": 5, "halfcheetah-medium-replay": 5, "hopper-medium-replay": 5, "walker2d-medium-replay": 1,
          "halfcheetah-medium-expert": 5, "hopper-medium-expert": 5, "walker2d-medium-expert": 1}

spec(method_id="MOReL", paper="2005.05951", table_methods=["MOReL"], lineage="model_based_offline_rl",
     base=base(batch_size=256, learning_rate=5e-4, hidden_dim=32, num_layers=2, planning_horizon=NA, context_length=NA,
               denoising_steps=NA, guidance_scale=NA, num_candidates=NA, num_q_networks=NA, expectile_tau=NA, n_seeds=3,
               sampler=NA, checkpoint_selection="last"),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {"model_rollout_length": 500 if env_of(ds) == "halfcheetah" else 400})

COMBO_LR = {"halfcheetah": 1e-4, "hopper": 1e-4, "walker2d": 1e-5}
COMBO_H = {"halfcheetah": 5, "hopper": 5, "walker2d": 1}
spec(method_id="COMBO", paper="2102.08363", table_methods=["COMBO"], lineage="model_based_offline_rl",
     base=base(planning_horizon=NA, context_length=NA, denoising_steps=NA, guidance_scale=NA, num_candidates=NA,
               num_q_networks=2, expectile_tau=NA, n_seeds=6, sampler=NA, checkpoint_selection="last"),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {"learning_rate": COMBO_LR[env_of(ds)], "model_rollout_length": COMBO_H[env_of(ds)]})

# --- return-conditioned supervised learning -----------------------------------------------------
spec(method_id="DecisionTransformer", paper="2106.01345", table_methods=["DT (Ours)"], lineage="return_conditioned_supervised_learning",
     base=base(batch_size=64, learning_rate=1e-4, training_steps=100000, hidden_dim=128, num_layers=3, discount=NA,
               planning_horizon=NA, context_length=20, denoising_steps=NA, guidance_scale=NA, num_candidates=NA,
               num_q_networks=NA, model_rollout_length=NA, expectile_tau=NA, n_seeds=3, sampler=NA),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {})

DC_BIG = {"hopper-medium", "hopper-medium-replay"}
spec(method_id="DecisionConvFormer", paper="2310.03022", table_methods=["DC"], lineage="return_conditioned_supervised_learning",
     base=base(batch_size=64, training_steps=100000, num_layers=3, discount=NA, planning_horizon=NA, context_length=8,
               denoising_steps=NA, guidance_scale=NA, num_candidates=NA, num_q_networks=NA, model_rollout_length=NA,
               expectile_tau=NA, n_seeds=5, sampler=NA),
     include=lambda r: {"main": "main", "embedding_dim=128": "embed_128", "K=20, L=6": "context_20"}.get(r["config_label"]),
     conditions=lambda r, ds: {"hidden_dim": 128 if r["config_label"] == "embedding_dim=128" else (256 if ds in DC_BIG else 128),
                               "learning_rate": 1e-4 if ds in (DC_BIG | {"walker2d-medium"}) else 1e-3,
                               **({"context_length": 20} if r["config_label"] == "K=20, L=6" else {})})

spec(method_id="QDT", paper="2209.03993", table_methods=["QDT"], lineage="return_conditioned_supervised_learning",
     base=base(batch_size=64, training_steps=100000, discount=NA, planning_horizon=NA, denoising_steps=NA, guidance_scale=NA,
               num_candidates=NA, num_q_networks=NA, model_rollout_length=NA, expectile_tau=NA, n_seeds=5, sampler=NA,
               checkpoint_selection="last"),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {})

spec(method_id="RvS", paper="2112.10751", table_methods=["RvS-R"], lineage="return_conditioned_supervised_learning",
     base=base(batch_size=16384, learning_rate=1e-3, hidden_dim=1024, num_layers=2, discount=NA, planning_horizon=NA,
               context_length=NA, denoising_steps=NA, guidance_scale=NA, num_candidates=NA, num_q_networks=NA,
               model_rollout_length=NA, expectile_tau=NA, n_seeds=5, sampler=NA),
     include=lambda r: "main" if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {})

# --- sequence-model planning --------------------------------------------------------------------
spec(method_id="TrajectoryTransformer", paper="2106.02039", table_methods=["TT (uniform)", "TT (quantile)"], lineage="sequence_model_planning",
     base=base(batch_size=256, learning_rate=2.5e-4, hidden_dim=128, num_layers=4, planning_horizon=15, context_length=5,
               denoising_steps=NA, guidance_scale=NA, num_candidates=256, num_q_networks=NA, model_rollout_length=NA,
               expectile_tau=NA, n_seeds=15, sampler=NA),
     include=lambda r: ("uniform" if "uniform" in r["_method"] else "quantile") if r["config_label"] == "main" else None,
     conditions=lambda r, ds: {"discretization": "uniform" if "uniform" in r["_method"] else "quantile"})

TAP_LABELS = {"main": ("main", {}), "horizon=3": ("horizon_3", {"planning_horizon": 3}),
              "horizon=9": ("horizon_9", {"planning_horizon": 9}), "horizon=21": ("horizon_21", {"planning_horizon": 21})}
spec(method_id="TAP", paper="2208.10291", table_methods=["TAP"], lineage="sequence_model_planning",
     base=base(batch_size=512, learning_rate=2e-4, hidden_dim=512, num_layers=4, discount=0.99, planning_horizon=15,
               denoising_steps=NA, guidance_scale=NA, num_candidates=64, num_q_networks=NA, model_rollout_length=NA,
               expectile_tau=NA, n_seeds=5, sampler=NA),
     include=lambda r: TAP_LABELS[r["config_label"]][0] if r["config_label"] in TAP_LABELS else None,
     conditions=lambda r, ds: TAP_LABELS[r["config_label"]][1])


# ---------------------------------------------------------------- assemble

def fmt(v):
    if isinstance(v, float):
        return repr(v) if v != int(v) else str(int(v))
    return str(v)


def build():
    out = []
    audit = defaultdict(lambda: defaultdict(int))
    for sp in SPECS:
        tab = json.load(open(os.path.join(TABLES, sp["paper"] + ".json")))
        rec = scanner_record(sp["paper"])
        struct = json.dumps(structure_tuples(rec, sp["method_id"]))
        for m in tab["methods"]:
            if m["method_name"] not in sp["table_methods"]:
                continue
            paper_version = (m.get("dataset_version") or {}).get("value", "unknown")
            for r in m["rows"]:
                r = dict(r, _method=m["method_name"])
                label = sp["include"](r)
                if label is None:
                    audit[sp["method_id"]]["excluded:" + r["config_label"]] += 1
                    continue
                ds, ver = normalize_dataset(r["dataset"])
                if ds is None:
                    audit[sp["method_id"]]["excluded:unparseable dataset " + r["dataset"]] += 1
                    continue
                if (r.get("notes") or "").lower().startswith("raw return") or "raw return" in (r.get("notes") or "").lower():
                    audit[sp["method_id"]]["excluded:raw return"] += 1
                    continue
                if r["config_label"] == "main (v2 datasets)":
                    ver = "v2"
                version = ver or (paper_version if paper_version in ("v0", "v2") else "unknown")
                cond = dict(sp["base"])
                cond.update(sp["conditions"](r, ds))
                row = {"paper_id": sp["paper"], "method_id": sp["method_id"], "method_structure": struct,
                       "lineage": sp["lineage"], "dataset": ds, "dataset_version": version, "config_label": label,
                       "outcome": r["outcome"], "outcome_std": "" if r.get("std") is None else r["std"],
                       "source": f"arXiv:{sp['paper']} {r['table']} [{m['method_name']}] dataset as printed: {r['dataset']}",
                       "value_type": "reported_mean_over_seeds" if r.get("outcome_is_mean_over_seeds") else "reported",
                       "notes": (r.get("notes") or "").replace("\n", " ")}
                row.update({k: fmt(cond[k]) for k in NUMERIC + CATEGORICAL})
                out.append(row)
                audit[sp["method_id"]]["included:" + label] += 1

    # --- Phase 23 pilot rows: 3 envs x 5 sampler configs, seed-averaged (score x100)
    p23 = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(ROOT, "phase23/results_collected.csv"))):
        p23[(r["env"], r["sampler"], int(r["steps"]))].append(float(r["score"]) * 100)
    diffuser = next(s for s in SPECS if s["method_id"] == "Diffuser")
    struct = json.dumps(structure_tuples(scanner_record("2205.09991"), "Diffuser"))
    for (env, sampler, steps), scores in sorted(p23.items()):
        ds, ver = normalize_dataset(env)
        cond = dict(diffuser["base"])
        cond.update({"sampler": sampler, "denoising_steps": steps, "planning_horizon": 4 if env.startswith("halfcheetah") else 32,
                     "guidance_scale": 0.001 if env.startswith("halfcheetah") else 0.1, "num_candidates": 64, "n_seeds": 3,
                     "training_steps": 800000, "checkpoint_selection": "last"})
        import statistics
        row = {"paper_id": "phase23_ddim_sweep", "method_id": "Diffuser", "method_structure": struct, "lineage": "diffusion_planning",
               "dataset": ds, "dataset_version": ver, "config_label": f"{sampler}-{steps}",
               "outcome": round(statistics.mean(scores), 2), "outcome_std": round(statistics.stdev(scores), 2),
               "source": f"phase23/results_collected.csv env={env} sampler={sampler} steps={steps}, mean of seeds 0-2 x100",
               "value_type": "own_run_mean_over_seeds",
               "notes": "Phase 23 pilot rows: gymnasium v4 dynamics with D4RL v2 data (see phase23_results.md); pretrained Diffuser checkpoint, 1000-step episodes"}
        row.update({k: fmt(cond[k]) for k in NUMERIC + CATEGORICAL})
        out.append(row)
        audit["Diffuser"][f"included:phase23 {sampler}-{steps}"] += 1

    with open(os.path.join(HERE, "d4rl_dataset.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    json.dump(audit, open(os.path.join(HERE, "sources", "build_audit.json"), "w"), indent=1)
    per_method = defaultdict(int)
    for r in out:
        per_method[r["method_id"]] += 1
    print(f"rows={len(out)} methods={len(per_method)} papers={len({r['paper_id'] for r in out})}")
    for k, v in sorted(per_method.items()):
        print(f"  {k:24}{v:4}")


if __name__ == "__main__":
    build()
