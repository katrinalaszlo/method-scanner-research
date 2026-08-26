"""Phase 26a: assemble atari100k_dataset.csv from sources/*.json + scanner records. Rules per schema26.md.
Outcome = 100 x human-normalized score, using the Random/Human columns printed in the same paper.
Usage: venv/bin/python outcome_predictor/phase26/build_atari.py"""
import csv, glob, json, os, re, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); P24 = os.path.dirname(HERE); ROOT = os.path.dirname(P24)
sys.path.insert(0, P24)
from build_dataset import structure_tuples  # same tuple rule as Phase 24

MISSING, NA = "missing", "not_applicable"
NUMERIC = ["env_steps", "replay_ratio", "batch_size", "learning_rate", "n_step", "discount", "hidden_dim", "num_layers",
           "imagination_horizon", "num_simulations", "model_params", "n_seeds", "eval_episodes"]
CATEGORICAL = ["sticky_actions", "data_augmentation", "checkpoint_selection"]
COLUMNS = ["paper_id", "method_id", "method_structure", "lineage", "dataset", "dataset_version", "config_label", "outcome",
           "raw_score", "random_ref", "human_ref"] + NUMERIC + CATEGORICAL + ["source", "value_type", "notes"]
GAMES = ["alien", "amidar", "assault", "asterix", "bankheist", "battlezone", "boxing", "breakout", "choppercommand", "crazyclimber",
         "demonattack", "freeway", "frostbite", "gopher", "hero", "jamesbond", "kangaroo", "krull", "kungfumaster", "mspacman",
         "pong", "privateeye", "qbert", "roadrunner", "seaquest", "upndown"]
ALIAS = {"choppercmd": "choppercommand"}


def game(name):
    g = re.sub(r"[^a-z]", "", name.lower()); g = ALIAS.get(g, g)
    return g if g in GAMES else None


def record(aid):
    recs = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results", "*.json"))]
    recs = [r for r in recs if (r.get("source") or "").startswith((f"test:{aid} ", f"arxiv:{aid} "))]
    assert len(recs) == 1, (aid, len(recs)); return recs[0]


def base(**kw):
    d = {c: MISSING for c in NUMERIC + CATEGORICAL}; d["env_steps"] = 100000; d.update(kw); return d


SPECS = [
 dict(method_id="SimPLe", paper="1903.00374", table_methods=["SimPLe"], lineage="world_model_imagination",
      labels={"main (Ours, SD long)": "main"},
      base=base(env_steps=102400, n_step=NA, discount=0.99, imagination_horizon=50, num_simulations=NA, model_params=74, n_seeds=5,
                sticky_actions="no", checkpoint_selection="last")),
 dict(method_id="DER", paper="1906.05243", table_methods=["Data-efficient Rainbow (DER)"], lineage="model_free_value", labels={"main": "main"},
      base=base(replay_ratio=1, batch_size=32, learning_rate=1e-4, n_step=20, discount=0.99, hidden_dim=256, imagination_horizon=NA,
                num_simulations=NA, n_seeds=5, sticky_actions="no", data_augmentation="no", checkpoint_selection="last")),
 dict(method_id="OTRainbow", paper="2003.10181", table_methods=["OTRainbow (Overtrained Rainbow)"], lineage="model_free_value", labels={"main": "main"},
      base=base(replay_ratio=8, imagination_horizon=NA, num_simulations=NA, data_augmentation="no", checkpoint_selection="last")),
 dict(method_id="CURL", paper="2004.04136", table_methods=["CURL (on Efficient Rainbow)"], lineage="model_free_representation", labels={"main": "main"},
      base=base(replay_ratio=1, batch_size=32, learning_rate=1e-4, n_step=20, discount=0.99, hidden_dim=256, imagination_horizon=NA,
                num_simulations=NA, n_seeds=20, data_augmentation="yes", checkpoint_selection="last")),
 dict(method_id="DrQ", paper="2004.13649", table_methods=["DrQ (Eff. DQN + DrQ)"], lineage="model_free_representation", labels={"main": "main"},
      base=base(replay_ratio=1, batch_size=32, learning_rate=1e-4, n_step=10, discount=0.99, hidden_dim=512, num_layers=3, imagination_horizon=NA,
                num_simulations=NA, n_seeds=5, data_augmentation="yes", checkpoint_selection="last")),
 dict(method_id="SPR", paper="2007.05929", table_methods=["SPR"], lineage="model_free_representation", labels={"main": "main", "no_aug": "no_aug"},
      base=base(replay_ratio=2, batch_size=32, learning_rate=1e-4, n_step=10, discount=0.99, num_layers=3, imagination_horizon=NA,
                num_simulations=NA, n_seeds=10, eval_episodes=100, data_augmentation="yes", checkpoint_selection="last"),
      overrides={"no_aug": {"data_augmentation": "no"}}),
 dict(method_id="PlayVirtual", paper="2106.04152", table_methods=["PlayVirtual"], lineage="model_free_representation", labels={"main": "main"},
      base=base(replay_ratio=2, batch_size=32, learning_rate=1e-4, n_step=10, discount=0.99, hidden_dim=256, num_layers=3, imagination_horizon=NA,
                num_simulations=NA, n_seeds=15, eval_episodes=100, data_augmentation="yes", checkpoint_selection="last")),
 dict(method_id="MLR", paper="2201.12096", table_methods=["MLR"], lineage="model_free_representation", labels={"main": "main"},
      base=base(replay_ratio=2, batch_size=32, learning_rate=1e-4, n_step=10, discount=0.99, hidden_dim=256, num_layers=3, imagination_horizon=NA,
                num_simulations=NA, n_seeds=3, eval_episodes=100, data_augmentation="yes", checkpoint_selection="last")),
 dict(method_id="EfficientZero", paper="2111.00210", table_methods=["EfficientZero"], lineage="mcts_planning",
      labels={"main": "main", "w.o. data augmentation": "no_aug"},
      base=base(replay_ratio=1.2, batch_size=256, learning_rate=0.2, n_step=5, discount=0.9974, imagination_horizon=NA, num_simulations=50,
                n_seeds=3, eval_episodes=32, data_augmentation="yes"),
      overrides={"no_aug": {"data_augmentation": "no", "n_seeds": MISSING}}),
 dict(method_id="IRIS", paper="2209.00588", table_methods=["IRIS"], lineage="world_model_imagination", labels={"main": "main"},
      base=base(replay_ratio=1, batch_size=64, learning_rate=1e-4, n_step=NA, discount=0.995, hidden_dim=256, num_layers=10, imagination_horizon=20,
                num_simulations=NA, n_seeds=5, eval_episodes=100, checkpoint_selection="last")),
 dict(method_id="TWM", paper="2303.07109", table_methods=["TWM"], lineage="world_model_imagination", labels={"main": "main"},
      base=base(batch_size=100, learning_rate=1e-4, n_step=NA, discount=0.99, hidden_dim=256, num_layers=10, imagination_horizon=15,
                num_simulations=NA, model_params=21.6, n_seeds=5, eval_episodes=100, checkpoint_selection="last")),
 dict(method_id="DreamerV3", paper="2301.04104", table_methods=["DreamerV3 (printed as 'Dreamer' in Table 9)"], lineage="world_model_imagination", labels={"main": "main"},
      base=base(replay_ratio=128, batch_size=16, learning_rate=4e-5, n_step=NA, discount=0.997, hidden_dim=1024, imagination_horizon=15,
                num_simulations=NA, model_params=200, n_seeds=5, checkpoint_selection="last")),
 dict(method_id="BBF", paper="2305.19452", table_methods=["BBF"], lineage="model_free_representation", labels={"main": "main"},
      base=base(replay_ratio=8, n_step=3, discount=0.997, num_layers=15, imagination_horizon=NA, num_simulations=NA, n_seeds=50,
                sticky_actions="no", checkpoint_selection="last")),
 dict(method_id="STORM", paper="2310.09615", table_methods=["STORM"], lineage="world_model_imagination", labels={"main": "main"},
      base=base(replay_ratio=1, batch_size=16, learning_rate=1e-4, n_step=NA, discount=0.985, hidden_dim=512, num_layers=2, imagination_horizon=16,
                num_simulations=NA, n_seeds=5, eval_episodes=20, checkpoint_selection="last")),
 dict(method_id="DIAMOND", paper="2405.12399", table_methods=["DIAMOND"], lineage="world_model_imagination", labels={"main": "main"},
      base=base(replay_ratio=4, batch_size=32, learning_rate=1e-4, n_step=NA, discount=0.985, imagination_horizon=15, num_simulations=NA, n_seeds=5)),
 dict(method_id="EfficientZeroV2", paper="2403.00564", table_methods=["EZ-V2"], lineage="mcts_planning", labels={"main": "main"},
      base=base(replay_ratio=1, batch_size=256, learning_rate=0.2, n_step=5, discount=0.997, imagination_horizon=NA, num_simulations=16)),
 dict(method_id="DeltaIRIS", paper="2406.19320", table_methods=["∆-IRIS", "Δ-IRIS", "Delta-IRIS"], lineage="world_model_imagination", labels={"main": "main"},
      base=base(batch_size=32, learning_rate=1e-4, n_step=NA, discount=0.997, hidden_dim=512, num_layers=3, imagination_horizon=15,
                num_simulations=NA, n_seeds=5, eval_episodes=100, checkpoint_selection="last")),
]


def fmt(v):
    return (repr(v) if v != int(v) else str(int(v))) if isinstance(v, float) else str(v)


def main():
    out, audit = [], defaultdict(lambda: defaultdict(int))
    for sp in SPECS:
        tab = json.load(open(os.path.join(HERE, "sources", sp["paper"] + ".json")))
        refs = {game(k): v for k, v in tab.get("reference_scores", {}).items() if game(k)}
        struct = json.dumps(structure_tuples(record(sp["paper"]), sp["method_id"]))
        ms = [m for m in tab["methods"] if m["method_name"] in sp["table_methods"]]
        assert ms, (sp["method_id"], [m["method_name"] for m in tab["methods"]])
        for m in ms:
            for r in m["rows"]:
                label = sp["labels"].get(r["config_label"])
                if label is None: audit[sp["method_id"]]["excluded:" + r["config_label"]] += 1; continue
                g = game(r["dataset"])
                if g is None: audit[sp["method_id"]]["excluded:game " + r["dataset"]] += 1; continue
                if g not in refs or refs[g].get("human") in (None, "-") or refs[g].get("random") is None:
                    audit[sp["method_id"]]["excluded:no reference " + g] += 1; continue
                hum, rnd = float(refs[g]["human"]), float(refs[g]["random"])
                if "normalized" in (r.get("notes") or "").lower(): audit[sp["method_id"]]["excluded:printed normalized"] += 1; continue
                hns = 100 * (float(r["outcome"]) - rnd) / (hum - rnd)
                cond = dict(sp["base"]); cond.update(sp.get("overrides", {}).get(label, {}))
                row = {"paper_id": sp["paper"], "method_id": sp["method_id"], "method_structure": struct, "lineage": sp["lineage"],
                       "dataset": g, "dataset_version": "atari100k", "config_label": label, "outcome": round(hns, 3),
                       "raw_score": r["outcome"], "random_ref": rnd, "human_ref": hum,
                       "source": f"arXiv:{sp['paper']} {r['table']} [{m['method_name']}] game as printed: {r['dataset']}; refs {refs[g].get('table','')}",
                       "value_type": "derived_hns_from_reported_mean", "notes": (r.get("notes") or "").replace("\n", " ")}
                row.update({k: fmt(cond[k]) for k in NUMERIC + CATEGORICAL})
                out.append(row); audit[sp["method_id"]]["included:" + label] += 1
    with open(os.path.join(HERE, "atari100k_dataset.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(out)
    json.dump({"numeric_conditions": NUMERIC, "categorical_conditions": CATEGORICAL, "exclude_config_labels": []},
              open(os.path.join(HERE, "schema26.json"), "w"), indent=1)
    json.dump(audit, open(os.path.join(HERE, "sources", "build_audit.json"), "w"), indent=1)
    per = defaultdict(int)
    for r in out: per[r["method_id"]] += 1
    print(f"rows={len(out)} methods={len(per)}"); [print(f"  {k:18}{v:4}") for k, v in sorted(per.items())]


if __name__ == "__main__":
    main()
