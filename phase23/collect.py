"""Phase 23: rebuild results_collected.csv from every logs/<env>/plans/*/p23_<sampler><steps>_s<seed>/rollout.json."""
import glob, json, re, csv, os
rows = []
for j in sorted(glob.glob("diffuser/logs/*/plans/*/p23_*/rollout.json")):
    env = j.split("/")[2]; suffix = j.split("/")[-2]
    m = re.match(r"p23_(ddpm|ddim)(\d+)_s(\d+)", suffix)
    d = json.load(open(j))
    rows.append([env, m.group(1), int(m.group(2)), int(m.group(3)), round(d["return"], 2), round(d["score"], 4), d["step"] + 1, int(d["term"])])
with open("results_collected.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["env", "sampler", "steps", "seed", "return", "score", "ep_len", "terminal"]); w.writerows(rows)
for r in rows: print(",".join(map(str, r)))
