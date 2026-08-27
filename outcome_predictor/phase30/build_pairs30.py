"""Phase 30: select leaderboards, papers, pairs. Writes papers.json (paper set with abstracts) and pairs.csv.
Papers keyed by arXiv id. Leaderboard rows are streamed one task at a time from evaluation-tables-*.parquet (the flattened JSON is 9.8 GB)."""
import csv, glob, json, os, random, re
from collections import defaultdict
import pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); PWC = os.path.join(HERE, "pwc")
LOWER = re.compile(r"error|loss|perplex|ppl|\bfid\b|\bmae\b|rmse|\bmse\b|\bwer\b|\bcer\b|distance|latency|time|params|flops|\bepe\b|abs rel|sq rel|lpips|nll|bits|\bkid\b", re.I)
MAX_PAPERS, MAX_BOARDS_PER_TASK, MIN_PAPERS, PAIR_CAP, BOARD_PAPER_CAP = 900, 3, 15, 200, 40

papers = {}
for f in sorted(glob.glob(os.path.join(PWC, "papers-with-abstracts-*.parquet"))):
    for r in pq.read_table(f, columns=["paper_url", "arxiv_id", "title", "abstract", "date", "methods", "tasks"]).to_pylist():
        if r["arxiv_id"] and r["abstract"] and r["paper_url"]:
            r["date"] = str(r["date"])[:10] if r["date"] else None
            r["methods"] = [{"name": m["name"], "collection": (m.get("main_collection") or {}).get("name")} for m in (r["methods"] or [])]
            papers[re.sub(r"v\d+$", "", r["arxiv_id"])] = r  # keyed by arxiv id; leaderboard rows link arxiv.org URLs
print("papers with arxiv abstract:", len(papers))
ARX = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/[0-9]{7})")


def num(v):
    m = re.search(r"-?\d+(\.\d+)?", str(v).replace(",", "")) if v is not None else None
    return float(m.group()) if m else None


boards = defaultdict(dict)  # (task,dataset) -> paper_url -> (score, metric)


def take(task, ds, sub=None):
    sota = ds.get("sota") or {}
    order = sota.get("metrics") or []
    name = ds["dataset"] + (f" / {sub}" if sub else "")
    for r in sota.get("rows") or []:
        m = ARX.search(r.get("paper_url") or ""); u = m.group(1) if m else None
        if not u or u not in papers or not order: continue
        metric = order[0]; s = num((r.get("metrics") or {}).get(metric))
        if s is None: continue
        key = (task, name)
        if u not in boards[key] or s > boards[key][u][0]:
            boards[key][u] = (s, metric)
    for sd in ds.get("subdatasets") or []:
        take(task, sd, sub=sd["dataset"] if not sub else f"{sub} / {sd['dataset']}")


def walk(t):
    for ds in t.get("datasets") or []: take(t["task"], ds)
    for st in t.get("subtasks") or []: walk(st)


for f in sorted(glob.glob(os.path.join(PWC, "evaluation-tables-*.parquet"))):
    pf = pq.ParquetFile(f)
    for b in pf.iter_batches(batch_size=1, columns=["task", "subtasks", "datasets"]):
        for t in b.to_pylist(): walk(t)
print("boards total:", len(boards))
random.seed(0)
cands = sorted(((k, v) for k, v in boards.items() if len(v) >= MIN_PAPERS), key=lambda kv: -len(kv[1]))
chosen, per_task, covered = [], defaultdict(int), set()
for k, v in cands:
    if per_task[k[0]] >= MAX_BOARDS_PER_TASK: continue
    if len(v) > BOARD_PAPER_CAP:  # amendment 1 (schema30.md): random subsample so big boards don't exhaust the paper budget
        keep = random.sample(sorted(v), BOARD_PAPER_CAP); v = {u: v[u] for u in keep}
    if len(covered | set(v)) > MAX_PAPERS: continue
    chosen.append((k, v)); per_task[k[0]] += 1; covered |= set(v)
pairs = []
for (task, ds), v in chosen:
    metric = next(iter(v.values()))[1]; lower = bool(LOWER.search(metric))
    items = list(v.items()); allp = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (pa, (sa, _)), (pb, (sb, _)) = items[i], items[j]
            if sa == sb: continue
            better = pa if ((sa < sb) if lower else (sa > sb)) else pb
            allp.append((pa, pb, better))
    random.shuffle(allp)
    for pa, pb, better in allp[:PAIR_CAP]:
        pairs.append({"board": f"{task} | {ds}", "task": task, "dataset": ds, "metric": metric, "lower_is_better": lower,
                      "paper_a": pa, "paper_b": pb, "score_a": v[pa][0], "score_b": v[pb][0], "label_a_better": int(better == pa)})
json.dump({u: papers[u] for u in covered}, open(os.path.join(HERE, "papers.json"), "w"))
with open(os.path.join(HERE, "pairs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
print(f"boards={len(chosen)} tasks={len({k[0] for k, _ in chosen})} papers={len(covered)} pairs={len(pairs)}")
for (task, ds), v in chosen: print(f"  {len(v):3} {task[:40]:40} | {ds[:35]:35} | {next(iter(v.values()))[1][:25]}")
