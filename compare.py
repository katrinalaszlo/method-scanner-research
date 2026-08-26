"""Paper x paper Jaccard on canonical ops (or raw ops with --raw). Reads the running server.

Usage: venv/bin/python compare.py [--raw] [--semantic [--strict|--relaxed]] [--top N] [--min-shared K] [--drop OP ...]
--drop removes an op from every paper's set before computing (for ablations like H1: drop predict).
--query "title substring" scores one record (any source, e.g. a test: abstract) against the corpus instead of the matrix.
--families mechanism relabels papers with the mechanism families in families.json (lineage is the default).
--semantic matches on (canonical_op, object_type) tuples from semantic_type.py instead of bare ops.
  --pass 2 uses the pass-2 typing; --agreed keeps only tuples where pass 1 and pass 2 gave the same type.
  --strict (default): object types must be identical. --relaxed: types in one compatibility group
  (semantic_ontology.json) count as equal. Writes semantic_similarity_matrix.csv.
Top pairs require at least K shared ops (default 2) — Jaccard 1.0 on one-op sets is noise.
Writes similarity_matrix.csv; prints within/between-family means and top cross-family pairs.
"""
import csv
import itertools
import json
import re
import sys
import urllib.request
from collections import defaultdict

SERVER = "http://localhost:8000"
ONTOLOGY = json.load(open("semantic_ontology.json"))
GROUP_OF = {t: g for g, ts in ONTOLOGY["compatibility_groups"].items() if not g.startswith("_") for t in ts}


OWN_ONLY = False  # set by --own


def semantic_set(it, relaxed, pass_no=1, agreed=False):
    """(canonical_op, object_type) per raw op. Unmapped raw ops keep their raw string as the op.
    'other' object types never match anything — they fall back to a bare-op tuple with type 'other:<paper>'."""
    raw_to_canon = {}
    canon, head, un = it.get("canonical_ops", []), set(it.get("head_matched_ops", [])), set(it.get("unmapped_ops", []))
    # rebuild the raw->canonical map the server used (exact, then head verb) so tuples line up with canonical_ops
    from app import RAW_TO_CANONICAL  # noqa: E402  (app is importable; this reuses the live vocab)
    out = set()
    p1, p2 = it.get("semantic_ops") or [], it.get("semantic_ops_pass2") or []
    if agreed:
        rows = [a for a, b in zip(p1, p2) if a["op"] == b["op"] and a["object_type"] == b["object_type"]]
    elif pass_no == "v2":
        rows = it.get("semantic_ops_v2") or p1  # v2 where re-typed, v1 elsewhere
        if OWN_ONLY and "semantic_ops_v2_own" in it:
            rows = it["semantic_ops_v2_own"]  # Phase 22: injected delta papers — drop the ops inherited from the base paper
    else:
        rows = p2 if pass_no == 2 else p1
    for s in rows:
        raw = s["op"].strip().lower()
        c = RAW_TO_CANONICAL.get(raw) or (RAW_TO_CANONICAL.get(raw.split()[0]) if " " in raw else None) or raw
        t = s["object_type"]
        if t in ("other", "unknown"):
            t = f"other:{it['paper'][:12]}"
        elif relaxed:
            t = GROUP_OF.get(t, t)
        out.add((c, t))
    return out


def load(raw_ops, semantic=False, relaxed=False, families="lineage", pass_no=1, agreed=False, query=None):
    override = json.load(open("families.json"))["mechanism"] if families == "mechanism" else {}
    items = json.load(urllib.request.urlopen(f"{SERVER}/api/results"))["results"]
    papers = []
    for it in items:
        src = it.get("source") or ""
        is_query = query is not None and query.lower() in it["paper"].lower()
        if not src.startswith("arxiv:") and not is_query:
            continue  # abstract-only records are a different input type; keep them out of the matrix
        m = re.search(r"\[([\w-]+)\]", src)
        fam = m.group(1) if m else "unknown"
        fam = override.get(src.split()[0].replace("arxiv:", ""), fam)
        if semantic:
            if not it.get("semantic_ops") or ((pass_no == 2 or agreed) and not it.get("semantic_ops_pass2")):
                continue
            if False:
                continue
            ops = semantic_set(it, relaxed, pass_no, agreed)
        else:
            ops = (it.get("data") or {}).get("operations") if raw_ops else it.get("canonical_ops")
            ops = set(o.lower() for o in ops or [])
        papers.append({"paper": it["paper"], "family": fam, "id": src.split()[0], "ops": ops, "query": is_query})
    return sorted(papers, key=lambda p: (p["family"], p["paper"]))


def run_query(papers, fmt, min_shared=1):
    q = next(p for p in papers if p["query"])
    corpus = [p for p in papers if not p["query"]]
    print(f"query: {q['paper'][:60]}\n  ops: {sorted(fmt(o) for o in q['ops'])}\n")
    scored = sorted(((jaccard(q["ops"], p["ops"]), p) for p in corpus), key=lambda t: -t[0])
    print(f"closest papers (>= {min_shared} shared):")
    for j, p in [t for t in scored if len(q["ops"] & t[1]["ops"]) >= min_shared][:8]:
        print(f"  {j:.2f}  {p['family']:17} {p['paper'][:44]:46} [{', '.join(sorted(fmt(o) for o in q['ops'] & p['ops']))}]")
    fams = defaultdict(list)
    for j, p in scored:
        fams[p["family"]].append(j)
    print("\nmean similarity by family:")
    for fam, v in sorted(fams.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {sum(v)/len(v):.2f}  {fam}  (n={len(v)}, max {max(v):.2f})")


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def main(argv):
    raw_ops = "--raw" in argv
    top = int(argv[argv.index("--top") + 1]) if "--top" in argv else 10
    min_shared = int(argv[argv.index("--min-shared") + 1]) if "--min-shared" in argv else 2
    drop = {argv[i + 1].lower() for i, a in enumerate(argv) if a == "--drop"}
    semantic = "--semantic" in argv
    relaxed = "--relaxed" in argv
    families = argv[argv.index("--families") + 1] if "--families" in argv else "lineage"
    pass_no = 1
    if "--pass" in argv:
        v = argv[argv.index("--pass") + 1]
        pass_no = "v2" if v == "v2" else int(v)
    agreed = "--agreed" in argv
    global OWN_ONLY
    OWN_ONLY = "--own" in argv
    query = argv[argv.index("--query") + 1] if "--query" in argv else None
    papers = load(raw_ops, semantic, relaxed, families, pass_no, agreed, query)
    for p in papers:
        p["ops"] = {o for o in p["ops"] if (o[0] if semantic else o) not in drop}
    if drop:
        print(f"dropped from every set: {sorted(drop)}")
    fmt = (lambda o: f"{o[0]}@{o[1]}") if semantic else (lambda o: o)
    matrix_file = "semantic_similarity_matrix.csv" if semantic else "similarity_matrix.csv"
    if query:
        run_query(papers, (lambda o: f"{o[0]}@{o[1]}") if semantic else (lambda o: o), min_shared)
        return
    mode = ("semantic-" + ("relaxed" if relaxed else "strict") + ("-agreed" if agreed else f"-pass{pass_no}") + ("-own" if OWN_ONLY else "")) if semantic else ("raw" if raw_ops else "canonical")
    n = len(papers)

    with open(matrix_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paper", "family"] + [p["paper"] for p in papers])
        for a in papers:
            w.writerow([a["paper"], a["family"]] + [f"{jaccard(a['ops'], b['ops']):.2f}" for b in papers])
    print(f"wrote {matrix_file} ({n}x{n}, {mode} ops)\n")

    within, between, pairs = defaultdict(list), defaultdict(list), []
    for a, b in itertools.combinations(papers, 2):
        j = jaccard(a["ops"], b["ops"])
        if a["family"] == b["family"]:
            within[a["family"]].append(j)
        else:
            between[tuple(sorted((a["family"], b["family"])))].append(j)
            pairs.append((j, a, b))

    sizes = {}
    for p in papers:
        sizes[p["family"]] = sizes.get(p["family"], 0) + 1
    print(f"within-family mean Jaccard ({families} families)")
    for fam, v in sorted(within.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {fam:17} {sum(v) / len(v):.2f}  (n={sizes[fam]})")
    for fam, n in sizes.items():
        if n == 1:
            print(f"  {fam:17}  —    (n=1, no within score)")
    print("\nbetween-family mean Jaccard (top 6)")
    for k, v in sorted(between.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))[:6]:
        print(f"  {k[0]:11} x {k[1]:11} {sum(v) / len(v):.2f}")

    print(f"\ntop {top} cross-family paper pairs (>= {min_shared} shared ops)")
    pairs = [t for t in pairs if len(t[1]["ops"] & t[2]["ops"]) >= min_shared]
    for j, a, b in sorted(pairs, key=lambda t: -t[0])[:top]:
        shared = ", ".join(sorted(fmt(o) for o in a["ops"] & b["ops"]))
        print(f"  {j:.2f}  {a['family']:11} {a['paper'][:28]:30} x {b['family']:11} {b['paper'][:28]:30} [{shared}]")


if __name__ == "__main__":
    main(sys.argv[1:])
