"""Pass 1 vs pass 2 typing agreement. Reads results/*.json; reports per-tuple agreement, what got
re-typed, and whether pass 1's `state` assignments survived a stricter prompt.

Usage: venv/bin/python agreement.py [--ids papers_phase15.txt]   # restrict to the arXiv ids listed in a papers file
"""
import glob
import json
from collections import Counter, defaultdict

import sys
only = None
if "--ids" in sys.argv:
    only = {l.split()[1] for l in open(sys.argv[sys.argv.index("--ids") + 1]) if l.strip() and not l.startswith("#")}
pairs, per_paper, retyped = [], {}, Counter()
state_p1 = Counter()
for f in sorted(glob.glob("results/*.json")):
    r = json.load(open(f))
    p1, p2 = r.get("semantic_ops"), r.get("semantic_ops_pass2")
    if not p1 or not p2:
        continue
    if only is not None and (r.get("source") or "").split()[0].replace("arxiv:", "") not in only:
        continue
    agree = 0
    for a, b in zip(p1, p2):
        if a["op"] != b["op"]:
            continue
        same = a["object_type"] == b["object_type"]
        agree += same
        pairs.append((r["paper"], a["op"], a["object_type"], b["object_type"], same))
        if not same:
            retyped[(a["object_type"], b["object_type"])] += 1
        if a["object_type"] == "state":
            state_p1[b["object_type"]] += 1
    per_paper[r["paper"]] = (agree, len(p1))

n = len(pairs)
same = sum(p[4] for p in pairs)
print(f"papers with both passes: {len(per_paper)}   tuples compared: {n}   exact agreement: {same}/{n} = {same/n:.2f}")
unknown2 = sum(1 for p in pairs if p[3] == "unknown")
print(f"pass 2 said unknown: {unknown2}/{n} = {unknown2/n:.2f}   (agreement excluding unknowns: {same/(n-unknown2):.2f})")

print("\npapers where typing changed (changed/total):")
for paper, (a, t) in sorted(per_paper.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True):
    if a < t:
        print(f"  {t-a}/{t}  {paper[:60]}")

print("\nmost frequent re-typings (pass1 -> pass2):")
for (a, b), c in retyped.most_common(12):
    print(f"  {c:2}  {a:13} -> {b}")

print("\nwhat pass 1 'state' became in pass 2:")
for t, c in state_p1.most_common():
    print(f"  {c:2}  {t}")

by_op = defaultdict(lambda: [0, 0])
for _, op, a, b, s in pairs:
    by_op[op][0] += s; by_op[op][1] += 1
print("\nagreement by operation (n>=3):")
for op, (s, t) in sorted(by_op.items(), key=lambda kv: kv[1][0] / kv[1][1]):
    if t >= 3:
        print(f"  {s}/{t}  {op}")
