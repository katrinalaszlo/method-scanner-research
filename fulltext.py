"""arXiv ID -> PDF -> text -> method section -> POST /api/extract.

Usage: venv/bin/python fulltext.py papers.txt            # all rows
       venv/bin/python fulltext.py 2301.08243 2006.11239  # specific ids
       venv/bin/python fulltext.py --from "The Integrated Agent" 1710.02298   # pin the section start to the first line matching this regex
       venv/bin/python fulltext.py --test 1605.06432      # unseen paper: source "test:" so it stays out of the corpus;
                                                            # section capped at 1000 words (corpus default 1500). Override: --words N
For prediction runs, type with pass 1 only (semantic_type.py, no --pass 2) — pass 2 is for reliability measurement.
PDFs and text cached in papers/. Needs `pdftotext` on PATH and app.py running on :8000.
"""
import json
import os
import re
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(ROOT, "papers")
SERVER = "http://localhost:8000"
NS = {"a": "http://www.w3.org/2005/Atom"}
SECTION_WORDS = 1500
HEADING = re.compile(r"^\s*(\d+(\.\d+)?|[IVX]+)\.?\s+([A-Z][^\n]{2,60})$")
SMALLCAPS = re.compile(r"\b([A-Z]) (?=[A-Z]{2,}\b)")  # pdftotext renders small caps as "M ETHOD"
NUMBER_ONLY = re.compile(r"^\s*(\d+|[IVX]+)\.?\s*$")
METHOD_LIKE = re.compile(r"method|approach|algorithm|framework|architecture|our |proposed", re.I)
SKIP_LIKE = re.compile(r"introduction|background|related work|preliminar|problem formulation|problem statement|notation|setup|motivation", re.I)
STOP_LIKE = re.compile(r"experiment|result|conclusion|discussion|evaluation|limitation|acknowledg|reference|appendix", re.I)


def arxiv_meta(arxiv_id):
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    entry = root.find("a:entry", NS)
    title = " ".join(entry.find("a:title", NS).text.split())
    abstract = " ".join(entry.find("a:summary", NS).text.split())
    return title, abstract


def fetch_text(arxiv_id):
    os.makedirs(PAPERS_DIR, exist_ok=True)
    pdf = os.path.join(PAPERS_DIR, f"{arxiv_id}.pdf")
    txt = os.path.join(PAPERS_DIR, f"{arxiv_id}.txt")
    if not os.path.exists(txt):
        if not os.path.exists(pdf):
            urllib.request.urlretrieve(f"https://arxiv.org/pdf/{arxiv_id}", pdf)
        subprocess.run(["pdftotext", pdf, txt], check=True)
    return open(txt, encoding="utf-8", errors="replace").read()


ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}


def section_number(token):
    """Section number as int, or None for anything that isn't one (e.g. 'XX' used as a placeholder)."""
    token = token.rstrip(".")
    if token in ROMAN:
        return ROMAN[token]
    head = token.split(".")[0]
    return int(head) if head.isdigit() else None


def find_headings(lines):
    """Top-level numbered headings, in either layout pdftotext produces:
    same line   -> "3. Method" / "III. METHODOLOGY"
    split lines -> "3" / "" / "Method"
    Only accepts a heading whose number is the next in sequence, which drops footnote
    markers, figure numbers, and equation labels. Returns [(line_index, title)]."""
    heads = []
    expected = 1
    for i, line in enumerate(lines):
        m = HEADING.match(line)
        if m:
            num, title = m.group(1), SMALLCAPS.sub(r"\1", m.group(3)).strip()
            idx = i
        elif NUMBER_ONLY.match(line) and i + 2 < len(lines) and not lines[i + 1].strip():
            num, title = line.strip(), SMALLCAPS.sub(r"\1", lines[i + 2].strip())
            idx = i + 2
        else:
            continue
        if "." in num.rstrip(".") and not num.rstrip(".").replace(".", "").isdigit():
            continue
        if "." in num.rstrip("."):
            continue  # subsection like 3.1 — only top-level headings drive the walk
        if not title or not title[0].isupper() or len(title.split()) > 8 or title.endswith((".", ",", ":")):
            continue
        if re.search(r"published as|conference paper|preprint|under review|arxiv|workshop", title, re.I):
            continue  # running page headers, not sections
        n = section_number(num)
        if n is None:
            continue
        if re.search(r"introduction", title, re.I):
            heads, expected = [], n  # footnote markers before the intro consume numbers; restart here
        if n != expected:
            continue
        heads.append((idx, title))
        expected += 1
    return heads


def pinned_section(text, pattern, max_words=SECTION_WORDS):
    """Phase 20: section starts at the first line matching `pattern` (case-insensitive). Raises if no line matches."""
    lines = text.split("\n")
    i = next(i for i, l in enumerate(lines) if re.search(pattern, l, re.I))
    words = " ".join(lines[i:]).split()
    return " ".join(words[:max_words]), f"pinned: {lines[i].strip()[:40]}"


def method_section(text, max_words=SECTION_WORDS):
    """Return (section_text, how). Walk numbered headings in order, stop at experiments/results/
    conclusion. Prefer a heading that names the method; else the first that isn't intro/background/
    related/preliminaries. Fall back to a fixed window 20% into the paper."""
    lines = text.split("\n")
    heads = find_headings(lines)
    candidates = []
    seen_front_matter = False
    for i, title in heads:
        if STOP_LIKE.search(title):
            break
        if SKIP_LIKE.search(title):
            seen_front_matter = True
            continue
        if seen_front_matter:
            candidates.append((i, title))
    chosen = next(((i, t) for i, t in candidates if METHOD_LIKE.search(t)), None) or (candidates[0] if candidates else None)
    if chosen is None:
        words = text.split()
        s = int(len(words) * 0.20)
        return " ".join(words[s:s + max_words]), "fallback: 20% window"
    i, title = chosen
    words = " ".join(lines[i:]).split()
    return " ".join(words[:max_words]), f"heading: {title}"


def post_extract(paper, body, source):
    req = urllib.request.Request(
        f"{SERVER}/api/extract",
        json.dumps({"paper": paper, "abstract": body, "source": source}).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=600))


def rows_from_file(path):
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        fam, aid = line.split()[:2]
        yield fam, aid


def main(argv):
    test = "--test" in argv
    max_words = int(argv[argv.index("--words") + 1]) if "--words" in argv else (1000 if test else SECTION_WORDS)
    if "--words" in argv:
        i = argv.index("--words"); del argv[i:i + 2]
    pin = None
    if "--from" in argv:
        i = argv.index("--from"); pin = argv[i + 1]; del argv[i:i + 2]
    argv = [a for a in argv if a != "--test"]
    if len(argv) == 1 and os.path.isfile(argv[0]):
        rows = list(rows_from_file(argv[0]))
    else:
        rows = [("?", a) for a in argv]
    for fam, aid in rows:
        title, _ = arxiv_meta(aid)
        text = fetch_text(aid)
        section, how = pinned_section(text, pin, max_words) if pin else method_section(text, max_words)
        prefix = "test" if test else "arxiv"
        fam = "unseen" if test else fam
        res = post_extract(title, section, f"{prefix}:{aid} [{fam}] {how}")
        if res.get("success"):
            d = res["result"]
            print(f"OK   {aid} {title[:40]:42} {d['elapsed_seconds']:5}s {how[:34]:36} ops={d['data']['operations']} canon={d['canonical_ops']}", flush=True)
        else:
            print(f"FAIL {aid} {title[:40]:42} {res.get('error')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
