import csv
import io
import json
import os
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
MODEL = "qwen3:8b"
OLLAMA_CHAT = "http://localhost:11434/api/chat"


SYSTEM_PROMPT = "You are a structured data extractor. Output ONLY valid JSON. No explanations, no markdown."

USER_PROMPT_TEMPLATE = """
Extract the method structure. "operations" must name what the algorithm computes at runtime,
not what the authors do in the paper. "loss" is the objective the method optimizes, named as a
loss function. "failure" is a weakness, limitation, or failure mode the description admits —
look for words like collapse, degrade, fail, limited, struggle, unstable, accumulate, sensitive.

Example
Description: In this paper we propose SORT. We formulate tracking as assignment: the tracker
predicts each track's next box with a Kalman filter, matches detections to tracks by IoU, and
updates the track states. We show state-of-the-art results on MOT15.
JSON: {"name": "SORT", "operations": ["predict", "match", "update"], "loss": null, "failure": null}

Example
Description: We study a masked autoencoder. The encoder embeds visible patches, the decoder
reconstructs the masked ones, and training minimizes mean squared error. Performance degrades
at very high mask ratios.
JSON: {"name": "Masked Autoencoder", "operations": ["embed", "reconstruct"], "loss": "mean squared error", "failure": "performance degrades at very high mask ratios"}

Example
Description: We describe a speech enhancement system. A recurrent network estimates a spectral
mask from noisy audio, the mask is applied to the spectrogram, and the result is inverted back to
a waveform. The method struggles at very low signal-to-noise ratios, where it introduces musical
noise artifacts.
JSON: {"name": "Mask-Based Speech Enhancement", "operations": ["estimate", "mask", "invert"], "loss": null, "failure": "musical noise artifacts at very low signal-to-noise ratios"}

Example
Description: We index a document collection with an inverted index. Each document is tokenized,
tokens are stemmed, and a posting list is built per term. Queries are answered by intersecting
posting lists and ranking by BM25.
JSON: {"name": "Inverted Index Retrieval", "operations": ["tokenize", "stem", "intersect", "rank"], "loss": null, "failure": null}

Rules for "failure": extract it only when the Description itself states a limitation, instability,
or breakdown condition (for example "diverges when", "struggles with", "fails under"). Quote the
Description's own words. If the Description states no weakness, "failure" is null — that is the
normal case, not a fallback. Never reuse wording from the examples.

Now do the same for:
Description: {text_snippet}
JSON:
"""


def normalize(data):
    for key in ("loss", "failure"):
        if str(data.get(key)).strip().lower() in ("none", "null", ""):
            data[key] = None
    ops = []
    for op in data.get("operations") or []:
        op = str(op).strip()
        if op and op.lower() not in (o.lower() for o in ops):
            ops.append(op)
    data["operations"] = ops
    return data


def extract_method(text_snippet):
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT_TEMPLATE.replace("{text_snippet}", text_snippet)
    # native Ollama API: the OpenAI-compat endpoint can't switch qwen3's thinking off
    body = json.dumps({
        "model": MODEL,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 300, "num_ctx": 8192},  # Phase 26: long sections overflowed the default context
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }).encode()
    start = time.time()
    with urlopen(Request(OLLAMA_CHAT, body, {"Content-Type": "application/json"})) as resp:
        raw = json.load(resp)["message"]["content"]
    elapsed = time.time() - start

    start_idx = raw.find("{")
    end_idx = raw.rfind("}") + 1
    if start_idx < 0 or end_idx <= start_idx:
        return {"success": False, "error": "no JSON object in output", "raw": raw[:500], "elapsed": elapsed,
                "system_prompt": system_prompt, "user_prompt": user_prompt}
    try:
        parsed = normalize(json.loads(raw[start_idx:end_idx]))
    except Exception as e:
        return {"success": False, "error": f"parse failed: {e}", "raw": raw[:500], "elapsed": elapsed,
                "system_prompt": system_prompt, "user_prompt": user_prompt}
    return {"success": True, "data": parsed, "raw": raw, "elapsed": elapsed,
            "system_prompt": system_prompt, "user_prompt": user_prompt}


# Canonical vocabulary. Derived at read time, never stored — edit here, every record updates.
# Exact match on the whole lowercased op. Multi-word ops ("compute gains") stay unmapped on purpose.
CANONICAL = {
    "encode": ["encode", "embed", "map", "represent", "extract features"],
    "predict": ["predict", "forecast", "infer", "estimate", "imagine"],
    "correct": ["correct", "update", "refine", "adjust"],
    "suppress": ["suppress"],
    "associate": ["associate"],
    "compare": ["compare", "contrast", "match", "score"],
    "reconstruct": ["reconstruct", "decode", "denoise", "generate"],
    "mask": ["mask", "drop", "occlude"],
    "intervene": ["intervene", "condition", "perturb"],
    "aggregate": ["aggregate", "combine"],
    "classify": ["classify", "recognize", "label"],
    "train": ["train", "learn", "optimize"],
    "transmit": ["transmit", "propagate", "pass"],
    # round 2 (2026-08-24), from unmapped ops in the 30-paper full-text run
    "project": ["project"],
    "sample": ["sample", "resample"],
    "augment": ["augment"],
    "diffuse": ["diffuse", "add noise", "noise"],
    "reverse": ["reverse", "reverse diffuse", "invert"],
    "plan": ["plan"],
    "act": ["act", "execute", "control", "select action", "interact"],
    # v2 (2026-08-24, Phase 18): RL / meta-learning
    "evaluate": ["evaluate", "compute advantage", "compute advantage estimates", "compute advantage values",
                 "estimate value", "estimate return", "compute return", "evaluate policy", "compute loss"],
    "improve": ["improve", "optimize", "optimize surrogate loss", "maximize objective",
                "solve constrained optimization", "policy improvement", "improve policy"],
    "transition": ["transition", "compute transition", "step"],
}
# multi-word ops whose head verb is generic — the object decides
RAW_TO_CANONICAL_PHRASE = {
    "compute error": "compare",
    "compute prediction error": "compare",
    "stop-gradient": "mask",
}
RAW_TO_CANONICAL = {raw: canon for canon, raws in CANONICAL.items() for raw in raws}
RAW_TO_CANONICAL.update(RAW_TO_CANONICAL_PHRASE)


def canonicalize(ops):
    """Exact match on the whole op first; then fall back to the head verb ("update covariance" -> update).
    Head matches are returned separately so they stay auditable."""
    canonical, head_matched, unmapped = [], [], []
    for op in ops or []:
        text = str(op).strip().lower()
        canon = RAW_TO_CANONICAL.get(text)
        if canon is None and " " in text:
            canon = RAW_TO_CANONICAL.get(text.split()[0])
            if canon is not None:
                head_matched.append(op)
        if canon is None:
            unmapped.append(op)
        elif canon not in canonical:
            canonical.append(canon)
    return canonical, head_matched, unmapped


EXAMPLE_FRAGMENTS = ["musical noise", "signal-to-noise", "asymmetric design", "mask ratio", "mot15", "posting list"]
NO_FAILURE_PHRASES = ["no failure", "not mentioned", "none mentioned", "not specified", "not stated"]


def guard_failure(item):
    """Null a failure that is an example echo or a prose 'none'. Records stay untouched on disk;
    the flag tells the UI/CSV why the field is empty."""
    data = item.get("data") or {}
    text = (data.get("failure") or "").lower()
    if not text:
        return item
    if any(f in text for f in EXAMPLE_FRAGMENTS):
        item["failure_flag"] = "echo: " + data["failure"]
        data["failure"] = None
    elif any(p in text for p in NO_FAILURE_PHRASES):
        item["failure_flag"] = "prose-null: " + data["failure"]
        data["failure"] = None
    return item


# `state` is for the hidden state of a dynamical system. The typing model assigns it to any vector-shaped
# object (BYOL projector output, ResNet identity path). Keep `state` only when the evidence carries a
# temporal-dynamics marker; otherwise demote to `latent` and say so. Records on disk are untouched.
import re as _re
TEMPORAL = _re.compile(r"\b(t\s*[-+−]\s*1|t\s*\|\s*t|x_?t|z_?t|h_?t|s_?t|\bt\b|time[- ]?step|over time|temporal|trajector|sequen|recurs|recurrent|dynamic|track|belief|filter|transition|previous|next state|state estimate)", _re.I)


NEVER_STATE = {"act": "action", "select": "action", "execute": "action", "sample": "sample", "resample": "sample", "store": "sample"}


def guard_state(item):
    text = (item.get("abstract") or "")
    for field in ("semantic_ops", "semantic_ops_pass2", "semantic_ops_v2"):
        for t in item.get(field) or []:
            head = t["op"].strip().lower().split()[0]
            if head in NEVER_STATE and t.get("object_type") in ("state", "other", "unknown"):
                # Phase 19: these verbs act on actions/samples, never on a tracked state. Phase 22: also fill in an untyped object
                t["demoted_from"] = t["object_type"]
                t["object_type"] = NEVER_STATE[head]
                continue
            if t.get("object_type") != "state":
                continue
            quote = t.get("quote") or ""
            evidence = quote
            if quote and quote[:40].lower() in text.lower():
                i = text.lower().find(quote[:40].lower())
                evidence = text[max(0, i - 200): i + len(quote) + 200]
            if not TEMPORAL.search(evidence):
                t["object_type"] = "latent"
                t["demoted_from"] = "state"
    return item


def with_canonical(item):
    guard_failure(item)
    guard_state(item)
    ops = (item.get("data") or {}).get("operations")
    item["canonical_ops"], item["head_matched_ops"], item["unmapped_ops"] = canonicalize(ops)
    return item


def save_result(data, paper_name, abstract, elapsed, system_prompt, user_prompt, raw, source=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in paper_name) or "untitled"
    filename = os.path.join(RESULTS_DIR, f"{safe}_{timestamp}.json")
    output = {
        "paper": paper_name,
        "timestamp": timestamp,
        "model": MODEL,
        "elapsed_seconds": round(elapsed, 2),
        "abstract": abstract,
        "source": source,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": raw,
        "data": data,
    }
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    output["file"] = os.path.basename(filename)
    return with_canonical(output)


def load_results():
    if not os.path.isdir(RESULTS_DIR):
        return []
    items = []
    for name in os.listdir(RESULTS_DIR):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(RESULTS_DIR, name)) as f:
            item = json.load(f)
        item["file"] = name
        items.append(with_canonical(item))
    items.sort(key=lambda i: i.get("timestamp", ""), reverse=True)
    return items


CSV_COLUMNS = ["paper", "source", "timestamp", "model", "elapsed_seconds", "name", "operations",
               "canonical_ops", "head_matched_ops", "unmapped_ops", "loss", "failure", "failure_flag", "abstract",
               "raw_response", "system_prompt", "user_prompt"]


TEMPLATE_ROWS = [
    ["paper", "abstract"],
    ["LeNet-5", "We train a convolutional network on handwritten digits, encoding images through "
                "convolution and pooling layers and classifying with a softmax output."],
    ["", "Paper name is optional — blank rows are named row_2, row_3, and so on."],
]


def template_csv():
    buf = io.StringIO()
    csv.writer(buf).writerows(TEMPLATE_ROWS)
    return buf.getvalue().encode()


def results_csv():
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_COLUMNS)
    for item in load_results():
        d = item.get("data") or {}
        writer.writerow([
            item.get("paper", ""),
            item.get("source") or "",
            item.get("timestamp", ""),
            item.get("model", ""),
            item.get("elapsed_seconds", ""),
            d.get("name", ""),
            "; ".join(d.get("operations") or []),
            "; ".join(item["canonical_ops"]),
            "; ".join(item["head_matched_ops"]),
            "; ".join(item["unmapped_ops"]),
            d.get("loss") or "",
            d.get("failure") or "",
            item.get("failure_flag") or "",
            item.get("abstract", ""),
            item.get("raw_response", ""),
            item.get("system_prompt", ""),
            item.get("user_prompt", ""),
        ])
    return buf.getvalue().encode()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="application/json"):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            with open(os.path.join(ROOT, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path in ("/api/results.csv", "/api/template.csv"):
            if path == "/api/template.csv":
                payload, filename = template_csv(), "method_scanner_template.csv"
            else:
                payload, filename = results_csv(), "method_scanner_results.csv"
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif path == "/api/vocab":
            self._send(200, {"canonical": CANONICAL})
        elif path == "/api/prompt":
            self._send(200, {"model": MODEL, "system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT_TEMPLATE})
        elif path == "/api/results":
            self._send(200, {"results": load_results()})
        else:
            self._send(404, {"error": "not found"})

    def do_DELETE(self):
        path = urlparse(self.path).path
        if not path.startswith("/api/results/"):
            self._send(404, {"error": "not found"})
            return
        name = os.path.basename(unquote(path[len("/api/results/"):]))
        target = os.path.join(RESULTS_DIR, name)
        if not name.endswith(".json") or not os.path.isfile(target):
            self._send(404, {"error": "no such result"})
            return
        os.remove(target)
        self._send(200, {"deleted": name})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/batch":
            self.handle_batch()
            return
        if path == "/api/rerun":
            self.handle_rerun()
            return
        if path != "/api/extract":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        abstract = (body.get("abstract") or "").strip()
        paper_name = (body.get("paper") or "").strip() or "untitled"
        source = (body.get("source") or "").strip() or None
        if not abstract:
            self._send(400, {"success": False, "error": "abstract is empty"})
            return

        result = extract_method(abstract)
        if not result["success"]:
            self._send(200, result)
            return
        saved = save_result(result["data"], paper_name, abstract, result["elapsed"],
                            result["system_prompt"], result["user_prompt"], result["raw"], source)
        self._send(200, {"success": True, "result": saved})

    def handle_rerun(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        name = os.path.basename(body.get("file", ""))
        target = os.path.join(RESULTS_DIR, name)
        if not name.endswith(".json") or not os.path.isfile(target):
            self._send(404, {"success": False, "error": "no such result"})
            return
        with open(target) as f:
            old = json.load(f)
        abstract = (old.get("abstract") or "").strip()
        if not abstract:
            self._send(400, {"success": False, "error": "this record has no stored abstract"})
            return
        result = extract_method(abstract)
        if not result["success"]:
            self._send(200, result)
            return
        saved = save_result(result["data"], old.get("paper", "untitled"), abstract, result["elapsed"],
                            result["system_prompt"], result["user_prompt"], result["raw"], old.get("source"))
        os.remove(target)
        self._send(200, {"success": True, "result": saved, "replaced": name})

    def handle_batch(self):
        length = int(self.headers.get("Content-Length", 0))
        text = self.rfile.read(length).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            self._send(400, {"success": False, "error": "CSV has no rows"})
            return
        fields = {(k or "").strip().lower(): k for k in rows[0].keys()}
        if "abstract" not in fields:
            self._send(400, {"success": False, "error": "CSV needs an 'abstract' column"})
            return

        outcomes = []
        for i, row in enumerate(rows, start=1):
            abstract = (row.get(fields["abstract"]) or "").strip()
            paper = (row.get(fields.get("paper", "")) or "").strip() or f"row_{i}"
            if not abstract:
                outcomes.append({"paper": paper, "success": False, "error": "empty abstract"})
                continue
            result = extract_method(abstract)
            if not result["success"]:
                outcomes.append({"paper": paper, "success": False, "error": result["error"]})
                continue
            save_result(result["data"], paper, abstract, result["elapsed"],
                        result["system_prompt"], result["user_prompt"], result["raw"])
            outcomes.append({"paper": paper, "success": True})
        ok = sum(1 for o in outcomes if o["success"])
        self._send(200, {"success": True, "extracted": ok, "total": len(outcomes), "rows": outcomes})

    def log_message(self, fmt, *args):
        print(f"{self.command} {self.path}")


if __name__ == "__main__":
    print("Method scanner UI: http://localhost:8000")
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
