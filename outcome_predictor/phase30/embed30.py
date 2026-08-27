"""Phase 30: nomic-embed-text embeddings of title + abstract via ollama. Writes embeddings.json {arxiv_id: [768 floats]}."""
import json, os, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
papers = json.load(open(os.path.join(HERE, "papers.json")))
out_path = os.path.join(HERE, "embeddings.json")
emb = json.load(open(out_path)) if os.path.exists(out_path) else {}
ids = [a for a in papers if a not in emb]
for i in range(0, len(ids), 16):
    chunk = ids[i:i + 16]
    body = json.dumps({"model": "nomic-embed-text", "input": [f"search_document: {papers[a]['title']}\n{papers[a]['abstract']}" for a in chunk]}).encode()
    req = urllib.request.Request("http://localhost:11434/api/embed", body, {"Content-Type": "application/json"})
    vecs = json.load(urllib.request.urlopen(req, timeout=600))["embeddings"]
    emb.update(dict(zip(chunk, vecs)))
    print(f"{len(emb)}/{len(papers)}", flush=True)
json.dump(emb, open(out_path, "w"))
print("dims", len(next(iter(emb.values()))))
