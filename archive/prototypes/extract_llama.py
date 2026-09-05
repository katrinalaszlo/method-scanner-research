import json
import time
import os
from datetime import datetime
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def extract_method(text_snippet: str):
    system_prompt = "You are a structured data extractor. Output ONLY valid JSON. No explanations, no markdown."
    user_prompt = f"""
Extract the algorithmic operations performed by the method (e.g., encode, predict, compare, update, estimate, associate, classify, transform). Do not list paper-level actions like 'derive', 'formulate', 'apply', or 'show'. Focus only on what the algorithm does step by step. Output a JSON object with:
- "name": a short name
- "operations": list of strings (e.g., ["encode", "predict", "compare", "update"])
- "loss": the loss function (or null)
- "failure": failure mode (or null)

Description: {text_snippet}

JSON:
"""
    start = time.time()
    response = client.chat.completions.create(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=200
    )
    elapsed = time.time() - start
    print(f"⏱️ Extraction took {elapsed:.2f} seconds")
    raw = response.choices[0].message.content
    print(f"📝 RAW: {raw[:200]}")

    try:
        start_idx = raw.find('{')
        end_idx = raw.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            parsed = json.loads(raw[start_idx:end_idx])
            return {"success": True, "data": parsed, "raw": raw}
    except Exception as e:
        print(f"⚠️ Parse error: {e}")
        return {"success": False, "error": "parse failed", "raw": raw[:500]}

def save_result(result, paper_name):
    if not result["success"]:
        print("❌ Extraction failed. Not saving.")
        return

    data = result["data"]
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{paper_name}_{timestamp}.json"

    output = {
        "paper": paper_name,
        "timestamp": timestamp,
        "data": data
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Saved to: {filename}")

# --- CONFIGURATION ---
# Change this to the paper you want to extract
PAPER_NAME = "PKF: Probabilistic Data Association Kalman Filter for Multi-Object Tracking"

test_text = """
In this letter, we derive a new Kalman filter (KF) with probabilistic data association between measurements and states. We formulate a variational inference problem to approximate the posterior density of the state conditioned on the measurement data. We view the unknown data association as a latent variable and apply Expectation Maximization (EM) to obtain a filter with the update step in the same form as the Kalman filter but with an expanded measurement vector of all potential associations. We show that the association probabilities can be computed as permanents of matrices with measurement likelihood entries.

"""

result = extract_method(test_text)
print(json.dumps(result, indent=2) if result["success"] else result)

if result["success"]:
    save_result(result, PAPER_NAME)
