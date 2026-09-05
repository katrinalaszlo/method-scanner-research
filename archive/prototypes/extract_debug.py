import json
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def extract_method(text_snippet: str):
    prompt = f"""
Given the following method description, output a JSON object with exactly these keys:
- "name": a short name for the method
- "operations": an array of operation strings like ["encode", "predict", "compare", "update"]
- "loss": the loss function used (or null)
- "failure": the failure mode (or null)

Description: {text_snippet}

JSON:
"""
    start = time.time()
    response = client.chat.completions.create(
        model="qwen3:4b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200
    )
    elapsed = time.time() - start
    print(f"⏱️ Extraction took {elapsed:.2f} seconds")
    
    raw = response.choices[0].message.content
    print(f"📝 RAW RESPONSE:\n{raw}\n{'-'*40}")
    
    # Try to extract JSON
    try:
        start_idx = raw.find('{')
        end_idx = raw.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            return json.loads(raw[start_idx:end_idx])
    except Exception as e:
        print(f"⚠️ JSON Parse Error: {e}")
        pass
    return {"error": "Could not parse JSON", "raw": raw}

test_text = """
The method uses an encoder to map raw pixels into a latent representation. 
A predictor then forecasts the next latent state given an action. 
The system compares the predicted latent state with the actual encoded state using a mean-squared error loss. 
Backpropagation updates the encoder and predictor. 
The model fails when the environment has high stochasticity.
"""

result = extract_method(test_text)
print(json.dumps(result, indent=2))
