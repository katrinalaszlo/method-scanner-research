import json
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def extract_method(text_snippet: str):
    # Simplified prompt - no complex schema
    prompt = f"""
Extract the method structure from this text. Output valid JSON with these fields only:
- method_name: string
- operations: list of strings (e.g., ["encode", "predict", "compare", "update"])
- loss_function: string or null
- failure_mode: string or null

Text: {text_snippet}

JSON:
"""
    
    start = time.time()
    response = client.chat.completions.create(
        model="qwen3:4b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    print(f"⏱️ Extraction took {time.time() - start:.2f} seconds")
    
    raw = response.choices[0].message.content
    # Extract JSON from the response (in case it has extra text)
    try:
        # Find the first { and last }
        start_idx = raw.find('{')
        end_idx = raw.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            return json.loads(raw[start_idx:end_idx])
    except:
        pass
    return {"error": "Could not parse JSON", "raw": raw}

# Test it
test_text = """
The method uses an encoder to map raw pixels into a latent representation. 
A predictor then forecasts the next latent state given an action. 
The system compares the predicted latent state with the actual encoded state using a mean-squared error loss. 
Backpropagation updates the encoder and predictor. 
The model fails when the environment has high stochasticity.
"""

result = extract_method(test_text)
print(json.dumps(result, indent=2))