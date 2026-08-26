import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def extract_method(text_snippet: str):
    prompt = f"""
Read this method description and list the key operations in order (like encode, predict, compare, update, etc).

Description: {text_snippet}

Operations:
"""
    
    start = time.time()
    response = client.chat.completions.create(
        model="qwen3:4b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=100
    )
    elapsed = time.time() - start
    print(f"⏱️ Extraction took {elapsed:.2f} seconds")
    print("📝 RESPONSE:")
    print(response.choices[0].message.content)

test_text = """
The method uses an encoder to map raw pixels into a latent representation. 
A predictor then forecasts the next latent state given an action. 
The system compares the predicted latent state with the actual encoded state using a mean-squared error loss. 
Backpropagation updates the encoder and predictor.
"""

extract_method(test_text)
