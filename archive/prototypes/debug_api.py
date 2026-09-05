import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

prompt = """
Read this method description and list the key operations in order.

Description: The method uses an encoder to map raw pixels into a latent representation. A predictor then forecasts the next latent state given an action. The system compares the predicted latent state with the actual encoded state using a mean-squared error loss.

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

print(f"⏱️ Time: {elapsed:.2f} seconds")
print(f"📝 Response object: {response}")
print(f"📝 Choices: {response.choices}")
if response.choices:
    print(f"📝 Choice content: {response.choices[0].message.content}")
    print(f"📝 Full message: {response.choices[0].message}")
print(f"📝 Usage: {response.usage}")
