from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

print("Sending request...")
response = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Say OK"}],
    temperature=0.1,
    timeout=10.0  # Add timeout
)
print(response.choices[0].message.content)
