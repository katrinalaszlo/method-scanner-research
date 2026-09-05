import json
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

# --- 1. CONNECT TO OLLAMA ---
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # dummy key
)

# --- 2. STRICT SCHEMA (Flat pipeline, escape hatch included) ---
class PipelineStep(BaseModel):
    step_number: int
    operation: str  # Allowed: encode, predict, compare, update, transform, aggregate, unknown
    input: str
    output: str

class ExtractedMethod(BaseModel):
    method_name: str = Field(description="Short name of the method")
    input_type: Optional[str] = None
    representation_type: Optional[str] = None
    loss_function: Optional[str] = None
    failure_mode: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    pipeline: List[PipelineStep] = Field(default_factory=list)
    source_quote: str = Field(description="Verbatim text this was extracted from")
    confidence: float = Field(ge=0.0, le=1.0)

# --- 3. EXTRACTION FUNCTION ---
def extract_method(text_snippet: str) -> ExtractedMethod:
    system_prompt = """You are a scientific method decomposer.
Extract the method details into the exact JSON schema provided.
CRITICAL RULES:
1. If a field is NOT explicitly mentioned, output NULL or [].
2. For pipeline operations, use ONLY: encode, predict, compare, update, transform, aggregate.
   If the text describes an operation not in this list, output "operation": "unknown".
3. Only output valid JSON. No extra words, no markdown."""
    
    # Get the schema as a JSON string (Pydantic v2 way - no deprecation warning)
    schema_json = json.dumps(ExtractedMethod.model_json_schema(), indent=2)
    
    user_prompt = f"""
Extract the method from this text into the given schema:
TEXT:
{text_snippet}

SCHEMA:
{schema_json}
"""
    
    response = client.chat.completions.create(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    raw_json = response.choices[0].message.content
    data = json.loads(raw_json)
    return ExtractedMethod(**data)

# --- 4. RUN ON A TEST SNIPPET (JEPA-style) ---
if __name__ == "__main__":
    test_text = """
    The method uses an encoder to map raw pixels into a latent representation. 
    A predictor then forecasts the next latent state given an action. 
    The system compares the predicted latent state with the actual encoded state using a mean-squared error loss. 
    Backpropagation updates the encoder and predictor. 
    The model fails when the environment has high stochasticity.
    """
    
    print("🧠 Extracting method from text...")
    result = extract_method(test_text)
    print("\n✅ EXTRACTION SUCCESSFUL!\n")
    print(json.dumps(result.model_dump(), indent=2))