# tools.py
import httpx
from pydantic import BaseModel
from langchain.tools import tool

class PromptResponse(BaseModel):
    prompt: str

@tool("get_new_prompt", return_direct=True)
def get_new_prompt() -> str:
    """
    Zero-input tool: Fetch the raw monitoring prompt.
    """
    resp = httpx.get("http://localhost:9005/new_prompt", timeout=10.0)
    resp.raise_for_status()
    return PromptResponse.parse_obj(resp.json()).prompt

@tool("infer_action", return_direct=True)
def infer_action(prompt: str) -> str:
    """
    Single-input tool: Send prompt to inference API and return decision.
    """
    resp = httpx.post(
        "http://localhost:8000/infer",
        json={"prompt": prompt},
        timeout=30.0
    )
    resp.raise_for_status()
    return resp.json().get("decision", "").strip()
