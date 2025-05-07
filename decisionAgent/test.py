import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

class DecisionResponse(BaseModel):
    prompt: str
    decision: str

app = FastAPI()

# Configure these addresses if your services run elsewhere or on different ports
PROMPT_API_URL = "http://localhost:9005/new_prompt"
INFER_API_URL = "http://localhost:8000/infer"

@app.get("/run_chain", response_model=DecisionResponse)
async def run_chain():
    async with httpx.AsyncClient() as client:
        # 1. Fetch the prompt
        try:
            prompt_res = await client.get(PROMPT_API_URL, timeout=10)
            prompt_res.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch prompt: {e}")

        data = prompt_res.json()
        prompt = data.get("prompt")
        if not prompt:
            raise HTTPException(status_code=500, detail="Prompt API returned empty prompt.")

        # 2. Send it to inference
        try:
            infer_res = await client.post(
                INFER_API_URL,
                json={"prompt": prompt},
                timeout=20,
            )
            infer_res.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Inference service failed: {e}")

        infer_data = infer_res.json()
        decision = infer_data.get("decision")
        if decision is None:
            raise HTTPException(status_code=500, detail="Inference API returned no decision.")

    return DecisionResponse(prompt=prompt, decision=decision)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8500)
