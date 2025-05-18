import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("inference_api.log"), logging.StreamHandler()]
)
logger = logging.getLogger("inference-api")

def log(message, level="INFO"):
    if level == "ERROR":
        logger.error(message)
    else:
        logger.info(message)

def run_inference(prompt, verbose=False):
    try:
        if verbose:
            log(f"Processing prompt: {prompt}")
        prompt = prompt.lower()
        if "high cpu" in prompt or "memory used: 90%" in prompt or "memory used: 95%" in prompt:
            decision = "SCALE_UP"
        elif "failed services" in prompt:
            decision = "RESTART_SERVICES"
        elif "anomalies" in prompt:
            decision = "RESTART_VM"
        else:
            decision = "NO_ACTION"
        if verbose:
            log(f"Decision: {decision}")
        return decision
    except Exception as e:
        log(f"Error in inference: {str(e)}", level="ERROR")
        raise

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str

class InferenceResponse(BaseModel):
    decision: str

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Inference API is running"}

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    try:
        decision = run_inference(prompt=req.prompt, verbose=False)
        return {"decision": decision}
    except Exception as e:
        log(f"Inference error: {e}", level="ERROR")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)