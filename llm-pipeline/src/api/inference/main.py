import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("inference_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("inference-api")

# FastAPI app
app = FastAPI(
    title="Inference API",
    description="API for making decisions based on monitoring data",
    version="1.0.0"
)

# Models
class InferenceRequest(BaseModel):
    prompt: str

class InferenceResponse(BaseModel):
    decision: str
    confidence: Optional[float] = None
    reason: Optional[str] = None

# Decision making logic
def analyze_prompt(prompt: str) -> tuple[str, float, str]:
    """
    Analyze the prompt and make a decision.
    Returns: (decision, confidence, reason)
    """
    prompt = prompt.lower()
    
    # High resource usage
    if "high cpu" in prompt or "memory used: 90%" in prompt or "memory used: 95%" in prompt:
        return "SCALE_UP", 0.9, "High resource usage detected"
    
    # Service issues
    if "failed services" in prompt:
        return "RESTART_SERVICES", 0.8, "Failed services detected"
    
    # System anomalies
    if "anomalies" in prompt:
        return "RESTART_VM", 0.7, "System anomalies detected"
    
    # No issues
    return "NO_ACTION", 1.0, "No critical issues detected"

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Inference API is running",
        "version": "1.0.0"
    }

@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest):
    """
    Make a decision based on the monitoring data.
    
    Args:
        req: The inference request containing the prompt
        
    Returns:
        InferenceResponse: The decision and supporting information
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        logger.info(f"Processing prompt: {req.prompt[:200]}...")  # Log first 200 chars
        
        decision, confidence, reason = analyze_prompt(req.prompt)
        
        logger.info(f"Decision: {decision} (confidence: {confidence}, reason: {reason})")
        
        return InferenceResponse(
            decision=decision,
            confidence=confidence,
            reason=reason
        )
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing inference request: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 