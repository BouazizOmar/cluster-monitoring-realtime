from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from handlers import Handlers
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()

class ActionPlan(BaseModel):
    action: str
    vm: str = None
    params: dict = {}

    @validator("action")
    def allowed(cls, v):
        allowed = {"start_vm","stop_vm","restart_vm","scale_vm","migrate_app","none"}
        if v not in allowed:
            raise ValueError(f"Unknown action: {v}")
        return v

ACTION_MAP = {
    "start_vm": Handlers.start_vm,
    "stop_vm": Handlers.stop_vm,
    "restart_vm": Handlers.restart_vm,
    "scale_vm": Handlers.scale_vm,
    "migrate_app": Handlers.migrate_app,
}

@app.post("/execute")
async def execute(plan: ActionPlan):
    if plan.action == "none":
        return {"status":"noop"}
    handler = ACTION_MAP.get(plan.action)
    try:
        out = handler(plan)
        return {"status":"ok","output":out}
    except Exception as e:
        logging.exception("Execution failed")
        raise HTTPException(status_code=500, detail=str(e))
