from http.client import HTTPException

from fastapi import status
from handlers import Handlers
from executor import ActionPlan, app

import subprocess, logging
logging.basicConfig(level=logging.INFO)

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
