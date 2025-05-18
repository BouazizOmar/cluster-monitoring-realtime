# main.py

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import json
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional

from snowflake_ai_agent import SnowflakeAIAgent

class QueryRequest(BaseModel):
    query: str

# Custom JSON encoder to handle problematic values
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle NumPy types
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle problematic floats
        elif isinstance(obj, (float, np.float64, np.float32, np.float16)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        # Handle timestamps and dates
        elif pd and hasattr(pd, 'Timestamp') and isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        # Handle decimal objects
        elif hasattr(obj, 'is_finite') and callable(getattr(obj, 'is_finite')):
            if obj.is_finite():
                return float(obj)
            return None
        
        # Try to convert other unserializable objects to string
        try:
            return super().default(obj)
        except TypeError:
            try:
                return str(obj)
            except:
                return None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in prod!
    allow_methods=["*"],
    allow_headers=["*"],
)

# No conversation tracking needed

# 1) Your API route lives at /api/…
@app.post("/api/process-query")
async def process_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query is required")
    agent = SnowflakeAIAgent()
    result = agent.process_question(req.query)
    
    # Add diagnostic information for empty results
    if result.get("success") and result.get("results", {}).get("row_count", 0) == 0:
        # Extract any VM names or metric names from the query
        vm_pattern = r'(\w+buntu\s*\w*\d*)'
        metric_pattern = r'(memory|cpu|disk|network)'
        
        import re
        vm_matches = re.findall(vm_pattern, req.query, re.IGNORECASE)
        metric_matches = re.findall(metric_pattern, req.query, re.IGNORECASE)
        
        vm_query = vm_matches[0] if vm_matches else None
        metric_query = metric_matches[0] if metric_matches else None
        
        diagnostic_info = {
            "possible_issues": [],
            "suggestions": []
        }
        
        if vm_query:
            diagnostic_info["possible_issues"].append(f"VM name format mismatch: '{vm_query}' might not match the exact database value")
            diagnostic_info["suggestions"].append("Try using a VM name variant like 'Lubuntu V2' or 'Ubuntu' or use LIKE operator with '%Lubuntu%'")
        
        if metric_query and metric_query.lower() == 'memory':
            diagnostic_info["possible_issues"].append("Memory metric name might not be 'memory_usage' exactly")
            diagnostic_info["suggestions"].append("Try using a more specific memory metric name or use LIKE operator with '%memory%'")
        
        # Extract any available VM and metric information from the query
        if '--' in result.get("query", ""):
            # Parse comments for VM and metric options
            lines = result.get("query", "").split('\n')
            for line in lines:
                if '-- Available VM names:' in line:
                    diagnostic_info["suggestions"].append(f"Available VMs: {line.replace('-- Available VM names:', '').strip()}")
                if '-- Available memory metrics:' in line:
                    diagnostic_info["suggestions"].append(f"Available metrics: {line.replace('-- Available memory metrics:', '').strip()}")
        
        result["diagnostic_info"] = diagnostic_info
    
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Processing failed"))
    
    # Use the custom JSON encoder to safely serialize the response
    response_data = {
        "success": True,
        "sql_query": result["query"],
        "results": result["results"],
        "explanation": result["explanation"],
        "diagnostic_info": result.get("diagnostic_info", {})
    }
    
    # Double-check for problematic values before serialization
    try:
        # First attempt serialization using our custom encoder
        json_string = json.dumps(response_data, cls=SafeJSONEncoder)
        
        # Then parse it back to ensure it worked
        validated_content = json.loads(json_string)
        
        # Return the validated content
        return JSONResponse(content=validated_content, status_code=200)
    except Exception as e:
        # If we still have serialization issues, return a simplified response
        print(f"JSON serialization error: {str(e)}")
        
        # Create a simplified version with just strings
        safe_response = {
            "success": True,
            "sql_query": result["query"],
            "explanation": result["explanation"],
            "error": "Some results contained values that couldn't be properly serialized.",
            "results": {
                "columns": result["results"].get("columns", []),
                "row_count": result["results"].get("row_count", 0),
                "rows": [{str(k): str(v) if v is not None else None 
                         for k, v in row.items()}
                         for row in result["results"].get("rows", [])]
            }
        }
        
        return JSONResponse(content=safe_response, status_code=200)

# LangGraph implementation has been removed

# 2) Serve static assets (CSS, JS, images) under their real paths
HERE = Path(__file__).parent.resolve()
REACT_DIST = HERE.parent / "chat-ui" / "dist"
if not REACT_DIST.exists():
    raise RuntimeError(f"Build not found at {REACT_DIST}")

app.mount(
    "/static", 
    StaticFiles(directory=str(REACT_DIST), html=False), 
    name="static"
)

# 3) Catch-all GET → serve index.html for your SPA
@app.get("/{full_path:path}")
async def spa_router(full_path: str, request: Request):
    # If someone requests /static/* it'll go to the StaticFiles above.
    # Anything else (including `/`), return index.html:
    return FileResponse(REACT_DIST / "index.html")
