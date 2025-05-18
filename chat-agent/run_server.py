#!/usr/bin/env python3
"""
Run the FastAPI server for the monitoring pipeline API
"""
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Run the FastAPI server
    """
    # Get port from environment variable or use default
    port = int(os.getenv("API_PORT", "5001"))
    
    # Start the server
    print(f"Starting server on port {port}...")
    print("Press Ctrl+C to stop")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True  # Enable auto-reload for development
    )

if __name__ == "__main__":
    main() 