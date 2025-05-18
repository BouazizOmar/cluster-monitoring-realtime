#!/bin/bash

# Start the monitoring API server in the background
echo "Starting Monitoring API server..."
uvicorn src.api.main:app --host 0.0.0.0 --port 9005 &
MONITOR_API_PID=$!

# Wait for monitoring API to start
sleep 5

# Start the inference API server in the background
echo "Starting Inference API server..."
uvicorn src.api.inference.main:app --host 0.0.0.0 --port 8000 &
INFER_API_PID=$!

# Wait for inference API to start
sleep 5

# Start the agent
echo "Starting agent..."
python src/models/agent.py &
AGENT_PID=$!

# Function to handle script termination
cleanup() {
    echo "Shutting down services..."
    kill $MONITOR_API_PID
    kill $INFER_API_PID
    kill $AGENT_PID
    exit 0
}

# Set up trap for cleanup on script termination
trap cleanup SIGINT SIGTERM

# Keep script running
wait 