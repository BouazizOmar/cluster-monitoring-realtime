# LLM Pipeline

A monitoring pipeline for LLM-based applications.

## Project Structure

```
llm-pipeline/
├── src/                    # Source code
│   ├── api/               # API endpoints
│   │   ├── main.py       # Monitoring API
│   │   └── inference/    # Inference API
│   ├── data/             # Data processing and pipeline components
│   ├── models/           # Model-related code and utilities
│   └── utils/            # Shared utilities and helpers
├── tests/                # Test files
├── notebooks/            # Jupyter notebooks for exploration
├── config/               # Configuration files
└── environment.yml       # Conda environment file
```

## Services

The project consists of three main components:

1. **Monitoring API** (Port 9005)
   - Provides monitoring data and prompts
   - Endpoints: `/health`, `/new_prompt`

2. **Inference API** (Port 8000)
   - Makes decisions based on monitoring data
   - Endpoints: `/health`, `/infer`
   - Returns decisions with confidence scores and reasons

3. **Agent**
   - Periodically fetches data from Monitoring API
   - Sends data to Inference API for decisions
   - Logs all activities

## Setup

1. Create and activate the conda environment:
```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate llm-pipeline
```

## Running the Services

You can run all services using the provided script:

```bash
./run_services.sh
```

Or run them separately:

1. Start the Monitoring API:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 9005
```

2. Start the Inference API:
```bash
uvicorn src.api.inference.main:app --host 0.0.0.0 --port 8000
```

3. Start the agent:
```bash
python src/models/agent.py
```

## Testing the Services

1. Check if the Monitoring API is running:
```bash
curl http://localhost:9005/health
```

2. Check if the Inference API is running:
```bash
curl http://localhost:8000/health
```

3. Get a new prompt:
```bash
curl http://localhost:9005/new_prompt
```

4. Test inference:
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "System Monitoring Report: High CPU usage detected"}'
```

5. Monitor the logs:
```bash
# Agent logs
tail -f agent.log

# Monitoring API logs
tail -f prompt_api.log

# Inference API logs
tail -f inference_api.log
```

## Environment Variables

The following environment variables can be configured:

- `PROMPT_API_URL`: URL of the monitoring API (default: http://localhost:9005/new_prompt)
- `INFER_API_URL`: URL of the inference API (default: http://localhost:8000/infer)
- `MINIO_ENDPOINT`: MinIO server endpoint (default: minio:9000)
- `MINIO_ACCESS_KEY`: MinIO access key (default: minioadmin)
- `MINIO_SECRET_KEY`: MinIO secret key (default: minioadmin)

## Development

- Use `black` for code formatting
- Use `pylint` for code linting
- Write tests in the `tests/` directory
- Keep notebooks in the `notebooks/` directory

## Environment Management

To update the environment after changes to `environment.yml`:
```bash
conda env update -f environment.yml --prune
```

To export your current environment:
```bash
conda env export > environment.yml
``` 