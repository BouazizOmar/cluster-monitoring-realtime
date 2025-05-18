# LangChain/LangGraph Orchestration Agent

An intelligent orchestration agent built with LangChain and LangGraph that uses OpenAI for smart decision classification and routing.

## Features

- **AI-Powered Decision Classification**: Uses GPT-4 to intelligently classify decisions
- **LangGraph Workflow**: Structured decision processing pipeline
- **Smart Routing**: Automatically routes critical operations to VM manager
- **Safety Delays**: Built-in delays for critical operations
- **RESTful API**: Easy integration with other services
- **CLI Interface**: Direct command-line access

## Setup

1. Copy environment template:
```bash
cp .env.example .env
```

2. Update `.env` with your OpenAI API key:
OPENAI_API_KEY=your_key_here


3. Start the orchestrator:
```bash
./run_orchestrator.sh
```

## Usage

### API Examples

Submit a migration decision:
```bash
curl -X POST http://localhost:8080/decision \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Migrate user service from VM-1 to VM-2 due to high CPU usage",
    "metadata": {
      "application": "user-service",
      "source_vm": "VM-1",
      "target_vm": "VM-2",
      "cpu_usage": 92
    }
  }'
```

### CLI Examples

```bash
# Simple decision
python cli.py "Scale up the web servers by 2 instances"

# With metadata
python cli.py "Deploy new version of API" '{"version": "v2.1.0", "environment": "production"}'
```

## Architecture

1. **Decision Input**: Receives decisions via API or CLI
2. **AI Classification**: GPT-4 analyzes and classifies decisions
3. **Workflow Processing**: LangGraph manages the processing pipeline
4. **Smart Routing**: Critical operations are routed to VM manager
5. **Result Collection**: All results are aggregated and returned

## Decision Types

- `critical_migration`: VM-to-VM application migrations
- `critical_scaling`: Infrastructure scaling operations
- `critical_deployment`: Application deployments/updates
- `standard_monitoring`: Regular monitoring tasks
- `configuration`: Configuration changes
- `information`: Informational queries