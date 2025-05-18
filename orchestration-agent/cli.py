import asyncio
import json
import sys
from orchestrator import OrchestrationAgent, DecisionInput

async def cli_main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py '<decision_content>' [metadata_json]")
        return
    
    content = sys.argv[1]
    metadata = {}
    
    if len(sys.argv) > 2:
        try:
            metadata = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            print("Invalid metadata JSON")
            return
    
    # Create agent and process decision
    agent = OrchestrationAgent()
    decision_input = DecisionInput(content=content, metadata=metadata, source="cli")
    
    print(f"Processing: {content}")
    result = await agent.process_decision(decision_input)
    
    print("\nResult:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(cli_main())