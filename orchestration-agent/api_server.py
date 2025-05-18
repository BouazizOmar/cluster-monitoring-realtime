from aiohttp import web, web_request
import json
import asyncio
import logging
from orchestrator import OrchestrationAgent, DecisionInput

class OrchestratorAPI:
    def __init__(self, agent: OrchestrationAgent):
        self.agent = agent
        self.app = web.Application()
        self._setup_routes()
        self.logger = logging.getLogger(__name__)
        self.running = True
    
    def _setup_routes(self):
        self.app.router.add_post('/decision', self.submit_decision)
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/status', self.get_status)
    
    async def submit_decision(self, request: web_request.Request):
        try:
            data = await request.json()
            decision_input = DecisionInput(**data)
            result = await self.agent.process_decision(decision_input)
            return web.json_response(result)
        except Exception as e:
            self.logger.error(f"Error processing decision: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=400)
    
    async def health_check(self, request: web_request.Request):
        return web.json_response({"status": "healthy", "running": self.running})
    
    async def get_status(self, request: web_request.Request):
        return web.json_response({
            "status": "running" if self.running else "stopped"
        })
    
    async def start_server(self, host='0.0.0.0', port=8082):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        print(f"Orchestration API running on http://{host}:{port}")
        return runner  # Return the runner so it can be cleaned up later

async def run_with_api():
    # Initialize agent
    agent = OrchestrationAgent()
    
    # Create API
    api = OrchestratorAPI(agent)
    
    # Start API server
    runner = await api.start_server()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        api.running = False
        # Clean up
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(run_with_api())