import os
import logging
import asyncio
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import utc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("agent.log"), logging.StreamHandler()]
)
logger = logging.getLogger("agent")

# API endpoints from environment variables
PROMPT_API = os.getenv("PROMPT_API_URL", "http://prompt_api:9005/new_prompt")
INFER_API = os.getenv("INFER_API_URL", "http://inference_api:8000/infer")

async def fetch_prompt():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(PROMPT_API)
            response.raise_for_status()
            return response.json().get("prompt")
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch prompt: {e}")
        return None

async def infer_decision(prompt):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(INFER_API, json={"prompt": prompt})
            response.raise_for_status()
            return response.json().get("decision")
    except httpx.HTTPError as e:
        logger.error(f"Failed to infer decision: {e}")
        return None

async def fetch_and_infer():
    logger.info("Fetching prompt...")
    prompt = await fetch_prompt()
    if not prompt:
        logger.warning("No prompt received")
        return

    logger.info(f"Prompt: {prompt}")
    decision = await infer_decision(prompt)
    if decision:
        logger.info(f"Decision: {decision}")
    else:
        logger.warning("No decision received")

async def main():
    scheduler = AsyncIOScheduler(timezone=utc)  # Use pytz UTC timezone
    scheduler.add_job(fetch_and_infer, "interval", minutes=2)
    scheduler.start()
    logger.info("Scheduler started")

    try:
        # Keep the script running
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped")

if __name__ == "__main__":
    asyncio.run(main())