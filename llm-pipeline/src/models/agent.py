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
PROMPT_API = os.getenv("PROMPT_API_URL", "http://localhost:9005/new_prompt")
INFER_API = os.getenv("INFER_API_URL", "http://localhost:8000/infer")

async def fetch_prompt():
    try:
        logger.info(f"Attempting to fetch prompt from: {PROMPT_API}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(PROMPT_API)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Received response: {data}")
            return data.get("prompt")
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch prompt: {str(e)}")
        logger.error(f"Response content: {e.response.content if hasattr(e, 'response') else 'No response content'}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while fetching prompt: {str(e)}")
        return None

async def infer_decision(prompt):
    try:
        logger.info(f"Attempting to get inference from: {INFER_API}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(INFER_API, json={"prompt": prompt})
            response.raise_for_status()
            data = response.json()
            logger.info(f"Received inference response: {data}")
            return data.get("decision")
    except httpx.HTTPError as e:
        logger.error(f"Failed to get inference: {str(e)}")
        logger.error(f"Response content: {e.response.content if hasattr(e, 'response') else 'No response content'}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while getting inference: {str(e)}")
        return None

async def fetch_and_infer():
    logger.info("Starting fetch_and_infer cycle...")
    prompt = await fetch_prompt()
    if not prompt:
        logger.warning("No prompt received, skipping inference")
        return

    logger.info(f"Successfully received prompt: {prompt[:200]}...")
    
    try:
        decision = await infer_decision(prompt)
        if decision:
            logger.info(f"Received decision: {decision}")
        else:
            logger.warning("No decision received from inference API")
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        logger.info("Continuing without inference result")

async def main():
    logger.info("Starting agent...")
    logger.info(f"Prompt API URL: {PROMPT_API}")
    logger.info(f"Inference API URL: {INFER_API}")
    
    scheduler = AsyncIOScheduler(timezone=utc)
    scheduler.add_job(fetch_and_infer, "interval", minutes=1)
    scheduler.start()
    logger.info("Scheduler started")

    try:
        # Keep the script running
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped")

if __name__ == "__main__":
    asyncio.run(main())