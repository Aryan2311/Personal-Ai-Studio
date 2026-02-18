"""
AI Worker - FastAPI Server
GPU executor, nothing more.

Worker guarantees:
- One job at a time (GPU lock)
- Pull inputs from S3
- Push outputs to S3
- No registry, no auth, no UI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import os

# Import worker modules
from app.api.worker_api import router as worker_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup spot instance interruption handler
# This handles graceful shutdown on spot interruption (2-minute warning)
try:
    from app.core.spot_handler import setup_interruption_handler
    setup_interruption_handler()
    logger.info("✅ Spot instance interruption handler registered | signals=SIGTERM,SIGINT")
except Exception as e:
    logger.warning(f"Could not setup interruption handler: {e} | Continuing without it...")

app = FastAPI(
    title="AI Worker",
    description="GPU executor for ML jobs",
    version="1.0.0"
)

# Include worker routes
app.include_router(worker_router)


@app.on_event("startup")
async def startup_sync():
    """
    Startup tasks:
    1. Sync state from S3
    2. Start SQS poller (if configured)
    """
    import asyncio
    
    logger.info("=" * 80)
    logger.info("Worker Startup: Initializing...")
    logger.info("=" * 80)
    
    # 1. Sync state from S3
    try:
        from app.core.s3_sync import S3Sync
        
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        if not backend_url or backend_url == "http://localhost:8000":
            logger.warning("BACKEND_URL not set, skipping S3 sync. Set BACKEND_URL env var to enable sync.")
        else:
            sync = S3Sync()
            summary = sync.sync_from_backend(backend_url)
            
            logger.info("=" * 80)
            logger.info(f"S3 Sync Summary:")
            logger.info(f"  - Identities synced: {summary['identities_synced']}")
            logger.info(f"  - LoRAs synced: {summary['loras_synced']}")
            logger.info(f"  - Base model checked: {summary['base_model_checked']}")
            logger.info(f"  - Errors: {len(summary['errors'])}")
            if summary['errors']:
                for error in summary['errors']:
                    logger.warning(f"    - {error}")
            logger.info("=" * 80)
    except Exception as e:
        logger.error(f"❌ S3 sync failed on startup: {e}", exc_info=True)
        logger.warning("Worker will continue, but models may need to be downloaded on-demand")
    
    # 2. Start SQS poller (if configured)
    sqs_jobs_queue_url = os.getenv("SQS_JOBS_QUEUE_URL")
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    if sqs_jobs_queue_url:
        try:
            from app.core.worker_poller import WorkerPoller
            
            poller = WorkerPoller(
                queue_url=sqs_jobs_queue_url,
                backend_url=backend_url if backend_url != "http://localhost:8000" else None
            )
            
            # Start poller in background
            asyncio.create_task(poller.start())
            logger.info(f"✅ SQS Worker Poller started | queue={sqs_jobs_queue_url}")
        except Exception as e:
            logger.error(f"❌ Failed to start SQS poller: {e}", exc_info=True)
            logger.warning("Worker will continue, but jobs must be sent via direct HTTP")
    else:
        logger.warning("⚠️ SQS_JOBS_QUEUE_URL not set, worker will not poll SQS. Jobs must be sent via direct HTTP.")


@app.get("/")
async def root():
    """Worker root endpoint"""
    return {
        "service": "AI Worker",
        "status": "online",
        "description": "GPU executor - no UI, no registry, no auth"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Worker runs on port 8001 (backend is 8000)
    port = int(os.getenv("WORKER_PORT", "8001"))
    host = os.getenv("WORKER_HOST", "0.0.0.0")
    
    logger.info(f"Starting AI Worker on {host}:{port}")
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False  # Never reload in production
    )

