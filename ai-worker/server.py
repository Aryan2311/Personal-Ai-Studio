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

app = FastAPI(
    title="AI Worker",
    description="GPU executor for ML jobs",
    version="1.0.0"
)

# Include worker routes
app.include_router(worker_router)


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

