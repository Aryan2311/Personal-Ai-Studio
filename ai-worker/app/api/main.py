"""
AI Worker - Main FastAPI Application
Minimal server for GPU execution only
"""

from fastapi import FastAPI
from app.api.worker_api import router as worker_router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Worker",
    description="GPU executor for ML jobs - no UI, no registry, no auth",
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
    import os
    
    # Worker runs on port 8001 (backend is 8000)
    port = int(os.getenv("WORKER_PORT", "8001"))
    host = os.getenv("WORKER_HOST", "0.0.0.0")
    
    logger.info(f"Starting AI Worker on {host}:{port}")
    uvicorn.run(
        "app.api.main:app",
        host=host,
        port=port,
        reload=False  # Never reload in production
    )

