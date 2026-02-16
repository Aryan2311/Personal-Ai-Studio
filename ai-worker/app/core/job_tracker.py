"""
Job Tracker - Tracks training and generation jobs
Provides visibility into system state
"""

import json
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JOBS_FILE = Path("/opt/ai-influencer/data/jobs.json")
JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_jobs() -> Dict:
    """Load jobs registry from JSON file"""
    if not JOBS_FILE.exists():
        return {}
    
    try:
        return json.loads(JOBS_FILE.read_text())
    except Exception as e:
        logger.error(f"Failed to load jobs: {e}")
        return {}


def save_jobs(jobs: Dict):
    """Save jobs registry to JSON file"""
    try:
        JOBS_FILE.write_text(json.dumps(jobs, indent=2))
    except Exception as e:
        logger.error(f"Failed to save jobs: {e}")
        raise


def create_job(
    job_type: str,
    target: str,
    metadata: Optional[Dict] = None
) -> str:
    """
    Create a new job entry.
    
    Args:
        job_type: Type of job ("dreambooth_training", "lora_training", "inference")
        target: Target identity/lora name
        metadata: Optional additional metadata
    
    Returns:
        Job ID (UUID)
    """
    jobs = load_jobs()
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "id": job_id,
        "type": job_type,
        "target": target,
        "status": "running",
        "created_at": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": metadata or {}
    }
    
    save_jobs(jobs)
    logger.info(f"Created job {job_id}: {job_type} for {target}")
    
    return job_id


def update_job(
    job_id: str,
    status: str,
    metadata: Optional[Dict] = None,
    error: Optional[str] = None
):
    """
    Update job status.
    
    Args:
        job_id: Job ID
        status: New status ("running", "completed", "failed")
        metadata: Optional additional metadata
        error: Error message if failed
    """
    jobs = load_jobs()
    
    if job_id not in jobs:
        logger.warning(f"Job {job_id} not found")
        return
    
    jobs[job_id]["status"] = status
    jobs[job_id]["updated_at"] = time.time()
    jobs[job_id]["updated_at_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    if metadata:
        jobs[job_id]["metadata"].update(metadata)
    
    if error:
        jobs[job_id]["error"] = error
    
    if status in ["completed", "failed"]:
        jobs[job_id]["completed_at"] = time.time()
        jobs[job_id]["completed_at_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
        duration = jobs[job_id]["completed_at"] - jobs[job_id]["created_at"]
        jobs[job_id]["duration_seconds"] = duration
    
    save_jobs(jobs)
    logger.info(f"Updated job {job_id}: {status}")


def get_job(job_id: str) -> Optional[Dict]:
    """Get job by ID"""
    jobs = load_jobs()
    return jobs.get(job_id)


def list_jobs(
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    target: Optional[str] = None
) -> List[Dict]:
    """
    List jobs with optional filters.
    
    Args:
        job_type: Filter by type
        status: Filter by status
        target: Filter by target
    
    Returns:
        List of job dictionaries
    """
    jobs = load_jobs()
    result = list(jobs.values())
    
    if job_type:
        result = [j for j in result if j["type"] == job_type]
    
    if status:
        result = [j for j in result if j["status"] == status]
    
    if target:
        result = [j for j in result if j["target"] == target]
    
    # Sort by created_at (newest first)
    result.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    
    return result


def get_running_jobs() -> List[Dict]:
    """Get all currently running jobs"""
    return list_jobs(status="running")

