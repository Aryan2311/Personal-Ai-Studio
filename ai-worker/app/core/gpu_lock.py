"""
GPU Lock - Prevents concurrent training/inference
MOST IMPORTANT: One GPU → one job at a time
"""

import os
import time
import json
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCK_PATH = "/opt/ai-influencer/locks/gpu.lock"
LOCK_DIR = os.path.dirname(LOCK_PATH)


def acquire_lock(owner: str, job_id: Optional[str] = None):
    """
    Acquire GPU lock.
    
    Args:
        owner: Who is acquiring the lock (e.g., "dreambooth_training", "lora_training", "inference")
        job_id: Optional job ID for tracking
    
    Raises:
        RuntimeError: If GPU is already locked
    """
    if os.path.exists(LOCK_PATH):
        # Read current lock info
        try:
            with open(LOCK_PATH, "r") as f:
                lock_info = json.load(f)
            current_owner = lock_info.get("owner", "unknown")
            current_job = lock_info.get("job_id", "unknown")
            raise RuntimeError(
                f"GPU is busy. Current owner: {current_owner} (job: {current_job})"
            )
        except:
            raise RuntimeError("GPU is busy (lock file exists)")
    
    # Create lock directory
    os.makedirs(LOCK_DIR, exist_ok=True)
    
    # Write lock file
    lock_info = {
        "owner": owner,
        "job_id": job_id,
        "acquired_at": time.time(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(LOCK_PATH, "w") as f:
        json.dump(lock_info, f, indent=2)
    
    logger.info(f"GPU lock acquired by {owner} (job: {job_id})")


def release_lock():
    """Release GPU lock"""
    if os.path.exists(LOCK_PATH):
        os.remove(LOCK_PATH)
        logger.info("GPU lock released")
    else:
        logger.warning("Attempted to release lock that doesn't exist")


def is_locked() -> bool:
    """Check if GPU is locked"""
    return os.path.exists(LOCK_PATH)


def get_lock_info() -> Optional[dict]:
    """Get current lock information"""
    if not os.path.exists(LOCK_PATH):
        return None
    
    try:
        with open(LOCK_PATH, "r") as f:
            return json.load(f)
    except:
        return {"owner": "unknown", "status": "locked"}


def wait_for_lock(timeout: int = 3600, check_interval: int = 5):
    """
    Wait for GPU lock to be released.
    
    Args:
        timeout: Maximum time to wait in seconds (default: 1 hour)
        check_interval: How often to check in seconds (default: 5)
    
    Raises:
        TimeoutError: If timeout is reached
    """
    start = time.time()
    
    while is_locked():
        elapsed = time.time() - start
        if elapsed > timeout:
            lock_info = get_lock_info()
            raise TimeoutError(
                f"GPU lock timeout after {timeout}s. "
                f"Current owner: {lock_info.get('owner', 'unknown') if lock_info else 'unknown'}"
            )
        
        logger.debug(f"Waiting for GPU lock... (elapsed: {elapsed:.0f}s)")
        time.sleep(check_interval)
    
    logger.info("GPU lock is now available")


def require_lock(owner: str, job_id: Optional[str] = None):
    """
    Context manager for GPU lock.
    
    Usage:
        with require_lock("dreambooth_training", job_id):
            # Training code here
    """
    class GPULockContext:
        def __init__(self, owner: str, job_id: Optional[str] = None):
            self.owner = owner
            self.job_id = job_id
        
        def __enter__(self):
            acquire_lock(self.owner, self.job_id)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            release_lock()
            return False
    
    return GPULockContext(owner, job_id)

