"""
Spot Instance Interruption Handler
Handles graceful shutdown on spot instance interruption
"""

import os
import signal
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import GPU lock and job tracker
from app.core.gpu_lock import release_lock, is_locked
from app.core.job_tracker import update_job, get_running_jobs


def handle_spot_interruption(signum, frame):
    """
    Handle spot instance interruption signal (2-minute warning).
    
    AWS sends SIGTERM 2 minutes before interruption.
    This handler:
    1. Releases GPU lock
    2. Updates running jobs to 'interrupted'
    3. Saves state
    """
    logger.warning("=" * 80)
    logger.warning("SPOT INSTANCE INTERRUPTION DETECTED")
    logger.warning("=" * 80)
    logger.warning("Received interruption signal - performing graceful shutdown...")
    
    try:
        # 1. Release GPU lock if held
        if is_locked():
            logger.info("[INTERRUPTION HANDLER] Releasing GPU lock...")
            release_lock()
            logger.info("[INTERRUPTION HANDLER] GPU lock released")
        
        # 2. Update all running jobs to 'interrupted'
        running_jobs = get_running_jobs()
        logger.info(f"[INTERRUPTION HANDLER] Found {len(running_jobs)} running jobs")
        
        for job_id, job in running_jobs.items():
            logger.info(f"[INTERRUPTION HANDLER] Marking job as interrupted | job_id={job_id} | type={job.get('type')}")
            update_job(
                job_id,
                status="interrupted",
                error="Spot instance interruption - job can be resumed after instance restart"
            )
        
        logger.info("[INTERRUPTION HANDLER] All jobs marked as interrupted")
        logger.info("[INTERRUPTION HANDLER] Graceful shutdown complete")
        logger.warning("=" * 80)
        
    except Exception as e:
        logger.error(f"[INTERRUPTION HANDLER] Error during shutdown: {e}", exc_info=True)
    
    # Exit gracefully
    os._exit(0)


def setup_interruption_handler():
    """
    Setup signal handlers for spot instance interruption.
    
    AWS sends SIGTERM 2 minutes before spot interruption.
    """
    signal.signal(signal.SIGTERM, handle_spot_interruption)
    signal.signal(signal.SIGINT, handle_spot_interruption)  # Also handle Ctrl+C
    
    logger.info("Spot interruption handler registered | signals=SIGTERM,SIGINT")


def check_spot_interruption_warning():
    """
    Check for spot instance interruption warning via metadata service.
    
    Returns:
        True if interruption warning detected, False otherwise
    """
    try:
        import urllib.request
        import json
        
        # AWS metadata service endpoint for spot interruption notice
        url = "http://169.254.169.254/latest/meta-data/spot/instance-action"
        
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                data = json.loads(response.read().decode())
                if data.get("action") == "terminate" or data.get("action") == "stop":
                    logger.warning(f"Spot interruption warning detected | action={data.get('action')} | time={data.get('time')}")
                    return True
        except urllib.error.URLError:
            # No interruption warning (normal case)
            return False
            
    except Exception as e:
        logger.debug(f"Could not check spot interruption warning: {e}")
        return False


if __name__ == "__main__":
    # Test interruption handler
    setup_interruption_handler()
    print("Interruption handler registered. Waiting for signal...")
    print("Send SIGTERM to test: kill -TERM <pid>")
    
    while True:
        if check_spot_interruption_warning():
            logger.warning("Interruption warning detected via metadata service")
        time.sleep(10)

