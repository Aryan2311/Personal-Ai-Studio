"""
Worker SQS Poller
Polls SQS queue for jobs and processes them.

Worker is stateless - calls backend API to update status.
"""

import boto3
import json
import logging
import time
import os
import asyncio
import subprocess
import shutil
from typing import Dict, Optional
import httpx
from app.core.status_sender import StatusSender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkerPoller:
    """
    Polls SQS queue for jobs and processes them.
    
    Flow:
    1. Poll SQS for messages
    2. Process job
    3. Update backend via API
    4. Delete message from queue
    """
    
    def __init__(
        self,
        queue_url: str,
        backend_url: Optional[str] = None,
        region: str = "us-east-1"
    ):
        """
        Initialize worker poller.
        
        Args:
            queue_url: SQS queue URL for jobs
            backend_url: Backend API URL (optional, for fallback status updates)
            region: AWS region
        """
        self.queue_url = queue_url
        self.backend_url = backend_url.rstrip('/') if backend_url else None
        self.region = region
        self.sqs = boto3.client('sqs', region_name=region)
        self.running = False
        
        # Initialize status sender (uses SQS if configured, falls back to HTTP)
        self.status_sender = StatusSender()
        
        logger.info(f"WorkerPoller initialized | queue={queue_url} | backend={backend_url or 'SQS only'}")
    
    async def update_backend_status(
        self,
        job_id: str,
        status: str,  # "done" or "failed"
        metadata: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """
        Update job status in backend via SQS (or HTTP fallback).
        
        Transitions job from 'running' → 'done' | 'failed'
        """
        # Try SQS first (preferred)
        if status == "done":
            success = self.status_sender.send_done(job_id, metadata)
        elif status == "failed":
            success = self.status_sender.send_failed(job_id, error or "Unknown error", metadata)
        else:
            success = self.status_sender.send_status(job_id, status, metadata, error)
        
        # Fallback to HTTP if SQS not configured
        if not success and self.backend_url:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        f"{self.backend_url}/worker/job/update",
                        json={
                            "job_id": job_id,
                            "status": status,
                            "metadata": metadata or {},
                            "error": error
                        }
                    )
                    response.raise_for_status()
                    logger.info(f"Backend status updated via HTTP | job_id={job_id} | status={status}")
            except Exception as e:
                logger.error(f"Failed to update backend status: {e}")
                raise  # Raise so message stays in queue for retry
    
    async def send_heartbeat(self, job_id: str):
        """Send heartbeat to backend via SQS (or HTTP fallback)"""
        # Try SQS first
        success = self.status_sender.send_heartbeat(job_id)
        
        # Fallback to HTTP if SQS not configured
        if not success and self.backend_url:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(
                        f"{self.backend_url}/worker/job/heartbeat",
                        params={"job_id": job_id}
                    )
            except Exception as e:
                logger.debug(f"Heartbeat failed: {e}")
    
    def receive_message(self) -> Optional[Dict]:
        """Receive message from SQS (long polling)"""
        try:
            response = self.sqs.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,  # Long polling
                MessageAttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            if messages:
                message = messages[0]
                body = json.loads(message['Body'])
                return {
                    "receipt_handle": message['ReceiptHandle'],
                    "message_id": message['MessageId'],
                    "body": body
                }
            return None
        except Exception as e:
            logger.error(f"Failed to receive message from SQS: {e}")
            return None
    
    def delete_message(self, receipt_handle: str):
        """Delete message from SQS after processing"""
        try:
            self.sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            logger.debug("Message deleted from queue")
        except Exception as e:
            logger.error(f"Failed to delete message: {e}")
    
    async def process_job(self, message_data: Dict):
        """
        Process a job from SQS message.
        
        Message format:
        {
            "job_id": "job-uuid",
            "type": "train_identity",
            "payload": {...}
        }
        """
        job_id = message_data.get("job_id")
        job_type = message_data.get("type")
        payload = message_data.get("payload", {})
        
        logger.info(f"Processing job | job_id={job_id} | type={job_type}")
        
        # Step 2: Notify backend that job started (via SQS or HTTP)
        worker_instance_id = os.getenv("INSTANCE_ID")
        success = self.status_sender.send_start(job_id, worker_instance_id)
        
        # Fallback to HTTP if SQS not configured
        if not success and self.backend_url:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        f"{self.backend_url}/worker/job/start",
                        json={
                            "job_id": job_id,
                            "worker_instance_id": worker_instance_id
                        }
                    )
                    response.raise_for_status()
                    logger.info(f"Job started in backend via HTTP | job_id={job_id}")
            except Exception as e:
                logger.error(f"Failed to start job in backend: {e}")
                raise
        else:
            logger.info(f"Job started notification sent | job_id={job_id}")
        
        # Start heartbeat loop in background
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(job_id))
        
        try:
            if job_type == "train_identity":
                # Pass job_id to payload for training script
                payload["job_id"] = job_id
                await self._process_train_identity(payload)
                metadata = {"s3_path": payload.get("output_s3_path")}
            elif job_type == "train_lora":
                await self._process_train_lora(payload)
                metadata = {"s3_path": payload.get("output_s3_path")}
            elif job_type == "generate_image":
                await self._process_generate_image(payload)
                metadata = {"s3_path": payload.get("output_s3_path")}
            elif job_type == "generate_video":
                await self._process_generate_video(payload)
                metadata = {"s3_path": payload.get("output_s3_path")}
            else:
                raise ValueError(f"Unknown job type: {job_type}")
            
            # Stop heartbeat
            heartbeat_task.cancel()
            
            # Step 4: Job completed successfully
            await self.update_backend_status(
                job_id,
                "done",
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Job failed: {e}", exc_info=True)
            
            # Stop heartbeat
            if 'heartbeat_task' in locals():
                heartbeat_task.cancel()
            
            # Step 4: Job failed
            await self.update_backend_status(
                job_id,
                "failed",
                error=str(e)
            )
    
    async def _heartbeat_loop(self, job_id: str):
        """Send heartbeat every 30 seconds while job is running"""
        try:
            while True:
                await asyncio.sleep(30)
                await self.send_heartbeat(job_id)
        except asyncio.CancelledError:
            logger.debug(f"Heartbeat loop cancelled for job {job_id}")
        except Exception as e:
            logger.warning(f"Heartbeat loop error: {e}")
    
    async def _process_train_identity(self, payload: Dict):
        """
        Process identity training job.
        
        Payload:
        {
            "identity": "Leela",
            "training_images_s3": [...],
            "token": "sks_leela",
            "output_s3_path": "s3://...",
            "steps": 800,
            "learning_rate": 2e-6
        }
        """
        from app.storage.s3_manager import S3Manager
        from app.core.gpu_lock import is_locked, get_lock_info
        
        identity = payload.get("identity")
        training_images_s3 = payload.get("training_images_s3", [])
        token = payload.get("token")
        output_s3_path = payload.get("output_s3_path")
        steps = payload.get("steps", 800)
        learning_rate = payload.get("learning_rate", 2e-6)
        
        logger.info(f"[TRAIN IDENTITY] Processing training job | identity={identity} | num_images={len(training_images_s3)}")
        
        # Check GPU availability
        if is_locked():
            lock_info = get_lock_info()
            raise RuntimeError(f"GPU is busy: {lock_info.get('owner', 'unknown')} (job: {lock_info.get('job_id', 'unknown')})")
        
        # Download images from S3
        s3_manager = S3Manager(bucket_name="ai-studio-dc275989")
        temp_dir = f"/opt/ai-influencer/data/training/{identity}"
        os.makedirs(temp_dir, exist_ok=True)
        
        logger.info(f"[TRAIN IDENTITY] Downloading {len(training_images_s3)} images from S3...")
        for idx, s3_path in enumerate(training_images_s3, 1):
            filename = os.path.basename(s3_path)
            local_path = os.path.join(temp_dir, filename)
            logger.info(f"[TRAIN IDENTITY] Downloading image {idx}/{len(training_images_s3)} | s3_path={s3_path}")
            s3_manager.download_file(s3_path, local_path)
        
        # Move images to location expected by training script
        identity_images_dir = f"/opt/ai-influencer/data/identities/{identity}/images"
        os.makedirs(identity_images_dir, exist_ok=True)
        
        logger.info(f"[TRAIN IDENTITY] Moving images to {identity_images_dir}...")
        for filename in os.listdir(temp_dir):
            src = os.path.join(temp_dir, filename)
            dst = os.path.join(identity_images_dir, filename)
            if os.path.isfile(src):
                shutil.move(src, dst)
        
        # Clean up temp directory
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        # Prepare training script arguments
        base_model_path = "/opt/ai-influencer/models/base/sd15"
        if not os.path.exists(base_model_path):
            raise RuntimeError(f"Base model not found at {base_model_path}")
        
        training_script = "/opt/ai-influencer/ai-worker/app/training/dreambooth.py"
        python_exec = "/opt/ai-venv/bin/python"
        
        # Get job_id from the process_job context (passed via closure or parameter)
        # For now, we'll need to pass it - let's modify the signature
        job_id = payload.get("job_id")  # Should be in payload or we need to pass it
        
        # Build command
        cmd = [
            python_exec,
            training_script,
            "--identity", identity,
            "--token", token,
            "--job-id", job_id,
            "--base-model", base_model_path,
            "--steps", str(steps),
            "--lr", str(learning_rate),
            "--output-s3-path", output_s3_path
        ]
        
        logger.info(f"[TRAIN IDENTITY] Launching training subprocess | cmd={' '.join(cmd)}")
        
        # Launch training in background subprocess
        log_dir = f"/opt/ai-influencer/logs/training/{identity}"
        os.makedirs(log_dir, exist_ok=True)
        
        stdout_file = open(f"{log_dir}/stdout.log", "w")
        stderr_file = open(f"{log_dir}/stderr.log", "w")
        
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd="/opt/ai-influencer/ai-worker",
            env=dict(os.environ, PYTHONUNBUFFERED="1")
        )
        
        logger.info(f"[TRAIN IDENTITY] ✅ Training process started | pid={process.pid} | job_id={job_id}")
        
        # Wait for process to complete (this is async, so it won't block the event loop)
        # We'll wait in a thread pool to avoid blocking
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return_code = await loop.run_in_executor(executor, process.wait)
        
        stdout_file.close()
        stderr_file.close()
        
        if return_code != 0:
            raise RuntimeError(f"Training process failed with return code {return_code}")
        
        logger.info(f"[TRAIN IDENTITY] ✅ Training completed | job_id={job_id}")
    
    async def _process_train_lora(self, job_data: Dict):
        """Process LoRA training job"""
        logger.info("LoRA training not yet implemented in poller")
        raise NotImplementedError("LoRA training via queue not yet implemented")
    
    async def _process_generate_image(self, job_data: Dict):
        """Process image generation job"""
        logger.info("Image generation not yet implemented in poller")
        raise NotImplementedError("Image generation via queue not yet implemented")
    
    async def _process_generate_video(self, job_data: Dict):
        """Process video generation job"""
        logger.info("Video generation not yet implemented in poller")
        raise NotImplementedError("Video generation via queue not yet implemented")
    
    async def start(self):
        """Start polling loop"""
        self.running = True
        logger.info("Worker poller started")
        
        while self.running:
            try:
                message = self.receive_message()
                if message:
                    message_body = message["body"]
                    receipt_handle = message["receipt_handle"]
                    
                    try:
                        # Process job (message format: {"job_id": "...", "type": "...", "payload": {...}})
                        await self.process_job(message_body)
                        
                        # Delete message after successful processing
                        self.delete_message(receipt_handle)
                        logger.info(f"Job processed and message deleted | job_id={message_body.get('job_id')}")
                    except Exception as e:
                        logger.error(f"Job processing failed: {e}")
                        # Don't delete message - let it become visible again for retry
                        # SQS will automatically retry based on visibility timeout
                else:
                    # No messages, wait a bit
                    await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in poller loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait before retrying
    
    def stop(self):
        """Stop polling loop"""
        self.running = False
        logger.info("Worker poller stopped")

