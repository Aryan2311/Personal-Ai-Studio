"""
Worker SQS Status Sender
Sends status updates to SQS queue for backend processing.

Worker sends status updates → SQS → Backend polls and processes
"""

import boto3
import json
import logging
import os
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatusSender:
    """
    Sends status updates to SQS queue.
    
    Worker is decoupled - sends status updates to queue instead of direct HTTP.
    """
    
    def __init__(
        self,
        queue_url: Optional[str] = None,
        region: str = "us-east-1"
    ):
        """
        Initialize status sender.
        
        Args:
            queue_url: SQS queue URL for status updates (from env var or parameter)
            region: AWS region
        """
        self.queue_url = queue_url or os.getenv("SQS_STATUS_QUEUE_URL")
        self.region = region
        
        if self.queue_url:
            self.sqs = boto3.client('sqs', region_name=region)
            logger.info(f"StatusSender initialized | queue={self.queue_url}")
        else:
            self.sqs = None
            logger.warning("⚠️ SQS_STATUS_QUEUE_URL not set, status updates will use direct HTTP (fallback)")
    
    def send_status(
        self,
        job_id: str,
        status: str,  # "start", "done", "failed", "heartbeat"
        metadata: Optional[Dict] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Send status update to SQS queue.
        
        Args:
            job_id: Job ID
            status: Status ("start", "done", "failed", "heartbeat")
            metadata: Optional metadata dict
            error: Optional error message (if status="failed")
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.sqs:
            logger.debug("SQS not configured, skipping status update")
            return False
        
        message_body = {
            "job_id": job_id,
            "status": status,
            "metadata": metadata or {},
            "error": error
        }
        
        try:
            response = self.sqs.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(message_body),
                MessageAttributes={
                    "status": {
                        "StringValue": status,
                        "DataType": "String"
                    },
                    "job_id": {
                        "StringValue": job_id,
                        "DataType": "String"
                    }
                }
            )
            
            message_id = response["MessageId"]
            logger.debug(f"Status update sent to SQS | job_id={job_id} | status={status} | message_id={message_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send status update to SQS: {e}")
            return False
    
    def send_start(self, job_id: str, worker_instance_id: Optional[str] = None) -> bool:
        """Send job start status"""
        return self.send_status(
            job_id=job_id,
            status="start",
            metadata={"worker_instance_id": worker_instance_id} if worker_instance_id else {}
        )
    
    def send_done(self, job_id: str, metadata: Optional[Dict] = None) -> bool:
        """Send job completion status"""
        return self.send_status(
            job_id=job_id,
            status="done",
            metadata=metadata or {}
        )
    
    def send_failed(self, job_id: str, error: str, metadata: Optional[Dict] = None) -> bool:
        """Send job failure status"""
        return self.send_status(
            job_id=job_id,
            status="failed",
            error=error,
            metadata=metadata or {}
        )
    
    def send_heartbeat(self, job_id: str) -> bool:
        """Send heartbeat status"""
        return self.send_status(
            job_id=job_id,
            status="heartbeat"
        )

