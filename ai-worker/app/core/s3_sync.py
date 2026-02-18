"""
S3 Sync - Worker
Syncs models and data from S3 on worker startup.

Since we don't have EBS, we need to download everything from S3.
"""

import os
import logging
from typing import List, Optional
import boto3
from botocore.exceptions import ClientError

from app.storage.s3_manager import S3Manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3Sync:
    """
    Syncs state from S3 on worker startup.
    
    Downloads:
    - Base models (if not present)
    - Identity models (for ready identities)
    - LoRA models (for ready LoRAs)
    """
    
    def __init__(self, s3_manager: Optional[S3Manager] = None):
        self.s3_manager = s3_manager or S3Manager(bucket_name="ai-studio-dc275989")
        self.bucket_name = self.s3_manager.bucket_name
        self.s3_client = self.s3_manager.s3_client
    
    def sync_from_backend(self, backend_url: str) -> dict:
        """
        Sync state from backend manifest.
        
        Gets list of ready identities and LoRAs from backend,
        then downloads their models from S3.
        
        Args:
            backend_url: Backend API URL
        
        Returns:
            Dict with sync summary
        """
        logger.info("Starting S3 sync from backend...")
        
        summary = {
            "identities_synced": 0,
            "loras_synced": 0,
            "base_model_checked": False,
            "errors": []
        }
        
        try:
            import httpx
            import asyncio
            
            async def fetch_state():
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Get identities from backend
                    response = await client.get(f"{backend_url.rstrip('/')}/identities")
                    response.raise_for_status()
                    identities = response.json()
                    
                    return identities
            
            # Fetch state from backend
            try:
                loop = asyncio.get_event_loop()
                identities = loop.run_until_complete(fetch_state())
            except RuntimeError:
                identities = asyncio.run(fetch_state())
            
            # Sync base model (if not present)
            base_model_path = "/opt/ai-influencer/models/base/sd15"
            if not os.path.exists(base_model_path):
                logger.warning(f"Base model not found at {base_model_path}. It should be installed during bootstrap.")
                summary["base_model_checked"] = False
            else:
                summary["base_model_checked"] = True
                logger.info("✅ Base model present")
            
            # Sync ready identities
            for identity in identities:
                identity_name = identity.get("id") or identity.get("identity_name")
                status = identity.get("status")
                
                if status == "ready":
                    try:
                        self._sync_identity_model(identity_name)
                        summary["identities_synced"] += 1
                    except Exception as e:
                        error_msg = f"Failed to sync identity {identity_name}: {e}"
                        logger.error(error_msg)
                        summary["errors"].append(error_msg)
            
            logger.info(f"✅ S3 sync complete | identities={summary['identities_synced']} | loras={summary['loras_synced']} | errors={len(summary['errors'])}")
            
        except Exception as e:
            logger.error(f"❌ S3 sync failed: {e}", exc_info=True)
            summary["errors"].append(f"Sync failed: {str(e)}")
        
        return summary
    
    def _sync_identity_model(self, identity_name: str):
        """Download identity model from S3 if not present locally"""
        local_path = f"/opt/ai-influencer/models/identities/{identity_name}"
        
        # Check if already exists
        if os.path.exists(local_path) and os.listdir(local_path):
            logger.debug(f"Identity model already present: {identity_name}")
            return
        
        # Download from S3
        s3_prefix = f"models/identities/{identity_name}/"
        logger.info(f"Downloading identity model from S3: {identity_name}")
        
        try:
            os.makedirs(local_path, exist_ok=True)
            
            # List all objects in the prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)
            
            downloaded = 0
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    # Skip directory markers
                    if s3_key.endswith('/'):
                        continue
                    
                    # Get relative path
                    relative_path = s3_key[len(s3_prefix):]
                    local_file = os.path.join(local_path, relative_path)
                    
                    # Create directory if needed
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    
                    # Download file
                    self.s3_client.download_file(self.bucket_name, s3_key, local_file)
                    downloaded += 1
            
            if downloaded > 0:
                logger.info(f"✅ Downloaded identity model: {identity_name} | files={downloaded}")
            else:
                logger.warning(f"No files found for identity model: {identity_name}")
                
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Identity model not found in S3: {identity_name}")
            else:
                raise
    
    def sync_loras(self, lora_names: List[str]):
        """Download LoRA models from S3"""
        for lora_name in lora_names:
            try:
                self._sync_lora_model(lora_name)
            except Exception as e:
                logger.error(f"Failed to sync LoRA {lora_name}: {e}")
    
    def _sync_lora_model(self, lora_name: str):
        """Download LoRA model from S3 if not present locally"""
        local_path = f"/opt/ai-influencer/models/loras/{lora_name}.safetensors"
        
        # Check if already exists
        if os.path.exists(local_path):
            logger.debug(f"LoRA model already present: {lora_name}")
            return
        
        # Download from S3
        s3_key = f"models/loras/{lora_name}.safetensors"
        logger.info(f"Downloading LoRA model from S3: {lora_name}")
        
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"✅ Downloaded LoRA model: {lora_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"LoRA model not found in S3: {lora_name}")
            else:
                raise

