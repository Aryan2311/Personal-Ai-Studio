"""
S3 Storage Manager - AI Worker
Handles downloading models/references from S3 and uploading outputs
Worker only: Pull → compute → push
"""

import boto3
import os
from pathlib import Path
from typing import List, Optional
import logging
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3Manager:
    """
    Manages S3 storage for AI Worker.
    
    Worker responsibilities:
    - Download training images from S3
    - Download models from S3 (if not cached)
    - Download references from S3
    - Upload trained models to S3
    - Upload generated outputs to S3
    """
    
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME", "ai-studio")
        self.region = region
        
        # Initialize S3 client
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            # Use default credentials (IAM role, env vars, etc.)
            self.s3_client = boto3.client('s3', region_name=region)
    
    def download_file(self, s3_key: str, local_path: str) -> str:
        """
        Download a single file from S3.
        
        Args:
            s3_key: S3 key (path in bucket)
            local_path: Local file path to save to
        
        Returns:
            Local path where file was downloaded
        """
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded from s3://{self.bucket_name}/{s3_key} to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            raise
    
    def upload_file(self, local_path: str, s3_key: str) -> str:
        """
        Upload a single file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 key (path in bucket)
        
        Returns:
            S3 key where file was uploaded
        """
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded to s3://{self.bucket_name}/{s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            raise
    
    def download_training_images(
        self,
        identity_name: str,
        local_dir: str,
        s3_key_prefix: Optional[str] = None
    ) -> List[str]:
        """
        Download training images from S3 to local disk.
        
        Args:
            identity_name: Name of the identity
            local_dir: Local directory to save images
            s3_key_prefix: Optional custom S3 prefix
        
        Returns:
            List of local file paths
        """
        prefix = s3_key_prefix or f"training-data/{identity_name}/images/"
        local_paths = []
        
        # List objects in S3
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No images found in s3://{self.bucket_name}/{prefix}")
                return []
            
            os.makedirs(local_dir, exist_ok=True)
            
            for obj in response['Contents']:
                s3_key = obj['Key']
                filename = Path(s3_key).name
                local_path = os.path.join(local_dir, filename)
                
                self.s3_client.download_file(self.bucket_name, s3_key, local_path)
                local_paths.append(local_path)
                logger.info(f"Downloaded {filename} from S3")
        
        except ClientError as e:
            logger.error(f"Failed to download training images: {e}")
            raise
        
        return local_paths
    
    def download_model(
        self,
        identity_name: str,
        local_path: str,
        s3_key_prefix: Optional[str] = None
    ) -> str:
        """
        Download model from S3 to local disk.
        
        Args:
            identity_name: Name of the identity
            local_path: Local directory to save model
            s3_key_prefix: Optional custom S3 prefix
        
        Returns:
            Local path where model was downloaded
        """
        prefix = s3_key_prefix or f"models/identities/{identity_name}/"
        
        # List all objects in the prefix
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No model found in s3://{self.bucket_name}/{prefix}")
                return None
            
            os.makedirs(local_path, exist_ok=True)
            
            for obj in response['Contents']:
                s3_key = obj['Key']
                relative_path = os.path.relpath(s3_key, prefix)
                local_file = os.path.join(local_path, relative_path)
                
                # Create directory if needed
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                
                self.s3_client.download_file(self.bucket_name, s3_key, local_file)
                logger.debug(f"Downloaded {relative_path} from S3")
            
            logger.info(f"Downloaded model for {identity_name} from S3")
            return local_path
        
        except ClientError as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    def upload_model(
        self,
        identity_name: str,
        model_path: str,
        s3_key_prefix: Optional[str] = None
    ) -> str:
        """
        Upload trained model to S3.
        
        Args:
            identity_name: Name of the identity
            model_path: Local path to model directory
            s3_key_prefix: Optional custom S3 prefix
        
        Returns:
            S3 key prefix where model was uploaded
        """
        prefix = s3_key_prefix or f"models/identities/{identity_name}/"
        
        # Upload entire directory
        for root, dirs, files in os.walk(model_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, model_path)
                s3_key = f"{prefix}{relative_path}"
                
                try:
                    self.s3_client.upload_file(local_file, self.bucket_name, s3_key)
                    logger.debug(f"Uploaded {relative_path} to S3")
                except ClientError as e:
                    logger.error(f"Failed to upload {relative_path}: {e}")
                    raise
        
        logger.info(f"Uploaded model for {identity_name} to s3://{self.bucket_name}/{prefix}")
        return prefix
    
    def upload_output(
        self,
        output_path: str,
        s3_key: str
    ) -> str:
        """
        Upload generated output to S3.
        
        Args:
            output_path: Local path to output file
            s3_key: S3 key where to save
        
        Returns:
            S3 key where output was uploaded
        """
        try:
            self.s3_client.upload_file(output_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded output to s3://{self.bucket_name}/{s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Failed to upload output: {e}")
            raise
