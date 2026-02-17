"""
Worker API - Private endpoints for backend communication
EC2 worker exposes minimal API for job execution
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/worker", tags=["worker"])

# Initialize PipelineManager globally
_pipeline_manager = None

def get_pipeline_manager():
    """Get or create the global PipelineManager instance"""
    global _pipeline_manager
    if _pipeline_manager is None:
        from app.pipeline_manager import PipelineManager
        
        # Determine model paths
        base_model_path = os.getenv("BASE_MODEL_PATH", "/mnt/models/base/sd15")
        if not os.path.exists(base_model_path):
            base_model_path = "/opt/ai-influencer/models/base/sd15"
        
        motion_path = os.getenv("MOTION_MODEL_PATH", "/mnt/models/motion/animatediff")
        if not os.path.exists(motion_path):
            motion_path = "/opt/ai-influencer/models/motion/animatediff"
        
        _pipeline_manager = PipelineManager(
            base_model_path=base_model_path,
            motion_path=motion_path
        )
        logger.info(f"PipelineManager initialized | base={base_model_path} | motion={motion_path}")
    
    return _pipeline_manager


# Request Models
class TrainIdentityRequest(BaseModel):
    identity: str
    training_images_s3: List[str]  # S3 paths
    output_s3_path: str  # Where to save model
    token: str


class TrainLoRARequest(BaseModel):
    identity: str
    lora_name: str
    training_images_s3: List[str]  # S3 paths
    output_s3_path: str  # Where to save LoRA
    token: str


class GenerateImageRequest(BaseModel):
    identity: str
    identity_model: str  # S3 path to identity model
    prompt: str
    negative_prompt: str
    loras: Optional[List[str]] = None  # LoRA names
    loras_s3: Optional[List[str]] = None  # S3 paths to LoRAs
    references_s3: Optional[List[str]] = None  # S3 paths to references
    steps: int = 30
    seed: Optional[int] = None
    output_s3_path: str  # Where to save image


class GenerateVideoRequest(BaseModel):
    identity: str
    identity_model: str  # S3 path to identity model
    prompt: str
    negative_prompt: str
    loras: Optional[List[str]] = None  # LoRA names
    loras_s3: Optional[List[str]] = None  # S3 paths to LoRAs
    references_s3: Optional[List[str]] = None  # S3 paths to references
    num_frames: int = 16
    steps: int = 25
    seed: Optional[int] = None
    output_s3_path: str  # Where to save video


@router.post("/train/identity")
async def train_identity(request: TrainIdentityRequest):
    """
    Train DreamBooth identity model.
    
    Backend sends:
    - Identity name
    - S3 paths to training images
    - S3 path for output model
    - Token
    
    Worker:
    - Downloads images from S3
    - Trains model
    - Uploads model to S3
    - Returns job ID
    """
    try:
        # Import worker modules
        import os
        from app.core.gpu_lock import acquire_lock, release_lock
        from app.core.job_tracker import create_job
        from app.storage.s3_manager import S3Manager
        from app.training.dreambooth import train_dreambooth
        
        # Acquire GPU lock
        acquire_lock("train_identity")
        
        try:
            # Create job
            job_id = create_job("train_identity", request.identity)
            
            # Download images from S3
            s3_manager = S3Manager(bucket_name=os.getenv("S3_BUCKET_NAME", "ai-studio"))
            local_dir = f"/opt/ai-influencer/data/training/{request.identity}"
            os.makedirs(local_dir, exist_ok=True)
            
            local_paths = []
            for s3_path in request.training_images_s3:
                filename = os.path.basename(s3_path)
                local_path = os.path.join(local_dir, filename)
                s3_manager.download_file(s3_path, local_path)
                local_paths.append(local_path)
            
            # Train (this will be async in real implementation)
            # For now, return job ID
            # train_dreambooth(...)
            
            return {
                "job_id": job_id,
                "status": "running",
                "message": "Training started"
            }
        finally:
            release_lock()
    
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/lora")
async def train_lora(request: TrainLoRARequest):
    """
    Train LoRA model.
    
    Similar to train_identity but for LoRAs.
    """
    try:
        import os
        import asyncio
        from app.core.gpu_lock import acquire_lock, release_lock
        from app.core.job_tracker import create_job, update_job
        from app.storage.s3_manager import S3Manager
        from app.training.lora import train_lora as train_lora_impl
        
        acquire_lock("train_lora")
        
        try:
            job_id = create_job("train_lora", f"{request.identity}_{request.lora_name}")
            
            # Download images from S3
            s3_manager = S3Manager(bucket_name=os.getenv("S3_BUCKET_NAME", "ai-studio"))
            local_dir = f"/opt/ai-influencer/data/training/{request.identity}/loras/{request.lora_name}"
            os.makedirs(local_dir, exist_ok=True)
            
            local_paths = []
            for s3_path in request.training_images_s3:
                s3_key = s3_path.replace(f"s3://{s3_manager.bucket_name}/", "")
                filename = os.path.basename(s3_key)
                local_path = os.path.join(local_dir, filename)
                s3_manager.download_file(s3_key, local_path)
                local_paths.append(local_path)
            
            # Train LoRA (async in background)
            async def train_async():
                try:
                    base_model_path = f"/opt/ai-influencer/models/identities/{request.identity}"
                    output_path = f"/opt/ai-influencer/models/loras/{request.lora_name}.safetensors"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    train_lora_impl(
                        base_model_path=base_model_path,
                        instance_data_dir=local_dir,
                        output_path=output_path,
                        token=request.token,
                        identity_name=request.identity
                    )
                    
                    # Upload LoRA to S3
                    s3_key = f"models/loras/{request.lora_name}.safetensors"
                    s3_manager.upload_file(output_path, s3_key)
                    
                    # Update job status
                    update_job(job_id, status="done")
                except Exception as e:
                    logger.error(f"LoRA training failed: {e}")
                    update_job(job_id, status="failed", error=str(e))
                finally:
                    release_lock()
            
            # Start training in background
            asyncio.create_task(train_async())
            
            return {
                "job_id": job_id,
                "status": "running",
                "message": "LoRA training started"
            }
        except Exception as e:
            release_lock()
            raise
    
    except Exception as e:
        logger.error(f"Failed to start LoRA training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/image")
async def generate_image(request: GenerateImageRequest):
    """
    Generate image.
    
    Backend sends:
    - Identity, prompt, LoRAs, references (S3 paths)
    - Output S3 path
    
    Worker:
    - Downloads references from S3 if needed
    - Generates image
    - Uploads to S3
    - Returns job ID and S3 path
    """
    try:
        import os
        import asyncio
        from app.core.gpu_lock import is_locked, get_lock_info, acquire_lock, release_lock
        from app.core.job_tracker import create_job, update_job
        from app.storage.s3_manager import S3Manager
        import torch
        
        if is_locked():
            lock_info = get_lock_info()
            raise HTTPException(
                status_code=409,
                detail=f"GPU is busy: {lock_info.get('owner', 'unknown')}"
            )
        
        job_id = create_job("generate_image", request.identity)
        acquire_lock("generate_image")
        
        # Generate image (async in background)
        async def generate_async():
            try:
                # Initialize managers
                model_manager = ModelManager()
                s3_manager = S3Manager(bucket_name=os.getenv("S3_BUCKET_NAME", "ai-studio"))
                ref_registry = ReferenceRegistry()
                
                # Download identity model if needed
                identity_model_path = f"/opt/ai-influencer/models/identities/{request.identity}"
                if not os.path.exists(identity_model_path):
                    s3_key = request.identity_model.replace(f"s3://{s3_manager.bucket_name}/", "")
                    s3_manager.download_model(request.identity, identity_model_path)
                
                # Download LoRAs if needed
                if request.loras and request.loras_s3:
                    for lora_name, lora_s3 in zip(request.loras, request.loras_s3):
                        lora_path = f"/opt/ai-influencer/models/loras/{lora_name}.safetensors"
                        if not os.path.exists(lora_path):
                            s3_key = lora_s3.replace(f"s3://{s3_manager.bucket_name}/", "")
                            s3_manager.download_file(s3_key, lora_path)
                
                # Use modern PipelineManager for image generation
                pipeline_manager = get_pipeline_manager()
                
                # Load image pipeline
                pipe = pipeline_manager.load_image_pipeline()
                
                # Load identity model if provided
                if request.identity_model:
                    identity_model_path = f"/opt/ai-influencer/models/identities/{request.identity}"
                    if os.path.exists(identity_model_path):
                        # Load identity model into pipeline
                        from diffusers import StableDiffusionPipeline
                        identity_pipe = StableDiffusionPipeline.from_pretrained(
                            identity_model_path,
                            torch_dtype=torch.float16,
                            safety_checker=None,
                            requires_safety_checker=False
                        ).to(pipeline_manager.device)
                        # Copy identity weights to base pipeline
                        pipe.unet = identity_pipe.unet
                        pipe.text_encoder = identity_pipe.text_encoder
                        del identity_pipe
                        torch.cuda.empty_cache()
                
                # Attach LoRAs if provided
                if request.loras and request.loras_s3:
                    for lora_name, lora_s3 in zip(request.loras, request.loras_s3):
                        lora_path = f"/opt/ai-influencer/models/loras/{lora_name}.safetensors"
                        if os.path.exists(lora_path):
                            pipe = pipeline_manager.attach_lora(pipe, lora_path)
                
                # Generate image
                generator = None
                if request.seed is not None:
                    generator = torch.Generator(device=pipeline_manager.device).manual_seed(request.seed)
                
                output = pipe(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    num_inference_steps=request.steps,
                    generator=generator,
                    guidance_scale=7.5
                )
                
                image = output.images[0]
                
                # Save image
                import uuid
                output_dir = "/opt/ai-influencer/outputs/images"
                os.makedirs(output_dir, exist_ok=True)
                filename = f"{request.identity}_{uuid.uuid4().hex[:8]}.png"
                output_path = os.path.join(output_dir, filename)
                image.save(output_path)
                
                result = {
                    "success": True,
                    "content_type": "photo",
                    "path": output_path,
                    "prompt_used": request.prompt
                }
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Upload output to S3
                output_s3_key = request.output_s3_path.replace(f"s3://{s3_manager.bucket_name}/", "")
                s3_manager.upload_file(result["path"], output_s3_key)
                
                # Update job status
                update_job(job_id, status="done", output_s3_path=request.output_s3_path)
            except Exception as e:
                logger.error(f"Image generation failed: {e}")
                update_job(job_id, status="failed", error=str(e))
            finally:
                release_lock()
        
        # Start generation in background
        asyncio.create_task(generate_async())
        
        return {
            "job_id": job_id,
            "status": "running",
            "output_s3_path": request.output_s3_path,
            "message": "Image generation started"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start image generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/video")
async def generate_video(request: GenerateVideoRequest):
    """
    Generate video.
    
    Similar to generate_image but for videos.
    """
    try:
        import os
        import asyncio
        from app.core.gpu_lock import is_locked, get_lock_info, acquire_lock, release_lock
        from app.core.job_tracker import create_job, update_job
        from app.storage.s3_manager import S3Manager
        import torch
        
        if is_locked():
            lock_info = get_lock_info()
            raise HTTPException(
                status_code=409,
                detail=f"GPU is busy: {lock_info.get('owner', 'unknown')}"
            )
        
        job_id = create_job("generate_video", request.identity)
        acquire_lock("generate_video")
        
        # Generate video (async in background)
        async def generate_async():
            try:
                # Initialize managers
                import torch
                s3_manager = S3Manager(bucket_name=os.getenv("S3_BUCKET_NAME", "ai-studio"))
                
                # Download identity model if needed
                identity_model_path = f"/opt/ai-influencer/models/identities/{request.identity}"
                if not os.path.exists(identity_model_path):
                    s3_key = request.identity_model.replace(f"s3://{s3_manager.bucket_name}/", "")
                    s3_manager.download_model(request.identity, identity_model_path)
                
                # Download LoRAs if needed
                if request.loras and request.loras_s3:
                    for lora_name, lora_s3 in zip(request.loras, request.loras_s3):
                        lora_path = f"/opt/ai-influencer/models/loras/{lora_name}.safetensors"
                        if not os.path.exists(lora_path):
                            s3_key = lora_s3.replace(f"s3://{s3_manager.bucket_name}/", "")
                            s3_manager.download_file(s3_key, lora_path)
                
                # Use modern PipelineManager for video generation
                import imageio
                
                pipeline_manager = get_pipeline_manager()
                
                # Load video pipeline
                pipe = pipeline_manager.load_video_pipeline()
                
                # Load identity model if provided
                if request.identity_model:
                    identity_model_path = f"/opt/ai-influencer/models/identities/{request.identity}"
                    if os.path.exists(identity_model_path):
                        # Load identity model and copy weights to video pipeline
                        from diffusers import StableDiffusionPipeline
                        identity_pipe = StableDiffusionPipeline.from_pretrained(
                            identity_model_path,
                            torch_dtype=torch.float16,
                            safety_checker=None,
                            requires_safety_checker=False
                        ).to(pipeline_manager.device)
                        # Copy identity weights to video pipeline
                        pipe.unet = identity_pipe.unet
                        pipe.text_encoder = identity_pipe.text_encoder
                        pipe.vae = identity_pipe.vae
                        del identity_pipe
                        torch.cuda.empty_cache()
                
                # Attach LoRAs if provided
                if request.loras and request.loras_s3:
                    for lora_name, lora_s3 in zip(request.loras, request.loras_s3):
                        lora_path = f"/opt/ai-influencer/models/loras/{lora_name}.safetensors"
                        if os.path.exists(lora_path):
                            pipe = pipeline_manager.attach_lora(pipe, lora_path)
                
                # Generate video frames
                generator = None
                if request.seed is not None:
                    generator = torch.Generator(device=pipeline_manager.device).manual_seed(request.seed)
                
                output = pipe(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    num_inference_steps=request.steps,
                    num_frames=request.num_frames,
                    generator=generator,
                    guidance_scale=8.0
                )
                
                # Convert frames to video
                frames = output.frames[0]  # List of PIL Images
                
                # Save video
                import uuid
                output_dir = "/opt/ai-influencer/outputs/videos"
                os.makedirs(output_dir, exist_ok=True)
                filename = f"{request.identity}_{uuid.uuid4().hex[:8]}.mp4"
                video_path = os.path.join(output_dir, filename)
                
                # Convert frames to MP4
                imageio.mimsave(
                    video_path,
                    frames,
                    fps=8,
                    codec="libx264",
                    quality=8
                )
                
                result = {
                    "success": True,
                    "content_type": "video",
                    "path": video_path,
                    "prompt_used": request.prompt,
                    "num_frames": request.num_frames,
                    "duration_seconds": request.num_frames / 8
                }
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Upload output to S3
                output_s3_key = request.output_s3_path.replace(f"s3://{s3_manager.bucket_name}/", "")
                s3_manager.upload_file(result["path"], output_s3_key)
                
                # Update job status
                update_job(job_id, status="done", output_s3_path=request.output_s3_path)
            except Exception as e:
                logger.error(f"Video generation failed: {e}")
                update_job(job_id, status="failed", error=str(e))
            finally:
                release_lock()
        
        # Start generation in background
        asyncio.create_task(generate_async())
        
        return {
            "job_id": job_id,
            "status": "running",
            "output_s3_path": request.output_s3_path,
            "message": "Video generation started"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start video generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """Worker health check - lightweight, no heavy imports"""
    import torch
    
    gpu_available = torch.cuda.is_available() if torch.cuda.is_available() else False
    gpu_name = None
    
    if gpu_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name
    }


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    from app.core.job_tracker import load_jobs
    
    jobs = load_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

