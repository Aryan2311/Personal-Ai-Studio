"""
Worker API - Private endpoints for backend communication
EC2 worker exposes minimal API for job execution
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import time

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
        
        # Determine model paths (using root disk, no EBS)
        base_model_path = os.getenv("BASE_MODEL_PATH", "/opt/ai-influencer/models/base/sd15")
        motion_path = os.getenv("MOTION_MODEL_PATH", "/opt/ai-influencer/models/motion/animatediff")
        
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
    job_id: Optional[str] = None  # Backend's job_id (optional for backward compatibility)
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
    - Checks GPU availability
    - Downloads images from S3
    - Launches training in background subprocess
    - Returns job ID immediately
    """
    import os
    import subprocess
    import shutil
    from pathlib import Path
    from app.core.gpu_lock import is_locked, get_lock_info
    from app.core.job_tracker import create_job
    from app.storage.s3_manager import S3Manager
    
    try:
        # Check if GPU is available
        if is_locked():
            lock_info = get_lock_info()
            raise HTTPException(
                status_code=409,
                detail=f"GPU is busy: {lock_info.get('owner', 'unknown')} (job: {lock_info.get('job_id', 'unknown')})"
            )
        
        # Create job FIRST (before any work)
        job_id = create_job(
            job_type="train_identity",
            target=request.identity,
            metadata={
                "token": request.token,
                "output_s3_path": request.output_s3_path
            }
        )
        
        logger.info(f"[TRAIN IDENTITY] Job created | job_id={job_id} | identity={request.identity}")
        
        # Download images from S3
        s3_manager = S3Manager(bucket_name="ai-studio-dc275989")
        temp_dir = f"/opt/ai-influencer/data/training/{request.identity}"
        os.makedirs(temp_dir, exist_ok=True)
        
        logger.info(f"[TRAIN IDENTITY] Downloading {len(request.training_images_s3)} images from S3...")
        for idx, s3_path in enumerate(request.training_images_s3, 1):
            filename = os.path.basename(s3_path)
            local_path = os.path.join(temp_dir, filename)
            logger.info(f"[TRAIN IDENTITY] Downloading image {idx}/{len(request.training_images_s3)} | s3_path={s3_path}")
            s3_manager.download_file(s3_path, local_path)
        
        # Move images to location expected by training script
        # Script expects: /opt/ai-influencer/data/identities/{identity}/images/
        identity_images_dir = f"/opt/ai-influencer/data/identities/{request.identity}/images"
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
        # Check for base model (using root disk, no EBS)
        base_model_path = "/opt/ai-influencer/models/base/sd15"
        if not os.path.exists(base_model_path):
            logger.error(f"❌ Base model not found at {base_model_path}. Training cannot proceed.")
            raise HTTPException(
                status_code=500,
                detail=f"Base model not found. Expected at /opt/ai-influencer/models/base/sd15"
            )
        
        logger.info(f"[TRAIN IDENTITY] Using base model: {base_model_path}")
        
        training_script = "/opt/ai-influencer/ai-worker/app/training/dreambooth.py"
        python_exec = "/opt/ai-venv/bin/python"
        
        # Build command
        cmd = [
            python_exec,
            training_script,
            "--identity", request.identity,
            "--token", request.token,
            "--job-id", job_id,
            "--base-model", base_model_path,
            "--steps", "800",  # Default training steps
            "--lr", "2e-6",    # Default learning rate
            "--output-s3-path", request.output_s3_path
        ]
        
        logger.info(f"[TRAIN IDENTITY] Launching training subprocess | cmd={' '.join(cmd)}")
        
        # Launch training in background
        # Use subprocess.Popen with proper logging
        log_dir = f"/opt/ai-influencer/logs/training/{request.identity}"
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
        logger.info(f"[TRAIN IDENTITY] Logs: {log_dir}/stdout.log and {log_dir}/stderr.log")
        
        # Don't wait for process - return immediately
        # The training script will:
        # 1. Acquire GPU lock
        # 2. Run training
        # 3. Upload model to S3
        # 4. Update job status
        # 5. Release GPU lock
        
        return {
            "job_id": job_id,
            "status": "running",
            "message": "Training started",
            "pid": process.pid,
            "log_dir": log_dir
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TRAIN IDENTITY] ❌ Failed to start training: {e}", exc_info=True)
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
            s3_manager = S3Manager(bucket_name="ai-studio-dc275989")
            local_dir = f"/opt/ai-influencer/data/training/{request.identity}/loras/{request.lora_name}"
            os.makedirs(local_dir, exist_ok=True)
            
            local_paths = []
            for s3_path in request.training_images_s3:
                # S3Manager.download_file now handles both full S3 paths and keys
                filename = os.path.basename(s3_path)
                local_path = os.path.join(local_dir, filename)
                logger.info(f"Downloading LoRA training image | s3_path={s3_path} | local_path={local_path}")
                s3_manager.download_file(s3_path, local_path)
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
        
        # Use backend's job_id if provided, otherwise create one
        job_id = request.job_id if request.job_id else create_job("generate_image", request.identity)
        
        # If using backend's job_id, still create local job tracker entry
        if request.job_id:
            from app.core.job_tracker import load_jobs, save_jobs
            jobs = load_jobs()
            if job_id not in jobs:
                jobs[job_id] = {
                    "job_type": "generate_image",
                    "target": request.identity,
                    "status": "running",
                    "created_at": time.time(),
                    "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "metadata": {}
                }
                save_jobs(jobs)
            logger.info(f"[GENERATE IMAGE] Using backend job_id | job_id={job_id}")
        else:
            logger.info(f"[GENERATE IMAGE] Created new job_id | job_id={job_id}")
        
        acquire_lock("generate_image")
        
        # Generate image (async in background)
        async def generate_async():
            try:
                logger.info(f"[GENERATE IMAGE] Starting generation | job_id={job_id} | identity={request.identity}")
                
                # Initialize managers
                s3_manager = S3Manager(bucket_name="ai-studio-dc275989")
                
                # Download identity model if needed
                identity_model_path = f"/opt/ai-influencer/models/identities/{request.identity}"
                if not os.path.exists(identity_model_path):
                    logger.info(f"[GENERATE IMAGE] Downloading identity model | identity={request.identity}")
                    s3_key = request.identity_model.replace(f"s3://{s3_manager.bucket_name}/", "")
                    s3_manager.download_model(request.identity, identity_model_path)
                else:
                    logger.info(f"[GENERATE IMAGE] Using cached identity model | identity={request.identity}")
                
                # Download LoRAs if needed
                if request.loras and request.loras_s3:
                    logger.info(f"[GENERATE IMAGE] Downloading {len(request.loras)} LoRA(s)")
                    for lora_name, lora_s3 in zip(request.loras, request.loras_s3):
                        lora_path = f"/opt/ai-influencer/models/loras/{lora_name}.safetensors"
                        if not os.path.exists(lora_path):
                            logger.info(f"[GENERATE IMAGE] Downloading LoRA | lora={lora_name}")
                            s3_key = lora_s3.replace(f"s3://{s3_manager.bucket_name}/", "")
                            s3_manager.download_file(s3_key, lora_path)
                        else:
                            logger.info(f"[GENERATE IMAGE] Using cached LoRA | lora={lora_name}")
                
                # Use modern PipelineManager for image generation
                logger.info(f"[GENERATE IMAGE] Loading image pipeline")
                pipeline_manager = get_pipeline_manager()
                
                # Load image pipeline
                pipe = pipeline_manager.load_image_pipeline()
                logger.info(f"[GENERATE IMAGE] Pipeline loaded")
                
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
                logger.info(f"[GENERATE IMAGE] Starting inference | prompt={request.prompt[:50]}... | steps={request.steps}")
                generator = None
                if request.seed is not None:
                    generator = torch.Generator(device=pipeline_manager.device).manual_seed(request.seed)
                    logger.info(f"[GENERATE IMAGE] Using seed={request.seed}")
                
                output = pipe(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    num_inference_steps=request.steps,
                    generator=generator,
                    guidance_scale=7.5
                )
                
                image = output.images[0]
                logger.info(f"[GENERATE IMAGE] Image generated successfully")
                
                # Save image
                import uuid
                output_dir = "/opt/ai-influencer/outputs/images"
                os.makedirs(output_dir, exist_ok=True)
                filename = f"{request.identity}_{uuid.uuid4().hex[:8]}.png"
                output_path = os.path.join(output_dir, filename)
                image.save(output_path)
                logger.info(f"[GENERATE IMAGE] Image saved locally | path={output_path}")
                
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
                logger.info(f"✅ Image uploaded to S3: {request.output_s3_path}")
                
                # Update local job status
                update_job(job_id, status="done", metadata={"output_s3_path": request.output_s3_path})
                
                # Send status update to backend via SQS (or HTTP fallback)
                from app.core.status_sender import StatusSender
                status_sender = StatusSender()
                status_sender.send_done(
                    job_id=job_id,
                    metadata={
                        "output_s3_path": request.output_s3_path,
                        "content_type": "photo",
                        "prompt_used": request.prompt
                    }
                )
                logger.info(f"✅ Status update sent to backend | job_id={job_id} | output={request.output_s3_path}")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[GENERATE IMAGE] Image generation failed | job_id={job_id} | error={error_msg}", exc_info=True)
                update_job(job_id, status="failed", error=error_msg)
                
                # Send failure status to backend via SQS
                from app.core.status_sender import StatusSender
                status_sender = StatusSender()
                success = status_sender.send_failed(job_id=job_id, error=error_msg)
                if success:
                    logger.info(f"[GENERATE IMAGE] ✅ Failure status sent to backend | job_id={job_id}")
                else:
                    logger.error(f"[GENERATE IMAGE] ❌ Failed to send failure status to backend | job_id={job_id}")
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
                s3_manager = S3Manager(bucket_name="ai-studio-dc275989")
                
                # Download identity model if needed
                identity_model_path = f"/opt/ai-influencer/models/identities/{request.identity}"
                if not os.path.exists(identity_model_path) and request.identity_model:
                    logger.info(f"Downloading identity model for video | s3_path={request.identity_model}")
                    s3_manager.download_model(request.identity, identity_model_path)
                
                # Download LoRAs if needed
                if request.loras and request.loras_s3:
                    for lora_name, lora_s3 in zip(request.loras, request.loras_s3):
                        lora_path = f"/opt/ai-influencer/models/loras/{lora_name}.safetensors"
                        if not os.path.exists(lora_path):
                            logger.info(f"Downloading LoRA for video | s3_path={lora_s3} | local_path={lora_path}")
                            s3_manager.download_file(lora_s3, lora_path)
                
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
                logger.info(f"✅ Video uploaded to S3: {request.output_s3_path}")
                
                # Update local job status
                update_job(job_id, status="done", metadata={"output_s3_path": request.output_s3_path})
                
                # Send status update to backend via SQS (or HTTP fallback)
                from app.core.status_sender import StatusSender
                status_sender = StatusSender()
                status_sender.send_done(
                    job_id=job_id,
                    metadata={
                        "output_s3_path": request.output_s3_path,
                        "content_type": "video",
                        "prompt_used": request.prompt
                    }
                )
                logger.info(f"✅ Status update sent to backend | job_id={job_id} | output={request.output_s3_path}")
            except Exception as e:
                logger.error(f"Video generation failed: {e}")
                update_job(job_id, status="failed", error=str(e))
                
                # Send failure status to backend via SQS
                from app.core.status_sender import StatusSender
                status_sender = StatusSender()
                status_sender.send_failed(job_id=job_id, error=str(e))
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


@router.post("/cache/clear")
async def clear_cache(request: dict):
    """
    Clear worker cache for an identity.
    
    Deletes:
    - Local model cache
    - Local LoRA cache
    - Local training data cache
    """
    try:
        identity_name = request.get("identity")
        if not identity_name:
            raise HTTPException(status_code=400, detail="identity field required")
        
        import shutil
        import os
        
        deleted_paths = []
        
        # Delete identity model cache
        identity_model_path = f"/opt/ai-influencer/models/identities/{identity_name}"
        if os.path.exists(identity_model_path):
            shutil.rmtree(identity_model_path)
            deleted_paths.append(identity_model_path)
            logger.info(f"Deleted identity model cache: {identity_model_path}")
        
        # Delete LoRA cache for this identity
        loras_dir = "/opt/ai-influencer/models/loras"
        if os.path.exists(loras_dir):
            for item in os.listdir(loras_dir):
                if item.startswith(f"{identity_name}_"):
                    lora_path = os.path.join(loras_dir, item)
                    if os.path.isfile(lora_path):
                        os.remove(lora_path)
                        deleted_paths.append(lora_path)
                    elif os.path.isdir(lora_path):
                        shutil.rmtree(lora_path)
                        deleted_paths.append(lora_path)
                    logger.info(f"Deleted LoRA cache: {lora_path}")
        
        # Delete training data cache
        training_data_path = f"/opt/ai-influencer/data/training/{identity_name}"
        if os.path.exists(training_data_path):
            shutil.rmtree(training_data_path)
            deleted_paths.append(training_data_path)
            logger.info(f"Deleted training data cache: {training_data_path}")
        
        return {
            "status": "ok",
            "identity": identity_name,
            "deleted_paths": deleted_paths,
            "count": len(deleted_paths)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

