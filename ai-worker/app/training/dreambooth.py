"""
DreamBooth Training Script - Production Implementation
Real DreamBooth implementation using Diffusers (simplified but correct)
"""

import argparse
import os
import sys
import time
import asyncio
import torch
from pathlib import Path
from PIL import Image
import logging
import bitsandbytes as bnb
from typing import Optional

# Add app to path for imports
sys.path.insert(0, "/opt/ai-influencer/ai-worker")

# Setup logging FIRST before any imports that might fail
# Ensure log directory exists
log_dir = "/opt/ai-influencer/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),  # Always log to stderr
        logging.FileHandler(f'{log_dir}/training_error.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

try:
    logger.info("[DreamBooth] Starting imports...")
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    from transformers import CLIPTokenizer
    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from accelerate.utils import ProjectConfiguration
    import numpy as np
    logger.info("[DreamBooth] ✅ Core ML libraries imported")
    
    from app.core.gpu_lock import acquire_lock, release_lock
    from app.core.job_tracker import create_job, update_job
    from app.storage.s3_manager import S3Manager
    import httpx
    logger.info("[DreamBooth] ✅ App modules imported")
except ImportError as e:
    logger.error(f"[DreamBooth] ❌ Import error: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.error(f"[DreamBooth] ❌ Unexpected error during imports: {e}", exc_info=True)
    sys.exit(1)


def train_dreambooth(
    base_model_path: str,
    instance_data_dir: str,
    output_dir: str,
    token: str,
    steps: int = 2000,
    lr: float = 1e-6,
    resolution: int = 512,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-6,
    max_train_steps: int = 800,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    use_ema: bool = True,
    job_id: Optional[str] = None
):
    """
    Train DreamBooth model for a specific identity.
    
    This is a real DreamBooth implementation using Diffusers.
    Single-identity, sequential, deterministic - perfect for influencers.
    """
    logger.info(f"Starting DreamBooth training for token: {token}")
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Training images: {instance_data_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs")
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16",
        project_config=accelerator_project_config
    )
    
    device = accelerator.device
    
    # Load base model
    # IMPORTANT: Load in float32 when using Accelerate mixed_precision
    # Accelerate will handle FP16 conversion automatically during training
    logger.info("Loading base Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,  # Always float32 - Accelerate handles FP16
        safety_checker=None,
        requires_safety_checker=False
    )
    
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    # Optimizer
    params_to_optimize = list(unet.parameters())
    if hasattr(text_encoder, "parameters"):
        params_to_optimize += list(text_encoder.parameters())
    
    # Use 8-bit Adam to reduce optimizer memory by ~75%
    # This solves the OOM during optimizer.step() without quality loss
    optimizer = bnb.optim.AdamW8bit(
        params_to_optimize,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08
    )
    
    # Learning rate scheduler
    lr_scheduler_obj = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )
    
    # Prepare with accelerator
    unet, text_encoder, optimizer, lr_scheduler_obj = accelerator.prepare(
        unet, text_encoder, optimizer, lr_scheduler_obj
    )
    
    # Setup EMA (must be after accelerator.prepare to ensure correct device)
    if use_ema:
        ema_unet = EMAModel(unet.parameters())
        # Ensure EMA is on the same device as the model
        model_device = next(unet.parameters()).device
        ema_unet.to(model_device)
    else:
        ema_unet = None
    
    # Load training images
    images = sorted([
        os.path.join(instance_data_dir, f)
        for f in os.listdir(instance_data_dir)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ])
    
    if not images:
        raise ValueError(f"No training images found in {instance_data_dir}")
    
    logger.info(f"Found {len(images)} training images")
    
    # Instance prompt with token
    instance_prompt = f"a photo of {token} person"
    
    # Move VAE to device
    vae.to(accelerator.device)
    vae.eval()
    
    # Training loop
    logger.info(f"Starting training for {max_train_steps} steps...")
    
    # Heartbeat tracking
    last_heartbeat = time.time()
    heartbeat_interval = 30.0  # Send heartbeat every 30 seconds
    
    for step in range(max_train_steps):
        unet.train()
        text_encoder.train()
        
        # Sample a random image
        image_path = images[step % len(images)]
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((resolution, resolution), Image.LANCZOS)
        
        # Convert to tensor
        # Models are in float32, Accelerate handles FP16 conversion
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = (image_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        image_tensor = image_tensor.to(device=device)
        
        # Encode to latent space
        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.sample()
            latent = latent * vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latent)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latent.shape[0],),
            device=latent.device
        ).long()
        
        # Add noise
        noisy_latent = noise_scheduler.add_noise(latent, noise, timesteps)
        
        # Get text embeddings
        token_ids = tokenizer(
            instance_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(accelerator.device)
        
        with accelerator.autocast():
            encoder_hidden_states = text_encoder(token_ids)[0]
        
        # Predict noise
        model_pred = unet(
            noisy_latent,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # Compute loss
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latent, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        # Backward pass
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(params_to_optimize, 1.0)
            optimizer.step()
            lr_scheduler_obj.step()
            optimizer.zero_grad()
            
            if use_ema:
                ema_unet.step(unet.parameters())
        
        # Send heartbeat every 30 seconds (non-blocking)
        current_time = time.time()
        if job_id and (current_time - last_heartbeat >= heartbeat_interval):
            last_heartbeat = current_time
            try:
                import threading
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                
                def send_heartbeat():
                    try:
                        async def heartbeat():
                            async with httpx.AsyncClient(timeout=5.0) as client:
                                response = await client.post(
                                    f"{backend_url}/worker/job/heartbeat",
                                    params={"job_id": job_id}
                                )
                                response.raise_for_status()
                        try:
                            loop = asyncio.get_event_loop()
                            loop.run_until_complete(heartbeat())
                        except RuntimeError:
                            asyncio.run(heartbeat())
                    except Exception as e:
                        logger.debug(f"Heartbeat failed: {e}")
                
                # Send in background thread (non-blocking)
                threading.Thread(target=send_heartbeat, daemon=True).start()
            except Exception as e:
                logger.debug(f"Failed to send heartbeat: {e}")
        
        if step % 100 == 0:
            logger.info(f"[DreamBooth] step={step}/{max_train_steps} loss={loss.item():.4f}")
    
    # Save model
    logger.info("Saving trained model...")
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        text_encoder = accelerator.unwrap_model(text_encoder)
        
        if use_ema:
            ema_unet.copy_to(unet.parameters())
        
        # Save pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            unet=unet,
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            scheduler=noise_scheduler
        )
        
        os.makedirs(output_dir, exist_ok=True)
        pipeline.save_pretrained(output_dir)
        logger.info(f"[DreamBooth] Model saved to {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    # Immediate log to verify script started
    print("[DreamBooth] Script started", file=sys.stderr, flush=True)
    logger.info("[DreamBooth] ========================================")
    logger.info("[DreamBooth] DreamBooth Training Script Starting")
    logger.info("[DreamBooth] ========================================")
    
    try:
        parser = argparse.ArgumentParser(description="Train DreamBooth model for identity")
        parser.add_argument("--identity", required=True, help="Identity name")
        parser.add_argument("--token", required=True, help="Unique token (e.g., sks_ava)")
        parser.add_argument("--job-id", help="Job ID for tracking")
        parser.add_argument("--base-model", default="/opt/ai-influencer/models/base/sd15", help="Base model path")
        parser.add_argument("--steps", type=int, default=800, help="Training steps")
        parser.add_argument("--lr", type=float, default=2e-6, help="Learning rate")
        parser.add_argument("--output-s3-path", help="S3 path to upload trained model")
        
        args = parser.parse_args()
        
        logger.info(f"[DreamBooth] Starting training script | identity={args.identity} | job_id={args.job_id}")
        logger.info(f"[DreamBooth] Base model: {args.base_model}")
        logger.info(f"[DreamBooth] Steps: {args.steps} | LR: {args.lr}")
        
        # Get or create job_id
        job_id = args.job_id or f"dreambooth_{args.identity}_{int(time.time())}"
        
        # Acquire GPU lock
        logger.info(f"[DreamBooth] Acquiring GPU lock | job_id={job_id}")
        try:
            acquire_lock("dreambooth_training", job_id)
            logger.info(f"[DreamBooth] ✅ GPU lock acquired")
        except RuntimeError as e:
            logger.error(f"[DreamBooth] ❌ Failed to acquire GPU lock: {e}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"[DreamBooth] ❌ Fatal error during initialization: {e}", exc_info=True)
        sys.exit(1)
    
    try:
        # Create job entry
        if not args.job_id:
            job_id = create_job(
                job_type="dreambooth_training",
                target=args.identity,
                metadata={
                    "token": args.token,
                    "steps": args.steps
                }
            )
        
        # Set job_id in environment for heartbeat function
        os.environ["CURRENT_JOB_ID"] = job_id
        
        # Step 2: Call backend to start job (transitions queued → running)
        # Note: If called through worker poller, job may already be started
        # If this times out or fails, we continue anyway (poller may have started it)
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        try:
            async def start_job():
                async with httpx.AsyncClient(timeout=30.0) as client:  # Increased timeout to 30s
                    try:
                        response = await client.post(
                            f"{backend_url}/worker/job/start",
                            json={
                                "job_id": job_id,
                                "worker_instance_id": os.getenv("INSTANCE_ID")  # Optional
                            }
                        )
                        # If job is already running (400 error), that's OK - poller may have started it
                        if response.status_code == 400:
                            error_text = response.text
                            if "already" in error_text.lower() or "running" in error_text.lower():
                                logger.info(f"Job already started (likely by poller) | job_id={job_id}")
                                return
                        response.raise_for_status()
                        logger.info(f"✅ Job started in backend | job_id={job_id}")
                    except httpx.TimeoutException as timeout_err:
                        logger.warning(f"⚠️ Timeout calling backend start endpoint (backend may be slow). Continuing anyway - job may already be started by poller.")
                        # Continue - if called through poller, job is already started
                        return
                    except httpx.HTTPStatusError as http_err:
                        # If it's a 400 error about job already running, continue
                        if http_err.response.status_code == 400:
                            error_text = http_err.response.text
                            if "already" in error_text.lower() or "running" in error_text.lower():
                                logger.info(f"Job already started (likely by poller) | job_id={job_id}")
                                return
                        raise  # Re-raise other HTTP errors
            
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(start_job())
            except RuntimeError:
                asyncio.run(start_job())
        except (httpx.TimeoutException, httpx.ReadTimeout) as timeout_err:
            # Timeout is not critical if called through poller - job may already be started
            logger.warning(f"⚠️ Timeout starting job in backend (backend may be slow). Continuing anyway - job may already be started by poller. Error type: {type(timeout_err).__name__}")
        except Exception as e:
            # Check if it's a timeout-related error (httpx wraps httpcore exceptions)
            error_type = type(e).__name__
            if "Timeout" in error_type or "timeout" in str(e).lower():
                logger.warning(f"⚠️ Timeout starting job in backend. Continuing anyway - job may already be started by poller. Error: {error_type}")
            # If it's a 400 error about job already running, continue
            elif "400" in str(e) and ("already" in str(e).lower() or "running" in str(e).lower()):
                logger.info(f"Job already started, continuing | job_id={job_id}")
            else:
                # For other errors, log but continue - if called through poller, job is already started
                logger.warning(f"⚠️ Failed to start job in backend (may already be started by poller). Continuing. Error: {error_type}: {e}")
        
        instance_data_dir = f"/opt/ai-influencer/data/identities/{args.identity}/images"
        output_dir = f"/opt/ai-influencer/models/identities/{args.identity}"
        
        # Train DreamBooth (pass job_id for heartbeat)
        train_dreambooth(
            base_model_path=args.base_model,
            instance_data_dir=instance_data_dir,
            output_dir=output_dir,
            token=args.token,
            max_train_steps=args.steps,
            learning_rate=args.lr,
            job_id=job_id  # Pass job_id for heartbeat mechanism
        )
        
        # Upload model to S3 if output_s3_path is provided
        # ATOMIC: Only update backend AFTER S3 upload completes
        s3_path = None
        if args.output_s3_path:
            logger.info(f"Uploading trained model to S3: {args.output_s3_path}")
            s3_manager = S3Manager(bucket_name="ai-studio-dc275989")
            
            # Upload entire model directory to S3
            # Extract S3 key from full path (s3://bucket/key)
            if args.output_s3_path.startswith("s3://"):
                s3_key_prefix = args.output_s3_path.replace("s3://ai-studio-dc275989/", "")
            else:
                s3_key_prefix = args.output_s3_path
            
            # Upload all files in the model directory
            uploaded_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    local_file = os.path.join(root, file)
                    # Get relative path from output_dir
                    rel_path = os.path.relpath(local_file, output_dir)
                    s3_key = f"{s3_key_prefix}/{rel_path}".replace("\\", "/")  # Windows path fix
                    logger.info(f"Uploading {local_file} to s3://ai-studio-dc275989/{s3_key}")
                    s3_manager.upload_file(local_file, s3_key)
                    uploaded_files.append(s3_key)
            
            s3_path = args.output_s3_path
            logger.info(f"✅ Model uploaded to S3: {s3_path} | files={len(uploaded_files)}")
        
        # ATOMIC: Update backend ONLY after all processing (training + S3 upload) is complete
        # This ensures backend state is consistent - job is "done" only when everything is finished
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        try:
            # Run async update in sync context
            async def update_backend():
                async with httpx.AsyncClient(timeout=30.0) as client:
                    request_data = {
                        "job_id": job_id,
                        "status": "done",  # Use "done" to match state machine
                        "metadata": {
                            "model_path": output_dir,
                            "s3_path": s3_path,
                            "uploaded_files": len(uploaded_files) if args.output_s3_path else 0
                        }
                    }
                    logger.info(f"Updating backend | job_id={job_id} | url={backend_url}/worker/job/update | data={request_data}")
                    response = await client.post(
                        f"{backend_url}/worker/job/update",
                        json=request_data
                    )
                    if response.status_code != 200:
                        error_detail = response.text
                        logger.error(f"Backend returned {response.status_code}: {error_detail}")
                    response.raise_for_status()
                    logger.info(f"✅ Backend updated | job_id={job_id} | status=done")
            
            # Run async update
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(update_backend())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(update_backend())
        except Exception as e:
            logger.error(f"❌ Failed to update backend: {e}")
            # Still update local job tracker for visibility
            update_job(job_id, status="completed", metadata={"s3_path": s3_path})
            raise  # Re-raise so caller knows backend update failed
        
        # Also update local job tracker for worker visibility
        update_job(
            job_id,
            status="completed",
            metadata={
                "model_path": output_dir,
                "s3_path": s3_path
            }
        )
        
        logger.info(f"✅ DreamBooth training completed: {args.identity}")
    
    except Exception as e:
        logger.error(f"DreamBooth training failed: {e}")
        if args.job_id:
            # Update backend on failure
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            try:
                async def update_backend_failed():
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        # First, update job status to failed
                        request_data = {
                            "job_id": args.job_id,
                            "status": "failed",
                            "error": str(e)
                        }
                        logger.info(f"Updating backend with failure | job_id={args.job_id} | url={backend_url}/worker/job/update | data={request_data}")
                        response = await client.post(
                            f"{backend_url}/worker/job/update",
                            json=request_data
                        )
                        if response.status_code != 200:
                            error_detail = response.text
                            logger.error(f"Backend returned {response.status_code}: {error_detail}")
                        response.raise_for_status()
                        logger.info(f"Backend updated with failure | job_id={args.job_id}")
                        
                        # Then, request cleanup of identity (backend will delete S3, manifest, etc.)
                        # Get identity name from job or args
                        identity_name = args.identity
                        if identity_name:
                            logger.info(f"Requesting cleanup of failed identity: {identity_name}")
                            try:
                                delete_response = await client.delete(
                                    f"{backend_url}/identities/{identity_name}"
                                )
                                if delete_response.status_code == 200:
                                    logger.info(f"✅ Cleaned up failed identity: {identity_name}")
                                else:
                                    logger.warning(f"Failed to cleanup identity {identity_name}: {delete_response.status_code}")
                            except Exception as cleanup_error:
                                logger.error(f"Failed to cleanup identity {identity_name}: {cleanup_error}")
                
                # Run async update
                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(update_backend_failed())
                except RuntimeError:
                    # No event loop, create one
                    asyncio.run(update_backend_failed())
            except Exception as backend_error:
                logger.error(f"Failed to update backend on failure: {backend_error}")
            
            # Clean up local worker cache for this identity
            try:
                identity_name = args.identity
                if identity_name:
                    import shutil
                    # Delete local model cache (if any partial files exist)
                    local_model_path = f"/opt/ai-influencer/models/identities/{identity_name}"
                    if os.path.exists(local_model_path):
                        shutil.rmtree(local_model_path)
                        logger.info(f"Cleaned up local model cache: {local_model_path}")
                    
                    # Delete training data cache
                    training_data_path = f"/opt/ai-influencer/data/training/{identity_name}"
                    if os.path.exists(training_data_path):
                        shutil.rmtree(training_data_path)
                        logger.info(f"Cleaned up local training data cache: {training_data_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup local cache: {cleanup_error}")
            
            # Also update local job tracker
            update_job(args.job_id, status="failed", error=str(e))
        raise
    
    finally:
        # Always release lock
        release_lock()
