"""
DreamBooth Training Script - Production Implementation
Real DreamBooth implementation using Diffusers (simplified but correct)
"""

import argparse
import os
import sys
import time
import torch
from pathlib import Path
from PIL import Image
import logging

# Add app to path for imports
sys.path.insert(0, "/opt/ai-influencer/ai-worker")

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from transformers import CLIPTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import numpy as np

from app.core.gpu_lock import acquire_lock, release_lock
from app.core.job_tracker import create_job, update_job
from app.storage.s3_manager import S3Manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    use_ema: bool = True
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
    logger.info("Loading base Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    # Enable gradient checkpointing for memory efficiency
    unet.enable_gradient_checkpointing()
    if hasattr(text_encoder, "gradient_checkpointing_enable"):
        text_encoder.gradient_checkpointing_enable()
    
    # Setup EMA
    if use_ema:
        ema_unet = EMAModel(unet.parameters())
    
    # Optimizer
    params_to_optimize = list(unet.parameters())
    if hasattr(text_encoder, "parameters"):
        params_to_optimize += list(text_encoder.parameters())
    
    optimizer = torch.optim.AdamW(
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
    
    for step in range(max_train_steps):
        unet.train()
        text_encoder.train()
        
        # Sample a random image
        image_path = images[step % len(images)]
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((resolution, resolution), Image.LANCZOS)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = (image_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        image_tensor = image_tensor.to(accelerator.device)
        
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
    try:
        parser = argparse.ArgumentParser(description="Train DreamBooth model for identity")
        parser.add_argument("--identity", required=True, help="Identity name")
        parser.add_argument("--token", required=True, help="Unique token (e.g., sks_ava)")
        parser.add_argument("--job-id", help="Job ID for tracking")
        parser.add_argument("--base-model", default="/mnt/models/base/sd15", help="Base model path")
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
        
        instance_data_dir = f"/opt/ai-influencer/data/identities/{args.identity}/images"
        output_dir = f"/opt/ai-influencer/models/identities/{args.identity}"
        
        # Train DreamBooth
        train_dreambooth(
            base_model_path=args.base_model,
            instance_data_dir=instance_data_dir,
            output_dir=output_dir,
            token=args.token,
            max_train_steps=args.steps,
            learning_rate=args.lr
        )
        
        # Upload model to S3 if output_s3_path is provided
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
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    local_file = os.path.join(root, file)
                    # Get relative path from output_dir
                    rel_path = os.path.relpath(local_file, output_dir)
                    s3_key = f"{s3_key_prefix}/{rel_path}".replace("\\", "/")  # Windows path fix
                    logger.info(f"Uploading {local_file} to s3://ai-studio-dc275989/{s3_key}")
                    s3_manager.upload_file(local_file, s3_key)
            
            logger.info(f"✅ Model uploaded to S3: {args.output_s3_path}")
        
        # Update job status
        update_job(
            job_id,
            status="completed",
            metadata={
                "model_path": output_dir,
                "s3_path": args.output_s3_path
            }
        )
        
        logger.info(f"✅ DreamBooth training completed: {args.identity}")
    
    except Exception as e:
        logger.error(f"DreamBooth training failed: {e}")
        if args.job_id:
            update_job(args.job_id, status="failed", error=str(e))
        raise
    
    finally:
        # Always release lock
        release_lock()
