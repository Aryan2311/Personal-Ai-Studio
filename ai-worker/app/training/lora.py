"""
LoRA Training Module
Trains LoRA adapters for associated entities (boyfriend, dog, house, etc.)
NOT for primary identity - that uses DreamBooth

Key rule:
- LoRA = secondary entities
- DreamBooth = primary identity only

LoRA training characteristics:
- VRAM: 6-10 GB
- Time: 10-20 min
- Re-trainable anytime
- Zero identity drift
"""

import os
import torch
import argparse
from typing import List
import logging
from pathlib import Path
from PIL import Image
import numpy as np

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from transformers import CLIPTokenizer
import json
import sys

# Add app to path for imports (if needed for future imports)
sys.path.insert(0, "/opt/ai-influencer/ai-worker")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_lora(
    base_model_path: str,
    instance_data_dir: str,
    output_dir: str,
    token: str,
    identity_name: str,
    max_train_steps: int = 400,
    learning_rate: float = 1e-4,
    resolution: int = 512,
    lora_rank: int = 16,
    lora_alpha: int = 32
):
    """
    Train a LoRA adapter for an associated entity.
    
    Example: Train LoRA for Leela's dog
    - Token: ldog_leela
    - Caption: "a photo of ldog_leela dog"
    - Takes ~10-20 minutes
    - Lightweight, reversible
    
    Args:
        base_model_path: Path to base SD model (or identity model)
        instance_data_dir: Directory with training images
        output_dir: Where to save LoRA
        token: Unique token (e.g., "ldog_leela")
        identity_name: Primary identity name (e.g., "leela")
        max_train_steps: Training steps (400 is usually enough for LoRA)
        learning_rate: Learning rate (higher than DreamBooth)
        resolution: Image resolution
        lora_rank: LoRA rank (16 is standard)
        lora_alpha: LoRA alpha (usually 2x rank)
    """
    logger.info(f"Starting LoRA training for token: {token}")
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Training images: {instance_data_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs")
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        project_config=accelerator_project_config
    )
    
    device = accelerator.device
    
    # Load base model (or identity model if available)
    # For LoRA, we can use the identity model as base
    identity_model_path = f"/opt/ai-influencer/models/identities/{identity_name}"
    if os.path.exists(identity_model_path):
        logger.info(f"Using identity model as base: {identity_model_path}")
        model_path = identity_model_path
    else:
        logger.info(f"Using base SD model: {base_model_path}")
        model_path = base_model_path
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    # Setup LoRA
    # PEFT 0.11.1+ supports TEXT_TO_IMAGE for Stable Diffusion
    # Check if TEXT_TO_IMAGE exists before using it
    if hasattr(TaskType, 'TEXT_TO_IMAGE'):
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            task_type=TaskType.TEXT_TO_IMAGE
        )
        logger.info("Using TaskType.TEXT_TO_IMAGE for LoRA config")
    else:
        # Fallback for older PEFT versions without TEXT_TO_IMAGE
        logger.warning("TaskType.TEXT_TO_IMAGE not available, creating LoRA config without task_type")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        )
    
    unet_lora = get_peft_model(unet, lora_config)
    
    # Enable gradient checkpointing
    unet_lora.enable_gradient_checkpointing()
    
    # Optimizer (only LoRA parameters)
    optimizer = torch.optim.AdamW(
        unet_lora.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08
    )
    
    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps
    )
    
    # Prepare with accelerator
    unet_lora, optimizer, lr_scheduler = accelerator.prepare(
        unet_lora, optimizer, lr_scheduler
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
    instance_prompt = f"a photo of {token}"
    
    # Move VAE to device
    vae.to(accelerator.device)
    vae.eval()
    text_encoder.to(accelerator.device)
    text_encoder.eval()
    
    # Training loop
    logger.info(f"Starting LoRA training for {max_train_steps} steps...")
    
    for step in range(max_train_steps):
        unet_lora.train()
        
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
        
        with torch.no_grad():
            encoder_hidden_states = text_encoder(token_ids)[0]
        
        # Predict noise
        model_pred = unet_lora(
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
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        if step % 50 == 0:
            logger.info(f"[LoRA] step={step}/{max_train_steps} loss={loss.item():.4f}")
    
    # Save LoRA
    logger.info("Saving LoRA adapter...")
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        unet_lora = accelerator.unwrap_model(unet_lora)
        
        os.makedirs(output_dir, exist_ok=True)
        unet_lora.save_pretrained(output_dir)
        
        logger.info(f"[LoRA] LoRA saved to {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA adapter for associated entity")
    parser.add_argument("--lora-name", required=True, help="LoRA name (e.g., leela_dog)")
    parser.add_argument("--token", required=True, help="Unique token (e.g., ldog_leela)")
    parser.add_argument("--identity", required=True, help="Primary identity name (e.g., leela)")
    parser.add_argument("--base-model", default="/opt/ai-influencer/models/base/sd15", help="Base model path")
    parser.add_argument("--steps", type=int, default=400, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    instance_data_dir = f"/opt/ai-influencer/data/loras/{args.lora_name}/images"
    output_dir = f"/opt/ai-influencer/models/loras/{args.lora_name}"
    
    train_lora(
        base_model_path=args.base_model,
        instance_data_dir=instance_data_dir,
        output_dir=output_dir,
        token=args.token,
        identity_name=args.identity,
        max_train_steps=args.steps,
        learning_rate=args.lr
    )
