"""
Image Generation with Reference Images
Supports img2img and ControlNet for reference image conditioning
"""

import os
import torch
from typing import List, Optional
import logging
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_with_references(
    model_manager,
    prompt: str,
    reference_images: List[str],
    steps: int = 30,
    seed: Optional[int] = None,
    lora_names: Optional[List[str]] = None,
    strength: float = 0.75
):
    """
    Generate image using reference images for conditioning.
    
    Uses img2img pipeline with the first reference image as init image.
    Additional reference images can be used for style/pose guidance.
    
    Args:
        model_manager: ModelManager instance
        prompt: Text prompt
        reference_images: List of paths to reference images
        steps: Inference steps
        seed: Random seed
        lora_names: Optional list of LoRAs
        strength: img2img strength (0.0 to 1.0, lower = more like reference)
    
    Returns:
        PIL Image
    """
    logger.info(f"Generating image with {len(reference_images)} reference images")
    
    # Load first reference image as init image
    init_image = Image.open(reference_images[0]).convert("RGB")
    init_image = init_image.resize((512, 512), Image.LANCZOS)
    
    # Apply LoRAs if specified
    if lora_names:
        model_manager.apply_loras(lora_names)
    
    try:
        # Use img2img pipeline
        generator = None
        if seed is not None:
            generator = torch.Generator(device=model_manager.device)
            generator.manual_seed(seed)
        
        # Convert PIL to tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Use diffusers img2img pipeline
        if hasattr(model_manager.pipe, 'image_to_image'):
            image = model_manager.pipe(
                prompt=prompt,
                image=init_image,
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=7.5,
                generator=generator
            ).images[0]
        else:
            # Fallback: regular generation if img2img not available
            logger.warning("img2img not available, using regular generation")
            image = model_manager.generate(
                prompt,
                steps=steps,
                seed=seed
            )
        
        return image
    
    finally:
        # Clear LoRAs
        if lora_names:
            model_manager.lora_manager.clear()
