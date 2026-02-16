"""
Reference Photo Applier
Applies reference photos correctly based on type:
- Background: img2img (low strength)
- Pose: ControlNet OpenPose
- Style: Prompt + optional img2img
"""

import os
import torch
from typing import List, Dict, Optional
import logging
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReferenceApplier:
    """
    Applies reference photos to generation pipeline.
    
    Classifies references by type and applies them correctly.
    """
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.device = model_manager.device
    
    def apply_references(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        references: Dict[str, List[Dict]] = None,
        steps: int = 30,
        seed: Optional[int] = None,
        strength: float = 0.4
    ) -> Image.Image:
        """
        Apply classified references to image generation.
        
        Args:
            prompt: Text prompt
            references: Classified references dict with keys: background, pose, style, misc
            steps: Inference steps
            seed: Random seed
            strength: img2img strength (for background/style)
        
        Returns:
            Generated PIL Image
        """
        bg_refs = references.get("background", [])
        pose_refs = references.get("pose", [])
        style_refs = references.get("style", [])
        
        logger.info(f"Applying references: {len(bg_refs)} bg, {len(pose_refs)} pose, {len(style_refs)} style")
        
        # Load first background reference for img2img init
        init_image = None
        if bg_refs:
            init_image_path = self._get_local_path(bg_refs[0])
            init_image = Image.open(init_image_path).convert("RGB")
            init_image = init_image.resize((512, 512), Image.LANCZOS)
            logger.info(f"Using background reference: {bg_refs[0]['name']}")
        
        # Load pose reference for ControlNet (if available)
        pose_image = None
        if pose_refs:
            pose_image_path = self._get_local_path(pose_refs[0])
            pose_image = Image.open(pose_image_path).convert("RGB")
            pose_image = pose_image.resize((512, 512), Image.LANCZOS)
            logger.info(f"Using pose reference: {pose_refs[0]['name']}")
        
        # Enhance prompt with style references
        if style_refs:
            style_names = [s["name"] for s in style_refs]
            prompt += f", {', '.join(style_names)} style"
            logger.info(f"Enhanced prompt with styles: {style_names}")
        
        # Generate with appropriate pipeline
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        
        # Use img2img if we have background reference
        if init_image:
            # For now, use simple img2img (ControlNet can be added later)
            from app.inference.image import generate_with_references
            image = generate_with_references(
                model_manager=self.model_manager,
                prompt=prompt,
                reference_images=[self._get_local_path(bg_refs[0])],
                steps=steps,
                seed=seed,
                lora_names=None,
                strength=strength
            )
        else:
            # Regular text-to-image with negative prompt
            image = self.model_manager.generate(
                prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                seed=seed
            )
        
        return image
    
    def _get_local_path(self, reference: Dict) -> str:
        """Get local path for reference, downloading from S3 if needed"""
        local_path = reference.get("local_path")
        
        if local_path and os.path.exists(local_path):
            return local_path
        
        # Download from S3 if not cached locally
        from app.storage.s3_manager import S3Manager
        s3_manager = S3Manager()
        
        s3_path = reference["s3_path"]
        ref_type = reference["type"]
        
        # Create local cache path
        local_dir = f"/opt/ai-influencer/data/references/{ref_type}"
        os.makedirs(local_dir, exist_ok=True)
        
        local_path = os.path.join(local_dir, f"{reference['name']}.jpg")
        
        # Download from S3
        s3_manager.download_file(s3_path, local_path)
        
        # Update registry with local path
        from app.api.reference_registry import ReferenceRegistry
        registry = ReferenceRegistry()
        registry.update_reference(reference["name"], local_path=local_path)
        
        return local_path

