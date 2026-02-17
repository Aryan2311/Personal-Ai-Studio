"""
Modern Pipeline Manager for Stable Diffusion + AnimateDiff
Supports image generation, video generation, and dynamic LoRA attachment
"""

import torch
from diffusers import (
    StableDiffusionPipeline,
    AnimateDiffPipeline,
)
from diffusers.models import MotionAdapter
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Manages Stable Diffusion and AnimateDiff pipelines with LoRA support.
    
    Features:
    - Lazy loading of pipelines
    - Dynamic LoRA attachment
    - Memory efficient attention (xformers)
    - A10G optimized (float16)
    """
    
    def __init__(self, base_model_path: str, motion_path: str):
        """
        Initialize the pipeline manager.
        
        Args:
            base_model_path: Path to Stable Diffusion 1.5 base model
            motion_path: Path to AnimateDiff motion adapter
        """
        self.base_model_path = base_model_path
        self.motion_path = motion_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.image_pipe = None
        self.video_pipe = None
        
        logger.info(f"PipelineManager initialized | device={self.device} | base={base_model_path} | motion={motion_path}")
    
    def load_image_pipeline(self):
        """
        Load or return cached Stable Diffusion pipeline for image generation.
        
        Returns:
            StableDiffusionPipeline instance
        """
        if self.image_pipe is None:
            logger.info(f"Loading image pipeline from {self.base_model_path}")
            self.image_pipe = StableDiffusionPipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Enable memory efficient attention
            try:
                self.image_pipe.enable_xformers_memory_efficient_attention()
                logger.info("✅ xformers memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable xformers: {e}")
            
            logger.info("✅ Image pipeline loaded")
        
        return self.image_pipe
    
    def load_video_pipeline(self):
        """
        Load or return cached AnimateDiff pipeline for video generation.
        
        Returns:
            AnimateDiffPipeline instance
        """
        if self.video_pipe is None:
            logger.info(f"Loading video pipeline | base={self.base_model_path} | motion={self.motion_path}")
            
            # Load motion adapter
            adapter = MotionAdapter.from_pretrained(self.motion_path)
            
            # Create AnimateDiff pipeline
            self.video_pipe = AnimateDiffPipeline.from_pretrained(
                self.base_model_path,
                motion_adapter=adapter,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Enable memory efficient attention
            try:
                self.video_pipe.enable_xformers_memory_efficient_attention()
                logger.info("✅ xformers memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable xformers: {e}")
            
            logger.info("✅ Video pipeline loaded")
        
        return self.video_pipe
    
    def attach_lora(self, pipe, lora_path: str, weight: float = 1.0):
        """
        Attach a LoRA adapter to a pipeline.
        
        Args:
            pipe: The pipeline to attach LoRA to
            lora_path: Path to LoRA weights file or directory
            weight: LoRA weight/scale (default: 1.0)
        
        Returns:
            Pipeline with LoRA attached
        """
        logger.info(f"Attaching LoRA | path={lora_path} | weight={weight}")
        
        try:
            pipe.load_lora_weights(lora_path, weight=weight)
            logger.info("✅ LoRA attached successfully")
        except Exception as e:
            logger.error(f"❌ Failed to attach LoRA: {e}")
            raise
        
        return pipe
    
    def clear_lora(self, pipe):
        """
        Clear LoRA weights from a pipeline (if supported).
        
        Args:
            pipe: The pipeline to clear LoRA from
        
        Returns:
            Pipeline with LoRA cleared
        """
        # Note: Diffusers doesn't have a direct "unload" method
        # You may need to reload the pipeline or use fuse/unfuse
        logger.info("Clearing LoRA weights")
        # For now, this is a placeholder - you may need to reload the pipeline
        return pipe

