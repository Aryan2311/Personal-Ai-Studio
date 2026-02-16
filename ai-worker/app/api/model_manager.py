"""
Model Manager - Hot-Swap Identity Models
CRITICAL PIECE: Enables multiple influencers on one GPU
Now supports LoRA composition for associated entities
"""

import os
import torch
from typing import Optional, List
import logging
import gc

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model loading and hot-swapping.
    
    This is what enables:
    - Multiple influencers
    - One GPU
    - No restarts
    - LoRA composition (associated entities)
    """
    
    def __init__(self, base_model_path: str = "/opt/ai-influencer/models/base/sd15"):
        self.base_model_path = base_model_path
        self.current_identity = None
        self.current_loras: List[str] = []
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # LoRA manager
        from app.api.lora_manager import LoRAManager
        self.lora_manager = LoRAManager()
    
    def load_identity(self, identity: str):
        """
        Load identity model, hot-swapping if needed.
        
        If same identity already loaded, returns immediately.
        If different identity, unloads current and loads new.
        """
        if self.current_identity == identity:
            logger.info(f"Identity '{identity}' already loaded")
            return
        
        # Unload current model if different identity
        if self.pipe:
            logger.info(f"Unloading current model (identity: {self.current_identity})")
            del self.pipe
            self.pipe = None
            self.current_identity = None
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load new identity model
        model_path = f"/opt/ai-influencer/models/identities/{identity}"
        
        if not os.path.exists(model_path):
            raise ValueError(
                f"Identity '{identity}' not found at {model_path}. "
                f"Train it first using /train/identity endpoint."
            )
        
        logger.info(f"Loading identity model: {identity}")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Use DPMSolver for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Enable memory efficient attention if available
        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
        
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()
        
        # Set pipeline in LoRA manager
        self.lora_manager.set_pipeline(self.pipe)
        self.lora_manager.clear()  # Clear any previous LoRAs
        
        self.current_identity = identity
        self.current_loras = []  # Reset LoRAs when identity changes
        logger.info(f"Identity '{identity}' loaded successfully")
    
    def apply_loras(self, lora_names: List[str], strengths: Optional[List[float]] = None):
        """
        Apply LoRAs to the current pipeline.
        
        Args:
            lora_names: List of LoRA names (e.g., ["leela_dog", "leela_house"])
            strengths: Optional list of strengths (default: 1.0 for all)
        """
        if not self.pipe:
            raise ValueError("No model loaded. Call load_identity() first.")
        
        if not lora_names:
            return
        
        # Clear previous LoRAs if different
        if set(self.current_loras) != set(lora_names):
            self.lora_manager.clear()
        
        # Apply new LoRAs
        self.lora_manager.load_multiple(lora_names, strengths)
        self.current_loras = lora_names
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        lora_names: Optional[List[str]] = None,
        lora_strengths: Optional[List[float]] = None
    ):
        """
        Generate image from prompt.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt (optional)
            steps: Number of inference steps
            guidance_scale: Guidance scale
            seed: Random seed for reproducibility
            lora_names: Optional list of LoRAs to apply
            lora_strengths: Optional list of LoRA strengths
        
        Returns:
            PIL Image
        """
        if not self.pipe:
            raise ValueError("No model loaded. Call load_identity() first.")
        
        # Apply LoRAs if specified
        if lora_names:
            self.apply_loras(lora_names, lora_strengths)
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        
        with torch.inference_mode():
            image = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        return image
    
    def check_gpu(self) -> bool:
        """Check if GPU is available"""
        return torch.cuda.is_available()
    
    def get_gpu_name(self) -> str:
        """Get GPU name"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"
