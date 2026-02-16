"""
LoRA Manager - Handles LoRA loading and composition
Enables associated entities (boyfriend, dog, house, etc.) without identity drift

Stackable magic: You can stack dog + boyfriend + house + clothes safely
"""

import os
import torch
from typing import List, Optional, Dict
import logging

from diffusers import StableDiffusionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRAManager:
    """
    Manages LoRA adapters for associated entities.
    
    Each associated thing (boyfriend, dog, house, etc.) becomes its own LoRA:
    - Keeps primary identity (DreamBooth) stable
    - Enables expandable universe
    - Fast and modular training
    
    Example tokens:
    - Primary: sks_leela (DreamBooth)
    - Boyfriend: lbf_leela (LoRA)
    - Dog: ldog_leela (LoRA)
    - House: lhouse_leela (LoRA)
    
    Generation flow:
    1. model_manager.load_identity("leela")
    2. lora_manager.load("models/loras/leela_dog")
    3. image = model_manager.generate(prompt)
    4. lora_manager.clear()
    """
    
    def __init__(self, pipe: Optional[StableDiffusionPipeline] = None):
        self.pipe = pipe
        self.loaded: List[str] = []
        self.loras_path = "/opt/ai-influencer/models/loras"
        os.makedirs(self.loras_path, exist_ok=True)
    
    def set_pipeline(self, pipe: StableDiffusionPipeline):
        """Set the pipeline to manage LoRAs for"""
        self.pipe = pipe
        self.loaded = []  # Clear loaded LoRAs when pipeline changes
    
    def load(self, lora_name: str, scale: float = 1.0) -> StableDiffusionPipeline:
        """
        Load a LoRA adapter.
        
        Args:
            lora_name: Name of the LoRA (e.g., "leela_dog")
            scale: LoRA strength (0.0 to 1.0)
        
        Returns:
            Pipeline with LoRA loaded
        """
        if not self.pipe:
            raise ValueError("Pipeline not set. Call set_pipeline() first.")
        
        lora_path = os.path.join(self.loras_path, f"{lora_name}.safetensors")
        lora_dir = os.path.join(self.loras_path, lora_name)
        
        # Check if LoRA exists
        if not os.path.exists(lora_path) and not os.path.exists(lora_dir):
            raise ValueError(f"LoRA '{lora_name}' not found at {lora_path} or {lora_dir}")
        
        # Use directory if it exists, otherwise use safetensors file
        actual_path = lora_dir if os.path.exists(lora_dir) else lora_path
        
        logger.info(f"Loading LoRA: {lora_name} (scale: {scale})")
        
        try:
            # Load LoRA weights using diffusers
            self.pipe.load_lora_weights(actual_path)
            
            # Fuse LoRA for better performance (optional, can also use dynamic loading)
            if hasattr(self.pipe, "fuse_lora"):
                self.pipe.fuse_lora(lora_scale=scale)
            
            self.loaded.append(lora_name)
            logger.info(f"LoRA '{lora_name}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LoRA '{lora_name}': {e}")
            raise
        
        return self.pipe
    
    def clear(self):
        """
        Clear all loaded LoRAs.
        
        Unfuses LoRAs and clears the loaded list.
        """
        if not self.pipe:
            return
        
        if self.loaded:
            try:
                if hasattr(self.pipe, "unfuse_lora"):
                    self.pipe.unfuse_lora()
                elif hasattr(self.pipe, "unload_lora_weights"):
                    self.pipe.unload_lora_weights()
                
                logger.info(f"Cleared {len(self.loaded)} LoRAs: {', '.join(self.loaded)}")
                self.loaded = []
            except Exception as e:
                logger.warning(f"Failed to clear LoRAs: {e}")
    
    def load_multiple(self, lora_names: List[str], scales: Optional[List[float]] = None) -> StableDiffusionPipeline:
        """
        Load multiple LoRAs (stackable magic).
        
        Args:
            lora_names: List of LoRA names (e.g., ["leela_dog", "leela_house"])
            scales: Optional list of scales (default: 1.0 for all)
        
        Returns:
            Pipeline with all LoRAs loaded
        """
        if scales is None:
            scales = [1.0] * len(lora_names)
        
        for lora_name, scale in zip(lora_names, scales):
            self.load(lora_name, scale)
        
        logger.info(f"Loaded {len(lora_names)} LoRAs: {', '.join(lora_names)}")
        return self.pipe
    
    def list_loras(self, identity: Optional[str] = None) -> List[str]:
        """
        List available LoRAs.
        
        Args:
            identity: Optional filter by identity (e.g., "leela")
        
        Returns:
            List of LoRA names
        """
        if not os.path.exists(self.loras_path):
            return []
        
        loras = []
        for item in os.listdir(self.loras_path):
            item_path = os.path.join(self.loras_path, item)
            
            # Check if it's a LoRA file or directory
            if item.endswith(".safetensors") or os.path.isdir(item_path):
                lora_name = item.replace(".safetensors", "")
                
                # Filter by identity if specified
                if identity and not lora_name.startswith(f"{identity}_"):
                    continue
                
                loras.append(lora_name)
        
        return sorted(loras)
    
    def lora_exists(self, lora_name: str) -> bool:
        """Check if a LoRA exists"""
        lora_path = os.path.join(self.loras_path, f"{lora_name}.safetensors")
        lora_dir = os.path.join(self.loras_path, lora_name)
        return os.path.exists(lora_path) or os.path.exists(lora_dir)
