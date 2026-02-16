"""
Unified Content Generation
Handles both photo and video generation with reference photos
"""

import os
import uuid
import logging
from typing import List, Dict, Optional
from fastapi import HTTPException

from app.api.model_manager import ModelManager
from app.api.reference_registry import ReferenceRegistry
from app.inference.reference_applier import ReferenceApplier
from app.inference.video import generate_video as generate_video_impl
from app.prompts.builder import PromptBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentGenerator:
    """
    Unified content generator for photos and videos.
    
    Handles:
    - Identity loading
    - LoRA application
    - Reference photo classification and application
    - Photo/video generation
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.reference_registry = ReferenceRegistry()
        self.reference_applier = ReferenceApplier(model_manager)
    
    def generate_content(
        self,
        identity: str,
        content_type: str,
        prompt_data: Dict,
        loras: Optional[List[str]] = None,
        references: Optional[List[str]] = None,
        steps: int = 30,
        seed: Optional[int] = None,
        num_frames: int = 16,
        motion_type: str = "walk"
    ) -> Dict:
        """
        Generate photo or video content.
        
        Args:
            identity: Identity name
            content_type: "photo" or "video"
            prompt_data: Structured prompt dict (action, location, mood, camera, custom)
            loras: List of LoRA names
            references: List of reference names
            steps: Inference steps
            seed: Random seed
            num_frames: Number of frames (for video)
            motion_type: Motion type (for video)
        
        Returns:
            Dict with success, path, prompt_used, etc.
        """
        # Check GPU lock
        from app.core.gpu_lock import is_locked, get_lock_info
        if is_locked():
            lock_info = get_lock_info()
            raise HTTPException(
                status_code=409,
                detail=f"Cannot generate content - GPU is busy. "
                f"Current owner: {lock_info.get('owner', 'unknown') if lock_info else 'unknown'}"
            )
        
        # Load identity
        self.model_manager.load_identity(identity)
        
        # Apply LoRAs if specified
        if loras:
            self.model_manager.apply_loras(loras)
        
        # Build prompt (returns dict with prompt and negative_prompt)
        prompt_result = self._build_prompt(identity, prompt_data, loras)
        final_prompt = prompt_result["prompt"]
        negative_prompt = prompt_result["negative_prompt"]
        
        # Classify references
        classified_refs = {}
        if references:
            classified_refs = self.reference_registry.classify_references(references)
        
        try:
            if content_type == "photo":
                return self._generate_photo(
                    prompt=final_prompt,
                    negative_prompt=negative_prompt,
                    classified_refs=classified_refs,
                    steps=steps,
                    seed=seed,
                    identity=identity
                )
            elif content_type == "video":
                return self._generate_video(
                    prompt=final_prompt,
                    negative_prompt=negative_prompt,
                    classified_refs=classified_refs,
                    steps=steps,
                    seed=seed,
                    num_frames=num_frames,
                    motion_type=motion_type,
                    identity=identity,
                    loras=loras
                )
            else:
                raise HTTPException(status_code=400, detail=f"Invalid content_type: {content_type}")
        
        finally:
            # Clear LoRAs
            if loras:
                self.model_manager.lora_manager.clear()
    
    def _build_prompt(self, identity: str, prompt_data: Dict, loras: Optional[List[str]]) -> Dict[str, str]:
        """
        Build final prompt using PromptBuilder (enforces all rules).
        Returns both prompt and negative_prompt.
        """
        # Validate prompt data
        validated_data = PromptBuilder.validate_prompt_data(prompt_data)
        
        # Build prompt using rules
        result = PromptBuilder.build_prompt(
            identity=identity,
            action=validated_data["action"],
            scene=validated_data["location"],
            style=validated_data.get("style", ""),
            camera=validated_data["camera"],
            custom=validated_data["custom"],
            content_type=prompt_data.get("content_type", "photo"),
            loras=loras
        )
        
        return result
    
    def _generate_photo(
        self,
        prompt: str,
        negative_prompt: str,
        classified_refs: Dict,
        steps: int,
        seed: Optional[int],
        identity: str
    ) -> Dict:
        """Generate photo with references"""
        # Apply references
        if any(classified_refs.values()):
            image = self.reference_applier.apply_references(
                prompt=prompt,
                negative_prompt=negative_prompt,
                references=classified_refs,
                steps=steps,
                seed=seed
            )
        else:
            # Regular generation with negative prompt
            image = self.model_manager.generate(
                prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                seed=seed
            )
        
        # Save image
        output_dir = "/opt/ai-influencer/outputs/images"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{identity}_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join(output_dir, filename)
        image.save(output_path)
        
        logger.info(f"Generated photo: {output_path}")
        
        return {
            "success": True,
            "content_type": "photo",
            "path": output_path,
            "prompt_used": prompt
        }
    
    def _generate_video(
        self,
        prompt: str,
        negative_prompt: str,
        classified_refs: Dict,
        steps: int,
        seed: Optional[int],
        num_frames: int,
        motion_type: str,
        identity: str,
        loras: Optional[List[str]]
    ) -> Dict:
        """Generate video with photo references"""
        # For video, we use photo references to condition the first frame
        # Then AnimateDiff propagates motion
        
        # Get reference paths for video pipeline
        reference_paths = []
        for ref_type, refs in classified_refs.items():
            for ref in refs:
                ref_path = self.reference_applier._get_local_path(ref)
                reference_paths.append(ref_path)
        
        # Generate video
        video_path = generate_video_impl(
            model_manager=self.model_manager,
            identity=identity,
            script=prompt,  # Full prompt
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=8.0,
            seed=seed,
            lora_names=loras if loras else None,
            fps=8,
            motion_type=motion_type,
            reference_images=reference_paths if reference_paths else None
        )
        
        logger.info(f"Generated video: {video_path}")
        
        return {
            "success": True,
            "content_type": "video",
            "path": video_path,
            "prompt_used": prompt,
            "num_frames": num_frames,
            "duration_seconds": num_frames / 8
        }

