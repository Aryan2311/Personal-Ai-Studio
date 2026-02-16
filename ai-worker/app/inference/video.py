"""
Video Generation Module
Generates 2-3 second silent videos using:
- Existing DreamBooth identity
- Existing LoRAs
- Same EC2 / GPU
- No retraining required

Pipeline:
Text → Stable Diffusion (identity + lora) → Motion module → Frames → MP4 video
"""

import os
import torch
import uuid
from typing import Optional, List
import logging
from pathlib import Path
import imageio

from diffusers import AnimateDiffPipeline, MotionAdapter
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_video_pipeline(identity_pipe):
    """
    Load video pipeline by reusing DreamBooth model weights.
    
    Key idea:
    - Reuse the DreamBooth model weights (vae, text_encoder, tokenizer, unet, scheduler)
    - Add a motion adapter on top
    - No retraining required
    
    Args:
        identity_pipe: The Stable Diffusion pipeline with identity model loaded
    
    Returns:
        AnimateDiffPipeline ready for video generation
    """
    motion_adapter_path = "/opt/ai-influencer/models/motion/animatediff"
    
    logger.info("Loading motion adapter...")
    motion_adapter = MotionAdapter.from_pretrained(
        motion_adapter_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    logger.info("Creating AnimateDiff pipeline from identity model...")
    video_pipe = AnimateDiffPipeline(
        vae=identity_pipe.vae,
        text_encoder=identity_pipe.text_encoder,
        tokenizer=identity_pipe.tokenizer,
        unet=identity_pipe.unet,
        motion_adapter=motion_adapter,
        scheduler=identity_pipe.scheduler,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_pipe = video_pipe.to(device)
    
    # Enable memory efficient attention if available
    if hasattr(video_pipe, "enable_xformers_memory_efficient_attention"):
        try:
            video_pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    video_pipe.enable_model_cpu_offload()
    
    logger.info("Video pipeline loaded successfully")
    return video_pipe


def generate_video_frames(
    video_pipe: AnimateDiffPipeline,
    prompt: str,
    negative_prompt: Optional[str] = None,
    num_frames: int = 16,
    steps: int = 25,
    guidance_scale: float = 8.0,
    seed: Optional[int] = None
) -> List[Image.Image]:
    """
    Generate video frames using AnimateDiff.
    
    Recommended defaults (A10G safe):
    - num_frames: 16-24 (16 = ~2 seconds at 8fps)
    - steps: 20-30
    - guidance_scale: 7-9
    
    Args:
        video_pipe: AnimateDiffPipeline instance
        prompt: Text prompt
        num_frames: Number of frames to generate (16 = ~2 seconds)
        steps: Number of inference steps
        guidance_scale: Guidance scale
        seed: Random seed for reproducibility
    
    Returns:
        List of PIL Images (frames)
    """
    logger.info(f"Generating {num_frames} frames with {steps} steps...")
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=video_pipe.device)
        generator.manual_seed(seed)
    
    with torch.inference_mode():
        output = video_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
    
    frames = output.frames[0]  # List of PIL images
    logger.info(f"Generated {len(frames)} frames")
    
    return frames


def frames_to_video(frames: List[Image.Image], output_path: str, fps: int = 8):
    """
    Convert frames to MP4 video file.
    
    Args:
        frames: List of PIL Images
        output_path: Output video file path
        fps: Frames per second (8 is good for 2-3 second clips)
    """
    logger.info(f"Converting {len(frames)} frames to video: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    imageio.mimsave(
        output_path,
        frames,
        fps=fps,
        codec="libx264"
    )
    
    logger.info(f"Video saved to {output_path}")


def download_motion_model():
    """
    Download motion model weights (run once during setup).
    
    This does not affect DreamBooth.
    """
    motion_model_path = "/opt/ai-influencer/models/motion/animatediff"
    
    if os.path.exists(motion_model_path):
        logger.info(f"Motion model already exists at {motion_model_path}")
        return
    
    logger.info("Downloading AnimateDiff motion adapter...")
    os.makedirs(motion_model_path, exist_ok=True)
    
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2"
    )
    adapter.save_pretrained(motion_model_path)
    
    logger.info(f"Motion model saved to {motion_model_path}")


def generate_video(
    model_manager,
    identity: str,
    script: str,
    negative_prompt: Optional[str] = None,
    num_frames: int = 16,
    steps: int = 25,
    guidance_scale: float = 8.0,
    seed: Optional[int] = None,
    lora_names: Optional[List[str]] = None,
    fps: int = 8,
    motion_type: str = "walk",
    reference_images: Optional[List[str]] = None
) -> str:
    """
    Generate a video for a specific identity.
    
    Full pipeline:
    1. Load identity model
    2. Apply LoRAs if specified
    3. Load video pipeline (reuses identity weights + motion adapter)
    4. Generate frames
    5. Convert to MP4
    
    Args:
        model_manager: ModelManager instance
        identity: Identity name
        script: User prompt
        num_frames: Number of frames (16 = ~2 seconds at 8fps)
        steps: Inference steps
        guidance_scale: Guidance scale
        seed: Random seed
        lora_names: Optional list of LoRAs to apply
        fps: Frames per second
    
    Returns:
        Path to generated video file
    """
    logger.info(f"Generating video for identity: {identity}")
    
    # Check GPU lock (inference should fail if training is running)
    from app.core.gpu_lock import is_locked, get_lock_info
    if is_locked():
        lock_info = get_lock_info()
        raise RuntimeError(
            f"Cannot generate video - GPU is busy. "
            f"Current owner: {lock_info.get('owner', 'unknown') if lock_info else 'unknown'}"
        )
    
    # Load identity model
    model_manager.load_identity(identity)
    
    # Apply LoRAs if specified
    if lora_names:
        model_manager.apply_loras(lora_names)
    
    try:
        # Build prompt with token
        identity_token = f"sks_{identity}"
        prompt_parts = [f"{identity_token} person"]
        
        # Add LoRA tokens to prompt if LoRAs are used
        if lora_names:
            for lora_name in lora_names:
                if lora_name.startswith(f"{identity}_"):
                    entity = lora_name.replace(f"{identity}_", "")
                    lora_token = f"l{entity}_{identity}"
                    prompt_parts.append(f"with her {entity} {lora_token}")
        
        # Add user script and cinematic motion
        prompt_parts.append(script)
        prompt_parts.append("cinematic motion, smooth animation")
        
        prompt = ", ".join(prompt_parts)
        
        # If reference images provided, use them to condition first frame
        # Then AnimateDiff propagates motion from that stable start
        if reference_images:
            logger.info(f"Using {len(reference_images)} photo references for video generation")
            # For video, we condition the first frame with photo references
            # This improves identity stability and background coherence
            from app.inference.reference_applier import ReferenceApplier
            ref_applier = ReferenceApplier(model_manager)
            
            # Classify references (for video, we use all types)
            from app.api.reference_registry import ReferenceRegistry
            registry = ReferenceRegistry()
            ref_names = [os.path.basename(ref).replace('.jpg', '') for ref in reference_images]
            classified_refs = registry.classify_references(ref_names)
            
            # Generate first frame with references
            first_frame = ref_applier.apply_references(
                prompt=prompt,
                negative_prompt=negative_prompt,
                references=classified_refs,
                steps=steps,
                seed=seed,
                strength=0.5  # Moderate strength for video
            )
            
            # Load video pipeline
            video_pipe = load_video_pipeline(model_manager.pipe)
            
            # Generate remaining frames using AnimateDiff
            # First frame is already conditioned, AnimateDiff adds motion
            frames = [first_frame]  # Start with conditioned frame
            remaining_frames = generate_video_frames(
                video_pipe,
                prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames - 1,  # One frame already generated
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
            frames.extend(remaining_frames)
        else:
            # No references - regular video generation
            video_pipe = load_video_pipeline(model_manager.pipe)
            frames = generate_video_frames(
                video_pipe,
                prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
        
        # Save video
        output_dir = "/opt/ai-influencer/outputs/videos"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{identity}_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(output_dir, filename)
        
        frames_to_video(frames, output_path, fps=fps)
        
        logger.info(f"Video generated: {output_path}")
        return output_path
    
    finally:
        # Clear LoRAs
        if lora_names:
            model_manager.lora_manager.clear()
