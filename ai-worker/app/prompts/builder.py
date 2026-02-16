"""
Prompt Engineering Rules - Backend-Controlled
Users never control raw Stable Diffusion prompts.
Backend builds safe, structured prompts from user intent.
"""

import re
import logging
from typing import Dict, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Enforces prompt engineering rules.
    
    Structure:
    [IDENTITY] - Always injected, non-negotiable
    [CORE ACTION] - User-controlled but sanitized
    [SCENE] - User-controlled but normalized
    [STYLE] - Semi-locked, user chooses from preset
    [CAMERA] - Backend-owned presets
    [QUALITY SAFEGUARDS] - Always appended
    """
    
    # Scene vocabulary mapping (normalized)
    SCENE_VOCABULARY = {
        "cafe": "at a cafe",
        "café": "at a cafe",
        "coffee shop": "at a cafe",
        "balcony": "on a balcony",
        "apartment": "inside a modern apartment",
        "home": "inside a modern apartment",
        "park": "in a park",
        "beach": "at the beach",
        "street": "on a city street",
        "studio": "in a photography studio",
        "outdoor": "outdoors",
        "indoor": "indoors",
        "urban": "in an urban setting",
        "nature": "in nature"
    }
    
    # Style mapping (semi-locked)
    STYLE_MAP = {
        "casual": "casual lifestyle aesthetic",
        "luxury": "luxury fashion, clean aesthetic",
        "sporty": "athletic wear, active lifestyle",
        "editorial": "editorial fashion photography",
        "street": "street style fashion",
        "minimalist": "minimalist aesthetic, clean composition"
    }
    
    # Camera presets (backend-owned)
    CAMERA_PRESETS = {
        "portrait": "85mm lens, shallow depth of field, portrait framing",
        "medium": "medium shot, eye-level camera, natural perspective",
        "wide": "wide angle shot, environmental context",
        "close": "close-up shot, detailed framing",
        "candid": "candid photography style, natural moment",
        "cinematic": "cinematic framing, film-like composition"
    }
    
    # Forbidden SD keywords (removed from user input)
    FORBIDDEN_KEYWORDS = [
        "sks_", "ldog_", "lbf_",  # Identity tokens
        "8k", "ultra", "hyper", "masterpiece", "best quality",  # Quality spam
        "nsfw", "nude", "naked",  # Safety
        "deformed", "ugly", "bad", "worst"  # Negative words in positive prompt
    ]
    
    # Quality safeguards (always appended)
    QUALITY_SAFEGUARDS = [
        "natural skin texture",
        "consistent facial features",
        "no distortion",
        "high realism"
    ]
    
    # Negative prompt (always appended)
    NEGATIVE_PROMPT = [
        "extra limbs",
        "deformed face",
        "blurry",
        "cartoon",
        "CGI",
        "3d render",
        "bad anatomy",
        "distorted",
        "low quality",
        "jpeg artifacts"
    ]
    
    @classmethod
    def sanitize_action(cls, action: str, max_length: int = 100) -> str:
        """
        Sanitize user action input.
        
        Rules:
        - Max length enforced
        - Removes SD keywords
        - No anatomy words
        - Lowercase, trimmed
        """
        if not action:
            return ""
        
        # Trim and lowercase
        action = action.strip().lower()
        
        # Enforce max length
        if len(action) > max_length:
            action = action[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary
            logger.warning(f"Action truncated to {max_length} chars")
        
        # Remove forbidden keywords
        for keyword in cls.FORBIDDEN_KEYWORDS:
            action = action.replace(keyword, "")
        
        # Remove multiple spaces
        action = re.sub(r'\s+', ' ', action).strip()
        
        # Basic anatomy word filter (simple)
        anatomy_words = ["breast", "chest", "nude", "naked"]
        words = action.split()
        words = [w for w in words if w not in anatomy_words]
        action = " ".join(words)
        
        return action
    
    @classmethod
    def normalize_scene(cls, scene: str) -> str:
        """
        Normalize scene input using vocabulary.
        """
        if not scene:
            return ""
        
        scene_lower = scene.strip().lower()
        
        # Direct match
        if scene_lower in cls.SCENE_VOCABULARY:
            return cls.SCENE_VOCABULARY[scene_lower]
        
        # Partial match
        for key, value in cls.SCENE_VOCABULARY.items():
            if key in scene_lower:
                return value
        
        # Default: use as-is but lowercase
        return scene_lower
    
    @classmethod
    def map_style(cls, style: str) -> str:
        """
        Map user style choice to backend style string.
        """
        if not style:
            return ""
        
        style_lower = style.strip().lower()
        return cls.STYLE_MAP.get(style_lower, "")
    
    @classmethod
    def get_camera_preset(cls, camera: str) -> str:
        """
        Get camera preset from user choice.
        Defaults to "medium" if not found.
        """
        if not camera:
            return cls.CAMERA_PRESETS["medium"]
        
        camera_lower = camera.strip().lower()
        return cls.CAMERA_PRESETS.get(camera_lower, cls.CAMERA_PRESETS["medium"])
    
    @classmethod
    def build_prompt(
        cls,
        identity: str,
        action: str = "",
        scene: str = "",
        style: str = "",
        camera: str = "",
        custom: str = "",
        content_type: str = "photo",
        loras: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Build structured prompt following all rules.
        
        Returns:
            {
                "prompt": str,  # Positive prompt
                "negative_prompt": str  # Negative prompt
            }
        """
        parts = []
        
        # 1. IDENTITY (NON-NEGOTIABLE)
        identity_token = f"sks_{identity}"
        parts.append(f"{identity_token} person")
        
        # 2. LoRA tokens (if any)
        if loras:
            for lora_name in loras:
                if lora_name.startswith(f"{identity}_"):
                    entity = lora_name.replace(f"{identity}_", "")
                    lora_token = f"l{entity}_{identity}"
                    parts.append(f"with her {entity} {lora_token}")
        
        # 3. CORE ACTION (sanitized)
        if action:
            sanitized_action = cls.sanitize_action(action)
            if sanitized_action:
                parts.append(sanitized_action)
        
        # 4. SCENE (normalized)
        if scene:
            normalized_scene = cls.normalize_scene(scene)
            if normalized_scene:
                parts.append(normalized_scene)
        
        # 5. STYLE (mapped)
        if style:
            mapped_style = cls.map_style(style)
            if mapped_style:
                parts.append(mapped_style)
        
        # 6. CUSTOM (sanitized)
        if custom:
            sanitized_custom = cls.sanitize_action(custom, max_length=50)
            if sanitized_custom:
                parts.append(sanitized_custom)
        
        # 7. CAMERA (preset)
        camera_preset = cls.get_camera_preset(camera)
        parts.append(camera_preset)
        
        # 8. CONTENT-SPECIFIC QUALITY
        if content_type == "video":
            # Video-specific: fewer adjectives, motion keywords
            parts.append("smooth motion")
            parts.append("cinematic movement")
            parts.append("stable camera")
        else:
            # Photo-specific: natural lighting
            parts.append("natural daylight")
        
        # 9. QUALITY SAFEGUARDS (ALWAYS APPENDED)
        parts.extend(cls.QUALITY_SAFEGUARDS)
        
        # Build final prompt
        prompt = ", ".join(parts)
        
        # Build negative prompt
        negative_prompt = ", ".join(cls.NEGATIVE_PROMPT)
        
        logger.info(f"Built prompt for {identity} ({content_type}): {prompt[:100]}...")
        
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt
        }
    
    @classmethod
    def validate_prompt_data(cls, prompt_data: Dict) -> Dict:
        """
        Validate and normalize prompt data from frontend.
        
        Ensures:
        - Required fields present
        - Values within constraints
        - No forbidden content
        """
        validated = {
            "action": cls.sanitize_action(prompt_data.get("action", "")),
            "location": cls.normalize_scene(prompt_data.get("location", "")),
            "mood": cls.sanitize_action(prompt_data.get("mood", ""), max_length=50),
            "camera": prompt_data.get("camera", ""),
            "custom": cls.sanitize_action(prompt_data.get("custom", ""), max_length=50),
            "style": prompt_data.get("style", "")  # Will be mapped in build_prompt
        }
        
        return validated

