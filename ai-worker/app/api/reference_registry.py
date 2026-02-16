"""
Reference Registry
Tracks all reference photos as first-class assets
References are reusable, shared between photo/video, and stored in S3
"""

import json
import os
from typing import Dict, Optional, List
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReferenceRegistry:
    """
    Manages reference photo registry - first-class assets.
    
    Each reference has:
    - name: Unique identifier (e.g., "cafe_01")
    - type: "background", "pose", "style", "misc"
    - s3_path: Path in S3 bucket
    - local_path: Cached local path (optional)
    - created_at: Timestamp
    - metadata: Additional info
    """
    
    def __init__(self, registry_path: str = "/opt/ai-influencer/registry/references.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load registry from JSON file"""
        if not self.registry_path.exists():
            logger.info("Creating new reference registry")
            return {}
        
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            logger.info(f"Loaded registry with {len(registry)} references")
            return registry
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return {}
    
    def _save_registry(self):
        """Save registry to JSON file"""
        try:
            with open(self.registry_path, "w") as f:
                json.dump(self.registry, f, indent=2)
            logger.debug("Registry saved")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise
    
    def register_reference(
        self,
        name: str,
        ref_type: str,
        s3_path: str,
        local_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Register a new reference photo.
        
        Args:
            name: Unique identifier (e.g., "cafe_01")
            ref_type: "background", "pose", "style", "misc"
            s3_path: Path in S3 bucket
            local_path: Optional cached local path
            metadata: Additional info
        """
        if name in self.registry:
            logger.warning(f"Reference '{name}' already exists, updating...")
        
        reference_record = {
            "name": name,
            "type": ref_type,
            "s3_path": s3_path,
            "local_path": local_path,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.registry[name] = reference_record
        self._save_registry()
        
        logger.info(f"Registered reference: {name} (type: {ref_type})")
        return reference_record
    
    def get_reference(self, name: str) -> Optional[Dict]:
        """Get reference record by name"""
        return self.registry.get(name)
    
    def list_references(self, ref_type: Optional[str] = None) -> List[Dict]:
        """List all references, optionally filtered by type"""
        references = list(self.registry.values())
        
        if ref_type:
            references = [r for r in references if r["type"] == ref_type]
        
        return sorted(references, key=lambda x: x["created_at"], reverse=True)
    
    def update_reference(
        self,
        name: str,
        ref_type: Optional[str] = None,
        s3_path: Optional[str] = None,
        local_path: Optional[str] = None,
        **kwargs
    ):
        """Update reference record"""
        if name not in self.registry:
            raise ValueError(f"Reference '{name}' not found in registry")
        
        if ref_type is not None:
            self.registry[name]["type"] = ref_type
        if s3_path is not None:
            self.registry[name]["s3_path"] = s3_path
        if local_path is not None:
            self.registry[name]["local_path"] = local_path
        
        # Update any other fields
        for key, value in kwargs.items():
            self.registry[name][key] = value
        
        self.registry[name]["updated_at"] = datetime.now().isoformat()
        self._save_registry()
        
        logger.info(f"Updated reference: {name}")
    
    def delete_reference(self, name: str):
        """Delete reference from registry (does not delete S3 file)"""
        if name not in self.registry:
            raise ValueError(f"Reference '{name}' not found")
        
        del self.registry[name]
        self._save_registry()
        logger.info(f"Deleted reference from registry: {name}")
    
    def classify_references(self, reference_names: List[str]) -> Dict[str, List[Dict]]:
        """
        Classify references by type.
        
        Returns:
            {
                "background": [ref1, ref2],
                "pose": [ref3],
                "style": [ref4],
                "misc": []
            }
        """
        classified = {
            "background": [],
            "pose": [],
            "style": [],
            "misc": []
        }
        
        for ref_name in reference_names:
            ref = self.get_reference(ref_name)
            if ref:
                ref_type = ref.get("type", "misc")
                if ref_type in classified:
                    classified[ref_type].append(ref)
                else:
                    classified["misc"].append(ref)
        
        return classified

