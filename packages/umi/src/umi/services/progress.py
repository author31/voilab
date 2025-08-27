import json
import time
from pathlib import Path
from typing import Any

from .base_service import BaseService


class ProgressService(BaseService):
    """Progress tracking service for pipeline stages."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        output_dir = self.config.get('output_dir', '.')
        self.output_dir = self._ensure_output_dir(output_dir)
        self.progress_file = self.output_dir / "progress.json"
        self._progress = self._load_progress()
    
    def _load_progress(self) -> dict:
        """Load existing progress or initialize new."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"stages": {}, "current_stage": None, "start_time": None, "end_time": None}
    
    def _save_progress(self):
        """Save progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self._progress, f, indent=2)
    
    def start_pipeline(self):
        """Mark pipeline start."""
        self._progress["start_time"] = time.time()
        self._save_progress()
    
    def start_stage(self, stage_name: str, total_items: int = 0):
        """Mark start of a processing stage."""
        self._progress["current_stage"] = stage_name
        self._progress["stages"][stage_name] = {
            "status": "running",
            "total": total_items,
            "completed": 0,
            "start_time": time.time(),
            "end_time": None
        }
        self._save_progress()
    
    def update_stage(self, stage_name: str, completed: int):
        """Update progress for a stage."""
        if stage_name in self._progress["stages"]:
            self._progress["stages"][stage_name]["completed"] = completed
            self._save_progress()
    
    def complete_stage(self, stage_name: str):
        """Mark stage as completed."""
        if stage_name in self._progress["stages"]:
            self._progress["stages"][stage_name]["status"] = "completed"
            self._progress["stages"][stage_name]["end_time"] = time.time()
            self._save_progress()
    
    def fail_stage(self, stage_name: str, error: str):
        """Mark stage as failed."""
        if stage_name in self._progress["stages"]:
            self._progress["stages"][stage_name]["status"] = "failed"
            self._progress["stages"][stage_name]["error"] = error
            self._progress["stages"][stage_name]["end_time"] = time.time()
            self._save_progress()
    
    def get_progress(self) -> dict:
        """Get current progress."""
        return self._progress.copy()
    
    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if stage is completed."""
        return (stage_name in self._progress["stages"] and 
                self._progress["stages"][stage_name]["status"] == "completed")
    
    def complete_pipeline(self):
        """Mark pipeline as completed."""
        self._progress["end_time"] = time.time()
        self._save_progress()
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute progress service (returns current progress)."""
        return self.get_progress()