from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from .base_service import BaseService


class BatchSLAMService(BaseService):
    """Service for batch processing SLAM on multiple videos."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.num_workers = self.config.get("num_workers") or self._get_num_workers(0.5)
        self.retry_attempts = self.config.get("retry_attempts", 3)

    def execute(self, input_dir: str, output_dir: str) -> dict:
        """
        Execute batch SLAM processing service.

        Args:
            input_dir: Directory containing videos and maps
            output_dir: Directory for batch SLAM outputs

        Returns:
            dict: Batch processing results
        """
        input_path = Path(input_dir)
        output_path = self._ensure_output_dir(output_dir)

        # Find all demo directories
        demo_dirs = [d for d in input_path.iterdir() if d.is_dir()]

        results = {"processed": [], "failed": []}

        # Process demos in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_demo = {
                executor.submit(self._process_demo, demo_dir, output_path): demo_dir
                for demo_dir in demo_dirs
            }

            for future in as_completed(future_to_demo):
                demo_dir = future_to_demo[future]
                try:
                    result = future.result()
                    if result:
                        results["processed"].append(result)
                except Exception as e:
                    results["failed"].append({"demo": demo_dir.name, "error": str(e)})

        return results

    def _process_demo(self, demo_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Process a single demo with retry logic."""
        demo_name = demo_dir.name
        batch_output_dir = output_dir / demo_name
        batch_output_dir.mkdir(exist_ok=True)

        # Find video and map files
        video_files = list(demo_dir.glob("*.MP4")) + list(demo_dir.glob("*.mp4"))
        map_files = list(demo_dir.glob("*.bin")) + list(demo_dir.glob("*.txt"))

        if not video_files:
            raise ValueError(f"No video files in {demo_dir}")

        # Retry logic for processing
        for attempt in range(self.retry_attempts):
            try:
                result = self._run_batch_slam(
                    video_files[0], map_files, batch_output_dir
                )
                if result:
                    return {
                        "demo": demo_name,
                        "output_dir": str(batch_output_dir),
                        "attempt": attempt + 1,
                    }
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise e

        return None

    def _run_batch_slam(
        self, video_file: Path, map_files: List[Path], output_dir: Path
    ) -> bool:
        """Run batch SLAM processing."""
        # This would implement actual batch SLAM processing
        # For now, create placeholder outputs

        trajectory_file = output_dir / "optimized_trajectory.txt"
        keyframe_file = output_dir / "keyframes.json"

        # Create placeholder files
        trajectory_file.touch()
        with open(keyframe_file, "w") as f:
            json.dump({"keyframes": []}, f, indent=2)

        return True

    def validate_output(self, output_dir: str) -> bool:
        """Validate batch SLAM results."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return False

        demo_dirs = [d for d in output_path.iterdir() if d.is_dir()]
        return len(demo_dirs) > 0
