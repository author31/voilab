from pathlib import Path
import subprocess
import json
import time
from typing import Dict, Any

from .base_service import BaseService


class SLAMMappingService(BaseService):
    """Service for creating SLAM maps using ORB-SLAM3."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.docker_image = self.config.get('docker_image', 'chicheng/orb_slam3:latest')
        self.timeout_multiple = self.config.get('timeout_multiple', 16)
        self.max_lost_frames = self.config.get('max_lost_frames', 60)
    
    def execute(self, input_dir: str, output_dir: str) -> dict:
        """
        Create SLAM map from videos in input directory.
        
        Args:
            input_dir: Directory containing videos and IMU data
            output_dir: Directory for SLAM map outputs
            
        Returns:
            dict: Mapping results with paths to map files
        """
        input_path = Path(input_dir)
        output_path = self._ensure_output_dir(output_dir)
        
        results = {"maps": [], "failed": []}
        
        # Find all demo directories
        demo_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        
        for demo_dir in demo_dirs:
            try:
                map_result = self._create_map_for_demo(demo_dir, output_path)
                if map_result:
                    results["maps"].append(map_result)
            except Exception as e:
                results["failed"].append({
                    "demo": demo_dir.name,
                    "error": str(e)
                })
        
        return results
    
    def _create_map_for_demo(self, demo_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Create SLAM map for a single demo."""
        demo_name = demo_dir.name
        map_dir = output_dir / demo_name
        map_dir.mkdir(exist_ok=True)
        
        # Find video files
        video_files = list(demo_dir.glob('*.MP4')) + list(demo_dir.glob('*.mp4'))
        if not video_files:
            raise ValueError(f"No video files found in {demo_dir}")
        
        # This would use actual ORB-SLAM3 Docker container
        # For now, create placeholder map files
        map_file = map_dir / "map.bin"
        trajectory_file = map_dir / "trajectory.txt"
        
        # Create placeholder files
        map_file.touch()
        with open(trajectory_file, 'w') as f:
            f.write("# SLAM trajectory placeholder\n")
        
        return {
            "demo": demo_name,
            "map_file": str(map_file),
            "trajectory_file": str(trajectory_file),
            "video_files": [str(v) for v in video_files]
        }
    
    def _run_docker_slam(self, video_file: Path, output_dir: Path) -> bool:
        """Run SLAM mapping using Docker container."""
        # This would implement actual Docker-based SLAM
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{video_file.parent}:/input",
            "-v", f"{output_dir}:/output",
            self.docker_image,
            "./Examples/Monocular/mono_tum",
            "Vocabulary/ORBvoc.txt",
            "Examples/Monocular/TUM1.yaml",
            f"/input/{video_file.name}"
        ]
        
        try:
            result = subprocess.run(cmd, timeout=self.timeout_multiple * 60, capture_output=True, text=True)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def validate_output(self, output_dir: str) -> bool:
        """Validate SLAM mapping results."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return False
        
        demo_dirs = [d for d in output_path.iterdir() if d.is_dir()]
        for demo_dir in demo_dirs:
            map_files = list(demo_dir.glob('*.bin')) + list(demo_dir.glob('*.txt'))
            if not map_files:
                return False
        
        return len(demo_dirs) > 0