from pathlib import Path
import json
import subprocess
from typing import List, Dict, Any

from .base_service import BaseService


class IMUExtractionService(BaseService):
    """Service for extracting IMU data from GoPro videos."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.num_workers = self.config.get('num_workers') or self._get_num_workers()
        self.stream_types = self.config.get('stream_types', ['ACCL', 'GYRO', 'GPS5', 'CORI', 'IORI'])
    
    def execute(self, input_dir: str, output_dir: str) -> dict:
        """
        Extract IMU data from videos in input directory.
        
        Args:
            input_dir: Directory containing organized demo videos
            output_dir: Directory for extracted IMU data
            
        Returns:
            dict: Extraction results with paths to IMU files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path = self._ensure_output_dir(output_dir)
        
        results = {"extracted": [], "failed": []}
        
        # Find all demo directories
        demo_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        
        for demo_dir in demo_dirs:
            # Find video files
            video_files = list(demo_dir.glob('*.MP4')) + list(demo_dir.glob('*.mp4'))
            
            for video_file in video_files:
                try:
                    imu_file = self._extract_imu_from_video(video_file, output_path)
                    if imu_file:
                        results["extracted"].append({
                            "video": str(video_file),
                            "imu_file": str(imu_file),
                            "demo": demo_dir.name
                        })
                except Exception as e:
                    results["failed"].append({
                        "video": str(video_file),
                        "error": str(e)
                    })
        
        return results
    
    def _extract_imu_from_video(self, video_file: Path, output_dir: Path) -> Path:
        """Extract IMU data from a single video file."""
        # Use py_gpmf_parser or ffmpeg to extract IMU data
        imu_file = output_dir / f"{video_file.stem}_imu.json"
        
        # This would use the actual IMU extraction logic
        # For now, create a placeholder
        imu_data = {
            "video_file": str(video_file),
            "streams": self.stream_types,
            "data": {}  # Actual IMU data would go here
        }
        
        with open(imu_file, 'w') as f:
            json.dump(imu_data, f, indent=2)
        
        return imu_file
    
    def validate_output(self, output_dir: str) -> bool:
        """Validate IMU extraction results."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return False
        
        imu_files = list(output_path.glob('*_imu.json'))
        return len(imu_files) > 0