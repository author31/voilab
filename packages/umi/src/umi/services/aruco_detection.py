from pathlib import Path
import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

from .base_service import BaseService


class ArucoDetectionService(BaseService):
    """Service for detecting ArUco markers in video frames."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.num_workers = self.config.get('num_workers') or self._get_num_workers()
        self.camera_intrinsics_path = self.config.get('camera_intrinsics_path')
        self.aruco_config_path = self.config.get('aruco_config_path')
        
        self.camera_matrix, self.dist_coeffs = self._load_camera_intrinsics()
        self.aruco_dict, self.aruco_params = self._load_aruco_config()
    
    def execute(self, input_dir: str, output_dir: str) -> dict:
        """
        Execute ArUco marker detection service.
        
        Args:
            input_dir: Directory containing videos
            output_dir: Directory for detection results
            
        Returns:
            dict: Detection results with marker positions
        """
        input_path = Path(input_dir)
        output_path = self._ensure_output_dir(output_dir)
        
        # Find all video files
        video_files = []
        for pattern in ['*.MP4', '*.mp4', '*.avi', '*.mov']:
            video_files.extend(input_path.rglob(pattern))
        
        results = {"detections": [], "failed": []}
        
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_video = {
                executor.submit(self._detect_in_video, video_file, output_path): video_file
                for video_file in video_files
            }
            
            for future in as_completed(future_to_video):
                video_file = future_to_video[future]
                try:
                    result = future.result()
                    if result:
                        results["detections"].append(result)
                except Exception as e:
                    results["failed"].append({
                        "video": str(video_file),
                        "error": str(e)
                    })
        
        return results
    
    def _load_camera_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load camera intrinsics from file."""
        if self.camera_intrinsics_path and Path(self.camera_intrinsics_path).exists():
            with open(self.camera_intrinsics_path, 'r') as f:
                calib = json.load(f)
            
            camera_matrix = np.array(calib.get('camera_matrix', [[1,0,0],[0,1,0],[0,0,1]]))
            dist_coeffs = np.array(calib.get('dist_coeffs', [0,0,0,0,0]))
            return camera_matrix, dist_coeffs
        
        # Default values
        return np.eye(3), np.zeros(5)
    
    def _load_aruco_config(self) -> Tuple[Any, Any]:
        """Load ArUco configuration."""
        if self.aruco_config_path and Path(self.aruco_config_path).exists():
            with open(self.aruco_config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Default ArUco dictionary
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Apply custom parameters if provided
        if 'parameters' in config:
            for key, value in config['parameters'].items():
                if hasattr(aruco_params, key):
                    setattr(aruco_params, key, value)
        
        return aruco_dict, aruco_params
    
    def _detect_in_video(self, video_file: Path, output_dir: Path) -> Dict[str, Any]:
        """Detect ArUco markers in a single video."""
        detection_file = output_dir / f"{video_file.stem}_aruco.json"
        
        # This would implement actual ArUco detection
        # For now, create placeholder detection results
        
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_file}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample every 30th frame for efficiency
        detections = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % 30 == 0:
                # Detect markers
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    frame, self.aruco_dict, parameters=self.aruco_params
                )
                
                if ids is not None:
                    detections.append({
                        "frame": frame_idx,
                        "timestamp": frame_idx / fps,
                        "markers": [
                            {
                                "id": int(ids[i][0]),
                                "corners": corners[i].tolist()
                            }
                            for i in range(len(ids))
                        ]
                    })
            
            frame_idx += 1
        
        cap.release()
        
        # Save detection results
        result = {
            "video_file": str(video_file),
            "total_frames": frame_count,
            "detections": detections
        }
        
        with open(detection_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def validate_output(self, output_dir: str) -> bool:
        """Validate ArUco detection results."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return False
        
        detection_files = list(output_path.glob('*_aruco.json'))
        return len(detection_files) > 0