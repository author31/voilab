import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class ArUcoDetectionLoader:
    """Loader for ArUco detection results synchronized with video frames."""
    
    def __init__(self, directory_path: str):
        """Initialize loader with directory containing raw_video.mp4 and tag_detection.pkl."""
        self.directory_path = Path(directory_path)
        self.video_path = self.directory_path / "raw_video.mp4"
        self.pkl_path = self.directory_path / "tag_detection.pkl"
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if not self.pkl_path.exists():
            raise FileNotFoundError(f"Detection results not found: {self.pkl_path}")
        
        # Load detection results
        with open(self.pkl_path, 'rb') as f:
            self.detection_results = pickle.load(f)
        
        # Create mapping from frame index to detection result
        self.frame_to_detection = {
            result['frame_idx']: result for result in self.detection_results
        }
        
        # Open video to get frame count and FPS
        cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Cache for video frames
        self._frame_cache = {}
        self._cap = None
    
    def _get_capture(self):
        """Lazy initialization of video capture."""
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.video_path))
        return self._cap
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Get video frame at specified index."""
        if frame_idx in self._frame_cache:
            return self._frame_cache[frame_idx]
        
        cap = self._get_capture()
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Frame {frame_idx} not found")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame_cache[frame_idx] = img
        return img
    
    def get_detection(self, frame_idx: int) -> Optional[Dict]:
        """Get detection result for specified frame."""
        return self.frame_to_detection.get(frame_idx)
    
    def get_detections_stats(self) -> Dict:
        """Get statistics about detections across all frames."""
        total_detections = sum(len(result.get('tag_dict', {})) for result in self.detection_results)
        frames_with_detections = sum(1 for result in self.detection_results if result.get('tag_dict', {}))
        
        # Get unique marker IDs
        all_marker_ids = set()
        for result in self.detection_results:
            all_marker_ids.update(result.get('tag_dict', {}).keys())
        
        return {
            'total_frames': self.total_frames,
            'frames_with_detections': frames_with_detections,
            'total_detections': total_detections,
            'unique_marker_ids': sorted(list(all_marker_ids)),
            'detection_rate': frames_with_detections / self.total_frames if self.total_frames > 0 else 0
        }
    
    def close(self):
        """Close video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()