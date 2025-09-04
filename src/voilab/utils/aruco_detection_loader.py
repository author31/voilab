import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import json
import yaml


class ArUcoDetectionLoader:
    """Loader for ArUco detection results synchronized with video frames."""
    
    def __init__(self, directory_path: str, camera_intrinsics_path: Optional[str] = None, 
                 aruco_config_path: Optional[str] = None):
        """Initialize loader with directory containing raw_video.mp4 and tag_detection.pkl.
        
        Args:
            directory_path: Path to directory containing raw_video.mp4 and tag_detection.pkl
            camera_intrinsics_path: Path to camera intrinsics JSON file
            aruco_config_path: Path to ArUco configuration YAML file
        """
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
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Cache for video frames
        self._frame_cache = {}
        self._undistorted_frame_cache = {}
        self._cap = None
        
        # Load camera intrinsics and ArUco config if provided
        self.camera_intrinsics_path = camera_intrinsics_path
        self.aruco_config_path = aruco_config_path
        self.camera_intrinsics = None
        self.aruco_config = None
        self.fisheye_intrinsics = None
        
        if camera_intrinsics_path:
            self.load_camera_intrinsics(camera_intrinsics_path)
        if aruco_config_path:
            self.load_aruco_config(aruco_config_path)
        
        # Cache for re-run detection results
        self._rerun_detection_cache = {}
    
    def load_camera_intrinsics(self, camera_intrinsics_path: str):
        """Load camera intrinsics from JSON file."""
        try:
            with open(camera_intrinsics_path, 'r') as f:
                self.camera_intrinsics = json.load(f)
            
            # Convert to OpenCV format (similar to parse_fisheye_intrinsics)
            intr_data = self.camera_intrinsics["intrinsics"]
            h = self.camera_intrinsics["image_height"]
            w = self.camera_intrinsics["image_width"]
            f = intr_data["focal_length"]
            px = intr_data["principal_pt_x"]
            py = intr_data["principal_pt_y"]
            
            # Kannala-Brandt non-linear parameters for distortion
            kb8 = [
                intr_data["radial_distortion_1"],
                intr_data["radial_distortion_2"],
                intr_data["radial_distortion_3"],
                intr_data["radial_distortion_4"],
            ]
            
            self.fisheye_intrinsics = {
                "DIM": np.array([w, h], dtype=np.int64),
                "K": np.array([[f, 0, px], [0, f, py], [0, 0, 1]], dtype=np.float64),
                "D": np.array([kb8]).T,
            }
            
            # Convert to video resolution if different
            video_res = np.array([self.video_width, self.video_height])
            if not np.array_equal(self.fisheye_intrinsics["DIM"], video_res):
                self.fisheye_intrinsics = self._convert_fisheye_intrinsics_resolution(
                    self.fisheye_intrinsics, video_res
                )
                
        except Exception as e:
            raise ValueError(f"Error loading camera intrinsics: {e}")
    
    def load_aruco_config(self, aruco_config_path: str):
        """Load ArUco configuration from YAML file."""
        try:
            with open(aruco_config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Parse ArUco config (similar to parse_aruco_config)
            aruco_dict_type = config_data["aruco_dict"]["predefined"]
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_type))
            
            # Parse marker size map
            marker_size_map = config_data["marker_size_map"]
            default_size = marker_size_map.get("default", None)
            
            n_markers = len(self.aruco_dict.bytesList)
            self.marker_size_map = {}
            
            for marker_id in range(n_markers):
                size = default_size
                if marker_id in marker_size_map:
                    size = marker_size_map[marker_id]
                self.marker_size_map[marker_id] = size
            
            self.aruco_config = config_data
            
        except Exception as e:
            raise ValueError(f"Error loading ArUco config: {e}")
    
    def _convert_fisheye_intrinsics_resolution(self, opencv_intr_dict: Dict, target_resolution: Tuple[int, int]) -> Dict:
        """Convert fisheye intrinsics parameter to a different resolution."""
        import copy
        
        iw, ih = opencv_intr_dict["DIM"]
        iK = opencv_intr_dict["K"]
        ifx = iK[0, 0]
        ify = iK[1, 1]
        ipx = iK[0, 2]
        ipy = iK[1, 2]

        ow, oh = target_resolution
        ofx = ifx / ih * oh
        ofy = ify / ih * oh
        opx = (ipx - (iw / 2)) / ih * oh + (ow / 2)
        opy = ipy / ih * oh
        oK = np.array([[ofx, 0, opx], [0, ofy, opy], [0, 0, 1]], dtype=np.float64)

        out_intr_dict = copy.deepcopy(opencv_intr_dict)
        out_intr_dict["DIM"] = np.array([ow, oh], dtype=np.int64)
        out_intr_dict["K"] = oK
        return out_intr_dict
    
    def _get_capture(self):
        """Lazy initialization of video capture."""
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.video_path))
        return self._cap
    
    def get_frame(self, frame_idx: int, undistorted: bool = False) -> np.ndarray:
        """Get video frame at specified index.
        
        Args:
            frame_idx: Frame index to retrieve
            undistorted: Whether to return undistorted frame (requires camera intrinsics)
        """
        if undistorted:
            if frame_idx in self._undistorted_frame_cache:
                return self._undistorted_frame_cache[frame_idx]
            
            # Get original frame first
            original_frame = self.get_frame(frame_idx, undistorted=False)
            
            # Apply undistortion if camera intrinsics are available
            if self.fisheye_intrinsics is not None:
                undistorted_frame = self._undistort_frame(original_frame)
                self._undistorted_frame_cache[frame_idx] = undistorted_frame
                return undistorted_frame
            return original_frame
        else:
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
    
    def get_detection(self, frame_idx: int, rerun: bool = False) -> Optional[Dict]:
        """Get detection result for specified frame.
        
        Args:
            frame_idx: Frame index to retrieve detection for
            rerun: Whether to re-run detection with current parameters
        """
        if rerun:
            if frame_idx in self._rerun_detection_cache:
                return self._rerun_detection_cache[frame_idx]
            
            # Re-run detection on this frame
            if self.fisheye_intrinsics is not None and self.aruco_dict is not None:
                img = self.get_frame(frame_idx, undistorted=False)
                detection_result = self._rerun_detection(img, frame_idx)
                self._rerun_detection_cache[frame_idx] = detection_result
                return detection_result
            else:
                # Missing required parameters, return original detection
                return self.frame_to_detection.get(frame_idx)
        else:
            return self.frame_to_detection.get(frame_idx)
    
    def _undistort_frame(self, img: np.ndarray) -> np.ndarray:
        """Apply fisheye undistortion to frame."""
        if self.fisheye_intrinsics is None:
            return img
        
        K = self.fisheye_intrinsics["K"]
        D = self.fisheye_intrinsics["D"]
        
        # Estimate new camera matrix for undistortion
        h, w = img.shape[:2]
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.0
        )
        
        # Undistort the image
        undistorted = cv2.fisheye.undistortImage(img, K, D, None, new_K)
        return undistorted
    
    def _rerun_detection(self, img: np.ndarray, frame_idx: int) -> Dict:
        """Re-run ArUco detection on a single frame."""
        if self.fisheye_intrinsics is None or self.aruco_dict is None:
            return {'frame_idx': frame_idx, 'time': 0.0, 'tag_dict': {}}
        
        # Apply predefined mask (similar to the original service)
        img_masked = self._draw_predefined_mask(img.copy())
        
        # Run detection
        param = cv2.aruco.DetectorParameters()
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            image=img_masked, dictionary=self.aruco_dict, parameters=param
        )
        
        if len(corners) == 0:
            return {'frame_idx': frame_idx, 'time': frame_idx / self.fps, 'tag_dict': {}}
        
        tag_dict = {}
        K = self.fisheye_intrinsics["K"]
        D = self.fisheye_intrinsics["D"]
        
        for this_id, this_corners in zip(ids, corners):
            this_id = int(this_id[0])
            if this_id not in self.marker_size_map:
                continue
            
            marker_size_m = self.marker_size_map[this_id]
            undistorted = cv2.fisheye.undistortPoints(this_corners, K, D, P=K)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                undistorted, marker_size_m, K, np.zeros((1, 5))
            )
            
            tag_dict[this_id] = {
                "rvec": rvec.squeeze(),
                "tvec": tvec.squeeze(),
                "corners": this_corners.squeeze(),
            }
        
        return {
            'frame_idx': frame_idx,
            'time': frame_idx / self.fps,
            'tag_dict': tag_dict,
        }
    
    def _draw_predefined_mask(self, img: np.ndarray, color=(0, 0, 0)) -> np.ndarray:
        """Draw predefined mask on image (similar to original service)."""
        # Simple mask implementation - mask out mirror regions
        h, w = img.shape[:2]
        
        # Mirror regions (simplified version)
        # Left mirror
        left_mirror = np.array([
            [int(w*0.1), int(h*0.4)],
            [int(w*0.25), int(h*0.3)],
            [int(w*0.2), int(h*0.1)],
            [int(w*0.05), int(h*0.15)],
            [int(w*0.05), int(h*0.45)],
            [int(w*0.1), int(h*0.45)],
        ], dtype=np.int32)
        
        # Right mirror (flipped)
        right_mirror = left_mirror.copy()
        right_mirror[:, 0] = w - right_mirror[:, 0]
        
        # Draw masks
        cv2.fillPoly(img, [left_mirror], color=color)
        cv2.fillPoly(img, [right_mirror], color=color)
        
        return img
    
    def get_detections_stats(self, rerun: bool = False) -> Dict:
        """Get statistics about detections across all frames."""
        if rerun:
            # For rerun detection, we'd need to process all frames
            # For now, return basic info
            return {
                'total_frames': self.total_frames,
                'frames_with_detections': 0,
                'total_detections': 0,
                'unique_marker_ids': [],
                'detection_rate': 0,
                'mode': 'rerun (not implemented for all frames)'
            }
        else:
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
                'detection_rate': frames_with_detections / self.total_frames if self.total_frames > 0 else 0,
                'mode': 'original'
            }
    
    def close(self):
        """Close video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()
