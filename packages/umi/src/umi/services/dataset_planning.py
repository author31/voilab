from pathlib import Path
import json
import numpy as np
from typing import Dict, Any, List

from .base_service import BaseService


class DatasetPlanningService(BaseService):
    """Service for generating dataset plans from processed data."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.tcp_offset = np.array(self.config.get('tcp_offset', [0.0, 0.0, 0.0]))
        self.nominal_z = self.config.get('nominal_z', 0.0)
        self.min_episode_length = self.config.get('min_episode_length', 10)
    
    def execute(self, input_dir: str, output_dir: str) -> dict:
        """
        Generate dataset plan from processed data.
        
        Args:
            input_dir: Directory containing processed videos, maps, and calibrations
            output_dir: Directory for dataset plan outputs
            
        Returns:
            dict: Dataset plan with episode definitions
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path = self._ensure_output_dir(output_dir)
        
        # Load all processed data
        calibrations = self._load_calibrations(input_path)
        trajectories = self._load_trajectories(input_path)
        aruco_detections = self._load_aruco_detections(input_path)
        
        # Generate dataset plan
        dataset_plan = self._create_dataset_plan(
            calibrations, trajectories, aruco_detections
        )
        
        # Save dataset plan
        plan_file = output_path / "dataset_plan.json"
        with open(plan_file, 'w') as f:
            json.dump(dataset_plan, f, indent=2)
        
        return {
            "plan_file": str(plan_file),
            "total_episodes": len(dataset_plan.get("episodes", [])),
            "total_frames": sum(ep.get("frame_count", 0) for ep in dataset_plan.get("episodes", [])),
            "plan": dataset_plan
        }
    
    def _load_calibrations(self, input_path: Path) -> Dict[str, Any]:
        """Load calibration data."""
        calibrations = {}
        
        # Load SLAM tag calibration
        tag_calib_file = input_path / "slam_tag_calibration.json"
        if tag_calib_file.exists():
            with open(tag_calib_file, 'r') as f:
                calibrations["slam_tag"] = json.load(f)
        
        # Load gripper range calibration
        gripper_calib_file = input_path / "gripper_range_calibration.json"
        if gripper_calib_file.exists():
            with open(gripper_calib_file, 'r') as f:
                calibrations["gripper_range"] = json.load(f)
        
        return calibrations
    
    def _load_trajectories(self, input_path: Path) -> Dict[str, Any]:
        """Load SLAM trajectories."""
        trajectories = {}
        
        # Find all trajectory files
        for traj_file in input_path.rglob("*trajectory*.txt"):
            demo_name = traj_file.parent.name
            trajectories[demo_name] = str(traj_file)
        
        return trajectories
    
    def _load_aruco_detections(self, input_path: Path) -> Dict[str, Any]:
        """Load ArUco detection results."""
        detections = {}
        
        # Find all detection files
        for det_file in input_path.rglob("*_aruco.json"):
            demo_name = det_file.parent.name
            with open(det_file, 'r') as f:
                detections[demo_name] = json.load(f)
        
        return detections
    
    def _create_dataset_plan(self, calibrations: Dict, trajectories: Dict, 
                           detections: Dict) -> Dict[str, Any]:
        """Create comprehensive dataset plan."""
        
        episodes = []
        
        # Process each demo
        for demo_name in set(list(trajectories.keys()) + list(detections.keys())):
            
            # Get trajectory and detection data
            traj_file = trajectories.get(demo_name)
            detection_data = detections.get(demo_name, {})
            
            # Create episode definition
            episode = {
                "demo_name": demo_name,
                "trajectory_file": traj_file,
                "detection_data": detection_data,
                "frame_count": detection_data.get("total_frames", 0),
                "duration": len(detection_data.get("detections", [])) * 0.033,  # 30fps
                "key_frames": [],
                "annotations": {},
                "metadata": {
                    "tcp_offset": self.tcp_offset.tolist(),
                    "nominal_z": self.nominal_z,
                    "calibrations": calibrations
                }
            }
            
            # Filter episodes by minimum length
            if episode["frame_count"] >= self.min_episode_length:
                episodes.append(episode)
        
        # Sort episodes by duration
        episodes.sort(key=lambda x: x["duration"], reverse=True)
        
        return {
            "episodes": episodes,
            "total_episodes": len(episodes),
            "total_duration": sum(ep["duration"] for ep in episodes),
            "config": {
                "tcp_offset": self.tcp_offset.tolist(),
                "nominal_z": self.nominal_z,
                "min_episode_length": self.min_episode_length
            }
        }
    
    def validate_output(self, output_dir: str) -> bool:
        """Validate dataset plan."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return False
        
        plan_file = output_path / "dataset_plan.json"
        if not plan_file.exists():
            return False
        
        with open(plan_file, 'r') as f:
            plan = json.load(f)
        
        return len(plan.get("episodes", [])) > 0