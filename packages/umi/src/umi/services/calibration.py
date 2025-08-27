from pathlib import Path
import json
import subprocess
import numpy as np
from typing import Dict, Any, List

from .base_service import BaseService


class CalibrationService(BaseService):
    """Service for running SLAM tag and gripper range calibrations."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.slam_tag_timeout = self.config.get('slam_tag_calibration_timeout', 300)
        self.gripper_range_timeout = self.config.get('gripper_range_timeout', 300)
    
    def execute(self, input_dir: str, output_dir: str) -> dict:
        """
        Execute calibration service.
        
        Args:
            input_dir: Directory containing processed data
            output_dir: Directory for calibration outputs
            
        Returns:
            dict: Calibration results
        """
        input_path = Path(input_dir)
        output_path = self._ensure_output_dir(output_dir)
        
        results = {
            "slam_tag_calibration": None,
            "gripper_range_calibration": None,
            "errors": []
        }
        
        try:
            # Run SLAM tag calibration
            slam_result = self._calibrate_slam_tag(input_path, output_path)
            results["slam_tag_calibration"] = slam_result
        except Exception as e:
            results["errors"].append(f"SLAM tag calibration failed: {str(e)}")
        
        try:
            # Run gripper range calibration
            gripper_result = self._calibrate_gripper_range(input_path, output_path)
            results["gripper_range_calibration"] = gripper_result
        except Exception as e:
            results["errors"].append(f"Gripper range calibration failed: {str(e)}")
        
        return results
    
    def _calibrate_slam_tag(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Calibrate SLAM tag positions."""
        calibration_file = output_path / "slam_tag_calibration.json"
        
        # Find ArUco detection files
        aruco_files = list(input_path.rglob('*_aruco.json'))
        if not aruco_files:
            raise ValueError("No ArUco detection files found")
        
        # This would implement actual SLAM tag calibration
        # For now, create placeholder calibration
        
        calibration_data = {
            "calibration_type": "slam_tag",
            "tag_positions": {},
            "world_to_camera_transforms": {},
            "validation_errors": []
        }
        
        # Process each detection file
        for aruco_file in aruco_files:
            with open(aruco_file, 'r') as f:
                detections = json.load(f)
            
            # Extract tag positions and poses
            # This is simplified - actual implementation would be more complex
            calibration_data["tag_positions"][aruco_file.stem] = {
                "markers": [d["markers"] for d in detections.get("detections", [])]
            }
        
        # Save calibration results
        with open(calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        return {
            "calibration_file": str(calibration_file),
            "num_tags_calibrated": len(calibration_data["tag_positions"]),
            "success": True
        }
    
    def _calibrate_gripper_range(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Calibrate gripper range from demonstration data."""
        calibration_file = output_path / "gripper_range_calibration.json"
        
        # Find trajectory files
        trajectory_files = list(input_path.rglob('*trajectory*.txt'))
        if not trajectory_files:
            raise ValueError("No trajectory files found")
        
        # This would implement actual gripper range calibration
        # For now, create placeholder calibration
        
        calibration_data = {
            "calibration_type": "gripper_range",
            "min_position": [0.0, 0.0, 0.0],
            "max_position": [1.0, 1.0, 1.0],
            "workspace_bounds": {
                "x": [-0.5, 0.5],
                "y": [-0.5, 0.5],
                "z": [0.0, 1.0]
            },
            "validation_errors": []
        }
        
        # Save calibration results
        with open(calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        return {
            "calibration_file": str(calibration_file),
            "workspace_volume": "1.0 cubic meters",
            "success": True
        }
    
    def validate_output(self, output_dir: str) -> bool:
        """Validate calibration results."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return False
        
        required_files = ["slam_tag_calibration.json", "gripper_range_calibration.json"]
        return all((output_path / f).exists() for f in required_files)