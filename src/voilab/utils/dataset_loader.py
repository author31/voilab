"""Loader for UMI dataset session directories.

This module provides utilities for loading and analyzing the structure
of a UMI pipeline session directory, including:
- Video organization (demos, mapping, gripper calibration)
- SLAM trajectory data
- ArUco tag detection results
- Calibration results
- Dataset planning results
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class DemoInfo:
    """Information about a single demo directory."""

    name: str
    path: Path
    has_video: bool = False
    has_tag_detection: bool = False
    has_camera_trajectory: bool = False
    has_imu_data: bool = False
    n_frames: int = 0
    n_detected_tags: int = 0
    detection_rate: float = 0.0
    n_lost_frames: int = 0
    trajectory_quality: str = "unknown"  # "good", "warning", "bad", "unknown"


@dataclass
class CalibrationInfo:
    """Information about calibration results."""

    has_slam_tag: bool = False
    slam_tag_position: Optional[List[float]] = None
    gripper_calibrations: Dict[str, dict] = field(default_factory=dict)


@dataclass
class DatasetPlanInfo:
    """Information about dataset planning results."""

    has_plan: bool = False
    plan_path: Optional[Path] = None
    n_episodes: int = 0
    total_frames: int = 0


@dataclass
class DatasetSessionInfo:
    """Complete information about a dataset session."""

    session_path: Path
    demos: List[DemoInfo] = field(default_factory=list)
    mapping: Optional[DemoInfo] = None
    gripper_calibrations: List[DemoInfo] = field(default_factory=list)
    calibration: CalibrationInfo = field(default_factory=CalibrationInfo)
    dataset_plan: DatasetPlanInfo = field(default_factory=DatasetPlanInfo)

    @property
    def n_demos(self) -> int:
        """Number of demo directories."""
        return len(self.demos)

    @property
    def demos_with_issues(self) -> List[DemoInfo]:
        """Get demos that have quality issues."""
        return [d for d in self.demos if d.trajectory_quality in ("warning", "bad")]


class DatasetLoader:
    """Loader for UMI dataset session directories."""

    def __init__(self, session_dir: str):
        """Initialize the loader.

        Args:
            session_dir: Path to the session directory
        """
        self.session_path = Path(session_dir).absolute()
        if not self.session_path.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        self.demos_dir = self.session_path / "demos"
        self._session_info: Optional[DatasetSessionInfo] = None

    def load(self) -> DatasetSessionInfo:
        """Load and analyze the session directory.

        Returns:
            DatasetSessionInfo with complete analysis
        """
        if self._session_info is not None:
            return self._session_info

        info = DatasetSessionInfo(session_path=self.session_path)

        if not self.demos_dir.exists():
            return info

        # Load mapping directory
        mapping_dir = self.demos_dir / "mapping"
        if mapping_dir.exists():
            info.mapping = self._load_demo_info(mapping_dir)

        # Load gripper calibration directories
        for cal_dir in sorted(self.demos_dir.glob("gripper_calibration*")):
            info.gripper_calibrations.append(self._load_demo_info(cal_dir))

        # Load demo directories
        for demo_dir in sorted(self.demos_dir.glob("demo_*")):
            info.demos.append(self._load_demo_info(demo_dir))

        # Load calibration results
        info.calibration = self._load_calibration_info()

        # Load dataset plan
        info.dataset_plan = self._load_dataset_plan_info()

        self._session_info = info
        return info

    def _load_demo_info(self, demo_dir: Path) -> DemoInfo:
        """Load information about a single demo directory."""
        demo_info = DemoInfo(name=demo_dir.name, path=demo_dir)

        # Check for video files
        video_path = demo_dir / "raw_video.mp4"
        converted_path = demo_dir / "converted_60fps_raw_video.mp4"
        demo_info.has_video = video_path.exists() or converted_path.exists()

        # Check for tag detection results
        tag_path = demo_dir / "tag_detection.pkl"
        demo_info.has_tag_detection = tag_path.exists()

        if tag_path.exists():
            try:
                with open(tag_path, "rb") as f:
                    tag_data = pickle.load(f)
                    demo_info.n_frames = len(tag_data)
                    # Count frames with at least one detected tag
                    frames_with_tags = sum(
                        1 for frame in tag_data if frame.get("tag_dict", {})
                    )
                    demo_info.n_detected_tags = sum(
                        len(frame.get("tag_dict", {})) for frame in tag_data
                    )
                    demo_info.detection_rate = (
                        frames_with_tags / len(tag_data) if tag_data else 0.0
                    )
            except Exception:
                pass

        # Check for camera trajectory
        csv_path = demo_dir / "camera_trajectory.csv"
        if not csv_path.exists():
            csv_path = demo_dir / "mapping_camera_trajectory.csv"

        demo_info.has_camera_trajectory = csv_path.exists()

        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                demo_info.n_lost_frames = df["is_lost"].sum()
                total_frames = len(df)

                # Assess trajectory quality
                lost_ratio = demo_info.n_lost_frames / total_frames if total_frames > 0 else 0
                if lost_ratio == 0:
                    demo_info.trajectory_quality = "good"
                elif lost_ratio < 0.05:
                    demo_info.trajectory_quality = "warning"
                else:
                    demo_info.trajectory_quality = "bad"
            except Exception:
                demo_info.trajectory_quality = "unknown"

        # Check for IMU data
        imu_path = demo_dir / "imu_data.json"
        demo_info.has_imu_data = imu_path.exists()

        return demo_info

    def _load_calibration_info(self) -> CalibrationInfo:
        """Load calibration results."""
        cal_info = CalibrationInfo()

        # Check for SLAM tag calibration
        slam_tag_path = self.demos_dir / "mapping" / "tx_slam_tag.json"
        if slam_tag_path.exists():
            try:
                with open(slam_tag_path) as f:
                    data = json.load(f)
                    cal_info.has_slam_tag = True
                    tx = data.get("tx_slam_tag", [])
                    if tx and len(tx) >= 3:
                        # Extract position from 4x4 matrix
                        if isinstance(tx[0], list):
                            cal_info.slam_tag_position = [tx[0][3], tx[1][3], tx[2][3]]
                        else:
                            cal_info.slam_tag_position = tx[:3]
            except Exception:
                pass

        # Check for gripper calibrations
        for cal_dir in self.demos_dir.glob("gripper_calibration*"):
            gripper_range_path = cal_dir / "gripper_range.json"
            if gripper_range_path.exists():
                try:
                    with open(gripper_range_path) as f:
                        data = json.load(f)
                        cal_info.gripper_calibrations[cal_dir.name] = data
                except Exception:
                    pass

        return cal_info

    def _load_dataset_plan_info(self) -> DatasetPlanInfo:
        """Load dataset planning results."""
        plan_info = DatasetPlanInfo()

        # Check for dataset plan
        for plan_name in ["dataset_plan.pkl", "dataset.pkl"]:
            plan_path = self.session_path / plan_name
            if plan_path.exists():
                plan_info.has_plan = True
                plan_info.plan_path = plan_path
                try:
                    with open(plan_path, "rb") as f:
                        plan_data = pickle.load(f)
                        plan_info.n_episodes = len(plan_data)
                        plan_info.total_frames = sum(
                            len(ep.get("episode_timestamps", [])) for ep in plan_data
                        )
                except Exception:
                    pass
                break

        return plan_info

    def get_demo_trajectory(self, demo_name: str) -> Optional[pd.DataFrame]:
        """Get trajectory data for a specific demo.

        Args:
            demo_name: Name of the demo directory

        Returns:
            DataFrame with trajectory data or None if not found
        """
        demo_dir = self.demos_dir / demo_name
        csv_path = demo_dir / "camera_trajectory.csv"
        if not csv_path.exists():
            csv_path = demo_dir / "mapping_camera_trajectory.csv"

        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None

    def get_pipeline_stages_status(self) -> Dict[str, dict]:
        """Get status of each pipeline stage.

        Returns:
            Dictionary mapping stage names to status info
        """
        info = self.load()
        stages = {}

        # Stage 0: Video Organization
        stages["00_process_video"] = {
            "name": "Video Organization",
            "status": "complete" if info.demos or info.mapping else "pending",
            "details": {
                "n_demos": len(info.demos),
                "has_mapping": info.mapping is not None,
                "n_gripper_cal": len(info.gripper_calibrations),
            },
        }

        # Stage 1: IMU Extraction
        has_imu = any(d.has_imu_data for d in info.demos)
        stages["01_extract_gopro_imu"] = {
            "name": "IMU Extraction",
            "status": "complete" if has_imu else "pending",
            "details": {
                "demos_with_imu": sum(1 for d in info.demos if d.has_imu_data),
            },
        }

        # Stage 2: Create Map
        mapping_has_traj = info.mapping and info.mapping.has_camera_trajectory
        stages["02_create_map"] = {
            "name": "SLAM Mapping",
            "status": "complete" if mapping_has_traj else "pending",
            "details": {
                "has_trajectory": mapping_has_traj,
                "n_lost_frames": info.mapping.n_lost_frames if info.mapping else 0,
            },
        }

        # Stage 3: Batch SLAM
        demos_with_traj = sum(1 for d in info.demos if d.has_camera_trajectory)
        stages["03_batch_slam"] = {
            "name": "Batch SLAM",
            "status": "complete" if demos_with_traj == len(info.demos) else "partial" if demos_with_traj > 0 else "pending",
            "details": {
                "demos_with_trajectory": demos_with_traj,
                "total_demos": len(info.demos),
                "demos_with_issues": len(info.demos_with_issues),
            },
        }

        # Stage 4: ArUco Detection
        demos_with_tags = sum(1 for d in info.demos if d.has_tag_detection)
        stages["04_detect_aruco"] = {
            "name": "ArUco Detection",
            "status": "complete" if demos_with_tags == len(info.demos) else "partial" if demos_with_tags > 0 else "pending",
            "details": {
                "demos_with_detection": demos_with_tags,
                "total_demos": len(info.demos),
            },
        }

        # Stage 5: Calibration
        stages["05_run_calibrations"] = {
            "name": "Calibration",
            "status": "complete" if info.calibration.has_slam_tag else "pending",
            "details": {
                "has_slam_tag": info.calibration.has_slam_tag,
                "n_gripper_calibrations": len(info.calibration.gripper_calibrations),
            },
        }

        # Stage 6: Dataset Planning
        stages["06_generate_dataset_plan"] = {
            "name": "Dataset Planning",
            "status": "complete" if info.dataset_plan.has_plan else "pending",
            "details": {
                "has_plan": info.dataset_plan.has_plan,
                "n_episodes": info.dataset_plan.n_episodes,
                "total_frames": info.dataset_plan.total_frames,
            },
        }

        # Stage 7: Replay Buffer (check for zarr file)
        zarr_exists = (self.session_path / "dataset.zarr.zip").exists()
        stages["07_generate_replay_buffer"] = {
            "name": "Replay Buffer",
            "status": "complete" if zarr_exists else "pending",
            "details": {
                "has_output": zarr_exists,
            },
        }

        return stages
