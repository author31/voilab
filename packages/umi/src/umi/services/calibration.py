import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.transform import Rotation
from skfda.exploratory.stats import geometric_median

from ..common.pose_util import pose_to_mat
from .base_service import BaseService


class CalibrationService(BaseService):
    """Service for running SLAM tag and gripper range calibrations."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.session_dir = self.config.get("session_dir")
        self.slam_tag_timeout = self.config.get("slam_tag_calibration_timeout", 300)
        self.gripper_range_timeout = self.config.get("gripper_range_timeout", 300)
        self.keyframe_only = self.config.get("keyframe_only", True)
        self.tag_id = self.config.get("tag_id", 13)
        self.dist_to_center_threshold = self.config.get("dist_to_center_threshold", 0)

    def execute(self) -> dict:
        """
        Execute calibration service.
        Returns:
            dict: Calibration results
        """
        assert self.session_dir, "Missing session_dir from configuration"
        assert self.tag_id, "Missing tag_id from configuration"
        input_path = Path(self.session_dir)
        demos_dir = input_path / "demos"
        mapping_dir = demos_dir / "mapping"
        slam_tag_path = mapping_dir / "tx_slam_tag.json"
        tag_path = mapping_dir / "tag_detection.pkl"
        assert tag_path.is_file()
        csv_path = mapping_dir / "camera_trajectory.csv"
        if not csv_path.is_file():
            csv_path = mapping_dir / "mapping_camera_trajectory.csv"
            logger.info("camera_trajectory.csv not found! using mapping_camera_trajectory.csv")
        assert csv_path.is_file()
        df = pd.read_csv(csv_path)
        tag_detection_results = pickle.load(open(tag_path, "rb"))
        is_valid = ~df["is_lost"]
        if self.keyframe_only:
            is_valid &= df["is_keyframe"]

        cam_pose_timestamps = df["timestamp"].loc[is_valid].to_numpy()
        cam_pos = df[["x", "y", "z"]].loc[is_valid].to_numpy()
        cam_rot_quat_xyzw = df[["q_x", "q_y", "q_z", "q_w"]].loc[is_valid].to_numpy()
        cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
        cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
        cam_pose[:, 3, 3] = 1
        cam_pose[:, :3, 3] = cam_pos
        cam_pose[:, :3, :3] = cam_rot.as_matrix()

        video_timestamps = np.array([x["time"] for x in tag_detection_results])
        tum_video_idxs = []
        for t in cam_pose_timestamps:
            tum_video_idxs.append(np.argmin(np.abs(video_timestamps - t)))

        all_tx_slam_tag = []
        all_idxs = []
        for tum_idx, video_idx in enumerate(tum_video_idxs):
            td = tag_detection_results[video_idx]
            tag_dict = td["tag_dict"]
            if self.tag_id not in tag_dict:
                continue

            tag = tag_dict[self.tag_id]
            pose = np.concatenate([tag["tvec"], tag["rvec"]])
            tx_cam_tag = pose_to_mat(pose)
            tx_slam_cam = cam_pose[tum_idx]
            dist_to_cam = np.linalg.norm(tx_cam_tag[:3, 3])
            if dist_to_cam < 0.3 or dist_to_cam > 2:
                continue

            corners = tag["corners"]
            tag_center_pix = corners.mean(axis=0)
            img_center = np.array([2704, 2028], dtype=np.float32) / 2
            dist_to_center = np.linalg.norm(tag_center_pix - img_center) / img_center[1]

            if dist_to_center > self.dist_to_center_threshold:
                continue

            tx_slam_tag = tx_slam_cam @ tx_cam_tag
            all_tx_slam_tag.append(tx_slam_tag)
            all_idxs.append(tum_idx)

        all_tx_slam_tag = np.array(all_tx_slam_tag)
        all_slam_tag_pos = all_tx_slam_tag[:, :3, 3]
        median = geometric_median(all_slam_tag_pos)
        dists = np.linalg.norm(all_tx_slam_tag[:, :3, 3] - median, axis=(-1))
        threshold = np.quantile(dists, 0.9)
        is_valid = dists < threshold
        std = all_slam_tag_pos[is_valid].std(axis=0)
        mean = all_slam_tag_pos[is_valid].mean(axis=0)
        dists = np.linalg.norm(all_tx_slam_tag[is_valid][:, :3, 3] - mean, axis=(-1))
        nn_idx = np.argmin(dists)
        tx_slam_tag = all_tx_slam_tag[is_valid][nn_idx]
        logger.info(f"Tag detection standard deviation (cm) < 0.9 quantile: {std * 100}")
        result = {"tx_slam_tag": tx_slam_tag.tolist()}
        json.dump(result, open(slam_tag_path, "w"), indent=2)
        logger.info(f"Saved result to {slam_tag_path}")
        return {
            "slam_tag_calibration": None,
            "gripper_range_calibration": None,
            "errors": [],
        }
