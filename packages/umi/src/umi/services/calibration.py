import collections
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.transform import Rotation
from skfda.exploratory.stats import geometric_median

from ..common.cv_util import get_gripper_width
from ..common.pose_util import pose_to_mat
from .base_service import BaseService


class CalibrationService(BaseService):
    """Service for running SLAM tag and gripper range calibrations."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.session_dir = self.config.get("session_dir")
        self.slam_tag_timeout = self.config.get(
            "slam_tag_calibration_timeout", 300)
        self.gripper_range_timeout = self.config.get(
            "gripper_range_timeout", 300)
        self.keyframe_only = self.config.get("keyframe_only", True)
        self.tag_id = self.config.get("tag_id", 13)
        self.dist_to_center_threshold = self.config.get(
            "dist_to_center_threshold", 0)
        self.tag_detection_threshold = self.config.get(
            "tag_detection_threshold", 0.1)
        self.nominal_z = self.config.get("nominal_z", 0.072)
        self.resolution = self.config.get("resolution")

    def execute(self) -> dict:
        """
        Execute calibration service.
        Returns:
            dict: Calibration results
        """
        logger.info("Starting calibration service execution")

        assert self.session_dir, "Missing session_dir from configuration"
        assert self.tag_id, "Missing tag_id from configuration"

        logger.info(f"Using session directory: {self.session_dir}")
        logger.info(f"Calibrating SLAM tag with ID: {self.tag_id}")

        # Calibrate SLAM tag
        slam_tag_result = self.calibrate_slam_tag()
        logger.info("SLAM tag calibration completed successfully")

        # Calibrate gripper range
        gripper_range_result = self.calibrate_gripper_range()
        logger.info(
            f"Gripper range calibration completed for {len(gripper_range_result)} gripper(s)")

        logger.info("Calibration service execution completed")
        return {
            "slam_tag_calibration": slam_tag_result,
            "gripper_range_calibration": gripper_range_result,
            "errors": [],
        }

    def calibrate_slam_tag(self) -> dict:
        """
        Calibrate SLAM tag position.
        Returns:
            dict: SLAM tag calibration result
        """
        logger.info("Starting SLAM tag calibration")
        assert self.resolution, "Missing resolution in configuration "

        input_path = Path(self.session_dir)
        demos_dir = input_path / "demos"
        mapping_dir = demos_dir / "mapping"
        slam_tag_path = mapping_dir / "tx_slam_tag.json"

        tag_path = mapping_dir / "tag_detection.pkl"
        assert tag_path.is_file(), f"Required file not found: {tag_path}"

        csv_path = mapping_dir / "camera_trajectory.csv"
        if not csv_path.is_file():
            csv_path = mapping_dir / "mapping_camera_trajectory.csv"
            logger.warning(
                f"camera_trajectory.csv not found, using fallback: {csv_path}")
        assert csv_path.is_file(), f"Required file not found: {csv_path}"

        logger.info(f"Loading camera trajectory from: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loading tag detections from: {tag_path}")
        tag_detection_results = pickle.load(open(tag_path, "rb"))

        # Filter valid frames
        is_valid = ~df["is_lost"]
        if self.keyframe_only:
            is_valid &= df["is_keyframe"]
            logger.info(
                f"Using keyframe-only mode, {is_valid.sum()} valid frames")
        else:
            logger.info(f"Using all frames, {is_valid.sum()} valid frames")

        # Extract camera poses
        cam_pose_timestamps = df["timestamp"].loc[is_valid].to_numpy()
        cam_pos = df[["x", "y", "z"]].loc[is_valid].to_numpy()
        cam_rot_quat_xyzw = df[["q_x", "q_y",
                                "q_z", "q_w"]].loc[is_valid].to_numpy()
        cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
        cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
        cam_pose[:, 3, 3] = 1
        cam_pose[:, :3, 3] = cam_pos
        cam_pose[:, :3, :3] = cam_rot.as_matrix()

        # Find corresponding video frames
        video_timestamps = np.array([x["time"] for x in tag_detection_results])
        tum_video_idxs = []
        for t in cam_pose_timestamps:
            tum_video_idxs.append(np.argmin(np.abs(video_timestamps - t)))

        # Collect valid tag detections
        all_tx_slam_tag = []
        all_idxs = []
        skipped_distance = 0
        skipped_center = 0

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
            if dist_to_cam < 0.3 or dist_to_cam > 4:
                logger.warning(f"{dist_to_cam:.2f}m to tag, skipping")
                skipped_distance += 1
                continue

            corners = tag["corners"]
            tag_center_pix = corners.mean(axis=0)
            img_center = np.array(self.resolution, dtype=np.float32) / 2
            dist_to_center = np.linalg.norm(
                tag_center_pix - img_center) / img_center[0]

            if dist_to_center > self.dist_to_center_threshold:
                skipped_center += 1
                continue

            tx_slam_tag = tx_slam_cam @ tx_cam_tag
            all_tx_slam_tag.append(tx_slam_tag)
            all_idxs.append(tum_idx)

        logger.info(
            f"Collected {len(all_tx_slam_tag)} valid tag detections (skipped {skipped_distance} due to distance, {skipped_center} due to center offset)")

        if len(all_tx_slam_tag) == 0:
            raise ValueError("No valid tag detections found after filtering")

        # Compute median and filter outliers
        all_tx_slam_tag = np.array(all_tx_slam_tag)
        all_slam_tag_pos = all_tx_slam_tag[:, :3, 3]
        median = geometric_median(all_slam_tag_pos)
        dists = np.linalg.norm(all_tx_slam_tag[:, :3, 3] - median, axis=(-1))
        threshold = np.quantile(dists, 0.9)
        is_valid = dists < threshold
        std = all_slam_tag_pos[is_valid].std(axis=0)
        mean = all_slam_tag_pos[is_valid].mean(axis=0)
        dists = np.linalg.norm(
            all_tx_slam_tag[is_valid][:, :3, 3] - mean, axis=(-1))
        nn_idx = np.argmin(dists)
        tx_slam_tag = all_tx_slam_tag[is_valid][nn_idx]

        logger.info(
            f"Tag detection precision (cm) - std: [{std[0]*100:.1f}, {std[1]*100:.1f}, {std[2]*100:.1f}], {is_valid.sum()}/{len(all_tx_slam_tag)} detections within 90th percentile")
        logger.info(
            f"Final tag position (meters): [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")

        result = {"tx_slam_tag": tx_slam_tag.tolist()}
        json.dump(result, open(slam_tag_path, "w"), indent=2)
        logger.info(f"Saved SLAM tag calibration result to: {slam_tag_path}")

        return result

    def calibrate_gripper_range(self) -> dict:
        """
        Calibrate gripper range.
        Returns:
            dict: Gripper range calibration result
        """
        logger.info("Starting gripper range calibration")

        input_path = Path(self.session_dir)
        demos_dir = input_path / "demos"
        gripper_dirs = list(demos_dir.glob("gripper_calibration*"))

        if not gripper_dirs:
            logger.warning("No gripper calibration directories found")
            return {}

        logger.info(
            f"Found {len(gripper_dirs)} gripper calibration directory(s)")

        results = {}
        for gripper_dir in gripper_dirs:
            logger.info(
                f"Processing gripper calibration directory: {gripper_dir.name}")
            gripper_range_path = gripper_dir/'gripper_range.json'
            tag_path = gripper_dir/'tag_detection.pkl'
            assert tag_path.is_file(), f"Required file not found: {tag_path}"

            logger.info(f"Loading tag detections from: {tag_path}")
            tag_detection_results = pickle.load(open(tag_path, 'rb'))

            # Identify gripper hardware ID
            n_frames = len(tag_detection_results)
            logger.info(f"Analyzing {n_frames} frames for gripper detection")

            tag_counts = collections.defaultdict(lambda: 0)
            for frame in tag_detection_results:
                for key in frame['tag_dict'].keys():
                    tag_counts[key] += 1

            tag_stats = collections.defaultdict(lambda: 0.0)
            for k, v in tag_counts.items():
                tag_stats[k] = v / n_frames

            max_tag_id = np.max(list(tag_stats.keys()))
            tag_per_gripper = 6
            max_gripper_id = max_tag_id // tag_per_gripper
            logger.info(
                f"Found tags up to ID {max_tag_id}, checking up to gripper ID {max_gripper_id}")

            # Calculate detection probability for each gripper
            gripper_prob_map = {}
            for gripper_id in range(max_gripper_id+1):
                left_id = gripper_id * tag_per_gripper
                right_id = left_id + 1
                left_prob = tag_stats.get(left_id, 0.0)
                right_prob = tag_stats.get(right_id, 0.0)
                gripper_prob = min(left_prob, right_prob)
                if gripper_prob > 0:
                    gripper_prob_map[gripper_id] = gripper_prob
                    logger.debug(
                        f"Gripper {gripper_id}: left_tag={left_id} ({left_prob:.2f}), right_tag={right_id} ({right_prob:.2f}), min_prob={gripper_prob:.2f}")

            if not gripper_prob_map:
                logger.error(
                    "No grippers detected with sufficient tag visibility")
                raise ValueError("No grippers detected")

            # Select gripper with highest detection probability
            gripper_probs = sorted(
                gripper_prob_map.items(), key=lambda x: x[1])
            gripper_id = gripper_probs[-1][0]
            gripper_prob = gripper_probs[-1][1]
            logger.info(
                f"Selected gripper ID {gripper_id} with detection probability {gripper_prob:.2f}")

            if gripper_prob < self.tag_detection_threshold:
                logger.error(
                    f"Detection rate {gripper_prob:.2f} below threshold {self.tag_detection_threshold}")
                raise ValueError(
                    f"Gripper detection rate too low: {gripper_prob:.2f} < {self.tag_detection_threshold}")

            left_id = gripper_id * tag_per_gripper
            right_id = left_id + 1
            logger.info(f"Using tag IDs: left={left_id}, right={right_id}")

            # Calculate gripper widths
            gripper_widths = list()
            valid_widths = 0
            for i, dt in enumerate(tag_detection_results):
                tag_dict = dt['tag_dict']
                width = get_gripper_width(
                    tag_dict, left_id, right_id, nominal_z=self.nominal_z)
                if width is None:
                    width = float('Nan')
                else:
                    valid_widths += 1
                gripper_widths.append(width)

            gripper_widths = np.array(gripper_widths)
            max_width = np.nanmax(gripper_widths)
            min_width = np.nanmin(gripper_widths)

            logger.info(
                f"Measured gripper range: {min_width*100:.1f}cm - {max_width*100:.1f}cm ({valid_widths}/{n_frames} valid measurements)")

            result = {
                'gripper_id': gripper_id,
                'left_finger_tag_id': left_id,
                'right_finger_tag_id': right_id,
                'max_width': max_width,
                'min_width': min_width
            }
            json.dump(result, open(gripper_range_path, 'w'), indent=2)
            logger.info(
                f"Saved gripper range calibration to: {gripper_range_path}")
            results[gripper_dir.name] = result

        return results
