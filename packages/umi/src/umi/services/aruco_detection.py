import json
import multiprocessing
import pickle
import concurrent.futures
from pathlib import Path
import av
import cv2
import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm
from ..common.cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    draw_predefined_mask,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)
from .base_service import BaseService


class ArucoDetectionService(BaseService):
    """Service for detecting ArUco markers in video frames."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.session_dir = self.config.get("session_dir")
        self.num_workers = self.config.get("num_workers", multiprocessing.cpu_count() // 2)
        self.camera_intrinsics_path = self.config.get("camera_intrinsics_path")
        self.aruco_config_path = self.config.get("aruco_config_path")

    def execute(self) -> dict:
        assert self.session_dir, "Missing session_dir from the configuration"

        cv2.setNumThreads(self.num_workers)
        input_path = Path(self.session_dir) / "demos"
        input_video_dirs = [x.parent for x in input_path.glob("*/raw_video.mp4")]
        logger.info(f"Found {len(input_video_dirs)} video dirs")
        with (
            tqdm(total=len(input_video_dirs)) as pbar,
            concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor,
        ):
            futures = set()
            for video_dir in tqdm(input_video_dirs):
                video_path = video_dir / "raw_video.mp4"
                if (converted_path:=video_dir/f"converted_60fps_{video_path.name}").is_file():
                    video_path = converted_path

                pkl_path = video_dir / "tag_detection.pkl"
                if pkl_path.is_file():
                    logger.info(f"tag_detection.pkl already exists, skipping {video_dir.name}")
                    continue

                if len(futures) >= self.num_workers:
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    pbar.update(len(completed))

                futures.add(executor.submit(self.detect_aruco, video_path, pkl_path))
            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))
            for future in completed:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in detect_aruco: {e}")
                    raise

        processed_videos = []
        skipped_videos = []
        for video_dir in input_video_dirs:
            pkl_path = video_dir / "tag_detection.pkl"
            video_name = video_dir.name
            if pkl_path.is_file():
                skipped_videos.append(video_name)
            else:
                processed_videos.append(video_name)

        return {
            "total_videos_found": len(input_video_dirs),
            "videos_processed": len(processed_videos),
            "videos_skipped": len(skipped_videos),
            "processed_video_names": processed_videos,
            "skipped_video_names": skipped_videos,
            "detection_results_dir": str(input_path),
        }

    def detect_aruco(self, video_path, tag_detection_dest):
        assert self.camera_intrinsics_path, "Missing camera_intrinsics_path from the configuration"
        assert self.aruco_config_path, "Missing aruco_config_path from the configuration"

        aruco_config = parse_aruco_config(yaml.safe_load(open(self.aruco_config_path, "r")))
        aruco_dict = aruco_config["aruco_dict"]
        marker_size_map = aruco_config["marker_size_map"]
        raw_fisheye_intr = parse_fisheye_intrinsics(json.load(open(self.camera_intrinsics_path, "r")))
        results = []
        with av.open(str(video_path)) as in_container:
            in_stream = in_container.streams.video[0]
            in_stream.thread_type = "AUTO"
            in_stream.thread_count = self.num_workers
            in_res = np.array([in_stream.height, in_stream.width])[::(-1)]
            fisheye_intr = convert_fisheye_intrinsics_resolution(
                opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res
            )
            for i, frame in tqdm(enumerate(in_container.decode(in_stream)), total=in_stream.frames):
                img = frame.to_ndarray(format="rgb24")
                frame_cts_sec = frame.pts * in_stream.time_base
                img = draw_predefined_mask(img, color=(0, 0, 0), mirror=True, gripper=False, finger=False)
                tag_dict = detect_localize_aruco_tags(
                    img=img,
                    aruco_dict=aruco_dict,
                    marker_size_map=marker_size_map,
                    fisheye_intr_dict=fisheye_intr,
                    refine_subpix=True,
                )
                result = {
                    "frame_idx": i,
                    "time": float(frame_cts_sec),
                    "tag_dict": tag_dict,
                }
                results.append(result)
        pickle.dump(results, open(str(tag_detection_dest), "wb"))
