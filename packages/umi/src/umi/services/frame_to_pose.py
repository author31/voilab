import glob
import os
import cv2
from cv2 import aruco
import numpy as np
import argparse
import json
from pathlib import Path
from loguru import logger
from .base_service import BaseService

def get_key_from_value(d, value):
    return next(k for k, v in d.items() if v == value)

ROOT = Path(__file__).resolve().parents[3]
intrinsics_path = ROOT / "defaults" / "calibration" / "gopro13_intrinsics_2_7k.json"

def process_and_save_with_axes(
    OBJ_ID,
    frame,
    filename,
    save_dir,
    marker_size_m=0.018,
    return_pose_list=False,
    intrinsics_path=intrinsics_path,
):
    # --- load camera intrinsics from JSON ---
    with open(intrinsics_path, "r") as f:
        data = json.load(f)

    intr = data["intrinsics"]

    # 1. 焦距與主點
    fx = intr["focal_length"]
    aspect_ratio = intr["aspect_ratio"]
    fy = fx * aspect_ratio          # 通常是這樣：fy = fx * aspect_ratio
    cx = intr["principal_pt_x"]
    cy = intr["principal_pt_y"]

    # 2. 組成 K
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 3. Fisheye 畸變係數
    D_fish = np.array([
        intr["radial_distortion_1"],
        intr["radial_distortion_2"],
        intr["radial_distortion_3"],
        intr["radial_distortion_4"],
    ], dtype=np.float64)

    # --- fisheye to pinhole ---
    h, w = frame.shape[:2]
    new_K = K.copy()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D_fish, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    undist = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # --- aruco detection ---
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    parameters = aruco.DetectorParameters()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.minMarkerPerimeterRate = 0.01
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.minCornerDistanceRate = 0.05
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(undist)
    if ids is None or len(ids) == 0:
        logger.info(f"{filename}: detected none")
        return [] if return_pose_list else None

    # --- tag 3D corner layout ---
    s = marker_size_m
    pts3D = np.array([
        [-s/2,  s/2, 0],
        [ s/2,  s/2, 0],
        [ s/2, -s/2, 0],
        [-s/2, -s/2, 0],
    ], dtype=np.float32)

    ids = ids.flatten()
    object_pose_list = []
    for i, id_val in enumerate(ids):
        if id_val not in OBJ_ID.values():
            continue

        pts2D = corners[i].reshape(-1, 2).astype(np.float32)
        flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(pts3D, pts2D, new_K, None, flags=flag)
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(pts3D, pts2D, new_K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            logger.info(f"{filename}: PnP failed for ID {id_val}")
            continue

        object_pose_list.append({
            "object_name": get_key_from_value(OBJ_ID, id_val),
            "rvec": rvec.reshape(3).tolist(),
            "tvec": tvec.reshape(3).tolist()
        })

    if return_pose_list:
        return object_pose_list

def run_frame_to_pose(
    task: str,
    session_dir: Path,
    marker_size_m: float,
    intrinsics_path: Path,
):
    # choose OBJ_ID
    if task == "kitchen":
        OBJ_ID = {'pink_cup': 310, 'blue_cup': 309}
    elif task == "dining_room":
        OBJ_ID = {'fork': 300, 'knife': 303}
    elif task == "living_room":
        OBJ_ID = {'blue_block': 305, 'green_block': 306, 'red_block': 304}
    else:
        raise ValueError(f"Unknown task: {task}")

    video_dir = session_dir / "raw_videos"
    save_dir = session_dir / "demos/mapping"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- open video list ---
    video_paths = sorted(
        glob.glob(os.path.join(video_dir, "*.mp4")) +
        glob.glob(os.path.join(video_dir, "*.MP4"))
    )

    all_video_results = []  # 這裡存每支影片的結果

    for video_path in video_paths:
        logger.info(f"\nProcessing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            window = total_frames
        else:
            window = int(fps * 1.0)
        end_frame = min(window, total_frames - 1)

        found_tags: dict[str, dict] = {}
        all_found = False

        for frame_index in range(0, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()
            if not success:
                continue

            filename = f"frame_{frame_index:04d}.png"
            object_pose_list = process_and_save_with_axes(
                OBJ_ID,
                frame,
                filename,
                save_dir,
                marker_size_m=marker_size_m,
                return_pose_list=True,
                intrinsics_path=intrinsics_path,
            )

            if not object_pose_list:
                continue

            # accumulate detections for this video
            for entry in object_pose_list:
                found_tags[entry["object_name"]] = entry

            if set(found_tags.keys()) == set(OBJ_ID.keys()):
                logger.info("All tags in OBJ_ID detected in this video.\n")
                all_found = True
                break

        cap.release()

        video_name = os.path.basename(video_path)

        if all_found:
            logger.info(f"[{video_name}] Saved FULL object poses.")
            all_video_results.append({
                "video_name": video_name,
                "objects": list(found_tags.values()),
                "status": "full",
            })
        else:
            if found_tags:
                logger.info(f"[{video_name}] Not all tags detected in same frame. Saving PARTIAL poses.")
                all_video_results.append({
                    "video_name": video_name,
                    "objects": list(found_tags.values()),
                    "status": "partial",
                })
            else:
                logger.info(f"[{video_name}] No tags detected. Skipping this video.")
                all_video_results.append({
                    "video_name": video_name,
                    "objects": [],
                    "status": "none",
                })

    out_json = os.path.join(save_dir, "object_poses.json")
    with open(out_json, "w") as f:
        json.dump(all_video_results, f, indent=4)
    logger.info(f"Saved object poses for {len(all_video_results)} video(s) to {out_json}")



class FrameToPoseService(BaseService):
    """Pipeline service wrapper for frame-to-pose."""

    def __init__(self, config: dict):
        super().__init__(config)

        self.session_dir = Path(self.config["session_dir"])

        self.task = self.config.get("task")
        if self.task is None:
            raise ValueError(
                "FrameToPoseService requires 'task' in config "
                "(kitchen / dining_room / living_room)."
            )

        self.marker_size_m = float(self.config.get("marker_size_m", 0.018))

        intrinsics_cfg = self.config.get(
            "intrinsics_path",
            "defaults/calibration/gopro13_intrinsics_2_7k.json",
        )
        self.intrinsics_path = (ROOT / intrinsics_cfg).resolve()

    def execute(self):
        logger.info("[FrameToPose] Service run() called.")
        run_frame_to_pose(
            task=self.task,
            session_dir=self.session_dir,
            marker_size_m=self.marker_size_m,
            intrinsics_path=self.intrinsics_path,
        )
