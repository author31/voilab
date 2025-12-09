import pickle
import cv2
from cv2 import aruco
import numpy as np
import json
from pathlib import Path
from loguru import logger
from .base_service import BaseService


def get_key_from_value(d, value):
    return next(k for k, v in d.items() if v == value)


ROOT = Path(__file__).resolve().parents[3]
intrinsics_path = ROOT / "defaults" / "calibration" / "gopro13_intrinsics_2_7k.json"

# REGISTRY: Maps task names to their corresponding object ID configurations
# Add new tasks here for future expansion
REGISTRY = {
    "kitchen": {
        "pink_cup": 310,
        "blue_cup": 309,
    },
    "dining_room": {
        "fork": 300,
        "knife": 303,
        "plate": 302,
    },
    "living_room": {
        "blue_block": 305,
        "green_block": 306,
        "red_block": 304,
    },
}


def process_frame_for_poses(
    OBJ_ID: dict,
    frame: np.ndarray,
    filename: str,
    K: np.ndarray,
    D_fish: np.ndarray,
    marker_size_m: float = 0.018,
    tx_slam_tag: np.ndarray = None,
):
    """
    Process a single frame to detect ArUco markers and estimate object poses.
    
    Args:
        OBJ_ID: Dictionary mapping object names to marker IDs
        frame: Input frame to process
        filename: Filename for logging
        K: Camera intrinsics matrix, shape (3, 3), dtype float64
        D_fish: Fisheye distortion coefficients, shape (4,), dtype float64
        marker_size_m: Size of ArUco markers in meters
        tx_slam_tag: Transform from SLAM tag frame to camera frame, shape (4, 4), dtype float64
    
    Returns:
        list: List of detected object poses [{object_name, rvec, tvec}, ...]
    """

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
        logger.debug(f"{filename}: detected none")
        return []

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
            logger.debug(f"{filename}: PnP failed for ID {id_val}")
            continue

        # ---------------------------
        # camera frame -> SLAM tag frame
        # ---------------------------
        R_cam_obj, _ = cv2.Rodrigues(rvec)
        T_cam_obj = np.eye(4, dtype=np.float64)
        T_cam_obj[:3, :3] = R_cam_obj
        T_cam_obj[:3, 3] = tvec.reshape(3)

        T_slamTag_cam = tx_slam_tag
        T_out = T_slamTag_cam @ T_cam_obj

        R_out = T_out[:3, :3]
        t_out = T_out[:3, 3]
        rvec_out, _ = cv2.Rodrigues(R_out)

        object_pose_list.append({
            "object_name": get_key_from_value(OBJ_ID, id_val),
            "rvec": rvec_out.reshape(3).tolist(),
            "tvec": t_out.reshape(3).tolist(),
        })

    return object_pose_list


def run_frame_to_pose_from_plan(
    task: str,
    session_dir: Path,
    marker_size_m: float,
    intrinsics_path: Path,
    dataset_plan_filename: str,
):
    """
    Run frame-to-pose extraction using dataset_plan.pkl.
    
    For each episode in the plan, processes all camera segments and detects
    object poses. Stops processing an episode once all expected tags are found.
    
    Outputs object_poses.json with schema:
    {
        "video_name": str,
        "episode_range": [global_start_frame, global_end_frame],
        "objects": [{object_name, rvec, tvec}, ...],
        "status": "full" | "partial" | "none"
    }
    """
    # Choose OBJ_ID from REGISTRY
    if task not in REGISTRY:
        raise ValueError(f"Unknown task: {task}. Available tasks: {list(REGISTRY.keys())}")
    
    OBJ_ID = REGISTRY[task]

    demos_dir = session_dir / "demos"
    save_dir = session_dir / "demos/mapping"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- load camera intrinsics from JSON (once for all frames) ---
    with open(intrinsics_path, "r") as f:
        data = json.load(f)

    intr = data["intrinsics"]

    fx = intr["focal_length"]
    aspect_ratio = intr["aspect_ratio"]
    fy = fx * aspect_ratio
    cx = intr["principal_pt_x"]
    cy = intr["principal_pt_y"]

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    D_fish = np.array([
        intr["radial_distortion_1"],
        intr["radial_distortion_2"],
        intr["radial_distortion_3"],
        intr["radial_distortion_4"],
    ], dtype=np.float64)

    # --- load SLAM tag transform ---
    tx_slam_tag_path = save_dir / "tx_slam_tag.json"
    if not tx_slam_tag_path.is_file():
        raise FileNotFoundError(f"tx_slam_tag.json not found at {tx_slam_tag_path}")

    with open(tx_slam_tag_path, "r") as f:
        tx_data = json.load(f)

    tx_slam_tag = np.array(tx_data["tx_slam_tag"], dtype=np.float64).reshape(4, 4)

    # --- load dataset plan ---
    plan_path = session_dir / dataset_plan_filename
    if not plan_path.is_file():
        raise FileNotFoundError(f"Dataset plan not found at {plan_path}")

    with open(plan_path, "rb") as f:
        plan = pickle.load(f)

    logger.info(f"Loaded dataset plan with {len(plan)} episodes")
    logger.info(f"Task: {task}, expected objects: {list(OBJ_ID.keys())}")

    all_episode_results = []
    global_frame_start = 0

    for episode_idx, plan_episode in enumerate(plan):
        cameras = plan_episode["cameras"]
        
        # Determine frame count for this episode (all cameras should have same frame count)
        n_frames = None
        for camera in cameras:
            video_start, video_end = camera["video_start_end"]
            cam_frames = video_end - video_start
            if n_frames is None:
                n_frames = cam_frames
            else:
                assert n_frames == cam_frames, f"Inconsistent frame counts in episode {episode_idx}"

        if n_frames is None:
            logger.warning(f"Skipping episode {episode_idx} because cameras list is empty.")
            continue
        global_frame_end = global_frame_start + n_frames
        episode_range = [global_frame_start, global_frame_end]

        logger.info(f"\nProcessing episode {episode_idx + 1}/{len(plan)}, "
                   f"global frames [{global_frame_start}, {global_frame_end})")

        # Track found tags for this episode across all cameras
        found_tags: dict[str, dict] = {}
        all_found = False
        video_names = []

        # Process each camera segment in the episode
        for cam_idx, camera in enumerate(cameras):
            if all_found:
                break

            video_path_rel = camera["video_path"]
            video_path = demos_dir.joinpath(video_path_rel).absolute()
            
            if not video_path.is_file():
                logger.warning(f"Video not found: {video_path}")
                continue

            video_name = video_path.name
            if video_name not in video_names:
                video_names.append(video_name)

            video_start, video_end = camera["video_start_end"]
            
            logger.info(f"  Camera {cam_idx}: {video_name} frames [{video_start}, {video_end})")

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                continue

            # Process frames in the planned range
            for frame_idx in range(video_start, video_end):
                if all_found:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                if not success:
                    continue

                filename = f"ep{episode_idx}_cam{cam_idx}_frame{frame_idx}"
                object_pose_list = process_frame_for_poses(
                    OBJ_ID,
                    frame,
                    filename,
                    K,
                    D_fish,
                    marker_size_m=marker_size_m,
                    tx_slam_tag=tx_slam_tag,
                )

                if not object_pose_list:
                    continue

                # Accumulate detections for this episode
                for entry in object_pose_list:
                    found_tags[entry["object_name"]] = entry

                # Check if all tags found
                if set(found_tags.keys()) == set(OBJ_ID.keys()):
                    logger.info(f"  All tags detected in episode {episode_idx + 1}")
                    all_found = True
                    break

            cap.release()

        # Determine status and build result
        video_name_str = ",".join(video_names) if video_names else "unknown"
        
        if all_found:
            status = "full"
            logger.info(f"[Episode {episode_idx + 1}] FULL - all object poses found")
        elif found_tags:
            status = "partial"
            logger.info(f"[Episode {episode_idx + 1}] PARTIAL - found: {list(found_tags.keys())}")
        else:
            status = "none"
            logger.info(f"[Episode {episode_idx + 1}] NONE - no tags detected")

        all_episode_results.append({
            "video_name": video_name_str,
            "episode_range": episode_range,
            "objects": list(found_tags.values()),
            "status": status,
        })

        # Update global frame counter
        global_frame_start = global_frame_end

    # Save results
    out_json = save_dir / "object_poses.json"
    with open(out_json, "w") as f:
        json.dump(all_episode_results, f, indent=4)
    
    logger.info(f"\nSaved object poses for {len(all_episode_results)} episode(s) to {out_json}")
    
    # Summary statistics
    full_count = sum(1 for r in all_episode_results if r["status"] == "full")
    partial_count = sum(1 for r in all_episode_results if r["status"] == "partial")
    none_count = sum(1 for r in all_episode_results if r["status"] == "none")
    logger.info(f"Summary: {full_count} full, {partial_count} partial, {none_count} none")

    return all_episode_results


class FrameToPoseService(BaseService):
    """Pipeline service wrapper for frame-to-pose using dataset_plan.pkl."""

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
        
        self.dataset_plan_filename = self.config.get(
            "dataset_plan_filename", "dataset_plan.pkl"
        )

        intrinsics_cfg = self.config.get(
            "intrinsics_path",
            "defaults/calibration/gopro13_intrinsics_2_7k.json",
        )
        self.intrinsics_path = (ROOT / intrinsics_cfg).resolve()

    def execute(self):
        logger.info("[FrameToPose] Service execute() called.")
        results = run_frame_to_pose_from_plan(
            task=self.task,
            session_dir=self.session_dir,
            marker_size_m=self.marker_size_m,
            intrinsics_path=self.intrinsics_path,
            dataset_plan_filename=self.dataset_plan_filename,
        )
        return {
            "status": "success",
            "num_episodes": len(results),
            "full_count": sum(1 for r in results if r["status"] == "full"),
            "partial_count": sum(1 for r in results if r["status"] == "partial"),
            "none_count": sum(1 for r in results if r["status"] == "none"),
        }
