import concurrent.futures
import multiprocessing
import subprocess
from pathlib import Path

import av
import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from ..common.cv_util import draw_predefined_mask
from .base_service import BaseService

CREATE_MAP_MODE = "create_map"
BATCH_SLAM_MODE = "batch_slam"


class SLAMMappingService(BaseService):
    """Service for creating SLAM maps using ORB-SLAM3."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.session_dir = self.config.get("session_dir")
        self.docker_image = self.config.get("docker_image", "chicheng/orb_slam3:latest")
        self.timeout_multiple = self.config.get("timeout_multiple", 16)
        self.max_lost_frames = self.config.get("max_lost_frames", 60)
        self.pull_docker = self.config.get("pull_docker", True)
        self.generate_mask = self.config.get("generate_mask", True)
        self.slam_process_mode = self.config.get("slam_process_mode", "slam_mapping")
        self.num_workers = self.config.get("num_workers", multiprocessing.cpu_count() // 2)
        self.force = self.config.get("force", False)

    def execute(self) -> dict:
        if self.slam_process_mode == CREATE_MAP_MODE:
            return self.execute_create_map_slam()
        elif self.slam_process_mode == BATCH_SLAM_MODE:
            return self.execute_slam_batch()
        raise ValueError(f"Unknown mode, only accepts: {CREATE_MAP_MODE}, {BATCH_SLAM_MODE}")

    def execute_create_map_slam(self) -> dict:
        assert self.session_dir, "Missing session_dir from the configuration"

        input_path = Path(self.session_dir) / "demos/mapping"
        for fn in ["raw_video.mp4", "imu_data.json"]:
            assert (input_path / fn).exists()

        map_path = input_path / "map_atlas.osa"
        if map_path.exists() and (not self.force):
            msg = "map_atlas exists, skipping. set 'force' to True to force re-run if needed."
            logger.info(msg)
            return {"msg": msg}

        self._pull_docker_image()
        mask_path = self._generate_mask_file(input_path) if self.generate_mask else None
        mount_target = Path("/data")
        csv_path = mount_target / "mapping_camera_trajectory.csv"
        video_path = mount_target / "raw_video.mp4"
        imu_path = mount_target / "imu_data.json"
        mask_target = mount_target / "slam_mask.png"
        map_mount_source = map_path
        map_mount_target = Path("/map") / map_mount_source.name
        settings_source = Path(__file__).parent.parent / "defaults" / "calibration" / "gopro10_maxlens_fisheye_setting_v1_720_compatible.yaml"
        settings_target = Path("/settings") / settings_source.name
        cmd = [
            "docker",
            "run",
            "--volume",
            f"{input_path.resolve()}:/data",
            "--volume",
            f"{map_mount_source.parent.resolve()}:{map_mount_target.parent}",
            "--volume",
            f"{settings_source.resolve()}:{settings_target}",
            self.docker_image,
            "/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam",
            "--vocabulary",
            "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
            "--setting",
            str(settings_target),
            "--input_video",
            str(video_path),
            "--input_imu_json",
            str(imu_path),
            "--output_trajectory_csv",
            str(csv_path),
            "--save_map",
            str(map_mount_target),
        ]
        logger.info(f"[DOCKER CMD]: {' '.join(cmd)}")
        if not mask_path:
            cmd.extend(["--mask_img", str(mask_target)])
        stdout_path = input_path / "slam_stdout.txt"
        stderr_path = input_path / "slam_stderr.txt"

        logger.info(f"Running SLAM mapping in {input_path}...")

        process = subprocess.Popen(
            cmd,
            cwd=str(input_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for line in iter(process.stdout.readline, ""):
            logger.info(f"SUBPROCESS STDOUT: {line.strip()}")
        for line in iter(process.stderr.readline, ""):
            logger.error(f"SUBPROCESS STDERR: {line.strip()}")

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"SLAM mapping failed. Check logs at {stdout_path} for details.")

        return {
            "map_path": str(map_path),
            "trajectory_csv": str(input_path / "mapping_camera_trajectory.csv"),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    def execute_slam_batch(self):
        assert self.session_dir, "Missing session_dir from the configuration"

        def runner(cmd, cwd, stdout_path, stderr_path, timeout, **kwargs):
            try:
                return subprocess.run(
                    cmd,
                    cwd=str(cwd),
                    stdout=stdout_path.open("w"),
                    stderr=stderr_path.open("w"),
                    timeout=timeout,
                    **kwargs,
                )
            except subprocess.TimeoutExpired as e:
                return e

        input_path = Path(self.session_dir) / "demos"
        input_video_dirs = [x.parent for x in input_path.glob("demo*/raw_video.mp4")]
        input_video_dirs += [x.parent for x in input_path.glob("map*/raw_video.mp4")]
        map_path = input_path / "mapping/map_atlas.osa"

        assert map_path.is_file(), "Missing map_atlas file, ensure the create_map process is executed before."
        self._pull_docker_image()

        processed_videos = []
        all_results = []
        processed_video_dirs = []
        with (
            tqdm(total=len(input_video_dirs)) as pbar,
            concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor,
        ):
            futures = {}
            for video_dir in input_video_dirs:
                video_dir = video_dir.absolute()
                if video_dir.joinpath("camera_trajectory.csv").is_file():
                    logger.warning(f"camera_trajectory.csv already exists, skipping {video_dir.name}")
                    continue

                mount_target = Path("/data")
                csv_path = mount_target / "camera_trajectory.csv"
                video_path = mount_target / "raw_video.mp4"
                json_path = mount_target / "imu_data.json"
                mask_path = mount_target / "slam_mask.png"
                mask_write_path = video_dir / "slam_mask.png"
                with av.open(str(video_dir.joinpath("raw_video.mp4").absolute())) as container:
                    video = container.streams.video[0]
                    duration_sec = float(video.duration * video.time_base)

                timeout = duration_sec * self.timeout_multiple
                slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
                slam_mask = draw_predefined_mask(slam_mask, color=255, mirror=True, gripper=False, finger=True)
                cv2.imwrite(str(mask_write_path.absolute()), slam_mask)
                map_mount_source = map_path
                map_mount_target = Path("/map") / map_mount_source.name
                settings_source = Path(__file__).parent.parent / "defaults" / "calibration" / "gopro10_maxlens_fisheye_setting_v1_720_compatible.yaml"
                settings_target = Path("/settings") / settings_source.name
                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "--volume",
                    f"{video_dir}:/data",
                    "--volume",
                    f"{map_mount_source.parent}:{str(map_mount_target.parent)}",
                    "--volume",
                    f"{settings_source.resolve()}:{settings_target}",
                    self.docker_image,
                    "/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam",
                    "--vocabulary",
                    "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
                    "--setting",
                    str(settings_target),
                    "--input_video",
                    str(video_path),
                    "--input_imu_json",
                    str(json_path),
                    "--output_trajectory_csv",
                    str(csv_path),
                    "--load_map",
                    str(map_mount_target),
                    "--mask_img",
                    str(mask_path),
                    "--max_lost_frames",
                    str(self.max_lost_frames),
                ]

                logger.info(f"[DOCKER CMD]: {' '.join(cmd)}")
                stdout_path = video_dir / "slam_stdout.txt"
                stderr_path = video_dir / "slam_stderr.txt"

                if len(futures) >= self.num_workers:
                    completed, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in completed:
                        result = futures[future]
                        all_results.append(future.result())
                        processed_video_dirs.append(result)
                        pbar.update(1)
                    for future in completed:
                        del futures[future]
                future = executor.submit(runner, cmd, str(video_dir), stdout_path, stderr_path, timeout)
                futures[future] = video_dir
            if futures:
                completed, _ = concurrent.futures.wait(futures)
                for future in completed:
                    result = futures[future]
                    all_results.append(future.result())
                    processed_video_dirs.append(result)
                    pbar.update(1)

        for video_dir, result in zip(processed_video_dirs, all_results):
            status = "success"
            if isinstance(result, subprocess.TimeoutExpired):
                status = "timeout"
            else:
                if getattr(result, "returncode", (-1)) != 0:
                    status = "failed"
            processed_videos.append(
                {
                    "video_dir": str(video_dir),
                    "trajectory_csv": str(video_dir / "camera_trajectory.csv"),
                    "stdout_log": str(video_dir / "slam_stdout.txt"),
                    "stderr_log": str(video_dir / "slam_stderr.txt"),
                    "status": status,
                }
            )

        return {
            "processed_videos": processed_videos,
            "total_processed": len(processed_videos),
        }

    def _pull_docker_image(self):
        """Pull Docker image if required."""
        if self.pull_docker:
            print(f"Pulling docker image {self.docker_image}")
            result = subprocess.run(["docker", "pull", self.docker_image])
            if result.returncode != 0:
                raise RuntimeError(f"Failed to pull docker image: {self.docker_image}")

    def _generate_mask_file(self, input_path: Path) -> Path:
        """Generate mask image for SLAM if enabled."""
        mask_path = input_path / "slam_mask.png"
        slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
        slam_mask = draw_predefined_mask(slam_mask, color=255, mirror=True, gripper=False, finger=True)
        cv2.imwrite(str(mask_path), slam_mask)
        return mask_path
