import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from .base_service import BaseService


class VisualizeSLAMGUI(BaseService):
    """Service for launching ORB-SLAM3 GUI visualization for debugging."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.session_dir = self.config.get("session_dir")
        self.video_path = self.config.get("video_path")
        self.docker_image = self.config.get("docker_image", "chicheng/orb_slam3:latest")
        self.slam_settings_file = self.config.get(
            "slam_settings_file",
            "/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml"
        )
        self.timeout_multiple = self.config.get("timeout_multiple", 10)
        self.pull_docker = self.config.get("pull_docker", True)

        # Always validate GUI setup since we only support GUI mode
        self._validate_gui_setup()

    def execute(self) -> dict:
        """Launch ORB-SLAM3 GUI for debugging specified video."""
        assert self.session_dir, "Missing session_dir from the configuration"
        assert self.video_path, "Missing video_path from the configuration"

        session_path = Path(self.session_dir)
        video_file = Path(self.video_path)

        if not session_path.exists():
            raise FileNotFoundError(f"Session directory does not exist: {session_path}")

        if not video_file.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_file}")

        logger.info(f"Processing video: {video_file}")
        logger.info(f"Session directory: {session_path}")

        # Pull Docker image if required
        self._pull_docker_image()

        # Build and execute Docker command
        docker_cmd = self._build_docker_command(session_path, video_file)

        logger.info(f"Launching ORB-SLAM3 GUI with command: {' '.join(docker_cmd)}")

        # Execute Docker command (synchronous for debugging)
        try:
            process = subprocess.Popen(
                docker_cmd,
                cwd=str(session_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Log output in real-time
            for line in iter(process.stdout.readline, ""):
                logger.info(f"ORB-SLAM3 STDOUT: {line.strip()}")

            for line in iter(process.stderr.readline, ""):
                logger.error(f"ORB-SLAM3 STDERR: {line.strip()}")

            # Wait for completion or allow user to terminate
            process.wait()

            return {
                "status": "completed" if process.returncode == 0 else "failed",
                "session_dir": str(session_path),
                "video_path": str(video_file),
                "return_code": process.returncode,
                "message": "ORB-SLAM3 GUI execution completed" if process.returncode == 0 else "ORB-SLAM3 GUI execution failed"
            }

        except KeyboardInterrupt:
            logger.info("User interrupted ORB-SLAM3 GUI execution")
            return {
                "status": "interrupted",
                "session_dir": str(session_path),
                "video_path": str(video_file),
                "message": "Execution interrupted by user"
            }
        except Exception as e:
            logger.error(f"Failed to launch ORB-SLAM3 GUI: {e}")
            raise RuntimeError(f"Failed to launch ORB-SLAM3 GUI: {e}")

    def _detect_slam_files(self, session_dir: Path) -> Dict[str, Path]:
        """Auto-detect SLAM files in session directory."""
        files = {}

        # Look for trajectory files in subdirectories
        trajectory_files = list(session_dir.glob("**/camera_trajectory.csv"))
        if trajectory_files:
            files["trajectory"] = trajectory_files[0]
            logger.info(f"Found trajectory file: {trajectory_files[0]}")

        # Look for video files
        video_files = list(session_dir.glob("**/raw_video.mp4"))
        if video_files:
            files["video"] = video_files[0]
            logger.info(f"Found video file: {video_files[0]}")

        # Look for map files
        map_files = list(session_dir.glob("**/map_atlas.osa"))
        if map_files:
            files["map"] = map_files[0]
            logger.info(f"Found map file: {map_files[0]}")

        # Look for IMU data files
        imu_files = list(session_dir.glob("**/imu_data.json"))
        if imu_files:
            files["imu"] = imu_files[0]
            logger.info(f"Found IMU file: {imu_files[0]}")

        return files

    def _validate_gui_setup(self):
        """Validate that GUI requirements are met and provide helpful warnings."""
        display_env = os.environ.get("DISPLAY")
        if not display_env:
            raise RuntimeError(
                "DISPLAY environment variable is not set. This service requires GUI mode with X11. "
                "Please run with X11 forwarding (ssh -X) or ensure DISPLAY is set."
            )

        # Check for X11 socket
        x11_socket = "/tmp/.X11-unix"
        if not os.path.exists(x11_socket):
            logger.warning(f"X11 socket not found at {x11_socket}. GUI may not work properly.")

        # Check for Xauthority file
        xauthority_file = os.environ.get("XAUTHORITY", os.path.expanduser("~/.Xauthority"))
        if not os.path.exists(xauthority_file):
            logger.warning(f"Xauthority file not found at {xauthority_file}. GUI authentication may fail.")

        logger.info("GUI validation passed. Docker container will run with X11 forwarding.")

    def _build_docker_command(self, session_path: Path, video_file: Path) -> List[str]:
        """Build Docker command for ORB-SLAM3 GUI with specified video."""
        cmd = ["docker", "run", "--rm"]

        # Add volume mounts for session data and video file
        cmd.extend(["--volume", f"{session_path.resolve()}:/data"])
        cmd.extend(["--volume", f"{video_file.resolve()}:/input/video.mp4"])

        # Add GUI mounts (always enabled)
        cmd.extend([
            "--volume", "/tmp/.X11-unix:/tmp/.X11-unix",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--ipc", "host"
        ])

        display_env = os.environ.get("DISPLAY")
        if display_env:
            cmd.extend(["--env", f"DISPLAY={display_env}"])

            # Set XDG_RUNTIME_DIR with fallback
            xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
            cmd.extend(["--env", f"XDG_RUNTIME_DIR={xdg_runtime_dir}"])

            # Set XAUTHORITY if it exists
            xauthority_file = os.environ.get("XAUTHORITY", os.path.expanduser("~/.Xauthority"))
            if os.path.exists(xauthority_file):
                cmd.extend(["--env", f"XAUTHORITY={xauthority_file}"])

            # Add basic graphics environment variables
            cmd.extend(["--env", "LIBGL_ALWAYS_SOFTWARE=1"])

        # Resolve settings file to absolute path
        settings_file_abs_path = self._resolve_settings_file_path()
        cmd.extend(["--volume", f"{settings_file_abs_path}:/slam_settings.yaml"])

        # Add ORB-SLAM3 command
        cmd.extend([
            self.docker_image,
            "/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam",
            "--vocabulary", "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
            "--setting", "/slam_settings.yaml",
            "--input_video", "/input/video.mp4",
            "--input_imu_json", "/data/imu_data.json",
            "--output_trajectory_csv", "/data/camera_trajectory.csv",
            "--save_map", "/data/map_atlas.osa",
            "--enable_gui"
        ])

        return cmd

    def _resolve_settings_file_path(self) -> Path:
        """Resolve and validate the SLAM settings file path to absolute path."""
        if not self.slam_settings_file:
            raise ValueError("slam_settings_file is not configured")

        settings_path = Path(self.slam_settings_file)

        # Convert to absolute path if it's a relative path
        if not settings_path.is_absolute():
            # Assume relative paths are relative to current working directory
            settings_path = Path.cwd() / settings_path

        # Validate that the file exists
        if not settings_path.exists():
            raise FileNotFoundError(f"SLAM settings file not found: {settings_path}")

        if not settings_path.is_file():
            raise ValueError(f"SLAM settings path is not a file: {settings_path}")

        return settings_path.resolve()

    def _pull_docker_image(self):
        """Pull Docker image if required."""
        if self.pull_docker:
            logger.info(f"Pulling docker image {self.docker_image}")
            result = subprocess.run(["docker", "pull", self.docker_image], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to pull docker image: {self.docker_image}. Error: {result.stderr}")
            logger.info(f"Successfully pulled docker image: {self.docker_image}")
