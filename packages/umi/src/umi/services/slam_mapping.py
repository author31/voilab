import subprocess
from pathlib import Path

import cv2
import numpy as np

from ..common.cv_util import draw_predefined_mask
from .base_service import BaseService


class SLAMMappingService(BaseService):
    """Service for creating SLAM maps using ORB-SLAM3."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.session_dir = self.config.get('session_dir')
        self.docker_image = self.config.get('docker_image', 'chicheng/orb_slam3:latest')
        self.timeout_multiple = self.config.get('timeout_multiple', 16)
        self.max_lost_frames = self.config.get('max_lost_frames', 60)
        self.pull_docker = self.config.get('pull_docker', True)
        self.generate_mask = self.config.get('generate_mask', True)
    
    def execute(self) -> dict:
        """
        Create SLAM map from videos in input directory.
        
        Args:
            input_dir: Directory containing videos and IMU data
            output_dir: Directory for SLAM map outputs
            
        Returns:
            dict: Mapping results with paths to map files
        """
        assert self.session_dir, "Missing session_dir from the configuration"
        input_path = Path(self.session_dir)/"demos/mapping"

        for fn in ['raw_video.mp4', 'imu_data.json']:
            assert (input_path / fn).exists()

        map_path = input_path/"map_atlas.osa"
        self._pull_docker_image()
        
        # Generate mask if needed
        mask_path = self._generate_mask_file(input_path) if self.generate_mask else None

        # Build docker command
        cmd = self._build_docker_command(input_path, map_path, mask_path)

        # Output logs
        stdout_path = input_path / 'slam_stdout.txt'
        stderr_path = input_path / 'slam_stderr.txt'


        print(f"Running SLAM mapping in {input_path}...")
        result = subprocess.run(
            cmd,
            cwd=str(input_path),
            stdout=stdout_path.open('w'),
            stderr=stderr_path.open('w')
        )

        if result.returncode != 0:
            raise RuntimeError(f"SLAM mapping failed. Check logs at {stdout_path} for details.")
       

        return {
            "map_path": str(map_path),
            "trajectory_csv": str(input_path / 'mapping_camera_trajectory.csv'),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    def _validate_inputs(self, input_path: Path):
        """Ensure input directory has required files."""
        for fn in ['raw_video.mp4', 'imu_data.json']:
            file_path = input_path / fn
            if not file_path.exists():
                raise FileNotFoundError(f"Missing required file: {file_path}")


    def _pull_docker_image(self):
        """Pull Docker image if required."""
        if self.pull_docker:
            print(f"Pulling docker image {self.docker_image}")
            result = subprocess.run(['docker', 'pull', self.docker_image])
            if result.returncode != 0:
                raise RuntimeError(f"Failed to pull docker image: {self.docker_image}")

    def _generate_mask_file(self, input_path: Path) -> Path:
        """Generate mask image for SLAM if enabled."""
        mask_path = input_path / 'slam_mask.png'
        slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
        slam_mask = draw_predefined_mask(
            slam_mask, color=255, mirror=True, gripper=False, finger=True
        )
        cv2.imwrite(str(mask_path), slam_mask)
        return mask_path

    def _build_docker_command(self, input_path: Path, map_path: Path, mask_path: Path | None = None):
        """Build the docker command for SLAM execution."""
        mount_target = Path('/data')
        csv_path = mount_target / 'mapping_camera_trajectory.csv'
        video_path = mount_target / 'raw_video.mp4'
        json_path = mount_target / 'imu_data.json'
        mask_target = mount_target / 'slam_mask.png'

        map_mount_source = map_path
        map_mount_target = Path('/map') / map_mount_source.name

        cmd = [
            'docker', 'run', '--rm',
            '--volume', f'{input_path}:/data',
            '--volume', f'{map_mount_source.parent}:/map',
            self.docker_image,
            '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
            '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
            '--setting', '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml',
            '--input_video', str(video_path),
            '--input_imu_json', str(json_path),
            '--output_trajectory_csv', str(csv_path),
            '--save_map', str(map_mount_target)
        ]
        if mask_path:
            cmd.extend(['--mask_img', str(mask_target)])

        return cmd
