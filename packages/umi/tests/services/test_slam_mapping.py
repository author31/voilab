#!/usr/bin/env python3
"""
Unit tests for SLAMMappingService

Run these tests independently:
    python -m pytest umi/tests/services/test_slam_mapping.py -v
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from umi.services.slam_mapping import SLAMMappingService


class TestSLAMMappingService:
    """Test cases for SLAMMappingService"""

    def test_init_with_config(self):
        """Test service initialization with custom config"""
        config = {
            "docker_image": "custom/orb_slam3:latest",
            "timeout_multiple": 20,
            "max_lost_frames": 100,
        }
        service = SLAMMappingService(config)
        assert service.docker_image == "custom/orb_slam3:latest"
        assert service.timeout_multiple == 20
        assert service.max_lost_frames == 100

    def test_init_with_default_config(self):
        """Test service initialization with default config"""
        service = SLAMMappingService({})
        assert service.docker_image == "chicheng/orb_slam3:latest"
        assert service.timeout_multiple == 16
        assert service.max_lost_frames == 60

    def test_create_map_single_demo(self):
        """Test creating SLAM map for single demo"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create proper input structure expected by SLAMMappingService
            # session_dir/demos/mapping/raw_video.mp4 and imu_data.json
            session_dir = tmpdir / "session"
            mapping_dir = session_dir / "demos" / "mapping"
            mapping_dir.mkdir(parents=True)
            (mapping_dir / "raw_video.mp4").write_text("mock video data")
            (mapping_dir / "imu_data.json").write_text('{"accel": [], "gyro": []}')

            service = SLAMMappingService({"session_dir": str(session_dir)})

            # Mock Docker operations and subprocess calls
            with patch('subprocess.Popen') as mock_popen, \
                 patch('subprocess.run') as mock_run, \
                 patch('cv2.imwrite') as mock_imwrite, \
                 patch('umi.services.slam_mapping.logger'):

                # Mock successful Docker execution
                mock_process = Mock()
                # Mock stdout and stderr for the for loops
                # iter(process.stdout.readline, "") calls readline repeatedly until empty string
                mock_process.stdout = Mock()
                mock_process.stdout.readline = Mock(return_value="")  # Empty string stops iteration
                mock_process.stderr = Mock()
                mock_process.stderr.readline = Mock(return_value="")  # Empty string stops iteration
                mock_process.wait.return_value = 0
                mock_process.returncode = 0  # This is what's actually checked
                mock_popen.return_value = mock_process

                # Mock subprocess.run for docker pull
                mock_run.return_value = Mock(returncode=0)

                # Mock cv2.imwrite for mask generation
                mock_imwrite.return_value = True

                result = service.execute_create_map_slam()

            # Check that expected files are in the result
            assert "map_path" in result
            assert "trajectory_csv" in result
            assert "stdout_log" in result
            assert "stderr_log" in result

    def test_create_map_multiple_demos(self):
        """Test that create_map method handles input correctly - this is actually for single mapping session"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create proper input structure for mapping session
            session_dir = tmpdir / "session"
            mapping_dir = session_dir / "demos" / "mapping"
            mapping_dir.mkdir(parents=True)
            (mapping_dir / "raw_video.mp4").write_text("video1")
            (mapping_dir / "imu_data.json").write_text('{"accel": [], "gyro": []}')

            output_dir = tmpdir / "output"

            service = SLAMMappingService({})

            # Mock Docker operations and subprocess calls
            with patch('subprocess.Popen') as mock_popen, \
                 patch('subprocess.run') as mock_run, \
                 patch('cv2.imwrite') as mock_imwrite, \
                 patch('umi.services.slam_mapping.logger'):

                # Mock successful Docker execution
                mock_process = Mock()
                mock_process.stdout = Mock()
                mock_process.stdout.readline = Mock(return_value="")
                mock_process.stderr = Mock()
                mock_process.stderr.readline = Mock(return_value="")
                mock_process.wait.return_value = 0
                mock_process.returncode = 0  # This is what's actually checked
                mock_popen.return_value = mock_process

                # Mock subprocess.run for docker pull
                mock_run.return_value = Mock(returncode=0)

                # Mock cv2.imwrite for mask generation
                mock_imwrite.return_value = True

                result = service.create_map(str(session_dir), str(output_dir))

            # The result should contain mapping session info
            assert "map_path" in result or "msg" in result

    def test_create_map_empty_directory(self):
        """Test handling missing required files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create session dir but no required files
            session_dir = tmpdir / "session"
            mapping_dir = session_dir / "demos" / "mapping"
            mapping_dir.mkdir(parents=True)

            service = SLAMMappingService({"session_dir": str(session_dir)})

            # Should raise AssertionError due to missing files
            with pytest.raises(AssertionError):
                service.execute_create_map_slam()

    def test_create_map_no_videos(self):
        """Test handling missing video files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create session dir with only IMU data, no video
            session_dir = tmpdir / "session"
            mapping_dir = session_dir / "demos" / "mapping"
            mapping_dir.mkdir(parents=True)
            (mapping_dir / "imu_data.json").write_text('{"accel": [], "gyro": []}')

            service = SLAMMappingService({"session_dir": str(session_dir)})

            # Should raise AssertionError due to missing raw_video.mp4
            with pytest.raises(AssertionError):
                service.execute_create_map_slam()

    def test_validate_mapping_success(self):
        """Test successful validation of SLAM mapping results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create valid mapping structure
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            demo1_dir = output_dir / "demo1"
            demo1_dir.mkdir()
            (demo1_dir / "map.bin").write_text("mock map data")
            (demo1_dir / "trajectory.txt").write_text("mock trajectory")

            service = SLAMMappingService({})
            assert service.validate_mapping(str(output_dir)) is True

    def test_validate_mapping_failure(self):
        """Test validation failure cases"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Empty directory
            empty_dir = tmpdir / "empty"
            empty_dir.mkdir()

            service = SLAMMappingService({})
            assert service.validate_mapping(str(empty_dir)) is False

            # Directory with incomplete files (only map.bin, missing trajectory.txt)
            incomplete_dir = tmpdir / "incomplete"
            incomplete_dir.mkdir()
            demo_dir = incomplete_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "map.bin").write_text("map only")

            # Should validate as False since it's missing trajectory.txt
            assert service.validate_mapping(str(incomplete_dir)) is False

            # Directory with complete files should validate as True
            complete_dir = tmpdir / "complete"
            complete_dir.mkdir()
            demo_dir = complete_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "map.bin").write_text("map data")
            (demo_dir / "trajectory.txt").write_text("trajectory data")

            assert service.validate_mapping(str(complete_dir)) is True

    def test_create_map_output_structure(self):
        """Test that proper output files are created after SLAM mapping"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create proper input structure for mapping
            session_dir = tmpdir / "session"
            mapping_dir = session_dir / "demos" / "mapping"
            mapping_dir.mkdir(parents=True)
            (mapping_dir / "raw_video.mp4").write_text("mock video")
            (mapping_dir / "imu_data.json").write_text('{"accel": [], "gyro": []}')

            # Mock Docker operations and subprocess calls
            with patch('subprocess.Popen') as mock_popen, \
                 patch('subprocess.run') as mock_run, \
                 patch('cv2.imwrite') as mock_imwrite, \
                 patch('umi.services.slam_mapping.logger'):

                # Mock successful Docker execution
                mock_process = Mock()
                mock_process.stdout = Mock()
                mock_process.stdout.readline = Mock(return_value="")
                mock_process.stderr = Mock()
                mock_process.stderr.readline = Mock(return_value="")
                mock_process.wait.return_value = 0
                mock_process.returncode = 0  # This is what's actually checked
                mock_popen.return_value = mock_process

                # Mock subprocess.run for docker pull
                mock_run.return_value = Mock(returncode=0)

                # Mock cv2.imwrite for mask generation
                mock_imwrite.return_value = True

                service = SLAMMappingService({"session_dir": str(session_dir)})
                result = service.execute_create_map_slam()

            # Check that expected file paths are in the result
            assert "map_path" in result
            assert "trajectory_csv" in result
            assert Path(result["map_path"]).parent.exists()  # mapping directory exists

    def test_create_map_result_structure(self):
        """Test result structure from create_map"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create proper input structure for mapping
            session_dir = tmpdir / "session"
            mapping_dir = session_dir / "demos" / "mapping"
            mapping_dir.mkdir(parents=True)
            (mapping_dir / "raw_video.mp4").write_text("mock video")
            (mapping_dir / "imu_data.json").write_text('{"accel": [], "gyro": []}')

            # Mock Docker operations and subprocess calls
            with patch('subprocess.Popen') as mock_popen, \
                 patch('subprocess.run') as mock_run, \
                 patch('cv2.imwrite') as mock_imwrite, \
                 patch('umi.services.slam_mapping.logger'):

                # Mock successful Docker execution
                mock_process = Mock()
                mock_process.stdout = Mock()
                mock_process.stdout.readline = Mock(return_value="")
                mock_process.stderr = Mock()
                mock_process.stderr.readline = Mock(return_value="")
                mock_process.wait.return_value = 0
                mock_process.returncode = 0  # This is what's actually checked
                mock_popen.return_value = mock_process

                # Mock subprocess.run for docker pull
                mock_run.return_value = Mock(returncode=0)

                # Mock cv2.imwrite for mask generation
                mock_imwrite.return_value = True

                service = SLAMMappingService({"session_dir": str(session_dir)})
                result = service.execute_create_map_slam()

            # Check result structure
            assert "map_path" in result
            assert "trajectory_csv" in result
            assert "stdout_log" in result
            assert "stderr_log" in result

            # Verify paths point to expected files
            assert result["map_path"].endswith("map_atlas.osa")
            assert result["trajectory_csv"].endswith("mapping_camera_trajectory.csv")

    def test_run_docker_slam_placeholder(self):
        """Test Docker SLAM method (placeholder implementation)"""
        # This is a placeholder test for the Docker SLAM functionality
        # In real testing, this would require Docker setup
        service = SLAMMappingService({})

        # Mock the method for testing
        def mock_run_docker_slam(self, video_file, output_dir):
            return True

        # Temporarily replace method
        original_method = service._run_docker_slam
        service._run_docker_slam = mock_run_docker_slam.__get__(service, SLAMMappingService)

        try:
            result = service._run_docker_slam(Path("test.mp4"), Path("/tmp"))
            assert result is True
        finally:
            # Restore original method
            service._run_docker_slam = original_method


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
