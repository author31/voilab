#!/usr/bin/env python3
"""
Unit tests for VisualizeSLAMGUI

Run these tests independently:
    python -m pytest umi/tests/services/test_visualize_slam_gui.py -v
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from umi.services.visualize_slam_gui import VisualizeSLAMGUI


class TestVisualizeSLAMGUI:
    """Test cases for VisualizeSLAMGUI"""

    def test_init_with_config(self):
        """Test service initialization with custom config"""
        config = {
            "session_dir": "/test/session",
            "video_path": "/test/video.mp4",
            "docker_image": "custom/orb_slam3:latest",
            "slam_settings_file": "/custom/settings.yaml",
            "timeout_multiple": 15,
            "pull_docker": False,
        }
        service = VisualizeSLAMGUI(config)
        assert service.session_dir == "/test/session"
        assert service.video_path == "/test/video.mp4"
        assert service.docker_image == "custom/orb_slam3:latest"
        assert service.slam_settings_file == "/custom/settings.yaml"
        assert service.timeout_multiple == 15
        assert service.pull_docker == False

    def test_init_with_default_config(self):
        """Test service initialization with default config"""
        service = VisualizeSLAMGUI({})
        assert service.session_dir is None
        assert service.video_path is None
        assert service.docker_image == "chicheng/orb_slam3:latest"
        assert service.slam_settings_file == "/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml"
        assert service.timeout_multiple == 10
        assert service.pull_docker == True

    def test_detect_slam_files(self):
        """Test auto-detection of SLAM files in session directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            session_dir = tmpdir / "session"
            demo_dir = session_dir / "demos" / "demo1"
            demo_dir.mkdir(parents=True)

            # Create mock SLAM files
            (demo_dir / "camera_trajectory.csv").write_text("frame_idx,timestamp,x,y,z\n1,1.0,0.0,0.0,0.0")
            (demo_dir / "raw_video.mp4").write_text("mock video data")
            (demo_dir / "imu_data.json").write_text('{"accel": [], "gyro": []}')
            (demo_dir / "map_atlas.osa").write_text("mock map data")

            service = VisualizeSLAMGUI({"session_dir": str(session_dir)})
            files = service._detect_slam_files(session_dir)

            assert "trajectory" in files
            assert "video" in files
            assert "imu" in files
            assert "map" in files
            assert files["trajectory"].name == "camera_trajectory.csv"
            assert files["video"].name == "raw_video.mp4"

    def test_detect_slam_files_empty_directory(self):
        """Test file detection with no SLAM files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            session_dir = tmpdir / "empty_session"
            session_dir.mkdir()

            service = VisualizeSLAMGUI({"session_dir": str(session_dir)})
            files = service._detect_slam_files(session_dir)

            assert files == {}

    def test_resolve_settings_file_path_absolute(self):
        """Test resolving absolute settings file path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "test_settings.yaml"
            settings_file.write_text("mock settings")

            service = VisualizeSLAMGUI({"slam_settings_file": str(settings_file)})
            resolved_path = service._resolve_settings_file_path()

            assert resolved_path == settings_file.resolve()

    def test_resolve_settings_file_path_relative(self):
        """Test resolving relative settings file path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            settings_file = tmpdir / "test_settings.yaml"
            settings_file.write_text("mock settings")

            # Change to temp directory for relative path test
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                service = VisualizeSLAMGUI({"slam_settings_file": "test_settings.yaml"})
                resolved_path = service._resolve_settings_file_path()
                assert resolved_path == settings_file.resolve()
            finally:
                os.chdir(original_cwd)

    def test_resolve_settings_file_path_not_found(self):
        """Test resolving non-existent settings file path"""
        service = VisualizeSLAMGUI({"slam_settings_file": "/nonexistent/path.yaml"})

        with pytest.raises(FileNotFoundError, match="SLAM settings file not found"):
            service._resolve_settings_file_path()

    @patch.dict(os.environ, {"DISPLAY": ":0"})
    def test_validate_gui_setup_success(self):
        """Test GUI validation with proper setup"""
        # Mock X11 socket and Xauthority file existence
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            service = VisualizeSLAMGUI({})
            # Should not raise any exception
            assert service is not None

    def test_validate_gui_setup_no_display(self):
        """Test GUI validation failure when DISPLAY is not set"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="DISPLAY environment variable is not set"):
                VisualizeSLAMGUI({})

    def test_build_docker_command(self):
        """Test building Docker command with GUI (always enabled)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            session_dir = tmpdir / "session"
            session_dir.mkdir(parents=True)

            video_file = tmpdir / "video.mp4"
            video_file.write_text("mock video data")

            settings_file = tmpdir / "settings.yaml"
            settings_file.write_text("mock settings")

            with patch.dict(os.environ, {"DISPLAY": ":0", "XAUTHORITY": "/tmp/.Xauthority"}):
                with patch("os.path.exists", return_value=True):
                    with patch("os.getuid", return_value=1000), patch("os.getgid", return_value=1000):
                        service = VisualizeSLAMGUI({
                            "session_dir": str(session_dir),
                            "video_path": str(video_file),
                            "docker_image": "test/orb_slam3:latest",
                            "slam_settings_file": str(settings_file)
                        })

                        cmd = service._build_docker_command(session_dir, video_file)

                        # Check basic Docker command structure
                        assert "docker" in cmd
                        assert "run" in cmd
                        assert "--rm" in cmd
                        assert "test/orb_slam3:latest" in cmd
                        assert "--enable_gui" in cmd

                        # Check GUI-related mounts and environment (always included)
                        assert "--volume" in cmd
                        assert "--user" in cmd
                        assert "--ipc" in cmd
                        assert "DISPLAY=:0" in cmd

                        # Check video volume mount
                        assert f"{video_file.resolve()}:/input/video.mp4" in cmd

                        # Check ORB-SLAM3 specific arguments
                        assert "--input_video" in cmd
                        assert "--output_trajectory_csv" in cmd
                        assert "--save_map" in cmd

    @patch('subprocess.run')
    def test_pull_docker_image_success(self, mock_run):
        """Test successful Docker image pull"""
        mock_run.return_value = Mock(returncode=0)

        service = VisualizeSLAMGUI({"docker_image": "test/orb_slam3:latest", "pull_docker": True})
        service._pull_docker_image()

        mock_run.assert_called_once_with(["docker", "pull", "test/orb_slam3:latest"])

    @patch('subprocess.run')
    def test_pull_docker_image_failure(self, mock_run):
        """Test Docker image pull failure"""
        mock_run.return_value = Mock(returncode=1, stderr="Error pulling image")

        service = VisualizeSLAMGUI({"docker_image": "test/orb_slam3:latest", "pull_docker": True})

        with pytest.raises(RuntimeError, match="Failed to pull docker image"):
            service._pull_docker_image()

    @patch('subprocess.run')
    def test_pull_docker_image_disabled(self, mock_run):
        """Test when Docker image pull is disabled"""
        service = VisualizeSLAMGUI({"docker_image": "test/orb_slam3:latest", "pull_docker": False})
        service._pull_docker_image()

        # Should not attempt to pull
        mock_run.assert_not_called()

    def test_execute_no_session_dir(self):
        """Test execute failure when session_dir is not provided"""
        service = VisualizeSLAMGUI({})

        with pytest.raises(AssertionError, match="Missing session_dir"):
            service.execute()

    def test_execute_no_video_path(self):
        """Test execute failure when video_path is not provided"""
        service = VisualizeSLAMGUI({"session_dir": "/test"})

        with pytest.raises(AssertionError, match="Missing video_path"):
            service.execute()

    def test_execute_session_dir_not_found(self):
        """Test execute failure when session directory doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = Path(tmpdir) / "video.mp4"
            video_file.write_text("mock video")
            service = VisualizeSLAMGUI({"session_dir": "/nonexistent/path", "video_path": str(video_file)})

            with pytest.raises(FileNotFoundError, match="Session directory does not exist"):
                service.execute()

    def test_execute_video_not_found(self):
        """Test execute failure when video file doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            service = VisualizeSLAMGUI({"session_dir": str(tmpdir), "video_path": "/nonexistent/video.mp4"})

            with pytest.raises(FileNotFoundError, match="Video file does not exist"):
                service.execute()

    @patch('umi.services.visualize_slam_gui.subprocess.Popen')
    @patch('umi.services.visualize_slam_gui.subprocess.run')
    def test_execute_success(self, mock_run, mock_popen):
        """Test successful service execution"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock files
            video_file = tmpdir / "video.mp4"
            video_file.write_text("mock video data")

            settings_file = tmpdir / "settings.yaml"
            settings_file.write_text("mock settings")

            # Mock Docker pull
            mock_run.return_value = Mock(returncode=0)

            # Mock Docker process
            mock_process = Mock()
            mock_process.stdout = []
            mock_process.stderr = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            # Mock GUI environment to avoid X11 validation
            with patch.dict(os.environ, {"DISPLAY": ":0"}):
                with patch("os.path.exists", return_value=True):
                    service = VisualizeSLAMGUI({
                        "session_dir": str(tmpdir),
                        "video_path": str(video_file),
                        "slam_settings_file": str(settings_file),
                        "pull_docker": True
                    })

                    result = service.execute()

                    assert result["status"] == "completed"
                    assert result["session_dir"] == str(tmpdir)
                    assert result["video_path"] == str(video_file)
                    assert result["return_code"] == 0

    @patch('umi.services.visualize_slam_gui.subprocess.Popen')
    @patch('umi.services.visualize_slam_gui.subprocess.run')
    def test_execute_keyboard_interrupt(self, mock_run, mock_popen):
        """Test service execution with keyboard interrupt"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock files
            video_file = tmpdir / "video.mp4"
            video_file.write_text("mock video data")

            settings_file = tmpdir / "settings.yaml"
            settings_file.write_text("mock settings")

            # Mock Docker pull
            mock_run.return_value = Mock(returncode=0)

            # Mock Docker process that raises KeyboardInterrupt
            mock_popen.side_effect = KeyboardInterrupt()

            # Mock GUI environment to avoid X11 validation
            with patch.dict(os.environ, {"DISPLAY": ":0"}):
                with patch("os.path.exists", return_value=True):
                    service = VisualizeSLAMGUI({
                        "session_dir": str(tmpdir),
                        "video_path": str(video_file),
                        "slam_settings_file": str(settings_file),
                        "pull_docker": True
                    })

                    result = service.execute()

                    assert result["status"] == "interrupted"
                    assert result["video_path"] == str(video_file)
                    assert "interrupted by user" in result["message"]