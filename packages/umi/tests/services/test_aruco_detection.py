#!/usr/bin/env python3
"""
Unit tests for ArucoDetectionService

Run these tests independently:
    python -m pytest umi/tests/services/test_aruco_detection.py -v
"""

import pytest
import tempfile
import json
import numpy as np
from pathlib import Path

from umi.services.aruco_detection import ArucoDetectionService


class TestArucoDetectionService:
    """Test cases for ArucoDetectionService"""

    def test_init_with_config(self):
        """Test service initialization with custom config"""
        config = {
            "num_workers": 4,
            "camera_intrinsics_path": None,
            "aruco_config_path": None,
        }
        service = ArucoDetectionService(config)
        assert service.num_workers == 4
        assert service.camera_intrinsics_path is None
        assert service.aruco_config_path is None

    def test_init_with_default_config(self):
        """Test service initialization with default config"""
        service = ArucoDetectionService({})
        assert service.num_workers is not None  # Auto-detected
        assert service.camera_intrinsics_path is None
        assert service.aruco_config_path is None

    def test_load_camera_intrinsics_default(self):
        """Test loading default camera intrinsics"""
        service = ArucoDetectionService({})
        camera_matrix, dist_coeffs = service._load_camera_intrinsics()

        assert camera_matrix.shape == (3, 3)
        assert dist_coeffs.shape == (5,)
        assert np.allclose(camera_matrix, np.eye(3))
        assert np.allclose(dist_coeffs, np.zeros(5))

    def test_load_camera_intrinsics_custom(self):
        """Test loading custom camera intrinsics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock calibration file
            calib_file = tmpdir / "calibration.json"
            calib_data = {
                "camera_matrix": [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]],
                "dist_coeffs": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
            calib_file.write_text(json.dumps(calib_data))

            config = {"camera_intrinsics_path": str(calib_file)}
            service = ArucoDetectionService(config)
            camera_matrix, dist_coeffs = service._load_camera_intrinsics()

            assert camera_matrix.shape == (3, 3)
            assert dist_coeffs.shape == (5,)
            assert not np.allclose(camera_matrix, np.eye(3))

    def test_load_aruco_config_default(self):
        """Test loading default ArUco configuration"""
        service = ArucoDetectionService({})
        aruco_dict, aruco_params = service._load_aruco_config()

        assert aruco_dict is not None
        assert aruco_params is not None

    def test_load_aruco_config_custom(self):
        """Test loading custom ArUco configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock config file
            config_file = tmpdir / "aruco_config.json"
            config_data = {
                "parameters": {
                    "adaptiveThreshConstant": 7,
                    "adaptiveThreshWinSizeMin": 3,
                    "adaptiveThreshWinSizeMax": 23,
                }
            }
            config_file.write_text(json.dumps(config_data))

            config = {"aruco_config_path": str(config_file)}
            service = ArucoDetectionService(config)
            aruco_dict, aruco_params = service._load_aruco_config()

            assert aruco_dict is not None
            assert aruco_params is not None

    def test_detect_aruco_single_video(self):
        """Test ArUco detection for single video"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create input structure
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"mock video data")

            output_dir = tmpdir / "output"

            service = ArucoDetectionService(
                {
                    "num_workers": 1,
                    "camera_intrinsics_path": None,
                    "aruco_config_path": None,
                }
            )
            result = service.detect_aruco(str(input_dir), str(output_dir))

            assert len(result["detections"]) == 1
            assert len(result["failed"]) == 0

    def test_detect_aruco_multiple_videos(self):
        """Test ArUco detection for multiple videos"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create multiple demo videos
            demo1_dir = input_dir / "demo1"
            demo1_dir.mkdir()
            (demo1_dir / "demo1.MP4").write_text(b"video1")

            demo2_dir = input_dir / "demo2"
            demo2_dir.mkdir()
            (demo2_dir / "demo2.mp4").write_text(b"video2")

            output_dir = tmpdir / "output"

            service = ArucoDetectionService(
                {
                    "num_workers": 1,
                    "camera_intrinsics_path": None,
                    "aruco_config_path": None,
                }
            )
            result = service.detect_aruco(str(input_dir), str(output_dir))

            assert len(result["detections"]) == 2
            assert len(result["failed"]) == 0

    def test_detect_aruco_empty_directory(self):
        """Test handling empty input directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            output_dir = tmpdir / "output"

            service = ArucoDetectionService(
                {
                    "num_workers": 1,
                    "camera_intrinsics_path": None,
                    "aruco_config_path": None,
                }
            )
            result = service.detect_aruco(str(input_dir), str(output_dir))

            assert len(result["detections"]) == 0
            assert len(result["failed"]) == 0

    def test_detect_in_video_placeholder(self):
        """Test video detection method (placeholder implementation)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock video file
            video_file = tmpdir / "test.MP4"
            video_file.write_text(b"mock video data")
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            service = ArucoDetectionService(
                {
                    "num_workers": 1,
                    "camera_intrinsics_path": None,
                    "aruco_config_path": None,
                }
            )

            result = service._detect_in_video(video_file, output_dir)

            assert isinstance(result, dict)
            assert "video_file" in result
            assert "total_frames" in result
            assert "detections" in result
            assert result["video_file"] == str(video_file)

    def test_validate_detections_success(self):
        """Test successful validation of detection results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            output_dir = tmpdir / "output"
            output_dir.mkdir()
            (output_dir / "test_aruco.json").write_text('{"test": true}')

            service = ArucoDetectionService({"camera_intrinsics_path": None, "aruco_config_path": None})
            assert service.validate_detections(str(output_dir)) is True

    def test_validate_detections_failure(self):
        """Test validation failure cases"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Empty directory
            empty_dir = tmpdir / "empty"
            empty_dir.mkdir()

            service = ArucoDetectionService({"camera_intrinsics_path": None, "aruco_config_path": None})
            assert service.validate_detections(str(empty_dir)) is False

    def test_num_workers_auto_detection(self):
        """Test automatic worker count detection"""
        service = ArucoDetectionService({"num_workers": None})
        assert service.num_workers is not None
        assert isinstance(service.num_workers, int)
        assert service.num_workers > 0

    def test_detect_aruco_no_matching_videos(self):
        """Test detection with no matching video files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()
            (input_dir / "readme.txt").write_text("no videos here")

            output_dir = tmpdir / "output"

            service = ArucoDetectionService(
                {
                    "num_workers": 1,
                    "camera_intrinsics_path": None,
                    "aruco_config_path": None,
                }
            )
            result = service.detect_aruco(str(input_dir), str(output_dir))

            assert len(result["detections"]) == 0
            assert len(result["failed"]) == 0

    def test_detect_aruco_output_structure(self):
        """Test that proper directory structure is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"mock video")

            output_dir = tmpdir / "output"

            service = ArucoDetectionService(
                {
                    "num_workers": 1,
                    "camera_intrinsics_path": None,
                    "aruco_config_path": None,
                }
            )
            service.detect_aruco(str(input_dir), str(output_dir))

            # Check output structure
            assert output_dir.exists()
            detection_files = list(output_dir.glob("*_aruco.json"))
            assert len(detection_files) == 1

    def test_detect_aruco_result_structure(self):
        """Test result structure from detect_aruco"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"mock video")

            output_dir = tmpdir / "output"

            service = ArucoDetectionService(
                {
                    "num_workers": 1,
                    "camera_intrinsics_path": None,
                    "aruco_config_path": None,
                }
            )
            result = service.detect_aruco(str(input_dir), str(output_dir))

            assert "detections" in result
            assert "failed" in result
            assert len(result["detections"]) == 1

            detection = result["detections"][0]
            assert "video_file" in detection
            assert "total_frames" in detection
            assert "detections" in detection


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
