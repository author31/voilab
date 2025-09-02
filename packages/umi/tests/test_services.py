import pytest
import tempfile
import json
from pathlib import Path

from umi.services.video_organization import VideoOrganizationService
from umi.services.imu_extraction import IMUExtractionService
from umi.services.slam_mapping import SLAMMappingService
from umi.services.batch_slam import BatchSLAMService
from umi.services.aruco_detection import ArucoDetectionService
from umi.services.calibration import CalibrationService
from umi.services.dataset_planning import DatasetPlanningService
from umi.services.replay_buffer import ReplayBufferService


class TestVideoOrganizationService:
    """Tests for VideoOrganizationService."""

    def test_organize_videos(self):
        """Test video organization functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test structure
            session_dir = tmpdir / "session"
            session_dir.mkdir()

            # Create test video files
            (session_dir / "demo1_video.MP4").touch()
            (session_dir / "demo2_video.mp4").touch()

            output_dir = tmpdir / "output"

            service = VideoOrganizationService({"input_patterns": ["*.MP4", "*.mp4"]})
            result = service.organize_videos(str(session_dir), str(output_dir))

            assert result["success"] is True
            assert "demos" in result
            assert len(result["demos"]) == 2
            assert result["total_videos"] == 2

    def test_validate_organization(self):
        """Test organization validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Valid structure
            valid_dir = tmpdir / "valid"
            valid_dir.mkdir()
            demo_dir = valid_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").touch()

            service = VideoOrganizationService({})
            assert service.validate_organization(str(valid_dir)) is True

            # Invalid structure
            invalid_dir = tmpdir / "invalid"
            invalid_dir.mkdir()
            assert service.validate_organization(str(invalid_dir)) is False


class TestIMUExtractionService:
    """Tests for IMUExtractionService."""

    def test_extract_imu(self):
        """Test IMU extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create input structure
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").touch()

            output_dir = tmpdir / "output"

            service = IMUExtractionService({"num_workers": 1})
            result = service.extract_imu(str(input_dir), str(output_dir))

            assert "extracted" in result
            assert "failed" in result
            assert len(result["extracted"]) == 1

    def test_validate_extraction(self):
        """Test extraction validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Valid extraction
            valid_dir = tmpdir / "valid"
            valid_dir.mkdir()
            (valid_dir / "test_imu.json").touch()

            service = IMUExtractionService({})
            assert service.validate_extraction(str(valid_dir)) is True


class TestSLAMMappingService:
    """Tests for SLAMMappingService."""

    def test_create_map(self):
        """Test SLAM map creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create input structure
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").touch()

            output_dir = tmpdir / "output"

            service = SLAMMappingService({"docker_image": "test:latest"})
            result = service.create_map(str(input_dir), str(output_dir))

            assert "maps" in result
            assert "failed" in result
            assert len(result["maps"]) == 1

    def test_validate_mapping(self):
        """Test mapping validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Valid mapping
            valid_dir = tmpdir / "valid"
            valid_dir.mkdir()
            demo_dir = valid_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "map.bin").touch()
            (demo_dir / "trajectory.txt").touch()

            service = SLAMMappingService({})
            assert service.validate_mapping(str(valid_dir)) is True


class TestBatchSLAMService:
    """Tests for BatchSLAMService."""

    def test_process_batch(self):
        """Test batch processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create input structure
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").touch()
            (demo_dir / "map.bin").touch()

            output_dir = tmpdir / "output"

            service = BatchSLAMService({"num_workers": 1})
            result = service.process_batch(str(input_dir), str(output_dir))

            assert "processed" in result
            assert "failed" in result
            assert len(result["processed"]) == 1

    def test_validate_batch_results(self):
        """Test batch validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Valid batch results
            valid_dir = tmpdir / "valid"
            valid_dir.mkdir()
            demo_dir = valid_dir / "demo1"
            demo_dir.mkdir()

            service = BatchSLAMService({})
            assert service.validate_batch_results(str(valid_dir)) is True


class TestArucoDetectionService:
    """Tests for ArucoDetectionService."""

    def test_detect_aruco(self):
        """Test ArUco detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create input structure
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").touch()

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

    def test_validate_detections(self):
        """Test detection validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Valid detections
            valid_dir = tmpdir / "valid"
            valid_dir.mkdir()
            (valid_dir / "test_aruco.json").touch()

            service = ArucoDetectionService({})
            assert service.validate_detections(str(valid_dir)) is True


class TestCalibrationService:
    """Tests for CalibrationService."""

    def test_run_calibrations(self):
        """Test calibration execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create input structure
            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock calibration files
            (input_dir / "slam_tag_calibration.json").write_text('{"test": true}')
            (input_dir / "gripper_range_calibration.json").write_text('{"test": true}')

            output_dir = tmpdir / "output"

            service = CalibrationService({"slam_tag_calibration_timeout": 10, "gripper_range_timeout": 10})
            result = service.run_calibrations(str(input_dir), str(output_dir))

            assert "slam_tag_calibration" in result
            assert "gripper_range_calibration" in result

    def test_validate_calibrations(self):
        """Test calibration validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Valid calibrations
            valid_dir = tmpdir / "valid"
            valid_dir.mkdir()
            (valid_dir / "slam_tag_calibration.json").touch()
            (valid_dir / "gripper_range_calibration.json").touch()

            service = CalibrationService({})
            assert service.validate_calibrations(str(valid_dir)) is True


class TestDatasetPlanningService:
    """Tests for DatasetPlanningService."""

    def test_generate_plan(self):
        """Test dataset plan generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create input structure
            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock dataset plan
            dataset_plan = {
                "episodes": [
                    {
                        "demo_name": "demo1",
                        "frame_count": 100,
                        "duration": 3.3,
                        "metadata": {},
                    }
                ]
            }

            (input_dir / "dataset_plan.json").write_text(json.dumps(dataset_plan))

            output_dir = tmpdir / "output"

            service = DatasetPlanningService(
                {
                    "tcp_offset": [0.0, 0.0, 0.0],
                    "nominal_z": 0.0,
                    "min_episode_length": 10,
                }
            )
            result = service.generate_plan(str(input_dir), str(output_dir))

            assert "plan_file" in result
            assert "total_episodes" in result
            assert result["total_episodes"] > 0

    def test_validate_plan(self):
        """Test plan validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Valid plan
            valid_dir = tmpdir / "valid"
            valid_dir.mkdir()

            plan_data = {"episodes": [{"demo_name": "test"}]}
            (valid_dir / "dataset_plan.json").write_text(json.dumps(plan_data))

            service = DatasetPlanningService({})
            assert service.validate_plan(str(valid_dir)) is True


class TestReplayBufferService:
    """Tests for ReplayBufferService."""

    def test_generate_replay_buffer(self):
        """Test replay buffer generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create input structure
            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock dataset plan
            dataset_plan = {"episodes": [{"demo_name": "demo1", "frame_count": 10, "metadata": {}}]}

            (input_dir / "dataset_plan.json").write_text(json.dumps(dataset_plan))

            # Create mock video
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"mock video data")

            output_dir = tmpdir / "output"

            service = ReplayBufferService(
                {
                    "output_resolution": [64, 64],
                    "output_fov": 90,
                    "compression_level": 1,
                    "num_workers": 1,
                }
            )
            result = service.generate_replay_buffer(str(input_dir), str(output_dir))

            assert "episodes" in result
            assert "summary" in result
            assert result["summary"]["total_episodes"] > 0

    def test_validate_replay_buffer(self):
        """Test replay buffer validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Valid replay buffer
            valid_dir = tmpdir / "valid"
            valid_dir.mkdir()

            summary_data = {"total_episodes": 1}
            (valid_dir / "replay_buffer_summary.json").write_text(json.dumps(summary_data))

            service = ReplayBufferService({})
            assert service.validate_replay_buffer(str(valid_dir)) is True


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v"])
