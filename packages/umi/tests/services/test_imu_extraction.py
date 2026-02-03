#!/usr/bin/env python3
"""
Unit tests for IMUExtractionService

Run these tests independently:
    python -m pytest umi/tests/services/test_imu_extraction.py -v
"""

import pytest
import tempfile
from pathlib import Path

from umi.services.imu_extraction import IMUExtractionService


class TestIMUExtractionService:
    """Test cases for IMUExtractionService"""

    def test_init_with_config(self):
        """Test service initialization with custom config"""
        config = {"num_workers": 4, "stream_types": ["ACCL", "GYRO"]}
        service = IMUExtractionService(config)
        assert service.num_workers == 4
        assert service.stream_types == ["ACCL", "GYRO"]

    def test_init_with_default_config(self):
        """Test service initialization with default config"""
        service = IMUExtractionService({})
        assert service.num_workers is not None  # Auto-detected
        assert service.stream_types == ["ACCL", "GYRO", "GPS5", "GPSP", "GPSU", "GPSF", "GRAV", "MAGN", "CORI", "IORI", "TMPC"]

    def test_extract_imu_single_demo(self):
        """Test IMU extraction for single demo"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create input structure
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text("mock video data")

            output_dir = tmpdir / "output"

            service = IMUExtractionService({"num_workers": 1})
            result = service.extract_imu(str(input_dir), str(output_dir))

            assert len(result["extracted"]) == 1
            assert len(result["failed"]) == 0
            assert result["extracted"][0]["demo"] == "demo1"

    def test_extract_imu_multiple_demos(self):
        """Test IMU extraction for multiple demos"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create multiple demo videos
            demo1_dir = input_dir / "demo1"
            demo1_dir.mkdir()
            (demo1_dir / "demo1.MP4").write_text("video1")

            demo2_dir = input_dir / "demo2"
            demo2_dir.mkdir()
            (demo2_dir / "demo2.mp4").write_text("video2")

            output_dir = tmpdir / "output"

            service = IMUExtractionService({"num_workers": 1})
            result = service.extract_imu(str(input_dir), str(output_dir))

            assert len(result["extracted"]) == 2
            assert len(result["failed"]) == 0
            assert any(e["demo"] == "demo1" for e in result["extracted"])
            assert any(e["demo"] == "demo2" for e in result["extracted"])

    def test_extract_imu_empty_directory(self):
        """Test handling empty input directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            output_dir = tmpdir / "output"

            service = IMUExtractionService({"num_workers": 1})
            result = service.extract_imu(str(input_dir), str(output_dir))

            assert len(result["extracted"]) == 0
            assert len(result["failed"]) == 0

    def test_extract_imu_from_video_creates_files(self):
        """Test that IMU files are created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            video_file = demo_dir / "demo1.MP4"
            video_file.write_text("mock video data")

            output_dir = tmpdir / "output"

            service = IMUExtractionService({"num_workers": 1})
            service.extract_imu(str(input_dir), str(output_dir))

            # Check that IMU file was created
            imu_files = list(output_dir.glob("*_imu.json"))
            assert len(imu_files) == 1

    def test_validate_extraction_success(self):
        """Test successful validation of extracted IMU data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            output_dir = tmpdir / "output"
            output_dir.mkdir()
            (output_dir / "test_imu.json").write_text('{"test": true}')

            service = IMUExtractionService({})
            assert service.validate_extraction(str(output_dir)) is True

    def test_validate_extraction_failure(self):
        """Test validation failure cases"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Empty directory
            empty_dir = tmpdir / "empty"
            empty_dir.mkdir()

            service = IMUExtractionService({})
            assert service.validate_extraction(str(empty_dir)) is False

    def test_num_workers_auto_detection(self):
        """Test automatic worker count detection"""
        service = IMUExtractionService({"num_workers": None})
        assert service.num_workers is not None
        assert isinstance(service.num_workers, int)
        assert service.num_workers > 0

    def test_custom_stream_types(self):
        """Test custom stream types configuration"""
        config = {"stream_types": ["ACCL", "GYRO", "TEST"]}
        service = IMUExtractionService(config)
        assert service.stream_types == ["ACCL", "GYRO", "TEST"]

    def test_extract_imu_no_matching_videos(self):
        """Test extraction with no matching video files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()
            (input_dir / "readme.txt").write_text("no videos here")

            output_dir = tmpdir / "output"

            service = IMUExtractionService({"num_workers": 1})
            result = service.extract_imu(str(input_dir), str(output_dir))

            assert len(result["extracted"]) == 0
            assert len(result["failed"]) == 0

    def test_extract_imu_file_structure(self):
        """Test that proper directory structure is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text("mock video")

            output_dir = tmpdir / "output"

            service = IMUExtractionService({"num_workers": 1})
            service.extract_imu(str(input_dir), str(output_dir))

            # Check output directory structure
            assert output_dir.exists()
            assert output_dir.is_dir()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
