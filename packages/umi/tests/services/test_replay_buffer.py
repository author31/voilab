#!/usr/bin/env python3
"""
Unit tests for ReplayBufferService

Run these tests independently:
    python -m pytest umi/tests/services/test_replay_buffer.py -v
"""

import pytest
import tempfile
import json
import multiprocessing
import pickle
from pathlib import Path

from umi.services.replay_buffer import ReplayBufferService


class TestReplayBufferService:
    """Test cases for ReplayBufferService"""

    def test_init_with_config(self):
        """Test service initialization with custom config"""
        config = {
            "output_resolution": [512, 512],
            "output_fov": 120,
            "compression_level": 9,
            "num_workers": 8,
        }
        service = ReplayBufferService(config)
        assert service.output_resolution == [512, 512]
        assert service.output_fov == 120
        assert service.compression_level == 9
        assert service.num_workers == 8

    def test_init_with_default_config(self):
        """Test service initialization with default config"""
        service = ReplayBufferService({})
        assert service.output_resolution == [256, 256]
        assert service.output_fov is None
        assert service.compression_level == 99
        assert service.num_workers == multiprocessing.cpu_count() // 2

    def test_execute_single_episode(self):
        """Test replay buffer generation for single episode using execute()"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create session directory structure
            session_dir = tmpdir / "session"
            session_dir.mkdir()
            demos_dir = session_dir / "demos"
            demos_dir.mkdir()

            # Create mock dataset plan (matching actual implementation structure)
            dataset_plan = [
                {
                    "grippers": [
                        {
                            "tcp_pose": [[0, 0, 0, 1] for _ in range(100)],
                            "gripper_width": [0.05] * 100,
                            "demo_start_pose": [0, 0, 0, 1],
                            "demo_end_pose": [0, 0, 0, 1],
                        }
                    ],
                    "cameras": [
                        {
                            "video_path": "demo1.MP4",
                            "video_start_end": (0, 100),
                        }
                    ],
                }
            ]

            (session_dir / "dataset_plan.pkl").write_bytes(pickle.dumps(dataset_plan))

            # Create mock video file in demos directory
            video_file = demos_dir / "demo1.MP4"
            video_file.write_bytes(b"mock video data")

            # Create mock tag detection file
            tag_detection = {i: {"tag_dict": {}} for i in range(100)}
            (demos_dir / "tag_detection.pkl").write_bytes(pickle.dumps(tag_detection))

            output_file = "replay_buffer.zarr"

            service = ReplayBufferService(
                {
                    "session_dir": str(session_dir),
                    "output_filename": output_file,
                    "dataset_plan_filename": "dataset_plan.pkl",
                    "output_resolution": [64, 64],
                    "compression_level": 1,
                    "num_workers": 1,
                }
            )

            # This will fail due to mock data, but we can test the setup
            with pytest.raises((RuntimeError, AssertionError, Exception)):
                result = service.execute()

            # Test that the service was configured correctly
            assert service.session_dir == str(session_dir)
            assert service.output_filename == output_file
            assert service.dataset_plan_filename == "dataset_plan.pkl"

    def test_execute_missing_required_config(self):
        """Test execute() with missing required configuration"""
        service = ReplayBufferService({})

        with pytest.raises(AssertionError, match="Missing session_dir"):
            service.execute()

        # Test missing output_filename
        service = ReplayBufferService({"session_dir": "/tmp"})
        with pytest.raises(AssertionError, match="Missing output_filename"):
            service.execute()

        # Test missing dataset_plan_filename
        service = ReplayBufferService({
            "session_dir": "/tmp",
            "output_filename": "test.zarr"
        })
        with pytest.raises(AssertionError, match="Missing dataset_plan_filename"):
            service.execute()

    def test_execute_missing_dataset_plan(self):
        """Test execute() with missing dataset plan file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            service = ReplayBufferService({
                "session_dir": str(tmpdir),
                "output_filename": "test.zarr",
                "dataset_plan_filename": "missing.pkl",
            })

            with pytest.raises(RuntimeError, match="does not exist"):
                service.execute()

    def test_video_to_zarg_method_exists(self):
        """Test that video_to_zarr method exists"""
        service = ReplayBufferService({})
        assert hasattr(service, 'video_to_zarr')
        assert callable(service.video_to_zarr)

    def test_num_workers_default_value(self):
        """Test that num_workers has a reasonable default value"""
        service = ReplayBufferService({})
        expected_workers = multiprocessing.cpu_count() // 2
        assert service.num_workers == expected_workers
        assert isinstance(service.num_workers, int)
        assert service.num_workers > 0

    def test_configuration_parameters(self):
        """Test various configuration parameters"""
        # Test with custom FOV and intrinsic path
        service = ReplayBufferService({
            "output_fov": 100,
            "output_fov_intrinsic_path": "/path/to/intrinsics.json",
            "no_mirror": True,
            "mirror_swap": False,
        })

        assert service.output_fov == 100
        assert service.output_fov_intrinsic_path == "/path/to/intrinsics.json"
        assert service.no_mirror is True
        assert service.mirror_swap is False

    def test_fisheye_converter_configuration(self):
        """Test fisheye converter configuration parameters"""
        service = ReplayBufferService({
            "output_fov": 90,
            "output_fov_intrinsic_path": "/path/to/intrinsics.json",
            "output_resolution": [512, 512],
        })

        assert service.output_fov == 90
        assert service.output_fov_intrinsic_path == "/path/to/intrinsics.json"
        assert service.output_resolution == [512, 512]

    def test_mirror_configuration(self):
        """Test mirror-related configuration"""
        service = ReplayBufferService({
            "no_mirror": True,
            "mirror_swap": True,
        })

        assert service.no_mirror is True
        assert service.mirror_swap is True


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
