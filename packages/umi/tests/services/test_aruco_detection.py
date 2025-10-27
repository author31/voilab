#!/usr/bin/env python3
"""
Unit tests for ArucoDetectionService

Run these tests independently:
    python -m pytest umi/tests/services/test_aruco_detection.py -v
"""

import pytest
import tempfile
import json
import pickle
import multiprocessing
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import numpy as np

from umi.services.aruco_detection import ArucoDetectionService


class TestArucoDetectionService:
    """Test cases for ArucoDetectionService"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            "session_dir": "/tmp/test_session",
            "num_workers": 2,
            "camera_intrinsics_path": "/tmp/camera_intrinsics.json",
            "aruco_config_path": "/tmp/aruco_config.yaml"
        }

    @pytest.fixture
    def sample_camera_intrinsics(self):
        """Sample camera intrinsics data"""
        return {
            "intrinsic_type": "FISHEYE",
            "image_height": 1080,
            "image_width": 1920,
            "intrinsics": {
                "focal_length": 420.5,
                "principal_pt_x": 959.8,
                "principal_pt_y": 542.8,
                "aspect_ratio": 1.0,
                "skew": 0.0,
                "radial_distortion_1": -0.011,
                "radial_distortion_2": -0.039,
                "radial_distortion_3": 0.018,
                "radial_distortion_4": -0.005
            }
        }

    @pytest.fixture
    def sample_aruco_config(self):
        """Sample ArUco configuration data"""
        return {
            "aruco_dict": {
                "predefined": "DICT_4X4_50"
            },
            "marker_size_map": {
                "default": 0.15,
                "12": 0.2
            }
        }

    @pytest.fixture
    def mock_video_structure(self, temp_dir):
        """Create mock video directory structure"""
        session_dir = temp_dir / "session"
        demos_dir = session_dir / "demos"
        demos_dir.mkdir(parents=True)

        # Create video directories with raw_video.mp4
        video_dirs = []
        for i in range(3):
            video_dir = demos_dir / f"video_{i}"
            video_dir.mkdir()
            (video_dir / "raw_video.mp4").touch()
            video_dirs.append(video_dir)

        return session_dir, demos_dir, video_dirs

    def test_init_with_config(self, sample_config):
        """Test service initialization with custom configuration"""
        service = ArucoDetectionService(sample_config)

        assert service.session_dir == sample_config["session_dir"]
        assert service.num_workers == sample_config["num_workers"]
        assert service.camera_intrinsics_path == sample_config["camera_intrinsics_path"]
        assert service.aruco_config_path == sample_config["aruco_config_path"]

    def test_init_with_default_config(self, sample_config):
        """Test service initialization with default num_workers"""
        config_without_workers = sample_config.copy()
        del config_without_workers["num_workers"]

        service = ArucoDetectionService(config_without_workers)

        expected_workers = multiprocessing.cpu_count() // 2
        assert service.num_workers == expected_workers

    def test_init_minimal_config(self):
        """Test service initialization with minimal configuration"""
        config = {"session_dir": "/tmp/test"}
        service = ArucoDetectionService(config)

        assert service.session_dir == "/tmp/test"
        assert service.camera_intrinsics_path is None
        assert service.aruco_config_path is None

    @patch('umi.services.aruco_detection.cv2.setNumThreads')
    @patch('umi.services.aruco_detection.Path')
    def test_execute_no_videos_found(self, mock_path, mock_cv2_threads, sample_config):
        """Test execute when no video directories are found"""
        # Setup mocks
        mock_session_path = Mock()
        mock_demos_path = Mock()
        mock_path.return_value = mock_session_path
        mock_session_path.__truediv__ = Mock(return_value=mock_demos_path)
        mock_demos_path.glob.return_value = []  # No videos found

        service = ArucoDetectionService(sample_config)
        result = service.execute()

        assert result["total_videos_found"] == 0
        assert result["videos_processed"] == 0
        assert result["videos_skipped"] == 0
        assert result["processed_video_names"] == []
        assert result["skipped_video_names"] == []
        mock_cv2_threads.assert_called_once_with(sample_config["num_workers"])

    def test_execute_missing_session_dir(self):
        """Test execute when session_dir is missing"""
        service = ArucoDetectionService({})

        with pytest.raises(AssertionError, match="Missing session_dir"):
            service.execute()

    @patch('umi.services.aruco_detection.cv2.setNumThreads')
    @patch('umi.services.aruco_detection.concurrent.futures.ThreadPoolExecutor')
    @patch('umi.services.aruco_detection.Path')
    def test_execute_all_videos_processed(self, mock_path, mock_executor, mock_cv2_threads,
                                        sample_config, mock_video_structure):
        """Test execute when all videos need processing"""
        session_dir, demos_dir, video_dirs = mock_video_structure

        # Update config to use real session dir
        config = sample_config.copy()
        config["session_dir"] = str(session_dir)

        # Setup executor mock
        mock_future = Mock()
        mock_executor_instance = Mock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor_instance.__enter__ = Mock(return_value=mock_executor_instance)
        mock_executor_instance.__exit__ = Mock(return_value=None)
        mock_executor.return_value = mock_executor_instance

        # Mock the service method to avoid actual video processing
        with patch.object(ArucoDetectionService, 'detect_aruco') as mock_detect:
            service = ArucoDetectionService(config)

            # Mock the detect_aruco method
            mock_detect.return_value = None

            # Create real Path objects for this test
            real_demos_path = Path(demos_dir)

            with patch('umi.services.aruco_detection.Path') as mock_path_class:
                mock_session_path = Mock()
                mock_session_path.__truediv__ = Mock(return_value=real_demos_path)
                mock_path_class.return_value = mock_session_path
                mock_path_class.return_value.glob = Mock(return_value=[
                    video_dir / "raw_video.mp4" for video_dir in video_dirs
                ])

                result = service.execute()

                assert result["total_videos_found"] == 3
                assert result["videos_processed"] == 3
                assert result["videos_skipped"] == 0
                assert len(result["processed_video_names"]) == 3
                assert mock_detect.call_count == 3

    @patch('umi.services.aruco_detection.cv2.setNumThreads')
    @patch('umi.services.aruco_detection.Path')
    def test_execute_some_videos_skipped(self, mock_path, mock_cv2_threads,
                                      sample_config, mock_video_structure):
        """Test execute when some videos already have detection files"""
        session_dir, demos_dir, video_dirs = mock_video_structure

        # Create existing detection file for one video
        (video_dirs[0] / "tag_detection.pkl").write_bytes(pickle.dumps([{"frame": 1}]))

        # Update config to use real session dir
        config = sample_config.copy()
        config["session_dir"] = str(session_dir)

        with patch.object(ArucoDetectionService, 'detect_aruco') as mock_detect:
            service = ArucoDetectionService(config)
            mock_detect.return_value = None

            # Create real Path objects for this test
            real_demos_path = Path(demos_dir)

            with patch('umi.services.aruco_detection.Path') as mock_path_class:
                mock_session_path = Mock()
                mock_session_path.__truediv__ = Mock(return_value=real_demos_path)
                mock_path_class.return_value = mock_session_path
                mock_path_class.return_value.glob = Mock(return_value=[
                    video_dir / "raw_video.mp4" for video_dir in video_dirs
                ])

                with patch('umi.services.aruco_detection.concurrent.futures.ThreadPoolExecutor') as mock_executor:
                    mock_executor_instance = Mock()
                    mock_executor_instance.submit.return_value = Mock()
                    mock_executor_instance.__enter__ = Mock(return_value=mock_executor_instance)
                    mock_executor_instance.__exit__ = Mock(return_value=None)
                    mock_executor.return_value = mock_executor_instance

                    result = service.execute()

                    assert result["total_videos_found"] == 3
                    assert result["videos_processed"] == 2  # Only videos without detection files
                    assert result["videos_skipped"] == 1   # Video with existing detection file
                    assert mock_detect.call_count == 2

    def test_detect_aruco_missing_camera_intrinsics(self, sample_config, temp_dir):
        """Test detect_aruco raises assertion error when camera_intrinsics_path is missing"""
        config = sample_config.copy()
        del config["camera_intrinsics_path"]

        service = ArucoDetectionService(config)
        video_path = temp_dir / "test.mp4"
        output_path = temp_dir / "output.pkl"

        with pytest.raises(AssertionError, match="Missing camera_intrinsics_path"):
            service.detect_aruco(video_path, output_path)

    def test_detect_aruco_missing_aruco_config(self, sample_config, temp_dir):
        """Test detect_aruco raises assertion error when aruco_config_path is missing"""
        config = sample_config.copy()
        del config["aruco_config_path"]

        service = ArucoDetectionService(config)
        video_path = temp_dir / "test.mp4"
        output_path = temp_dir / "output.pkl"

        with pytest.raises(AssertionError, match="Missing aruco_config_path"):
            service.detect_aruco(video_path, output_path)

    @patch('umi.services.aruco_detection.yaml.safe_load')
    @patch('umi.services.aruco_detection.json.load')
    @patch('umi.services.aruco_detection.parse_aruco_config')
    @patch('umi.services.aruco_detection.parse_fisheye_intrinsics')
    @patch('umi.services.aruco_detection.convert_fisheye_intrinsics_resolution')
    @patch('umi.services.aruco_detection.detect_localize_aruco_tags')
    @patch('umi.services.aruco_detection.draw_predefined_mask')
    @patch('umi.services.aruco_detection.av.open')
    @patch('umi.services.aruco_detection.pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_detect_aruco_successful_processing(self, mock_file_open, mock_pickle_dump,
                                              mock_av_open, mock_draw_mask, mock_detect_tags,
                                              mock_convert_intrinsics, mock_parse_fisheye,
                                              mock_parse_aruco, mock_json_load, mock_yaml_load,
                                              sample_config, sample_camera_intrinsics,
                                              sample_aruco_config, temp_dir):
        """Test successful ArUco detection processing"""
        # Setup mocks
        mock_yaml_load.return_value = sample_aruco_config
        mock_json_load.return_value = sample_camera_intrinsics

        mock_aruco_dict = Mock()
        mock_marker_size_map = {"default": 0.15}
        mock_parse_aruco.return_value = {
            "aruco_dict": mock_aruco_dict,
            "marker_size_map": mock_marker_size_map
        }

        mock_fisheye_intr = {"K": np.eye(3), "D": np.zeros(4)}
        mock_parse_fisheye.return_value = mock_fisheye_intr

        mock_converted_intrinsics = {"K": np.eye(3), "D": np.zeros(4), "DIM": np.array([1920, 1080])}
        mock_convert_intrinsics.return_value = mock_converted_intrinsics

        # Mock video processing
        mock_container = Mock()
        mock_stream = Mock()
        mock_stream.height = 1080
        mock_stream.width = 1920
        mock_stream.frames = 10
        mock_stream.time_base = 1/30.0
        mock_container.streams.video = [mock_stream]

        mock_frame = Mock()
        mock_frame.pts = 0
        mock_frame.to_ndarray.return_value = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        mock_container.decode.return_value = [mock_frame]
        mock_av_open.return_value.__enter__.return_value = mock_container

        # Mock ArUco detection
        mock_tag_dict = {"tag_id": {"rvec": np.array([0, 0, 0]), "tvec": np.array([0, 0, 0])}}
        mock_detect_tags.return_value = mock_tag_dict

        # Mock image mask drawing
        mock_draw_mask.return_value = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        service = ArucoDetectionService(sample_config)
        video_path = temp_dir / "test.mp4"
        output_path = temp_dir / "output.pkl"

        service.detect_aruco(video_path, output_path)

        # Verify all functions were called
        mock_av_open.assert_called_once_with(str(video_path))
        mock_draw_mask.assert_called()
        mock_detect_tags.assert_called()
        mock_pickle_dump.assert_called_once()

        # Verify file operations
        assert mock_file_open.call_count >= 2  # Once for config, once for camera intrinsics

    @patch('umi.services.aruco_detection.yaml.safe_load')
    @patch('umi.services.aruco_detection.json.load')
    @patch('umi.services.aruco_detection.parse_aruco_config')
    @patch('umi.services.aruco_detection.parse_fisheye_intrinsics')
    @patch('umi.services.aruco_detection.convert_fisheye_intrinsics_resolution')
    @patch('umi.services.aruco_detection.detect_localize_aruco_tags')
    @patch('umi.services.aruco_detection.draw_predefined_mask')
    @patch('umi.services.aruco_detection.av.open')
    @patch('umi.services.aruco_detection.pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_detect_aruco_multiple_frames(self, mock_file_open, mock_pickle_dump,
                                        mock_av_open, mock_draw_mask, mock_detect_tags,
                                        mock_convert_intrinsics, mock_parse_fisheye,
                                        mock_parse_aruco, mock_json_load, mock_yaml_load,
                                        sample_config, sample_camera_intrinsics,
                                        sample_aruco_config, temp_dir):
        """Test ArUco detection with multiple video frames"""
        # Setup mocks similar to previous test
        mock_yaml_load.return_value = sample_aruco_config
        mock_json_load.return_value = sample_camera_intrinsics
        mock_parse_aruco.return_value = {
            "aruco_dict": Mock(),
            "marker_size_map": {"default": 0.15}
        }
        mock_parse_fisheye.return_value = {"K": np.eye(3), "D": np.zeros(4), "DIM": np.array([1920, 1080])}
        mock_convert_intrinsics.return_value = {"K": np.eye(3), "D": np.zeros(4), "DIM": np.array([1920, 1080])}

        # Mock video with multiple frames
        mock_container = Mock()
        mock_stream = Mock()
        mock_stream.height = 1080
        mock_stream.width = 1920
        mock_stream.frames = 3
        mock_stream.time_base = 1/30.0
        mock_container.streams.video = [mock_stream]

        mock_frames = []
        for i in range(3):
            mock_frame = Mock()
            mock_frame.pts = i
            mock_frame.to_ndarray.return_value = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            mock_frames.append(mock_frame)

        mock_container.decode.return_value = mock_frames
        mock_av_open.return_value.__enter__.return_value = mock_container

        mock_detect_tags.return_value = {"tag_id": {"rvec": np.array([0, 0, 0]), "tvec": np.array([0, 0, 0])}}
        mock_draw_mask.return_value = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        service = ArucoDetectionService(sample_config)
        video_path = temp_dir / "test.mp4"
        output_path = temp_dir / "output.pkl"

        service.detect_aruco(video_path, output_path)

        # Verify pickle was called with results containing all frames
        mock_pickle_dump.assert_called_once()
        saved_results = mock_pickle_dump.call_args[0][0]
        assert len(saved_results) == 3  # Should have 3 frame results

        # Verify each result has correct structure
        for i, result in enumerate(saved_results):
            assert result["frame_idx"] == i
            assert "time" in result
            assert "tag_dict" in result

    def test_integration_end_to_end(self, sample_config, mock_video_structure,
                                  sample_camera_intrinsics, sample_aruco_config):
        """Test end-to-end integration with mocked dependencies"""
        session_dir, demos_dir, video_dirs = mock_video_structure

        # Update config
        config = sample_config.copy()
        config["session_dir"] = str(session_dir)

        # Create actual config files
        camera_intrinsics_path = Path(tempfile.gettempdir()) / "test_camera_intrinsics.json"
        aruco_config_path = Path(tempfile.gettempdir()) / "test_aruco_config.yaml"

        with open(camera_intrinsics_path, 'w') as f:
            json.dump(sample_camera_intrinsics, f)

        with open(aruco_config_path, 'w') as f:
            import yaml
            yaml.dump(sample_aruco_config, f)

        config["camera_intrinsics_path"] = str(camera_intrinsics_path)
        config["aruco_config_path"] = str(aruco_config_path)

        with patch('umi.services.aruco_detection.cv2.setNumThreads'), \
             patch('umi.services.aruco_detection.av.open') as mock_av_open, \
             patch('umi.services.aruco_detection.pickle.dump') as mock_pickle_dump:

            # Mock video processing
            mock_container = Mock()
            mock_stream = Mock()
            mock_stream.height = 108
            mock_stream.width = 192
            mock_stream.frames = 2
            mock_stream.time_base = 1/30.0
            mock_container.streams.video = [mock_stream]

            mock_frame = Mock()
            mock_frame.pts = 0
            mock_frame.to_ndarray.return_value = np.random.randint(0, 255, (108, 192, 3), dtype=np.uint8)
            mock_container.decode.return_value = [mock_frame, mock_frame]
            mock_av_open.return_value.__enter__.return_value = mock_container

            service = ArucoDetectionService(config)

            with patch('umi.services.aruco_detection.concurrent.futures.ThreadPoolExecutor') as mock_executor:
                mock_executor_instance = Mock()
                mock_future = Mock()
                mock_executor_instance.submit.return_value = mock_future
                mock_executor_instance.__enter__ = Mock(return_value=mock_executor_instance)
                mock_executor_instance.__exit__ = Mock(return_value=None)
                mock_executor.return_value = mock_executor_instance

                result = service.execute()

                assert result["total_videos_found"] == 3
                assert result["videos_processed"] == 3

    def test_error_handling_in_execute(self, sample_config, mock_video_structure):
        """Test error handling during execute"""
        session_dir, demos_dir, video_dirs = mock_video_structure
        config = sample_config.copy()
        config["session_dir"] = str(session_dir)

        service = ArucoDetectionService(config)

        with patch('umi.services.aruco_detection.cv2.setNumThreads'), \
             patch('umi.services.aruco_detection.concurrent.futures.ThreadPoolExecutor') as mock_executor:

            # Setup executor to raise an exception
            mock_executor_instance = Mock()
            mock_future = Mock()
            mock_future.result.side_effect = Exception("Test error")
            mock_executor_instance.submit.return_value = mock_future
            mock_executor_instance.__enter__ = Mock(return_value=mock_executor_instance)
            mock_executor_instance.__exit__ = Mock(return_value=None)
            mock_executor.return_value = mock_executor_instance

            with patch('umi.services.aruco_detection.Path') as mock_path_class:
                real_demos_path = Path(demos_dir)
                mock_session_path = Mock()
                mock_session_path.__truediv__ = Mock(return_value=real_demos_path)
                mock_path_class.return_value = mock_session_path
                mock_path_class.return_value.glob = Mock(return_value=[
                    video_dir / "raw_video.mp4" for video_dir in video_dirs
                ])

                with pytest.raises(Exception, match="Test error"):
                    service.execute()

    def test_empty_video_handling(self, sample_config, temp_dir,
                                sample_camera_intrinsics, sample_aruco_config):
        """Test handling of videos with no frames"""
        # Create config files
        camera_intrinsics_path = temp_dir / "camera_intrinsics.json"
        aruco_config_path = temp_dir / "aruco_config.yaml"

        with open(camera_intrinsics_path, 'w') as f:
            json.dump(sample_camera_intrinsics, f)

        with open(aruco_config_path, 'w') as f:
            import yaml
            yaml.dump(sample_aruco_config, f)

        config = sample_config.copy()
        config["camera_intrinsics_path"] = str(camera_intrinsics_path)
        config["aruco_config_path"] = str(aruco_config_path)

        with patch('umi.services.aruco_detection.yaml.safe_load') as mock_yaml, \
             patch('umi.services.aruco_detection.json.load') as mock_json, \
             patch('umi.services.aruco_detection.parse_aruco_config') as mock_parse_aruco, \
             patch('umi.services.aruco_detection.parse_fisheye_intrinsics') as mock_parse_fisheye, \
             patch('umi.services.aruco_detection.av.open') as mock_av_open, \
             patch('umi.services.aruco_detection.pickle.dump') as mock_pickle_dump:

            # Setup mocks
            mock_yaml.return_value = sample_aruco_config
            mock_json.return_value = sample_camera_intrinsics
            mock_parse_aruco.return_value = {
            "aruco_dict": Mock(),
            "marker_size_map": {"default": 0.15}
        }
            mock_parse_fisheye.return_value = {"K": np.eye(3), "D": np.zeros(4), "DIM": np.array([1920, 1080])}

            # Mock empty video
            mock_container = Mock()
            mock_stream = Mock()
            mock_stream.height = 1080
            mock_stream.width = 1920
            mock_stream.frames = 0
            mock_container.streams.video = [mock_stream]
            mock_container.decode.return_value = []  # No frames
            mock_av_open.return_value.__enter__.return_value = mock_container

            service = ArucoDetectionService(config)
            video_path = temp_dir / "empty_video.mp4"
            output_path = temp_dir / "output.pkl"

            service.detect_aruco(video_path, output_path)

            # Verify empty results are saved
            mock_pickle_dump.assert_called_once()
            saved_results = mock_pickle_dump.call_args[0][0]
            assert len(saved_results) == 0