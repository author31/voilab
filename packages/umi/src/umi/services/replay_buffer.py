from pathlib import Path
import json
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
import gzip

from .base_service import BaseService


class ReplayBufferService(BaseService):
    """Service for generating replay buffers from processed data."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.output_resolution = tuple(self.config.get("output_resolution", [256, 256]))
        self.output_fov = self.config.get("output_fov", 90)
        self.compression_level = self.config.get("compression_level", 6)
        self.num_workers = self.config.get("num_workers", 4)

    def execute(self, input_dir: str, output_dir: str) -> dict:
        """
        Generate replay buffer from processed data.

        Args:
            input_dir: Directory containing processed videos, plans, and calibrations
            output_dir: Directory for replay buffer outputs

        Returns:
            dict: Replay buffer generation results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path = self._ensure_output_dir(output_dir)

        # Load dataset plan
        plan_file = input_path / "dataset_plan.json"
        if not plan_file.exists():
            raise ValueError("Dataset plan not found")

        with open(plan_file, "r") as f:
            dataset_plan = json.load(f)

        # Generate replay buffer for each episode
        results = {"episodes": [], "failed": []}

        episodes = dataset_plan.get("episodes", [])

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_episode = {
                executor.submit(
                    self._process_episode, episode, input_path, output_path
                ): episode
                for episode in episodes
            }

            for future in as_completed(future_to_episode):
                episode = future_to_episode[future]
                try:
                    result = future.result()
                    if result:
                        results["episodes"].append(result)
                except Exception as e:
                    results["failed"].append(
                        {
                            "episode": episode.get("demo_name", "unknown"),
                            "error": str(e),
                        }
                    )

        # Create summary
        summary = self._create_summary(results)
        summary_file = output_path / "replay_buffer_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        results["summary"] = summary
        results["summary_file"] = str(summary_file)

        return results

    def _process_episode(
        self, episode: Dict[str, Any], input_path: Path, output_path: Path
    ) -> Dict[str, Any]:
        """Process a single episode into replay buffer format."""
        demo_name = episode["demo_name"]
        episode_dir = output_path / demo_name
        episode_dir.mkdir(exist_ok=True)

        # Find corresponding video file
        video_files = list(input_path.rglob(f"{demo_name}.MP4")) + list(
            input_path.rglob(f"{demo_name}.mp4")
        )

        if not video_files:
            raise ValueError(f"Video file not found for {demo_name}")

        video_file = video_files[0]

        # Process video frames
        frames_data = self._extract_frames(video_file, episode)

        # Save episode data
        episode_data = {
            "demo_name": demo_name,
            "frames": frames_data,
            "metadata": {
                "resolution": self.output_resolution,
                "fov": self.output_fov,
                "compression_level": self.compression_level,
                "episode_config": episode,
            },
        }

        episode_file = episode_dir / "replay_buffer.json.gz"
        with gzip.open(episode_file, "wt", compresslevel=self.compression_level) as f:
            json.dump(episode_data, f, indent=2)

        return {
            "demo_name": demo_name,
            "episode_file": str(episode_file),
            "frame_count": len(frames_data),
            "episode_dir": str(episode_dir),
        }

    def _extract_frames(
        self, video_file: Path, episode: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract and process frames from video."""
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_file}")

        frames = []
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to target resolution
            resized = cv2.resize(frame, self.output_resolution)

            # Convert to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Encode as base64 for JSON storage (simplified)
            _, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_data = encoded.tobytes()

            # Create frame metadata
            frame_info = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps,
                "image_shape": rgb_frame.shape,
                "camera_pose": self._get_camera_pose(frame_idx, episode),
                "action": self._get_action(frame_idx, episode),
            }

            frames.append(frame_info)
            frame_idx += 1

        cap.release()
        return frames

    def _get_camera_pose(self, frame_idx: int, episode: Dict[str, Any]) -> List[float]:
        """Get camera pose for frame (placeholder)."""
        # This would implement actual pose extraction
        return [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    def _get_action(self, frame_idx: int, episode: Dict[str, Any]) -> Dict[str, Any]:
        """Get action data for frame (placeholder)."""
        return {
            "gripper_position": [0.0, 0.0, 0.0],
            "gripper_rotation": [0.0, 0.0, 0.0, 1.0],
            "gripper_open": 0.5,
        }

    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of replay buffer generation."""
        episodes = results.get("episodes", [])

        return {
            "total_episodes": len(episodes),
            "total_frames": sum(ep.get("frame_count", 0) for ep in episodes),
            "total_size_mb": 0,  # Would calculate actual size
            "resolution": self.output_resolution,
            "compression_level": self.compression_level,
            "episodes": [
                {
                    "demo_name": ep["demo_name"],
                    "frame_count": ep["frame_count"],
                    "file_path": ep["episode_file"],
                }
                for ep in episodes
            ],
        }

    def validate_output(self, output_dir: str) -> bool:
        """Validate replay buffer generation."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return False

        summary_file = output_path / "replay_buffer_summary.json"
        if not summary_file.exists():
            return False

        with open(summary_file, "r") as f:
            summary = json.load(f)

        return summary.get("total_episodes", 0) > 0
