import json
from pathlib import Path

from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor

from .base_service import BaseService

SECS_TO_MS = 1e3

class IMUExtractionService(BaseService):
    """Service for extracting IMU data from GoPro videos."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.session_dir = self.config.get('session_dir')
        self.num_workers = self.config.get('num_workers') or self._get_num_workers()
        self.stream_types = self.config.get(
            'stream_types', 
            ["ACCL", "GYRO", "GPS5", "GPSP", "GPSU", "GPSF", "GRAV", "MAGN", "CORI", "IORI", "TMPC"]
        )

    def execute(self) -> dict:
        """
        Extract IMU data from videos in input directory.

        Args:
            input_dir: Directory containing organized demo videos
            output_dir: Directory for extracted IMU data

        Returns:
            dict: Extraction results with paths to IMU files
        """
        assert self.session_dir, "Missing session_dir from the configuration."
        input_path = Path(self.session_dir) / "demos"

        results = {"extracted": [], "failed": []}

        # Find all demo directories
        demo_dirs = [d for d in input_path.iterdir() if d.is_dir()]

        for demo_dir in demo_dirs:
            # Find video files
            try:
                imu_file = self._extract_imu_from_video(demo_dir)
                if imu_file:
                    results["extracted"].append({
                        "imu_file": str(imu_file),
                        "demo": demo_dir.name
                    })
            except Exception as e:
                results["failed"].append({
                    "path": str(demo_dir),
                    "error": str(e)
                })

        return results

    def _extract_imu_from_video(self, video_dir: str | Path):
        """Extract IMU data from a single video directory using py_gpmf_parser."""
        src = Path(video_dir).absolute()
        video_path = src / 'raw_video.mp4'
        output_path = src / 'imu_data.json'

        if not video_path.exists():
            raise FileNotFoundError(f"raw_video.mp4 not found in {video_dir}")

        extractor = GoProTelemetryExtractor(str(video_path))
        try:
            extractor.open_source()

            output = {
                "1": {
                    "streams": {},
                },
                "frames/second": 0.0 # TODO: update
            }

            for stream in self.stream_types:
                payload = extractor.extract_data(stream)
                if payload and len(payload[0]) > 0:
                    output["1"]["streams"][stream] = {
                        "samples": [{"value": data.tolist(), "cts": (ts*SECS_TO_MS).tolist()} for data, ts in zip(*payload)]
                    }

            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)

            return True

        except Exception as e:
            print(f"Error processing {video_dir}: {str(e)}")
            return False

        finally:
            extractor.close_source()

