from pathlib import Path
import shutil

from .base_service import BaseService

import os
from exiftool import ExifToolHelper
from umi.common.timecode_util import mp4_get_start_datetime


class VideoOrganizationService(BaseService):
    """Service for organizing raw videos into structured demo directories."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.session_dir = self.config.get("session_dir")
        self.input_patterns = self.config.get("input_patterns", ["*.MP4", "*.mp4"])

    def execute(self) -> dict:
        assert self.session_dir, "Missing session_dir in pipeline configuration."

        session = Path(self.session_dir).absolute()
        input_dir = session.joinpath("raw_videos")
        output_dir = session.joinpath("demos")
        output_dir.mkdir(parents=True, exist_ok=True)

        result_summary = {
            "moved_to_raw_videos": 0,
            "mapping_video": None,
            "gripper_calibration": {},
            "organized_demos": 0,
        }

        # --- 1. Ensure raw_videos exists and move MP4s there ---
        if not input_dir.is_dir():
            input_dir.mkdir()
            for pattern in self.input_patterns:
                for mp4_path in session.glob(f"**/{pattern}"):
                    shutil.move(mp4_path, input_dir / mp4_path.name)
                    result_summary["moved_to_raw_videos"] += 1

        # --- 2. Ensure mapping.mp4 exists ---
        mapping_vid_path = input_dir / "mapping.mp4"
        if not mapping_vid_path.exists() and not mapping_vid_path.is_symlink():
            largest_file = max(
                (p for pattern in self.input_patterns for p in input_dir.glob(f"**/{pattern}")),
                key=lambda p: p.stat().st_size,
                default=None,
            )
            if largest_file:
                shutil.move(largest_file, mapping_vid_path)
                result_summary["mapping_video"] = mapping_vid_path.name

        # --- 3. Ensure gripper_calibration folder ---
        gripper_cal_dir = input_dir / "gripper_calibration"
        if not gripper_cal_dir.is_dir():
            gripper_cal_dir.mkdir()
            serial_start_dict, serial_path_dict = {}, {}

            with ExifToolHelper() as et:
                for pattern in self.input_patterns:
                    for mp4_path in input_dir.glob(f"**/{pattern}"):
                        if mp4_path.name.startswith("map"):
                            continue
                        start_date = mp4_get_start_datetime(str(mp4_path))
                        meta = list(et.get_metadata(str(mp4_path)))[0]
                        cam_serial = meta["QuickTime:CameraSerialNumber"]

                        if cam_serial not in serial_start_dict or start_date < serial_start_dict[cam_serial]:
                            serial_start_dict[cam_serial] = start_date
                            serial_path_dict[cam_serial] = mp4_path

            for serial, path in serial_path_dict.items():
                shutil.move(path, gripper_cal_dir / path.name)
                result_summary["gripper_calibration"][serial] = path.name

        # --- 4. Organize videos into demo directories ---
        input_mp4_paths = [p for pattern in self.input_patterns for p in input_dir.glob(f"**/{pattern}")]

        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                if mp4_path.is_symlink():
                    continue

                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta["QuickTime:CameraSerialNumber"]

                # Naming logic
                if mp4_path.name.startswith("mapping"):
                    out_dname = "mapping"
                elif mp4_path.name.startswith("gripper_cal") or mp4_path.parent.name.startswith("gripper_cal"):
                    out_dname = f"gripper_calibration_{cam_serial}_{start_date.strftime('%Y.%m.%d_%H.%M.%S.%f')}"
                else:
                    out_dname = f"demo_{cam_serial}_{start_date.strftime('%Y.%m.%d_%H.%M.%S.%f')}"

                this_out_dir = output_dir / out_dname
                this_out_dir.mkdir(parents=True, exist_ok=True)

                # Move video and create symlink
                out_video_path = this_out_dir / "raw_video.mp4"
                shutil.move(mp4_path, out_video_path)
                dots = os.path.join(*[".."] * len(mp4_path.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                mp4_path.symlink_to(os.path.join(dots, rel_path))

                result_summary["organized_demos"] += 1

        return result_summary
