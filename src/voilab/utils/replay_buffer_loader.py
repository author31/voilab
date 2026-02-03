import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Layout
from IPython.display import display

try:
    from zarr.storage import ZipStore
except ImportError:
    ZipStore = None


class ReplayBufferLoader:
    """
    Viewer for replay buffer datasets with the latest structure:
    /data
      ├── camera0_rgb (T, H, W, 3) uint8
      ├── robot0_demo_end_pose (T, 6)
      ├── robot0_demo_start_pose (T, 6)
      ├── robot0_eef_pos (T, 3)
      ├── robot0_eef_rot_axis_angle (T, 3)
      └── robot0_gripper_width (T, 1)
    """

    def __init__(self, zarr_path: str):
        self.zarr_path = zarr_path
        self.store = self._open_store(zarr_path)

        # Load datasets under /data
        self.data = self.store["data"]
        self.imgs = self.data["camera0_rgb"]
        self.eef_pos = self.data["robot0_eef_pos"]
        self.eef_rot = self.data["robot0_eef_rot_axis_angle"]
        self.gripper_width = self.data["robot0_gripper_width"]
        self.demo_start_pose = self.data["robot0_demo_start_pose"]
        self.demo_end_pose = self.data["robot0_demo_end_pose"]

        self.total_frames = self.imgs.shape[0]

    def _open_store(self, path):
        """Handles zipped and unzipped zarr paths."""
        if os.path.isdir(path):
            return zarr.open(path, mode="r")
        elif path.endswith(".zip"):
            if ZipStore is None:
                raise ImportError(
                    "Your zarr version does not support ZipStore. Run: pip install 'zarr<3' OR unzip the archive."
                )
            return zarr.open(ZipStore(path, mode="r"), mode="r")
        else:
            raise ValueError(f"Unrecognized Zarr path format: {path}")
