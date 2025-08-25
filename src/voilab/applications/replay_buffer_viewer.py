import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Layout

from ..utils.replay_buffer_loader import ReplayBufferLoader



def show(dataset_path: str):
    """Interactive slider to navigate frames."""
    zarr_loader = ReplayBufferLoader(dataset_path)

    def plot_frame(frame_idx):
        """Plot image frame with basic info."""
        img = zarr_loader.imgs[frame_idx]
        eef = zarr_loader.eef_pos[frame_idx]
        rot = zarr_loader.eef_rot[frame_idx]
        grip = zarr_loader.gripper_width[frame_idx][0]
        start_pose = zarr_loader.demo_start_pose[frame_idx]
        end_pose = zarr_loader.demo_end_pose[frame_idx]

        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"Frame {frame_idx}\n"
            f"EEF Pos: {np.round(eef, 3)}\n"
            f"EEF Rot: {np.round(rot, 3)}\n"
            f"Gripper: {np.round(grip, 3)}\n"
            f"Start Pose: {np.round(start_pose, 3)}\n"
            f"End Pose: {np.round(end_pose, 3)}"
        )
        plt.show()

    def show_frame(frame_idx=0):
        frame_idx = min(frame_idx, zarr_loader.total_frames - 1)
        plot_frame(frame_idx)


    interact(
        show_frame,
        frame_idx=IntSlider(
            min=0, max=zarr_loader.total_frames - 1, step=1, value=0,
            description="Frame", layout=Layout(width="80%")
        )
    )
