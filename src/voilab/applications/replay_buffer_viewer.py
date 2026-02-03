import cv2
import numpy as np
import ipywidgets as w
from ipywidgets import IntSlider, Layout, VBox, HTML, HBox, Dropdown
import plotly.graph_objects as go

from umi.infrastructure.replay_buffer import ReplayBuffer


from imagecodecs.numcodecs import Jpegxl
from imagecodecs import numcodecs

numcodecs.register_codec(Jpegxl)


def show(dataset_path: str):
    """Interactive slider to navigate frames."""
    zarr_loader = ReplayBuffer.create_from_path(dataset_path)
    
    # Get episode boundaries
    episode_ends = zarr_loader.meta.episode_ends[:]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
    current_episode_idx = 0
    
    # Create episode dropdown options
    episode_options = [f"Episode {i+1} (frames {start}-{end})" 
                      for i, (start, end) in enumerate(zip(episode_starts, episode_ends))]

    image_widget = w.Image(
        format='jpeg',
        width=400,
        height=400
    )
    info_widget = HTML()
    episode_widget = HTML()
    trajectory_widget = go.FigureWidget(
        data=go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines+markers',
            marker=dict(size=5, color='cyan', symbol='diamond'),
            line=dict(width=4, color='magenta')
        )
    )
    
    # Create episode dropdown
    episode_dropdown = Dropdown(
        options=episode_options,
        value=episode_options[0],
        description='Episode:',
        layout=Layout(width='300px')
    )
    

    def plot_frame(frame_idx=0):
        """Create image and info widgets for the given frame."""
        frame_idx = min(frame_idx, zarr_loader.meta.episode_ends[-1])
        
        # Find current episode
        nonlocal current_episode_idx
        current_episode_idx = np.searchsorted(episode_ends, frame_idx, side='right')
        
        img = zarr_loader.data.camera0_rgb[frame_idx]
        eef = zarr_loader.data.robot0_eef_pos[frame_idx]
        rot = zarr_loader.data.robot0_eef_rot_axis_angle[frame_idx]
        grip = zarr_loader.data.robot0_gripper_width[frame_idx][0]
        start_pose = zarr_loader.data.robot0_demo_start_pose[frame_idx]
        end_pose = zarr_loader.data.robot0_demo_end_pose[frame_idx]

        # Convert numpy array to PIL Image and then to bytes for ipywidget.Image
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        _, buffer = cv2.imencode('.jpg', img[:, :, ::-1])
        image_widget.value = buffer.tobytes()

        # Create info text widget
        info_text = f"""
            <b>Frame {frame_idx}</b><br>
            <b>EEF Pos:</b> {np.round(eef, 3)}<br>
            <b>EEF Rot:</b> {np.round(rot, 3)}<br>
            <b>Gripper:</b> {np.round(grip, 3)}<br>
            <b>Start Pose:</b> {np.round(start_pose, 3)}<br>
            <b>End Pose:</b> {np.round(end_pose, 3)}
            """

        info_widget.value = info_text
        
        # Update episode info
        episode_text = f"""
            <b>Episode {current_episode_idx + 1}/{len(episode_ends)}</b><br>
            <b>Episode Start:</b> {episode_starts[current_episode_idx]}<br>
            <b>Episode End:</b> {episode_ends[current_episode_idx]}<br>
            <b>Episode Length:</b> {episode_ends[current_episode_idx] - episode_starts[current_episode_idx] + 1}
        """
        episode_widget.value = episode_text

    def plot_trajectory(frame_idx=0):
        with trajectory_widget.batch_update():
            # Get current episode start frame
            episode_start = episode_starts[current_episode_idx]
            
            # Only show trajectory from current episode start to current frame
            eef = zarr_loader.data.robot0_eef_pos[episode_start:frame_idx+1]
            trajectory_widget.data[0].x = eef[:, 0]
            trajectory_widget.data[0].y = eef[:, 1]
            trajectory_widget.data[0].z = eef[:, 2]


    def show_frame(frame_idx):
        frame_idx = min(frame_idx, zarr_loader.meta.episode_ends[-1]-1)
        plot_frame(frame_idx)
        plot_trajectory(frame_idx)

    def on_episode_change(change):
        """Handle episode dropdown change."""
        selected_episode = episode_dropdown.index
        current_episode_idx = selected_episode
        slider.min = episode_starts[selected_episode]
        slider.max = episode_ends[selected_episode]
        slider.value = episode_starts[selected_episode]
        show_frame(episode_starts[selected_episode])

    slider = IntSlider(
        min=0, 
        max=zarr_loader.meta.episode_ends[-1] - 1, 
        step=1, value=0, description="Frame", layout=Layout(width="80%")
    )
    slider.observe(lambda change: show_frame(change['new']), names='value')
    episode_dropdown.observe(on_episode_change, names='value')

    # Create UI layout
    controls = VBox([
        episode_dropdown,
        HBox([image_widget, trajectory_widget]), 
        HBox([info_widget, episode_widget]), 
        slider
    ])

    show_frame(0)
    display(controls)
