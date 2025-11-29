import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, Layout, VBox, HTML, Output, Text, Checkbox, Button, HBox, Image
from pathlib import Path
import io
import base64

from ..utils.aruco_detection_loader import ArUcoDetectionLoader


def draw_aruco_markers(img: np.ndarray, tag_dict: dict) -> np.ndarray:
    """Draw ArUco markers and their poses on the image."""
    img_with_markers = img.copy()

    for tag_id, tag_data in tag_dict.items():
        # Draw marker corners with thicker lines
        corners = tag_data['corners'].astype(int)
        cv2.polylines(img_with_markers, [corners], True, (0, 255, 0), 15)

        # Draw marker ID with larger font
        center = np.mean(corners, axis=0).astype(int)
        cv2.putText(img_with_markers, f"ID: {tag_id}",
                    (center[0] - 30, center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

        # Draw coordinate axes if pose is available
        if 'rvec' in tag_data and 'tvec' in tag_data:
            # Create axis points (15cm length for better visibility)
            axis_length = 0.15
            axis_points = np.array([
                [0, 0, 0],
                [axis_length, 0, 0],  # X-axis (red)
                [0, axis_length, 0],  # Y-axis (green)
                [0, 0, axis_length]   # Z-axis (blue)
            ])

            # Project 3D points to 2D
            # Note: This is a simplified projection. For accurate results,
            # you would need the camera intrinsics matrix
            rvec = tag_data['rvec']
            tvec = tag_data['tvec']

            # Simple orthographic projection for visualization
            projected_points = []
            for point in axis_points:
                # Rotate and translate
                rotated = cv2.Rodrigues(rvec)[0] @ point + tvec
                # Simple projection (assuming camera at origin looking down Z) - increased scale for visibility
                x_2d = int(center[0] + rotated[0] * 1500)
                y_2d = int(center[1] + rotated[1] * 1500)
                projected_points.append([x_2d, y_2d])

            projected_points = np.array(projected_points)

            # Draw axes with thicker lines and larger labels
            origin = projected_points[0].astype(int)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB
            labels = ['X', 'Y', 'Z']

            for i in range(1, 4):
                end_point = projected_points[i].astype(int)
                # Draw axis lines with increased thickness
                cv2.line(img_with_markers, origin, end_point, colors[i-1], 6)
                # Draw larger axis labels
                cv2.putText(img_with_markers, labels[i-1],
                            end_point + np.array([8, -8]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i-1], 2)

            # Draw origin point for better visibility
            cv2.circle(img_with_markers, origin, 5, (255, 255, 255), -1)

    return img_with_markers


def show(directory_path: str, figsize=(8, 6), dpi=100):
    """Interactive slider to navigate frames with ArUco detection visualization.

    Args:
        directory_path: Path to directory containing raw_video.mp4 and tag_detection.pkl
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for display
    """
    # UI Controls
    camera_intrinsics_widget = Text(
        value="",
        placeholder="Path to camera intrinsics JSON file",
        description="Camera Intrinsics:",
        layout=Layout(width="500px")
    )

    aruco_config_widget = Text(
        value="",
        placeholder="Path to ArUco config YAML file",
        description="ArUco Config:",
        layout=Layout(width="500px")
    )

    undistort_checkbox = Checkbox(
        value=False,
        description="Show undistorted image"
    )

    rerun_checkbox = Checkbox(
        value=False,
        description="Re-run detection"
    )

    load_params_button = Button(
        description="Load Parameters",
        button_style="info"
    )

    # Initialize loader
    loader = ArUcoDetectionLoader(directory_path)

    # Create output widget for image display and status
    image_widget = Image(
        format='jpeg',
        width=figsize[0] * dpi,
        height=figsize[1] * dpi
    )

    info_widget = HTML()
    status_output = Output()

    def load_parameters(_):
        """Load camera intrinsics and ArUco config."""
        with status_output:
            status_output.clear_output()
            try:
                camera_path = camera_intrinsics_widget.value.strip()
                aruco_path = aruco_config_widget.value.strip()

                # Create new loader with parameters
                new_loader = ArUcoDetectionLoader(
                    directory_path,
                    camera_intrinsics_path=camera_path if camera_path else None,
                    aruco_config_path=aruco_path if aruco_path else None
                )

                # Update the loader reference
                nonlocal loader
                loader = new_loader

                print("✓ Parameters loaded successfully")
                if camera_path:
                    print(f"  Camera intrinsics: {camera_path}")
                if aruco_path:
                    print(f"  ArUco config: {aruco_path}")

            except Exception as e:
                print(f"✗ Error loading parameters: {e}")

    load_params_button.on_click(load_parameters)

    def plot_frame(frame_idx):
        """Plot video frame with ArUco detections."""
        try:
            # Get frame with optional undistortion
            img = loader.get_frame(
                frame_idx, undistorted=undistort_checkbox.value)

            # Get detection with optional re-run
            detection = loader.get_detection(
                frame_idx, rerun=rerun_checkbox.value)

            if detection and detection.get('tag_dict'):
                # Draw markers on the image
                img_with_markers = draw_aruco_markers(
                    img, detection['tag_dict'])

                # Create info text
                tag_dict = detection['tag_dict']
                mode_info = []
                if undistort_checkbox.value:
                    mode_info.append("Undistorted")
                if rerun_checkbox.value:
                    mode_info.append("Re-run Detection")

                info_lines = [
                    f"Frame: {frame_idx} | Time: {detection['time']:.3f}s",
                    f"Detected markers: {len(tag_dict)}",
                ]

                if mode_info:
                    info_lines.append(f"Mode: {', '.join(mode_info)}")

                info_lines.append("")

                for tag_id, tag_data in tag_dict.items():
                    rvec = tag_data.get('rvec', np.zeros(3))
                    tvec = tag_data.get('tvec', np.zeros(3))
                    info_lines.extend([
                        f"Marker {tag_id}:",
                        f"  Position: ({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f})",
                        f"  Rotation: ({rvec[0]:.3f}, {rvec[1]:.3f}, {rvec[2]:.3f})"
                    ])

                info_text = '<br>'.join(info_lines)
                info_widget.value = f'<pre style="background-color: wheat; padding: 10px; border-radius: 5px;">{info_text}</pre>'

                # Convert image to JPEG and display
                _, buffer = cv2.imencode('.jpg', img_with_markers[:, :, ::-1])
                image_widget.value = buffer.tobytes()

            else:
                mode_info = []
                if undistort_checkbox.value:
                    mode_info.append("Undistorted")
                if rerun_checkbox.value:
                    mode_info.append("Re-run Detection")

                frame_text = f"Frame: {frame_idx}"
                if mode_info:
                    frame_text += f" ({', '.join(mode_info)})"
                frame_text += "\nNo markers detected"

                info_widget.value = f'<pre style="background-color: lightcoral; padding: 10px; border-radius: 5px;">{frame_text}</pre>'

                # Convert image to JPEG and display
                _, buffer = cv2.imencode('.jpg', img)
                image_widget.value = buffer.tobytes()

        except Exception as e:
            info_widget.value = f'<pre style="color: red;">Error: {e}</pre>'

    def show_frame(frame_idx):
        frame_idx = min(frame_idx, loader.total_frames - 1)
        plot_frame(frame_idx)

    # Create UI layout
    controls = VBox([
        HBox([camera_intrinsics_widget, aruco_config_widget]),
        HBox([undistort_checkbox, rerun_checkbox, load_params_button]),
        status_output
    ])

    # Create slider
    slider = IntSlider(
        min=0, max=loader.total_frames - 1, step=1, value=0,
        description="Frame", layout=Layout(width="80%")
    )

    # Connect slider to update function
    slider.observe(lambda change: show_frame(change['new']), names='value')

    # Connect checkboxes to update function
    undistort_checkbox.observe(
        lambda change: show_frame(slider.value), names='value')
    rerun_checkbox.observe(
        lambda change: show_frame(slider.value), names='value')

    # Initial display
    stats = loader.get_detections_stats()
    with status_output:
        print(f"Loaded {loader.total_frames} frames")
        print(f"Detection rate: {stats['detection_rate']:.2%}")
        print(f"Unique marker IDs: {stats['unique_marker_ids']}")

    show_frame(0)

    # Display widgets
    display(VBox([controls, slider, image_widget, info_widget]))


def show_batch(directory_path: str, frame_indices: list, subplot_size=(3, 2.5), dpi=100):
    """Show multiple frames in a grid layout.

    Args:
        directory_path: Path to directory containing raw_video.mp4 and tag_detection.pkl
        frame_indices: List of frame indices to display
        subplot_size: Size of each subplot as (width, height) in inches
        dpi: Dots per inch for display
    """
    loader = ArUcoDetectionLoader(directory_path)

    n_frames = len(frame_indices)
    cols = min(4, n_frames)
    rows = (n_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(
        subplot_size[0]*cols, subplot_size[1]*rows), dpi=dpi)
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, frame_idx in enumerate(frame_indices):
        row = i // cols
        col = i % cols

        img = loader.get_frame(frame_idx)
        detection = loader.get_detection(frame_idx)

        if detection and detection.get('tag_dict'):
            img = draw_aruco_markers(img, detection['tag_dict'])
            title = f"Frame {frame_idx}\n{len(detection['tag_dict'])} markers"
        else:
            title = f"Frame {frame_idx}\nNo markers"

        axes[row, col].imshow(img)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    # Hide empty subplots
    for i in range(n_frames, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()
