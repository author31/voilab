"""Dataset Visualizer application for reviewing UMI pipeline results.

This module provides an interactive UI for data collectors to review
and refine their collected human demonstrations. It displays:
- Pipeline stage status
- Demo quality metrics
- SLAM trajectory visualization
- Issue identification
"""

import cv2
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from ipywidgets import (
    HTML,
    Dropdown,
    IntSlider,
    Layout,
    Output,
    Tab,
    VBox,
    Image,
)

from ..utils.dataset_loader import DatasetLoader


def get_status_icon(status: str) -> str:
    """Get an emoji icon for a status."""
    icons = {
        "complete": "✅",
        "partial": "⚠️",
        "pending": "⏳",
        "warning": "⚠️",
        "bad": "❌",
        "good": "✅",
        "unknown": "❓",
    }
    return icons.get(status, "❓")


def create_pipeline_status_html(stages: dict) -> str:
    """Create HTML table showing pipeline status."""
    rows = []
    for stage_id, stage_info in stages.items():
        icon = get_status_icon(stage_info["status"])
        name = stage_info["name"]
        status = stage_info["status"].capitalize()

        # Create details string
        details = stage_info.get("details", {})
        details_str = ", ".join(f"{k}: {v}" for k, v in details.items())

        rows.append(
            f"""
            <tr>
                <td style="padding: 8px;">{stage_id}</td>
                <td style="padding: 8px;">{name}</td>
                <td style="padding: 8px;">{icon} {status}</td>
                <td style="padding: 8px; font-size: 0.9em; color: #666;">{details_str}</td>
            </tr>
        """
        )

    return f"""
    <style>
        .pipeline-table {{ border-collapse: collapse; width: 100%; }}
        .pipeline-table th, .pipeline-table td {{ border: 1px solid #ddd; text-align: left; }}
        .pipeline-table th {{ background-color: #4CAF50; color: white; padding: 12px; }}
        .pipeline-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .pipeline-table tr:hover {{ background-color: #ddd; }}
    </style>
    <table class="pipeline-table">
        <thead>
            <tr>
                <th>Stage ID</th>
                <th>Stage Name</th>
                <th>Status</th>
                <th>Details</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """


def create_demos_table_html(demos: list) -> str:
    """Create HTML table showing demo information."""
    if not demos:
        return "<p>No demos found.</p>"

    rows = []
    for demo in demos:
        quality_icon = get_status_icon(demo.trajectory_quality)
        video_icon = "📹" if demo.has_video else "❌"
        traj_icon = "📈" if demo.has_camera_trajectory else "❌"
        tags_icon = "🏷️" if demo.has_tag_detection else "❌"

        detection_rate_str = f"{demo.detection_rate:.1%}"
        lost_frames_color = "red" if demo.n_lost_frames > 10 else "orange" if demo.n_lost_frames > 0 else "green"

        rows.append(
            f"""
            <tr>
                <td style="padding: 8px;">{demo.name}</td>
                <td style="padding: 8px;">{video_icon}</td>
                <td style="padding: 8px;">{traj_icon}</td>
                <td style="padding: 8px;">{tags_icon}</td>
                <td style="padding: 8px;">{demo.n_frames}</td>
                <td style="padding: 8px; color: {lost_frames_color};">{demo.n_lost_frames}</td>
                <td style="padding: 8px;">{detection_rate_str}</td>
                <td style="padding: 8px;">{quality_icon}</td>
            </tr>
        """
        )

    return f"""
    <style>
        .demos-table {{ border-collapse: collapse; width: 100%; }}
        .demos-table th, .demos-table td {{ border: 1px solid #ddd; text-align: left; }}
        .demos-table th {{ background-color: #2196F3; color: white; padding: 12px; }}
        .demos-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .demos-table tr:hover {{ background-color: #ddd; }}
    </style>
    <table class="demos-table">
        <thead>
            <tr>
                <th>Demo Name</th>
                <th>Video</th>
                <th>Trajectory</th>
                <th>Tags</th>
                <th>Frames</th>
                <th>Lost Frames</th>
                <th>Tag Detection Rate</th>
                <th>Quality</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """


def create_summary_html(loader: DatasetLoader) -> str:
    """Create summary HTML for the session."""
    info = loader.load()

    n_good = sum(1 for d in info.demos if d.trajectory_quality == "good")
    n_warning = sum(1 for d in info.demos if d.trajectory_quality == "warning")
    n_bad = sum(1 for d in info.demos if d.trajectory_quality == "bad")
    n_unknown = sum(1 for d in info.demos if d.trajectory_quality == "unknown")

    cal_status = "✅ Complete" if info.calibration.has_slam_tag else "❌ Incomplete"
    plan_status = f"✅ {info.dataset_plan.n_episodes} episodes" if info.dataset_plan.has_plan else "❌ Not generated"

    return f"""
    <div style="background-color: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
        <h3>📊 Session Summary</h3>
        <p><strong>Session Path:</strong> {info.session_path}</p>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4>📁 Demos</h4>
                <p>Total: <strong>{info.n_demos}</strong></p>
                <p>✅ Good: <strong>{n_good}</strong></p>
                <p>⚠️ Warning: <strong>{n_warning}</strong></p>
                <p>❌ Bad: <strong>{n_bad}</strong></p>
                <p>❓ Unknown: <strong>{n_unknown}</strong></p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4>🔧 Calibration</h4>
                <p>SLAM Tag: {cal_status}</p>
                <p>Gripper Calibrations: <strong>{len(info.calibration.gripper_calibrations)}</strong></p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4>📋 Dataset Plan</h4>
                <p>{plan_status}</p>
                <p>Total Frames: <strong>{info.dataset_plan.total_frames:,}</strong></p>
            </div>
        </div>
    </div>
    """


def create_trajectory_plot(df: pd.DataFrame) -> go.FigureWidget:
    """Create a 3D trajectory plot from trajectory DataFrame."""
    # Filter out lost frames
    valid_df = df[~df["is_lost"]]

    fig = go.FigureWidget()

    # Add trajectory line
    fig.add_trace(
        go.Scatter3d(
            x=valid_df["x"],
            y=valid_df["y"],
            z=valid_df["z"],
            mode="lines",
            name="Trajectory",
            line=dict(color="blue", width=3),
        )
    )

    # Add start point
    if len(valid_df) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=[valid_df["x"].iloc[0]],
                y=[valid_df["y"].iloc[0]],
                z=[valid_df["z"].iloc[0]],
                mode="markers",
                name="Start",
                marker=dict(color="green", size=10),
            )
        )

        # Add end point
        fig.add_trace(
            go.Scatter3d(
                x=[valid_df["x"].iloc[-1]],
                y=[valid_df["y"].iloc[-1]],
                z=[valid_df["z"].iloc[-1]],
                mode="markers",
                name="End",
                marker=dict(color="red", size=10),
            )
        )

    # Add lost frame markers
    lost_df = df[df["is_lost"]]
    if len(lost_df) > 0:
        # For lost frames, use the last known position
        fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                name=f"Lost Frames ({len(lost_df)})",
                marker=dict(color="orange", size=5, symbol="x"),
            )
        )

    fig.update_layout(
        title="Camera Trajectory",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=500,
    )

    return fig


def show(session_dir: str):
    """Show the dataset visualizer for a session directory.

    Args:
        session_dir: Path to the session directory
    """
    # Initialize loader
    loader = DatasetLoader(session_dir)
    info = loader.load()
    stages = loader.get_pipeline_stages_status()

    # Create output widgets
    summary_widget = HTML()
    pipeline_widget = HTML()
    demos_widget = HTML()
    trajectory_output = Output()

    # Update summary and tables
    summary_widget.value = create_summary_html(loader)
    pipeline_widget.value = create_pipeline_status_html(stages)
    demos_widget.value = create_demos_table_html(info.demos)

    # Create demo selector
    demo_options = ["(Select a demo)"] + [d.name for d in info.demos]
    if info.mapping:
        demo_options.append(info.mapping.name)
    for gc in info.gripper_calibrations:
        demo_options.append(gc.name)

    demo_dropdown = Dropdown(
        options=demo_options,
        value=demo_options[0],
        description="Select Demo:",
        layout=Layout(width="400px"),
    )

    # Create image widget for video frame display
    image_widget = Image(format="jpeg", width=400, height=300)
    frame_slider = IntSlider(
        min=0, max=100, step=1, value=0, description="Frame", layout=Layout(width="400px")
    )
    frame_info = HTML()

    # Video capture for demo video
    video_cap = {"cap": None, "n_frames": 0}

    def update_trajectory(demo_name: str):
        """Update trajectory plot for selected demo."""
        with trajectory_output:
            trajectory_output.clear_output()
            if demo_name == "(Select a demo)":
                print("Please select a demo to view its trajectory.")
                return

            df = loader.get_demo_trajectory(demo_name)
            if df is not None:
                fig = create_trajectory_plot(df)
                display(fig)
            else:
                print(f"No trajectory data found for {demo_name}")

    def update_video_viewer(demo_name: str):
        """Update video viewer for selected demo."""
        # Close previous capture
        if video_cap["cap"] is not None:
            video_cap["cap"].release()
            video_cap["cap"] = None

        if demo_name == "(Select a demo)":
            frame_info.value = "Please select a demo to view its video."
            return

        # Find demo info
        demo_info = None
        for d in info.demos + info.gripper_calibrations + ([info.mapping] if info.mapping else []):
            if d and d.name == demo_name:
                demo_info = d
                break

        if demo_info is None or not demo_info.has_video:
            frame_info.value = f"No video found for {demo_name}"
            return

        # Try to open video
        video_path = demo_info.path / "raw_video.mp4"
        converted_path = demo_info.path / "converted_60fps_raw_video.mp4"

        if converted_path.exists():
            video_path = converted_path
        elif not video_path.exists():
            frame_info.value = f"Video file not found: {video_path}"
            return

        try:
            cap = cv2.VideoCapture(str(video_path))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_cap["cap"] = cap
            video_cap["n_frames"] = n_frames

            frame_slider.max = max(0, n_frames - 1)
            frame_slider.value = 0
            update_frame(0)
        except Exception as e:
            frame_info.value = f"Error opening video: {e}"

    def update_frame(frame_idx: int):
        """Update displayed frame."""
        cap = video_cap["cap"]
        if cap is None:
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # Resize for display
            h, w = frame.shape[:2]
            scale = 400 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h))

            # Convert to JPEG
            _, buffer = cv2.imencode(".jpg", frame_resized)
            image_widget.value = buffer.tobytes()

            frame_info.value = f"Frame: {frame_idx + 1}/{video_cap['n_frames']}"
        else:
            frame_info.value = f"Error reading frame {frame_idx}"

    def on_demo_change(change):
        """Handle demo selection change."""
        demo_name = change["new"]
        update_trajectory(demo_name)
        update_video_viewer(demo_name)

    def on_frame_change(change):
        """Handle frame slider change."""
        update_frame(change["new"])

    demo_dropdown.observe(on_demo_change, names="value")
    frame_slider.observe(on_frame_change, names="value")

    # Create tabs
    tab = Tab()
    tab.children = [
        VBox([summary_widget, pipeline_widget]),
        VBox([demos_widget]),
        VBox(
            [
                demo_dropdown,
                trajectory_output,
            ]
        ),
        VBox(
            [
                demo_dropdown,
                image_widget,
                frame_slider,
                frame_info,
            ]
        ),
    ]
    tab.set_title(0, "Overview")
    tab.set_title(1, "Demos")
    tab.set_title(2, "Trajectory")
    tab.set_title(3, "Video")

    # Show initial state
    display(tab)
