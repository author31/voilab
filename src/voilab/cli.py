import os
import click
import subprocess
import sys
from pathlib import Path

from voila.app import Voila


@click.group()
def cli():
    pass


@cli.command()
def launch_viewer():
    """Launch the replay buffer viewer."""
    argv = ["--no-browser", "nbs/replay_buffer_viewer.ipynb"]
    Voila.launch_instance(argv=argv)


@cli.command()
def launch_dataset_visualizer():
    """Launch the dataset visualizer for reviewing collected demonstrations."""
    argv = ["--no-browser", "nbs/dataset_visualizer.ipynb"]
    Voila.launch_instance(argv=argv)


@cli.command()
@click.option('--task', type=click.Choice(['kitchen', 'dining-table', 'living-room']),
              required=True, help='Task environment to load')
@click.option('--width', default=1280, help='Window width')
@click.option('--height', default=720, help='Window height')
def launch_simulator(task, width, height):
    """Launch Isaac Sim with ROS2 bridge enabled"""
    
    try:
        # Prepare environment
        env_vars = os.environ.copy()
        env_vars.update({
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "DISPLAY": os.getenv("DISPLAY", ":1"),
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "all,graphics,display,utility,compute",
            "ROS_LOCALHOST_ONLY": "0",
            "ROS_DOMAIN_ID": "0",
            "ROS_DISTRO": "humble",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            "TASK_NAME": task,
            "WINDOW_WIDTH": str(width),
            "WINDOW_HEIGHT": str(height),
        })
        
        click.echo(f"[CLI] Launching Isaac Sim + ROS2: task={task}, resolution={width}x{height}")
        
        # Build image
        click.echo("[CLI] Building Docker image...")
        build_cmd = ["docker", "compose", "build", "isaac-sim"]
        subprocess.run(build_cmd, env=env_vars, check=True)
        
        # Run container with host network
        click.echo("[CLI] Starting Docker container with host network...")
        compose_run_cmd = [
            "docker", "compose", "run",
            "-e", "OMNI_KIT_ACCEPT_EULA=yes",
            "--rm",
            "isaac-sim",
            "python3",
            "/workspace/voilab/scripts/launch_isaacsim_workspace.py",
            "--task", task
        ]
        subprocess.run(compose_run_cmd, env=env_vars, check=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"[ERROR] Docker execution failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"[ERROR] {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--session-dir', type=str, required=True, help='Path to the session directory to replay')
def replay_trajectory(session_dir):
    """Replay a trajectory from a recorded session directory."""
    try:
        # Validate session directory
        if not session_dir:
            click.echo("[ERROR] Session directory is required", err=True)
            sys.exit(1)

        # Prepare environment
        env_vars = os.environ.copy()

        click.echo(f"[CLI] Replaying trajectory from session: {session_dir}")

        # Run pose publisher in docker container
        compose_docker_cmd = [
            "docker", "compose", "run", "voilab-workspace",
            "python", "packages/diffusion_policy/examples/run_dataset_pose_publisher.py",
            "--session_dir", session_dir
        ]

        subprocess.run(compose_docker_cmd, env=env_vars, check=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"[ERROR] Docker execution failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"[ERROR] {str(e)}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()
