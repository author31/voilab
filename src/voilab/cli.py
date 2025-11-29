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
@click.option('--headless', is_flag=True, default=True, 
              help='Run in headless mode (ROS2 only)')
def launch_simulator(task, width, height, headless):
    """Launch Isaac Sim with ROS2 bridge enabled"""
    
    try:
        # Prepare environment
        env_vars = os.environ.copy()
        env_vars.update({
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "DISPLAY": os.getenv("DISPLAY", ":1"),
            
            # GPU Configuration
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "all,graphics,display,utility,compute",
            
            # ROS2 Configuration
            "ROS_LOCALHOST_ONLY": "0",
            "ROS_DOMAIN_ID": "0",
            "ROS_DISTRO": "humble",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            
            # Library paths
            "LD_LIBRARY_PATH": "/opt/ros/humble/lib:/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib",
            
            # Task parameters
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

if __name__ == "__main__":
    cli()
