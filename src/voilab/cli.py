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
    """Launch Isaac Sim simulation with specified robot and task environment."""

    try:
        click.echo(f"Preparing Isaac Sim for task: {task}")
        click.echo(f"Resolution: {width}x{height}")

        # Prepare environment variables for Docker
        env_vars = os.environ.copy()
        env_vars.update({
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "DISPLAY": os.getenv("DISPLAY", ":1"),
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "NVIDIA_DRIVER_CAPABILITIES": "all,graphics,display,utility,compute",
            "XAUTHORITY": os.getenv("XAUTHORITY", f"{os.getenv('HOME', '')}/.Xauthority"),
            "ROS_LOCALHOST_ONLY": "0",
            "ROS_DOMAIN_ID": "0",
            "TASK_NAME": task,
            "WINDOW_WIDTH": str(width),
            "WINDOW_HEIGHT": str(height)
        })

        # Build Docker command to use Isaac Sim's built-in Python launcher
        workspace_path = Path.cwd()
        docker_cmd = [
            "docker", "run",
            "--name", "isaac-sim",
            "--entrypoint", "bash",
            "--runtime=nvidia", "--gpus", "all",
            "-e", "ACCEPT_EULA=Y",
            "-e", "PRIVACY_CONSENT=Y",
            "-e", f"DISPLAY={env_vars['DISPLAY']}",
            "-e", f"VK_ICD_FILENAMES={env_vars['VK_ICD_FILENAMES']}",
            "-e", f"NVIDIA_DRIVER_CAPABILITIES={env_vars['NVIDIA_DRIVER_CAPABILITIES']}",
            "-e", f"XAUTHORITY={env_vars['XAUTHORITY']}",
            "-e", f"ROS_LOCALHOST_ONLY={env_vars['ROS_LOCALHOST_ONLY']}",
            "-e", f"ROS_DOMAIN_ID={env_vars['ROS_DOMAIN_ID']}",
            "-e", f"TASK_NAME={env_vars['TASK_NAME']}",
            "-e", f"WINDOW_WIDTH={env_vars['WINDOW_WIDTH']}",
            "-e", f"WINDOW_HEIGHT={env_vars['WINDOW_HEIGHT']}",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
            "-v", f"{workspace_path}:/workspace/voilab",
            "-v", "/usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro",
            "--network", "host",
            "--ipc", "host",
            "--rm", "-it",
            "nvcr.io/nvidia/isaac-sim:5.0.0",
            "/isaac-sim/python.sh", "/workspace/voilab/scripts/launch_isaacsim_workspace.py",
            "--task", task
        ]

        click.echo(f"Launching Isaac Sim in Docker container...")
        click.echo(f"Task: {task}")
        click.echo(f"Resolution: {width}x{height}")

        # Launch Docker container
        subprocess.run(docker_cmd, env=env_vars, check=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"Error launching Docker container: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error launching Isaac Sim: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
