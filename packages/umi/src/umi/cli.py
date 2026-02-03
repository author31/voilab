import click
from umi.pipeline_executor import PipelineExecutor
from umi.services.visualize_slam_gui import VisualizeSLAMGUI


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_path")
@click.option("--session-dir", type=click.Path(exists=True), help="Override session directory from config file")
@click.option("--task", type=click.Choice(["kitchen", "living_room", "dining_room"]), help="Specify task type")
def run_slam_pipeline(config_path: str, session_dir: str, task:str):
    executor = PipelineExecutor(config_path, session_dir_override=session_dir, task_override=task)
    executor.execute_all()


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--session-dir", type=click.Path(exists=True), required=True,
              help="Session directory for SLAM output")
@click.option("--docker-image", default="chicheng/orb_slam3:latest",
              help="ORB-SLAM3 Docker image")
@click.option("--settings-file",
              help="ORB-SLAM3 settings file path")
@click.option("--force", is_flag=True,
              help="Force re-run even if GUI already running")
def visualize_slam_gui(video_path: str, session_dir: str, docker_image: str,
                       settings_file: str, force: bool):
    """Launch ORB-SLAM3 GUI for debugging specified video."""
    config = {
        "session_dir": session_dir,
        "video_path": video_path,
        "docker_image": docker_image,
        "slam_settings_file": settings_file,
        "force": force
    }

    try:
        service = VisualizeSLAMGUI(config)
        result = service.execute()

        if result["status"] == "completed":
            click.echo(f"SLAM GUI execution completed successfully")
        elif result["status"] == "interrupted":
            click.echo(f"SLAM GUI execution interrupted by user")
        else:
            click.echo(f"SLAM GUI execution failed with return code {result.get('return_code', 'unknown')}")

        click.echo(f"Session directory: {result['session_dir']}")
        click.echo(f"Video file: {result['video_path']}")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.ClickException(str(e))


if __name__ == "__main__":
    cli()
