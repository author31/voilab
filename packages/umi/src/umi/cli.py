import click
from umi.pipeline_executor import PipelineExecutor


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_path")
@click.option("--session-dir", type=click.Path(exists=True), help="Override session directory from config file")
def run_slam_pipeline(config_path: str, session_dir: str):
    executor = PipelineExecutor(config_path, session_dir_override=session_dir)
    executor.execute_all()


if __name__ == "__main__":
    cli()
