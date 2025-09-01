import click
from umi.pipeline_executor import PipelineExecutor

@click.group()
def cli():
    pass

@cli.command()
@click.argument("config_path")
def run_slam_pipeline(config_path: str):
    executor = PipelineExecutor(config_path)
    executor.execute_all()


if __name__ == "__main__":
    cli()
