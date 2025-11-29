import click

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


if __name__ == "__main__":
    cli()
