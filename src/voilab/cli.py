import click

from voila.app import Voila

@click.group()
def cli():
    pass

@cli.command()
def launch_viewer():
    argv = [
        "--no-browser",
        "nbs/replay_buffer_viewer.ipynb"
    ]
    Voila.launch_instance(argv=argv)

if __name__ == "__main__":
    cli()
