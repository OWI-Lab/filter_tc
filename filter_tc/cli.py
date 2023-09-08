"""Console script for filter_tc."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for filter_tc."""
    click.echo("Replace this message by putting your code into "
               "filter_tc.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
