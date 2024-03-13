import click

from .examples import exmpl_fresnelzone 

@click.group()
def cli():
    pass

@cli.command()
@click.argument('exmpl', type=click.Choice(["fresn"]))
def example(exmpl):
    if exmpl == 'fresn':
        exmpl_fresnelzone.run()