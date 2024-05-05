from datetime import datetime
from pathlib import Path

import click

from .training import Trainer


@click.command()
@click.option("--fold", default=0)
@click.option(
    "--exp-id",
    type=str,
    help="A name for this experiment. Paths will always get timestamps as well.",
)
@click.option("--epochs", default=100)
@click.option(
    "--data-dir",
    envvar="DATA_DIR",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--results-dir",
    envvar="RESULTS_DIR",
    required=True,
    type=click.Path(exists=False, file_okay=False, path_type=Path),
)
def train(
    data_dir: Path, results_dir: Path, epochs=100, exp_id: str | None = None, fold=0
):
    date = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = results_dir / f"{date}_{exp_id or 'default'}" / f"fold{fold}"
    save_dir.mkdir(exist_ok=True, parents=True)

    trainer = Trainer(data_dir, save_dir, fold=fold, epochs=epochs)
    trainer.train()


@click.group()
def cli():
    pass


cli.add_command(train)

if __name__ == "__main__":
    cli()
