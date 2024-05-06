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
@click.option(
    "--batch-size",
    default=8,
)
@click.option(
    "--segmentation-weight",
    envvar="SEGMENTATION_WEIGHT",
    type=click.FloatRange(min=0),
    default=1.0,
)
@click.option(
    "--noduletype-weight",
    envvar="NODULETYPE_WEIGHT",
    type=click.FloatRange(min=0),
    default=1.0,
)
@click.option(
    "--malignancy-weight",
    envvar="MALIGNANCY_WEIGHT",
    type=click.FloatRange(min=0),
    default=1.0,
)
def train(
    data_dir: Path,
    results_dir: Path,
    epochs=100,
    batch_size=8,
    exp_id: str | None = None,
    fold=0,
    segmentation_weight: float = 1.0,
    noduletype_weight: float = 1.0,
    malignancy_weight: float = 1.0,
):
    date = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = results_dir / f"{date}_{exp_id or 'default'}" / f"fold{fold}"
    save_dir.mkdir(exist_ok=True, parents=True)

    trainer = Trainer(
        data_dir,
        save_dir,
        fold=fold,
        epochs=epochs,
        batch_size=batch_size,
        task_weights={
            "segmentation": segmentation_weight,
            "noduletype": noduletype_weight,
            "malignancy": malignancy_weight,
        },
    )
    trainer.train()


@click.group()
def cli():
    pass


cli.add_command(train)

if __name__ == "__main__":
    cli()
