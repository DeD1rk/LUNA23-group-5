import shutil
from datetime import datetime
from pathlib import Path

import click

from .inference import perform_inference_on_test_set
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
@click.option(
    "--perform-inference/--no-perform-inference",
    default=False,
    type=bool,
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
    perform_inference: bool = False,
):
    date = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = results_dir / f"{date}_{exp_id or 'default'}_fold{fold}"
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

    if perform_inference:
        perform_inference_on_test_set(data_dir, save_dir)
        shutil.make_archive(
            save_dir / "predictions", "zip", save_dir / "test_set_predictions"
        )


@click.command()
@click.option(
    "--data-dir",
    envvar="DATA_DIR",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--result-dir",
    envvar="RESULT_DIR",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def inference(
    data_dir: Path,
    result_dir: Path,
):
    perform_inference_on_test_set(data_dir, result_dir)
    shutil.make_archive(
        result_dir / "predictions.zip", "zip", result_dir / "test_set_predictions"
    )


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(inference)

if __name__ == "__main__":
    cli()
