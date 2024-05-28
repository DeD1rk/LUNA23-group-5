import shutil
from datetime import datetime
from pathlib import Path

import click

from .inference import perform_inference_on_test_set
from .process import perform_inference_on_one_sample
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
    "--dropout",
    type=click.FloatRange(min=0, max=1),
    default=0.0,
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
@click.option(
    "--aug-mirror-x/--no-aug-mirror-x",
    default=False,
    type=bool,
    help="Enable data augmentation by random mirroring along the x-axis (left-right).",
)
@click.option(
    "--aug-mirror-y/--no-aug-mirror-y",
    default=False,
    type=bool,
    help="Enable data augmentation by random mirroring along the y-axis (front-back).",
)
@click.option(
    "--aug-mirror-z/--no-aug-mirror-z",
    default=False,
    type=bool,
    help="Enable data augmentation by random mirroring along the z-axis (head-toe).",
)
@click.option(
    "--aug-noise-std",
    default=0.0,
    type=click.FloatRange(min=0),
    help="Enable data augmentation with normal noise with this STD.",
)
@click.option(
    "--aug-scaling/--no-aug-scaling",
    default=False,
    type=bool,
    help="Enable data augmentation by randomly scaling (0.8-1.2x) along each axis (individually).",
)
@click.option(
    "--learning-rate",
    default=1e-4,
    type=click.FloatRange(min=0),
)
@click.option(
    "--weight-decay",
    default=0.0,
    type=click.FloatRange(min=0),
)
def train(
    data_dir: Path,
    results_dir: Path,
    epochs: int = 100,
    batch_size: int = 8,
    exp_id: str | None = None,
    fold: int = 0,
    dropout: float = 0.0,
    segmentation_weight: float = 1.0,
    noduletype_weight: float = 1.0,
    malignancy_weight: float = 1.0,
    perform_inference: bool = False,
    aug_mirror_x: bool = False,
    aug_mirror_y: bool = False,
    aug_mirror_z: bool = False,
    aug_noise_std: float = 0,
    aug_scaling: bool = False,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
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
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        task_weights={
            "segmentation": segmentation_weight,
            "noduletype": noduletype_weight,
            "malignancy": malignancy_weight,
        },
        augmentation_mirrorings=(aug_mirror_z, aug_mirror_y, aug_mirror_x),
        augmentation_noise_std=aug_noise_std,
        augmentation_scaling=aug_scaling,
    )
    trainer.train()

    if perform_inference:
        perform_inference_on_test_set(data_dir, save_dir, dropout=dropout)
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
@click.option(
    "--dropout",
    type=click.FloatRange(min=0, max=1),
    default=0.0,
)
def inference(data_dir: Path, result_dir: Path, dropout: float = 0.0):
    perform_inference_on_test_set(data_dir, result_dir, dropout=dropout)
    shutil.make_archive(
        result_dir / "predictions.zip", "zip", result_dir / "test_set_predictions"
    )


def proces(dropout: float = 0.3, checkpoint_name: str = "best-model"):
    perform_inference_on_one_sample(dropout=dropout, checkpoint_name=checkpoint_name)


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(inference)
cli.add_command(proces)

if __name__ == "__main__":
    cli()
