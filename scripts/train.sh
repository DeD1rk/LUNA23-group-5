#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=02:30:00
#SBATCH --mem=32G
#SBATCH --output=/home/danne/slurm_output/%j.out


# Load modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Copy data to scratch storage
echo "Copying data to $TMPDIR"
cp -r $HOME/LUNA23-group-5/dataset/ $TMPDIR/dataset/

# Set up environment
python3 -m pip install --user --upgrade pip
python3 -m pip install --user scikit-build
python3 -m pip install --user -r $HOME/LUNA23-group-5/requirements.txt

# Run training script

python -u -m luna train \
    --data-dir=$TMPDIR/dataset \
    --results-dir=$TMPDIR/results \
    --batch-size=16 \
    --epochs=300 \
    --fold=0 \
    --segmentation-weight=2.0 \
    --noduletype-weight=1.0 \
    --malignancy-weight=1.0 \
    --aug-mirror-x \
    --aug-mirror-y \
    --aug-mirror-z \
    --dropout=0.2 \
    --exp-id="baseline-b16-e300-w211-aug-xyz-d02-lr1e-4" \
    --perform-inference

cp -r $TMPDIR/results/* $HOME/results/

