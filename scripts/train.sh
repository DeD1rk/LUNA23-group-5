#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=04:20:00
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
    --epochs=800 \
    --fold=0 \
    --segmentation-weight=1.0 \
    --noduletype-weight=1.0 \
    --malignancy-weight=1.0 \
    --aug-mirror-x \
    --aug-mirror-y \
    --aug-mirror-z \
    --dropout=0.3 \
    --weight-decay=0.005 \
    --learning-rate=0.0001 \
    --aug-scaling \
    --exp-id="02sampler-b16-e800-w111-aug-xyz-d03-lr1e-4-wd0005-scaling" \
    --perform-inference

cp -r $TMPDIR/results/* $HOME/results/

