#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/danne/slurm_output/%j.out

# Load modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Copy data to scratch storage
echo "Copying data to $TMPDIR"
cp -r $HOME/LUNA23-group-5/dataset/ $TMPDIR/dataset/
cp -r $HOME/results/ $TMPDIR/results/

# Set up environment
python3 -m pip install --user --upgrade pip
python3 -m pip install --user scikit-build
python3 -m pip install --user -r $HOME/LUNA23-group-5/requirements.txt

# Run training script
WORKSPACE_PATH=$TMPDIR python3 -u $HOME/LUNA23-group-5/luna-base/inference.py
cp -r $TMPDIR/results $HOME
