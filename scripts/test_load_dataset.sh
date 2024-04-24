#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

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
pwd
python3 -u $HOME/LUNA23-group-5/luna/dataset.py
