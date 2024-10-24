#!/bin/bash

# Path to Miniconda installation directory
MINICONDA_DIR="/home/toan_le/miniconda3"

# Name of the Conda environment
CONDA_ENV="MA"

# Activate the Conda environment
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

# Path to the Python script
SCRIPT_PATH="/home/toan_le/Masterarbeit/Palette-Image-to-Image-Diffusion-Models/run.py"

# Parameters for train.py
CONFIGPATH="/home/toan_le/Masterarbeit/Palette-Image-to-Image-Diffusion-Models/config/uncropping_custom_train.json"
BATCHSIZE=3

# Run the Python script with parameters
python $SCRIPT_PATH --config $CONFIGPATH --phase train --batch $BATCHSIZE --gpu_ids 5,6,7

# Deactivate the Conda environment
conda deactivate
