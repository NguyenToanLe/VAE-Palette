#!/bin/bash

# Path to Miniconda installation directory
MINICONDA_DIR="/home/toan_le/miniconda3"

# Name of the Conda environment
CONDA_ENV="MA"

# Activate the Conda environment
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

# Path to the Python script
SCRIPT_PATH="/home/toan_le/Masterarbeit/Palette-Image-to-Image-Diffusion-Models/eval.py"

# Parameters for eval.py
GT="/home/toan_le/Masterarbeit/Palette-Image-to-Image-Diffusion-Models/experiments/finetune_combine_VAE_240808_181503/results/test/0/GT"
GENERATED="/home/toan_le/Masterarbeit/Palette-Image-to-Image-Diffusion-Models/experiments/finetune_combine_VAE_240808_181503/results/test/0/Out"
FILE="/home/toan_le/Masterarbeit/Palette-Image-to-Image-Diffusion-Models/experiments/finetune_combine_VAE_240808_181503/results/test/0"

# Run the Python script with parameters
python $SCRIPT_PATH --src $GT --dst $GENERATED --file $FILE

# Deactivate the Conda environment
conda deactivate
