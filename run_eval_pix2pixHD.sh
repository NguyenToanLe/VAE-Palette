#!/bin/bash

# Path to Miniconda installation directory
MINICONDA_DIR="/home/toan_le/miniconda3"

# Name of the Conda environment
CONDA_ENV="MA"

# Activate the Conda environment
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

# Path to the Python script
SCRIPT_PATH="/home/toan_le/Masterarbeit/Palette-Image-to-Image-Diffusion-Models/eval_pix2pixHD.py"

# Parameters for eval.py
INPUT="/home/toan_le/Masterarbeit/collimation-eval_scripts/results/original_collimated_input_reverse_masks/test_latest/images"
FILE="/home/toan_le/Masterarbeit/collimation-eval_scripts/results/original_collimated_input_reverse_masks/test_latest"

# Run the Python script with parameters
python $SCRIPT_PATH --src $INPUT --file $FILE

# Deactivate the Conda environment
conda deactivate
