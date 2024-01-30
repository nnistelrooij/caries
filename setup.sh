#!/bin/bash

# GCC
sudo apt update
sudo apt -y install build-essential

# Conda
conda create -n caries -y python=3.11
source ~/anaconda3/etc/profile.d/conda.sh
conda activate caries

# PyTorch
pip3 install torch torchvision torchaudio

# MMEngine, MMCV
pip3 install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# MMDetection
pip3 install -v -e $(dirname "$0")/mmdetection

# Pip requirements
pip3 install -r $(dirname "$0")/requirements.txt
