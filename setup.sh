# GCC
sudo apt update
sudo apt -y install build-essential

# PyTorch
pip3 install torch torchvision torchaudio

# MMEngine, MMCV
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# MMDetection
pip install -v -e mmdetection

# Pip requirements
pip install -r requirements.txt
