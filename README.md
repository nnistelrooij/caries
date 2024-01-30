# Detection of caries around restorations in bitewing radiographs

This is the code repository accompanying the paper *"Eduardo Chaves et al, 2024. Detection of caries around restorations on bitewings using deep learning"* submitted to Journal of Dentistry.


## Installation

Please refer to `setup.sh` for the installation of a virtual environment and the necessary packages.


## Reproduction

### Pre-processing

With the bitewing radiographs and annotations as supplied on OSF, splits for 10-fold cross-validation can be made by running `caries/split_bitewings.py`.

### Training

Specify the cross-validation split you would like to train within `caries/config.py` and run the following in the terminal:

``` shell
PYTHONPATH=code python code/mmdetection/tools/train.py code/caries/config.py
```

While training is running, several metrics are logged to TensorBoard in the working directory specified by `work_dir` in the configuration file.

### Inference

Choose the latest or best checkpoint from the working directory and run the following to save the annotations and predictions to a pickle file in the working directory.

```shell
PYTHONPATH=code python code/mmdetection/tools/test.py code/caries/config.py <work_dir>/<checkpoint>.pth --tta [--show [--wait-time <seconds>]] 
```

### Evaluation

The annotations can be compared to the predictions visually, by specifying `--show` and `--wait-time` after the inference terminal command above.

Additionally, a confusion matrix and an FROC curve can be made by running `caries/evaluation/confusion_matrix.py` and `caries/evaluation/froc.py`, respectively.


## Citation

```bib
@article{
    title={Detection of caries around restorations on bitewings using deep learning},
    author={Chaves, Eduardo Trota and {van Nistelrooij}, Niels and Vinayahalingam, Shankeeth and Xi, Tong and Romero, Vitor Henrique Digmayer and Shwendicke, Falk and Lima, Giana da Silveira and Loomans, Bas and Huysmans, Marie-Charlotte and Mendes, Fausto Medeiros and Cenci, Maximiliano Sergio},
    journal={Journal of Dentistry},
    year={2024},
}
```
