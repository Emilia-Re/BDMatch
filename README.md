**Binary Decomposition: A Problem Transformation Perspective for Open-Set Semi-Supervised Learning**

Jun-Yi Hang, Min-Ling Zhang

*Accepted by ICML 2024*



*This code package can be used freely for academic, non-profit purposes. For other usage,  please contact us for further information (Prof. Min-Ling Zhang: zhangml@seu.edu.cn ).*

## Usage

### 1. Required Packages

We suggest first creating a conda environment:

```
conda create --name bdmatch python=3.8
```

then use pip to install required packages:

```
pip install -r requirements.txt
```

### 2. Download and Prepare the Data

All data sets can be downloaded from the Google [drive](https://drive.google.com/file/d/1Rkm6BHFH2Vr2kduDoStn9UJKil7ZRsaj/view?usp=sharing).

Download and then unrar the data into the current directory. Please ensure the file structure the same as follows:

```
BDMatch
├── config
    └── ...
├── data
    ├── cifar10
        └── cifar-10-python.tar
    └── cifar100
        └── cifar-100-python.tar
    └── imagenet30
        └── filelist
        └── one_class_test
        └── one_class_train
    └── ...
├── semilearn
    └── ...
└── ...  
```

### 3. Train and Test BDMatch

For example, to train BDMatch on CIFAR-100-80-4 data set:

```python
# CIFAR100, seen/unseen split of 80/20, 4 labels per seen class (CIFAR-80-4), seed = 0  
python train.py --c config/classic_cv_os/bdmatch/bdmatch_cifar100_80_4_0.yaml
```

For another example, to train BDMatch on ImageNet-30-20-p1 data set:

```python
# ImageNet-30, seen/unseen split of 20/10, 1% labeled data (ImageNet-30-20-p1), seed = 1  
python train.py --c config/classic_cv_os/bdmatch/bdmatch_imagenet30_20_p1_1.yaml
```

Training BDMatch on other data sets can be specified by corresponding config files. We provide config files for all data sets used in the paper. Please check the folder `config/classic_cv_os/bdmatch/`.

## Additional Usage

BDMatch is implemented based on the codebase of [USB](https://github.com/microsoft/Semi-supervised-learning) and its usage is kept as consistent as possible with USB. Please refer to [USB](https://github.com/microsoft/Semi-supervised-learning) for additional usage.

## Acknowledgments

We sincerely thank the authors of [USB (NeurIPS'22)](https://github.com/microsoft/Semi-supervised-learning) for creating such an awesome SSL codebase.
