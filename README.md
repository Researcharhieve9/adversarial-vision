# Progressive Alignment for Robust Domain Adaptation

## Overview

This repository contains code for **Progressive Alignment for Robust Domain Adaptation (PARDA)**. It aims to improve robustness under domain shift in Unsupervised Domain Adaptation (UDA) under clean and adversarial settings.


## Supported Benchmarks

* OfficeHome (12 pairs)
* PACS (12 pairs)
* DIGIT (3 pairs)
* VisDA (2 pairs)

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```


## Running the Code

To train on a specific domain pair:

```bash
python main.py \
  --dataset OFFICEHOME \
  --source Art \
  --target Real_World \
  --batch_size 16 \
  --lr 1e-4 \
  --val_split 0.1 \
  --seed 42 \
  --multigpu

```

To test a saved model on clean and adversarial target data:

```bash
python test.py \
  --dataset OFFICEHOME \
  --source Art \
  --target Real_World \
  --attack pgd \
  --eps 0.008 \
  --atk_lr 0.002 \
  --atk_iter 20 \
  --ckpt_path ./checkpoints/officehome_art_to_real_world.pth \
  

```



## Optional t-SNE Visualization

To plot feature alignment via t-SNE:

```bash
python test.py --tsne
```

## Directory Structure

```
├── main.py                  # Training entry point
├── test.py                  # Evaluation script
├── net.py                   # FeatureExtractor and Classifier definitions
├── config.py                # Dataset and class configuration
├── utils.py                 # Utility functions (PGD, loaders, t-SNE, etc.)
├── resnet.py                # Contain ResNet Backbone
├── train.py                 # Contain Warm-start and MCD steps
├── requirements.txt         # Environment setup
```

## References & Acknowledgements

This work builds on codebases and implementations from:

* [DART](https://github.com/google-research/domain-robust)
* [MCD (Saito et al.)](https://github.com/mil-tokyo/MCD_DA)
