

## Introduction

This repository contains the implementation of **Label-Efficient Point Cloud Semantic Segmentation: A
Holistic Active Learning Approach**([arXiv](https://arxiv.org/abs/2101.06931)) in tensorflow.

If you find our work useful in your research, please consider citing:
```
@article{2021Label,
  title={Label-Efficient Point Cloud Semantic Segmentation: An Active Learning Approach},
  author={ Shi, X.  and  Xu, X.  and  Chen, K.  and  Cai, L.  and  Foo, C. S.  and  Jia, K. },
  year={2021},
}
```
## Installation
This implementation has been tested on Ubuntu 16.04 and Linux.

## Experiments
We provide scripts for ShapeNet experiments.

## Download training, validation and testing data for ShapeNet dataset. 
Run prepareDataset.sh

## Active Training
Run training script at ./RandomSamp/train_DGCNN_SP_AL.py

## Active Strategies
Different active learning strategies in ./Strategies

