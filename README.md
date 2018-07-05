## Perspective-Aware CNN For Crowd Counting
By SHI Miaojing, YANG Zhaohui, XU Chao and CHEN Qijun.
This implementation is written by SHI Miaojing and YANG Zhaohui

### Introduction
This project is an implementation of the proposed method in our arXiv paper - [Perspective-Aware CNN For Crowd Counting (PACNN)](http://arxiv.org/abs/). A major challenge of the crowd counting lies in the drastic changes of scales and perspectives in images. Recent trends employs different (large) sized filters or conduct patch-based estimations to tackle it. This work does not follow them, it instead directly predicts a perspective map in the network and encodes it as a perspective-aware weighting layer to adaptively combine the density outputs from multi-scale feature maps. The weights are learned at every pixel of the map such that the final combination is robust to perspective changes and pedestrian size variations. PACNN achieves state-of-the-art results and runs as fast as the fastest. The ground truth perspective maps used in this work is also provided. 

### License
This code is released under the MIT License (Please refer to the LICENSE file for details). It can only be used for academic research purposes.

### Citation
```
@article{shi18pacnn,
Author = {Miaojing Shi, Zhaohui Yang, Chao Xu and Qijun Chen},
Title = {Perspective-Aware CNN For Crowd Counting},
booktitle= = {arXiv},
Year = {2018}
}
```

### Dependencies and Installation
We tested the implementation on Linux 14.04 with GPU Nvidia Titan, CUDA8 and CuDNN v5. The other versions should work as well. You should first be able to compile tools/lmdb2txt and sigmoid\_learn layer with Caffe.

### Train and Test
Train and deploy prototxt are available in folder TrainTest. You can use the matlab code to load configurations, prepare data and test model. Our trained model on ShanghaiTech PartB is also included for test.

### Perspective Maps for ShanghaiTech Dataset
We also provide the ground truth perspective map for every image in the ShanghaiTech Dataset. Please refer to our paper about the generation of GT perspective maps. If you find the perspective data useful in your research, please kindly cite our paper above. 

### Q&A
Please submit a bug report on the Github site of the project if you run into any problems with the code.
