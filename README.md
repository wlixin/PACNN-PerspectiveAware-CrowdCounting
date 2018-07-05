## Perspective-Aware CNN For Crowd Counting
By SHI Miaojing, Yang Zhaohui, Xu Chao
This implementation is written by SHI Miaojing and YANG Zhaohui

### Introduction
This project is an implementation of the crowd counting method proposed in "Perspective-Aware CNN For Crowd Counting".

### License
This code is released under the MIT License (Please refer to the LICENSE file for details). It can only be used for academic research purposes.

### Citation

### Dependencies and Installation
We tested the implementation on Linux 14.04 with GPU Nvidia Titan, CUDA8 and CuDNN v5. The other version should also working.
You should first compile tools/lmdb2txt and sigmoid\_learn layer with caffe.

### Train and Test
Two stages train prototxt and deploy prototxt are in folder stage\_1 and stage\_2, you may use matlab code to load configurations, prepare data and test model.

### ShanghaiTech Dataset
ShanghaiTech can be download using dropbox: https://www.dropbox.com/s/ci3nghwhe9wy8h5/ShanghaiTech.zip?dl=0

### Q&A
Please submit a bug report on the Github site of the project if you run into any problems with the code.
