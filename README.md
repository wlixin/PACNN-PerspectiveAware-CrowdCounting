## Perspective-Aware CNN For Crowd Counting
By SHI Miaojing, YANG Zhaohui, XU Chao and CHEN Qijun
This implementation is written by SHI Miaojing and YANG Zhaohui

### Introduction
This project is an implementation of our arXiv paper - [Perspective-Aware CNN for Crowd Counting(PACNN)](http). A major challenge of the crowd counting task lies in the drastic changes of scales and perspectives in images. Recent trends employ either large-sized filters or patch-based inference to address this problem. PACNN does not follow them, instead it directly predicts a perspective map in the network and encodes it as a perspective-aware weighting layer to adaptively combine the density outputs from multi-scale feature maps. The weights are learned at every pixel of the map such that the final combination is robust to perspective changes and pedestrian size variations. PACNN achieves state-of-the-art results and runs as fast as the fastest. The ground truth perspective maps on ShanghaiTech dataset are provided. 

### License
The code & data is released under the MIT License (Please refer to the LICENSE file for details). It can only be used for academic research purposes.

### Citation
```
@article{shi18pacnn,
Author = {Miaojing Shi, Zhaohui Yang, Chao Xu and Qijun Chen},
Title = {Perspective-Aware CNN for Crowd Counting},
booktitle= = {arXiv},
Year = {2018}
}
```
### Dependencies and Installation
We tested the implementation on Linux 14.04 with GPU Nvidia Titan, CUDA8 and CuDNN v5. The other version should also working.
You should first compile tools/lmdb2txt and sigmoid\_learn layer with caffe.

### Train and Test
Two stages train prototxt and deploy prototxt are in folder stage\_1 and stage\_2, you may use matlab code to load configurations, prepare data and test model. Our trained model on ShanghaiTech PartB is provided.  

### Perspective Maps for ShanghaiTech Dataset
We provide the ground truth perspective maps for each image in ShanghaiTech dataset. Please refer to our paper for the generation of GT perspective maps. If you find the perspective information useful in your research, please kindly cite our paper above. 

### Q&A
Please submit a bug report on the Github site of the project if you run into any problems with the code.

