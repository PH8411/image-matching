# Image Registration

## Introduction

This is a Python Code for image registration task . It contains the OpenCV implemetation of traditional registration method: SIFT and ORB; and the Pytorch implementation of deep learning method: SuperPoint and SuperGlue.  

SuperPoint and SuperGlue are respectively  CVPR2018  and CVPR2020 research project done by Magic Leap . SuperPoint is a CNN framework used for feature extraction and feature description. SuperGlue use deep graph matching method to replace the traditional local feature matching method, it use attention mechanism  aggregating  the context information . For more details, please see their paper and github repo:

- SuperPoint Paper: [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629).
- SuperPoint Github repo: https://github.com/magicleap/SuperPointPretrainedNetwork.
- SuperGlue Paper:[SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763).
- SuperGlue Github repo: https://github.com/magicleap/SuperGluePretrainedNetwork.

Due to the author only open the test code and pretrained network,  you need to realize the training code for your own implementation.  There also exist some unofficial project for reference, in this project, the SuperPoint training code  is  based on the Pytorch implementation: https://github.com/eric-yyjau/pytorch-superpoint. The SuperGlue training code is base on the Pytorch implementation:  https://github.com/HeatherJiaZG/SuperGlue-pytorch.

## Requirement

### Dependencies

This repo depends on a few standard pythonic modules, plus OpenCV and PyTorch.

- Python3==3.7
- Pytorch>=1.1
- opencv-python==4.5.1.48
- opencv-contrib-python==4.5.1.48
- cuda (tested in cuda10.1)

```
conda create --name matching python=3.7
conda activate matching
pip install -r requirements.txt
```

### Datasets

Datasets should be prepared by yourself. The folder structure contain two parts: the datasets for training and evaluation. 

The training dataset should look like this:

```
|-- ALLSS(Your datasets)
|   |-- train
|   |   |-- file.jpg
|   |   `-- ...
|   `-- val
|       |-- file.jpg
|   |   `-- ...
```

Our background is defect inspection in Industrial. We have many pictures which are the same product, but there exist shift、rotation、scaling between of them. And our task is  to align all the images to the same coordinate system which benefit to downstream task. So the evaluation dataset(for our using) use one image for template image, others for source image. It should look like this:

```
datasets/
|-- Amazon
|   `-- template
|       |-- template.jpg
|   |-- source
|   |   |-- source1.jpg
|   |   |-- source2.jpg
|   |   `-- ...
```

## Run the Code

### Traditional method

There are two main top-level scripts for testing in this repo:

- `traditional.py` : runs the traditional method: SIFT or ORB for feature extraction and description, KNN method(flann) for feature matching and RANSAC method for outliers rejection. 

  Run the SIFT method  for testing

  ```
  python traditional.py --Method SIFT --img_dir datasets/Amazon/ --Result_dir Results/Amazon/ --resize_scale 0.5 --match_viz True
  ```

  Run the ORB method  for testing

  ```
  python traditional.py --Method ORB --img_dir datasets/Amazon/ --Result_dir Results/Amazon/ --resize_scale 0.5 --match_viz True
  ```


### Superpoint method

#### Training 

Superpoint use a Self-Supervisied Training method, which contain three steps:                                                                               

- **Step 1:** train an initial interest point detector(Magic point) on synthetic data , we don’t realize it in our task and just using the pretrained model weights provided by the implementaion:  https://github.com/eric-yyjau/pytorch-superpoint

- **Step 2:** apply a novel homographic Adaptation procedure to automatically label images from a target, unlabeled domain 

  Export the pesudo label on training dataset:

  ```
  python superpoint_export_pseudo.py --config superpoint/configs/magicpoint_allss_export.yaml --export_task train --outputImg
  ```

  Export the pesudo label on validation dataset:

  ```
  python superpoint_export_pseudo.py --config superpoint/configs/magicpoint_allss_export.yaml --export_task val --outputImg
  ```

- **Step 3:** train a fully-convolutional network that jointly extracts interest points and descriptors from an image   

  ```
  python superpoint_train_descriptor.py --config superpoint/configs/superpoint_allss_train_heatmap.yaml
  ```

#### Testing 

After training on your own dataset, we test on the evaluation datasets. We apply the SuperPoint for feature extraction and feature description. KNN method(flann) for feature matching and RANSAC method for outliers rejection. 

Run the SuperPoint +FLANN method  for testing:

```
python superpoint_flann_test.py --img_dir datasets/Amazon/ --descriptor_dim 128 --img_dir --weights_path superpoint/models/weights/superPointNet_allss_descriptor_128.pth.tar
```

**noted：** the **descriptor_dim** and the **weights**  should be paired , if descriptor_dim=64, the weights_path should be superpoint/models/weights/superPointNet_allss_descriptor_64.pth.tar

###  SuperGlue method

#### Training

SuperGlue is GNN method for feature matching and outliers rejection. The input of superglue is the output of superpoint, and you  can using other deep learning or traditional(SIFT or ORB) feature extraction and description methods to replace superpoint.  Run the superglue training code with superpoint pretrained model :

```
python superpoint_glue_train.py --descriptor_dim 128 --keypoint_encoder [32,64,128] --superpoint_weights superpoint/models/weights/superPointNet_allss_descriptor_128.pth.tar  --sinkhorn_iterations 30
```

**noted：**  the **descriptor_dim** 、**weights**  and **keypoint _encoder** should be paired, like this:

superPointNet_allss_descriptor_64.pth.tar --descriptor_dim 64 --keypoint_encoder [32,64]

superPointNet_allss_descriptor_128.pth.tar --descriptor_dim 128 --keypoint_encoder [32,64,128]

superPointNet_allss_descriptor_256.pth.tar --descriptor_dim 256 --keypoint_encoder [32,64,128, 256]

#### Testing

Run the SuperPoint+SuperGlue code whose model is trained  by ourself for evaluation:

```
python superpoint_glue_test.py --descriptor_dim 128 --keypoint_encoder [32, 64, 128] --sinkhorn_iterations 30 --superpoint_weights superpoint/models/weights/superPointNet_allss_descriptor_128.pth.tar --superglue_weights superglue/models/weights/SuperGlue_allss_descriptor_128.pth
```

**noted：** We didn’t have a good result from the superpoint+superglue method on our training , the reason is that our training datasets is too small, you can training on a big dataset(like coco) and then fine tune on your own dataset.  You also can test the official model pretrained by Megicleap for comparing, which has a good result. 

Run the SuperPoint+SuperGlue official code for evaluation：

```
python superpoint_glue_official_test.py --descriptor_dim 256 --superpoint_weights supeeglue/models/weights/superpoint_v1.pth
--superglue_weights superglue/models/weights/superglue_indoor.pth --sinkhorn_iterations 30
```



