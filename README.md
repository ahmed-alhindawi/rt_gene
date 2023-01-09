# RT-GENE & RT-BENE: Real-Time Eye Gaze and Blink Estimation in Natural Environments
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![stars](https://img.shields.io/github/stars/Tobias-Fischer/rt_gene.svg?style=flat-square)](https://github.com/Tobias-Fischer/rt_gene/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Tobias-Fischer/rt_gene.svg?style=flat-square)](https://github.com/Tobias-Fischer/rt_gene/issues)
[![GitHub repo size](https://img.shields.io/github/repo-size/Tobias-Fischer/rt_gene.svg?style=flat-square)](./README.md)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rt-gene-real-time-eye-gaze-estimation-in/gaze-estimation-on-mpii-gaze&style=flat-square)](https://paperswithcode.com/sota/gaze-estimation-on-mpii-gaze?p=rt-gene-real-time-eye-gaze-estimation-in?style=square)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rt-gene-real-time-eye-gaze-estimation-in/gaze-estimation-on-rt-gene&style=flat-square)](https://paperswithcode.com/sota/gaze-estimation-on-rt-gene?p=rt-gene-real-time-eye-gaze-estimation-in)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rt-gene-real-time-eye-gaze-estimation-in/gaze-estimation-on-ut-multi-view&style=flat-square)](https://paperswithcode.com/sota/gaze-estimation-on-ut-multi-view?p=rt-gene-real-time-eye-gaze-estimation-in)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rt-bene-a-dataset-and-baselines-for-real-time/blink-estimation-on-eyeblink8&style=flat-square)](https://paperswithcode.com/sota/blink-estimation-on-eyeblink8?p=rt-bene-a-dataset-and-baselines-for-real-time)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rt-bene-a-dataset-and-baselines-for-real-time/blink-estimation-on-researcher-s-night&style=flat-square)](https://paperswithcode.com/sota/blink-estimation-on-researcher-s-night?p=rt-bene-a-dataset-and-baselines-for-real-time)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rt-bene-a-dataset-and-baselines-for-real-time/blink-estimation-on-rt-bene&style=flat-square)](https://paperswithcode.com/sota/blink-estimation-on-rt-bene?p=rt-bene-a-dataset-and-baselines-for-real-time)

This repository contains a fork of the code for two papers: [RT-GENE (Gaze Estimation; ECCV2018)](http://openaccess.thecvf.com/content_ECCV_2018/html/Tobias_Fischer_RT-GENE_Real-Time_Eye_ECCV_2018_paper.html) and [RT-BENE (Blink Estimation; ICCV2019 Workshops)](http://openaccess.thecvf.com/content_ICCVW_2019/html/GAZE/Cortacero_RT-BENE_A_Dataset_and_Baselines_for_Real-Time_Blink_Estimation_in_ICCVW_2019_paper.html).

## RT-GENE (Gaze Estimation)

### License + Attribution
The RT-GENE code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Commercial usage is not permitted. If you use this dataset or the code in a scientific publication, please cite the following [paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Tobias_Fischer_RT-GENE_Real-Time_Eye_ECCV_2018_paper.html):

### Overview + Accompanying Dataset
The code is split into four parts, each having its own README contained. There is also an accompanying [dataset](https://zenodo.org/record/2529036) [(alternative link)](https://goo.gl/tfUaDm) to the code. For more information, other datasets and more open-source software please visit the Personal Robotic Lab's website: <https://www.imperial.ac.uk/personal-robotics/software/>.

#### RT-GENE ROS package
The [rt_gene](./rt_gene) directory contains a ROS package for real-time eye gaze and blink estimation. This contains all the code required at inference time.

<p align="center">
  <img src="./assets/dataset_video.gif" alt="RT-GENE inference example"/>
</p>

#### RT-GENE Inpainting
The [rt_gene_inpainting](./rt_gene_inpainting) directory contains code to inpaint the region covered by the eyetracking glasses.

![Inpaining example](./assets/inpaint_example.jpg)

#### RT-GENE Model Training
The [rt_gene_model_training](./rt_gene_model_training) directory allows using the inpainted images to train a deep neural network for eye gaze estimation.

<p align="center">
  <img src="./assets/accuracy_prl.jpg" alt="Accuracy on RT-GENE dataset"/>
</p>

## Faster-Better Blink Estimation

Code TBA
