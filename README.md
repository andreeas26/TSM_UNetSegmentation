# UNet Segmentation

## Introduction
This is the code presented in the [article for Today Software Magazine](https://www.todaysoftmag.ro/article/3229/segmentarea-automata-in-imagistica-medicala) from September 2020.

## Environment
Initially this code was developed using the below versions for the main libraries, but it should work fine with the most recent ones too:
- Keras: deep learning framework, version 2.2.4;
- Tensorflow: for GPU as backend, version 1.12.0;
- numpy: library for multi-dimensional arrays computations, version 1.15.0;
- OpenCV: library for python, used for the pre-processing steps, version 4.1.0.25;
- imgaug: for data augmentation, version 0.2.9.

For training and evaluation I used [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb) environment as it offers free access to a VM instance with GPU. The dataset and all the necessary files must be in Google drive. That's why the first step when running a notebook is mounting the drive.

## Dataset
This project uses the [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#97944ee0e4a54b15a8479efafa2064dd) dataset. It contains annotated mammograms with two type of tumors: calcification and mass. For each image it is known the type of tumor, its pathology (benign or malignant) and it has one or more segmentation masks coresponding for each tumor present in that image. 

## Project structure
The project contains the following files:
- data_analysis
  - data_analysis.ipynb: gives a short overview of the dataset
  - create_jpgs.ipynb: converts DICOM files into JPG
  - create_cropped_imgs_masks.ipynb: cropps the original images and masks
  - imgs_masks_dims.ipynb: analysis of the data from the image dimension point of view
  - tumor_stats.ipynb: analysis of the breast tumors.
- config_train.yaml: training parameters
- config_evaluate.yaml: evaluation parameters
- UNetModel.py: the UNet architecture
- generators.py: custom generator for loading the data batch-wise
- custom_metrics.py: contains the Dice Coefficient loss definition
- train.ipynb: notebook for the training process 
- evaluate.ipynb: notebook for the evaluation process
- history_stats.ipynb: notebook for checking the status of the training process
- overlap_masks.ipynb: notebook for ploting the overlapping between the ground truth and the predicted mask

## References
1.  Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi , Daniel Rubin  (2016). Curated Breast Imaging Subset of DDSM [Dataset]. The Cancer Imaging Archive. DOI:  https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
