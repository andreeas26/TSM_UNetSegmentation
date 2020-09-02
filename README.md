# TSM_UNetSegmentation

## Introduction

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

## References
1.  Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi , Daniel Rubin  (2016). Curated Breast Imaging Subset of DDSM [Dataset]. The Cancer Imaging Archive. DOI:  https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
