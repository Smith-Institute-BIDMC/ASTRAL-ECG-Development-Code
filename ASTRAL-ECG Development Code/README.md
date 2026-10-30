# AI-ECG: Deep Learning Analysis of 12-Lead Electrocardiograms

## Overview

This repository contains Python code for the development, evaluation, and external validation of convolutional neural network (CNN) models using 12-lead electrocardiogram (ECG) data.

The analysis includes ECG preprocessing, CNN model training, internal performance evaluation, external validation using the EchoNext dataset, multi-label classification, and Gradient-weighted Class Activation Mapping (Grad-CAM) visualization.


## Repository Structure

The analysis is organized into nine Jupyter notebooks.

| Notebook | Description |
|---|---|
| `01_preprocess_ecg.ipynb` | Preprocess ECG signals, including baseline wander removal, winsorization, and normalization |
| `02_build_cnn_model.ipynb` | Build and train the CNN model using the development dataset |
| `03_evaluate_cnn_model.ipynb` | Evaluate model performance on the internal test dataset |
| `04_predict_echonext.ipynb` | Apply the trained CNN model to the external EchoNext dataset to generate predictions |
| `05_evaluate_echonext.ipynb` | Evaluate model performance on the external EchoNext dataset |
| `06_build_multilabel_cnn.ipynb` | Build and train the multi-label CNN model |
| `07_evaluate_multilabel_cnn.ipynb` | Evaluate multi-label CNN model performance |
| `08_generate_gradcam.ipynb` | Generate Grad-CAM heatmaps to interpret model predictions |
| `09_plot_gradcam.ipynb` | Visualize Grad-CAM results across ECG leads |

## Requirements

The analysis was developed using Python and TensorFlow/Keras.

