# FairFace Fairness Classification

## Overview
This notebook demonstrates race classification using the FairFace dataset with a deep learning model based on ResNet50. It includes exploratory data analysis (EDA), model training, and evaluation while focusing on fairness metrics across race groups. The notebook explores baseline modeling, pre-processing with oversampling to balance race groups, and fairness evaluation using the `fairlearn` library.

## Features
- Data loading and preparation from FairFace dataset via KaggleHub
- Visualization of race and gender distribution and sample images by race
- Image data augmentation and generator setup with TensorFlow Keras
- Race classification using transfer learning with ResNet50
- Accuracy, classification report, confusion matrix, and ROC analysis
- Fairness evaluation via accuracy, selection rate, false positive rate, disparate impact, and equal opportunity difference
- Pre-processing approach with oversampling for class balancing
- Reusable evaluation function for model performance and fairness

## Tech Stack
- Python  
- Jupyter Notebook / Google Colab  
- TensorFlow with Keras  
- Matplotlib, Seaborn for visualization  
- scikit-learn for metrics and preprocessing  
- fairlearn for fairness metrics  
- AIF360 (installed but not used explicitly in shown code)  
- kagglehub for dataset downloading  
- reportlab (installed but not used explicitly in shown code)

## How to Use
1. Clone or download the repository.
2. Open the notebook `fairface-fairness-classification.ipynb` in Jupyter or Google Colab.
3. Run all cells sequentially to install dependencies, load data, train models, and evaluate results.
4. Results include model accuracy, fairness metrics, and plots saved in the working directory.

## Status
This notebook is well-organized and prepared for GitHub presentation with clean, reproducible workflows for fairness-aware race classification on FairFace.