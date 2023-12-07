# Emotion Recognition Using Convolutional Neural Networks

## Description

This project aims to implement emotion recognition using convolutional neural networks (CNNs), inspired by the architectures and methodologies described in "[A New Method for Face Recognition Using Convolutional Neural Network](http://advances.vsb.cz/index.php/AEEE/article/view/2389/1292)" and "[Facial Expression Recognition in the Wild via Deep Attentive Networks](https://iopscience.iop.org/article/10.1088/1742-6596/1844/1/012004/pdf)". The project utilizes TensorFlow and Keras for model development, with a focus on achieving high accuracy in detecting human emotions from facial images.

## Features

- Utilizes a custom-built CNN architecture for emotion recognition.
- Employs Haar cascades for effective face detection in preprocessing.
- Implements data augmentation to enhance model robustness.
- Split data into training and validation sets for effective model evaluation.
- Can be extended to real-time emotion recognition with minimal modifications.

## Model Architecture

The model architecture is based on the principles outlined in the referenced papers, adapted for emotion recognition. Key features include:

- Multiple convolutional layers for feature extraction.
- MaxPooling layers for spatial data reduction.
- Dropout layers to mitigate overfitting.
- Dense layers for classification, with a softmax activation function in the output layer to handle multiple emotion classes.

## Installation

To set up the project, follow these steps:
