# Flower Classification with ResNet50

This project involves classifying images of flowers into one of 17 categories using a deep learning model based on the ResNet50 architecture. The model is trained using the **17 Flowers** dataset and fine-tuned to improve accuracy. It is capable of predicting the type of flower in a given image.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Overview
This project uses a pre-trained **ResNet50** model from Keras for feature extraction, which is fine-tuned on the flower classification task. The model is first trained with the ResNet50 layers frozen and then fine-tuned by unfreezing the last few layers of the model for improved accuracy.

## Dependencies
To run this project, the following Python libraries are required:

- `tensorflow` >= 2.5
- `numpy`
- `matplotlib`
- `pandas`

You can install these dependencies by running:

```bash
pip install tensorflow numpy matplotlib pandas
