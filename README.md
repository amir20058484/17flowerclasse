# Flower Classification with ResNet50

This project classifies images of flowers into 17 categories using a deep learning model based on **ResNet50**. The model leverages transfer learning, first training on features extracted using a pre-trained ResNet50 model and then fine-tuning for the specific task of flower classification.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Usage](#usage)
- [License](#license)

## Overview
This project uses the **17 Flowers** dataset to classify flower images into 17 distinct categories. The model starts by using a pre-trained **ResNet50** model, which helps the model to learn complex image features more quickly. We then fine-tune the model to improve performance on the specific flower classification task. The entire process is done in a few stages: data preprocessing, model training, and prediction.

## Requirements
Make sure to install the following dependencies to run the project:

```bash
pip install tensorflow numpy matplotlib
Additionally, you need to have access to the 17 Flowers Dataset. You can download it from Kaggle.

Dataset
The 17 Flowers Dataset consists of images representing 17 different species of flowers. The dataset is divided into training and test sets. The training set is used to train the model, while the test set is used to evaluate its performance.

Dataset Link: 17 Flowers Dataset on Kaggle

The images are stored in the following directory structure:

train/: Contains the images for training the model.

test/: Contains the images for testing the model.

Model Architecture
The model is built on top of ResNet50, a deep convolutional neural network pre-trained on ImageNet, which is used as a feature extractor. The custom layers added on top of the ResNet50 base model help fine-tune the model for the specific task of flower classification.

Base Model: ResNet50 (pre-trained on ImageNet).

Added Layers:

Global Average Pooling: This reduces the spatial dimensions of the feature maps.

Dense Layer (256 units) with ReLU activation: Adds a fully connected layer for classification.

Dropout (0.5): Reduces overfitting during training.

Output Layer (17 units) with Softmax activation: Outputs the probability of each class (flower type).

The model is trained in two phases:

Initial Training: We freeze the layers of the ResNet50 model and train the custom layers.

Fine-Tuning: After the initial training, we unfreeze the last few layers of ResNet50 and continue training the entire model to improve performance.

Training
The model is trained using the 17 Flowers Dataset with data augmentation techniques such as rotation, shifting, zooming, and flipping. These techniques help to improve the model’s ability to generalize to new, unseen images.

Data Augmentation: The ImageDataGenerator is used to apply random transformations to the images to make the model more robust.

Batch Training: The model is trained in batches of 32 images, and the training lasts for 20 epochs with a validation split of 20%.

Fine-Tuning: After the initial training phase, we unfreeze the last few layers of the ResNet50 model to allow for fine-tuning and more accurate learning.

Here is how the accuracy improves during training:

python
Copy code
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
Usage
To use the trained model for predicting the class of a new flower image, follow the steps below:

1. Load and Preprocess the Image:
Use the load_img and img_to_array functions from Keras to load and preprocess the image.

2. Predict the Label:
Use the model’s predict function to get the predicted class for the image.

Example Code:
python
Copy code
def predict_label(model, img_path):
    img = load_img(img_path, target_size=(224, 224))  # Load image
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image for ResNet50
    
    pred = model.predict(img_array)  # Predict class probabilities
    class_index = np.argmax(pred)  # Get the index of the highest probability
    
    return labels[class_index]  # Return the corresponding flower class

# Example Usage:
img_path = "path_to_flower_image.jpg"
result = predict_label(model, img_path)
print("Predicted label:", result)
This function will output the predicted flower class for the input image
