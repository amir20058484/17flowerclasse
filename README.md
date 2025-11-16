# Flower Classification Model using ResNet50

This repository contains a Python script for training and fine-tuning a deep learning model to classify 17 different flower species using transfer learning with ResNet50. The model is built with TensorFlow and Keras, incorporating data augmentation, training on a dataset, fine-tuning, and inference on sample images.

## Project Overview

- **Task**: Multi-class image classification for 17 flower categories.
- **Model Architecture**: Pre-trained ResNet50 (from ImageNet) as the base, with additional layers for classification.
- **Dataset**: The [17 Category Flower Dataset](https://www.kaggle.com/datasets/rajatkumar30/flower-recognition) from Kaggle, consisting of training and test images.
- **Key Features**:
  - Data augmentation for improved generalization.
  - Initial training with frozen base layers.
  - Fine-tuning of the last 10 layers for better performance.
  - Visualization of training and validation accuracy.
  - Sample prediction on a new image.

This code was originally developed in a Kaggle notebook environment but can be adapted for local runs with minor modifications (e.g., dataset paths).

## Dataset

The dataset includes:
- **Training Set**: Images in `/train` directory, organized by class folders.
- **Test Set**: Images in `/test` directory.
- **Classes**: 17 flower types (e.g., Bluebell, Buttercup, etc.). The exact class names are derived from the directory structure.

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rajatkumar30/flower-recognition) and place it in a local directory (e.g., `./data/train` and `./data/test`). Update the `train_path` and `test_path` variables in the script accordingly.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Pillow (for image loading)

Install dependencies via pip:

```bash
pip install tensorflow numpy matplotlib pillow
```

## Usage

1. **Run the Script**:
   - Execute the Python file:
     ```bash
     python flower_classification.py
     ```
   - The script will:
     - Load and preprocess data.
     - Build and train the model for 20 epochs.
     - Plot initial training accuracy.
     - Fine-tune the model for 5 more epochs.
     - Plot fine-tuning accuracy.
     - Predict the label for a sample image (update `img_path` as needed).

2. **Customization**:
   - Adjust hyperparameters like `batch_size`, `epochs`, or learning rate.
   - For local runs, replace Kaggle paths (e.g., `/kaggle/input/...`) with local paths.
   - To predict on a new image, call `predict_label(model, your_image_path)`.

3. **Output**:
   - Training history plots will be displayed (accuracy vs. epochs).
   - Sample prediction: Prints the predicted flower class for the given image.

## Model Architecture

- Base: ResNet50 (pre-trained on ImageNet, frozen initially).
- Top Layers:
  - Global Average Pooling.
  - Dense (256 units, ReLU).
  - Dropout (0.5).
  - Dense (17 units, Softmax).
- Optimizer: Adam.
- Loss: Categorical Crossentropy.
- Metrics: Accuracy.

During fine-tuning, the last 10 layers of ResNet50 are unfrozen, and the learning rate is reduced to 1e-5.

## Results

- **Initial Training**: Expect training accuracy to improve over 20 epochs, with validation accuracy monitoring for overfitting.
- **Fine-Tuning**: Further improvement in accuracy after unfreezing layers.
- **Sample Prediction**: For the provided image (`6028_SpanishBluebells_CGC8649sq-scaled.jpg`), the model outputs the predicted class (e.g., "Spanish Bluebells").

Example plots (generated during runtime):
- Training vs. Validation Accuracy (initial and fine-tuned).

## Limitations

- This script assumes a GPU-enabled environment (like Kaggle) for faster training. On CPU, reduce `batch_size` or epochs.
- Dataset paths are Kaggle-specific; adapt for local use.
- No model saving/exporting in the script—add `model.save('flower_model.h5')` if needed.

## Contributing

Feel free to fork the repository and submit pull requests for improvements, such as adding model evaluation on the test set or hyperparameter tuning.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: Provided by [rajatkumar30 on Kaggle](https://www.kaggle.com/datasets/rajatkumar30/flower-recognition).
- Inspired by standard transfer learning practices for image classification.
