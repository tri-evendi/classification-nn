# Mango Ripeness Detection using GLCM and Backpropagation Neural Network

## Setup and Running the Program

1. Environment Setup: Ensure Python is installed and set up the environment. Install all required dependencies or activate an existing environment (e.g., kmenv).
2. Dataset Extraction: Execute python3 glcm.py to process the mango images (ripe and unripe). The program applies the GLCM algorithm to each image and saves the feature data to a CSV file.
3. Model Training: With the dataset prepared, run python3 backprop.py to train the neural network. The trained model is saved to network-train.npy.
4. Model Testing via GUI: Launch python3 master_gui.py and select an image to evaluate. The program preprocesses each image, ensuring a consistent aspect ratio for accurate predictions. The GUI displays the prediction results.

## GUI Instructions

1. Image Selection: Use the interface to choose an image file for ripeness evaluation.
2. Result Display: The GUI presents the prediction outcome, indicating the ripeness status of the selected mango.

## Utilizing the Model

1. Load the network-train.npy file to use the trained model.
2. Apply the predict function to evaluate new images, obtaining their ripeness classification.

## Employing GLCM for Feature Extraction

1. Use glcm.py with an input image to compute GLCM features.
2. The script generates a CSV file with the GLCM data, suitable for model training or further analysis.

## Training with Backpropagation

1. Input the CSV file containing feature data to backprop.py for model training.
2. The script fine-tunes the neural network and outputs a trained model file.

## Dataset Details

1. The dataset consists of images processed through glcm.py.
2. It includes 189 unripe and 125 ripe mango images, used for both training and testing the model.

## Interpreting Results

The model's performance is quantified using a confusion matrix, recall, precision, and accuracy metrics, showcased in the console or GUI after testing.#

```python
    Confusion Matrix: 
    [[189   1]
    [  1 125]]
    Recall Confusion Matrix:  [0.99473684 0.99206349]
    Precission Confusion Matrix:  [0.99473684 0.99206349]
    Accuracy:  99.4
```

## Summary

This project presents an automated system for detecting mango ripeness using image processing and neural network techniques. It demonstrates the process from feature extraction using GLCM to training a backpropagation neural network and evaluating mango images through a user-friendly GUI.
