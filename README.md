# CNN Pokémon Classification

## Overview
This project focuses on classifying Pokémon images using Convolutional Neural Networks (CNN). The repository contains the code, datasets, and resources needed to train and evaluate a CNN model for this task.

## Contents
- `TrainImages/`: Directory containing training images.
- `TestImages/`: Directory containing test images.
- `main.ipynb`: Jupyter notebook with data preprocessing, data engineering, implementation of the CNN model, model tunning, and model evaluation.
- `PokemonStats.csv`: CSV file with Pokémon statistics.
- `prediction.csv`: CSV file with the model's predictions.

## Implementation Details

The implementation of the CNN Pokémon classification project involves several key steps:

1. **Data Preparation:**
   - Images are stored in the `TrainImages/` and `TestImages/` directories.
   - Pokémon statistics are provided in `PokemonStats.csv`.

2. **Model Architecture:**
   - The Convolutional Neural Network (CNN) is built using layers such as convolutional layers, pooling layers, and fully connected layers to extract features from the images and perform classification.

3. **Training the Model:**
   - The Jupyter notebook `main.ipynb` includes code to load the dataset, preprocess the images, and train the CNN model using a training set of Pokémon images.
   - Techniques like data augmentation and normalization are used to improve model performance.

4. **Evaluation and Prediction:**
   - The model is evaluated on a separate test set to assess its accuracy and generalization.
   - Predictions are generated and saved in `prediction.csv`.

5. **Hyperparameter Tuning:**
   - Parameters like learning rate, batch size, and the number of epochs are tuned to optimize the model's performance.

6. **Libraries and Frameworks:**
   - The project utilizes popular machine learning libraries such as TensorFlow or PyTorch for building and training the CNN model.