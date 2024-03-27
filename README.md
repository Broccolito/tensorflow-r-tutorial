# Sentiment Analysis on Movie Reviews

This project demonstrates the process of performing sentiment analysis on a dataset of movie reviews using R with TensorFlow, Keras, and other relevant libraries. The objective is to classify movie reviews into positive or negative categories based on their content.

## Overview

The project utilizes the IMDb movie review dataset, a widely-used resource for text classification tasks. The dataset comprises 50,000 reviews split evenly into training and test sets, with balanced subsets of positive and negative reviews.

## Prerequisites

To run the scripts in this directory, ensure you have the following R libraries installed:

- `tensorflow`: Provides an R interface to TensorFlow, a powerful open-source software library for machine learning.
- `keras`: Offers a simple and flexible R interface to the deep learning library Keras, which is built on top of TensorFlow.
- `tfdatasets`: Aids in setting up input pipelines for TensorFlow models directly from R.
- `coro`: Facilitates asynchronous programming in R, useful for handling I/O-bound tasks efficiently.
- `readr`: Part of the tidyverse, readr is used for reading and writing data in R.

## Dataset

The IMDb movie review dataset is automatically downloaded and prepared for processing. The dataset includes a training set, a validation set, and a test set. Unsupervised samples are removed to maintain a focus on supervised learning.

## Data Preparation

Text data is preprocessed to convert the raw movie reviews into a more manageable format for the neural network. This involves standardizing the text (e.g., converting to lowercase, removing HTML tags, and stripping punctuation).

## Model

The project employs a sequential neural network model with the following layers:

- An embedding layer for text representation,
- Dropout layers for regularization,
- A pooling layer to reduce the dimensionality,
- A dense layer for output.

The model is compiled with a binary cross-entropy loss function and an Adam optimizer, aiming to classify reviews into positive (1) or negative (0) sentiments.

## Training

The model is trained on the preprocessed training dataset while monitoring its performance on a validation set to prevent overfitting. Performance metrics include accuracy and loss over epochs.

## Evaluation and Prediction

After training, the model's performance is evaluated on a separate test dataset. Additionally, the model can predict sentiments for new, raw movie review texts.

## How to Run

1. Ensure all required libraries are installed.
2. Execute the script to train and evaluate the model.
3. Use the model to predict sentiments of new movie reviews.

This project provides a comprehensive example of applying deep learning for natural language processing (NLP) tasks using R with TensorFlow and Keras.