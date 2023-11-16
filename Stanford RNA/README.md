# RNA Reactivity Prediction

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Keras-orange.svg)
![Framework](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)

## Overview

This repository contains code for predicting RNA reactivity using deep learning techniques. The project involves data preprocessing, model development using LSTM networks, and evaluation of the model's performance.

## Table of Contents

- [Importing Necessary Libraries](#importing-necessary-libraries)
- [Data Loading and Preliminary Exploration](#data-loading-and-preliminary-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Deep Learning Model with LSTM](#deep-learning-model-with-lstm)
- [Model Evaluation](#model-evaluation)
- [Visualizing Loss](#visualizing-loss)

## Importing Necessary Libraries <a name="importing-necessary-libraries"></a>

This section includes the importation of essential libraries for data manipulation, visualization, and deep learning using Keras.

## Data Loading and Preliminary Exploration <a name="data-loading-and-preliminary-exploration"></a>

The dataset is loaded from '/kaggle/input/stanford-ribonanza-rna-folding/train_data.csv'. Preliminary exploration can include examining the dataset's structure, summary statistics, and sample data points.

## Data Preprocessing <a name="data-preprocessing"></a>

Data preprocessing involves:
- Finding the maximum sequence length for padding
- Padding the sequences with a specific character
- Encoding sequences to numerical format
- Converting the encoded sequences to a matrix format
- Splitting the dataset into training and validation sets

## Deep Learning Model with LSTM <a name="deep-learning-model-with-lstm"></a>

In this section, a deep learning model is developed using LSTM layers. The architecture includes bidirectional LSTM layers with dropout regularization and fully connected layers. The model is compiled with the mean absolute error as the loss function.

## Model Evaluation <a name="model-evaluation"></a>

The model's performance is evaluated using the mean absolute error (MAE) on the validation set.

## Visualizing Loss <a name="visualizing-loss"></a>

The training and validation loss curves are visualized over epochs.
## Acknowledgments

- [Keras](https://keras.io/)
- [Scikit-Learn](https://scikit-learn.org/)
- [Kaggle](https://www.kaggle.com/)
