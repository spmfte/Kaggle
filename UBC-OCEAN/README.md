
# Ovarian Cancer Subtype Classification and Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Ovarian_Cancer_Subtypes-green.svg)](https://www.kaggle.com/competitions/UBC-OCEAN/data)

## Overview

This repository houses the project for classifying ovarian cancer subtypes using histopathology images. The project's goal is to advance medical diagnostics by developing a model that accurately classifies ovarian cancer subtypes.

## Data

The dataset is accessible [here](https://www.kaggle.com/competitions/UBC-OCEAN/data). It includes histopathology images from over 20 medical centers, making it the largest dataset of its kind for ovarian cancer research.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- PyTorch
- OpenCV
- Matplotlib

### Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/spmfte/ovarian-cancer-subtype-classification.git
   ```

2. Navigate to the project directory:

   ```shell
   cd ovarian-cancer-subtype-classification
   ```

3. Install the required packages:

   ```shell
   pip install -r requirements.txt
   ```

### Usage

1. Open `cancer_classification.ipynb` for a comprehensive guide on the project's methodology.

2. Adapt the model and techniques according to your research needs.

## Data Exploration

The notebook includes an in-depth analysis of the image dataset, along with preprocessing steps like normalization and augmentation.

## Preprocessing

Image preprocessing involves resizing, normalization, and data augmentation to optimize the model's performance.

## Model Development

A PyTorch-based convolutional neural network (CNN) model is developed for image classification tasks.

## Evaluation

Model performance is assessed using accuracy, precision, recall, and AUC metrics to ensure robustness in clinical applications.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/)
- [PyTorch](https://pytorch.org/)
