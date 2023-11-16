# Detecting AI-Generated Writing in Student Essays

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-AI_Generated_Text-green.svg)](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data)

## Overview

This repository contains code for detecting AI-generated content in student essays. The project aims to develop a deep learning model that can differentiate between human-written and AI-generated text, addressing the growing challenge of AI in education.

## Data

The dataset is available [here](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data) and includes a collection of essays from middle and high school students. It contains features indicating whether the text is human or AI-generated.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- TensorFlow
- Pandas
- NumPy

### Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/spmfte/ai-generated-writing-detection.git
   ```

2. Navigate to the project directory:

   ```shell
   cd ai-generated-writing-detection
   ```

3. Install the required packages:

   ```shell
   pip install -r requirements.txt
   ```

### Usage

1. Run the Jupyter notebook `ai_detection.ipynb` to see the implementation and analysis.

2. Adjust the model and parameters to suit different datasets or requirements.

## Data Exploration

Initial data exploration is conducted in the Jupyter notebook, where you can examine the characteristics of both human-written and AI-generated texts.

## Preprocessing

Data preprocessing involves cleaning, tokenization, and vectorization of text data to prepare it for model training.

## Model Development

TensorFlow to develop a neural network model that classifies text as either human-written or AI-generated.

## Evaluation

The model is evaluated based on its accuracy and F1 score, ensuring its effectiveness in real-world scenarios.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/)
- [TensorFlow](https://www.tensorflow.org/)
