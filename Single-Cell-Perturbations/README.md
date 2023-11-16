# Single-Cell Gene Expression Prediction

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/en/latest/)
[![Dataset](https://img.shields.io/badge/Dataset-Single_Cell_Perturbations-green.svg)](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data)

## Overview

This repository contains code for predicting how small molecules change gene expression in different cell types. The goal is to accelerate drug discovery and basic biology research by developing methods to accurately predict chemical perturbations in new cell types. 

## Data

The dataset used for this project can be found [here](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data). It includes various data files, such as `adata_obs_meta.csv`, `adata_train.parquet`, `de_train.parquet`, and more. These files are used for training and evaluation.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- LightGBM
- Scikit-learn
- PyArrow

### Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/spmfte/single-cell-gene-expression-prediction.git
   ```

2. Navigate to the project directory:

   ```shell
   cd single-cell-gene-expression-prediction
   ```

3. Install the required packages:

   ```shell
   pip install -r requirements.txt
   ```

### Usage

1. Run the Jupyter notebook `pert30.ipynb` for a step-by-step walkthrough of the project.

2. Modify the code as needed for your specific use case and dataset.

## Data Exploration

In the Jupyter notebook, we perform data exploration, visualize the dataset, and analyze its characteristics.

## Preprocessing

We preprocess the data by handling missing values and scaling the features to prepare it for model development.

## Model Development

We train a LightGBM regression model to predict chemical perturbations' impact on gene expression in different cell types.

## Evaluation

The model's performance is evaluated using root mean squared error (RMSE) on a test dataset.
## Acknowledgments

- [Open Problems in Single-Cell Analysis](https://www.openproblems.com/cellarity-competition)
- [Cellarity](https://www.cellarity.com/)

