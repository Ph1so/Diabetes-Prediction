### Diabetes Prediction Machine Learning Project

# Overview

This repository contains a machine learning project focused on predicting diabetes using relevant medical data. The aim of this project is to develop a model that can accurately predict the likelihood of an individual having diabetes based on various input features.

# Dataset

The dataset used for this project consists of 2460 samples with 8 features, including age, BMI, insulin, and more. The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/ehababoelnaga/diabetes-dataset).

# Approach

In this project, I employed several different machine learning models from Scikit-learn to train and evaluate the predictive model. The steps involved in the process include:

- Importing data: Cleaning the dataset, handling missing values, and scaling features.
- Understanding data: Understanding data distribution and correlation matrix
- Model training: Training the selected algorithm on the preprocessed dataset.
- Model evaluation: Assessing the performance of the trained model using appropriate metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

# Results

Currently, the trained model achieved an accuracy score of 80%. 

# Usage

To run this project locally, follow these steps:

1. Clone this repository to your local machine.
2. Download necessary files
3. Run Diabetes_Prediction.ipynb script to preprocess the data, train the model, and make predictions.
   
# Technologies Used

Programming languages: `Python`

Libraries: `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn`

# Future Improvements

Some potential enhancements for this project include:

- Exploring additional features or data sources to improve the model's predictive performance.
- Tuning hyperparameters to optimize the model's performance.
- Deploying the trained model as a web application or API for real-time predictions.

# Acknowledgments
Special thanks to Ehab Aboelnaga for providing the dataset used in this project.

