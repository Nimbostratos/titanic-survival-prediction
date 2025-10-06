# 🚢 Titanic Survival Prediction

A comprehensive machine learning project predicting passenger survival on the Titanic using AdaBoost with extensive feature engineering and model evaluation.

## 📊 Project Overview

This project analyzes the famous Titanic dataset to predict which passengers survived the disaster. It includes thorough exploratory data analysis, intelligent feature engineering, and comparison of multiple ML algorithms.

## 🎯 Key Features

- **Smart Data Cleaning**: Class-based age imputation, handling missing values intelligently
- **Feature Engineering**: 
  - Title extraction from passenger names
  - Fare per ticket calculation (accounting for group bookings)
  - Age and fare binning
  - Family size features
- **Comprehensive EDA**: Visual analysis of survival patterns
- **Model Comparison**: Tested 7+ algorithms (AdaBoost, Random Forest, Gradient Boosting, etc.)
- **Hyperparameter Optimization**: RandomizedSearchCV + GridSearchCV
- **Advanced Evaluation**: Confusion matrix, ROC curves, precision-recall analysis

## 📈 Results

- **Test Accuracy**: ~82%
- **AUC-ROC Score**: ~0.85
- **Best Model**: AdaBoost with optimized hyperparameters

### Key Insights
- **Title** (Mr/Mrs/Miss) was the strongest predictor
- **Passenger Class** and **Sex** were highly correlated with survival
- **Family Size** showed non-linear relationship with survival
- Model successfully captures "women and children first" evacuation protocol

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **ML Algorithm**: AdaBoost Classifier
- **Tools**: Jupyter Notebook, Kaggle

## Cell breakdown
| **Cell Range** | **Section Title**                             | **Purpose and Description**                                                                                                                                                                              |
| -------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1–2**        | **Library Importation & Data Loading**        | Imports analytical and visualization libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`). Loads the Titanic dataset into a Pandas DataFrame (`combined_pd`) as the foundational dataset for analysis. |
| **3–5**        | **Exploratory Data Overview**                 | Performs initial inspection of the dataset using `.head()`, `.info()`, and `.describe()`. Identifies data types, missing values, and provides an overview of numerical and categorical features.         |
| **6–8**        | **Feature Engineering Setup**                 | Extracts additional information from raw data (e.g., `Title` from `Name`), defines relevant variables, and prepares the dataset for preprocessing and encoding.                                          |
| **9–12**       | **Data Cleaning & Encoding**                  | Handles missing values (`Age`, `Fare`, `Embarked`) and applies one-hot encoding using `pd.get_dummies()` to convert categorical variables (`Sex`, `Embarked`, `Title`) into numerical form.              |
| **13–15**      | **Feature Selection & Normalization**         | Removes non-predictive columns (e.g., `Name`, `Ticket`, `Cabin`) and ensures numerical features are normalized or standardized where necessary for model compatibility.                                  |
| **16–18**      | **Train–Test Split**                          | Divides the dataset into training (`x_train`, `y_train`) and testing (`x_test`, `y_test`) subsets using `train_test_split()`. Confirms shapes and consistency of resulting data matrices.                |
| **19–22**      | **Random Forest Model Training**              | Initializes and trains a `RandomForestClassifier` using the training set. Defines hyperparameters such as `n_estimators`, `max_depth`, and `criterion` for initial model fitting.                        |
| **23–26**      | **Hyperparameter Optimization (Grid Search)** | Implements `GridSearchCV` to test multiple hyperparameter configurations. Identifies and saves the best-performing model (`best_model`) based on cross-validation accuracy.                              |
| **27–30**      | **Model Evaluation (Random Forest)**          | Evaluates the optimized model using metrics such as accuracy, confusion matrix, precision, recall, and F1-score. Visual results are generated to validate predictive performance.                        |
| **31–34**      | **Feature Importance Analysis**               | Extracts and visualizes the Random Forest’s `feature_importances_`. Maps encoded variable names back to readable categories (e.g., `Title=Mr`, `Sex=male`) and displays a horizontal bar chart.          |
| **35–38**      | **AdaBoost Model Implementation**             | Introduces an `AdaBoostClassifier` for comparative analysis. Trains and evaluates it on the same dataset using identical metrics for performance benchmarking against Random Forest.                     |
| **39–42**      | **Comparative Visualization & Results**       | Visualizes accuracy, confusion matrices, and feature importances of both models. Provides comparative insight into bias–variance balance and model robustness.                                           |
| **43–End**     | **Final Outputs & Summary**                   | Summarizes findings, highlighting key predictive features (e.g., `Sex`, `Fare`, `Pclass`) and final model accuracies. Optionally exports results or trained models for deployment or reporting.          |


![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
