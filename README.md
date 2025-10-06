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
| Cell Range | Purpose                                                                                                        |
| ---------- | -------------------------------------------------------------------------------------------------------------- |
| 1          | **Library Imports** — importing Python packages.                                                               |
| 2          | **Data Loading** — reading your Titanic dataset.                                                               |
| 3–5        | **Utility / Setup code** — probably display settings, print checks, or early tests.                            |
| 6          | **Library Imports** — additional packages (likely ML or visualization).                                        |
| 7–8        | **Utility / Setup code** — more configuration or helper definitions.                                           |
| 9–11       | **Data Cleaning / Feature Engineering** — handling missing data, encoding, mapping categorical variables, etc. |
| 12–15      | **Utility / Intermediate steps** — possibly checking data shape or distributions.                              |
| 16         | **Data Cleaning / Feature Engineering** — more transformations.                                                |
| 17–20      | **Utility / Checks / Setup** — validation, summaries, or dataset splits.                                       |

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
