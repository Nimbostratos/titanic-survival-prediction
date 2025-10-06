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

## 📁 Project Structure
