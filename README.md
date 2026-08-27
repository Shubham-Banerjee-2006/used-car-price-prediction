# 🚗 Used Car Price Prediction

A Machine Learning-based web application that predicts the resale price of a used car based on its specifications and details.

The project uses **Random Forest Regression** to estimate the expected price of a used car and provides a simple web interface using **Flask**.

---

## 📌 Project Overview

Buying or selling a used car can be difficult because the price depends on many factors such as:

- Car brand and model
- Manufacturing year
- Kilometers driven
- Fuel type
- Transmission type
- Engine specifications
- Mileage
- Number of previous owners
- Other vehicle specifications

This project uses Machine Learning to analyze these factors and predict an estimated resale price.

The goal is to provide a simple and practical solution that can help users understand the approximate market value of a used car.

---

## 🎯 Objectives

- Predict the approximate resale price of used cars.
- Apply Machine Learning to a real-world problem.
- Perform data preprocessing and feature engineering.
- Train and evaluate a Random Forest Regression model.
- Build a user-friendly web application using Flask.
- Allow users to enter car details and receive a predicted price.

---

## 🧠 Machine Learning Model

The project uses:

**Random Forest Regressor**

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees to produce a more reliable prediction.

### Why Random Forest?

- Handles non-linear relationships well.
- Works with multiple input features.
- Provides good prediction performance.
- Less sensitive to overfitting compared with a single decision tree.
- Suitable for tabular datasets.

---

## 🔄 Project Workflow

```text
Used Car Dataset
       ↓
Data Cleaning
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Train-Test Split
       ↓
Random Forest Regression
       ↓
Model Evaluation
       ↓
Save Trained Model
       ↓
Flask Web Application
       ↓
User Enters Car Details
       ↓
Predicted Used Car Price
