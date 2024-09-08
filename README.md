# Loan Approval Prediction using Machine Learning

## Project Overview

This project focuses on predicting loan approval outcomes using machine learning techniques. The prediction model aims to assist lenders in assessing whether an applicant is eligible for a loan based on certain factors. The project follows a structured approach starting from data analysis to deploying the model using FastAPI.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Steps Involved](#steps-involved)
   - [1. Data Analysis](#1-data-analysis)
   - [2. Feature Engineering](#2-feature-engineering)
   - [3. Training ML Models](#3-training-ml-models)
   - [4. FastAPI App Creation](#4-fastapi-app-creation)
4. [How to Run the Project](#how-to-run-the-project)
5. [Results](#results)


## Technologies Used

- Python
- Pandas
- Scikit-learn
- FastAPI
- NumPy
- Matplotlib / Seaborn for visualization


## Steps Involved

### 1. Data Analysis

- The dataset contains features related to applicants, such as their Gender(Male/Female) ,Education(Graduate/Not),income, loan amount, credit history, etc.
- Exploratory data analysis (EDA) was conducted to understand patterns and trends in the data.
- Various visualization techniques were used to highlight key relationships between the features and the target variable.

### 2. Feature Engineering

- Feature selection and transformation were performed to enhance the model's predictive power.
- Missing values were handled appropriately.
- Categorical variables were encoded using techniques such Label Encoder.

### 3. Training ML Models

- Several machine learning algorithms were evaluated, including Logistic Regression, SVM, KNearestNeighbors.
- The model was trained on a split of training and test datasets.
- Hyperparameter tuning was performed to optimize the models using techniques like GridSearchCV.
- The best-performing model was selected based on evaluation metrics like accuracy.

### 4. FastAPI App Creation

- A FastAPI web application was created to serve the loan approval model as an API.
- Users can send a request with loan-related details and get an immediate prediction on whether the loan will be approved or not.
- The API can handle both GET and POST requests.


