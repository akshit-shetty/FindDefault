
# Find Default - Credit Card fraud detection

## Problem Statement:
A credit card is one of the most used financial products to make online purchases and payments. Though the Credit cards can be a convenient way to manage your finances, they can also be risky. Credit card fraud is the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash.

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. 
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

We have to build a classification model to predict whether a transaction is fraudulent or not.

## Installation

- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, imblearn, sklearn, scipy, xgboost, warnings, boto3, sagemaker, os


## Setup

Clone the project repository:

```bash
  git clone https://github.com/akshit-shetty/FindDefault.git
```

Navigate to the project directory:

```bash
  cd FindDefault
```

## Execution

### Data Preprocessing

- Loading the dataset using pandas.
- Checking data types and missing values.
- Summary statistics of numerical features.
- Handling missing values if any.

### Exploratory Data Analysis (EDA)

- Visualizing the distribution of the 'Class' variable (legitimate vs. fraudulent transactions).
- Visualizing the distribution of 'Time' and 'Amount'.
- Analyzing temporal patterns and transaction amounts.
- Creating boxplots for features V1 to V28 to identify potential outliers.

### Feature Engineering

- Handling outliers using power transformation.
- Scaling features using Min-Max scaling.
- Separating features and target variable.
- Addressing class imbalance using SMOTE.

### Splitting Dataset

- Splitting data into training, validation, and testing sets.
- Maintaining data integrity and reproducibility.

### Model Selection

- Several machine learning models are trained and evaluated on the validation data to select the best-performing model for the task.
- Models trained include Logistic Regression, K-Nearest Neighbors (KNN), Gaussian Naive Bayes (GaussianNB), Decision Trees, Random Forest XGBoost, and Gradient Boosting Classifier.
- Each model's performance is evaluated using accuracy, precision, recall, and F1-score on the validation data.
- The XGBoost classifier is selected as the best-performing model based on its high accuracy, precision, recall, and F1-score on the validation data.

### Model Training & Evaluation

- Training XGBoost classifier on the training data.
- Evaluating model performance on validation data.
- Hyperparameter tuning using Randomized Search.
- Evaluating model with best hyperparameters.
- Testing model on unseen test data.
- Assessing model performance with evaluation metrics and confusion matrix.

### Model Deployment

- Saving data in CSV format.
- Uploading data to Amazon S3.
- Creating XGBoost Estimator for training.
- Setting hyperparameters and defining training inputs.
- Training model and retrieving model data.
- Deploying model as an endpoint.
- Making predictions using the deployed endpoint and evaluating performance.
- Deleting the endpoint to avoid additional costs.

## Results

The model achieved high accuracy, precision, recall, and F1-score on both validation and test datasets, indicating strong performance in detecting fraudulent transactions.

## Future Work
- Explore advanced feature engineering techniques.
- Experiment with ensemble methods for model improvement.
- Integrate advanced anomaly detection techniques for enhanced - fraud detection.
- Implement continuous monitoring and updating mechanisms for the model.
- Enhance model interpretability and explainability for better decision-making.

## Conclusion
This project demonstrates a robust approach to credit card fraud detection, emphasizing thoughtful design choices, rigorous evaluation, and ongoing refinement for building reliable fraud detection systems.
