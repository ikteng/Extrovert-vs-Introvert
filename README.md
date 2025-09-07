# Extrovert-vs-Introvert

Predict the Introverts from the Extroverts
Kaggle Competition/Dataset: https://www.kaggle.com/competitions/playground-series-s5e7

This project predicts whether a person is an Introvert or Extrovert, based on their social behavior and personality traits.
It uses machine learning models to train on the provided dataset (train.csv) and generate predictions for the test set (test.csv) for Kaggle submission.

## How the Code works (main.py)
Load the Data
   - The script extracts and reads train.csv and test.csv directly from the downloaded competition zip file.
   - Displays dataset shapes for sanity check.

Prepare Features and Target
   - Drops the id column (not useful for prediction).
   - Defines Personality as the target variable, encoding: Introvert → 0, Extrovert → 1

Handle Different Data Types
- Splits features into numeric columns and categorical columns for separate preprocessing
- Numeric columns:
  - Missing values imputed using median.
  - Values scaled with StandardScaler.
- Categorical columns:
  - Missing values imputed with the most frequent category.
  - Encoded using OneHotEncoder.

Machine Learning Models
- Three classifiers are tested: Logistic Regression, Random Forest (n=300 trees), XGBoost (n=500 trees)

Train/Validation Split: Splits data into 80% training and 20% validation, stratified to preserve class balance

Model Training & Evaluation
- Each model is trained inside a Pipeline (preprocessing + classifier).
- Evaluates performance with: Accuracy, Classification Report (Precision, Recall, F1-score), and Confusion Matrix
- Tracks the best model based on validation accuracy.

Generate Predictions for Submission
- Applies the best model to the test.csv dataset.
- Decodes predictions back to "Introvert" / "Extrovert".
- Saves final output as submission.csv.
