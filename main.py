# main.py
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load data
with zipfile.ZipFile("playground-series-s5e7.zip") as z:
    with z.open("train.csv") as f:
        train = pd.read_csv(f)
    with z.open("test.csv") as f:
        test = pd.read_csv(f)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# 2. Separate features and target
X = train.drop(columns=["Personality", "id"])
y = train["Personality"]

# 3. Split into numeric and categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# 4. Preprocessing
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", LabelEncoder())  # NOTE: we’ll adjust this
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# 5. Build model pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 6. Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 7. Fit model
clf.fit(X_train, y_train)

# 8. Evaluate
y_pred = clf.predict(X_val)
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))

# 9. Predict on test.csv (if needed for submission)
test_ids = test["id"]
X_test = test.drop(columns=["id"])
test_preds = clf.predict(X_test)

submission = pd.DataFrame({
    "id": test_ids,
    "Personality": test_preds
})
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as submission.csv")
