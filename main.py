# main.py
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

# Encode target labels (Extrovert -> 1, Introvert -> 0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

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
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# 5. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=500, random_state=42)
}

# 6. Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# 7. Evaluate all models
best_model = None
best_acc = 0

for name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_val, y_pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

    if acc > best_acc:
        best_acc = acc
        best_model = clf
        best_name = name

print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")

# 8. Predict on test.csv with best model
test_ids = test["id"]
X_test = test.drop(columns=["id"])
test_preds_encoded = best_model.predict(X_test)

# Decode predictions back to original labels
test_preds = le.inverse_transform(test_preds_encoded)

submission = pd.DataFrame({
    "id": test_ids,
    "Personality": test_preds
})
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as submission.csv")