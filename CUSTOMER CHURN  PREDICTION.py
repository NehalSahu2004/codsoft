import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# STEP 1: DOWNLOAD DATASET
# =====================================================

print("Downloading dataset...")

path = kagglehub.dataset_download(
    "shantanudhakadd/bank-customer-churn-prediction"
)

print(f"\nDataset downloaded to:\n{path}")

# Locate CSV file
csv_file = os.path.join(path, "Churn_Modelling.csv")

# Load dataset
df = pd.read_csv(csv_file)

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# =====================================================
# STEP 2: DATA CLEANING
# =====================================================

print("\nMissing Values:")
print(df.isnull().sum())

# Drop unnecessary columns
df.drop(
    columns=[
        "RowNumber",
        "CustomerId",
        "Surname"
    ],
    inplace=True
)

# =====================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# =====================================================

# Gender Encoding
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

# Geography One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=["Geography"],
    drop_first=True
)

# Convert boolean columns to integers
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

print("\nColumns after Encoding:")
print(df.columns)

# =====================================================
# STEP 4: FEATURES AND TARGET
# =====================================================

X = df.drop("Exited", axis=1)
y = df["Exited"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# =====================================================
# STEP 5: TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# STEP 6: FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# STEP 7: LOGISTIC REGRESSION
# =====================================================

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_scaled, y_train)

lr_predictions = lr_model.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f"Accuracy: {lr_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, lr_predictions))

# =====================================================
# STEP 8: RANDOM FOREST
# =====================================================

print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"Accuracy: {rf_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

# =====================================================
# STEP 9: GRADIENT BOOSTING
# =====================================================

print("\n" + "=" * 50)
print("GRADIENT BOOSTING")
print("=" * 50)

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_predictions)

print(f"Accuracy: {gb_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, gb_predictions))

# =====================================================
# STEP 10: MODEL COMPARISON
# =====================================================

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy": [
        lr_accuracy,
        rf_accuracy,
        gb_accuracy
    ]
})

print(results)

best_model = results.loc[
    results["Accuracy"].idxmax(),
    "Model"
]

print(f"\nBest Model: {best_model}")

# =====================================================
# STEP 11: CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, rf_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Random Forest Confusion Matrix")
plt.show()

# =====================================================
# STEP 12: FEATURE IMPORTANCE
# =====================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Features Affecting Churn")
plt.tight_layout()
plt.show()

# =====================================================
# STEP 13: SAVE BEST MODEL
# =====================================================

import joblib

joblib.dump(rf_model, "customer_churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")
print("customer_churn_model.pkl")
print("scaler.pkl")

print("\nProject Completed Successfully!")