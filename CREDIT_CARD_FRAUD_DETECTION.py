import os
import warnings
import kagglehub
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# =====================================================
# DOWNLOAD DATASET
# =====================================================

print("Downloading dataset...")

path = kagglehub.dataset_download("kartik2112/fraud-detection")

print("Dataset Path:", path)

# =====================================================
# FIND CSV FILE
# =====================================================

csv_files = []

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

print("\nCSV Files Found:")
for i, file in enumerate(csv_files):
    print(f"{i+1}. {file}")

dataset_path = csv_files[0]

print("\nLoading Dataset...")
df = pd.read_csv(dataset_path)

# =====================================================
# BASIC INFO
# =====================================================

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# =====================================================
# REMOVE MISSING VALUES
# =====================================================

df.dropna(inplace=True)

# =====================================================
# HANDLE DATETIME COLUMN
# =====================================================

if 'trans_date_trans_time' in df.columns:

    df['trans_date_trans_time'] = pd.to_datetime(
        df['trans_date_trans_time']
    )

    df['trans_year'] = df['trans_date_trans_time'].dt.year
    df['trans_month'] = df['trans_date_trans_time'].dt.month
    df['trans_day'] = df['trans_date_trans_time'].dt.day
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour

    df.drop('trans_date_trans_time', axis=1, inplace=True)

# =====================================================
# HANDLE DOB COLUMN
# =====================================================

if 'dob' in df.columns:

    df['dob'] = pd.to_datetime(df['dob'])

    df['birth_year'] = df['dob'].dt.year
    df['birth_month'] = df['dob'].dt.month

    df.drop('dob', axis=1, inplace=True)

# =====================================================
# DROP USELESS ID COLUMNS
# =====================================================

drop_columns = [
    'Unnamed: 0',
    'trans_num',
    'first',
    'last',
    'street'
]

for col in drop_columns:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# =====================================================
# ENCODE CATEGORICAL FEATURES
# =====================================================

categorical_columns = df.select_dtypes(include=['object']).columns

print("\nEncoding Columns:")
print(categorical_columns)

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))

# =====================================================
# TARGET VARIABLE
# =====================================================

TARGET = "is_fraud"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found!"
    )

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# =====================================================
# VERIFY DATA TYPES
# =====================================================

print("\nFeature Types:")
print(X.dtypes.value_counts())

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    print("\n" + "=" * 60)
    print(model.__class__.__name__)
    print("=" * 60)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    return accuracy, precision, recall, f1

# =====================================================
# LOGISTIC REGRESSION
# =====================================================

print("\nTraining Logistic Regression...")

log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_model.fit(X_train_scaled, y_train)

log_results = evaluate_model(
    log_model,
    X_test_scaled,
    y_test
)

# =====================================================
# DECISION TREE
# =====================================================

print("\nTraining Decision Tree...")

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=15
)

dt_model.fit(X_train, y_train)

dt_results = evaluate_model(
    dt_model,
    X_test,
    y_test
)

# =====================================================
# RANDOM FOREST
# =====================================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_results = evaluate_model(
    rf_model,
    X_test,
    y_test
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features")
print(importance_df.head(15))

# =====================================================
# MODEL COMPARISON
# =====================================================

results_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        log_results[0],
        dt_results[0],
        rf_results[0]
    ],
    "Precision": [
        log_results[1],
        dt_results[1],
        rf_results[1]
    ],
    "Recall": [
        log_results[2],
        dt_results[2],
        rf_results[2]
    ],
    "F1 Score": [
        log_results[3],
        dt_results[3],
        rf_results[3]
    ]
})

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.sort_values(
        by="F1 Score",
        ascending=False
    )
)

# =====================================================
# BEST MODEL
# =====================================================

best_model = results_df.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]

print("\nBest Model:")
print(best_model["Model"])
print("F1 Score:", round(best_model["F1 Score"], 4))