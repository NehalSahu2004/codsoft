import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# DOWNLOAD DATASET
# ==========================================

print("Downloading dataset...")

path = kagglehub.dataset_download(
    "uciml/sms-spam-collection-dataset"
)

print("Dataset Path:", path)

# Find dataset file
dataset_file = None

for file in os.listdir(path):
    if file.endswith(".csv") or file.endswith(".tsv"):
        dataset_file = os.path.join(path, file)
        break

print("Dataset File:", dataset_file)

# ==========================================
# LOAD DATASET
# ==========================================

try:
    df = pd.read_csv(dataset_file, encoding="latin-1")
except:
    df = pd.read_csv(
        dataset_file,
        sep="\t",
        names=["label", "message"]
    )

print("\nDataset Shape:", df.shape)

print(df.head())

# ==========================================
# CLEAN DATA
# ==========================================

if "v1" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

df.drop_duplicates(inplace=True)

print("\nClass Distribution:")
print(df["label"].value_counts())

# ==========================================
# CONVERT LABELS
# ham = 0
# spam = 1
# ==========================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df["message"]
y = df["label"]

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(X)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# MODEL 1 - NAIVE BAYES
# ==========================================

print("\n" + "=" * 50)
print("NAIVE BAYES")
print("=" * 50)

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)

print("Accuracy:", round(nb_acc, 4))

print(classification_report(
    y_test,
    nb_pred
))

# ==========================================
# MODEL 2 - LOGISTIC REGRESSION
# ==========================================

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)

print("Accuracy:", round(lr_acc, 4))

print(classification_report(
    y_test,
    lr_pred
))

# ==========================================
# MODEL 3 - SUPPORT VECTOR MACHINE
# ==========================================

print("\n" + "=" * 50)
print("SUPPORT VECTOR MACHINE")
print("=" * 50)

svm_model = LinearSVC()

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)

print("Accuracy:", round(svm_acc, 4))

print(classification_report(
    y_test,
    svm_pred
))

# ==========================================
# MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Naive Bayes",
        "Logistic Regression",
        "SVM"
    ],
    "Accuracy": [
        nb_acc,
        lr_acc,
        svm_acc
    ]
})

print("\n")
print(results)

best_model_name = results.loc[
    results["Accuracy"].idxmax(),
    "Model"
]

print(f"\nBest Model: {best_model_name}")

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, svm_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("SVM Confusion Matrix")
plt.show()

# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

sns.barplot(
    x=results["Model"],
    y=results["Accuracy"]
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# ==========================================
# SAVE BEST MODEL
# ==========================================

joblib.dump(
    svm_model,
    "spam_sms_model.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)

print("\nModel Saved Successfully!")

# ==========================================
# CUSTOM PREDICTION
# ==========================================

sample_message = [
    "Congratulations! You won a free iPhone. Click now!"
]

sample_vector = vectorizer.transform(
    sample_message
)

prediction = svm_model.predict(
    sample_vector
)

print("\nCustom Prediction:")

if prediction[0] == 1:
    print("SPAM")
else:
    print("HAM")