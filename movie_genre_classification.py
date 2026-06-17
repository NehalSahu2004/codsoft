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
    confusion_matrix
)

# ==================================================
# DOWNLOAD DATASET
# ==================================================

print("Downloading dataset...")

path = kagglehub.dataset_download(
    "hijest/genre-classification-dataset-imdb"
)

print("Dataset Path:", path)

# ==================================================
# FIND TRAIN FILE
# ==================================================

train_file = None

for root, dirs, files in os.walk(path):
    for file in files:
        if "train" in file.lower():
            train_file = os.path.join(root, file)
            break

print("Train File:", train_file)

# ==================================================
# LOAD DATASET
# ==================================================

try:
    df = pd.read_csv(train_file)
except:
    df = pd.read_csv(
        train_file,
        sep=":::",
        engine="python",
        header=None,
        names=[
            "id",
            "genre",
            "title",
            "description"
        ]
    )

print("\nDataset Shape:", df.shape)

print(df.head())

# ==================================================
# DATA CLEANING
# ==================================================

df = df.dropna()

df["description"] = df["description"].astype(str)

print("\nGenres Found:")
print(df["genre"].value_counts().head())

# ==================================================
# FEATURES AND TARGET
# ==================================================

X = df["description"]

y = df["genre"]

# ==================================================
# TF-IDF FEATURE EXTRACTION
# ==================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_tfidf = vectorizer.fit_transform(X)

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==================================================
# MODEL 1 : NAIVE BAYES
# ==================================================

print("\n" + "=" * 60)
print("NAIVE BAYES")
print("=" * 60)

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

nb_acc = accuracy_score(
    y_test,
    nb_pred
)

print("Accuracy:", round(nb_acc, 4))

# ==================================================
# MODEL 2 : LOGISTIC REGRESSION
# ==================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(
    X_train,
    y_train
)

lr_pred = lr_model.predict(
    X_test
)

lr_acc = accuracy_score(
    y_test,
    lr_pred
)

print("Accuracy:", round(lr_acc, 4))

# ==================================================
# MODEL 3 : SUPPORT VECTOR MACHINE
# ==================================================

print("\n" + "=" * 60)
print("SUPPORT VECTOR MACHINE")
print("=" * 60)

svm_model = LinearSVC()

svm_model.fit(
    X_train,
    y_train
)

svm_pred = svm_model.predict(
    X_test
)

svm_acc = accuracy_score(
    y_test,
    svm_pred
)

print("Accuracy:", round(svm_acc, 4))

# ==================================================
# CLASSIFICATION REPORT
# ==================================================

print("\nClassification Report (SVM)\n")

print(
    classification_report(
        y_test,
        svm_pred
    )
)

# ==================================================
# MODEL COMPARISON
# ==================================================

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

print("\nModel Comparison\n")
print(results)

# ==================================================
# ACCURACY GRAPH
# ==================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=results,
    x="Model",
    y="Accuracy"
)

plt.title(
    "Movie Genre Classification Models"
)

plt.show()

# ==================================================
# SAVE BEST MODEL
# ==================================================

joblib.dump(
    svm_model,
    "movie_genre_model.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)

print("\nModel Saved Successfully!")

# ==================================================
# CUSTOM TEST
# ==================================================

sample_plot = [
    """
    A detective investigates a mysterious
    murder case involving a serial killer.
    """
]

sample_vector = vectorizer.transform(
    sample_plot
)

prediction = svm_model.predict(
    sample_vector
)

print("\nPredicted Genre:")
print(prediction[0])