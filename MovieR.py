# =========================================================
# MOVIE GENRE PREDICTION USING IMDB DATASET
# TF-IDF + LOGISTIC REGRESSION
# FULL WORKING CODE
# =========================================================

# =========================================================
# 1. INSTALL REQUIRED LIBRARIES
# =========================================================
# RUN ONLY ONCE IF LIBRARIES ARE NOT INSTALLED

# pip install kagglehub
# pip install pandas
# pip install numpy
# pip install nltk
# pip install scikit-learn
# pip install joblib

# =========================================================
# 2. IMPORT LIBRARIES
# =========================================================

import os
import pandas as pd
import numpy as np
import kagglehub
import nltk
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# =========================================================
# 3. DOWNLOAD NLTK DATA
# =========================================================

nltk.download('stopwords')
nltk.download('punkt')

# =========================================================
# 4. DOWNLOAD DATASET
# =========================================================

print("Downloading Dataset...\n")

path = kagglehub.dataset_download(
    "hijest/genre-classification-dataset-imdb"
)

print("Dataset Downloaded Successfully!")
print("Dataset Path:", path)

# =========================================================
# 5. CHECK DATASET FILES
# =========================================================

print("\nMain Folder Files:")
print(os.listdir(path))

# Dataset folder
dataset_folder = os.path.join(
    path,
    "Genre Classification Dataset"
)

print("\nDataset Folder Files:")
print(os.listdir(dataset_folder))

# =========================================================
# 6. LOAD TRAINING DATA
# =========================================================

train_file = os.path.join(
    dataset_folder,
    "train_data.txt"
)

print("\nLoading Dataset...\n")

df = pd.read_csv(
    train_file,
    sep=" ::: ",
    engine="python",
    header=None,
    names=[
        "ID",
        "TITLE",
        "GENRE",
        "DESCRIPTION"
    ]
)

print("Dataset Loaded Successfully!")

# =========================================================
# 7. DISPLAY DATASET INFO
# =========================================================

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nGenre Distribution:\n")
print(df["GENRE"].value_counts())

# =========================================================
# 8. REMOVE MISSING VALUES
# =========================================================

df.dropna(inplace=True)

# =========================================================
# 9. FEATURES AND LABELS
# =========================================================

X = df["DESCRIPTION"]
y = df["GENRE"]

# =========================================================
# 10. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# =========================================================
# 11. CREATE MACHINE LEARNING PIPELINE
# =========================================================
# TF-IDF converts text into vectors
# Logistic Regression classifies genres

model = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",

        LogisticRegression(
            max_iter=1000
        )
    )
])

# =========================================================
# 12. TRAIN MODEL
# =========================================================

print("\nTraining Model...\n")

model.fit(X_train, y_train)

print("Model Training Completed!")

# =========================================================
# 13. MAKE PREDICTIONS
# =========================================================

print("\nPredicting Genres...\n")

y_pred = model.predict(X_test)

# =========================================================
# 14. MODEL EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================================================
# 15. TEST CUSTOM MOVIE PLOTS
# =========================================================

sample_movies = [

    "A young wizard discovers magical powers and battles evil forces.",

    "Two lovers struggle against family conflicts and emotional problems.",

    "A detective investigates a horrifying serial murder case.",

    "Astronauts travel across galaxies to save humanity.",

    "A hilarious comedian gets trapped in funny situations."
]

predictions = model.predict(sample_movies)

print("\n==============================")
print("CUSTOM MOVIE PREDICTIONS")
print("==============================\n")

for movie, genre in zip(sample_movies, predictions):

    print("Movie Plot:")
    print(movie)

    print("\nPredicted Genre:")
    print(genre)

    print("\n" + "-" * 60 + "\n")

# =========================================================
# 16. SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "movie_genre_prediction_model.pkl"
)

print("\nModel Saved Successfully!")

# =========================================================
# 17. LOAD MODEL AGAIN
# =========================================================

loaded_model = joblib.load(
    "movie_genre_prediction_model.pkl"
)

# =========================================================
# 18. TEST LOADED MODEL
# =========================================================

test_plot = [

    "A superhero protects the city from dangerous villains."
]

prediction = loaded_model.predict(test_plot)

print("\nPrediction From Loaded Model:")
print(prediction[0])

# =========================================================
# END
# =========================================================