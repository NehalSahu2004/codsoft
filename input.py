# =========================================================
# MOVIE GENRE PREDICTION USING SAVED MODEL
# =========================================================

import joblib

# =========================================================
# LOAD SAVED MODEL
# =========================================================

model = joblib.load("movie_genre_prediction_model.pkl")

print("Model Loaded Successfully!")

# =========================================================
# USER INPUT LOOP
# =========================================================

print("\n===================================")
print("MOVIE GENRE PREDICTION SYSTEM")
print("Type 'exit' to stop")
print("===================================\n")

while True:

    # Take input from user
    movie_plot = input("Enter Movie Plot:\n")

    # Exit condition
    if movie_plot.lower() == "exit":
        print("\nProgram Ended.")
        break

    # Predict genre
    prediction = model.predict([movie_plot])

    # Display result
    print("\nPredicted Genre:", prediction[0])

    print("\n" + "=" * 50 + "\n")