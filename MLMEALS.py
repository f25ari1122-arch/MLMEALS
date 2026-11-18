import sys, json, pickle, zipfile
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

REQUIRED_SKLEARN = "1.6.1"
print("Python:", sys.version.split()[0], "| sklearn:", sklearn.__version__)
if sklearn.__version__ != REQUIRED_SKLEARN:
    raise RuntimeError(
        f"Install scikit-learn=={REQUIRED_SKLEARN} in Colab, restart runtime, "
        "then run this file again."
    )

# Grocery database
groceries = pd.DataFrame([
    ["Rice", "Grains", "kg", 300], ["Chicken", "Meat", "kg", 750],
    ["Eggs", "Protein", "piece", 30], ["Potatoes", "Vegetables", "kg", 100],
    ["Onions", "Vegetables", "kg", 120], ["Tomatoes", "Vegetables", "kg", 150],
    ["Lentils", "Pulses", "kg", 280], ["Bread", "Bakery", "piece", 80],
    ["Milk", "Dairy", "litre", 220], ["Oil", "Cooking", "litre", 550]
], columns=["Item", "Category", "Unit", "Price"])

# Demonstration historical price data (replace with real historical data for
# meaningful real-world forecasting).
price_factors = [0.90, 0.92, 0.95, 0.97, 0.96, 1.00, 1.02, 1.05, 1.03, 1.06, 1.08, 1.10]
price_rows = []
for _, item in groceries.iterrows():
    for month, factor in enumerate(price_factors, start=1):
        price_rows.append([item["Item"], month, round(item["Price"] * factor, 2)])
price_df = pd.DataFrame(price_rows, columns=["Item", "Month", "Price"])

# Train Random Forest price prediction model
preprocessor = ColumnTransformer([
    ("item", OneHotEncoder(handle_unknown="ignore"), ["Item"]),
    ("month", "passthrough", ["Month"])
])
price_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])
price_model.fit(price_df[["Item", "Month"]], price_df["Price"])

next_month = (datetime.now().month % 12) + 1
groceries["Predicted_Price"] = [
    round(float(price_model.predict(pd.DataFrame({"Item": [item], "Month": [next_month]}))[0]), 2)
    for item in groceries["Item"]
]

# Meal database
meals = pd.DataFrame([
    ["Omelette", "Breakfast", "Vegetarian", {"Eggs": 2, "Onions": 0.05}, 10],
    ["Boiled Eggs", "Breakfast", "Vegetarian", {"Eggs": 2}, 10],
    ["Bread and Milk", "Breakfast", "Vegetarian", {"Bread": 1, "Milk": 0.25}, 5],
    ["Chicken Rice", "Lunch", "Non-Vegetarian", {"Rice": 0.15, "Chicken": 0.15, "Onions": 0.05}, 30],
    ["Potato Curry", "Lunch", "Vegetarian", {"Potatoes": 0.2, "Onions": 0.05, "Tomatoes": 0.05}, 25],
    ["Lentil Curry", "Lunch", "Vegetarian", {"Lentils": 0.12, "Onions": 0.05, "Tomatoes": 0.05}, 25],
    ["Chicken Biryani", "Dinner", "Non-Vegetarian", {"Rice": 0.15, "Chicken": 0.15, "Potatoes": 0.1}, 40],
    ["Egg Curry", "Dinner", "Vegetarian", {"Eggs": 2, "Tomatoes": 0.1, "Onions": 0.05}, 25],
    ["Lentil Rice", "Dinner", "Vegetarian", {"Rice": 0.12, "Lentils": 0.1}, 25]
], columns=["Meal", "Category", "Preference", "Ingredients", "Preparation_Time"])

# Train TF-IDF meal recommendation model
meal_text = [
    row["Meal"] + " " + row["Category"] + " " + row["Preference"] + " " +
    " ".join(row["Ingredients"].keys())
    for _, row in meals.iterrows()
]
vectorizer = TfidfVectorizer()
meal_vectors = vectorizer.fit_transform(meal_text)
meal_recommender = {"vectorizer": vectorizer, "meal_vectors": meal_vectors, "meals": meals}

# Save artifacts and version metadata
folder = Path.cwd()
artifacts = {
    "price_model.pkl": price_model,
    "meal_recommender.pkl": meal_recommender,
    "grocery_database.pkl": groceries
}
for filename, obj in artifacts.items():
    with open(folder / filename, "wb") as f:
        pickle.dump(obj, f)

price_df.to_csv(folder / "price_history.csv", index=False)
metadata = {
    "project": "SmartBite",
    "scikit_learn_version": sklearn.__version__,
    "created_at": datetime.now().isoformat()
}
with open(folder / "model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

zip_path = folder / "SmartBite_Trained_Models.zip"
files_to_zip = list(artifacts.keys()) + ["price_history.csv", "model_metadata.json"]
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for filename in files_to_zip:
        z.write(folder / filename, arcname=filename)

print("Training complete. Created:", zip_path)
try:
    from google.colab import files
    files.download(str(zip_path))
except ImportError:
    print("ZIP saved in the current folder.")
