import json
import pickle
import zipfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SmartBite | Weekly Meal Planner", page_icon="🍽️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "SmartBite_Trained_Models.zip"
REQUIRED_SKLEARN = "1.6.1"

st.title("🍽️ SmartBite")
st.subheader("Smart Weekly Grocery Budget & Meal Planner")
st.write("Generate a weekly meal schedule and grocery shopping list according to your budget and food preferences.")

def load_models():
    required = ["price_model.pkl", "meal_recommender.pkl", "grocery_database.pkl", "model_metadata.json"]
    if any(not (BASE_DIR / name).exists() for name in required):
        if not ZIP_PATH.exists():
            raise FileNotFoundError("Add SmartBite_Trained_Models.zip beside app.py in your GitHub repository.")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(BASE_DIR)

    with open(BASE_DIR / "model_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    trained_version = metadata.get("scikit_learn_version")
    if sklearn.__version__ != REQUIRED_SKLEARN or trained_version != sklearn.__version__:
        raise RuntimeError(
            f"Scikit-learn version mismatch. Model version: {trained_version}; "
            f"app version: {sklearn.__version__}. Set scikit-learn=={REQUIRED_SKLEARN} "
            "in requirements.txt and retrain the models with that same version."
        )

    with open(BASE_DIR / "price_model.pkl", "rb") as f:
        price_model = pickle.load(f)
    with open(BASE_DIR / "meal_recommender.pkl", "rb") as f:
        recommender = pickle.load(f)
    with open(BASE_DIR / "grocery_database.pkl", "rb") as f:
        groceries = pickle.load(f)
    return price_model, recommender, groceries

@st.cache_resource
def get_models():
    return load_models()

def create_plan(budget, people, preference, preferred_foods, groceries, meals, vectorizer, meal_vectors):
    query = preference + " " + preferred_foods
    similarities = cosine_similarity(vectorizer.transform([query]), meal_vectors)[0]
    meals = meals.copy()
    meals["Similarity"] = similarities

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    categories = ["Breakfast", "Lunch", "Dinner"]
    grocery_quantities = {}
    selected = []
    counts = {name: 0 for name in meals["Meal"]}

    for day in days:
        for category in categories:
            options = meals[meals["Category"] == category].copy()
            if preference != "Any":
                options = options[options["Preference"] == preference]

            best_meal, best_score, best_quantities = None, None, None
            for _, meal in options.iterrows():
                quantities = grocery_quantities.copy()
                for item, amount in meal["Ingredients"].items():
                    quantities[item] = quantities.get(item, 0) + amount * people

                cost = sum(
                    amount * float(groceries.loc[groceries["Item"] == item, "Predicted_Price"].iloc[0])
                    for item, amount in quantities.items()
                )
                if cost <= budget:
                    score = (1 - meal["Similarity"]) * 100 + counts[meal["Meal"]] * 20 + (cost / budget) * 10
                    if best_score is None or score < best_score:
                        best_meal, best_score, best_quantities = meal, score, quantities

            if best_meal is None:
                selected.append({"Day": day, "Category": category, "Meal": "No meal available",
                                 "Ingredients": {}, "Preparation Time (min)": 0})
            else:
                grocery_quantities = best_quantities
                counts[best_meal["Meal"]] += 1
                selected.append({"Day": day, "Category": category, "Meal": best_meal["Meal"],
                                 "Ingredients": best_meal["Ingredients"],
                                 "Preparation Time (min)": best_meal["Preparation_Time"]})

    grocery_rows = []