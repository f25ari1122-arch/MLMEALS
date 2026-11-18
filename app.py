
import json
import pickle
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import sklearn
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartBite | Weekly Meal Planner",
    page_icon="🍽️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "SmartBite_Trained_Models.zip"
REQUIRED_SKLEARN = "1.6.1"


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    required_files = [
        "price_model.pkl",
        "meal_recommender.pkl",
        "grocery_database.pkl",
        "model_metadata.json"
    ]

    missing_files = [
        name for name in required_files
        if not (BASE_DIR / name).exists()
    ]

    if missing_files:

        if not ZIP_PATH.exists():
            raise FileNotFoundError(
                "SmartBite_Trained_Models.zip was not found. "
                "Upload it beside app.py."
            )

        with zipfile.ZipFile(ZIP_PATH, "r") as archive:

            available = archive.namelist()

            missing_in_zip = [
                name for name in missing_files
                if name not in available
            ]

            if missing_in_zip:
                raise FileNotFoundError(
                    f"Missing files inside ZIP: {missing_in_zip}"
                )

            for name in missing_files:
                with archive.open(name) as source:
                    with open(BASE_DIR / name, "wb") as destination:
                        destination.write(source.read())

    with open(BASE_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    trained_version = metadata.get("scikit_learn_version")

    if sklearn.__version__ != REQUIRED_SKLEARN:
        raise RuntimeError(
            f"Installed Scikit-learn version is {sklearn.__version__}. "
            f"Required version is {REQUIRED_SKLEARN}."
        )

    if trained_version != sklearn.__version__:
        raise RuntimeError(
            f"Model trained with Scikit-learn {trained_version}, "
            f"but application uses {sklearn.__version__}."
        )

    with open(BASE_DIR / "price_model.pkl", "rb") as f:
        price_model = pickle.load(f)

    with open(BASE_DIR / "meal_recommender.pkl", "rb") as f:
        recommender = pickle.load(f)

    with open(BASE_DIR / "grocery_database.pkl", "rb") as f:
        groceries = pickle.load(f)

    return price_model, recommender, groceries, metadata


# ============================================================
# WEEKLY MEAL PLAN GENERATION
# ============================================================

def generate_plan(
    budget,
    people,
    preference,
    preferred_foods,
    recommender,
    groceries
):

    meals = recommender["meals"].copy()
    vectorizer = recommender["vectorizer"]
    meal_vectors = recommender["meal_vectors"]

    # User preference query
    query = preferred_foods.strip()

    if not query:
        query = "balanced healthy meals"

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        meal_vectors
    )[0]

    meals["Similarity"] = similarities

    # Filter preference
    if preference != "Any":
        meals = meals[
            meals["Preference"] == preference
        ].copy()

    if meals.empty:
        raise ValueError("No meals available for this preference.")

    # Grocery price dictionary
    price_dict = dict(
        zip(groceries["Item"], groceries["Predicted_Price"])
    )

    unit_dict = dict(
        zip(groceries["Item"], groceries["Unit"])
    )

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    categories = [
        "Breakfast",
        "Lunch",
        "Dinner"
    ]

    grocery_quantities = {}
    meal_counts = {
        name: 0 for name in meals["Meal"]
    }

    selected_meals = []

    # Generate 21 meals
    for day in days:

        for category in categories:

            options = meals[
                meals["Category"] == category
            ].copy()

            best_meal = None
            best_score = float("inf")
            best_quantities = None
            best_cost = 0

            for _, meal in options.iterrows():

                ingredients = meal["Ingredients"]

                new_quantities = grocery_quantities.copy()

                for item, amount in ingredients.items():

                    if item not in price_dict:
                        continue

                    new_quantities[item] = (
                        new_quantities.get(item, 0)
                        + float(amount) * people
                    )

                # Calculate total weekly grocery cost
                total_cost = sum(
                    amount * float(price_dict[item])
                    for item, amount in new_quantities.items()
                )

                # Never exceed the weekly budget
                if total_cost > budget:
                    continue

                # Recommendation score
                repetition_penalty = (
                    meal_counts[meal["Meal"]] * 2
                )

                cost_penalty = (
                    total_cost / max(budget, 1)
                )

                score = (
                    (1 - meal["Similarity"]) * 10
                    + repetition_penalty
                    + cost_penalty
                )

                if score < best_score:

                    best_score = score
                    best_meal = meal
                    best_quantities = new_quantities
                    best_cost = total_cost

            if best_meal is None:

                selected_meals.append({
                    "Day": day,
                    "Category": category,
                    "Meal": "No meal available",
                    "Preparation Time (min)": 0,
                    "Estimated Total Cost (PKR)": 0
                })

            else:

                grocery_quantities = best_quantities

                meal_counts[best_meal["Meal"]] += 1

                selected_meals.append({
                    "Day": day,
                    "Category": category,
                    "Meal": best_meal["Meal"],
                    "Preparation Time (min)": int(
                        best_meal["Preparation_Time"]
                    ),
                    "Estimated Total Cost (PKR)": round(
                        best_cost, 2
                    )
                })

    # ========================================================
    # CREATE GROCERY LIST
    # ========================================================

    grocery_rows = []

    for item, quantity in grocery_quantities.items():

        price = float(price_dict[item])

        grocery_rows.append({
            "Grocery Item": item,
            "Quantity": round(quantity, 2),
            "Unit": unit_dict[item],
            "Predicted Unit Price (PKR)": round(price, 2),
            "Total Price (PKR)": round(quantity * price, 2)
        })

    grocery_df = pd.DataFrame(grocery_rows)

    if not grocery_df.empty:
        grocery_df = grocery_df.sort_values(
            "Grocery Item"
        ).reset_index(drop=True)

        total_grocery_cost = grocery_df[
            "Total Price (PKR)"
        ].sum()

    else:
        total_grocery_cost = 0

    remaining_budget = budget - total_grocery_cost

    meal_df = pd.DataFrame(selected_meals)

    return (
        meal_df,
        grocery_df,
        round(total_grocery_cost, 2),
        round(remaining_budget, 2)
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🍽️ SmartBite")

st.subheader("Smart Weekly Grocery Budget & Meal Planner")

st.write(
    "Plan your weekly meals and grocery shopping using "
    "machine learning. Enter your budget and preferences "
    "to generate a personalized weekly plan."
)

st.divider()


# ============================================================
# LOAD APPLICATION DATA
# ============================================================

try:

    price_model, recommender, groceries, metadata = load_models()

    st.success("Machine learning models loaded successfully.")

except Exception as error:

    st.error("Unable to load the trained models.")

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.title("⚙️ Meal Planner Settings")

st.sidebar.subheader("Your Weekly Requirements")

budget = st.sidebar.number_input(
    "Weekly Grocery Budget (PKR)",
    min_value=100,
    max_value=1000000,
    value=10000,
    step=500
)

people = st.sidebar.number_input(
    "Number of People",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)

preference = st.sidebar.selectbox(
    "Food Preference",
    [
        "Any",
        "Vegetarian",
        "Non-Vegetarian"
    ]
)

preferred_foods = st.sidebar.text_input(
    "Preferred Foods or Ingredients",
    placeholder="e.g. Rice, Chicken, Eggs"
)

st.sidebar.caption(
    "Enter your weekly budget and food preferences "
    "to generate your meal plan."
)

generate_button = st.sidebar.button(
    "Generate Weekly Plan 🍽️",
    use_container_width=True,
    type="primary"
)


# ============================================================
# DISPLAY DEFAULT INFORMATION
# ============================================================

if "plan_generated" not in st.session_state:
    st.session_state.plan_generated = False


if generate_button:

    try:

        with st.spinner(
            "Generating your personalized weekly meal plan..."
        ):

            meal_df, grocery_df, total_cost, remaining = generate_plan(
                budget,
                people,
                preference,
                preferred_foods,
                recommender,
                groceries
            )

            st.session_state.meal_df = meal_df
            st.session_state.grocery_df = grocery_df
            st.session_state.total_cost = total_cost
            st.session_state.remaining = remaining
            st.session_state.budget = budget
            st.session_state.people = people
            st.session_state.plan_generated = True

    except Exception as error:

        st.error("Unable to generate the meal plan.")
        st.exception(error)


# ============================================================
# RESULTS
# ============================================================

if st.session_state.plan_generated:

    meal_df = st.session_state.meal_df
    grocery_df = st.session_state.grocery_df
    total_cost = st.session_state.total_cost
    remaining = st.session_state.remaining
    current_budget = st.session_state.budget

    st.header("📊 Weekly Budget Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Weekly Budget",
        f"Rs. {current_budget:,.2f}"
    )

    col2.metric(
        "Estimated Grocery Cost",
        f"Rs. {total_cost:,.2f}"
    )

    col3.metric(
        "Remaining Budget",
        f"Rs. {remaining:,.2f}",
        delta=f"Rs. {remaining:,.2f}"
    )

    if total_cost > 0:

        st.progress(
            min(total_cost / current_budget, 1.0)
        )

    if (meal_df["Meal"] == "No meal available").any():

        st.warning(
            "Some meals could not be selected within your budget. "
            "Consider increasing your budget."
        )

    else:

        st.success(
            "Your complete 7-day meal plan was generated!"
        )

    st.divider()

    # ========================================================
    # WEEKLY MEAL SCHEDULE
    # ========================================================

    st.header("📅 Your Weekly Meal Schedule")

    st.caption(
        "Breakfast, lunch and dinner for all seven days."
    )

    for day in meal_df["Day"].unique():

        day_data = meal_df[
            meal_df["Day"] == day
        ]

        with st.expander(day, expanded=True):

            for _, meal in day_data.iterrows():

                col1, col2, col3 = st.columns(
                    [2, 3, 2]
                )

                col1.write(f"**{meal['Category']}**")

                col2.write(f"🍽️ {meal['Meal']}")

                col3.write(
                    f"⏱️ {meal['Preparation Time (min)']} min"
                )

    st.divider()

    # ========================================================
    # GROCERY SHOPPING LIST
    # ========================================================

    st.header("🛒 Weekly Grocery Shopping List")

    st.caption(
        f"Quantities calculated for "
        f"{st.session_state.people} people."
    )

    if not grocery_df.empty:

        st.dataframe(
            grocery_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            f"Total Grocery Cost: Rs. {total_cost:,.2f}"
        )

        csv_data = grocery_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Grocery List (CSV)",
            data=csv_data,
            file_name="SmartBite_Grocery_List.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "No groceries were selected. Try increasing your budget."
        )

    st.divider()

    # ========================================================
    # MEAL SUMMARY
    # ========================================================

    st.header("📈 Meal Plan Summary")

    valid_meals = meal_df[
        meal_df["Meal"] != "No meal available"
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Meals Generated",
        len(valid_meals)
    )

    col2.metric(
        "Total Planned Meals",
        21
    )

    col3.metric(
        "People",
        st.session_state.people
    )

    if not valid_meals.empty:

        st.subheader("Meal Category Distribution")

        category_counts = valid_meals[
            "Category"
        ].value_counts()

        st.bar_chart(category_counts)

else:

    st.info(
        "👈 Enter your weekly budget and food preferences "
        "in the sidebar, then click Generate Weekly Plan."
    )

    st.subheader("What SmartBite Can Do")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### 🤖 Machine Learning
            - Random Forest grocery price prediction
            - TF-IDF meal recommendation
            - Personalized meal selection
            """
        )

    with col2:
        st.markdown(
            """
            ### 🛒 Weekly Planning
            - Seven-day meal schedule
            - Grocery quantity calculation
            - Budget-based grocery list
            - Estimated grocery expenses
            """
        )

st.divider()

st.caption(
    "SmartBite | Machine Learning Based Weekly Meal Planner"
)
