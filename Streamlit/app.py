import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("model.pkl")

st.title("🏠 Immo Eliza Price Predictor")
st.write("Fill in the property details to estimate the price.")

# -------------------------------------------------------------
# Step 1 — Minimal UI for important inputs
# -------------------------------------------------------------
locality = st.text_input("Locality", "")
property_type = st.selectbox("Property Type", ["APARTMENT", "HOUSE"])
nbr_bedrooms = st.number_input("Number of bedrooms", 0, 20, 2)
total_area_sqm = st.number_input("Total area (sqm)", 10, 10000, 100)
zip_code = st.number_input("Zip Code", 1000, 9999, 1000)
fl_garden = st.checkbox("Has garden?")
garden_sqm = st.number_input("Garden size (sqm)", 0, 5000, 0)

# -------------------------------------------------------------
# Step 2 — Get ALL training columns from the model pipeline
# -------------------------------------------------------------
# Extract the preprocessor inside the pipeline
preprocessor = model.named_steps['preprocessor']

# Numeric + categorical columns used during training
numeric_cols = list(preprocessor.transformers_[0][2])
categorical_cols = list(preprocessor.transformers_[1][2])

ALL_COLUMNS = numeric_cols + categorical_cols

# -------------------------------------------------------------
# Step 3 — Create a complete row for prediction
# -------------------------------------------------------------
def build_input_row():
    """
    Build a FULL row containing ALL columns that the model expects.
    Missing columns get default values (None).
    """
    row = {col: None for col in ALL_COLUMNS}

    # Fill the columns we actually have UI for:
    if "locality" in row: row["locality"] = locality
    if "property_type" in row: row["property_type"] = property_type
    if "nbr_bedrooms" in row: row["nbr_bedrooms"] = nbr_bedrooms
    if "total_area_sqm" in row: row["total_area_sqm"] = total_area_sqm
    if "zip_code" in row: row["zip_code"] = zip_code
    if "fl_garden" in row: row["fl_garden"] = fl_garden
    if "garden_sqm" in row: row["garden_sqm"] = garden_sqm

    return pd.DataFrame([row])


# -------------------------------------------------------------
# Step 4 — Predict
# -------------------------------------------------------------
if st.button("Predict Price"):

    df_input = build_input_row()

    try:
        prediction = model.predict(df_input)[0]
        st.success(f"💶 Estimated price: **€ {round(prediction):,}**")

    except Exception as e:
        st.error(f"⚠️ Error during prediction:\n\n{e}")



