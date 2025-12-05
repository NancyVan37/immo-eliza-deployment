import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Config (title + layout)
# -----------------------------
st.set_page_config(
    page_title="Immo Eliza Price Predictor",
    layout="centered",
)

# -----------------------------
# Custom CSS (Navy theme)
# -----------------------------
st.markdown("""
<style>

body {
    background-color: #001f3f !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #001f3f;
}

[data-testid="stSidebar"] {
    background-color: #001933;
}

h1, h2, h3, h4, label, p, span {
    color: #ffffff !important;
}

.stButton > button {
    background-color: #0074D9 !important;
    color: white !important;
    width: 100%;
    border-radius: 10px;
    padding: 0.6rem;
}

.stTextInput > div > div > input,
.stNumberInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
}

.stSelectbox > div > div {
    background-color: #ffffff !important;
    color: black !important;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["🏠 Home", "🔮 Predict Price", "ℹ️ About"]
)

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("model.pkl")
preprocessor = model.named_steps["preprocessor"]

numeric_cols = list(preprocessor.transformers_[0][2])
categorical_cols = list(preprocessor.transformers_[1][2])
ALL_COLUMNS = numeric_cols + categorical_cols


# ============================================================
#  HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.title("🏠 Immo Eliza Price Predictor")
    st.write("""
        Welcome to the **Immo Eliza Price Predictor**.

        Use the sidebar to navigate to the prediction tool.
    """)


# ============================================================
#  PREDICT PRICE PAGE
# ============================================================
elif page == "🔮 Predict Price":

    st.title("🔮 Predict the Property Price")

    locality = st.text_input("Locality", "")
    property_type = st.selectbox("Property Type", ["APARTMENT", "HOUSE"])
    nbr_bedrooms = st.number_input("Number of bedrooms", 0, 20, 2)
    total_area_sqm = st.number_input("Total area (sqm)", 10, 10000, 100)
    zip_code = st.number_input("Zip Code", 1000, 9999, 1000)
    fl_garden = st.checkbox("Has garden?")
    garden_sqm = st.number_input("Garden size (sqm)", 0, 5000, 0)

    def build_input_row():
        row = {col: None for col in ALL_COLUMNS}

        # Fill the columns we actually have UI for
        row.update({
            "locality": locality,
            "property_type": property_type,
            "nbr_bedrooms": nbr_bedrooms,
            "total_area_sqm": total_area_sqm,
            "zip_code": zip_code,
            "fl_garden": fl_garden,
            "garden_sqm": garden_sqm,
        })

        return pd.DataFrame([row])

    if st.button("💶 Predict Price"):
        df_input = build_input_row()
        try:
            prediction = model.predict(df_input)[0]
            st.success(f"💶 Estimated price: **€ {round(prediction):,}**")
        except Exception as e:
            st.error(f"⚠️ Error during prediction:\n\n{e}")


# ============================================================
#  ABOUT PAGE
# ============================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.write("""
        This app predicts real-estate prices in Belgium using a machine learning model.
        
        Built with:
        - Streamlit  
        - Scikit-learn  
        - Python  
        - Immo Eliza public dataset  
    """)




