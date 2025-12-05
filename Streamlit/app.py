import streamlit as st
import pandas as pd
import joblib

# =====================================================================
# ----------------------  PAGE CONFIG & THEME  -------------------------
# =====================================================================
st.set_page_config(
    page_title="Immo Eliza Price Predictor",
    layout="centered"
)

# Light beige base + modern clean card UI
st.markdown("""
<style>
    body {
        background-color: #f7f2e8;
    }
    .main {
        background-color: #f7f2e8;
    }
    .stTextInput>div>div>input {
        background-color: #fff;
    }
    .stSelectbox>div>div>div {
        background-color: #fff;
    }
    .card {
        padding: 25px;
        background-color: white;
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e6dfd3;
        margin-top: 20px;
    }
    .title {
        text-align: center;
        font-size: 36px !important;
        font-weight: 700 !important;
        color: #5c5246;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #7e7468;
        margin-bottom: 20px;
        font-size: 17px;
    }
    .predict-btn button {
        width: 100%;
        background-color: #c9b8a6 !important;
        color: white !important;
        border-radius: 10px !important;
        font-size: 18px !important;
        padding: 10px 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# -------------------------  LOAD MODEL  -------------------------------
# =====================================================================
model = joblib.load("model.pkl")

# Page Title
st.markdown('<div class="title">🏠 Immo Eliza Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter property details to estimate the price</div>', unsafe_allow_html=True)


# =====================================================================
# -------------------------  INPUT CARD  -------------------------------
# =====================================================================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📌 Property Details")

    col1, col2 = st.columns(2)

    with col1:
        locality = st.text_input("Locality")
        property_type = st.selectbox("Property Type", ["APARTMENT", "HOUSE"])
        nbr_bedrooms = st.number_input("Bedrooms", 0, 20, 2)

    with col2:
        total_area_sqm = st.number_input("Total area (sqm)", 10, 10000, 100)
        zip_code = st.number_input("Zip Code", 1000, 9999, 1000)
        fl_garden = st.checkbox("Has garden?")

    garden_sqm = st.number_input("Garden size (sqm)", 0, 5000, 0)

    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# -----------  Extract Training Columns From Model Pipeline -----------
# =====================================================================
preprocessor = model.named_steps['preprocessor']
numeric_cols = list(preprocessor.transformers_[0][2])
categorical_cols = list(preprocessor.transformers_[1][2])
ALL_COLUMNS = numeric_cols + categorical_cols


# =====================================================================
# ----------------------  BUILD INPUT ROW  -----------------------------
# =====================================================================
def build_input_row():
    row = {col: None for col in ALL_COLUMNS}

    if "locality" in row: row["locality"] = locality
    if "property_type" in row: row["property_type"] = property_type
    if "nbr_bedrooms" in row: row["nbr_bedrooms"] = nbr_bedrooms
    if "total_area_sqm" in row: row["total_area_sqm"] = total_area_sqm
    if "zip_code" in row: row["zip_code"] = zip_code
    if "fl_garden" in row: row["fl_garden"] = fl_garden
    if "garden_sqm" in row: row["garden_sqm"] = garden_sqm

    return pd.DataFrame([row])


# =====================================================================
# ---------------------------  PREDICT  -------------------------------
# =====================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

if st.button("Predict Price", type="primary"):
    df_input = build_input_row()

    try:
        prediction = model.predict(df_input)[0]

        st.success(f"💶 Estimated price: **€ {round(prediction):,}**")

    except Exception as e:
        st.error(f"⚠️ Error during prediction:\n\n{e}")

st.markdown('</div>', unsafe_allow_html=True)



