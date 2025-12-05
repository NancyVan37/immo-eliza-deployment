# Streamlit/app.py
import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
import base64

def get_base64_image(image_path):
    # SAFER: check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Background image not found at: {image_path}")
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()
    
# Page config
st.set_page_config(page_title="Immo Eliza", page_icon="🏠", layout="centered")

# ----------------------------
# Path
# ----------------------------
IMAGE_PATH = "Streamlit/background.jpeg" 
encoded_img = get_base64_image(IMAGE_PATH)

# CSS with background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #001f3f;
        background-image: url("data:image/jpg;base64,{encoded_img}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #f5f5f5;
    }}
    .card {{
        background: rgba(245, 245, 240, 0.92);
        color: #0b2545;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }}
    .stButton>button {{
        background-color: #f5c16c !important;
        color: black !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar navigation
# ----------------------------
page = st.sidebar.radio("Navigation", ["Predict", "About", "Debug"])

# ----------------------------
# Utility: cached model loader (safer on deployment)
# ----------------------------
@st.cache_resource
def load_model(path: str = "model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    model = joblib.load(path)
    return model

# Try to load model and show friendly message if failure
try:
    model = load_model("model.pkl")
except Exception as e:
    st.sidebar.error("Model load error")
    st.error(
        "The model could not be loaded. Check model.pkl exists in the repository and is <=100MB.\n\n"
        f"Error: {e}"
    )
    st.stop()

# Helper to get expected columns from training preprocessor
def get_expected_columns(pipeline):
    """
    If you saved a pipeline where preprocessor is pipeline.named_steps['preprocessor'],
    it will expose transformers_ with column lists used at fit-time.
    """
    try:
        pre = pipeline.named_steps["preprocessor"]
        num_cols = list(pre.transformers_[0][2])
        cat_cols = list(pre.transformers_[1][2])
        return num_cols + cat_cols
    except Exception:
        # fallback: if you saved a raw model (not a pipeline)
        return None

EXPECTED_COLUMNS = get_expected_columns(model)

# ----------------------------
# PAGES
# ----------------------------
if page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🏠 Immo Eliza Price Predictor")
    st.write(
        "This app uses a trained pipeline (preprocessor + regressor). "
        "Fill inputs and press Predict."
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Property details")
    col1, col2 = st.columns(2)

    with col1:
        locality = st.text_input("Locality (city)", "")
        property_type = st.selectbox("Property Type", ["APARTMENT", "HOUSE"])
        subproperty_type = st.text_input("Subtype (optional)", "")
        region = st.text_input("Region (optional)", "")
        zip_code = st.number_input("Zip Code", min_value=1000, max_value=9999, value=1000, step=1)

    with col2:
        total_area_sqm = st.number_input("Total area (sqm)", min_value=1, max_value=10000, value=80)
        nbr_bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=2)
        garden_sqm = st.number_input("Garden size (sqm)", min_value=0, max_value=10000, value=0)
        terrace_sqm = st.number_input("Terrace size (sqm)", min_value=0, max_value=10000, value=0)
        epc = st.text_input("EPC (optional)", "")

    fl_garden = st.checkbox("Has garden")
    fl_terrace = st.checkbox("Has terrace")
    fl_furnished = st.checkbox("Furnished")
    fl_double_glazing = st.checkbox("Double glazing")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Predict Price"):
        # row with defaults for expected columns
        if EXPECTED_COLUMNS is None:
            # if we can't retrieve columns, try predict directly with minimal DF
            df_input = pd.DataFrame([{
                "locality": locality,
                "property_type": property_type,
                "nbr_bedrooms": nbr_bedrooms,
                "total_area_sqm": total_area_sqm,
                "zip_code": zip_code,
                "fl_garden": fl_garden,
                "garden_sqm": garden_sqm
            }])
        else:
            row = {c: None for c in EXPECTED_COLUMNS}
            #known fields if present in EXPECTED_COLUMNS
            mapping = {
                "locality": locality,
                "property_type": property_type,
                "subproperty_type": subproperty_type,
                "region": region,
                "zip_code": zip_code,
                "total_area_sqm": total_area_sqm,
                "nbr_bedrooms": nbr_bedrooms,
                "garden_sqm": garden_sqm,
                "terrace_sqm": terrace_sqm,
                "epc": epc,
                "fl_garden": fl_garden,
                "fl_terrace": fl_terrace,
                "fl_furnished": fl_furnished,
                "fl_double_glazing": fl_double_glazing
            }
            for k, v in mapping.items():
                if k in row:
                    row[k] = v
            df_input = pd.DataFrame([row])

        # Show debug info in collapsed expander
        with st.expander("Input preview (for debugging)"):
            st.write(df_input.T)

        # Predict with try/except and helpful missing-columns message
        try:
            pred = model.predict(df_input)[0]
            st.success(f"💶 Estimated price: € {int(round(pred)):,}")
        except Exception as e:
            st.error("Prediction error: the model raised an exception. See details below.")
            st.exception(e)
            # If expected columns available, show which are missing
            if EXPECTED_COLUMNS is not None:
                provided = set(df_input.columns)
                expected = set(EXPECTED_COLUMNS)
                missing = expected - provided
                extra = provided - expected
                st.write("Expected columns count:", len(expected))
                st.write("Provided columns count:", len(provided))
                st.write("Missing columns:", sorted(list(missing)))
                st.write("Extra columns (provided but not expected):", sorted(list(extra)))

elif page == "Debug":
    st.header("Diagnostics")
    st.write("App running. Model path:", Path("model.pkl").absolute())
    st.write("Model size (bytes):", os.path.getsize("model.pkl") if os.path.exists("model.pkl") else "not found")
    st.write("Expected columns (first 50):", EXPECTED_COLUMNS[:50] if EXPECTED_COLUMNS else "Not available")
    st.write("Python version:", f"{os.sys.version}")
    st.write("Streamlit version:", st.__version__)





