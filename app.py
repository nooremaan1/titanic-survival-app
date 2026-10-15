
import streamlit as st
import joblib
import pandas as pd

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Titanic | Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CUSTOM DESIGN 
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #101D32 0%, #172940 100%);
        color: #F5F1E8;
    }

    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p, label {
        color: #F5F1E8 !important;
    }

    .hero {
        text-align: center;
        padding: 28px 18px;
        border: 1px solid #BFA36A;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.04);
        margin-bottom: 24px;
    }

    .ship-icon {
        font-size: 48px;
        margin-bottom: 8px;
    }

    .hero h1 {
        color: #E5C780 !important;
        font-size: 36px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #D6DFE9 !important;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-title {
        color: #E5C780 !important;
        font-size: 22px;
        font-weight: 600;
        margin: 20px 0 12px;
    }

    div[data-baseweb="select"] > div {
        background-color: #F8F6F0;
        border-radius: 9px;
    }

    div[data-baseweb="select"] * {
        color: #172940;
    }

    div.stButton > button {
        width: 100%;
        background: #D6B66D;
        color: #142238;
        border: none;
        border-radius: 10px;
        padding: 13px 20px;
        font-weight: 700;
        font-size: 17px;
        transition: 0.2s;
    }

    div.stButton > button:hover {
        background: #E8CD8C;
        color: #101D32;
        border: none;
    }

    .result-card {
        padding: 22px;
        border-radius: 14px;
        text-align: center;
        margin-top: 22px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid #BFA36A;
    }

    .result-card h2 {
        color: #E5C780 !important;
        margin-bottom: 8px;
    }

    .footer {
        text-align: center;
        color: #AAB8C8;
        font-size: 12px;
        margin-top: 35px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# LOAD TRAINED MODEL
@st.cache_resource
def load_model():
    return joblib.load("svc_trained_model.pkl")

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

# HEADER 
st.markdown("""
<div class="hero">
    <div class="ship-icon">🚢</div>
    <h1>Titanic Survival Predictor</h1>
    <p>Explore how passenger characteristics relate to survival
    using a machine learning model.</p>
</div>
""", unsafe_allow_html=True)

#PASSENGER INPUTS
st.markdown(
    '<div class="section-title">Passenger Information</div>',
    unsafe_allow_html=True
)

st.write("Choose the passenger details below to generate a prediction.")

col1, col2 = st.columns(2)

with col1:
    passenger_class = st.selectbox(
        "Passenger Class",
        ["First", "Second", "Third"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    sibling = st.selectbox(
        "Sibling Category",
        ["Zero", "One", "Two", "Three"]
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["Southampton", "Cherbourg", "Queenstown"]
    )

st.caption(
    "The form uses the four input features required by the trained model."
)

# ENCODING 
# These mappings follow alphabetical LabelEncoder class ordering.
pclass_mapping = {
    "First": 0,
    "Second": 1,
    "Third": 2
}

gender_mapping = {
    "Female": 0,
    "Male": 1
}

sibling_mapping = {
    "One": 0,
    "Three": 1,
    "Two": 2,
    "Zero": 3
}

embarked_mapping = {
    "Cherbourg": 0,
    "Queenstown": 1,
    "Southampton": 2
}

#PREDICTION 
st.markdown("---")

if st.button("⚓ Predict Survival"):

    input_data = pd.DataFrame(
        [[
            pclass_mapping[passenger_class],
            gender_mapping[gender],
            sibling_mapping[sibling],
            embarked_mapping[embarked]
        ]],
        columns=["PClass", "Gender", "Sibling", "Embarked"]
    )

    try:
        prediction = model.predict(input_data)[0]

        st.markdown(
            '<div class="result-card">'
            '<h2>Prediction Result</h2>',
            unsafe_allow_html=True
        )

        # Assumes the target encoding is No = 0 and Yes = 1.
        if prediction == 1:
            st.success("Likely to Survive")
            st.write(
                "The model predicts that this passenger "
                "is likely to survive."
            )
        else:
            st.warning("Unlikely to Survive")
            st.write(
                "The model predicts that this passenger "
                "is unlikely to survive."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# MODEL INFORMATION 
with st.expander("About this machine learning project"):
    st.write(
        "This application uses a Support Vector Classifier (SVC) "
        "trained on Titanic passenger data."
    )
    st.write(
        "Inputs: passenger class, gender, sibling category, "
        "and port of embarkation."
    )
    st.write(
        "The prediction is an estimate from the model, "
        "not a guarantee of the actual outcome."
    )

# FOOTER
st.markdown("""
<div class="footer">
    MACHINE LEARNING FUNDAMENTALS · TITANIC SURVIVAL PROJECT
</div>
""", unsafe_allow_html=True)
