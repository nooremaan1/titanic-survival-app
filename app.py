
import streamlit as st
import joblib
import pandas as pd

# Configure the Streamlit webpage
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Load the trained machine learning model
@st.cache_resource
def load_model():
    return joblib.load("svc_trained_model.pkl")

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# Display the app title and introduction
st.title("🚢 Titanic Passenger Survival Prediction")
st.write(
    "Enter the passenger details below to predict "
    "whether the passenger is likely to survive."
)

# Collect passenger information
passenger_class = st.selectbox(
    "Passenger Class",
    ["First", "Second", "Third"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

sibling = st.selectbox(
    "Number of Siblings",
    ["Zero", "One", "Two", "Three"]
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["Cherbourg", "Queenstown", "Southampton"]
)

# Convert the categorical inputs to the integer
# encodings used by the trained model.
# These mappings assume the notebook's LabelEncoder
# classes were fitted in alphabetical order.
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

# Predict survival when the button is clicked
if st.button("Predict Survival", type="primary"):

    # Arrange the input features in the model's expected order
    input_data = pd.DataFrame(
        [[
            pclass_mapping[passenger_class],
            gender_mapping[gender],
            sibling_mapping[sibling],
            embarked_mapping[embarked]
        ]],
        columns=["PClass", "Gender", "Sibling", "Embarked"]
    )

    # Generate the prediction
    prediction = model.predict(input_data)[0]

    # The target LabelEncoder is assumed to use No=0, Yes=1
    if prediction == 1:
        st.success("Prediction: The passenger is likely to survive.")
    else:
        st.error("Prediction: The passenger is unlikely to survive.")

    st.caption(
        "This result is a machine learning prediction, "
        "not a guarantee of the actual outcome."
    )
