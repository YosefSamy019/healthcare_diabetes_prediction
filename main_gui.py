import math
import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Load encoders and scalers
ENCODERS_SCALERS_DIR = "encoders_scalers"
MODELS_CACHE_DIR = "models_cache"

loaded_objects = {}
if os.path.exists(ENCODERS_SCALERS_DIR):
    for file in os.listdir(ENCODERS_SCALERS_DIR):
        if file.endswith(".pickle"):
            with open(os.path.join(ENCODERS_SCALERS_DIR, file), "rb") as f:
                loaded_objects[file] = pickle.load(f)

st.title("🩺 Diabetes Prediction App")
st.warning("This project is **educational only**. It is **not a medical tool** and should not be used for real-world diagnosis or treatment.")

# Feature input tabs
main_tabs = st.tabs(["Patient Input", "Evaluation Dataset"])

with main_tabs[0]:
    st.header("Patient Input")

    # Model picker
    model_files = [f for f in os.listdir(MODELS_CACHE_DIR) if f.endswith(".pickle")] if os.path.exists(MODELS_CACHE_DIR) else []
    selected_model_file = st.selectbox("Choose model pickle to load", model_files)

    if selected_model_file:
        with open(os.path.join(MODELS_CACHE_DIR, selected_model_file), "rb") as f:
            model = pickle.load(f)
    else:
        model = None

    # Input fields
    col0, col1, col2 = st.columns([1,1,1])
    with col0:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI", min_value=9.0, max_value=60.0, value=22.0)
    with col1:
        blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
        glucose = st.number_input("Glucose", min_value=50, max_value=300, value=100)
    with col2:
        family_history = st.selectbox("Family History", ["Yes", "No"])
        physical_activity = st.selectbox("Physical Activity", ['High', 'Medium', 'Low'])

    if st.button("Predict"):
        # Preprocess input
        data_dict = {
            "Age": age,
            "BMI": bmi,
            "BloodPressure": blood_pressure,
            "GlucoseLevel": glucose,
            "PhysicalActivity": physical_activity,
            'exp (Age)': math.exp(age),
            'exp (BMI)':math.exp(bmi),
            'exp (BloodPressure)':math.exp(blood_pressure),
            'exp (GlucoseLevel)':math.exp(glucose),
            'Gender_Female': gender == 'Female',
            'Gender_Male': gender == 'Male',
            'FamilyHistory_No': family_history == 'No',
            'FamilyHistory_Yes':  family_history == 'Yes'
        }

        df_input = pd.DataFrame([data_dict])

        CATEGORICAL_ORDINAL = 'PhysicalActivity'

        label_encoder = loaded_objects["label_encoder.pickle"]
        min_max_scaller = loaded_objects["min-max-scaler.pickle"]

        df_input.loc[:, CATEGORICAL_ORDINAL] = label_encoder.transform(df_input[CATEGORICAL_ORDINAL]) + 1

        NUMERICAL_FEATURES = ['Age',
         'BMI',
         'BloodPressure',
         'GlucoseLevel',
         'exp (Age)',
         'exp (BMI)',
         'exp (BloodPressure)',
         'exp (GlucoseLevel)']

        df_input.loc[:, NUMERICAL_FEATURES] = min_max_scaller.transform(df_input[NUMERICAL_FEATURES])

        if model:
            st.write(df_input)
            prediction = model.predict(df_input)

            if prediction[0] >= 0.5:
                st.error("Diabetes detected")
            else:
                st.success("No Diabetes detected")

            if hasattr(model, "predict_proba") :
                proba = model.predict_proba(df_input)[0]
                if proba[0] not in [0, 1]:
                    st.write("Probability of Diabetes:", proba[1])
        else:
            st.warning("Please select a model to run prediction.")

with main_tabs[1]:
    st.header("Evaluation Dataset")
    eval_path = "eval_dataset.csv"
    if os.path.exists(eval_path):
        df_eval = pd.read_csv(eval_path, index_col=0)
        st.dataframe(df_eval)
    else:
        st.info("No evaluation dataset found.")
