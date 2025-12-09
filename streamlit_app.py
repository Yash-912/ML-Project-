import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.exception import CustomException

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("Student Performance Predictor")

# Navigation Menu
menu = st.sidebar.radio("Navigation", ["Predict", "Train Model"])

# -------------------- PREDICTION PAGE --------------------
if menu == "Predict":
    st.header("Enter Student Details")

    gender = st.selectbox("Gender", ["male", "female"])
    ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_education = st.selectbox(
        "Parental Level of Education",
        ["associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"]
    )
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
    test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)

    if st.button("Predict Score"):
        try:
            data = CustomData(
                gender=gender,
                race_ethnicity=ethnicity,
                parental_level_of_education=parental_education,
                lunch=lunch,
                test_preparation_course=test_prep,
                reading_score=reading_score,
                writing_score=writing_score
            )

            pred_df = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            result = predict_pipeline.predict(pred_df)[0]

            st.success(f"🎯 **Predicted Math Score: {result}**")

        except Exception as e:
            st.error(f"Error: {e}")

# -------------------- TRAINING PAGE --------------------
elif menu == "Train Model":
    st.header("Train the Model")

    if st.button("Start Training"):
        try:
            train_pipeline = TrainPipeline()
            r2 = train_pipeline.run()
            st.success(f"Model Trained Successfully! R2 Score: {r2}")

        except Exception as e:
            st.error(f"Training failed: {e}")
