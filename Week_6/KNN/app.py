# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🌸 Iris Flower Classifier (KNN)")
st.write("Enter the flower's measurements to predict its species.")

# User input form
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    st.success(f"Predicted Species: **{prediction}**")
