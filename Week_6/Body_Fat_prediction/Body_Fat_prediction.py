import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Body Fat Predictor", layout="centered")
st.title('🏋 Body Fat Predictor')
st.subheader("Predict your body fat percentage based on measurements")


@st.cache_resource
def load_or_train_model():
    model_path = "body_fat_model.pkl"
    mae_path = "body_fat_mae.npy"

    if os.path.exists(model_path) and os.path.exists(mae_path):
        model = joblib.load(model_path)
        mae = np.load(mae_path)
        return model, mae

    df = pd.read_csv('bodyfat.csv')
    df.drop(['Density', 'Age', 'Height', 'Ankle'], axis=1, inplace=True)
    df['Weight'] = (df['Weight'] * 0.453592).round(2)

    X = df.drop(['BodyFat'], axis=1)
    y = df['BodyFat']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)

    
    joblib.dump(model, model_path)
    np.save(mae_path, mae)

    return model, mae


model, mae = load_or_train_model()

st.markdown("### Enter Your Measurements:")

cols = st.columns(2)
with cols[0]:
    weight = st.number_input("Weight (kg)", 40.0, 200.0, 70.0, step=0.1)
    neck = st.number_input("Neck (cm)", 25.0, 60.0, 36.0, step=0.1)
    chest = st.number_input("Chest (cm)", 70.0, 150.0, 95.0, step=0.1)
    abdomen = st.number_input("Abdomen (cm)", 60.0, 150.0, 85.0, step=0.1)
    hip = st.number_input("Hip (cm)", 70.0, 150.0, 95.0, step=0.1)

with cols[1]:
    thigh = st.number_input("Thigh (cm)", 40.0, 90.0, 60.0, step=0.1)
    knee = st.number_input("Knee (cm)", 30.0, 60.0, 37.0, step=0.1)
    biceps = st.number_input("Biceps (cm)", 20.0, 50.0, 32.0, step=0.1)
    forearm = st.number_input("Forearm (cm)", 20.0, 40.0, 27.0, step=0.1)
    wrist = st.number_input("Wrist (cm)", 13.0, 25.0, 17.0, step=0.1)

# -------------------- PREDICTION --------------------
if st.button("Predict Body Fat"):
    # Basic validation
    if abdomen <= neck:
        st.error("Abdomen circumference must be greater than neck circumference for realistic results.")
    else:
        features = np.array([[weight, neck, chest, abdomen, hip, thigh, knee, biceps, forearm, wrist]])
        prediction = model.predict(features)[0]

        lower_bound = prediction - mae
        upper_bound = prediction + mae

        st.success(f"Predicted Body Fat: **{prediction:.2f}%**")
        st.write(f"Estimated Range: **{lower_bound:.2f}%** - **{upper_bound:.2f}%**")

        
        if prediction < 14:
            category = "Athlete"
        elif prediction < 20:
            category = "Fit"
        elif prediction < 25:
            category = "Average"
        else:
            category = "Obese"
        st.info(f"Category: **{category}**")

        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Body Fat %"},
            gauge={
                'axis': {'range': [0, 50]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 14], 'color': "green"},
                    {'range': [14, 20], 'color': "lightgreen"},
                    {'range': [20, 25], 'color': "yellow"},
                    {'range': [25, 50], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

st.write("---")
st.write('📧 For Feedback: **mail.habiburrehman@gmail.com**')
