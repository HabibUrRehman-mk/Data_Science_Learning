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

# -------------------- MODEL LOADING --------------------
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
    cols=['Neck','Chest','Abdomen','Hip','Thigh','Knee','Biceps','Forearm','Wrist']
    X[cols]=X[cols]* 0.3937
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
    weight = st.number_input("Weight (kg)",min_value=10.0,value=69.0, step=0.1,)
    neck = st.number_input("Neck Circumference (Inches)",min_value=7.0, value= 14.0, step=0.1)
    chest = st.number_input("Chest Circumference (Inches)",min_value=10.0,value= 38.0, step=0.1)
    abdomen = st.number_input("Abdomen Circumference (Inches)",min_value=10.0, value=33.0, step=0.1)
    hip = st.number_input("Hip Circumference (Inches)",min_value=10.0, value= 37.0, step=0.1)

with cols[1]:
    thigh = st.number_input("Thigh Circumference (Inches)",min_value=10.0, value= 24.0, step=0.1)
    knee = st.number_input("Knee Circumference (Inches)",min_value=8.0,value= 15.0, step=0.1)
    biceps = st.number_input("Biceps Circumference (Inches)",min_value=5.0, value= 13.0, step=0.1)
    forearm = st.number_input("Forearm Circumference (Inches)",min_value=5.0, value= 10.0, step=0.1)
    wrist = st.number_input("Wrist Circumference (Inches)",min_value=3.0,value= 7.0, step=0.1)

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

        # st.success(f"Predicted Body Fat: **{prediction:.2f}%**")
        # st.success(f"Estimated Range: **{lower_bound:.2f}%** - **{upper_bound:.2f}%**")

        
        if prediction < 14:
            st.success(f"Predicted Body Fat: **{prediction:.2f}%**")
            st.success(f"Estimated Range: **{lower_bound:.2f}%** - **{upper_bound:.2f}%**")
            category = "Athlete"
        elif prediction < 20:
            st.success(f"Predicted Body Fat: **{prediction:.2f}%**")
            st.success(f"Estimated Range: **{lower_bound:.2f}%** - **{upper_bound:.2f}%**")
            category = "Fit"
        elif prediction < 25:
            st.warning(f"Predicted Body Fat: **{prediction:.2f}%**")
            st.warning(f"Estimated Range: **{lower_bound:.2f}%** - **{upper_bound:.2f}%**")
            category = "Average"
        else:
            st.error(f"Predicted Body Fat: **{prediction:.2f}%**")
            st.error(f"Estimated Range: **{lower_bound:.2f}%** - **{upper_bound:.2f}%**")
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
