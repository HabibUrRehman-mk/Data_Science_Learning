import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
st.title('Body Fat predictor')
st.subheader("This app can predict you body fat percentage")
df=pd.read_csv('bodyfat.csv')
df.drop(['Density','Age','Height','Ankle'],axis=1,inplace=True)
df['Weight'] = (df['Weight'] * 0.453592).round(2)
x=df.drop(['BodyFat'],axis=1)
y=df['BodyFat']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestRegressor()
model.fit(x_train, y_train)
# Predictions
y_pred = model.predict(x_test)

# print("R² Score:", r2_score(y_test, y_pred))
# def calculate_density_male(height_cm, neck_cm, abdomen_cm):
#     return 495 / (86.010 * np.log10(abdomen_cm - neck_cm) 
#                   - 70.041 * np.log10(height_cm) 
#                   + 36.76 + 450)
# density=calculate_density_male(67.75,36.2,85.2)
# print(density)
weight = st.number_input("Weight (kg)", min_value=40.0, max_value=200.0, value=70.0, step=0.1)
neck = st.number_input("Neck circumference (cm)", min_value=25.0, max_value=60.0, value=36.0, step=0.1)
chest = st.number_input("Chest circumference (cm)", min_value=70.0, max_value=150.0, value=95.0, step=0.1)
abdomen = st.number_input("Abdomen circumference (cm)", min_value=60.0, max_value=150.0, value=85.0, step=0.1)
hip = st.number_input("Hip circumference (cm)", min_value=70.0, max_value=150.0, value=95.0, step=0.1)
thigh = st.number_input("Thigh circumference (cm)", min_value=40.0, max_value=90.0, value=60.0, step=0.1)
knee = st.number_input("Knee circumference (cm)", min_value=30.0, max_value=60.0, value=37.0, step=0.1)
biceps = st.number_input("Biceps circumference (cm)", min_value=20.0, max_value=50.0, value=32.0, step=0.1)
forearm = st.number_input("Forearm circumference (cm)", min_value=20.0, max_value=40.0, value=27.0, step=0.1)
wrist = st.number_input("Wrist circumference (cm)", min_value=13.0, max_value=25.0, value=17.0, step=0.1)

# Prepare features in same order as training after dropping ['Density','Age','Height','Ankle']
features = np.array([[weight, neck, chest, abdomen, hip, thigh, knee, biceps, forearm, wrist]])

# Prediction
if st.button("Predict Body Fat"):
    prediction = model.predict(features)
    lower_bound = prediction[0] * 0.8 
    upper_bound = prediction[0]        

    st.success(f"Predicted Body Fat: **{lower_bound:.2f}%** to **{upper_bound:.2f}%**")

    if prediction[0] < 14:
        category = "Athlete"
    elif prediction[0] < 20:
        category = "Fit"
    elif prediction[0] < 25:
        category = "Average"
    else:
        category = "Obese"

    st.info(f"Category: **{category}**")


    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction[0],
        title={'text': "Body Fat %"},
        gauge={'axis': {'range': [0, 50]},
            'bar': {'color': "darkblue"}}
    ))
    st.plotly_chart(fig)


st.write('For Feedback Feel free to contact at mail.habiburrehman@gmail.com')





