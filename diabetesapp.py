import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title
st.title("Diabetes Disease Progression Predictor")

# Load dataset
df = pd.read_csv("diabetes_linear_regression_dataset.csv")

# Display dataset
st.subheader("Dataset Preview")
st.dataframe(df)

# Prepare data
X = df[['insulin']]
y = df['disease_progression']

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
st.subheader("Enter Patient Insulin Level")

insulin_input = st.number_input("Insulin Level", min_value=0.0)

# Prediction
if st.button("Predict Disease Progression"):
    
    prediction = model.predict([[insulin_input]])
    
    st.success(f"Predicted Disease Progression Score: {prediction[0]:.2f}")
