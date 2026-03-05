import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Diabetes Progression Predictor")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    # Check required columns
    if 'insulin' in df.columns and 'disease_progression' in df.columns:

        X = df[['insulin']]
        y = df['disease_progression']

        model = LinearRegression()
        model.fit(X, y)

        st.subheader("Enter Insulin Level")
        insulin_value = st.number_input("Insulin", min_value=0.0)
        if st.button("Predict Progression"):
            prediction = model.predict([[insulin_value]])
            st.success(f"Predicted Disease Progression: {prediction[0]:.2f}")
    else:
        st.error("Dataset must contain 'insulin' and 'disease_progression' columns.")