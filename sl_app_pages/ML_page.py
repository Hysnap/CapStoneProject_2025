import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load("ML_model.pkl")  # Replace with your actual model path
    vectorizer = joblib.load("vectorizer.pkl")  # Load the text vectorizer
    return model, vectorizer

model, vectorizer = load_model()

st.title("Fake News Detection Dashboard")
st.write("This dashboard allows users to analyze the model's"
         " performance and test new articles for realness.")

# Sidebar navigation
option = st.sidebar.radio("Choose an option",
                          ["Model Overview",
                           "Test a News Article"])

if option == "Model Overview":
    st.header("Model Overview")
    st.write("### Model Type: RandomForestClassifier")
    st.write("### Feature Importance:")
    feature_importances = model.feature_importances_
    st.bar_chart(pd.Series(feature_importances, name="Feature Importance"))

    st.write("### Model Performance")
    st.write("The model was trained using real and fake news data"
             " and evaluated based on accuracy, precision, and recall.")

elif option == "Test a News Article":
    st.header("Test a News Article")
    uploaded_file = st.file_uploader("Upload a text file containing"
                                     " a news article", type=["txt"])

    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.write("### Uploaded Text:")
        st.write(text[:500] + "...")  # Displaying a preview

        # Preprocess and predict
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        realness_score = model.predict_proba(text_vectorized)[0][1]
        
        st.write("### Prediction Result:")
        if prediction == 1:
            st.success(f"The model predicts this article"
                       " is **Real** with a confidence of {realness_score:.2f}")
        else:
            st.error(f"The model predicts this article"
                     " is **Fake** with a confidence of {1 - realness_score:.2f}")
