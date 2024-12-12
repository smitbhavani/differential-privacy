import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

with open("./log_reg_model.pkl", "rb") as f:
    log_reg_model = pickle.load(f)

with open("./svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("./nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("./rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("./ada_boost_model.pkl", "rb") as f:
    ada_boost_model = pickle.load(f)

with open("./vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

label_mapping = {0: 'Negative', 1: 'Positive'}

st.title("Tweet Sentiment Classifier")

model_option = st.selectbox("Select Model", ["Logistic Regression", "SVM", "Naive Bayes", "Random Forest", "ADA Boost"])

new_text = st.text_input("Enter a tweet for prediction:")

if model_option == "Logistic Regression":
    model = log_reg_model
elif model_option == "SVM":
    model = svm_model
elif model_option == "Naive Bayes":
    model = nb_model
elif model_option == "Random Forest":
    model = rf_model
elif model_option == "ADA Boost":
    model = ada_boost_model

if st.button("Predict"):
    if new_text:
        new_text_vec = vectorizer.transform([new_text])

        prediction = model.predict(new_text_vec)

        prediction_label = label_mapping[prediction[0]]

        st.write(f"Prediction: {prediction_label}")
    else:
        st.write("Please enter some text to classify.")
