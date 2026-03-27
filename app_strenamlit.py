from pickle import load
import streamlit as st
import joblib

model = joblib.load("model.pkl")  # ruta relativa al proyecto

st.title("Cancer Prediccion")

val1 = st.slider("f1", min_value = 0.0, max_value = 50.0, step = 0.1)
val2 = st.slider("f2", min_value = 0.0, max_value = 50.0, step = 0.1)
val3 = st.slider("f3", min_value = 0.0, max_value = 50.0, step = 0.1)
val4 = st.slider("f4", min_value = 0.0, max_value = 50.0, step = 0.1)
val5 = st.slider("f5", min_value = 0.0, max_value = 50.0, step = 0.1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4, val5]])[0])
    st.write("Prediction:", prediction)