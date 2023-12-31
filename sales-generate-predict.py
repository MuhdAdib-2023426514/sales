import streamlit as st
import pandas as pd
import sklearn
import xgboost
import pickle

st.write("# Aplikasi Ramalan Jualan Pernigaan")
st.write("Aplikasi ini meramalkan **Jualan** anda!")

st.sidebar.header('Berikan perbelanjaan pengiklanan anda')

def user_input_features():
    tv = st.sidebar.slider('TV', 0.0, 300.0, 1.0)
    radio = st.sidebar.slider('Radio', 0.0, 50.0, 1.0)
    newspaper = st.sidebar.slider('Newspaper', 0.0, 115.0, 1.0)
    data = {'TV': tv,
            'Radio': radio,
            'Newspaper': newspaper,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Perbelanjaan pengiklanan anda')
st.write(df)

regressor = pickle.load(open("salesrandomforestregressor.h5", "rb"))
predictions = regressor.predict(df)

st.subheader('Ramalan')
st.write(predictions)

