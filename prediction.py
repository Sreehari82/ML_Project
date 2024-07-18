import streamlit as st
import pandas as pd
from PIL import Image  # For handling images
import pickle
import matplotlib.pyplot as plt
from joblib import load


# st.set_page_config(
#         page_title="Liver Cirrhosis Classification App",
#         initial_sidebar_state="expanded"
#     )

st.markdown(
        """
        <style>
        .reportview-container {
            background: url('v3-DARK-HEXAGONAL-PATTERN-4K.png');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def show_prediction():
    st.title(':violet[LIVER CIRRHOSIS STAGE PREDICTION]')
    st.write("This webpage helps you to predict the stages of liver cirrhosis based on patient data.")

    # storing variable
    xb = load('XG_model.joblib')
    scaler = load('scaler.joblib')

    col1, col2, col3 = st.columns(3)
    with col1:
        N_Days = st.text_input("Treatment Period(in years)")
        Hepatomegaly = st.selectbox('Hepatomegaly', ('Present', 'Absent'))
        if Hepatomegaly == 'Present':
            hep = 1
        else:
            hep = 0
        Bilirubin = st.text_input('Bilirubin')
        SGOT = st.text_input('SGOT')

    with col2:
        Age = st.text_input('Age')
        Spiders = st.selectbox('Spiders', ('Present', 'Absent'))
        if Spiders == 'Present':
            sp = 1
        else:
            sp = 0
        Albumin = st.text_input('Albumin')
        Platelets = st.text_input('Platelets')

    with col3:
        Ascites = st.selectbox('Ascites', ('Present', 'Absent'))
        if Ascites == 'Present':
            asc = 1
        else:
            asc = 0
        Edema = st.selectbox('Edema', ('Present', 'Absent'))
        if Edema == 'Present':
            ed = 1
        else:
            ed = 0
        Copper = st.text_input('Copper')
        Prothrombin = st.text_input('Prothrombin')

    pred = st.button("PREDICT")


    if pred:
        try:
            prediction=xb.predict(scaler.transform([[N_Days,Age,asc,hep,sp,
                                                     ed,Bilirubin,Albumin,
                                                     Copper,SGOT,Platelets,Prothrombin]]))
            if prediction==0:
                st.header("STAGE 1 LIVER CIRRHOSIS")
            elif prediction==1:
                st.header("STAGE 2 LIVER CIRRHOSIS")
            elif prediction==2:
                st.header("STAGE 3 LIVER CIRRHOSIS")
        except:
            st.write("INPUT ERROR : input cannot be blank!")



show_prediction()
