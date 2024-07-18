import streamlit as st
import base64


def show_conclusion():
    st.header(":violet[Conclusion]",divider='gray')

    st.header("Summary of the Project")
    st.write("""
    This project focused on developing and training a machine learning model using data from a Mayo Clinic study, is designed to classify the stage of liver cirrhosis by analyzing a range of medical and demographic variables. 
    Leveraging advanced machine learning techniques, it aims to provide accurate assessments that can assist healthcare professionals in making informed decisions about patient care and treatment strategies. 
    By integrating insights from extensive clinical data, this model seeks to enhance diagnostic precision and contribute to improved outcomes for individuals affected by liver cirrhosis."
    """)

    st.header("Key Findings")
    st.write("""
    Bioinformatics, which combines biology, computer science, and statistics, relies heavily on machine learning to analyze complex biological data and vast amounts of clinical information. These methods allow us to:

    - Find hidden patterns in the data.
    - Improve the accuracy of diagnoses.
    - Personalize medical treatments.
    
    Machine learning in bioinformatics goes beyond predicting the stage of diseases like liver cirrhosis. It can also be used for:

    - Early detection of genetic diseases.
    - Identifying biomarkers for various medical conditions.
    - Optimizing drug development processes.

    """)

    st.header("Model Performance")
    st.write("""
    The machine learning model was evaluated using various performance metrics. This project successfully showed that machine learning can predict the stage of liver cirrhosis with high accuracy. 
    This achievement highlights the potential of these techniques for predicting diseases based on clinical data.Furthermore, this analysis provides a strong foundation for applying machine learning to predict other diseases. 
    The steps followed in this notebook, from data preparation to model training and evaluation, can be applied to many medical datasets.This showcases the versatility and potential of machine learning algorithms in bioinformatics.
    """)

    st.write("- [Link to my colab notebook](https://colab.research.google.com/drive/1NRdaDH6amJkXqZF5_Iw6rZjrDoWbeHjt#scrollTo=nGwY7p5BIkyX)")

show_conclusion()
