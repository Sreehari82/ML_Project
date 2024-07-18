import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,auc,roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import base64


# st.set_page_config(
#     page_title="Insights into the ML model",
#     page_icon=":robot_face:",
# )


def show_insights_model():
    st.header(":violet[**Insights into the ML model**]",divider='gray')

    st.header("SYNOPSIS OF THE MODEL")
    st.write("""
    This machine learning model uses an :violet[**XGBoost Classifier**]. Trained on data from a Mayo Clinic study, it aims to classify the stage of liver cirrhosis based on medical and demographic variables.
    """)




    st.header("DATA PREPROCESSING")

    st.write("""
    To start training the model, first prepared the data by handling missing values, encoding categorical variables, and scaling numerical features.
    """)

    st.header("OVER SAMPLING")
    st.write("""
            The dataset is almost balanced,but still i have performed oversampling on the dataset to check whether it enhances the model performance. To tackle the issue, used the SMOTE (Synthetic Minority Over-sampling Technique) to oversample the data, ensuring a balanced dataset suitable for training purposes.
            """)

    st.header("Model Evaluation")
    st.write("""
        The models evaluated in this study include:
        - K-Nearest Neighbors (KNN)
        - Support Vector Classifier (SVC)
        - Naive Bayes (NB)
        - Decision Tree (DT)
        - Random Forest (RF)
        - AdaBoost
        - Gradient Boosting
        - XGBoost""")

    st.subheader("Model Perfromance Comparison")
    acd = pd.read_csv('macc.csv')
    st.dataframe(acd)

    st.write("""
            The XGBoost Classifier was selected for this project based on its exceptional performance in accuracy, precision, and recall compared to alternative models. XGBoost is a powerful gradient boosting algorithm known for its ability to iteratively improve weak learners, thereby significantly enhancing predictive capabilities.
            """)

    st.header("Hyperparameter Tuning")
    st.write("""
    The model's performance was improved by tuning its hyperparameters. This involved testing different parameter combinations using grid search and cross-validation. Key hyperparameters adjusted included:
    - Learning Rate
    - Maximum depth
    - Estimators
    """)

    st.write(""" The best combination of hyperparameters was selected based on the performance on the validation set.
            """)

    st.header("Training the Model")
    st.write("""
           The model was trained on both the original and oversampled datasets. The training process involved fitting the model to the training data and evaluating its performance using metrics such as accuracy, precision, and recall.
           """)

    st.header("Evaluation and Validation")
    st.write("""
           After training the model, it was evaluated on the test dataset to assess its performance. The evaluation metrics provided insights into how well the model generalizes to unseen data. The confusion matrix, accuracy, precision, and recall were key metrics used in this evaluation.
           """)

    df = pd.read_csv('liver_cirrhosis1.csv')
    df.drop(['Status','Alk_Phos','Tryglicerides','Sex','Drug','Cholesterol'], axis=1, inplace=True)
    columns = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']
    le = LabelEncoder()
    for i in columns:
        df[i] = le.fit_transform(df[i])
    df['Copper'] = df['Copper'].fillna(df['Copper'].mode()[0])
    df['N_Days'] = df['N_Days'] // 365
    df['Age'] = df['Age'] // 365

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scaler = MinMaxScaler()
    os = SMOTE(random_state=1)
    X_os, y_os = os.fit_resample(X, y)
    X_os_scaled = scaler.fit_transform(X_os)
    X_os_train, X_os_test, y_os_train, y_os_test = train_test_split(X_os_scaled, y_os, test_size=0.3, random_state=1)
    xb=XGBClassifier(learning_rate=0.3,max_depth=6,n_estimators=300)
    xb.fit(X_os_train, y_os_train)
    y_os_pred = xb.predict(X_os_test)
    acs = accuracy_score(y_os_test, y_os_pred)
    report = classification_report(y_os_test, y_os_pred, output_dict=True)
    cm = confusion_matrix(y_os_test, y_os_pred)
    report_df = pd.DataFrame(report).transpose()

    st.header(':violet[XGBClassifier]')

    st.subheader(" Accuracy score : 92 %")
    # st.text(f'{acs * 100:.2f} %')

    st.subheader(":violet[Classification Report]")
    st.dataframe(pd.read_csv('clf_report.csv'))

    st.header(":violet[Confusion Matrix]",divider='gray')

    on = st.toggle('Show Confusion Matrix')
    if on:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        st.pyplot(plt)




show_insights_model()