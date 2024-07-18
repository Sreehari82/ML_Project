import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64





def load_data(path):
    data = pd.read_csv(path)
    return data

df = load_data('liver_cirrhosis1.csv')

def show_data_overview():
    st.header(":violet[**Patient's Medical Details**]",divider='gray')
    st.write("This dataset stems from a Mayo Clinic investigation of primary biliary cirrhosis (PBC) of the liver, "
             "carried out between 1974 and 1984. It comprises 25,000 entries and 19 attributes, providing a substantial "
             "foundation for analyzing and forecasting the stages of liver cirrhosis through machine learning methods.")

    st.dataframe(df)

    st.subheader("Dataset Dimensions and Description")
    p=st.selectbox("Select an option for the specified information",['Size','Shape','Describe'])
    if p=='Size':
        dataset_size = df.size
        st.write(f"**Size**: {dataset_size}")
    elif p=='Shape':
        dataset_shape = df.shape
        st.write(f"**Shape**: {dataset_shape}")
    elif p=='Describe':
        st.dataframe(df.describe())

    dfc = load_data('df_cleaned.csv')
    # Select columns to visualize
    Data_columns = dfc.select_dtypes(['float64', 'int64']).columns

    load=st.button("Visualizations of Dataset")

    if "load_state" not in st.session_state:
        st.session_state.load_state=False

    if load or st.session_state.load_state:
        st.session_state.load_state=True
        opt=st.radio("Select an option for visualization",['Pie-Chart','KDE-Plot','Count-Plot','Correlation Heatmap'])


        if opt=='Pie-Chart':
            c=df['Stage'].value_counts()
            labels = ['Stage 1', 'Stage 2', 'Stage 3']
            explode = [0, 0, 0.1]
            # if on:
            fig, ax = plt.subplots()
            ax.pie(c, labels=labels, explode=explode, shadow=True, autopct='%1.1f%%')
            ax.legend(labels, loc='upper left')
            ax.set_title("Count of Patients in Different Stages of Liver Cirrhosis")
            st.pyplot(fig)


        elif opt == 'KDE-Plot':

            col = dfc.select_dtypes('number').columns
            palettes = ['#1CEC01', '#9E0068', '#000875']

            # if on1:
            selected_kde_column = st.selectbox("Select column for KDE plot", Data_columns, key="kde")
            selected_kde_hue = st.selectbox("Select hue for KDE plot", Data_columns, key="kde_hue")
            fig, ax = plt.subplots()

            sns.kdeplot(data=dfc, x=selected_kde_column, hue=selected_kde_hue, palette=palettes,
                        alpha=0.5, linewidth=0, fill=True, ax=ax)
            ax.legend(['Stage 1', 'Stage 2', 'Stage 3'])

            st.pyplot(fig)


        elif opt == 'Count-Plot':


            selected_count_column = st.selectbox("Select column for count plot", Data_columns, key="count")
            selected_count_hue = st.selectbox("Select hue for count plot", Data_columns, key="count_hue")
            fig, ax = plt.subplots(figsize=(20, 20))
            sns.countplot(x=selected_count_column, hue=selected_count_hue, data=dfc, ax=ax)
            plt.xticks(rotation=90)  # Rotate x-axis labels if necessary
            ax.legend(['Stage 1', 'Stage 2', 'Stage 3'])
            st.pyplot(fig)

        elif opt == 'Correlation Heatmap':
            # if on3:
            df1 = load_data('df_cleaned.csv')
            corr_matrix = df1.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", annot_kws={"size": 5}, linewidths=.5, cmap='coolwarm',
                        ax=ax, )
            plt.title('Correlation Heatmap')
            st.pyplot(fig)




show_data_overview()