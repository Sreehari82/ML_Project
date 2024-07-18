import streamlit as st
from streamlit_option_menu import option_menu
import base64

from data_overview import show_data_overview
from insights_model import show_insights_model
from prediction import show_prediction
from conclusion import show_conclusion

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    add_bg_from_local('Minimalist Desktop Wallpaper (2).png')

    # Initialize session state for page selection
    if 'selected_page' not in st.session_state:
        st.session_state['selected_page'] = "Home"

    # Sidebar for navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Data Overview", "Insights into the Model", "Prediction", "Takeaways"],
            icons=["house", "bar-chart", "info-circle", "activity", "check-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"background-color": "#044355"},
                "icon": {"color": "white", "font-size": "25px"},
                "nav-link": {
                    "--hover-color": "#C70039"
                },
                "nav-link-selected": {"background-color": "#C70039"},
            },
            key="selected_page"
        )

    selected = st.session_state['selected_page']

    # Home Page
    if selected == "Home":
        st.header(":violet[**Welcome to the LIVER CIRRHOSIS PREDICTION APP**]", divider='gray')
        st.write("""
        This app predicts the stage of liver cirrhosis based on medical data.

        Use the navigation menu on the left to explore the different sections of the app:
        - **Home**: You're here! This is the home page.
        - **Data Overview**: Details about the data used.
        - **Insights into the Model**: Information about the model.
        - **Prediction**: Make predictions using the model.
        - **Takeaways**: Summary and conclusion of the project.
        """)

        st.write("""
        ### Note:
        This app is a demonstration and the predictions should not be used as a substitute for professional medical advice.
        """)

        st.write(" ")
        st.write(" ")

        st.write("CREATED BY : ", ":blue[SREEHARI S]")

    # Data Overview Page
    elif selected == "Data Overview":
        show_data_overview()

    # Insights into the Model Page
    elif selected == "Insights into the Model":
        show_insights_model()

    # Prediction Page
    elif selected == "Prediction":
        show_prediction()

    # Conclusion Page
    elif selected == "Takeaways":
        show_conclusion()
