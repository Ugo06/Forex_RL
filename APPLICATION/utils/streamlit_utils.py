import streamlit as st

def apply_css():
    st.markdown(
        """
        <style>
        * {
            font-family: 'Times New Roman', Times, serif;
        }
        .stApp {
            background-color: white;
        }
        [data-testid="stSidebar"] {
            background-color: black;
        }
        [data-testid="stSidebar"] * {
        color: white !important;
        }
        h1 {
        font-family: 'Times New Roman', Times, serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
