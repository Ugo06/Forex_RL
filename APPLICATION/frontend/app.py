import streamlit as st
import os

# Set page configuration (title, icon, etc.)
st.set_page_config(page_title="The Wall Street Master", page_icon="💼", layout="wide")

# Sidebar Navigation
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Config Page", "Training Page", "Test Page"])

# Customize sidebar background color to dark grey
sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        background-color: #333333;
    }
    </style>
    """
st.markdown(sidebar_style, unsafe_allow_html=True)

# Load image (Make sure you have an image in the 'Images' folder)
image_path = os.path.join("Images", "your_image.jpg")  # Replace 'your_image.jpg' with your actual image file

# Define the Welcome Page
if page == "Welcome":
    # Custom title in the center with styling
    st.markdown(
        """
        <style>
        .title {
            font-size:50px;
            text-align:center;
            color:black;
            font-family: 'Arial', sans-serif;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown('<p class="title">The Wall Street Master</p>', unsafe_allow_html=True)

    # Display an image below the title
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    
    # File uploader and Config button
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Upload data in CSV format
        uploaded_file = st.file_uploader("Import Data File (CSV)", type="csv")
        if uploaded_file is not None:
            st.write("Uploaded Data Preview:")
            st.write(pd.read_csv(uploaded_file).head())  # Display first few rows of the data
    
    with col2:
        # Button to navigate to Config page
        if st.button("Go to Config Page"):
            st.session_state.page = "Config Page"

# Load other pages dynamically from the 'pages' folder
elif page == "Config Page":
    exec(open("pages/config.py").read())

elif page == "Training Page":
    exec(open("pages/training.py").read())

elif page == "Test Page":
    exec(open("pages/test.py").read())