import streamlit as st
from PIL import Image
from utils.streamlit_utils import apply_css
import os

st.set_page_config(page_title="The Wall Street Master", layout="wide")
# Custom CSS to set background color and sidebar color

apply_css()


st.markdown("<h1 style='text-align: center; font-family: \"Times New Roman\", Times, serif;'>The Wall Street Master</h1>", unsafe_allow_html=True)

# Load and display image
image_path = os.path.join("C:/Users/Ugo/Documents/AI/Forex_ML/APPLICATION/frontend/images", "trading_bot.jpg")  # Update with your image path
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, use_column_width=False, width=300)
else:
    st.warning("Image not found. Please add an image to the Images folder.")


