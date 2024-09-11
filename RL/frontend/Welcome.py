import streamlit as st
from PIL import Image
from utils.streamlit_utils import apply_css
import os

st.set_page_config(page_title="The Wall Street Master", layout="wide")
# Custom CSS to set background color and sidebar color

apply_css()


st.markdown("<h1 style='text-align: center; font-family: \"Times New Roman\", Times, serif;'>The Wall Street Master</h1>", unsafe_allow_html=True)

# Load and display image
image_path = os.path.join("C:/Users/Ugo/Documents/AI/Forex_ML/RL/frontend/images", "trading_bot.jpg")  # Update with your image path

left_co, cent_co,last_co = st.columns(3)

if os.path.exists(image_path):
    image = Image.open(image_path)
    with cent_co:
        st.image(image, use_column_width=False, width=300)
else:
    st.warning("Image not found. Please add an image to the Images folder.")

body_1 = """
This project is an introduction to quantitative finance, aimed at exploring the influence of macroeconomic indicators on the EUR/USD exchange rate. The exchange rate between the euro and the dollar is influenced by several external factors, including monetary policies, political situations, economic health, and conflicts in each region. Macroeconomic indicators from both the eurozone and the United States, reflecting the economic health of these areas, play a crucial role.

We hypothesize that the value of EUR/USD can be expressed as a weighted sum of various economic indicators. Specifically, we assume that the price of EUR/USD at time $t+T$ (where $T$ represents a certain time horizon) can be approximated by the price at time $t$, added to a weighted sum of the indicator values $I_i$ at time $t$, transformed by a function $f$, as expressed by the following equation:

"""
st.markdown(f"{body_1}", unsafe_allow_html=True)
body = r"""
P_{t+T} = P_t + \sum_{i=1}^n w_i f_i(I_i(t)) + \epsilon
"""
st.latex(body)
body_2 = """
Where:
- $P_{t+T}$ is the EUR/USD price at time $t+T$, where $T$ represents a certain time horizon,
- $P_t$ is the EUR/USD price at time $t$,
- $I_i(t)$ represents the value of the $i$-th macroeconomic indicator at time $t$,
- $f_i$ is a transformation function (for example, normalization, logarithmic transformation, differencing, etc.),
- $w_i$ is the weight associated with the indicator $I_i(t)$,
- $\\epsilon$ is an error term.

To test this hypothesis, we implemented a **reinforcement learning** algorithm, which allows us to design an agent capable of learning a trading strategy autonomously based on these economic indicators. The key advantage of this approach is that the algorithm dynamically adjusts its strategy as it interacts with market data, allowing trading decisions to be adapted in response to evolving economic conditions.

Thus, the goal of this study is to verify the validity of our hypothesis through the learning of an optimal strategy, where the trading agent, through **reinforcement learning**, learns to make buy or sell decisions based on macroeconomic indicators, with the aim of maximizing long-term gains.

"""
st.markdown(f"{body_2}", unsafe_allow_html=True)

