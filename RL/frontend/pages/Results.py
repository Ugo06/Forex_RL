import json
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

if "is_trained" not in st.session_state:
    st.session_state.is_trained= False

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

from utils.streamlit_utils import apply_css
apply_css()

st.markdown("<h1>Results of Learning</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:20px;'>Here you can load the results of your last training.</p>", unsafe_allow_html=True)

load_setup = st.selectbox('Do you prefer to load an existing results or set up and train the Agent?', ["Load", "Set up"])
if load_setup == "Load":
    config_path = st.text_input("Config File Path", value="")
    if config_path:
        try:
            config = load_config(config_path)
            st.success('Config file loaded successfully!')
        except Exception as e:
            st.warning('Enter a valid Config File Path!!', icon="⚠️")
            config = None
    else:
        st.warning('Please provide a valid config file path.', icon="⚠️")
        config = None
else:
    if "config" not in st.session_state:
        st.warning('Set up and train the agent!!', icon="⚠️")
        config = None
    else:
        config = st.session_state.config
        if st.session_state.is_trained:
            st.success('Config file loaded successfully!')
        else:
            st.warning('Train the Agent!!', icon="⚠️")
            config = None
        


if config or st.session_state.is_trained:
    config = load_config(config_path)
    st.markdown("<h1 style='font-size:30px;'>Configuration used for the training</h1>", unsafe_allow_html=True)
    st.session_state.config = config
    st.json(st.session_state.config)
    
    st.markdown("<h1 style='font-size:30px;'>Results</h1>")
    run_folder = os.path.join(st.session_state.config['SAVE_DIR'], f"config_{st.session_state.config['RUN_ID']}")
    
    path = os.path.join(run_folder, "rolling_mean_scores_plot.png")
    st.image(path)

    path = os.path.join(run_folder, "scores_plot.png")
    st.image(path)

    path = os.path.join(run_folder, "time_order_plot.png")
    st.image(path)

    episode=st.slider("Numero of the backup",min_value=0,max_value=st.session_state.config["NB_EPISODE"],step=st.session_state.config["ITER_SAVE_MODEL_SCORE"])
    if episode == 0:
        episode = 1
    path=os.path.join(run_folder,f'agent_trading_episode_{episode}.mp4')
    st.video(path, format="video/mp4")

    